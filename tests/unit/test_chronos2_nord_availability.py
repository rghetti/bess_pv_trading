"""Exercise publication cutoffs and actual-then-forecast Chronos inputs."""

import numpy as np
import pandas as pd

from scripts.prepare_timesfm_realized import available_at_gate, fill_history


POLICY = {
    'mode': 'deadline_assumption',
    'revision_policy': 'latest_retrospective',
    'timezone': 'Europe/Rome',
    'actual_lag_minutes': 60,
    'forecast_deadline_local': {
        'load_forecast': '10:00',
        'solar_forecast': '18:00',
    },
}


def rows(dataset, starts):
    """Create records whose latest revisions were obtained after the gate."""
    return pd.DataFrame({
        'dataset': dataset,
        'product_start': pd.to_datetime(starts, utc=True),
        'product_end': pd.to_datetime(starts, utc=True) + pd.Timedelta(minutes=15),
        'source_document_time': pd.Timestamp('2026-11-01T00:00Z'),
        'historical_available_at': pd.NaT,
    })


def test_actual_mask_at_da_cutoff_and_fallback():
    """Only finished intervals with a full one-hour lag may supply actuals."""
    frame = rows('actual_load', ['2026-01-01T09:30Z', '2026-01-01T09:45Z'])
    eligible = available_at_gate(frame, '2026-01-01T10:45Z', 'actual', POLICY)
    assert eligible.index.tolist() == [0]
    actual = np.array([12.0, np.nan])
    filled, used, interpolated, zeros = fill_history(actual, [8.0, 9.0])
    assert filled.tolist() == [12.0, 9.0]
    assert (used, interpolated, zeros) == (1, 0, 0)


def test_forecast_deadlines_and_retrospective_revision():
    """Renewable horizons become eligible at 18:00, while load is due at 10:00."""
    frame = rows('solar_forecast', ['2026-01-02T00:00Z'])
    assert available_at_gate(frame, '2026-01-01T16:59Z', 'forecast', POLICY).empty
    assert len(available_at_gate(frame, '2026-01-01T17:00Z', 'forecast', POLICY)) == 1
    load = rows('load_forecast', ['2026-01-02T00:00Z'])
    assert len(available_at_gate(load, '2026-01-01T09:00Z', 'forecast', POLICY)) == 1
    strict = {**POLICY, 'revision_policy': 'exclude_later_document'}
    assert available_at_gate(frame, '2026-01-01T17:00Z', 'forecast', strict).empty


def test_known_publication_and_dst_use_local_clock():
    """Recorded availability overrides the fallback and local DST is respected."""
    frame = rows('solar_forecast', ['2026-07-02T00:00Z'])
    frame['historical_available_at'] = pd.Timestamp('2026-07-01T13:00Z')
    assert available_at_gate(frame, '2026-07-01T12:59Z', 'forecast', POLICY).empty
    assert len(available_at_gate(frame, '2026-07-01T13:00Z', 'forecast', POLICY)) == 1
    frame['historical_available_at'] = pd.NaT
    assert available_at_gate(frame, '2026-07-01T15:59Z', 'forecast', POLICY).empty
    assert len(available_at_gate(frame, '2026-07-01T16:00Z', 'forecast', POLICY)) == 1


def test_recorded_only_does_not_invent_publication():
    """Strict mode cannot turn a legal deadline into a verified timestamp."""
    frame = rows('actual_load', ['2026-01-01T09:30Z'])
    strict = {**POLICY, 'mode': 'recorded_only'}
    assert available_at_gate(frame, '2026-01-02T00:00Z', 'actual', strict).empty


def test_builder_binds_chronos_and_a75_actuals(tmp_path):
    """Prepare a model-free bundle with gated actuals and a past-only solar series."""
    import json

    from bess_pv_trading.forecasting.studies.zero_shot_da import digest, write_json
    from scripts.prepare_timesfm_realized import prepare

    base = tmp_path / 'base'
    source = tmp_path / 'source'
    actuals = tmp_path / 'actuals'
    (base / 'requests').mkdir(parents=True)
    source.mkdir()
    actuals.mkdir()
    history = pd.date_range('2026-01-01T00:00Z', periods=7 * 96, freq='15min')
    horizon = pd.date_range('2026-01-08T00:00Z', periods=96, freq='15min')
    forecasts = []
    for name in ('load_forecast', 'generation_forecast', 'solar_forecast', 'wind_onshore_forecast'):
        for stamp in history.append(horizon):
            forecasts.append({'dataset': name, 'zone': 'NORD', 'product_start': stamp,
                              'product_end': stamp + pd.Timedelta(minutes=15), 'value': 4.0,
                              'historical_available_at': (pd.Timestamp('2026-01-07T09:00Z')
                                  if name == 'solar_forecast' and stamp >= horizon[0] else pd.NaT)})
    # This test needs only load and solar; exclude unused forecast streams.
    pd.DataFrame(forecasts).to_parquet(source / 'forecasts.parquet')
    source_entry = {'path': 'forecasts.parquet', 'sha256': digest(source / 'forecasts.parquet')}
    write_json(source / 'manifest.json', {'tables': {'nord_forecasts': source_entry}})
    measurements = []
    for stamp in history:
        for dataset, direction, psr in [('actual_load', 'load', None), ('actual_generation_by_type', 'production', 'B16')]:
            measurements.append({'dataset': dataset, 'zone': 'NORD', 'direction': direction,
                                 'psr_type': psr, 'product_start': stamp,
                                 'product_end': stamp + pd.Timedelta(minutes=15),
                                 'value': 3.0, 'unit': 'MW'})
    pd.DataFrame(measurements).to_parquet(actuals / 'actual.parquet')
    write_json(actuals / 'manifest.json', {'tables': {'nord_actuals': {
        'path': 'actual.parquet', 'sha256': digest(actuals / 'actual.parquet')}}})
    request = {'delivery_day': '2026-01-08', 'decision_at': '2026-01-07T11:45Z',
               'context_start': history[0].isoformat(), 'context_end_exclusive': horizon[0].isoformat(),
               'context': [100.] * len(history),
               'product_starts_utc': [stamp.isoformat() for stamp in horizon], 'horizon': 96}
    write_json(base / 'requests/2026-01-08.json', request)
    write_json(base / 'manifest.json', {
        'files': {'requests/2026-01-08.json': digest(base / 'requests/2026-01-08.json')},
        'settings': {'timezone': 'UTC', 'market': 'DA'}, 'features': ['price'],
        'models': {'timesfm3': {}}, 'source_manifest_sha256': digest(source / 'manifest.json')})
    model_config = tmp_path / 'model.yaml'
    model_config.write_text('zero_shot_price_study:\n  models:\n    chronos2:\n      revision: pinned\n')
    policy = {**POLICY, 'timezone': 'UTC'}
    settings = {'forecast_snapshot': str(source), 'minimum_context_days': 7,
                'availability_policy': policy, 'model_config': str(model_config), 'models': ['chronos2'],
                'calendar': {'weekday_one_hot': True, 'italian_holiday': True},
                'future_forecast_earliest_gate': {'solar_forecast': 'IDA2'},
                'actual_dataset_selectors': {'actual_solar_generation': {
                    'dataset': 'actual_generation_by_type', 'psr_type': 'B16'}},
                'variants': {'actual': {'exclude_forecast': ['generation_forecast', 'wind_onshore_forecast'],
                    'replace_forecast_history': {'load_forecast': 'actual_load',
                                                 'solar_forecast': 'actual_solar_generation'}}}}
    manifest = prepare(base, actuals, tmp_path / 'result', 'actual', settings)
    result = json.loads((tmp_path / 'result/requests/2026-01-08.json').read_text())
    assert manifest['models'] == {'chronos2': {'revision': 'pinned'}}
    assert result['forecast_covariates']['load_forecast'][0] == 3.0
    last_available = history.get_loc(pd.Timestamp('2026-01-07T10:30Z'))
    first_unavailable = history.get_loc(pd.Timestamp('2026-01-07T10:45Z'))
    assert result['forecast_covariates']['load_forecast'][last_available] == 3.0
    assert result['forecast_covariates']['load_forecast'][first_unavailable] == 4.0
    assert result['forecast_covariates']['load_forecast'][len(history) - 1] == 4.0
    assert result['forecast_covariates']['load_forecast'][len(history)] == 4.0
    assert result['past_covariates']['solar_forecast'][0] == 3.0
    assert 'solar_forecast' not in result['forecast_covariates']
    assert result['realized_fill_counts']['actual_load_unavailable_at_gate'] > 0
    assert len(result['forecast_covariates']['italian_national_holiday']) == len(history) + len(horizon)
    assert sum(result['forecast_covariates'][f'weekday_{day}'][-1] for day in range(7)) == 1
