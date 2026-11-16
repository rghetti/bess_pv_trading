"""Check the explicit gap threshold and mixed forecast-availability batching."""

import numpy as np
import pandas as pd

from scripts.prepare_timesfm_realized import fill_history
from scripts.run_da_zero_shot_model import request_batches


def test_forecast_precedes_short_interpolation_and_long_gap_zero():
    """Forecast values are retained; eight quarters interpolate and nine become zero."""
    result, forecasts, interpolated, zeros = fill_history([1, np.nan, np.nan, np.nan, 5], [0, 2, np.nan, np.nan, 0])
    assert np.allclose(result, [1, 2, 3, 4, 5])
    assert (forecasts, interpolated, zeros) == (1, 2, 0)
    assert fill_history([1] + [np.nan] * 8 + [10])[2:] == (8, 0)
    result, _, interpolated, zeros = fill_history([1] + [np.nan] * 9 + [10])
    assert interpolated == 0 and zeros == 9 and (result[1:-1] == 0).all()


def test_missing_ida3_overnight_is_not_interpolated_across():
    """IDA3's overnight discontinuity is not mistaken for a quarter-hour step."""
    clock = pd.DatetimeIndex(['2026-01-01T22:45Z', '2026-01-02T12:00Z', '2026-01-02T12:15Z'])
    values, _, _, _ = fill_history([9, np.nan, 7], timestamps=clock)
    assert values.tolist() == [9, 7, 7]


def test_batching_separates_future_and_past_only_covariates():
    """A missing future forecast creates a different, valid worker batch."""
    request = {'context': [1], 'horizon': 1, 'forecast_covariates': {'wind': [1, 2]}, 'past_covariates': {}}
    fallback = {'context': [1], 'horizon': 1, 'forecast_covariates': {}, 'past_covariates': {'wind': [1]}}
    assert len(list(request_batches([('a', 'a', request), ('b', 'b', fallback)], 10))) == 2


def test_actual_history_with_future_forecast_and_past_only_fallback(tmp_path):
    """The same series contains actual history and forecast horizon, or history alone."""
    import json
    from bess_pv_trading.forecasting.studies.zero_shot_da import digest, write_json
    from scripts.prepare_timesfm_realized import prepare

    base = tmp_path / 'base'
    source = tmp_path / 'forecasts'
    actual = tmp_path / 'actual'
    (base / 'requests').mkdir(parents=True)
    source.mkdir()
    actual.mkdir()
    history = pd.date_range('2026-01-01T00:00Z', periods=7 * 96, freq='15min')
    horizon = pd.date_range('2026-01-08T00:00Z', periods=96, freq='15min')
    forecast_rows = []
    for name in ('load_forecast', 'generation_forecast', 'solar_forecast', 'wind_onshore_forecast'):
        for time in history.append(horizon):
            if name == 'solar_forecast' and time == horizon[5]:
                continue
            forecast_rows.append({'dataset': name, 'zone': 'NORD', 'product_start': time,
                                  'product_end': time + pd.Timedelta(minutes=15), 'value': 4.0, 'unit': 'MW'})
    pd.DataFrame(forecast_rows).to_parquet(source / 'forecast.parquet')
    write_json(source / 'manifest.json', {'tables': {'nord_forecasts': {'path': 'forecast.parquet', 'sha256': digest(source / 'forecast.parquet')}}})
    actual_rows = []
    for dataset, direction in [('actual_load', 'load'), ('actual_solar_generation', 'production'), ('actual_wind_onshore_generation', 'production')]:
        for index, time in enumerate(history):
            if dataset == 'actual_solar_generation' and index == 5:
                continue
            actual_rows.append({'dataset': dataset, 'zone': 'NORD', 'direction': direction, 'product_start': time,
                                'product_end': time + pd.Timedelta(minutes=15), 'value': 3.0, 'unit': 'MW'})
    pd.DataFrame(actual_rows).to_parquet(actual / 'actual.parquet')
    write_json(actual / 'manifest.json', {'tables': {'nord_actuals': {'path': 'actual.parquet', 'sha256': digest(actual / 'actual.parquet')}}})
    request = {'delivery_day': '2026-01-08', 'context_start': history[0].isoformat(), 'context_end_exclusive': horizon[0].isoformat(),
               'context': [100.] * len(history), 'product_starts_utc': [time.isoformat() for time in horizon], 'horizon': 96}
    write_json(base / 'requests/2026-01-08.json', request)
    write_json(base / 'manifest.json', {'files': {'requests/2026-01-08.json': digest(base / 'requests/2026-01-08.json')},
               'settings': {'timezone': 'UTC', 'market': 'DA'}, 'features': ['price'], 'models': {'timesfm3': {}},
               'source_manifest_sha256': digest(source / 'manifest.json')})
    settings = {'forecast_snapshot': str(source), 'minimum_context_days': 7, 'variants': {'actual': {
        'exclude_forecast': ['generation_forecast'], 'replace_forecast_history': {
            'load_forecast': 'actual_load', 'solar_forecast': 'actual_solar_generation',
            'wind_onshore_forecast': 'actual_wind_onshore_generation'}}}}
    result = prepare(base, actual, tmp_path / 'result', 'actual', settings)
    data = json.loads((tmp_path / 'result/requests/2026-01-08.json').read_text())
    assert result['eligible_days'] == 1
    assert data['forecast_covariates']['load_forecast'] == [3.] * len(history) + [4.] * len(horizon)
    assert 'solar_forecast' not in data['forecast_covariates']
    assert data['past_covariates']['solar_forecast'][5] == 4.0
    assert data['past_covariates']['solar_forecast'][6] == 3.0
    assert 'generation_forecast' not in data['forecast_covariates']
