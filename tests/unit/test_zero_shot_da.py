"""Verify publication-safe target-only DA inputs, DST and restart integrity."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.studies.rolling_price_data import PriceIndex, delivery_grid
from bess_pv_trading.forecasting.studies.zero_shot_da import prepare_day, prepare_bundle, report, digest, write_json
from scripts.da_zero_shot_probe import verify_bundle
from scripts.run_da_zero_shot_model import object_digest, verify_prediction, load_model


def settings():
    """Use a short synthetic history to exercise the same calendar alignment."""
    return {'timezone': 'Europe/Rome', 'context_days': 28, 'decision_time': '11:45',
            'decision_day_offset': -1, 'publication_time': '12:57', 'publication_day_offset': -1,
            'start': '2025-10-01', 'end_exclusive': None, 'november_holdout_start': '2026-11-01'}


def source(first='2025-09-01', end='2025-11-01'):
    """Create hourly pre-transition prices and native quarter-hour later prices."""
    rows = []
    for day in pd.date_range(first, end, inclusive='left'):
        minutes = 60 if day < pd.Timestamp('2025-10-01') else 15
        starts = delivery_grid(day, minutes, 'Europe/Rome')
        clock = pd.Timestamp(f"{(day - pd.Timedelta(days=1)).date()} 12:57", tz='Europe/Rome').tz_convert('UTC')
        for index, start in enumerate(starts):
            rows.append({'product_start': start, 'product_end': start + pd.Timedelta(minutes=minutes),
                         'available_at': clock, 'price_eur_per_mwh': float(index) - 10})
    return pd.DataFrame(rows)


def test_hourly_history_is_repeated_but_target_is_native():
    """October 1 sees only published September hourly history and 96 native targets."""
    frame = source()
    request, labels, coverage = prepare_day(PriceIndex(frame), frame, '2025-10-01', settings())
    assert coverage['hourly_context_products'] == 2688
    assert request['horizon'] == 96
    assert 'actual' not in request and 'target_values' not in request
    assert request['feature_names'] == ['past_NORD_DA_price']
    assert request['context'][:4] == [-10.0] * 4
    assert labels.actual.iloc[:4].tolist() == [-10.0, -9.0, -8.0, -7.0]
    assert pd.Timestamp(request['max_context_publication_at']) <= pd.Timestamp(request['decision_at'])


def test_future_labels_cannot_change_context_and_unpublished_history_excludes_day():
    """A large target perturbation never leaks into model inputs."""
    frame = source()
    request, _, _ = prepare_day(PriceIndex(frame), frame, '2025-10-01', settings())
    target = frame.product_start >= pd.Timestamp('2025-10-01', tz='Europe/Rome').tz_convert('UTC')
    frame.loc[target, 'price_eur_per_mwh'] = 1e6
    other, _, _ = prepare_day(PriceIndex(frame), frame, '2025-10-01', settings())
    assert request == other
    frame.loc[~target, 'available_at'] = pd.Timestamp('2025-10-02', tz='UTC')
    other, _, coverage = prepare_day(PriceIndex(frame), frame, '2025-10-01', settings())
    assert other is None and coverage['context_missing'] == 2688


def test_missing_history_and_cancellation_are_explicit():
    """Missing inputs exclude the day and cancelled target prices remain unscored."""
    frame = source()
    frame = frame.loc[frame.product_start.ne(pd.Timestamp('2025-09-10T10:00Z'))]
    request, _, coverage = prepare_day(PriceIndex(frame), frame, '2025-10-01', settings())
    assert request is None and coverage['context_missing'] == 4
    frame = source()
    request, labels, coverage = prepare_day(PriceIndex(frame), frame, '2025-10-01', settings(), cancelled=True)
    assert request is None and labels.actual.isna().all() and coverage['cancelled']


def test_autumn_products_and_reference_fallback_preserve_both_occurrences():
    """An autumn delivery day has 100 targets; each repeated-hour reference is explicit."""
    frame = source()
    request, labels, _ = prepare_day(PriceIndex(frame), frame, '2025-10-26', settings())
    assert request['horizon'] == 100 and len(labels) == 100
    local = pd.DatetimeIndex(labels.product_start).tz_convert('Europe/Rome')
    two = labels.loc[local.hour == 2]
    assert len(two) == 8
    np.testing.assert_array_equal(two.weekly_reference.iloc[:4], two.weekly_reference.iloc[4:])
    assert labels.product_start.is_unique
    next_day, _, _ = prepare_day(PriceIndex(frame), frame, '2025-10-27', settings())
    assert len(next_day['context']) == 2692


def test_spring_products_are_not_fabricated():
    """Native spring delivery contains only the actual 92 UTC products."""
    frame = source('2026-02-01', '2026-03-31')
    request, labels, _ = prepare_day(PriceIndex(frame), frame, '2026-03-29', settings())
    assert request['horizon'] == 92 and len(labels) == 92


def test_restart_rejects_modified_output_or_run_identity():
    """Previously completed results need matching fingerprints and intact content."""
    saved = {'run_identity': 'run', 'request_sha256': 'request', 'quantiles': [[-1.0]]}
    saved['payload_sha256'] = object_digest(saved)
    verify_prediction(saved, 'run', 'request')
    with pytest.raises(ValueError, match='different'):
        verify_prediction(saved, 'another_run', 'request')
    saved['quantiles'] = [[999]]
    with pytest.raises(ValueError, match='checksum'):
        verify_prediction(saved, 'run', 'request')


def snapshot(tmp_path):
    """Build an immutable synthetic catalog export for preparation/report integration."""
    root = tmp_path / 'snapshot'
    root.mkdir()
    frame = source(end='2025-10-04')
    frame = frame.rename(columns={'price_eur_per_mwh': 'value', 'available_at': 'published_at'})
    frame = frame.assign(dataset='day_ahead_prices', zone='NORD', unit='EUR/MWh')
    frame.to_parquet(root / 'prices.parquet', index=False)
    write_json(root / 'manifest.json', {'tables': {'nord_auctions': {
        'path': 'prices.parquet', 'sha256': digest(root / 'prices.parquet')}}})
    return root


def test_bundle_endpoint_coverage_and_checksum(tmp_path):
    """Preparation infers the last complete native day and checks input corruption."""
    root = snapshot(tmp_path)
    bundle = tmp_path / 'bundle'
    manifest = prepare_bundle(root, bundle, settings(), {'model_a': {}})
    assert manifest['eligible_days'] == 3
    assert manifest['settings']['end_exclusive'] == '2025-10-04'
    verify_bundle(bundle)
    path = bundle / 'requests/2025-10-01.json'
    path.write_text('{}')
    with pytest.raises(ValueError, match='checksum'):
        verify_bundle(bundle)


def test_report_common_products_and_incomplete_status(tmp_path):
    """Partial model runs cannot claim complete evaluation or compare different days."""
    bundle = tmp_path / 'bundle'
    prepare_bundle(snapshot(tmp_path), bundle, settings(), {'model_a': {}, 'model_b': {}})
    manifest = verify_bundle(bundle)
    run = tmp_path / 'run'
    for name in ['model_a', 'model_b']:
        write_json(run / name / 'shard-0-of-1' / 'identity.json', {'model': name})
    for model, dates in [('model_a', ['2025-10-01', '2025-10-02']), ('model_b', ['2025-10-01'])]:
        for day in dates:
            request = json.loads((bundle / f'requests/{day}.json').read_text())
            result = {'delivery_day': day, 'product_starts_utc': request['product_starts_utc'],
                      'quantiles': [[float(q) for q in range(9)] for _ in range(96)],
                      'request_sha256': manifest['files'][f'requests/{day}.json'],
                      'bundle_sha256': digest(bundle / 'manifest.json')}
            result['run_identity'] = object_digest({'model': model})
            result['payload_sha256'] = object_digest(result)
            write_json(run / model / 'shard-0-of-1' / 'predictions' / f'{day}.json', result)
    report(bundle, run, tmp_path / 'report')
    matched = pd.read_parquet(tmp_path / 'report/matched_forecasts.parquet')
    assert len(matched) == 192
    assert matched.delivery_day.nunique() == 1
    assert json.loads((tmp_path / 'report/status.json').read_text())['complete'] is False


def test_gpu_discovery_preserves_allocated_visibility(monkeypatch):
    """Physical idle GPU 3 maps to the correct ordinal inside a restricted allocation."""
    from types import SimpleNamespace
    from scripts.da_zero_shot_probe import available_devices
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '3,1')
    monkeypatch.setattr('scripts.da_zero_shot_probe.subprocess.run', lambda *a, **k: SimpleNamespace(
        stdout='0, GPU-a, 0, 0\n1, GPU-b, 2000, 80\n2, GPU-c, 0, 0\n3, GPU-d, 4, 0\n'))
    assert available_devices() == ['cuda:0']
    assert __import__('os').environ['CUDA_VISIBLE_DEVICES'] == '3,1'


def test_three_gpu_jobs_keep_models_independent():
    """Each frozen model uses one GPU; calendar days are batched within that worker."""
    from scripts.da_zero_shot_probe import forecast_jobs
    jobs = forecast_jobs(['chronos2', 't0_gamma', 'timesfm3'], ['cuda:0', 'cuda:1', 'cuda:2'])
    assert len(jobs) == 3
    assert len({job['device'] for job in jobs}) == 3
    assert all(job['shards'] == 1 for job in jobs)


def test_timesfm_adapter_disables_positive_clipping_and_uses_full_context(monkeypatch):
    """Model-free contract test ensures negative prices and complete histories survive."""
    import sys
    from types import SimpleNamespace
    class Forecaster:
        global_context = 15360
        def __init__(self, config):
            pass
        def predict_batch(self, contexts, **options):
            assert len(contexts[0]) == 2692
            assert options['make_positive'] is False
            assert options['use_symmetric_averaging'] is False
            return iter([SimpleNamespace(quantiles=np.full((100, 9), -20.0))])
    monkeypatch.setitem(sys.modules, 'timesfm3', SimpleNamespace(ModelConfig=lambda **kw: kw, TimesFM3Evaluator=Forecaster))
    predict = load_model('timesfm3', 'pinned-checkpoint', 'cpu', SimpleNamespace())
    result = predict(np.zeros(2692), 100, np.arange(1, 10) / 10)
    assert (result == -20).all()


def test_launcher_preserves_virtual_environment_interpreter_symlinks(tmp_path):
    """Resolving a venv executable to its base Python would silently load the wrong packages."""
    import subprocess
    import sys
    bundle = tmp_path / 'bundle'
    prepare_bundle(snapshot(tmp_path), bundle, settings(), {'chronos2': {}})
    executable = tmp_path / 'envs/chronos/bin/python'
    executable.parent.mkdir(parents=True)
    executable.symlink_to(sys.executable)
    completed = subprocess.run([sys.executable, '-m', 'scripts.da_zero_shot_probe', 'run',
        '--bundle', str(bundle), '--output', str(tmp_path / 'run'), '--python-root', str(tmp_path / 'envs'),
        '--devices', 'cpu'], capture_output=True, text=True, check=True)
    plan = json.loads(completed.stdout)
    assert plan['action'] == 'plan_only'
    assert plan['commands'][0][1][0] == str(executable)


def test_batching_preserves_dst_shapes_and_fixed_restart_membership():
    """Only equal-sized daily contexts and target horizons share a forward pass."""
    from scripts.run_da_zero_shot_model import request_batches
    requests = [(str(index), 'hash', {'context': [0] * length, 'horizon': horizon})
                for index, (length, horizon) in enumerate([(2688, 96), (2688, 96), (2688, 100), (2692, 96)])]
    batches = list(request_batches(requests, 8))
    assert [[item[0] for item in batch] for batch in batches] == [['0', '1'], ['2'], ['3']]
    assert list(request_batches(requests, 1))[0][0][0] == '0'
