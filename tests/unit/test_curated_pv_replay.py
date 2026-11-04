"""Verify curated replay fills only reviewed exclusions and retains provenance."""
import hashlib
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.simulation.curated_pv import load_curated_realized_pv


def test_curated_replay_uses_reviewed_targets_and_exclusion_reference(tmp_path):
    config = load_config('configs/standard.yaml')
    config['forecast']['curated_realized'] = {'enabled': True, 'dataset_root': str(tmp_path)}
    times = pd.DatetimeIndex(['2026-01-05T09:00Z', '2026-01-05T09:15Z',
                              '2026-01-05T09:30Z', '2026-01-05T23:00Z'])
    frame = pd.DataFrame({'interval_start_utc': times, 'split': 'test',
        'night': [False, False, False, True], 'model_target_mw': [1., 2., 3., 4.],
        'model_target_source': ['metered_ac', 'pvlib_historical_ghi', 'metered_ac', 'metered_ac'],
        'retained_prior_exclusion': [False, False, True, False],
        'pvlib_historical_ghi_mw': [5., 5., 2.5, 0.]})
    folder = tmp_path / 'farm_1'
    folder.mkdir()
    corpus = folder / 'corpus.parquet'
    frame.to_parquet(corpus, index=False)
    frame.to_parquet(folder / 'modeling_views.parquet', index=False)
    sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps({'farms': {'1': {'corpus_sha256': sha}}}))
    grid = SimpleNamespace(starts=times, interval_ids=['a', 'b', 'c', 'd'])
    values, provenance = load_curated_realized_pv(config, 1, grid)
    np.testing.assert_allclose(values, [1., 2., 2.5, 0.])
    assert [row['source'] for row in provenance['interval_sources']] == [
        'metered_ac', 'pvlib_historical_ghi', 'pvlib_retained_exclusion', 'monthly_night_zero']
    # A missing reference is an explicit gap, not a fabricated zero or forecast.
    frame.loc[2, 'pvlib_historical_ghi_mw'] = np.nan
    frame.to_parquet(corpus, index=False)
    frame.to_parquet(folder / 'modeling_views.parquet', index=False)
    sha = hashlib.sha256(corpus.read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps({'farms': {'1': {'corpus_sha256': sha}}}))
    with pytest.raises(ValueError, match='remain unresolved'):
        load_curated_realized_pv(config, 1, grid)


def test_pv_loader_uses_curated_replay_without_reading_raw_observations(monkeypatch):
    """The optimization execution trajectory comes only from the selected dataset."""
    from bess_pv_trading.simulation.gme_oracle import _load_pv

    config = load_config('configs/standard.yaml', overrides={'farm_id': 1})
    config['forecast']['curated_realized'] = {'enabled': True, 'dataset_root': 'unused'}
    times = pd.date_range('2026-05-01T10:00Z', periods=4, freq='15min')
    grid = SimpleNamespace(starts=times, interval_ids=list('abcd'), delta_hours=np.full(4, .25))
    cycle = pd.Timestamp('2026-04-30T00:00Z')
    cutoff = pd.Timestamp('2026-04-30T10:00Z')
    pv = SimpleNamespace(values=pd.DataFrame({'pv_ac_mw': [.1] * 4}), forecast_vintage=cycle,
        available_at=cycle + pd.Timedelta(hours=6), members=['deterministic'], member_origin='test',
        provenance={'forecast_missing_filled_night_zero_intervals': 0})
    @dataclass
    class Farm:
        ac_capacity_mw: float = 1.
    monkeypatch.setattr('bess_pv_trading.simulation.gme_oracle.open_store',
                        lambda config: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr('bess_pv_trading.simulation.gme_oracle.load_metadata', lambda *args: {1: Farm()})
    monkeypatch.setattr('bess_pv_trading.simulation.gme_oracle._forecast_pv_at_cutoff', lambda *args: pv)
    monkeypatch.setattr('bess_pv_trading.simulation.gme_oracle.load_observations',
                        lambda *args, **kwargs: pytest.fail('raw observations must not be loaded'))
    monkeypatch.setattr('bess_pv_trading.simulation.curated_pv.load_curated_realized_pv',
                        lambda *args: (np.array([.2, .3, .4, .5]), {'source_counts': {'metered_ac': 4},
                            'raw_missing_intervals': 0}))
    _, actual, _, provenance = _load_pv(config, None, grid, decision_cutoff=cutoff)
    np.testing.assert_allclose(actual, [.2, .3, .4, .5])
    assert provenance['curated_realized_pv']['source_counts'] == {'metered_ac': 4}
    assert provenance['realized_power_fallback_interval_ids'] == []


def test_curated_source_is_cached_per_unchanged_farm(tmp_path, monkeypatch):
    """Many replay days must not reread and rehash the same farm corpus."""
    from bess_pv_trading.simulation import curated_pv

    config = load_config('configs/standard.yaml')
    config['forecast']['curated_realized'] = {'enabled': True, 'dataset_root': str(tmp_path)}
    times = pd.DatetimeIndex(['2026-01-05T09:00Z', '2026-01-05T09:15Z'])
    frame = pd.DataFrame({'interval_start_utc': times, 'split': 'test', 'night': False,
        'model_target_mw': [1., 2.], 'model_target_source': 'metered_ac',
        'measured_ac_raw_mw': [1., 2.], 'retained_prior_exclusion': False,
        'pvlib_historical_ghi_mw': [1., 2.]})
    folder = tmp_path / 'farm_1'
    folder.mkdir()
    corpus = folder / 'corpus.parquet'
    frame.to_parquet(corpus, index=False)
    frame.to_parquet(folder / 'modeling_views.parquet', index=False)
    checksum = hashlib.sha256(corpus.read_bytes()).hexdigest()
    (tmp_path / 'manifest.json').write_text(json.dumps({'farms': {'1': {'corpus_sha256': checksum}}}))
    grid = SimpleNamespace(starts=times, interval_ids=['a', 'b'])
    original = pd.read_parquet
    reads = []
    def counted(path, *args, **kwargs):
        reads.append(path)
        return original(path, *args, **kwargs)
    monkeypatch.setattr(curated_pv.pd, 'read_parquet', counted)
    first, _ = load_curated_realized_pv(config, 1, grid)
    second, _ = load_curated_realized_pv(config, 1, grid)
    np.testing.assert_array_equal(first, second)
    assert len(reads) == 2
