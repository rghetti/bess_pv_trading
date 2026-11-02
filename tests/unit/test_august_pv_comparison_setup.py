"""The August launcher selects matching, isolated farm studies."""
from pathlib import Path
import importlib.util
import sys

import pytest

from bess_pv_trading.config import load_config


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('farm_id', [1, 345, 901, 910])
def test_august_farm_configs_match_and_use_complete_ensemble_period(farm_id):
    paths = [ROOT / f'configs/full_experiments/continuous_2026_08_pv_{kind}_four_farms.yaml'
             for kind in ('deterministic', 'scenarios')]
    deterministic, scenarios = [load_config(p, overrides={'farm_id': farm_id}) for p in paths]
    assert deterministic['pv'] == scenarios['pv']
    assert deterministic['plant'] == scenarios['plant']
    assert deterministic['battery'] == scenarios['battery']
    assert deterministic['state'] == scenarios['state']
    for config in (deterministic, scenarios):
        study = config['continuous_optimization']
        assert (study['start_date'], study['end_date']) == ('2026-08-02', '2026-08-31')
        assert config['trading']['optimize_imbalances'] is False
    study = scenarios['continuous_optimization']
    assert study['pv_ensemble']['member_count'] == 50
    assert study['pv_ensemble']['zarr'].endswith('nwp_ens_20260801_20260930.zarr')
    assert study['perfect_pv_reference'] is True
    assert study['perfect_pv_reoptimize_at_each_auction'] is False
    assert deterministic['continuous_optimization']['perfect_pv_reference'] is False
    assert (scenarios['pv']['forecast_replacement']['models_by_farm'][str(farm_id)] is None) == (farm_id == 345)


def test_august_launcher_runs_both_replays_then_compares(monkeypatch):
    scripts = ROOT / 'scripts'
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location('august_pv_launcher', scripts / 'run_august_pv_farm_comparison.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []
    monkeypatch.setattr(module.subprocess, 'run', lambda command, **kwargs: calls.append(command))
    monkeypatch.setattr(module, 'compare', lambda *paths: calls.append(paths))
    monkeypatch.setattr(sys, 'argv', ['launcher', '--farm-id', '345'])
    module.main()
    assert len(calls) == 3
    assert all('--farm-id' in command and command[-1] == '345' for command in calls[:2])
    assert '2026_08_mixed_models_deterministic_exact' in str(calls[2][0])
    assert '2026_08_mixed_models_scenarios_exact' in str(calls[2][1])
    assert 'farm_345/2026_08_mixed_models_pv_comparison' in str(calls[2][2])
