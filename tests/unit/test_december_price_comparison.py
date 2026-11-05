"""Check December comparison configuration and paired reporting."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.participation import participation_profiles

ROOT = Path(__file__).resolve().parents[2]


def _runner():
    """Load the standalone study runner without executing its CLI."""
    spec = importlib.util.spec_from_file_location('december_runner', ROOT / 'scripts/run_december_price_comparison.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_december_configuration_information_and_profiles():
    """Both arms use the same physical settings and eight DA profiles."""
    for farm in (1, 345, 901, 910):
        configs = [load_config(ROOT / f'configs/research/december_2025_price_{mode}_farm_{farm}.yaml')
                   for mode in ('forecast', 'oracle')]
        for config in configs:
            profiles = participation_profiles(config['continuous_participation_study'])
            assert len(profiles) == 8
            assert all('DA' in profile for profile in profiles)
            assert config['pv']['output']['method'] == 'explicit_efficiency'
            assert config['execution']['dispatch_policy'] == 'track_committed_grid'
        for key in ('battery', 'plant', 'state', 'degradation', 'pv', 'policy', 'trading', 'settlement'):
            assert configs[0][key] == configs[1][key]
        assert _runner().matched_settings(configs[0]) == _runner().matched_settings(configs[1])
        assert configs[0]['energy_price_forecast']['auction_enabled']
        assert not configs[1]['energy_price_forecast']['auction_enabled']


def test_report_pairs_profits_and_components(tmp_path, monkeypatch):
    """Pair monthly outcomes only after both source manifests are complete."""
    runner = _runner()
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    monkeypatch.setattr(runner, 'FARMS', (1,))
    profiles = runner.expected_profiles(load_config(runner.config_path(1, 'forecast')))
    (tmp_path / 'preflight').mkdir()
    for mode in ('forecast', 'oracle'):
        root = tmp_path / 'farm_1' / f'december_2025_price_{mode}_v1'
        (root / 'json').mkdir(parents=True)
        (root / 'csv').mkdir()
        config = load_config(runner.config_path(1, mode))
        evidence = {'status': 'complete', 'completed': profiles,
                    'delivery_days': [f'2025-12-{day:02}' for day in range(1, 32)],
                    'input_fingerprint': mode, 'config_fingerprint': mode}
        (root / 'json/manifest.json').write_text(json.dumps(evidence))
        provenance = {'days': {day: {'pv_provenance': {'curated_realized_pv': {
            'source_counts': {'metered_ac': 96}}}} for day in evidence['delivery_days']}}
        (root / 'json/input_provenance.json').write_text(json.dumps(provenance))
        (root / 'json/study_config.json').write_text(json.dumps(config))
        (tmp_path / 'preflight' / f'farm_1_{mode}.json').write_text(json.dumps(
            {**evidence, 'days': evidence['delivery_days']}))
        for profile in profiles:
            diagnostics = root / 'strategies' / profile / 'json'
            diagnostics.mkdir(parents=True)
            (diagnostics / 'optimization_attempts.json').write_text(json.dumps(
                [{'status': 'optimal', 'mip_gap': 0.0, 'fallback_applied': False}]))
        profit = 310 if mode == 'forecast' else 341
        pd.DataFrame({'profile': profiles, 'total_net_profit_eur': [profit]*9,
                      'final_soc_percent': [50]*9, 'final_energy_mwh': [0.375]*9,
                      'fallback_count': [0]*9}).to_csv(root / 'csv/monthly_participation.csv', index=False)
        rows = [{'profile': profile, 'delivery_day': f'2025-12-{day:02}',
                 **{key: 0 for key in runner.COMPONENTS}, 'auction_revenue_eur': profit / 31,
                 'net_profit_eur': profit / 31, 'initial_energy_mwh': 0.375, 'final_energy_mwh': 0.375}
                for profile in profiles for day in range(1, 32)]
        pd.DataFrame(rows).to_csv(root / 'csv/daily_participation.csv', index=False)
    runner.report()
    paired = pd.read_csv(tmp_path / 'paired_monthly.csv')
    assert set(paired.profit_forecast_minus_oracle_eur) == {-31}
    assert set(paired.auction_revenue_eur_difference) == {-31}
    summary = (tmp_path / 'summary.md').read_text()
    assert 'Settlement | Degradation | Import fee | Trading fee' in summary
    assert 'retrospective' in summary

    # A partial audit cannot be presented as validation of the paired study.
    audit = {'status': 'passed', 'trajectory_count': 17, 'totals': {
        'auction_price_curve_checks': 1, 'pv_daily_fingerprint_checks': 1,
        'settlement_fingerprint_checks': 1}}
    (tmp_path / 'ledger_validation.json').write_text(json.dumps(audit))
    with pytest.raises(ValueError, match='Ledger validation evidence is incomplete'):
        runner.report()
    audit['trajectory_count'] = 18
    (tmp_path / 'ledger_validation.json').write_text(json.dumps(audit))
    runner.report()
    assert 'Independent ledger audit passed for 18 trajectories' in (tmp_path / 'summary.md').read_text()

    # Corruptions must be rejected even when row counts are unchanged.
    daily_path = root / 'csv/daily_participation.csv'
    frame = pd.read_csv(daily_path)
    frame.loc[0, 'delivery_day'] = '2026-01-01'
    frame.to_csv(daily_path, index=False)
    with pytest.raises(ValueError, match='Incomplete December dates'):
        runner.report()


@pytest.mark.parametrize('corruption,message', [
    ('profit', 'cash components'), ('state', 'Discontinuous SOC'),
    ('nonfinite', 'Nonfinite'), ('profile', 'participation profiles'),
    ('monthly', 'Monthly profit')])
def test_result_validation_rejects_invalid_accounting(corruption, message):
    """Counts alone cannot establish a valid, continuous monthly cash comparison."""
    runner = _runner()
    daily = pd.DataFrame([{'profile': 'DA', 'delivery_day': f'2025-12-{day:02}',
        **{key: 0.0 for key in runner.COMPONENTS}, 'auction_revenue_eur': 10.0,
        'net_profit_eur': 10.0, 'initial_energy_mwh': 0.375, 'final_energy_mwh': 0.375}
        for day in range(1, 32)])
    monthly = pd.DataFrame([{'profile': 'DA', 'total_net_profit_eur': 310.0, 'final_energy_mwh': 0.375}])
    if corruption == 'profit':
        daily.loc[0, 'net_profit_eur'] = 100.0
    elif corruption == 'state':
        daily.loc[1, 'initial_energy_mwh'] = 0.4
    elif corruption == 'nonfinite':
        daily.loc[0, 'degradation_eur'] = float('nan')
    elif corruption == 'profile':
        monthly.loc[0, 'profile'] = 'wrong'
    else:
        monthly.loc[0, 'total_net_profit_eur'] = 0.0
    with pytest.raises(ValueError, match=message):
        runner.validate_tables(monthly, daily, ['DA'])


def test_farm_runner_bounds_threads_and_runs_arms_serially(tmp_path, monkeypatch):
    """Inherited parallelism must not multiply the two-farm concurrency budget."""
    runner = _runner()
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    monkeypatch.setenv('BESS_PARTICIPATION_WORKERS', '48')
    monkeypatch.setenv('OPENBLAS_NUM_THREADS', '48')
    calls = []
    monkeypatch.setattr(runner.subprocess, 'run', lambda command, **options: calls.append((command, options)))
    runner.run_farm(1)
    assert len(calls) == 2
    for mode, (command, options) in zip(runner.MODES, calls):
        assert command[-1] == str(runner.config_path(1, mode))
        assert options['check']
        assert options['env']['BESS_PARTICIPATION_WORKERS'] == '1'
        assert all(options['env'][name] == '1' for name in runner.THREAD_VARIABLES)
        assert options['env']['PYTHONPATH'] == str(runner.ROOT / 'src')
