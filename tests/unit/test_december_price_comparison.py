"""Check December comparison configuration and paired reporting."""
import importlib.util
import json
from pathlib import Path

import pandas as pd

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
        assert configs[0]['energy_price_forecast']['auction_enabled']
        assert not configs[1]['energy_price_forecast']['auction_enabled']


def test_report_pairs_profits_and_components(tmp_path, monkeypatch):
    """Pair monthly outcomes only after both source manifests are complete."""
    runner = _runner()
    monkeypatch.setattr(runner, 'OUTPUT', tmp_path)
    monkeypatch.setattr(runner, 'FARMS', (1,))
    profiles = [f'profile_{i}' for i in range(9)]
    fields = ['auction_revenue_eur', 'imbalance_cashflow_eur', 'degradation_eur',
              'import_fee_eur', 'transaction_fee_eur', 'absolute_imbalance_mwh',
              'charge_mwh', 'discharge_mwh', 'curtailment_mwh']
    for mode in ('forecast', 'oracle'):
        root = tmp_path / 'farm_1' / f'december_2025_price_{mode}_v1'
        (root / 'json').mkdir(parents=True)
        (root / 'csv').mkdir()
        (root / 'json/manifest.json').write_text(json.dumps({'status': 'complete'}))
        profit = 100 if mode == 'forecast' else 110
        pd.DataFrame({'profile': profiles, 'total_net_profit_eur': [profit]*9,
                      'final_soc_percent': [50]*9}).to_csv(root / 'csv/monthly_participation.csv', index=False)
        rows = [{'profile': profile, 'delivery_day': f'2025-12-{day:02}', **{key: 1 for key in fields}}
                for profile in profiles for day in range(1, 32)]
        pd.DataFrame(rows).to_csv(root / 'csv/daily_participation.csv', index=False)
    runner.report()
    paired = pd.read_csv(tmp_path / 'paired_monthly.csv')
    assert set(paired.profit_forecast_minus_oracle_eur) == {-10}
    assert set(paired.auction_revenue_eur_difference) == {0}
