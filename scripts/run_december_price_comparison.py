"""Preflight, run, and report the December auction-price participation study."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from bess_pv_trading.config import load_config

FARMS = (1, 345, 901, 910)
MODES = ('forecast', 'oracle')
OUTPUT = ROOT / 'outputs/continuous_optimization/december_2025_price_study'


def config_path(farm, mode):
    """Resolve one of the declared per-farm comparison configurations."""
    return ROOT / 'configs/research' / f'december_2025_price_{mode}_farm_{farm}.yaml'


def preflight(farm):
    """Prepare both information modes without fitting or optimizing."""
    from bess_pv_trading.simulation.continuous_inputs import prepare_continuous_inputs
    for mode in MODES:
        config = load_config(config_path(farm, mode))
        prepared = prepare_continuous_inputs(config, progress=lambda msg: print(f'farm {farm} {mode}: {msg}', flush=True))
        print(f"farm {farm} {mode}: preflight passed ({len(prepared['days'])} days, {prepared['input_fingerprint']})", flush=True)


def run_farm(farm):
    """Run the two modes serially so each worker owns one farm at a time."""
    env = dict(os.environ)
    env['BESS_PARTICIPATION_WORKERS'] = '1'
    for mode in MODES:
        print(f'Starting farm {farm}, {mode}', flush=True)
        subprocess.run([sys.executable, '-m', 'bess_pv_trading', 'experiment-continuous',
                        '--config', str(config_path(farm, mode))], cwd=ROOT, env=env, check=True)


def report():
    """Join only completed monthly trajectories and write a paired summary."""
    import numpy as np
    import pandas as pd

    records = []
    daily_parts = []
    for farm in FARMS:
        for mode in MODES:
            directory = OUTPUT / f'farm_{farm}' / f'december_2025_price_{mode}_v1'
            manifest = json.loads((directory / 'json/manifest.json').read_text())
            if manifest['status'] != 'complete':
                raise ValueError(f'Incomplete source run: {directory}')
            monthly = pd.read_csv(directory / 'csv/monthly_participation.csv')
            daily = pd.read_csv(directory / 'csv/daily_participation.csv')
            if len(monthly) != 9 or len(daily) != 9 * 31 or daily.duplicated(['profile', 'delivery_day']).any():
                raise ValueError(f'Incomplete participation results: {directory}')
            component_fields = ['auction_revenue_eur', 'imbalance_cashflow_eur', 'degradation_eur',
                                'import_fee_eur', 'transaction_fee_eur', 'absolute_imbalance_mwh',
                                'charge_mwh', 'discharge_mwh', 'curtailment_mwh']
            sums = daily.groupby('profile', as_index=False)[component_fields].sum()
            monthly = monthly.merge(sums, on='profile', validate='one_to_one')
            monthly['farm'] = farm
            monthly['mode'] = mode
            records.append(monthly)
            daily['farm'] = farm
            daily['mode'] = mode
            daily_parts.append(daily)
    table = pd.concat(records, ignore_index=True)
    forecast = table[table['mode'].eq('forecast')].set_index(['farm', 'profile'])
    oracle = table[table['mode'].eq('oracle')].set_index(['farm', 'profile'])
    if not forecast.index.equals(oracle.index):
        oracle = oracle.reindex(forecast.index)
    if oracle.total_net_profit_eur.isna().any():
        raise ValueError('Forecast and oracle profiles do not match')
    paired = forecast.join(oracle, lsuffix='_forecast', rsuffix='_oracle')
    paired['profit_forecast_minus_oracle_eur'] = paired.total_net_profit_eur_forecast - paired.total_net_profit_eur_oracle
    for field in ['auction_revenue_eur', 'imbalance_cashflow_eur', 'degradation_eur',
                  'import_fee_eur', 'transaction_fee_eur', 'absolute_imbalance_mwh']:
        paired[f'{field}_difference'] = paired[f'{field}_forecast'] - paired[f'{field}_oracle']
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paired.reset_index().to_csv(OUTPUT / 'paired_monthly.csv', index=False)
    pd.concat(daily_parts, ignore_index=True).to_csv(OUTPUT / 'daily_all_farms.csv', index=False)
    lines = ['# December 2025: forecast versus perfect auction prices', '',
        'The forecast model is a frozen pre-October-2025 ridge model per auction, with causally updated price lags. '
        'The perfect-price arm knows actual auction curves in planning; both arms use actual settlements and independent continuous trajectories.', '',
        '| Farm | Profile | Forecast profit EUR | Oracle profit EUR | Difference EUR | Forecast final SOC % | Oracle final SOC % |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for (farm, profile), row in paired.iterrows():
        vals = [row.total_net_profit_eur_forecast, row.total_net_profit_eur_oracle,
                row.profit_forecast_minus_oracle_eur, row.final_soc_percent_forecast, row.final_soc_percent_oracle]
        if not np.isfinite(vals).all():
            raise ValueError('Nonfinite report values')
        lines.append(f'| {farm} | {profile} | ' + ' | '.join(f'{x:.2f}' for x in vals) + ' |')
    lines.extend(['', 'The model produces point forecasts, not stochastic price scenarios. '
        'PV uses unfitted pvlib conversion and the reviewed counterfactual execution targets; '
        'forecasts follow the configured night-zero and six-hour weather availability rules. '
        'The comparison is retrospective; price publication times and some source availability are assumed. '
        'Final SOC is shown because remaining inventory has no cash value in profit.', ''])
    (OUTPUT / 'summary.md').write_text('\n'.join(lines))
    print(f'Report: {OUTPUT / "summary.md"}', flush=True)


def main():
    """Use two farm workers, then produce a report after every run completes."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    if args.report_only:
        report()
        return
    action = preflight if args.check_only else run_farm
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(action, FARMS))
    if args.run:
        report()


if __name__ == '__main__':
    main()
