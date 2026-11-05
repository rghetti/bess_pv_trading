"""Summarize a completed matched curtailment-recovery MPC study."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(root):
    """Pair completed controller outcomes and report actual recovery and costs."""
    root = Path(root)
    manifest = json.loads((root / 'manifest.json').read_text())
    jobs = manifest['results']
    if manifest['status'] != 'complete' or len(jobs) != 8:
        raise ValueError('The complete eight-job comparison is required')
    records, daily = [], []
    for job in jobs:
        if job['status'] != 'complete':
            raise ValueError(f"Incomplete job: {job['job_id']}")
        directory = Path(job['result']['output_dir'])
        rows = pd.read_csv(directory / 'controller_comparison.csv').set_index('controller')
        for controller in ('track_committed_grid', 'deterministic_mpc'):
            intervals = pd.read_csv(directory / 'controllers' / controller / 'csv/controller_intervals.csv')
            if len(intervals) != 7 * 96:
                raise ValueError(f'Unexpected interval count: {directory} {controller}')
            recovery = intervals.curtailment_recovered_mw * intervals.duration_hours
            item = {'farm': str(job['farm_id']), 'case': job['case_id'], 'controller': controller,
                'recovered_pv_mwh': float(recovery.sum()), 'recovery_intervals': int((recovery > 1e-8).sum())}
            keys = ('absolute_imbalance_mwh', 'imbalance_cashflow_eur', 'auction_revenue_eur',
                'degradation_eur', 'import_cost_eur', 'trading_fees_eur', 'realized_operating_profit_eur',
                'pv_curtailment_mwh', 'throughput_mwh', 'final_energy_mwh', 'soc_final_percent',
                'mpc_solves', 'mpc_fallbacks', 'mpc_timeouts', 'physical_safety_overrides')
            item.update({key: float(rows.loc[controller, key]) for key in keys})
            records.append(item)
            for day, frame in intervals.assign(recovered_pv_mwh=recovery).groupby('delivery_day'):
                daily.append({'farm': str(job['farm_id']), 'case': job['case_id'], 'controller': controller,
                    'delivery_day': day, 'recovered_pv_mwh': float(frame.recovered_pv_mwh.sum()),
                    'curtailment_mwh': float((frame.curtailment_mw * frame.duration_hours).sum()),
                    'absolute_imbalance_mwh': float(frame.imbalance_mwh.abs().sum()),
                    'profit_eur': float(frame.net_profit_eur.sum()), 'energy_end_mwh': float(frame.energy_end_mwh.iloc[-1])})
    table = pd.DataFrame(records)
    daily = pd.DataFrame(daily)
    target = table[table.controller.eq('deterministic_mpc')].set_index(['farm', 'case'])
    fields = ['absolute_imbalance_mwh','imbalance_cashflow_eur','auction_revenue_eur','degradation_eur',
              'import_cost_eur','trading_fees_eur','realized_operating_profit_eur','pv_curtailment_mwh',
              'throughput_mwh','final_energy_mwh','soc_final_percent','recovered_pv_mwh']
    contrasts = []
    for farm in ('1','345','901','910'):
        old = target.loc[(farm, 'mpc_original')]
        new = target.loc[(farm, 'mpc_recovery')]
        difference = {field: float(new[field] - old[field]) for field in fields}
        profit = (difference['auction_revenue_eur'] + difference['imbalance_cashflow_eur']
            - difference['degradation_eur'] - difference['import_cost_eur'] - difference['trading_fees_eur'])
        if not np.isclose(difference['realized_operating_profit_eur'], profit, atol=1e-5):
            raise ValueError(f'Profit reconciliation failed for farm {farm}')
        contrasts.append({'farm': farm, **difference})
    contrast = pd.DataFrame(contrasts)
    table.to_csv(root / 'recovery_controller_totals.csv', index=False)
    daily.to_csv(root / 'recovery_daily_metrics.csv', index=False)
    contrast.to_csv(root / 'recovery_paired_differences.csv', index=False)
    lines = ['# One-week MPC curtailment-recovery comparison', '',
        '15–21 January 2026; four farms; continuous closed-loop trajectories. '
        'Changes are recovery minus the original MPC.', '',
        '| Farm | Recovered PV MWh | Curtailment change MWh | Imbalance change MWh | Wear change EUR | Profit change EUR | Final energy change MWh |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in contrast.itertuples(index=False):
        lines.append(f'| {row.farm} | {row.recovered_pv_mwh:.3f} | {row.pv_curtailment_mwh:+.3f} | '
            f'{row.absolute_imbalance_mwh:+.3f} | {row.degradation_eur:+.2f} | '
            f'{row.realized_operating_profit_eur:+.2f} | {row.final_energy_mwh:+.3f} |')
    lines.extend(['', '| Farm | Tracking imbalance MWh | Original MPC MWh | Recovery MPC MWh | Original final SOC % | Recovery final SOC % |',
        '| --- | ---: | ---: | ---: | ---: | ---: |'])
    tracker = table[table.controller.eq('track_committed_grid')].set_index(['farm','case'])
    for farm in ('1','345','901','910'):
        old, new = target.loc[(farm,'mpc_original')], target.loc[(farm,'mpc_recovery')]
        baseline = tracker.loc[(farm,'mpc_original')]
        lines.append(f'| {farm} | {baseline.absolute_imbalance_mwh:.3f} | {old.absolute_imbalance_mwh:.3f} | '
            f'{new.absolute_imbalance_mwh:.3f} | {old.soc_final_percent:.2f} | {new.soc_final_percent:.2f} |')
    lines.extend(['', 'The recovery action preserves protected grid exchange at that interval; subsequent '
        'SOC, auction orders and dispatch may differ. Recovered PV is AC energy; battery energy also reflects '
        'efficiency and avoided discharge. Planning penalties are not actual charges. '
        'The one-week sample, old validation-selected PV models, current-interval average PV disclosure '
        'and perfect auction-price planning make this a retrospective development comparison. '
        'Final SOC must be inspected against the final-study target.', ''])
    (root / 'recovery_summary.md').write_text('\n'.join(lines))
    return contrast


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    print(summarize(parser.parse_args().root).to_string(index=False))
