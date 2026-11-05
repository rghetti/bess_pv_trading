"""Audit and summarize a completed current-imbalance priority comparison."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import pandas as pd


def report(root):
    """Reconstruct objective values and pair actual outcomes without new solves."""
    root = Path(root)
    batch = json.loads((root / 'manifest.json').read_text())
    if batch['status'] != 'complete' or len(batch['results']) != 12:
        raise ValueError('The complete twelve-job study is required')
    records, daily, diagnostics, trackers, violations = [], [], [], {}, []
    for job in batch['results']:
        if job['status'] != 'complete':
            raise ValueError(f"Incomplete job: {job['job_id']}")
        config = json.loads((root / f"config__{job['job_id']}.json").read_text())
        directory = Path(job['result']['output_dir'])
        if not directory.is_dir():
            directory = root / 'runs' / f"farm_{job['farm_id']}" / job['job_id'] / 'continuous'
        summary = pd.read_csv(directory / 'controller_comparison.csv').set_index('controller')
        farm, case = str(job['farm_id']), job['case_id']
        for controller in ('track_committed_grid', 'deterministic_mpc'):
            folder = directory / 'controllers' / controller
            intervals = pd.read_csv(folder / 'csv/controller_intervals.csv')
            if len(intervals) != 672 or intervals.interval_id.duplicated().any():
                raise ValueError(f'Unexpected interval grid: {job["job_id"]} {controller}')
            if controller == 'track_committed_grid':
                columns = ['net_position_mwh', 'grid_mw', 'energy_end_mwh', 'imbalance_mwh', 'net_profit_eur']
                previous = trackers.setdefault(farm, intervals[columns].copy())
                np.testing.assert_allclose(previous, intervals[columns], atol=1e-7, rtol=0)
            keys = ('absolute_imbalance_mwh', 'imbalance_cashflow_eur', 'auction_revenue_eur',
                'degradation_eur', 'import_cost_eur', 'trading_fees_eur', 'realized_operating_profit_eur',
                'pv_curtailment_mwh', 'throughput_mwh', 'final_energy_mwh', 'soc_final_percent',
                'mpc_solves', 'mpc_fallbacks', 'mpc_timeouts', 'physical_safety_overrides',
                'solver_runtime_total_seconds')
            value = {'farm': farm, 'case': case, 'controller': controller}
            value.update({key: float(summary.loc[controller, key]) for key in keys})
            records.append(value)
            for day, frame in intervals.groupby('delivery_day'):
                daily.append({'farm': farm, 'case': case, 'controller': controller, 'delivery_day': day,
                    'absolute_imbalance_mwh': float(frame.imbalance_mwh.abs().sum()),
                    'signed_settlement_eur': float(frame.imbalance_cashflow_eur.sum()),
                    'profit_eur': float(frame.net_profit_eur.sum()), 'energy_end_mwh': float(frame.energy_end_mwh.iloc[-1])})
            if controller != 'deterministic_mpc':
                continue
            solves = json.loads((folder / 'json/mpc_solves.json').read_text())
            if len(solves) != len(intervals):
                raise ValueError(f'Incomplete MPC diagnostics for {job["job_id"]}')
            stage_count = Counter()
            max_objective_error = 0.0
            max_cap_excess = 0.0
            for solve in solves:
                stage_count[solve['status']] += 1
                stage_count['fallbacks'] += int(bool(solve.get('fallback_applied')))
                stage_count['stage1_certified'] += int(bool(solve.get('first_interval_minimum_certified')))
                stage_count['stage2_used'] += int(bool(solve.get('first_interval_refinement_used')))
                if solve.get('fallback_applied'):
                    continue
                components = solve['components']
                first = float(solve['first_interval_expected_imbalance_mwh'])
                costs = sum(components[key] for key in ('degradation_eur','import_cost_eur','terminal_cost_eur'))
                settings = config['realtime_mpc']
                weight = float(settings.get('first_interval_weight', 1.0))
                penalty = float(settings['planning_imbalance_penalty_eur_per_mwh'])
                expected = penalty * (components['absolute_imbalance_mwh'] + (weight - 1) * first) + costs
                if case == 'strict_first' and not solve.get('first_interval_refinement_used'):
                    expected = first
                error = abs(expected - solve['planning_objective_value'])
                max_objective_error = max(max_objective_error, error)
                if error > 1e-5:
                    violations.append({'job_id': job['job_id'], 'interval': solve['interval_id'],
                        'reason': 'objective_decomposition', 'error': error})
                if case == 'strict_first' and solve.get('first_interval_minimum_certified'):
                    excess = max(0.0, first - solve['first_interval_cap_mwh'])
                    max_cap_excess = max(max_cap_excess, excess)
                    tolerance = 2 * config['solver']['feasibility_tolerance'] + 1e-8
                    if excess > tolerance:
                        violations.append({'job_id': job['job_id'], 'interval': solve['interval_id'],
                            'reason': 'first_interval_cap', 'excess_mwh': excess})
                if case == 'strict_first' and not solve.get('first_interval_minimum_certified') and solve.get('first_interval_refinement_used'):
                    violations.append({'job_id': job['job_id'], 'interval': solve['interval_id'],
                        'reason': 'uncertified_first_stage_refined'})
            diagnostics.append({'farm': farm, 'case': case, 'solve_status_counts': dict(stage_count),
                'max_objective_error_eur_or_mwh': max_objective_error, 'max_first_cap_excess_mwh': max_cap_excess})
    if violations:
        raise ValueError(f'First-interval priority audit failed: {violations[:5]}')
    totals, days = pd.DataFrame(records), pd.DataFrame(daily)
    totals.to_csv(root / 'first_priority_controller_totals.csv', index=False)
    days.to_csv(root / 'first_priority_daily_metrics.csv', index=False)
    (root / 'first_priority_objective_audit.json').write_text(json.dumps({'jobs': diagnostics, 'violations': violations}, indent=2) + '\n')
    target = totals[totals.controller.eq('deterministic_mpc')].set_index(['farm','case'])
    contrasts = []
    for farm in ('1','345','901','910'):
        old = target.loc[(farm, 'original')]
        for case in ('weighted_k10', 'strict_first'):
            new = target.loc[(farm, case)]
            fields = ('absolute_imbalance_mwh', 'imbalance_cashflow_eur', 'auction_revenue_eur',
                      'degradation_eur', 'import_cost_eur', 'trading_fees_eur',
                      'realized_operating_profit_eur', 'final_energy_mwh', 'soc_final_percent')
            diff = {field: float(new[field] - old[field]) for field in fields}
            expected_profit = (diff['auction_revenue_eur'] + diff['imbalance_cashflow_eur']
                - diff['degradation_eur'] - diff['import_cost_eur'] - diff['trading_fees_eur'])
            if not np.isclose(diff['realized_operating_profit_eur'], expected_profit, atol=1e-5):
                raise ValueError(f'Profit attribution failed for {farm} {case}')
            contrasts.append({'farm': farm, 'case': case, **diff})
    pd.DataFrame(contrasts).to_csv(root / 'first_priority_paired_differences.csv', index=False)
    lines = ['# First-interval MPC priority: one-week comparison', '',
        '15–21 January 2026; four farms; continuous closed-loop cases. All use planned IDA3 and disable recovery.', '',
        '| Farm | Tracker imbalance MWh | Original MPC MWh | Weight 10 MWh | Strict first MWh | Original profit EUR | Weight 10 profit EUR | Strict first profit EUR |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for farm in ('1','345','901','910'):
        baseline = totals[(totals.farm.eq(farm)) & (totals.case.eq('original')) & totals.controller.eq('track_committed_grid')].iloc[0]
        a, b, c = [target.loc[(farm, case)] for case in ('original','weighted_k10','strict_first')]
        lines.append(f'| {farm} | {baseline.absolute_imbalance_mwh:.4f} | {a.absolute_imbalance_mwh:.4f} | '
            f'{b.absolute_imbalance_mwh:.4f} | {c.absolute_imbalance_mwh:.4f} | {a.realized_operating_profit_eur:.2f} | '
            f'{b.realized_operating_profit_eur:.2f} | {c.realized_operating_profit_eur:.2f} |')
    lines += ['', 'Cash profit excludes planning penalties and terminal values; final inventory must be compared. '
        'Later auction orders and battery states can differ. This is a retrospective seven-day development '
        'sample with old PV models, disclosed current interval-average PV and actual auction-price curves.', '',
        'The per-case daily and whole-run CSVs and `first_priority_objective_audit.json` retain the detailed '
        'results. The ordinary independent physical/accounting audit must also pass.', '']
    (root / 'first_priority_summary.md').write_text('\n'.join(lines))
    print(totals[totals.controller.eq('deterministic_mpc')][['farm','case','absolute_imbalance_mwh',
        'realized_operating_profit_eur','degradation_eur','soc_final_percent','mpc_fallbacks','mpc_timeouts']].to_string(index=False))
    print(json.dumps({'jobs': diagnostics, 'violations': violations}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    report(parser.parse_args().root)
