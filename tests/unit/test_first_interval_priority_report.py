"""Audit the paired first-priority report against synthetic source artifacts."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def test_priority_report_reconstructs_weighted_and_strict_objectives(tmp_path):
    """A corrupt saved objective is rejected instead of reported as valid."""
    script = Path(__file__).resolve().parents[2] / 'scripts/report_first_interval_priority.py'
    spec = importlib.util.spec_from_file_location('priority_report', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields = ('absolute_imbalance_mwh', 'imbalance_cashflow_eur', 'auction_revenue_eur',
        'degradation_eur', 'import_cost_eur', 'trading_fees_eur', 'realized_operating_profit_eur',
        'pv_curtailment_mwh', 'throughput_mwh', 'final_energy_mwh', 'soc_final_percent',
        'mpc_solves', 'mpc_fallbacks', 'mpc_timeouts', 'physical_safety_overrides', 'solver_runtime_total_seconds')
    jobs = []
    bad_path = None
    for farm in (1, 345, 901, 910):
        for case in ('original', 'weighted_k10', 'strict_first'):
            job_id = f'{farm}_{case}'
            output = tmp_path / job_id
            output.mkdir()
            config = {'realtime_mpc': {'planning_imbalance_penalty_eur_per_mwh': 500,
                'first_interval_weight': 10 if case == 'weighted_k10' else 1},
                'solver': {'feasibility_tolerance': 1e-8}}
            (tmp_path / f'config__{job_id}.json').write_text(json.dumps(config))
            jobs.append({'job_id': job_id, 'status': 'complete', 'farm_id': farm,
                'case_id': case, 'result': {'output_dir': str(output)}})
            comparison = []
            for controller in ('track_committed_grid', 'deterministic_mpc'):
                values = {field: 0.0 for field in fields}
                values['soc_final_percent'] = 50.0
                values['controller'] = controller
                comparison.append(values)
                folder = output / 'controllers' / controller
                (folder / 'csv').mkdir(parents=True)
                days = [f'2026-01-{day:02}' for day in range(15,22) for _ in range(96)]
                frame = pd.DataFrame({'interval_id': range(672), 'delivery_day': days,
                    'net_position_mwh': 0.0, 'grid_mw': 0.0, 'energy_end_mwh': 0.5,
                    'imbalance_mwh': 0.0, 'net_profit_eur': 0.0, 'imbalance_cashflow_eur': 0.0})
                frame.to_csv(folder / 'csv/controller_intervals.csv', index=False)
                if controller == 'deterministic_mpc':
                    (folder / 'json').mkdir()
                    components = {'absolute_imbalance_mwh': 0.1, 'degradation_eur': 1.0,
                        'import_cost_eur': 0.0, 'terminal_cost_eur': 0.0}
                    objective = 276.0 if case == 'weighted_k10' else 51.0
                    solve = {'status': 'optimal', 'fallback_applied': False, 'components': components,
                        'first_interval_expected_imbalance_mwh': 0.05, 'planning_objective_value': objective,
                        'first_interval_minimum_certified': case == 'strict_first',
                        'first_interval_refinement_used': case == 'strict_first', 'first_interval_cap_mwh': 0.05}
                    solve_path = folder / 'json/mpc_solves.json'
                    solve_path.write_text(json.dumps([dict(solve, interval_id=str(i)) for i in range(672)]))
                    if farm == 1 and case == 'weighted_k10':
                        bad_path = solve_path
            pd.DataFrame(comparison).to_csv(output / 'controller_comparison.csv', index=False)
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'complete', 'results': jobs}))
    module.report(tmp_path)
    assert (tmp_path / 'first_priority_summary.md').is_file()
    corrupt = json.loads(bad_path.read_text())
    corrupt[0]['planning_objective_value'] += 1
    bad_path.write_text(json.dumps(corrupt))
    with pytest.raises(ValueError, match='objective_decomposition'):
        module.report(tmp_path)
