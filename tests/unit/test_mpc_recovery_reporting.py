"""Verify paired recovery reports from complete synthetic artifacts."""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest


def test_recovery_report_pairs_realized_outcomes(tmp_path):
    """Keep recovery and profit differences paired by farm and controller."""
    script = Path(__file__).resolve().parents[2] / 'scripts/report_mpc_recovery.py'
    spec = importlib.util.spec_from_file_location('recovery_report', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    results = []
    fields = ('absolute_imbalance_mwh', 'imbalance_cashflow_eur', 'auction_revenue_eur',
              'degradation_eur', 'import_cost_eur', 'trading_fees_eur', 'realized_operating_profit_eur',
              'pv_curtailment_mwh', 'throughput_mwh', 'final_energy_mwh', 'soc_final_percent',
              'mpc_solves', 'mpc_fallbacks', 'mpc_timeouts', 'physical_safety_overrides')
    for farm in (1, 345, 901, 910):
        for case in ('mpc_original', 'mpc_recovery'):
            path = tmp_path / f'{farm}_{case}'
            path.mkdir()
            results.append({'status': 'complete', 'job_id': path.name, 'farm_id': farm,
                            'case_id': case, 'result': {'output_dir': str(path)}})
            comparison = []
            for controller in ('track_committed_grid', 'deterministic_mpc'):
                values = {field: 0.0 for field in fields}
                values['soc_final_percent'] = 50.0
                values['controller'] = controller
                comparison.append(values)
                destination = path / 'controllers' / controller / 'csv'
                destination.mkdir(parents=True)
                recovery = 0.2 if case == 'mpc_recovery' and controller == 'deterministic_mpc' else 0.0
                dates = [f'2026-01-{day:02}' for day in range(15, 22) for _ in range(96)]
                frame = pd.DataFrame({'delivery_day': dates, 'curtailment_recovered_mw': recovery,
                    'duration_hours': 0.25, 'curtailment_mw': 0.0, 'imbalance_mwh': 0.0,
                    'net_profit_eur': 0.0, 'energy_end_mwh': 0.5})
                frame.to_csv(destination / 'controller_intervals.csv', index=False)
            pd.DataFrame(comparison).to_csv(path / 'controller_comparison.csv', index=False)
    (tmp_path / 'manifest.json').write_text(json.dumps({'status': 'complete', 'results': results}))
    contrast = module.summarize(tmp_path)
    assert len(contrast) == 4
    assert contrast.recovered_pv_mwh.to_numpy() == pytest.approx([33.6] * 4)
    assert (tmp_path / 'recovery_summary.md').exists()
