#!/usr/bin/env python
"""Prepare, run and report member-wise probabilistic PV studies."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    from bess_pv_trading.config import load_config
    from bess_pv_trading.forecasting.data.pv_probabilistic_data import prepare_farm, pvlib_farm, study_root, validate_study, delivery_split
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--stage', choices=['prepare', 'pvlib', 'search', 'report', 'all'], default='all')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--farms', nargs='+', type=int)
    parser.add_argument('--workers', type=int)
    parser.add_argument('--force', action='store_true', help='Recreate prepared input and pvlib artifacts')
    args = parser.parse_args()
    overrides = {}
    # When `args.farms is not None`, set `overrides['farm_ids']` using `args.farms`.
    if args.farms is not None: overrides['farm_ids'] = args.farms
    # When `args.workers is not None`, set `overrides['workers']` using `args.workers`.
    if args.workers is not None: overrides['workers'] = args.workers
    config = load_config(args.config, overrides={'pv_probabilistic_study': overrides})
    section = config['pv_probabilistic_study']
    # Perform main when `args.plan_only`.
    if args.plan_only:
        from bess_pv_trading.forecasting.training.pv_probabilistic_search import trial_plan
        trials = trial_plan(section)
        days = validate_study(section)
        splits = {name: [d.strftime('%Y-%m-%d') for d in days if delivery_split(d, section) == name]
                  for name in ('train','validation')}
        print(json.dumps({'study': section, 'delivery_days_by_split': splits, 'test': None,
            'purpose': 'Compare same-cycle deterministic and member-wise PV predictions and quantile calibration',
            'families': sorted({x['family'] for x in trials}), 'total_configurations': len(trials),
            'output': str(study_root(config)), 'trials': trials}, indent=2))
        return
    root = study_root(config)
    root.mkdir(parents=True, exist_ok=True)
    # Perform main when `args.stage in {'prepare', 'all'}`.
    if args.stage in {'prepare', 'all'}:
        # Perform main for each `farm` in `section['farm_ids']`.
        for farm in section['farm_ids']: prepare_farm(config, int(farm), force=args.force)
    # Perform main when `args.stage in {'pvlib', 'all'}`.
    if args.stage in {'pvlib', 'all'}:
        from bess_pv_trading.forecasting.plotting.pv_probabilistic_reporting import evaluate_pvlib, write_pvlib_overview
        # Perform main for each `farm` in `section['farm_ids']`.
        for farm in section['farm_ids']:
            pvlib_farm(config, int(farm), force=args.force)
            evaluate_pvlib(config, int(farm))
        write_pvlib_overview(config)
        from bess_pv_trading.forecasting.evaluation.pv_probabilistic_diagnostics import summarize_diagnostics
        summarize_diagnostics(config)
    # Perform main when `args.stage in {'search', 'all'}`.
    if args.stage in {'search', 'all'}:
        from bess_pv_trading.forecasting.training.pv_probabilistic_search import run_search
        run_search(config)
    # Perform main when `args.stage in {'report', 'all'}`.
    if args.stage in {'report', 'all'}:
        from bess_pv_trading.forecasting.plotting.pv_probabilistic_reporting import report_search
        report_search(config)
        from bess_pv_trading.forecasting.evaluation.pv_probabilistic_diagnostics import summarize_diagnostics
        summarize_diagnostics(config, include_winners=True)


# Process this part of the workflow when `__name__ == '__main__'`.
if __name__ == '__main__':
    main()
