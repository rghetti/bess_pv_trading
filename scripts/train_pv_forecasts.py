#!/usr/bin/env python
"""Inspect or run the reproducible PV forecast search from the repository root.

Examples:
  python scripts/train_pv_forecasts.py --config configs/pv_experiments/forecast_search_901_910.yaml --plan-only
  python scripts/train_pv_forecasts.py --config configs/pv_experiments/forecast_search_901_910.yaml --workers 42
  python scripts/train_pv_forecasts.py --config configs/pv_experiments/forecast_search_901_910.yaml --farms 901 --run-id farm901_trial --no-resume

Hyperparameter grids, seeds, budgets, inputs and output roots live in YAML.
The launcher changes no installed packages and performs no network requests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

# Support an uninstalled checkout without requiring PYTHONPATH shell overrides.
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))


def main():
    from bess_pv_trading.config import load_config
    from bess_pv_trading.forecasting.training.pv_forecast_search import generate_trials,run_forecast_search
    parser=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config',required=True,type=Path,help='YAML importing configs/standard.yaml')
    parser.add_argument('--plan-only',action='store_true',help='Print models and parameters without fitting')
    parser.add_argument('--workers',type=int,help='Single-threaded CPU processes; at most 42 with six reserved')
    parser.add_argument('--farms',type=int,nargs='+',help='Override the farm list; use a separate run ID')
    parser.add_argument('--run-id',help='Override the output directory name below the configured root')
    parser.add_argument('--resume',action=argparse.BooleanOptionalAction,default=None,help='Resume matching completed fits, or replace the run when disabled')
    args=parser.parse_args()
    options={name:value for name,value in {'workers':args.workers,'farm_ids':args.farms,
        'run_id':args.run_id,'resume':args.resume}.items() if value is not None}
    config=load_config(args.config,overrides={'pv_forecast_search':options})
    if args.plan_only:
        section=config['pv_forecast_search']
        print(json.dumps({'workers':section['workers'],'reserved_cpu_cores':section['reserved_cpu_cores'],
            'input_contract':'weather and cyclic time; no proxy geometry, auction identity, or aging',
            'output_root':section['output_root'],'run_id':section['run_id'],
            'training':'2025 frozen corpus train rows', 'validation':'Jan-Apr 2026 frozen validation rows',
            'reporting':'May-Aug 2026 reused holdout',
            'trials':{str(farm):generate_trials(section,farm) for farm in section['farm_ids']}},indent=2))
    else:
        print(json.dumps(run_forecast_search(config,progress=lambda line:print(line,flush=True)),indent=2))


if __name__=='__main__':main()
