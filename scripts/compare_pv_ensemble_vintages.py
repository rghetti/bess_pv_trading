#!/usr/bin/env python
"""Compare completed preceding-day and retrospective same-day PV studies."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.evaluation.pv_probabilistic_compare import compare_experiments

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preceding-config',required=True,type=Path)
    parser.add_argument('--same-day-config',required=True,type=Path)
    args=parser.parse_args()
    compare_experiments(load_config(args.preceding_config),load_config(args.same_day_config))
