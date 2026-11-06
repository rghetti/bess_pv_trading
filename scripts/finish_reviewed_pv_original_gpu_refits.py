"""Finish the reviewed PV search's frozen refits on their original GPUs."""
from __future__ import annotations

import argparse
import json

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.training.pv_reviewed_original_device_refits import check_only, run_refits


def main():
    """Require explicit preflight or refit execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.check_only:
        plan, *_ = check_only(config)
        print(json.dumps(plan, indent=2, default=str))
        return
    run_refits(config)


if __name__ == '__main__':
    main()
