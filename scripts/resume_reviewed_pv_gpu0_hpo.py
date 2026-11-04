"""Check or resume the interrupted reviewed multi-GPU PV search on GPU 0."""
from __future__ import annotations

import argparse
import json

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.training.pv_reviewed_gpu0_recovery import check_only, run_hpo


def main():
    """Require an explicit preflight or training mode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.check_only:
        report, _, _, _ = check_only(config)
        print(json.dumps(report, indent=2, default=str))
        return
    run_hpo(config)


if __name__ == '__main__':
    main()
