"""Apply user-reviewed PV target decisions without starting a training run."""
from __future__ import annotations

import argparse

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.data.pv_reviewed_targets import run_reviewed_preparation


def main():
    """Resolve the study configuration and prepare its versioned data artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    arguments = parser.parse_args()
    run_reviewed_preparation(load_config(arguments.config))


if __name__ == '__main__':
    main()
