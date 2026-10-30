#!/usr/bin/env python
"""Rearrange and rescore saved PV quantiles without rerunning the study."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    """Keep the original fit outputs intact and create a separate report."""
    from bess_pv_trading.forecasting.studies.pv_mayer import rearrange_existing
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    for result in rearrange_existing(args.run_dir):
        print(result, flush=True)


if __name__ == '__main__':
    main()
