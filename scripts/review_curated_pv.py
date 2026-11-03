"""Prepare the requested PV corpus and a full-history episode review; never train."""
from __future__ import annotations

import argparse

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.data.pv_curated_review import run_review


def main():
    """Load one study configuration and materialize reviewable data artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--refresh-review', action='store_true',
                        help='Reuse fingerprint-checked prepared data and publish missing review plots.')
    arguments = parser.parse_args()
    run_review(load_config(arguments.config), reuse_prepared=arguments.refresh_review)


if __name__ == '__main__':
    main()
