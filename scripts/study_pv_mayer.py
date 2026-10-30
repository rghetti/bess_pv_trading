#!/usr/bin/env python
"""Check coverage or run an approved Mayer-style hourly PV forecast study."""
import argparse
import json
from pathlib import Path
import sys
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    """Load documented settings and keep the coverage check side-effect free."""
    from bess_pv_trading.config import load_config
    from bess_pv_trading.forecasting.studies.pv_mayer_data import check_coverage
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    # Apply check coverage when `args.check_only`.
    if args.check_only:
        try:
            routes = check_coverage(config)
        except (ValueError, FileNotFoundError) as error:
            parser.exit(2, f'Coverage check failed: {error}\n')
        section = config['pv_mayer_postprocessing']
        missing = [str(day.date()) for day in pd.date_range(section['start_day'], section['end_day']) if day not in routes]
        print(json.dumps({'status': 'declared_windows_complete', 'days': len(routes), 'uncovered_days': len(missing),
                          'study': config['pv_mayer_postprocessing']}, indent=2))
    else:
        from bess_pv_trading.forecasting.studies.pv_mayer import run
        run(config)


# Process this part of the workflow when `__name__ == '__main__'`.
if __name__ == '__main__':
    main()
