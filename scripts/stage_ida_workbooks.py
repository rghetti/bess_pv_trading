"""Stage manually downloaded GME IDA workbooks for a checked DuckDB merge."""

from __future__ import annotations

import argparse
import json

from bess_pv_trading.config import load_config
from bess_pv_trading.data.manual_market_import import stage_market_workbooks


def main():
    """Resolve runtime paths and prepare the existing overlap-safe import workflow."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/manual_ida_updates.yaml")
    parser.add_argument("--database", help="Existing target market DuckDB")
    parser.add_argument("--source-directory", help="Directory containing GME XLSX files")
    parser.add_argument("--start", help="First local delivery date for the coverage report")
    parser.add_argument("--end-exclusive", help="End of the coverage report")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["manual_market_import"]
    for name, value in {"target_database": args.database, "source_directory": args.source_directory,
                        "date_start": args.start, "date_end_exclusive": args.end_exclusive}.items():
        if value is not None:
            settings[name] = value
    report = stage_market_workbooks(config, progress=print)
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
