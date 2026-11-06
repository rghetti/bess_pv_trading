"""Download/import Swissgrid physical-border history without requiring any farm files."""

from __future__ import annotations

import argparse
import json

from bess_pv_trading.config import load_config
from bess_pv_trading.data.swissgrid_energy import acquire_energy_statistics


def main() -> None:
    """Apply explicit runtime paths and execute the configured dataset acquisition."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/api_experiments/swissgrid_energy_history.yaml")
    parser.add_argument("--database")
    parser.add_argument("--raw-directory")
    parser.add_argument("--report")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["swissgrid_energy"]
    for key, value in {"database_path": args.database, "raw_directory": args.raw_directory,
                       "report_path": args.report}.items():
        if value is not None:
            settings[key] = value
    report = acquire_energy_statistics(config, offline=args.offline, refresh=args.refresh,
                                       prepare_only=args.prepare_only)
    print(json.dumps({"status": report["status"], "database_path": report["database_path"],
                      "years": [{"year": item["year"], "status": item["status"],
                                 "rows": item.get("metadata", item.get("manifest", {}).get("metadata", {})).get("source_rows")}
                                for item in report["sources"]]}, indent=2))


if __name__ == "__main__":
    main()
