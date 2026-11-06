"""Rebuild price-search inputs from market.duckdb without farm files or saved matrices."""

from __future__ import annotations

import argparse
import json

from bess_pv_trading.config import load_config
from bess_pv_trading.data.price_search_inputs import build_price_search_inputs


def main() -> None:
    """Apply machine-specific storage paths without changing the scientific source selection."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/price_search_inputs.yaml")
    parser.add_argument("--database")
    parser.add_argument("--output")
    parser.add_argument("--pinned-manifest")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["price_search_inputs"]
    if args.database:
        settings["database_path"] = args.database
    if args.output:
        settings["output_directory"] = args.output
    report = build_price_search_inputs(config, pinned_manifest=args.pinned_manifest)
    print(json.dumps({"status": report["status"], "farm_files_required": False,
                      "sources": {name: details["rows"] for name, details in report["tables"].items()}}, indent=2))


if __name__ == "__main__":
    main()
