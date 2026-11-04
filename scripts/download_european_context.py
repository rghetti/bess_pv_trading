"""Acquire public RTE and Swissgrid market context into the configured DuckDB."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.config import load_config
from bess_pv_trading.data.country_balances import download_french
from bess_pv_trading.data.swiss_border import download_swiss_border


def main(argv=None):
    """Run explicitly selected providers and save a readable acquisition report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/european_context.yaml")
    parser.add_argument("--provider", choices=("rte", "swissgrid", "both"), default="both")
    parser.add_argument("--max-rte-months", type=int)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.max_rte_months is not None and args.max_rte_months < 0:
        parser.error("--max-rte-months must be nonnegative")
    config = load_config(args.config)
    results = {}
    if args.provider in {"rte", "both"}:
        results["rte"] = download_french(config, refresh=args.refresh, max_requests=args.max_rte_months,
            progress=lambda message: print(message, file=sys.stderr, flush=True))
    if args.provider in {"swissgrid", "both"}:
        results["swissgrid"] = download_swiss_border(config, refresh=args.refresh)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(results, indent=2, default=str, allow_nan=False) + "\n")
    print(json.dumps({key: value["status"] for key, value in results.items()}))
    return 0 if all(value["status"] in {"complete", "cached"} for value in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
