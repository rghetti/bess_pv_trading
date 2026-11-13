"""Download archived city weather through the shared features checkout."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


CHILD = r'''
import json
import sys
from bess_pv_trading.config import load_config
from bess_pv_trading.data.feature_sources import acquire_regional_weather

settings = json.load(sys.stdin)
config = load_config(settings["config"])
feature = config["price_features"]
feature["database_path"] = settings["database"]
feature["start"] = settings["start"]
feature["end_exclusive"] = settings["end_exclusive"]
feature["output_root"] = settings["output_root"]
feature["regional_weather"]["run_hours_utc"] = settings["cycles"]
feature["regional_weather"]["publication_latency_hours"] = settings["lag_hours"]
result = acquire_regional_weather(config, dry_run=not settings["run"],
    max_requests=settings["max_requests"], progress=lambda message: print(message, file=sys.stderr, flush=True))
print(json.dumps(result, indent=2, default=str), flush=True)
if settings["run"] and result["status"] not in ("complete", "complete_with_gaps"):
    sys.exit(2)
'''


def main():
    """Run bounded acquisition with a separately imported features package."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features-repo", required=True)
    parser.add_argument("--database", required=True)
    parser.add_argument("--start", required=True, help="First initialization day, YYYY-MM-DD")
    parser.add_argument("--end-exclusive", required=True, help="Exclusive initialization day, YYYY-MM-DD")
    parser.add_argument("--cycles", type=int, nargs="+", default=[12])
    parser.add_argument("--lag-hours", type=float, default=6)
    parser.add_argument("--max-requests", type=int, default=100)
    parser.add_argument("--output", default="outputs/rolling_price_weather")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if any(hour not in (0, 6, 12, 18) for hour in args.cycles) or args.lag_hours < 0:
        parser.error("Cycles must be among 0, 6, 12, 18 and lag must be nonnegative")
    repo = Path(args.features_repo).resolve()
    config = repo / "configs/standard.yaml"
    if not config.is_file():
        raise FileNotFoundError(f"Features checkout configuration not found: {config}")
    # The legacy feature downloader includes the previous day and excludes the
    # last day; shift both bounds so this CLI uses an ordinary exclusive range.
    from datetime import date, timedelta
    first, stop = date.fromisoformat(args.start), date.fromisoformat(args.end_exclusive)
    if first >= stop:
        parser.error("--end-exclusive must be later than --start")
    lag = int(args.lag_hours) if args.lag_hours.is_integer() else args.lag_hours
    settings = {"config": str(config), "database": str(Path(args.database).resolve()),
        "start": str(first + timedelta(days=1)), "end_exclusive": str(stop + timedelta(days=1)),
        "output_root": str(Path(args.output).resolve()), "cycles": args.cycles,
        "lag_hours": lag, "max_requests": args.max_requests, "run": args.run}
    environment = {**os.environ, "PYTHONPATH": str(repo)}
    result = subprocess.run([sys.executable, "-c", CHILD], input=json.dumps(settings),
                            text=True, cwd=repo, env=environment, check=False)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
