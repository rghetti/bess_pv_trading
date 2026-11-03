"""Plan, preflight or explicitly run paired all-auction MPC/feedback comparisons."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import date
import json
from pathlib import Path
import sys

# Allow the repository's usual direct-script invocation without installing it.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.config import load_config
from bess_pv_trading.markets import AuctionCalendar
from bess_pv_trading.optimization.mpc import MPC_POLICIES
from bess_pv_trading.simulation.continuous_inputs import prepare_continuous_inputs
from bess_pv_trading.simulation.realtime_mpc_study import (
    run_realtime_mpc_experiment, validate_mpc_study,
)


def comparison_plan(config, modes):
    """Count the exact calendar work without reading data or creating solver models."""
    validate_mpc_study(config)
    auctions = ["DA", "IDA1", "IDA2", "IDA3"]
    if config["gme_oracle_experiment"].get("participating_auctions") != auctions:
        raise ValueError("This comparison script requires explicit DA/IDA1/IDA2/IDA3 participation")
    settings = config["continuous_optimization"]
    first, last = (date.fromisoformat(settings[key]) for key in ("start_date", "end_date"))
    import pandas as pd

    days = [day.date().isoformat() for day in pd.date_range(first, last, freq="D")]
    calendar = AuctionCalendar.from_config(config)
    intervals = sum(len(calendar.delivery_grid(day).to_domain().interval_ids) for day in days)
    auction_events = sum(event.available for day in days for event in calendar.schedule(day))
    controllers = config["realtime_mpc_comparison"]["controllers"]
    mpc_count = sum(controller in MPC_POLICIES for controller in controllers)
    auction_solves = sum(auction_events * (1 if mode == "frozen_commitments" else len(controllers)) for mode in modes)
    mpc_solves = len(modes) * intervals * mpc_count
    return {"status": "plan_only", "delivery_days": len(days), "start_date": days[0], "end_date": days[-1],
        "timezone": calendar.timezone, "controllers": controllers, "modes": list(modes),
        "participating_auctions": auctions, "physical_intervals_per_trajectory": intervals,
        "mpc_decisions": mpc_solves, "auction_solves_if_sources_are_generated": auction_solves,
        "mpc_time_limit_seconds": config.get("realtime_mpc", {}).get("solver_time_limit_seconds", 10),
        "auction_time_limit_seconds": config["solver"]["time_limit_seconds"],
        "threads": config["solver"]["threads"], "gpus": 0,
        "current_pv_information": config.get("realtime_mpc", {}).get("current_pv_information", "forecast"),
        "forecast_model": config["pv"]["forecast_replacement"]["models_by_farm"][str(config["farm_id"])]["name"],
        "retrospective_model_artifacts": config["pv"]["forecast_replacement"].get("allow_retrospective_artifacts", False),
        "run_id": settings["run_id"], "data_read": False, "solves_started": 0,
        "detailed_daily_plot_day_limit": min(3, config["realtime_mpc_comparison"].get("representative_days", 3))}


def main(argv=None):
    """Default to a reviewable plan; --run is required to start historical replays."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
        default=ROOT / "configs/full_experiments/realtime_farm1_january_2026_all_auctions.yaml")
    parser.add_argument("--mode", choices=("both", "frozen_commitments", "continuous"), default="both")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true", help="Pin cached inputs once; start no optimization")
    actions.add_argument("--run", action="store_true", help="Execute the explicitly approved historical study")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    config["_project_root"] = str(ROOT)
    modes = ("frozen_commitments", "continuous") if args.mode == "both" else (args.mode,)
    plan = comparison_plan(config, modes)
    print(json.dumps(plan, indent=2))
    if not args.run and not args.check_only:
        return 0
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    # Both comparisons use one immutable set of actuals, prices and forecast snapshots.
    prepared = prepare_continuous_inputs(config, progress=progress)
    results = []
    for mode in modes:
        selected = deepcopy(config)
        selected["realtime_mpc_comparison"].update(mode=mode, check_only=bool(args.check_only))
        results.append(run_realtime_mpc_experiment(selected, progress=progress, prepared_inputs=prepared))
    print(json.dumps({"status": "preflight_passed" if args.check_only else "complete", "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
