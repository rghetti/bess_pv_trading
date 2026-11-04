"""Plan, preflight or run continuous MPC versus grid-tracking research cases."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.simulation.energy_research_study import research_plan, run_research_study


def main(argv=None):
    """Default to a plan; explicitly select --run to execute historical comparisons."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/energy_market_extensions.yaml")
    parser.add_argument("--case", action="append", dest="case_ids", help="Repeat to select explicit case IDs")
    parser.add_argument("--farm", action="append", dest="farms", help="Repeat to select configured farms")
    parser.add_argument("--window", action="append", dest="window_ids", help="Repeat to select configured date windows")
    parser.add_argument("--run-id", help="Use a new batch identifier without modifying the study YAML")
    parser.add_argument("--workers", type=int, help="Use up to 42 spawned workers for this batch")
    parser.add_argument("--solver-threads", type=int, help="Set Gurobi threads per worker within the 48-core budget")
    parser.add_argument("--license-file", type=Path, help="Use this existing Gurobi license file in spawned workers")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true", help="Read and validate selected inputs without fitting or solving")
    actions.add_argument("--run", action="store_true", help="Execute selected continuous comparisons")
    args = parser.parse_args(argv)
    if args.license_file is not None:
        if not args.license_file.is_file():
            parser.error("--license-file must identify an existing file")
        os.environ["GRB_LICENSE_FILE"] = str(args.license_file.resolve())
    selection = {"case_ids": args.case_ids, "farms": args.farms, "window_ids": args.window_ids,
        "project_root": str(ROOT), "run_id": args.run_id, "workers": args.workers,
        "solver_threads": args.solver_threads}
    plan = research_plan(args.config, **selection)
    # Keep explicit planning detailed, but avoid repeating thousands of resolved
    # input records during execution; the manifest retains the complete plan.
    displayed_plan = plan if not (args.run or args.check_only) else {
        key: value for key, value in plan.items() if key != "jobs"}
    print(json.dumps(displayed_plan, indent=2, allow_nan=False), flush=True)
    if not args.check_only and not args.run:
        return 0
    result = run_research_study(args.config, check_only=args.check_only, **selection,
        progress=lambda message: print(message, file=sys.stderr, flush=True))
    displayed_result = {key: value for key, value in result.items()
        if key not in {"plan", "results"}}
    displayed_result["job_statuses"] = {item["job_id"]: item["status"]
        for item in result.get("results", [])}
    displayed_result["failures"] = [item for item in result.get("results", [])
        if item["status"] not in {"complete", "preflight_passed"}]
    print(json.dumps(displayed_result, indent=2, allow_nan=False), flush=True)
    return 0 if result["status"] in {"complete", "preflight_passed"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
