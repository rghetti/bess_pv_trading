"""Plan, preflight or run the January IDA3 extended-horizon grid tracker study."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import json
import multiprocessing
from pathlib import Path
import re
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np
import pandas as pd

from bess_pv_trading.config import _read_mapping, merge_mapping, resolve_config
from bess_pv_trading.parallel import configure_worker_threads
from bess_pv_trading.reporting.ida3_horizon_comparison import write_comparison
from bess_pv_trading.simulation.energy_research_study import _artifact_requirements
from bess_pv_trading.simulation.continuous_inputs import prepare_continuous_inputs
from bess_pv_trading.simulation.synthetic import write_json

CONFIG = ROOT / "configs/research/ida3_next_day_january_2026.yaml"
FARMS = ("1", "345", "901", "910")


def historical_paths():
    """Identify the completed trajectory and provenance for each baseline farm."""
    result = {}
    for farm in FARMS:
        batch = "deterministic_daily_hard_v2" if farm in {"1", "345"} else "farms_901_910_equal_v1"
        base = ROOT / "outputs/energy_research" / batch / "runs" / f"farm_{farm}" / (
            f"january_to_august__farm_{farm}__deterministic_original") / "continuous"
        result[farm] = base
    return result


def resolved_jobs(farms, batch, solver_threads):
    """Resolve each farm from a fresh common study configuration."""
    raw = _read_mapping(CONFIG)
    jobs = []
    for farm in farms:
        forced = {"farm_id": int(farm), "_project_root": str(ROOT),
                  "solver": {"threads": solver_threads},
                  "continuous_optimization": {"output_root": str(batch / "runs"), "run_id": f"farm_{farm}"}}
        config = resolve_config(merge_mapping(raw, forced))
        if config["execution"]["dispatch_policy"] != "track_committed_grid":
            raise ValueError("This study runs only the grid tracker")
        jobs.append({"farm": farm, "config": config})
    return jobs


def compare_prepared_to_history(prepared, baseline):
    """Reject different January PV, prices or reviewed model before optimizing."""
    old = json.loads((baseline / "input_provenance__track_committed_grid.json").read_text())
    new = prepared["provenance"]
    for day in pd.date_range("2026-01-01", "2026-01-31").strftime("%Y-%m-%d"):
        for key in ("realized_pv_sha256", "auction_price_sha256", "skipped_auctions"):
            if old["days"][day][key] != new["days"][day][key]:
                raise ValueError(f"January historical {key} differs for {day}")
    old_model = old.get("forecast_model_artifacts", {})
    new_model = new.get("forecast_model_artifacts", {})
    if not old_model or any(old_model.get(key) != new_model.get(key) for key in ("model", "checksums")):
        raise ValueError("Reviewed forecast model differs from the historical tracker")
    # Compare actual settlement inputs by product, even though batch-wide
    # fingerprints differ due to the window and planning configuration.
    old_dispatch = pd.read_csv(baseline / "controllers/track_committed_grid/csv/realized_dispatch.csv",
                               usecols=["interval_id", "delivery_day", "imbalance_price_eur_mwh"])
    old_dispatch = old_dispatch.loc[old_dispatch.delivery_day.astype(str).between("2026-01-01", "2026-01-31")]
    prices = dict(zip(prepared["full_grid"].interval_ids, prepared["imbalance_prices_eur_per_mwh"]))
    values = old_dispatch.interval_id.map(prices)
    if len(old_dispatch) != 31 * 96 or values.isna().any() or not np.allclose(
            old_dispatch.imbalance_price_eur_mwh, values, rtol=0, atol=1e-9):
        raise ValueError("January historical settlement prices differ")
    return {"january_days": 31, "settlement_products": len(old_dispatch), "model": old_model["model"]}


def run_farm(job, batch, baseline, check_only):
    """Preflight one farm, then optionally run its single continuous tracker."""
    farm, config = job["farm"], job["config"]
    base = Path(batch) / "runs" / f"farm_{farm}"
    log = base / "tracker.log"
    result = {"farm": farm, "status": "failed", "log": str(log)}
    with log.open("w", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            progress = lambda message: print(f"[farm {farm}] {message}", flush=True)
            progress("Preparing forecasts, production, auction prices and next-day context")
            prepared = prepare_continuous_inputs(config, progress=progress)
            if prepared.get("imbalance_prices_eur_per_mwh") is None:
                raise ValueError("Actual imbalance prices are required")
            result["input_fingerprint"] = prepared["input_fingerprint"]
            result["common_inputs"] = compare_prepared_to_history(prepared, baseline)
            write_json(base / "input_provenance.json", prepared["provenance"])
            if check_only:
                result["status"] = "preflight_passed"
                return result
            from bess_pv_trading.simulation.continuous import ContinuousReplay
            destination = base / "controllers/track_committed_grid"
            replay = ContinuousReplay(config, prepared, "integrated", destination, progress=progress)
            progress("Starting continuous grid-tracking trajectory")
            summary = replay.run()
            write_json(base / "summary.json", summary)
            result["status"] = "complete"
            progress("Grid-tracking trajectory completed")
        except Exception as error:
            result.update(error=str(error), traceback=traceback.format_exc())
            print(result["traceback"], file=stream, flush=True)
    return result


def main(argv=None):
    """Show a concrete plan without solving unless an action is explicitly selected."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--farm-ids", nargs="+", choices=FARMS, default=list(FARMS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--solver-threads", type=int, default=5)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id must contain letters, digits, underscores, hyphens or periods")
    if len(set(args.farm_ids)) != len(args.farm_ids):
        parser.error("farm IDs must be unique")
    workers = min(args.workers, len(args.farm_ids))
    if workers < 1 or args.solver_threads < 1 or workers * args.solver_threads > 48:
        parser.error("worker and solver thread counts must be positive and fit the 48-thread budget")
    batch = ROOT / "outputs/energy_research" / args.run_id
    jobs = resolved_jobs(args.farm_ids, batch, args.solver_threads)
    baselines = historical_paths()
    missing = [str(baselines[farm]) for farm in args.farm_ids
               if not (baselines[farm] / "input_provenance__track_committed_grid.json").is_file()]
    requirements = {job["farm"]: _artifact_requirements(job["config"]) for job in jobs}
    missing_inputs = {farm: [item for item in items if not item["present"]]
                      for farm, items in requirements.items()}
    plan = {"status": "plan_only", "output_dir": str(batch), "farms": args.farm_ids,
        "delivery_period": ["2026-01-01", "2026-01-31"], "planning_context_through": "2026-02-01",
        "initial_auction": "2025-12-31 DA gate for 2026-01-01; no 2025-12-31 IDA3",
        "controller": "track_committed_grid", "trajectory_count": len(jobs), "workers": workers,
        "solver_threads_per_farm": args.solver_threads, "maximum_solver_threads": workers * args.solver_threads,
        "auction_horizon": "IDA3 remainder of delivery day plus next delivery day", "midnight_soc_floor_percent": 40,
        "forecast": "latest eligible deterministic 48-hour weather cycles and selected retrospective neural models",
        "forecast_availability_lag_hours": jobs[0]["config"]["source"]["availability_lag_hours"],
        "pv_models": {job["farm"]: job["config"]["pv"]["forecast_replacement"]["models_by_farm"][job["farm"]]["name"]
                      for job in jobs},
        "model_training": "frozen prior models; no training, tuning or calibration in this study",
        "actual_pv_preprocessing": "reviewed AC, monthly night zeros and explicitly labeled synthetic or historical-GHI replacements",
        "actual_price_information": "retrospective auction-price oracle including future next-day prices",
        "pv_information": "curated actual PV, including synthetic replacements; realized interval-average known to tracker",
        "comparison": "January historical same-day auction plans with a hard 50% target; horizon and SOC policy both differ",
        "historical_baselines": {farm: str(baselines[farm]) for farm in args.farm_ids},
        "missing_baselines": missing, "missing_inputs": missing_inputs, "solves_started": 0, "models_fitted": 0}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.check_only and not args.run:
        return 0
    if missing:
        parser.error("Historical baselines are missing; see plan")
    if any(missing_inputs.values()):
        parser.error("Study inputs are missing; see plan")
    batch.mkdir(parents=True, exist_ok=False)
    for job in jobs:
        base = batch / "runs" / f"farm_{job['farm']}"
        base.mkdir(parents=True)
        write_json(base / "resolved_config.json", job["config"])
    manifest = {"status": "running", "plan": plan,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "results": []}
    write_json(batch / "manifest.json", manifest)
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                                 initializer=configure_worker_threads, initargs=(1, True)) as pool:
            futures = {pool.submit(run_farm, job, batch, baselines[job["farm"]], args.check_only): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as error:
                    result = {"farm": job["farm"], "status": "failed", "error": str(error)}
                manifest["results"].append(result)
                print(f"Farm {result['farm']}: {result['status']}", flush=True)
                write_json(batch / "manifest.json", manifest)
        failures = [result for result in manifest["results"] if result["status"] not in {"complete", "preflight_passed"}]
        if failures:
            manifest.update(status="failed", failures=failures)
        elif args.check_only:
            manifest["status"] = "preflight_passed"
        else:
            sources = {farm: str(baselines[farm] / "controllers/track_committed_grid") for farm in args.farm_ids}
            path = write_comparison(batch, args.farm_ids, sources)
            manifest.update(status="complete", summary_markdown=str(path))
        return 1 if failures else 0
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        write_json(batch / "manifest.json", manifest)


if __name__ == "__main__":
    raise SystemExit(main())
