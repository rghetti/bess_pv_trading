"""Plan, preflight, or execute the 2026 same-day SOC sweep."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
from datetime import date
import hashlib
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

from bess_pv_trading.parallel import configure_worker_threads
from bess_pv_trading.config import _read_mapping, merge_mapping, resolve_config
from bess_pv_trading.simulation.continuous_inputs import _fingerprint, prepare_continuous_inputs
from bess_pv_trading.simulation.energy_research_study import _artifact_requirements
from bess_pv_trading.simulation.synthetic import write_json
from k25_preflight_cache import clear_preflight_cache, preparation_key, prepare_shared
FARMS = ("1", "345", "901", "910")

CONFIG = ROOT / "configs/research/same_day_soc_sweep_2026.yaml"


def source_checksums():
    """Pin the exact source and launcher files used for a batch."""
    checksums = {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                 for path in sorted((ROOT / "src").rglob("*.py"))}
    for path in (Path(__file__), ROOT / "scripts/run_ida3_next_day_january.py",
                 ROOT / "scripts/k25_preflight_cache.py"):
        checksums[str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return checksums


def make_jobs(farms, targets, batch, solver_threads, *, raw=None):
    """Keep farm inputs identical while changing each planning SOC target."""
    raw = _read_mapping(CONFIG) if raw is None else raw
    base = {}
    for farm in farms:
        forced = {"farm_id": int(farm), "_project_root": str(ROOT), "solver": {"threads": solver_threads},
                  "continuous_optimization": {"output_root": str(batch / "runs"), "run_id": f"farm_{farm}"}}
        base[farm] = resolve_config(merge_mapping(raw, forced))
        battery = base[farm]["battery"]
        if any(not battery["soc_min_percent"] <= target <= battery["soc_max_percent"] for target in targets):
            raise ValueError(f"A configured SOC target is outside farm {farm}'s battery limits")
    jobs = []
    # Interleave farms so the shared per-farm inputs are prepared promptly.
    for target in targets:
        for farm in farms:
            config = deepcopy(base[farm])
            config["state"]["terminal_soc_percent"] = float(target)
            config["continuous_optimization"]["run_id"] = f"farm_{farm}_soc_{target}"
            jobs.append({"farm": farm, "target_soc_percent": target, "config": config})
    return jobs


def prepare_for_target(config, batch, progress):
    """Share immutable source inputs while accurately pinning each SOC target."""
    canonical = deepcopy(config)
    # The target changes only planning constraints, not forecasts or actuals.
    canonical["state"]["terminal_soc_percent"] = float(
        config["soc_sweep"]["preflight_reference_target_soc_percent"])
    prepared = prepare_shared(canonical, batch, prepare_continuous_inputs, progress)
    target = float(config["state"]["terminal_soc_percent"])
    prepared["terminal_energy_mwh"] = prepared["plant"].battery.capacity_mwh * target / 100
    prepared["provenance"]["asset_input_settings"]["state"] = deepcopy(config["state"])
    prepared["input_fingerprint"] = _fingerprint(prepared["provenance"])
    return prepared, preparation_key(canonical)


def run_job(job, batch, *, check_only=False):
    """Preflight one farm or replay one independent target trajectory."""
    farm = job["farm"]
    target = job["target_soc_percent"]
    directory = Path(batch) / "runs" / f"farm_{farm}" / f"soc_{target}"
    log = (Path(batch) / "preflight" / f"farm_{farm}.log") if check_only else directory / "tracker.log"
    result = {"farm": farm, "target_soc_percent": target, "status": "failed", "log": str(log)}
    with log.open("w", encoding="utf-8", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            progress = lambda message: print(f"[farm {farm}, target {target}%] {message}", flush=True)
            progress("Preparing and checking shared forecast and market inputs")
            prepared, key = prepare_for_target(job["config"], Path(batch), progress)
            if prepared.get("imbalance_prices_eur_per_mwh") is None:
                raise ValueError("Actual imbalance prices are required")
            result.update(input_fingerprint=prepared["input_fingerprint"],
                          preflight_cache_key=key)
            if check_only:
                # One persistent provenance record per farm is shared by all
                # target runs after the temporary pickle cache is removed.
                write_json(Path(batch) / "preflight" / f"farm_{farm}.provenance.json", prepared["provenance"])
                result["status"] = "preflight_passed"
                return result
            from bess_pv_trading.simulation.continuous import ContinuousReplay
            # The only target-specific provenance difference is the state
            # configuration; the referenced canonical record remains immutable.
            write_json(directory / "input_provenance.json", {
                "shared_record": str(Path(batch) / "preflight" / f"farm_{farm}.provenance.json"),
                "state_override": job["config"]["state"], "input_fingerprint": prepared["input_fingerprint"]})
            replay = ContinuousReplay(job["config"], prepared, "integrated",
                                      directory / "controllers/track_committed_grid", progress=progress)
            progress("Starting continuous grid-tracking trajectory")
            summary = replay.run()
            write_json(directory / "summary.json", summary)
            result["status"] = "complete"
            progress("Trajectory completed")
        except Exception as error:
            result.update(error=str(error), traceback=traceback.format_exc())
            print(result["traceback"], file=stream, flush=True)
    return result


def collect(pool, futures, manifest, path, result_key):
    """Persist each completed status and refill process-pool slots automatically."""
    for future in as_completed(futures):
        job = futures[future]
        try:
            result = future.result()
        except Exception as error:
            result = {"farm": job["farm"], "target_soc_percent": job["target_soc_percent"],
                      "status": "failed", "error": str(error)}
        manifest[result_key].append(result)
        print(f"Farm {result['farm']} SOC {result['target_soc_percent']}%: {result['status']}", flush=True)
        write_json(path, manifest)


def main(argv=None):
    """Print the run plan by default; actions require explicit flags."""
    raw = _read_mapping(CONFIG)
    sweep = raw["soc_sweep"]
    targets = tuple(sweep["target_soc_percent"])
    reference_target = sweep["preflight_reference_target_soc_percent"]
    if (not targets or any(isinstance(value, bool) or not isinstance(value, int) for value in targets)
            or len(set(targets)) != len(targets) or reference_target not in targets):
        raise ValueError("Sweep targets must be unique integer SOC percentages containing the preflight reference")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--farm-ids", nargs="+", choices=FARMS, default=list(sweep["farm_ids"]))
    parser.add_argument("--workers", type=int, default=int(sweep["workers"]))
    parser.add_argument("--solver-threads", type=int, default=int(sweep["solver_threads"]))
    parser.add_argument("--keep-preflight-cache", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id must contain letters, digits, underscores, hyphens or periods")
    if len(set(args.farm_ids)) != len(args.farm_ids):
        parser.error("farm IDs must be unique")
    if args.workers < 1 or args.solver_threads < 1:
        parser.error("worker and solver thread counts must be positive")
    batch = ROOT / "outputs/energy_research" / args.run_id
    jobs = make_jobs(args.farm_ids, targets, batch, args.solver_threads, raw=raw)
    requirements = {farm: _artifact_requirements(next(job["config"] for job in jobs if job["farm"] == farm))
                    for farm in args.farm_ids}
    missing_inputs = {farm: [item for item in items if not item["present"]]
                      for farm, items in requirements.items()}
    plan = {"status": "plan_only", "output_dir": str(batch), "farms": args.farm_ids,
        "soc_targets_percent": list(targets), "trajectory_count": len(jobs),
        "workers": min(args.workers, len(jobs)), "solver_threads_per_job": args.solver_threads,
        "maximum_solver_threads": min(args.workers, len(jobs)) * args.solver_threads,
        "preflight_farms": len(args.farm_ids), "forecast_cache": "one shared immutable snapshot per farm",
        "delivery_period": [str(jobs[0]["config"]["continuous_optimization"]["start_date"]),
                            str(jobs[0]["config"]["continuous_optimization"]["end_date"])],
        "initial_auction": "2025-12-31 DA for 2026-01-01; no 2025-12-31 IDA3",
        "controller": "track_committed_grid", "planning": "same delivery day at every auction, including IDA3",
        "soc_policy": "hard equality at the delivery-day midnight at every auction",
        "checkpoint_enabled": jobs[0]["config"]["continuous_optimization"].get("checkpoint_enabled", True),
        "initial_soc_percent": jobs[0]["config"]["state"]["initial_soc_percent"],
        "preflight_reference_target_soc_percent": reference_target,
        "forecast": "frozen retrospective four-cycle neural models, latest available 48-hour weather issue, six-hour lag",
        "realized_pv": "reviewed AC including labeled synthetic values and night zeros",
        "price_information": "retrospective actual auction price oracle including future prices",
        "auction_submissions_upper_bound": len(jobs) * 4 * (
            (date.fromisoformat(str(jobs[0]["config"]["continuous_optimization"]["end_date"]))
             - date.fromisoformat(str(jobs[0]["config"]["continuous_optimization"]["start_date"]))).days + 1),
        "missing_inputs": missing_inputs, "solves_started": 0, "models_fitted": 0}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.check_only and not args.run:
        return 0
    if any(missing_inputs.values()):
        parser.error("Study inputs are missing; see plan")
    batch.mkdir(parents=True, exist_ok=False)
    (batch / "preflight").mkdir()
    for job in jobs:
        directory = batch / "runs" / f"farm_{job['farm']}" / f"soc_{job['target_soc_percent']}"
        directory.mkdir(parents=True)
        write_json(directory / "resolved_config.json", job["config"])
    path = batch / "manifest.json"
    # Pin executable source even when a user launches from a dirty checkout.
    manifest = {"status": "running", "plan": plan, "git_revision": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(), "source_checksums": source_checksums(),
        "preflight_results": [], "results": []}
    write_json(path, manifest)
    try:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=plan["workers"], mp_context=context,
                                 initializer=configure_worker_threads, initargs=(1, True)) as pool:
            # All farms pass preflight before any optimization starts.
            probes = [next(job for job in jobs if job["farm"] == farm and job["target_soc_percent"] == reference_target)
                      for farm in args.farm_ids]
            futures = {pool.submit(run_job, job, batch, check_only=True): job for job in probes}
            collect(pool, futures, manifest, path, "preflight_results")
            if any(result["status"] != "preflight_passed" for result in manifest["preflight_results"]):
                manifest["status"] = "failed_preflight"
                return 1
            if args.check_only:
                manifest["status"] = "preflight_passed"
                return 0
            print("All farm preflights passed; starting the SOC sweep.", flush=True)
            futures = {pool.submit(run_job, job, batch): job for job in jobs}
            collect(pool, futures, manifest, path, "results")
        failures = [result for result in manifest["results"] if result["status"] != "complete"]
        if any(result["status"] == "complete" for result in manifest["results"]):
            from bess_pv_trading.reporting.same_day_soc_sweep import write_soc_sweep_report
            summary = write_soc_sweep_report(batch, manifest["results"], args.farm_ids, targets)
            manifest["summary_markdown"] = str(summary)
        manifest.update(status="failed" if failures else "complete", failures=failures)
        return 1 if failures else 0
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        raise
    finally:
        if manifest["status"] in {"complete", "preflight_passed"} and not args.keep_preflight_cache:
            manifest["preflight_cache_cleanup"] = clear_preflight_cache(batch)
        else:
            manifest["preflight_cache_cleanup"] = {"status": "retained"}
        write_json(path, manifest)


if __name__ == "__main__":
    raise SystemExit(main())
