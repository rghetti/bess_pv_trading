"""Plan or run independently scheduled continuous tracker and MPC trajectories."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, FIRST_COMPLETED, wait
from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from bess_pv_trading.parallel import configure_worker_threads
from bess_pv_trading.optimization.mpc import MpcSettings
from bess_pv_trading.simulation.energy_research_study import (
    _reports, _write_json, expand_research_jobs, research_plan,
)
from bess_pv_trading.simulation.continuous_inputs import prepare_continuous_inputs
from bess_pv_trading.simulation.realtime_mpc_study import run_realtime_mpc_experiment
from bess_pv_trading.simulation.synthetic import write_json
from k25_preflight_cache import clear_preflight_cache, preparation_key, prepare_shared
from k25_provenance import link_provenance, publish_provenance


STUDIES = {
    "january-ensemble": "configs/research/ensemble_k25_january_2026.yaml",
    "january-august-deterministic": "configs/research/deterministic_k25_jan_aug_2026.yaml",
    "year-deterministic": "configs/research/deterministic_k25_full_year_2026.yaml",
}


def trajectory_destination(job):
    """Place each controller under its case's independent output directory."""
    settings = job["config"]["continuous_optimization"]
    return Path(settings["output_root"]) / f"farm_{job['farm_id']}" / settings["run_id"] / "continuous"


def run_trajectory(task):
    """Prepare one independent input snapshot and execute exactly one controller."""
    job = task["job"]
    controller = task["controller"]
    destination = trajectory_destination(job)
    log_path = destination / "logs" / f"{controller}.log"
    result = {"job_id": job["job_id"], "farm_id": job["farm_id"], "controller": controller,
              "status": "failed", "log": str(log_path)}
    with log_path.open("w", encoding="utf-8", buffering=1) as stream:
        with redirect_stdout(stream), redirect_stderr(stream):
            try:
                config = deepcopy(job["config"])
                progress = lambda message: print(f"[{job['job_id']} / {controller}] {message}", flush=True)
                progress("preparing inputs and forecast cutoffs")
                if task.get("preflight_cache", True):
                    # The cache lives in this batch only; each consumer receives
                    # its own deserialized copy of the pinned prepared inputs.
                    batch = Path(config["continuous_optimization"]["output_root"]).parent
                    prepared = prepare_shared(config, batch, prepare_continuous_inputs, progress)
                    result["preflight_cache_key"] = preparation_key(config)
                else:
                    prepared = prepare_continuous_inputs(config, progress=progress)
                if prepared["imbalance_prices_eur_per_mwh"] is None:
                    raise ValueError("Complete actual settlement prices are required")
                result["input_fingerprint"] = prepared["input_fingerprint"]
                if task["check_only"]:
                    config["realtime_mpc_comparison"]["check_only"] = True
                    run_realtime_mpc_experiment(config, prepared_inputs=prepared, progress=progress)
                    result["status"] = "preflight_passed"
                    return result
                # Matching workers share one immutable record while retaining
                # the existing controller-specific provenance file paths.
                batch = Path(config["continuous_optimization"]["output_root"]).parent
                record = publish_provenance(batch, job["farm_id"], prepared["input_fingerprint"],
                    prepared["provenance"], destination / f"input_provenance__{controller}.json")
                result["shared_input_provenance"] = str(record.relative_to(batch))
                selected = deepcopy(config)
                selected["execution"]["dispatch_policy"] = controller
                from bess_pv_trading.simulation.continuous import ContinuousReplay
                from bess_pv_trading.reporting.realtime_mpc import save_mpc_outputs

                replay = ContinuousReplay(selected, prepared, "integrated",
                    destination / "controllers" / controller, progress=progress)
                progress("starting continuous trajectory")
                summary = replay.run()
                if "mpc_metrics" not in summary:
                    dispatch, _, _ = replay._final_dispatch_and_ledger()
                    summary.update(save_mpc_outputs(selected, replay.destination, dispatch, None, replay.projections))
                write_json(destination / f"summary__{controller}.json", summary)
                result["status"] = "complete"
                progress("trajectory complete")
            except Exception as error:
                result.update(error=str(error), traceback=traceback.format_exc())
                print(result["traceback"], file=sys.stderr, flush=True)
    return result


def run_queue(tasks, workers, per_farm, native_threads, completed):
    """Refill each available slot while enforcing the maximum for every farm."""
    pending = list(tasks)
    active = {}
    results = []
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context,
                             initializer=configure_worker_threads, initargs=(native_threads, True)) as pool:
        while pending or active:
            counts = {}
            for task in active.values():
                farm = task["job"]["farm_id"]
                counts[farm] = counts.get(farm, 0) + 1
            # Submit only an eligible task, so one busy farm cannot block another.
            while pending and len(active) < workers:
                index = next((i for i, task in enumerate(pending)
                    if counts.get(task["job"]["farm_id"], 0) < per_farm), None)
                if index is None:
                    break
                task = pending.pop(index)
                farm = task["job"]["farm_id"]
                counts[farm] = counts.get(farm, 0) + 1
                print(f"Starting farm {farm}, {task['job']['case_id']}, {task['controller']}", flush=True)
                active[pool.submit(run_trajectory, task)] = task
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                task = active.pop(future)
                try:
                    result = future.result()
                except Exception as error:
                    result = {"job_id": task["job"]["job_id"], "farm_id": task["job"]["farm_id"],
                        "controller": task["controller"], "status": "failed", "error": str(error)}
                results.append(result)
                print(f"Finished {result['job_id']} / {result['controller']}: {result['status']}", flush=True)
                completed(results)
    return results


def deduplicate_tracking(tasks):
    """Identify identical tracker trajectories while retaining every MPC task."""
    unique = []
    aliases = []
    trackers = {}
    for task in tasks:
        if task["controller"] != "track_committed_grid":
            unique.append(task)
            continue
        key = preparation_key(task["job"]["config"])
        if key in trackers:
            aliases.append((task, trackers[key]))
        else:
            trackers[key] = task
            unique.append(task)
    return unique, aliases


def reuse_tracking_results(results, aliases):
    """Attach the completed baseline to each compatible comparison case."""
    by_job = {(item["job_id"], item["controller"]): item for item in results}
    for target, source in aliases:
        source_result = by_job[(source["job"]["job_id"], "track_committed_grid")]
        result = dict(source_result)
        result.update(job_id=target["job"]["job_id"], reused_from_job_id=source["job"]["job_id"])
        if result["status"] == "complete":
            source_dir = trajectory_destination(source["job"])
            target_dir = trajectory_destination(target["job"])
            # The copied summary retains the original controller output path,
            # so both comparisons read the one set of baseline artifacts.
            name = "summary__track_committed_grid.json"
            shutil.copyfile(source_dir / name, target_dir / name)
            name = "input_provenance__track_committed_grid.json"
            link_provenance(source_dir / name, target_dir / name)
        results.append(result)
    return results


def main(argv=None):
    """Show a plan by default; execute only a specifically selected action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=STUDIES, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--per-farm", type=int, default=2)
    parser.add_argument("--solver-threads", type=int, help="Override solver threads per trajectory")
    parser.add_argument("--no-preflight-cache", action="store_true",
                        help="Prepare each trajectory independently instead of sharing within this batch")
    parser.add_argument("--keep-preflight-cache", action="store_true",
                        help="Retain the batch-local cache after successful completion")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers < 1 or args.per_farm < 1 or args.per_farm > args.workers:
        parser.error("worker counts must be positive and --per-farm cannot exceed --workers")
    path = ROOT / STUDIES[args.study]
    options = {"project_root": str(ROOT), "run_id": args.run_id, "workers": args.workers,
               "solver_threads": args.solver_threads}
    plan = research_plan(path, **options)
    study, batch, jobs = expand_research_jobs(path, **options)
    tasks = [{"job": job, "controller": controller, "check_only": args.check_only,
              "preflight_cache": not args.no_preflight_cache}
        for job in jobs for controller in job["config"]["realtime_mpc_comparison"]["controllers"]]
    tasks, tracking_aliases = deduplicate_tracking(tasks)
    if tracking_aliases:
        # The original nominal count includes the omitted tracker trajectories.
        plan["auction_submission_solves_upper_bound"] = plan.pop("auction_submission_solves")
    farm_count = len({job["farm_id"] for job in jobs})
    effective_workers = min(args.workers, len(tasks), farm_count * args.per_farm)
    formulations = []
    for job in jobs:
        # Expose the actual daily constraints and MPC priorities before launch.
        settings = MpcSettings.from_config(job["config"])
        formulations.append({"farm": job["farm_id"], "case": job["case_id"],
            "auction_terminal_soc_scope": job["config"]["continuous_optimization"].get(
                "terminal_soc_scope", "final_replay_day"),
            "auction_terminal_soc_percent": job["config"].get("state", {}).get("terminal_soc_percent"),
            "mpc_objective": settings.planning_objective, "first_interval_weight": settings.first_interval_weight,
            "mpc_terminal_inventory_mode": settings.terminal_inventory_mode,
            "mpc_terminal_soc_percent": settings.terminal_soc_target_percent,
            "mpc_fallback_controller": settings.fallback_controller,
            "validation_diagnostics": job["config"]["continuous_optimization"].get("validation_diagnostics", "full")})
    plan.update(scheduler="controller_trajectories", workers=effective_workers, per_farm=args.per_farm,
        configured_cpu_threads=effective_workers * max(study["native_threads"], study["solver_threads"]),
        maximum_solver_threads=effective_workers * study["solver_threads"],
        preflight_cache="disabled" if args.no_preflight_cache else "batch_local_shared",
        preflight_cache_cleanup="keep" if args.keep_preflight_cache else "after_success",
        physical_controller_trajectories=len(tasks), reused_tracking_trajectories=len(tracking_aliases),
        formulations=formulations,
        note="Compatible trajectories share a batch-local snapshot; paired input fingerprints must match."
            if not args.no_preflight_cache else "Each trajectory independently prepares inputs; fingerprints must match.")
    displayed = {key: value for key, value in plan.items() if key != "jobs"}
    displayed["trajectories"] = [{"farm": task["job"]["farm_id"], "case": task["job"]["case_id"],
        "controller": task["controller"]} for task in tasks]
    print(json.dumps(displayed, indent=2), flush=True)
    if not args.check_only and not args.run:
        return 0
    if plan["missing_input_jobs"]:
        print("Missing input artifacts; no trajectory started.", file=sys.stderr)
        return 1
    # A unique batch ID reserves all case and controller outputs without replacement.
    batch.parent.mkdir(parents=True, exist_ok=True)
    batch.mkdir(exist_ok=False)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    checksums = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "src").rglob("*.py"))}
    manifest = {"status": "running", "plan": plan, "git_revision": revision,
                "source_checksums": checksums, "results": []}
    _write_json(batch / "manifest.json", manifest)
    for job in jobs:
        destination = trajectory_destination(job)
        (destination / "logs").mkdir(parents=True, exist_ok=False)
        write_json(destination / "resolved_config.json", job["config"])
        _write_json(destination / "manifest.json", {"status": "running", "mode": "continuous",
            "git_revision": revision, "source_checksums": checksums,
            "controllers": job["config"]["realtime_mpc_comparison"]["controllers"],
            "completed_controllers": []})

    def update(results):
        """Persist completed task statuses as soon as a worker slot frees up."""
        manifest["results"] = results
        _write_json(batch / "manifest.json", manifest)
        for job in jobs:
            selected = [item for item in results if item["job_id"] == job["job_id"]]
            path = trajectory_destination(job) / "manifest.json"
            case_manifest = json.loads(path.read_text())
            case_manifest["completed_controllers"] = [item["controller"] for item in selected
                                                      if item["status"] == "complete"]
            case_manifest["trajectory_results"] = selected
            _write_json(path, case_manifest)

    try:
        results = run_queue(tasks, args.workers, args.per_farm, study["native_threads"], update)
        reuse_tracking_results(results, tracking_aliases)
        update(results)
        failures = [item for item in results if item["status"] not in {"complete", "preflight_passed"}]
        comparisons = []
        if not args.check_only:
            from bess_pv_trading.reporting.realtime_mpc import save_controller_comparison

            for job in jobs:
                selected = [item for item in results if item["job_id"] == job["job_id"]]
                if len(selected) != 2 or any(item["status"] != "complete" for item in selected):
                    continue
                # Reject comparisons if independent workers read different data snapshots.
                if len({item["input_fingerprint"] for item in selected}) != 1:
                    failures.append({"job_id": job["job_id"], "error": "Input fingerprints differ"})
                    continue
                destination = trajectory_destination(job)
                link_provenance(destination / "input_provenance__track_committed_grid.json",
                                destination / "input_provenance.json")
                summaries = {item["controller"]: json.loads(
                    (destination / f"summary__{item['controller']}.json").read_text()) for item in selected}
                comparison = save_controller_comparison(job["config"], destination, summaries, "continuous")
                case_manifest_path = destination / "manifest.json"
                case_manifest = json.loads(case_manifest_path.read_text())
                case_manifest.update(status="complete", input_fingerprint=selected[0]["input_fingerprint"],
                                     comparison=comparison)
                _write_json(case_manifest_path, case_manifest)
                comparisons.append({**{key: job[key] for key in job if key != "config"}, "status": "complete",
                    "result": {"status": "complete", "mode": "continuous", "output_dir": str(destination),
                               "comparison": comparison}})
            if comparisons:
                manifest["reports"] = _reports(batch, comparisons, jobs, study)
            if not failures and len(comparisons) == len(jobs) and manifest.get("reports", {}).get("status") == "complete":
                # Generate one readable batch summary after every comparison is complete.
                from bess_pv_trading.reporting.k25_batch_summary import write_k25_batch_summary

                summary_path = write_k25_batch_summary(batch, jobs, results,
                    plots=study.get("reporting", {}).get("plots", True))
                manifest["summary_markdown"] = str(summary_path.relative_to(batch))
        for job in jobs:
            path = trajectory_destination(job) / "manifest.json"
            case_manifest = json.loads(path.read_text())
            if case_manifest["status"] == "running":
                selected = [item for item in results if item["job_id"] == job["job_id"]]
                passed = args.check_only and len(selected) == 2 and all(
                    item["status"] == "preflight_passed" for item in selected)
                case_manifest["status"] = "preflight_passed" if passed else "failed"
                _write_json(path, case_manifest)
        manifest.update(status="failed" if failures else "preflight_passed" if args.check_only else "complete",
                        failures=failures)
        return 1 if failures else 0
    except BaseException as error:
        manifest.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                        error=str(error) or type(error).__name__)
        raise
    finally:
        # All worker processes have exited by this point. Preserve the cache
        # on interrupted or failed batches for diagnosis; successful batches
        # no longer need their batch-local preflight and model-inference data.
        if manifest["status"] in {"complete", "preflight_passed"} and not args.keep_preflight_cache:
            try:
                manifest["preflight_cache_cleanup"] = clear_preflight_cache(batch)
            except (OSError, ValueError) as error:
                manifest["preflight_cache_cleanup"] = {"status": "failed", "error": str(error)}
                print(f"Warning: could not remove preflight cache: {error}", file=sys.stderr, flush=True)
        else:
            manifest["preflight_cache_cleanup"] = {"status": "retained"}
        _write_json(batch / "manifest.json", manifest)


if __name__ == "__main__":
    raise SystemExit(main())
