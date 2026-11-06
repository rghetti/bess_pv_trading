"""Plan, preflight, or run independent 2026 greedy grid-tracking farm replays."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "configs/research/annual_2026_grid_tracking_farm_1.yaml"
MODEL_ROOT = ROOT / "models/pv/reviewed_four_cycle_2026_selected"
OUTPUT_ROOT = ROOT / "outputs/continuous_optimization/annual_2026_grid_tracking/batches"
RUNNER = ROOT / "scripts/run_realtime_mpc_comparison.py"
FARMS = (1, 345, 901, 910)


def make_config(template, farm, model_name, batch_id, config_dir):
    """Build a standalone study override with the selected farm and frozen model."""
    config = deepcopy(template)
    config["include"] = os.path.relpath(ROOT / "configs/standard.yaml", config_dir.resolve())
    config["farm_id"] = farm
    config["continuous_optimization"]["run_id"] = f"farm_{farm}_four_cycle_{batch_id}"
    config["continuous_optimization"]["output_root"] = str(OUTPUT_ROOT / batch_id / "runs")
    config["continuous_optimization"]["resume"] = False
    config["experiment_outputs"]["overwrite_existing"] = False
    config["pv"]["forecast_replacement"]["models_by_farm"] = {
        str(farm): {
            "name": model_name,
            "kind": "reviewed_hpo",
            "artifact_path": f"models/pv/reviewed_four_cycle_2026_selected/farm_{farm}/result.json",
            "available_at_utc": "2026-11-06T00:00:00Z",
        }
    }
    return config


def invoke(farm, config_path, action, log_path):
    """Run the existing single-farm entry point and preserve its complete log."""
    command = [sys.executable, str(RUNNER), "--config", str(config_path), "--mode", "continuous", action]
    environment = os.environ.copy()
    # Bound native library pools; the YAML independently sets Gurobi threads.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = "1"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=ROOT, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, check=False)
    return {"farm": farm, "exit_code": completed.returncode, "log": str(log_path)}


def execute_phase(configs, action, workers, log_dir):
    """Run independent farms with bounded process concurrency and report failures."""
    records = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        jobs = {}
        for farm, path in configs.items():
            log_path = log_dir / f"farm_{farm}_{action.removeprefix('--')}.log"
            jobs[executor.submit(invoke, farm, path, action, log_path)] = farm
        for future in as_completed(jobs):
            farm = jobs[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"farm": farm, "exit_code": 1, "error": str(error)}
            records.append(result)
            print(json.dumps(result), flush=True)
    return sorted(records, key=lambda item: item["farm"])


def main(argv=None):
    """Present the run plan by default; require a distinct flag for preflight or execution."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farms", nargs="+", type=int, choices=FARMS, required=True)
    parser.add_argument("--workers", type=int, help="Maximum concurrent farm processes (default: up to 2)")
    parser.add_argument("--run-id", required=True, help="New batch identifier containing letters, digits, underscores or hyphens")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if args.workers is None:
        args.workers = min(2, len(args.farms))
    if len(set(args.farms)) != len(args.farms):
        parser.error("Each farm may be selected only once")
    if args.workers < 1 or args.workers > len(args.farms):
        parser.error("--workers must be between 1 and the number of selected farms")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", args.run_id):
        parser.error("Invalid --run-id")

    # Pin the named four-cycle model for each farm and check the study template.
    template = yaml.safe_load(TEMPLATE.read_text(encoding="utf-8"))
    manifest = json.loads((MODEL_ROOT / "manifest.json").read_text(encoding="utf-8"))
    models = {farm: manifest["models"][str(farm)]["model_name"] for farm in args.farms}
    if (template["execution"]["dispatch_policy"] != "track_committed_grid"
            or template["realtime_mpc_comparison"]["controllers"] != ["track_committed_grid"]
            or template["solver"]["threads"] != 9):
        parser.error("The study template must specify greedy tracking only and nine solver threads")
    batch_dir = OUTPUT_ROOT / args.run_id
    plan = {"farms": args.farms, "models": models, "workers": args.workers,
        "solver_threads_per_farm": 9, "maximum_solver_threads": 9 * args.workers,
        "logical_cpus": os.cpu_count(), "start_date": template["continuous_optimization"]["start_date"],
        "end_date": template["continuous_optimization"]["end_date"], "controller": "track_committed_grid",
        "forecast_updates": template["research_information"]["pv_updates"],
        "auction_price_information": template["gme_oracle_experiment"]["price_information"],
        "curated_realized": template["forecast"]["curated_realized"], "output_directory": str(batch_dir),
        "note": "Retrospective models and curated targets; preflight every farm before optimization."}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run and not args.check_only:
        return 0

    # Reserve a new batch only for execution; check-only uses disposable files.
    temporary = tempfile.TemporaryDirectory(prefix="grid_tracking_preflight_") if args.check_only else None
    try:
        if temporary:
            work_dir = Path(temporary.name)
        else:
            batch_dir.mkdir(parents=True, exist_ok=False)
            work_dir = batch_dir
        config_dir = work_dir / "configs"
        log_dir = work_dir / "logs"
        config_dir.mkdir()
        log_dir.mkdir()
        configs = {}
        for farm in args.farms:
            path = config_dir / f"farm_{farm}.yaml"
            config = make_config(template, farm, models[farm], args.run_id, config_dir)
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
            configs[farm] = path
        checks = execute_phase(configs, "--check-only", args.workers, log_dir)
        if any(item["exit_code"] for item in checks):
            # Print the failure tail before temporary preflight logs are removed.
            for item in checks:
                if item["exit_code"] and "log" in item:
                    print(Path(item["log"]).read_text(encoding="utf-8")[-6000:], file=sys.stderr)
            print("Preflight failed; no optimization was started.", file=sys.stderr)
            return 1
        if args.check_only:
            print("All selected farms passed preflight; no optimization was started.")
            return 0
        results = execute_phase(configs, "--run", args.workers, log_dir)
        (batch_dir / "batch_results.json").write_text(json.dumps({"plan": plan, "preflight": checks,
            "runs": results}, indent=2), encoding="utf-8")
        return 1 if any(item["exit_code"] for item in results) else 0
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
