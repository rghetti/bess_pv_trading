"""Plan, preflight or fit small chronological PV ensemble calibrations."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.pv_scenario_calibration import CalibrationSettings, run_farm_calibration
from bess_pv_trading.parallel import configure_worker_threads


def calibration_plan(config):
    """State all modeling and availability assumptions before any fitting occurs."""
    settings = config["pv_scenario_calibration"]
    allowed = {"farm_ids", "workers", "native_threads", "seed", "include_deterministic_member", "training",
               "evaluation_windows", "parameters", "output_directory", "plots"}
    if set(settings) - allowed:
        raise ValueError("Unknown pv_scenario_calibration settings")
    workers = settings["workers"]
    if isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 8:
        raise ValueError("PV calibration uses one to eight spawned workers")
    if settings["native_threads"] != 1 or isinstance(settings["native_threads"], bool):
        raise ValueError("PV calibration requires one native thread per worker")
    if not isinstance(settings["include_deterministic_member"], bool) or not isinstance(settings["plots"], bool):
        raise ValueError("PV calibration flags must be booleans")
    if isinstance(settings["seed"], bool) or not isinstance(settings["seed"], int) or settings["seed"] < 0:
        raise ValueError("PV calibration seed must be a nonnegative integer")
    farms = settings["farm_ids"]
    if not farms or len(set(map(str, farms))) != len(farms) or set(map(str, farms)) - set(config["farms"]):
        raise ValueError("PV calibration farms must be unique configured farms")
    parameters = CalibrationSettings(**settings.get("parameters", {}))
    training = settings["training"]
    if pd.Timestamp(training["start_date"]) > pd.Timestamp(training["end_date"]):
        raise ValueError("PV calibration training dates are reversed")
    cutoff = pd.Timestamp(training["fit_cutoff_utc"])
    if cutoff.tzinfo is None:
        raise ValueError("PV calibration fit cutoff must be timezone aware")
    windows = settings["evaluation_windows"]
    if not windows or len({window["id"] for window in windows}) != len(windows):
        raise ValueError("PV calibration evaluation windows must have unique identifiers")
    for window in windows:
        if (not isinstance(window["id"], str) or not window["id"].replace("_", "").isalnum()
                or not pd.Timestamp(training["end_date"]) < pd.Timestamp(window["start_date"]) <= pd.Timestamp(window["end_date"])):
            raise ValueError("PV calibration evaluation windows must be named, ordered and follow training")
    return {"workflow": "pv_affine_marginal_calibration", "farm_ids": farms, "training": training,
        "evaluation_windows": windows, "model": "global normalized bias and nonnegative spread scale per farm",
        "training_loss": "proper empirical CRPS divided by AC nameplate; no profit-driven parameter selection",
        "preprocessing": "retain raw AC targets; exclude nonfinite/negative and versioned reviewed-episode rows; never impute targets",
        "forecast_source": "preceding delivery-day 00 UTC ensemble, plus optional eligible deterministic member",
        "availability": "cycle plus configured latency; target interval completion; hypothetical immediate model fit at cutoff",
        "retrospective_computation": True, "evaluation": "CRPS, mean MAE, central interval coverage; raw versus calibrated; identical ECC/shuffle marginals",
        "selection": "one declared bounded two-parameter Powell fit; evaluation windows never select or refit",
        "parameter_settings": parameters.__dict__, "compute": {"workers": workers, "native_threads_per_worker": 1, "gpus": 0},
        "outputs": settings["output_directory"], "runtime_application": "apply exported parameters to currently eligible raw MPC forecasts",
        "limitations": "January-only calibration; August transfer is retrospective seasonal generalization; no claim of joint calibration"}


def _worker(config_path, farm, root, check_only):
    """Each process owns its forecast readers and fits only its configured farm."""
    config = load_config(config_path, overrides={"farm_id": int(farm)})
    config["_project_root"] = root
    return run_farm_calibration(config, check_only=check_only)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/pv_scenario_calibration.yaml")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    plan = calibration_plan(config)
    print(json.dumps(plan, indent=2, allow_nan=False), flush=True)
    if not args.check_only and not args.run:
        return 0
    settings = config["pv_scenario_calibration"]
    results = []
    with ProcessPoolExecutor(max_workers=min(settings["workers"], len(settings["farm_ids"])),
            mp_context=multiprocessing.get_context("spawn"), initializer=configure_worker_threads,
            initargs=(1, False)) as executor:
        tasks = {executor.submit(_worker, str(args.config.resolve()), farm, str(ROOT), args.check_only): farm
                 for farm in settings["farm_ids"]}
        for task in as_completed(tasks):
            result = task.result()
            results.append(result)
            print(f"Farm {tasks[task]}: {result['status']}", file=sys.stderr, flush=True)
    if args.run:
        output = ROOT / settings["output_directory"]
        output.mkdir(parents=True, exist_ok=True)
        payload = {"plan": plan, "status": "complete", "farms": sorted(results, key=lambda row: row["farm_id"])}
        (output / "calibration_summary.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        lines = ["# Frozen PV ensemble calibration", "",
            "Raw AC targets remain untouched. January calibration parameters are frozen for both evaluation windows.", "",
            "| Farm | Window | Forecast | Eligible quarters | CRPS MW | Mean MAE MW | Central coverage |",
            "|---|---|---|---:|---:|---:|---:|"]
        for result in payload["farms"]:
            for evaluation in result["evaluations"]:
                for name, scores in evaluation["scores"].items():
                    lines.append(f"| {result['farm_id']} | {evaluation['window']['id']} | {name} | {scores['eligible_intervals']} | "
                        f"{scores['crps_mw']} | {scores['mae_mean_mw']} | {scores['central_interval_coverage']} |")
        (output / "calibration_summary.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"status": "preflight_passed" if args.check_only else "complete", "farm_count": len(results),
        "fits_started": 0 if args.check_only else len(results)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
