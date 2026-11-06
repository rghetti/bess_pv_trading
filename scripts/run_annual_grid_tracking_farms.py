"""Plan, preflight, or run independent 2026 greedy grid-tracking farm replays."""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import csv
import json
import math
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


def gross_pv_production(interval_path):
    """Integrate available AC PV before curtailment across actual interval durations."""
    contributions = []
    with interval_path.open(newline="", encoding="utf-8") as stream:
        # Reject incomplete or invalid interval data instead of reporting a misleading ratio.
        for row in csv.DictReader(stream):
            power = float(row["pv_ac_mw"])
            duration = float(row["duration_hours"])
            if not math.isfinite(power) or power < 0 or not math.isfinite(duration) or duration <= 0:
                raise ValueError("Invalid PV power or interval duration in controller results")
            contributions.append(power * duration)
    if not contributions:
        raise ValueError("Controller interval results contain no production intervals")
    return math.fsum(contributions)


def write_farm_summary(batch_dir, plan, farm, preflight, run):
    """Summarize one completed or failed replay using its recorded artifacts."""
    reports = batch_dir / "reports"
    reports.mkdir(exist_ok=True)
    report = reports / f"farm_{farm}.md"
    run_id = f"farm_{farm}_four_cycle_{batch_dir.name}"
    destination = batch_dir / "runs" / f"farm_{farm}" / run_id / "continuous"
    controller = destination / "controllers/track_committed_grid"
    model_name = plan["models"].get(farm, plan["models"].get(str(farm)))
    if model_name is None:
        raise ValueError(f"No model recorded for farm {farm}")
    lines = [f"# Farm {farm}: continuous greedy grid tracking", "",
        f"- Period: {plan['start_date']} to {plan['end_date']} (Europe/Rome).",
        f"- Model: `{model_name}`; latest available deterministic forecast cycle.",
        "- Controller: `track_committed_grid`; no MPC decisions.",
        "- Auction planning uses actual price curves; model and curated PV inputs are retrospective.",
        f"- Preflight: {'passed' if preflight['exit_code'] == 0 else 'failed'}."]
    if run is None:
        lines.append("- Run: not started because at least one preflight failed.")
    else:
        lines.append(f"- Run process exit code: {run['exit_code']}.")
    # Incomplete runs do not expose partial numbers as completed results.
    manifest_path = destination / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
    if run is not None:
        lines.append(f"- Replay manifest status: {manifest.get('status', 'unavailable')}.")
    if run and run["exit_code"] == 0 and manifest.get("status") == "complete":
        metrics = json.loads((controller / "json/controller_kpis.json").read_text())
        production = gross_pv_production(controller / "csv/controller_intervals.csv")
        imbalance = metrics.get("absolute_imbalance_mwh")
        curtailment = metrics.get("pv_curtailment_mwh")
        metrics["gross_pv_production_mwh"] = production
        # A zero-production replay has no defined percentage denominator.
        metrics["absolute_imbalance_production_percent"] = (
            100 * imbalance / production if production > 0 and imbalance is not None else None)
        metrics["curtailment_production_percent"] = (
            100 * curtailment / production if production > 0 and curtailment is not None else None)
        measures = [
            ("Realized operating profit", "realized_operating_profit_eur", "EUR"),
            ("Auction revenue", "auction_revenue_eur", "EUR"),
            ("Signed imbalance settlement", "imbalance_cashflow_eur", "EUR"),
            ("Degradation cost", "degradation_eur", "EUR"),
            ("Import cost", "import_cost_eur", "EUR"),
            ("Trading fees", "trading_fees_eur", "EUR"),
            ("Gross PV production before curtailment", "gross_pv_production_mwh", "MWh"),
            ("Absolute imbalance", "absolute_imbalance_mwh", "MWh"),
            ("Absolute imbalance / gross PV production", "absolute_imbalance_production_percent", "%"),
            ("Signed net imbalance", "signed_net_imbalance_mwh", "MWh"),
            ("PV curtailment", "pv_curtailment_mwh", "MWh"),
            ("PV curtailment / gross PV production", "curtailment_production_percent", "%"),
            ("Battery charge", "charge_mwh", "MWh"),
            ("Battery discharge", "discharge_mwh", "MWh"),
            ("Final SOC", "soc_final_percent", "%"),
            ("Physical safety overrides", "physical_safety_overrides", "intervals"),
            ("MPC solves", "mpc_solves", "solves"),
        ]
        lines.extend(["", "## Realized results", "", "| Metric | Value |", "| --- | ---: |"])
        for label, key, unit in measures:
            value = metrics.get(key)
            if value is None:
                display = "unavailable"
            elif unit in {"intervals", "solves"}:
                display = f"{value:,} {unit}"
            else:
                display = f"{value:,.3f} {unit}"
            lines.append(f"| {label} | {display} |")
        lines.extend(["", "Gross PV production is the replay's available PV before curtailment; "
            "it may include synthetic values. Percentages use that entire-period production as the denominator.",
            "Realized profit does not assign a value to final battery inventory."])
        # Keep auction solve diagnostics separate from the zero MPC-solve count.
        solves_path = controller / "json/solves.json"
        if solves_path.is_file():
            solves = json.loads(solves_path.read_text())
            statuses = Counter(str(row.get("status", "unknown")) for row in solves)
            lines.extend(["", "## Auction optimization", "", f"- Solve records: {len(solves)}.",
                f"- Fallbacks: {sum(bool(row.get('fallback_applied')) for row in solves)}."])
            for status, count in sorted(statuses.items()):
                lines.append(f"- {status}: {count}.")
        provenance_path = destination / "input_provenance.json"
        if provenance_path.is_file():
            provenance = json.loads(provenance_path.read_text())
            sources = Counter()
            for day in provenance.get("days", {}).values():
                curated = day.get("pv_provenance", {}).get("curated_realized_pv", {})
                sources.update(curated.get("source_counts", {}))
            if sources:
                lines.extend(["", "## Curated PV input sources", ""])
                for source, count in sorted(sources.items()):
                    lines.append(f"- `{source}`: {count:,} intervals.")
                lines.append("\nSynthetic and night-zero inputs are not independent metered production.")
    elif run is not None:
        lines.extend(["", "No complete metrics are reported. Inspect the run log and manifest for details."])
    lines.extend(["", "## Artifacts", ""])
    # Use relative links so a batch can be moved together with its reports.
    paths = [("Configuration", batch_dir / "configs" / f"farm_{farm}.yaml")]
    if preflight.get("log"):
        paths.append(("Preflight log", Path(preflight["log"])))
    if run and run.get("log"):
        paths.append(("Run log", Path(run["log"])))
    paths.extend([("Replay manifest", manifest_path), ("Daily metrics", controller / "csv/controller_daily_kpis.csv"),
        ("Input provenance", destination / "input_provenance.json"), ("Plots", controller / "plots")])
    for label, path in paths:
        if path.exists():
            lines.append(f"- [{label}]({os.path.relpath(path, reports)})")
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report)


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
            if args.run:
                for item in checks:
                    write_farm_summary(batch_dir, plan, item["farm"], item, None)
            print("Preflight failed; no optimization was started.", file=sys.stderr)
            return 1
        if args.check_only:
            print("All selected farms passed preflight; no optimization was started.")
            return 0
        results = execute_phase(configs, "--run", args.workers, log_dir)
        checks_by_farm = {item["farm"]: item for item in checks}
        report_failures = []
        for result in results:
            farm = result["farm"]
            try:
                result["summary"] = write_farm_summary(batch_dir, plan, farm, checks_by_farm[farm], result)
            except (OSError, ValueError, KeyError, TypeError) as error:
                report_failures.append({"farm": farm, "error": str(error)})
                print(f"Farm {farm} summary failed: {error}", file=sys.stderr)
        (batch_dir / "batch_results.json").write_text(json.dumps({"plan": plan, "preflight": checks,
            "runs": results, "summary_failures": report_failures}, indent=2), encoding="utf-8")
        return 1 if report_failures or any(item["exit_code"] for item in results) else 0
    finally:
        if temporary:
            temporary.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
