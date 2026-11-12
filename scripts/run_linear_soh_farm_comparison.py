"""Run eight independent 2026 battery-aging trajectories concurrently.

Without --run this prints the study plan and does not start a simulation.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stderr, redirect_stdout
import fcntl
import gc
import json
import multiprocessing
from pathlib import Path
import re
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bess_pv_trading.config import load_config
from bess_pv_trading.parallel import configure_worker_threads
from bess_pv_trading.simulation.continuous import _configuration_fingerprint


FARMS = (1, 345, 901, 910)
CASES = {"linear_soh": ROOT / "configs/research/linear_soh_all_farms_2026.yaml",
         "throughput": ROOT / "configs/research/throughput_all_farms_2026.yaml"}


def resolve_jobs(batch):
    """Resolve every farm and model independently, in round-robin order."""
    jobs = []
    for mode in CASES:
        for farm in FARMS:
            config = load_config(CASES[mode], overrides={"farm_id": farm})
            config["continuous_optimization"]["output_root"] = str(batch / "runs")
            config["continuous_optimization"]["run_id"] = mode
            jobs.append({"farm": farm, "mode": mode, "config": config})
    return jobs


def run_job(job, batch, resume=False):
    """Preflight and run one case; persist its log and return failure details."""
    from bess_pv_trading.simulation.continuous import run_continuous_experiment
    from bess_pv_trading.simulation.continuous_inputs import _replacement_evidence

    farm, mode = job["farm"], job["mode"]
    log = Path(batch) / "logs" / f"farm_{farm}_{mode}.log"
    result = {"farm": farm, "mode": mode, "status": "failed", "log": str(log)}
    with log.open("a" if resume else "w", buffering=1) as stream, redirect_stdout(stream), redirect_stderr(stream):
        try:
            def progress(message):
                """Mirror compact chronological progress to the console and log."""
                line = f"[farm {farm} / {mode}] {message}"
                print(line, file=stream, flush=True)
                print(line, file=sys.__stdout__, flush=True)

            progress("Resuming offline preflight" if resume else "Starting offline preflight")
            config = job["config"]
            config["continuous_optimization"]["resume"] = resume
            initial_model = _replacement_evidence(config)
            if initial_model is None or initial_model.get("status") != "pinned":
                raise ValueError("The requested forecast model bundle is missing or not pinned")
            run_continuous_experiment(config, progress=progress)
            final_model = _replacement_evidence(config)
            if final_model.get("checksums") != initial_model.get("checksums"):
                raise ValueError("Forecast model files changed during the trajectory")
            result["forecast_model_checksums"] = initial_model["checksums"]
            result["status"] = "complete"
            progress("Finished")
        except Exception as error:
            result["status"] = getattr(error, "status", "failed")
            result["error"] = str(error)
            result["traceback"] = traceback.format_exc()
            print(result["traceback"], file=stream, flush=True)
            print(f"[farm {farm} / {mode}] {result['status'].upper()}: {error}; see {log}",
                  file=sys.__stdout__, flush=True)
        finally:
            # A worker can receive another case; collect unreachable preflight
            # and replay objects once this run has returned or failed.
            gc.collect()
    return result


def job_key(job):
    """Identify a case without relying on list position."""
    return f"farm_{job['farm']}:{job['mode']}"


def resume_jobs(batch, jobs, manifest):
    """Check identities and exclude completed runs without altering their artifacts."""
    expected = {job_key(job): _configuration_fingerprint(job["config"]) for job in jobs}
    if manifest.get("job_fingerprints") != expected:
        raise ValueError("Cannot resume: batch configuration has changed")
    terminal_statuses = {"complete", "stopped_at_battery_end_of_life"}
    completed = {(item["farm"], item["mode"]): item["status"] for item in manifest.get("results", [])
                 if item.get("status") in terminal_statuses}
    explicitly_failed = {(item["farm"], item["mode"]) for item in manifest.get("results", [])
                         if item.get("status") not in terminal_statuses}
    recovered = []
    for job in jobs:
        identity = (job["farm"], job["mode"])
        path = Path(batch) / "runs" / f"farm_{job['farm']}" / job["mode"] / "json/manifest.json"
        if not path.is_file():
            if identity in completed:
                raise ValueError(f"Completed run metadata is missing: {path}")
            continue
        saved = json.loads(path.read_text())
        if identity in explicitly_failed and saved.get("status") == "complete":
            raise ValueError(f"Finished trajectory failed a post-run check; use a new batch: {path}")
        if identity in completed and (saved.get("status") != completed[identity]
                or saved.get("config_fingerprint") != expected[job_key(job)]):
            raise ValueError(f"Finished run identity or status does not match: {path}")
        # A worker can finish before the parent records its completion.
        if identity not in completed and saved.get("status") in terminal_statuses:
            if saved.get("config_fingerprint") != expected[job_key(job)]:
                raise ValueError(f"Recovered completed run identity does not match: {path}")
            from bess_pv_trading.simulation.continuous_inputs import _replacement_evidence

            provenance_path = path.parent / "input_provenance.json"
            if not provenance_path.is_file():
                raise ValueError(f"Recovered run lacks input provenance: {provenance_path}")
            provenance = json.loads(provenance_path.read_text())
            recorded = provenance.get("forecast_model_artifacts", {})
            current = _replacement_evidence(job["config"])
            if current is None or current.get("checksums") != recorded.get("checksums"):
                raise ValueError(f"Recovered run's forecast model differs from current files: {path}")
            completed[identity] = saved["status"]
            recovered.append({"farm": job["farm"], "mode": job["mode"], "status": saved["status"],
                              "forecast_model_checksums": current["checksums"], "recovered_from_run_manifest": True})
    pending = [job for job in jobs if (job["farm"], job["mode"]) not in completed]
    return pending, completed, recovered


def write_manifest(path, manifest):
    """Atomically preserve the latest queue status for later resume."""
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)


def write_comparison_figures(batch, rows, health_by_farm):
    """Plot comparable market cash, physical throughput and reported daily SOH."""
    if not rows:
        return []
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    output = Path(batch) / "figures"
    output.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame(rows)
    farms = sorted(table.farm.unique())
    positions = np.arange(len(farms), dtype=float)
    figures = []
    # Both panels use identical, matched delivery intervals for each farm pair.
    for field, ylabel, title in (("market_cash_eur", "EUR", "Realized market cash after imbalance"),
                                  ("market_cash_minus_common_proxy_wear_eur", "EUR", "Market cash minus common proxy wear"),
                                  ("throughput_mwh", "AC charge + discharge (MWh)", "Battery throughput")):
        figure, axis = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for offset, mode, label in ((-.2, "throughput", "Fixed capacity, throughput wear"),
                                    (.2, "linear_soh", "Linear SOH")):
            selected = table[table["mode"] == mode].set_index("farm").reindex(farms)
            axis.bar(positions + offset, selected[field].to_numpy(), width=.4, label=label)
        axis.set_xticks(positions, [str(farm) for farm in farms])
        axis.set_xlabel("Farm")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.axhline(0, color="black", linewidth=.6)
        axis.legend()
        path = output / f"{field}.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        figures.append(str(path))
    if health_by_farm:
        figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
        for farm, frame in sorted(health_by_farm.items()):
            last = frame.groupby("delivery_day", sort=True).tail(1)
            axis.plot(pd.to_datetime(last.delivery_day), 100 * last.battery_soh_end, label=f"Farm {farm}")
        axis.set_xlabel("Delivery day")
        axis.set_ylabel("Committed SOH (%)")
        axis.set_title("Linear-SOH trajectory")
        axis.legend()
        path = output / "daily_soh.png"
        figure.savefig(path, dpi=160)
        plt.close(figure)
        figures.append(str(path))
    return figures


def common_linear_wear_proxy(frame, battery, settings):
    """Revalue one executed trajectory using the same provisional SOH assumptions.

    This post-processing counterfactual does not change the executed dispatch,
    enforce shrinking capacity on a fixed-capacity baseline, or measure health.
    """
    import numpy as np
    import pandas as pd

    if settings.mode != "linear_soh" or battery.capacity_mwh <= 0:
        raise ValueError("Common wear proxy requires a positive-capacity linear SOH calibration")
    p = settings.linear_soh
    duration = pd.to_numeric(frame.duration_hours).to_numpy(dtype=float)
    charge = pd.to_numeric(frame.charge_mw).to_numpy(dtype=float)
    discharge = pd.to_numeric(frame.discharge_mw).to_numpy(dtype=float)
    if (not np.isfinite(duration).all() or not np.isfinite(charge).all()
            or not np.isfinite(discharge).all() or np.any(duration <= 0)
            or np.any(charge < 0) or np.any(discharge < 0)):
        raise ValueError("Common wear proxy requires finite, nonnegative executed powers and positive durations")
    internal_throughput = float(np.dot(duration, battery.eta_charge * charge + discharge / battery.eta_discharge))
    calendar_loss = float(p.calendar_loss_per_year * duration.sum() * 3600 / p.calendar_year_seconds)
    cycling_loss = float(p.cycle_capacity_loss_at_reference_life * internal_throughput /
        (2 * p.cycle_life * p.reference_depth_of_discharge * battery.capacity_mwh))
    total_loss = calendar_loss + cycling_loss
    wear = float(settings.replacement_cost_eur / (1 - p.end_of_life_soh) * total_loss)
    return {"common_proxy_internal_throughput_mwh": internal_throughput,
            "common_proxy_calendar_loss_fraction": calendar_loss,
            "common_proxy_cycle_loss_fraction": cycling_loss,
            "common_proxy_total_soh_loss_fraction": total_loss,
            "common_proxy_final_soh": 1 - total_loss,
            "common_proxy_noncash_wear_eur": wear}


def paired_summary(batch, results):
    """Compare only complete matched intervals and explicitly report exclusions."""
    import numpy as np
    import pandas as pd
    from bess_pv_trading.domain import battery_parameters_from_config
    from bess_pv_trading.optimization.degradation import degradation_settings_from_config

    completed = {(x["farm"], x["mode"]) for x in results if x["status"] == "complete"}
    rows = []
    auction_rows = []
    health_by_farm = {}
    for farm in FARMS:
        if not all((farm, mode) in completed for mode in CASES):
            continue
        roots = {mode: Path(batch) / "runs" / f"farm_{farm}" / mode for mode in CASES}
        provenance = {mode: json.loads((root / "json/input_provenance.json").read_text())
                      for mode, root in roots.items()}
        left, right = (provenance[mode] for mode in CASES)
        if set(left["days"]) != set(right["days"]):
            raise ValueError(f"Farm {farm} has different delivery-day provenance")
        for day in left["days"]:
            for key in ("realized_pv_sha256", "auction_price_sha256", "skipped_auctions"):
                if left["days"][day].get(key) != right["days"][day].get(key):
                    raise ValueError(f"Farm {farm} differs in {key} on {day}")
            for auction, source in left["days"][day]["price_provenance"].items():
                other = right["days"][day]["price_provenance"].get(auction)
                if other is None:
                    raise ValueError(f"Farm {farm} lacks matching {auction} provenance on {day}")
                def snapshot_ids(item):
                    """Compare immutable imported snapshots and their normalized checksums."""
                    return sorted((entry.get("storage", {}).get("snapshot_id") or "", entry.get("normalized_sha256") or "")
                                  for entry in item.get("snapshots", []))
                if snapshot_ids(source) != snapshot_ids(other):
                    raise ValueError(f"Farm {farm} has different {auction} snapshots on {day}")
        if left.get("settlement_values_sha256") != right.get("settlement_values_sha256"):
            raise ValueError(f"Farm {farm} has different imbalance prices")
        for key in ("model", "checksums"):
            if left["forecast_model_artifacts"].get(key) != right["forecast_model_artifacts"].get(key):
                raise ValueError(f"Farm {farm} has different forecast model {key}")
        weather_keys = ("cutoff_utc", "forecast_vintage_utc", "aligned_weather_pv_sha256")
        weather = lambda item: [tuple(record.get(key) for key in weather_keys)
                                for record in item.get("weather_forecasts", [])]
        if weather(left) != weather(right):
            raise ValueError(f"Farm {farm} has different weather forecast vintages or hashes")
        saved_configs = {mode: json.loads((root / "json/study_config.json").read_text())
                         for mode, root in roots.items()}
        if any(config.get("state", {}).get("battery_health") is not None for config in saved_configs.values()):
            raise ValueError(f"Farm {farm} common wear proxy requires a fresh battery without prior history")
        batteries = {mode: battery_parameters_from_config(config["battery"])
                     for mode, config in saved_configs.items()}
        battery = batteries["linear_soh"]
        baseline_battery = batteries["throughput"]
        if (battery.capacity_mwh != baseline_battery.capacity_mwh
                or battery.eta_charge != baseline_battery.eta_charge
                or battery.eta_discharge != baseline_battery.eta_discharge):
            raise ValueError(f"Farm {farm} has different battery capacity or efficiencies")
        common_settings = degradation_settings_from_config(saved_configs["linear_soh"])
        frames = {mode: pd.read_csv(root / "strategies/integrated/csv/realized_dispatch.csv")
                  for mode, root in roots.items()}
        first, second = (frames[mode] for mode in CASES)
        if (not first.interval_id.equals(second.interval_id)
                or not np.allclose(first.pv_ac_mw, second.pv_ac_mw, atol=1e-9, rtol=0)):
            raise ValueError(f"Farm {farm} has different interval or realized PV data")
        for mode, frame in frames.items():
            # Attribute only booked trades belonging to the delivered intervals.
            trade_path = roots[mode] / "strategies/integrated/csv/accepted_trades.csv"
            trades = pd.read_csv(trade_path)
            trades = trades[trades.interval_id.isin(set(frame.interval_id))].copy()
            for auction in ("DA", "IDA1", "IDA2", "IDA3"):
                selected = trades[trades.auction == auction]
                auction_rows.append({"farm": farm, "mode": mode, "auction": auction,
                    "cashflow_eur": float(selected.cashflow_eur.sum()),
                    "purchased_mwh": float(selected.purchased_mwh.sum()),
                    "sold_mwh": float(selected.sold_mwh.sum()),
                    "transaction_fee_eur": float(selected.transaction_fee_eur.sum())})
            if not np.isclose(trades.cashflow_eur.sum(), frame.auction_revenue_eur.sum(), atol=1e-6, rtol=1e-8):
                raise ValueError(f"Farm {farm} {mode} auction cash does not reconcile with delivered dispatch")
            if "battery_soh_start" in frame and not np.isclose(frame.battery_soh_start.iloc[0], 1., atol=1e-10, rtol=0):
                raise ValueError(f"Farm {farm} common wear proxy requires an initial SOH of 100%")
            cash = (frame.auction_revenue_eur + frame.imbalance_cashflow_eur
                    - frame.import_fee_eur - frame.transaction_fee_eur).sum()
            proxy = common_linear_wear_proxy(frame, battery, common_settings)
            rows.append({"farm": farm, "mode": mode, "days": frame.delivery_day.nunique(),
                "auction_cash_eur": frame.auction_revenue_eur.sum(),
                "imbalance_cash_eur": frame.imbalance_cashflow_eur.sum(),
                "market_cash_eur": cash, "model_wear_eur": frame.degradation_eur.sum(),
                "cash_minus_model_wear_eur": cash - frame.degradation_eur.sum(),
                "throughput_mwh": frame.throughput_mwh.sum(),
                "final_soh": frame.battery_soh_end.iloc[-1] if "battery_soh_end" in frame else None,
                **proxy,
                "market_cash_minus_common_proxy_wear_eur": cash - proxy["common_proxy_noncash_wear_eur"]})
            if mode == "linear_soh" and "battery_soh_end" in frame:
                health_by_farm[farm] = frame
    destination = Path(batch) / "comparison.csv"
    pd.DataFrame(rows).to_csv(destination, index=False)
    pd.DataFrame(auction_rows).to_csv(Path(batch) / "auction_comparison.csv", index=False)
    # Persist a machine-readable summary and a compact report for complete pairs.
    summary = {"status": "complete" if len(rows) == 2 * len(FARMS) else "partial",
               "completed_farms": sorted({row["farm"] for row in rows}),
               "excluded_farms": sorted(set(FARMS) - {row["farm"] for row in rows}),
               "cases": rows, "auctions": auction_rows}
    (Path(batch) / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    lines = ["# Matched battery-aging comparison", "",
             f"Status: {summary['status']}. Completed farm pairs: {len(summary['completed_farms'])} of {len(FARMS)}.", "",
             "Market cash includes auction and imbalance cash, less transaction and import fees. "
             "Wear is a noncash model estimate; the common proxy revalues both executed trajectories.", "",
             "| Farm | Case | Days | Market cash (€) | Model wear (€) | Common proxy wear (€) | Throughput (MWh) | Final SOH |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in rows:
        soh = "—" if row["final_soh"] is None else f"{100 * row['final_soh']:.3f}%"
        lines.append(f"| {row['farm']} | {row['mode']} | {row['days']} | {row['market_cash_eur']:.2f} | "
                     f"{row['model_wear_eur']:.2f} | {row['common_proxy_noncash_wear_eur']:.2f} | "
                     f"{row['throughput_mwh']:.3f} | {soh} |")
    lines.extend(["", "## Booked auction cash flows", "",
                  "Signed cash flow is sales less purchases, before transaction fees; volumes are gross.", "",
                  "| Farm | Case | Auction | Cash flow (€) | Purchased (MWh) | Sold (MWh) | Fees (€) |",
                  "| --- | --- | --- | ---: | ---: | ---: | ---: |"])
    for row in auction_rows:
        lines.append(f"| {row['farm']} | {row['mode']} | {row['auction']} | {row['cashflow_eur']:.2f} | "
                     f"{row['purchased_mwh']:.3f} | {row['sold_mwh']:.3f} | {row['transaction_fee_eur']:.2f} |")
    lines.extend(["", "The common proxy is a counterfactual, not measured battery health; "
                  "the baseline dispatch was not rerun with fading capacity. Prices use a historical oracle."])
    if summary["excluded_farms"]:
        lines.extend(["", "Excluded incomplete or unmatched farms: " + ", ".join(map(str, summary["excluded_farms"])) + "."])
    (Path(batch) / "comparison.md").write_text("\n".join(lines) + "\n")
    write_comparison_figures(batch, rows, health_by_farm)
    return str(destination), len(rows)


def write_intraday_analysis(batch, results):
    """Reuse the standard intraday profiles for complete matched farm pairs."""
    from bess_pv_trading.reporting.k25_intraday_profiles import write_intraday_profiles

    completed = {(item["farm"], item["mode"]) for item in results if item["status"] == "complete"}
    rows = []
    for farm in FARMS:
        if not all((farm, mode) in completed for mode in CASES):
            continue
        for mode in CASES:
            root = Path(batch) / "runs" / f"farm_{farm}" / mode
            strategy = root / "strategies/integrated"
            rows.append({"farm": farm, "case": mode, "controller": "track_committed_grid",
                "label": mode, "config": json.loads((root / "json/study_config.json").read_text()),
                "summary": json.loads((strategy / "json/kpis.json").read_text()),
                "csv_dir": strategy / "csv"})
    if not rows:
        return None
    return str(write_intraday_profiles(batch, rows, plots=True))


def main(argv=None):
    """Describe the study, or run its eight jobs in a shared worker queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True, help="Fresh batch identifier under outputs/energy_research")
    parser.add_argument("--workers", type=int, default=8, choices=range(1, 9),
                        help="Maximum concurrent trajectories (default: 8)")
    parser.add_argument("--run", action="store_true", help="Actually launch the requested trajectories")
    parser.add_argument("--resume", action="store_true", help="Resume incomplete cases without rerunning completed cases")
    args = parser.parse_args(argv)
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        parser.error("run-id must contain letters, digits, underscores, hyphens or periods")
    batch = ROOT / "outputs/energy_research" / args.run_id
    jobs = resolve_jobs(batch)
    plan = {"period": ["2026-01-01", "2026-08-29"], "farms": FARMS,
            "cases": list(CASES), "jobs": len(jobs), "workers": args.workers, "solver_threads_per_job": 3,
            "native_threads_at_worker_start": 1, "pv_inference_threads_per_worker": 3,
            "daily_limit_efc": 2.0,
            "daily_limit_basis": "AC charge plus discharge / (2 * daily committed available MWh); baseline fixed, SOH case fading",
            "solve_time_limit_seconds": 60, "relative_mip_gap": 1e-4,
            "mode": "integrated auction planning and grid tracking",
            "price_information": "retrospective actual auction-price oracle",
            "pv_information": "latest eligible deterministic forecast, six-hour lag; retrospectively selected models",
            "pv_models": {str(job["farm"]): job["config"]["pv"]["forecast_replacement"]["models_by_farm"][str(job["farm"])]["name"]
                          for job in jobs if job["mode"] == "linear_soh"},
            "input": "reviewed farm PV with historical replacements plus read-only DuckDB GME/TERNA snapshots",
            "missing_auction_policy": "skip entirely absent products, reject partial gaps",
            "model_training_and_selection": "none in this study; previously selected artifacts are retrospective",
            "calibration": "2025 $70/kWh pack price; 6000 EFC/17% cycling plus 1%/year calendar for SOH; baseline 2500 cycles at 80% depth",
            "metrics": "realized cash, own-model wear, throughput, SOH, solver status and common linear aging proxy revaluation",
            "output_directory": str(batch)}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        return 0
    if args.resume:
        if not batch.is_dir():
            parser.error("resume requires an existing batch directory")
    else:
        batch.mkdir(parents=True, exist_ok=False)
        (batch / "logs").mkdir()
    # Hold the batch lock for the complete queue, including manifest updates.
    lock = (batch / "batch.lock").open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise RuntimeError(f"Study batch is already running: {batch}") from None
    manifest_path = batch / "manifest.json"
    if args.resume:
        if not manifest_path.is_file():
            parser.error("resume requires the existing batch manifest")
        manifest = json.loads(manifest_path.read_text())
        jobs, completed, recovered = resume_jobs(batch, jobs, manifest)
        # Retain earlier outcomes for audit, but leave only completed results active.
        previous = manifest.get("results", [])
        manifest.setdefault("previous_results", []).extend(item for item in previous
            if item.get("status") not in {"complete", "stopped_at_battery_end_of_life"})
        manifest["results"] = [item for item in previous if item.get("status") in {"complete", "stopped_at_battery_end_of_life"}] + recovered
        # Worker count may change on resume without changing study inputs.
        manifest.setdefault("worker_counts", []).append(manifest.get("plan", {}).get("workers"))
        manifest["plan"]["workers"] = args.workers
        print(f"Resuming {len(jobs)} unfinished cases; preserving {len(completed)} finished cases.", flush=True)
    else:
        manifest = {"status": "running", "plan": plan, "results": [],
                    "job_fingerprints": {job_key(job): _configuration_fingerprint(job["config"]) for job in jobs}}
    manifest["status"] = "running"
    results = manifest["results"]
    write_manifest(manifest_path, manifest)
    # Eight independent jobs share a queue; each free slot takes the next job.
    pool = ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn"),
                               initializer=configure_worker_threads, initargs=(1, True))
    try:
        futures = {pool.submit(run_job, job, batch, args.resume): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"farm": job["farm"], "mode": job["mode"], "status": "failed", "error": str(error)}
            results.append(result)
            write_manifest(manifest_path, manifest)
    except KeyboardInterrupt:
        # ProcessPoolExecutor otherwise waits for multi-hour trajectories on Ctrl-C.
        for worker in tuple(pool._processes.values()):
            worker.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        manifest["status"] = "interrupted"
        write_manifest(manifest_path, manifest)
        raise
    else:
        pool.shutdown(wait=True)
    completed_pairs = {(item["farm"], item["mode"]) for item in results if item["status"] == "complete"}
    required_pairs = {(farm, mode) for farm in FARMS for mode in CASES}
    manifest["status"] = "complete" if completed_pairs == required_pairs else "completed_with_failures"
    try:
        manifest["comparison"], manifest["comparison_rows"] = paired_summary(batch, results)
        manifest["summary"] = str(batch / "summary.json")
        manifest["comparison_markdown"] = str(batch / "comparison.md")
        manifest["auction_comparison"] = str(batch / "auction_comparison.csv")
        manifest["comparison_figures"] = sorted(str(path) for path in (batch / "figures").glob("*.png"))
        manifest["intraday_analysis"] = write_intraday_analysis(batch, results)
    except Exception as error:
        manifest["status"] = "comparison_failed"
        manifest["comparison_error"] = str(error)
    write_manifest(manifest_path, manifest)
    print(json.dumps(manifest, indent=2))
    return 0 if manifest["status"] == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
