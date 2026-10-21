"""Comparable Gurobi runtimes for individual solves and experiment batches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def timing_statistics(values):
    """Aggregate recorded seconds; missing timings never become zero samples."""
    samples = np.asarray([value for value in values if value is not None], dtype=float)
    if np.any(~np.isfinite(samples)) or np.any(samples < 0):
        raise ValueError("Optimization timings must be finite nonnegative seconds")
    return {"count": len(samples), "total": float(samples.sum()),
        "mean": float(samples.mean()) if len(samples) else None,
        "min": float(samples.min()) if len(samples) else None,
        "max": float(samples.max()) if len(samples) else None,
        "std": float(samples.std(ddof=0)) if len(samples) else None}


def solve_timing(diagnostics, *, category, **context):
    """Keep Gurobi Runtime separate from Python solve and model-build wall times."""
    return {**context, "category": category,
        **{key: diagnostics.get(key) for key in (
            "runtime_seconds", "solve_seconds", "build_seconds", "optimize_call_runtimes_seconds",
            "variables", "binary_variables", "constraints")}}


def optimization_timing_report(solves):
    """Pool raw solve samples, with explicit categories and auction breakdowns."""
    solves = list(solves)

    def summary(rows):
        return {"solve_count": len(rows),
            "missing_runtime_count": sum(row.get("runtime_seconds") is None for row in rows),
            "gurobi_runtime_seconds": timing_statistics(row.get("runtime_seconds") for row in rows),
            "build_seconds": timing_statistics(row.get("build_seconds") for row in rows),
            "solve_wall_seconds": timing_statistics(row.get("solve_seconds") for row in rows)}

    return {"runtime_source": "Gurobi Model.Runtime; summed across optimize calls when a model is retried",
        "units": "seconds", "std_ddof": 0, "timing_available": True,
        "scope": "Actual solver invocations, including unsuccessful solves; excludes data preparation, plotting, execution and frozen-plan simulation",
        **summary(solves),
        "by_category": {category: summary([row for row in solves if row["category"] == category])
            for category in sorted({row["category"] for row in solves})},
        "by_auction": {auction: summary([row for row in solves if row.get("auction") == auction])
            for auction in sorted({row["auction"] for row in solves if row.get("auction")})},
        "solves": solves}


def load_experiment_timing(output):
    """Read a completed report, or recover existing raw timing diagnostics."""
    output = Path(output)
    report = output / "json/optimization_timing.json"
    if report.is_file():
        return json.loads(report.read_text())
    solves = []
    found_diagnostics = False
    # Older experiments already stored per-solve Runtime without an aggregate.
    for name in ("solves.json", "infeasible_auction_decisions.json"):
        path = output / "json/oracle" / name
        if path.is_file():
            found_diagnostics = True
            for row in json.loads(path.read_text()):
                solves.append(solve_timing(row, category="replay",
                    auction=row.get("current_auction"), cutoff_utc=row.get("cutoff"),
                    trigger=row.get("solve_trigger"), status=row.get("status")))
    for category in ("perfect_pv_counterfactual", "perfect_pv_only_counterfactual"):
        path = output / "json" / f"{category}.json"
        if path.is_file():
            found_diagnostics = True
            row = json.loads(path.read_text())
            solves.append(solve_timing(row, category=category, status=row.get("status")))
    return {**optimization_timing_report(solves), "timing_available": found_diagnostics,
        "recovered_from_legacy_diagnostics": True,
        "legacy_runtime_note": "Older diagnostics may retain only the last optimize call when Gurobi was retried"}


def participation_timing_report(experiments):
    """Compare daily replay totals by profile, and pool all actual solve times."""
    cases, solves, seen = [], [], set()
    for experiment in experiments:
        identity = experiment["profile"], experiment["study_date"]
        if identity in seen:
            raise ValueError("Duplicate profile/day in optimization timing aggregation")
        seen.add(identity)
        report = load_experiment_timing(experiment["output_dir"])
        replay = report["by_category"].get("replay", {})
        all_complete = report["timing_available"] and report["missing_runtime_count"] == 0
        replay_complete = bool(replay.get("solve_count")) and replay["missing_runtime_count"] == 0
        replay_total = (replay["gurobi_runtime_seconds"]["total"] if replay_complete else
            0.0 if report["timing_available"] and experiment.get("auction_solves") == 0 else None)
        cases.append({key: experiment.get(key) for key in (
            "profile", "study_date", "auction_planning_mode", "output_dir")}
            | {"solve_count": report["solve_count"],
               "missing_runtime_count": report["missing_runtime_count"],
               "total_runtime_seconds": report["gurobi_runtime_seconds"]["total"] if all_complete else None,
               "replay_runtime_seconds": replay_total})
        solves.extend({**row, "profile": experiment["profile"], "study_date": experiment["study_date"]}
            for row in report["solves"])

    def summary(rows):
        return {"experiment_count": len(rows),
            "missing_experiment_runtime_count": sum(row["total_runtime_seconds"] is None for row in rows),
            "experiment_total_runtime_seconds": timing_statistics(row["total_runtime_seconds"] for row in rows),
            "experiment_replay_runtime_seconds": timing_statistics(row["replay_runtime_seconds"] for row in rows)}

    return {**optimization_timing_report(solves), **summary(cases),
        "experiment_statistics_scope": "One sample per profile/day; total includes benchmarks, replay excludes benchmarks; incomplete timings excluded rather than treated as zero",
        "by_profile": {profile: summary([row for row in cases if row["profile"] == profile])
            for profile in sorted({row["profile"] for row in cases})},
        "experiments": cases}


def report_saved_optimization_timing(config):
    """Refresh timing JSON from saved diagnostics without solving or using APIs."""
    from ..data import config_path
    from ..simulation.synthetic import write_json

    study = config.get("participation_study", {})
    if study.get("run_id") is not None:
        output = config_path(config, study["output_root"]) / str(study["run_id"])
        manifest = json.loads((output / "json/manifest.json").read_text())
        for experiment in manifest["experiments"]:
            child = Path(experiment["output_dir"])
            child_report = load_experiment_timing(child)
            if child_report["timing_available"]:
                write_json(child / "json/optimization_timing.json", child_report)
        report = {**participation_timing_report(manifest["experiments"]),
            "status": "complete" if manifest["status"] == "complete" else "partial",
            "requested_experiments": manifest["requested_experiments"]}
    else:
        settings = config["experiment_outputs"]
        if settings.get("run_id") is None:
            raise ValueError("Saved timing reports require an explicit experiment_outputs.run_id or participation_study.run_id")
        output = config_path(config, settings["root"]) / str(settings["run_id"]) / (
            f"farm_{config['plant']['farm_id']}") / str(config["gme_oracle_experiment"]["delivery_day"])
        report = load_experiment_timing(output)
        if not report["timing_available"]:
            raise ValueError(f"No saved optimization timing diagnostics at {output}")
        manifest_path = output / "json/manifest.json"
        complete = manifest_path.is_file() and json.loads(manifest_path.read_text())["status"] == "complete"
        report["status"] = "complete" if complete else "partial"
    path = output / "json/optimization_timing.json"
    write_json(path, report)
    return {"status": report["status"], "report": str(path),
        "solve_count": report["solve_count"], "missing_runtime_count": report["missing_runtime_count"],
        "gurobi_runtime_seconds": report["gurobi_runtime_seconds"],
        **{key: report[key] for key in ("experiment_count", "experiment_total_runtime_seconds",
            "experiment_replay_runtime_seconds") if key in report}}
