"""Audit and report a completed shared-first ensemble weighting comparison."""
from collections import Counter
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from audit_energy_research_results import _scenario_horizons


def report(root):
    """Check stored 51-member horizons, objective, realized PV and paired cash."""
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    if manifest["status"] != "complete" or len(manifest["results"]) != 8:
        raise ValueError("The eight-job batch must complete before reporting")
    records = []
    audits = []
    baselines = {}
    violations = []
    for job in manifest["results"]:
        if job["status"] != "complete":
            raise ValueError(f"Incomplete job: {job['job_id']}")
        farm = str(job["farm_id"])
        case = job["case_id"]
        config = json.loads((root / f"config__{job['job_id']}.json").read_text())
        directory = root / "runs" / f"farm_{farm}" / job["job_id"] / "continuous"
        summary = pd.read_csv(directory / "controller_comparison.csv").set_index("controller")
        for controller in ("track_committed_grid", "ensemble_mpc_adaptive"):
            folder = directory / "controllers" / controller
            intervals = pd.read_csv(folder / "csv/controller_intervals.csv")
            if len(intervals) != 672 or intervals.interval_id.duplicated().any():
                raise ValueError(f"Invalid interval grid: {job['job_id']} {controller}")
            if controller == "track_committed_grid":
                cols = ["net_position_mwh", "grid_mw", "energy_end_mwh", "imbalance_mwh", "net_profit_eur"]
                previous = baselines.setdefault(farm, intervals[cols].copy())
                np.testing.assert_allclose(previous, intervals[cols], atol=1e-7, rtol=0)
            keys = ("absolute_imbalance_mwh", "imbalance_cashflow_eur", "auction_revenue_eur",
                "degradation_eur", "import_cost_eur", "trading_fees_eur", "realized_operating_profit_eur",
                "pv_curtailment_mwh", "throughput_mwh", "final_energy_mwh", "soc_final_percent",
                "mpc_solves", "mpc_fallbacks", "mpc_timeouts", "physical_safety_overrides",
                "solver_runtime_total_seconds")
            records.append({"farm": farm, "case": case, "controller": controller,
                **{key: float(summary.loc[controller, key]) for key in keys}})
            if controller == "track_committed_grid":
                continue
            solves = json.loads((folder / "json/mpc_solves.json").read_text())
            if len(solves) != len(intervals):
                raise ValueError(f"Missing solver diagnostics: {job['job_id']}")
            diagnostics = {row["solve_id"]: row for row in solves}
            actual = intervals.set_index("interval_id")
            statuses = Counter(row["status"] for row in solves)
            sources = Counter()
            checked = 0
            max_error = 0.0
            for solve_id, frame in _scenario_horizons(folder):
                solve = diagnostics[solve_id]
                first_id = solve["interval_id"]
                first = frame[frame.interval_id == first_id]
                if len(first) != 51 or first.scenario_id.nunique() != 51:
                    violations.append((job["job_id"], solve_id, "scenario_count"))
                    continue
                weights = first.weight.to_numpy(dtype=float)
                pv = first.pv_ac_mw.to_numpy(dtype=float)
                commands = first[["charge_mw", "discharge_mw", "pv_utilization_fraction"]].to_numpy(dtype=float)
                expected_pv = float(actual.loc[first_id, "pv_ac_mw"])
                if not np.allclose(weights, 1 / 51, atol=1e-10, rtol=0):
                    violations.append((job["job_id"], solve_id, "weights"))
                if not np.allclose(pv, expected_pv, atol=1e-7, rtol=0):
                    violations.append((job["job_id"], solve_id, "current_pv"))
                if not np.allclose(commands, commands[0], atol=1e-7, rtol=0):
                    violations.append((job["job_id"], solve_id, "shared_first_action"))
                later = frame[frame.interval_id != first_id]
                if not later.empty and later.groupby("interval_id").control_node.nunique().min() != 51:
                    violations.append((job["job_id"], solve_id, "future_scenario_nodes"))
                first_imbalance = float(np.dot(weights, np.abs(first.imbalance_mwh.to_numpy(dtype=float))))
                parts = solve["components"]
                penalty = config["realtime_mpc"]["planning_imbalance_penalty_eur_per_mwh"]
                weight = config["realtime_mpc"].get("first_interval_weight", 1)
                expected = penalty * (parts["absolute_imbalance_mwh"] + (weight - 1) * first_imbalance)
                expected += sum(parts[key] for key in ("degradation_eur", "terminal_cost_eur", "import_cost_eur"))
                error = abs(expected - solve["planning_objective_value"])
                max_error = max(max_error, error)
                if error > 1e-5:
                    violations.append((job["job_id"], solve_id, "objective", error))
                checked += 1
            expected_horizons = sum(not row.get("fallback_applied", False) for row in solves)
            if checked != expected_horizons:
                violations.append((job["job_id"], "horizon_count", checked, expected_horizons))
            provenance = json.loads((directory / "input_provenance.json").read_text())
            for day in provenance.get("days", {}).values():
                curated = day.get("pv_provenance", {}).get("curated_realized_pv") or {}
                sources.update(curated.get("source_counts") or {})
            audits.append({"farm": farm, "case": case, "status_counts": dict(statuses),
                "horizons_checked": checked, "target_sources": dict(sources), "maximum_objective_error_eur": max_error})
    if violations:
        raise ValueError(f"Ensemble first-priority audit violations: {violations[:10]}")
    totals = pd.DataFrame(records)
    totals.to_csv(root / "ensemble_priority_totals.csv", index=False)
    target = totals[totals.controller.eq("ensemble_mpc_adaptive")].set_index(["farm", "case"])
    fields = ("absolute_imbalance_mwh", "imbalance_cashflow_eur", "auction_revenue_eur", "degradation_eur",
        "import_cost_eur", "trading_fees_eur", "realized_operating_profit_eur", "final_energy_mwh", "soc_final_percent")
    comparisons = []
    for farm in sorted(totals.farm.unique()):
        old = target.loc[(farm, "shared_first_original")]
        new = target.loc[(farm, "shared_first_weighted_k10")]
        diff = {field: float(new[field] - old[field]) for field in fields}
        cash = diff["auction_revenue_eur"] + diff["imbalance_cashflow_eur"] - diff["degradation_eur"] - diff["import_cost_eur"] - diff["trading_fees_eur"]
        if not np.isclose(cash, diff["realized_operating_profit_eur"], atol=1e-5):
            raise ValueError(f"Profit attribution mismatch: {farm}")
        comparisons.append({"farm": farm, **diff})
    pd.DataFrame(comparisons).to_csv(root / "ensemble_priority_paired_differences.csv", index=False)
    (root / "ensemble_priority_objective_audit.json").write_text(json.dumps({"jobs": audits, "violations": violations}, indent=2) + "\n")
    print(totals.to_string(index=False))
    print(json.dumps(audits, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    report(parser.parse_args().root)
