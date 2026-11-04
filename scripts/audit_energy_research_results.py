"""Independently audit completed continuous MPC/tracking artifacts without solving."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.domain import battery_parameters_from_config
from bess_pv_trading.reporting.mpc_history import iter_mpc_history
from bess_pv_trading.scenario_research import observable_history_nodes


def _read_json(path):
    """Read an already completed artifact; the auditor never changes the source."""
    return json.loads(Path(path).read_text())


def _maximum(values):
    """Return an absolute residual while preserving invalid numeric input as failure."""
    values = np.asarray(values, dtype=float)
    return float(np.max(np.abs(values))) if values.size and np.isfinite(values).all() else None


def _scenario_batches(directory, batch_size=65536):
    """Stream both current checksummed partitions and legacy CSV horizon history."""
    if (directory / "json/mpc_horizon_history.json").is_file():
        yield from iter_mpc_history(directory, kind="scenarios", batch_size=batch_size)
    elif (directory / "csv/mpc_scenarios.csv").is_file():
        try:
            yield from pd.read_csv(directory / "csv/mpc_scenarios.csv", chunksize=batch_size)
        except pd.errors.EmptyDataError:
            return


def _scenario_horizons(directory):
    """Carry at most one unfinished solve between bounded file batches."""
    pending = None
    for frame in _scenario_batches(directory):
        if frame.empty:
            continue
        if "charge_mw" not in frame:
            return
        frame = frame if pending is None else pd.concat([pending, frame], ignore_index=True)
        last = frame.solve_id.iloc[-1]
        complete = frame[frame.solve_id != last]
        for solve_id, selected in complete.groupby("solve_id", sort=False):
            yield solve_id, selected
        pending = frame[frame.solve_id == last].copy()
    if pending is not None and not pending.empty:
        yield pending.solve_id.iloc[0], pending


def audit_conditional_horizons(directory, config, actual, ledger, solves, tolerance=1e-6):
    """Independently check conditional physics and observation-based nonanticipativity."""
    battery = battery_parameters_from_config(config["battery"])
    # Older solve summaries may lack identifiers and have no conditional path
    # history. A recorded conditional horizon still requires its matching solve.
    diagnostics = {row["solve_id"]: row for row in solves if "solve_id" in row}
    actual = actual.set_index("interval_id")
    ledger_frame = pd.DataFrame(ledger)
    if not ledger_frame.empty:
        ledger_frame["submission_time"] = pd.to_datetime(ledger_frame.submission_time, utc=True)
    maxima = defaultdict(float)
    failures = []
    checked = 0
    for solve_id, frame in _scenario_horizons(directory):
        diagnostic = diagnostics.get(solve_id)
        if diagnostic is None:
            failures.append({"solve_id": solve_id, "reason": "missing_solve_diagnostic"})
            continue
        scenarios = diagnostic["scenario_ids"]
        intervals = sorted(frame.interval_id.unique(), key=pd.Timestamp)
        if len(intervals) != diagnostic["horizon_intervals"]:
            failures.append({"solve_id": solve_id, "reason": "incomplete_horizon"})
            continue
        def matrix(name):
            return frame.pivot(index="scenario_id", columns="interval_id", values=name).reindex(
                index=scenarios, columns=intervals).to_numpy(float)
        pv, charge, discharge = (matrix(name) for name in ("pv_ac_mw", "charge_mw", "discharge_mw"))
        utilization, start, end = (matrix(name) for name in ("pv_utilization_fraction", "energy_start_mwh", "energy_end_mwh"))
        grid, imbalance, nodes = (matrix(name) for name in ("grid_mw", "imbalance_mwh", "control_node"))
        weights = matrix("weight")
        cutoff = pd.Timestamp(diagnostic["cutoff_utc"])
        committed = np.zeros(len(intervals))
        if not ledger_frame.empty:
            eligible = ledger_frame[ledger_frame.submission_time <= cutoff]
            committed = eligible.groupby("interval_id").quantity_mwh.sum().reindex(intervals, fill_value=0).to_numpy()
        predicted_energy = (start * (1 - battery.self_discharge_per_hour) ** 0.25
            + battery.eta_charge * charge * 0.25 - discharge * 0.25 / battery.eta_discharge)
        expected_nodes = np.zeros(pv.shape, dtype=int)
        if diagnostic["controller"] == "ensemble_mpc_adaptive":
            settings = config.get("realtime_mpc", {})
            stages = settings.get("recourse_stages")
            stages = None if stages is None else [stage for stage in stages if stage < len(intervals)]
            expected_nodes = observable_history_nodes(pv,
                revelation_lag_intervals=settings.get("recourse_observation_lag_intervals", 1),
                branch_intervals=stages, observation_quantum_mw=settings.get("recourse_pv_bin_width_mw", 0) or None)
        residuals = {"energy_transition_mwh": _maximum(predicted_energy - end),
            "energy_continuity_mwh": _maximum(end[:, :-1] - start[:, 1:]) if len(intervals) > 1 else 0.0,
            "initial_energy_mwh": _maximum(start[:, 0] - actual.loc[intervals[0], "energy_start_mwh"]),
            "soc_lower_mwh": float(np.maximum(battery.energy_min_mwh - np.minimum(start, end), 0).max()),
            "soc_upper_mwh": float(np.maximum(np.maximum(start, end) - battery.maximum_energy, 0).max()),
            "power_lower_mw": float(np.maximum(-np.minimum(charge, discharge), 0).max()),
            "power_upper_mw": float(np.maximum(np.maximum(charge, discharge) - battery.max_power_mw, 0).max()),
            "simultaneous_charge_discharge_mw": float(np.minimum(charge, discharge).max()),
            "pv_utilization_bounds": float(np.maximum(np.maximum(-utilization, utilization - 1), 0).max()),
            "grid_balance_mw": _maximum(grid - pv * utilization - discharge + charge),
            "poi_import_mw": float(np.maximum(-config["plant"]["poi_import_mw"] - grid, 0).max()),
            "poi_export_mw": float(np.maximum(grid - config["plant"]["poi_export_mw"], 0).max()),
            "committed_deviation_mwh": _maximum(grid * 0.25 - committed[None, :] - imbalance),
            "weights_constant": _maximum(weights - weights[:, :1]),
            "probability_mass": _maximum(weights.sum(axis=0) - 1),
            "nonanticipativity_mw": 0.0, "history_node_identity": _maximum(nodes - expected_nodes)}
        controls = np.stack([charge, discharge, utilization], axis=2)
        for t in range(len(intervals)):
            for node in np.unique(expected_nodes[:, t]):
                members = controls[expected_nodes[:, t] == node, t]
                residuals["nonanticipativity_mw"] = max(residuals["nonanticipativity_mw"], _maximum(members - members[0]))
        for name, value in residuals.items():
            if value is None or not np.isfinite(value) or value > tolerance:
                failures.append({"solve_id": solve_id, "reason": name, "residual": value})
            elif value > maxima[name]:
                maxima[name] = value
        checked += 1
    return {"status": "checked" if checked else "no_conditional_scenario_controls_in_history",
        "horizons_checked": checked, "maximum_residuals": dict(maxima), "violations": failures,
        "input_formats": "checksummed daily Parquet partitions or bounded legacy CSV batches"}


def audit_controller(directory, config, *, tolerance=1e-6):
    """Reconstruct energy, cash and immutable trade positions from interval outputs."""
    directory = Path(directory)
    frame = pd.read_csv(directory / "csv/controller_intervals.csv")
    summary = _read_json(directory / "json/controller_kpis.json")
    solves_path = directory / "json/mpc_solves.json"
    solves = _read_json(solves_path) if solves_path.is_file() else []
    ledger = _read_json(directory / "json/accepted_trades.json")
    publications = _read_json(directory / "json/price_publications.json")
    battery = battery_parameters_from_config(config["battery"])
    duration = (pd.to_datetime(frame.end, utc=True) - pd.to_datetime(frame.start, utc=True)).dt.total_seconds().to_numpy() / 3600
    expected_energy = (frame.energy_start_mwh.to_numpy() * (1 - battery.self_discharge_per_hour) ** duration
        + battery.eta_charge * frame.charge_mw.to_numpy() * duration
        - frame.discharge_mw.to_numpy() * duration / battery.eta_discharge)
    negative_price = frame.get("imbalance_negative_price_eur_mwh", frame.imbalance_price_eur_mwh)
    expected_settlement = np.maximum(frame.imbalance_mwh, 0) * frame.imbalance_price_eur_mwh
    expected_settlement -= np.maximum(-frame.imbalance_mwh, 0) * negative_price
    expected_profit = frame.auction_revenue_eur + expected_settlement - frame.degradation_eur
    expected_profit -= frame.transaction_fee_eur + frame.import_fee_eur
    positions, revenues, trade_ids = defaultdict(float), defaultdict(float), []
    # Each guaranteed submission must appear once, at its published clearing price.
    for trade in ledger:
        positions[trade["interval_id"]] += trade["quantity_mwh"]
        revenues[trade["interval_id"]] += trade["quantity_mwh"] * trade["price_eur_per_mwh"]
        trade_ids.append(trade["trade_id"])
    reconstructed_position = np.array([positions[product] for product in frame.interval_id])
    reconstructed_revenue = np.array([revenues[product] for product in frame.interval_id])
    residuals = {
        "quarter_hour_duration": _maximum(duration - 0.25),
        "energy_transition_mwh": _maximum(expected_energy - frame.energy_end_mwh),
        "energy_continuity_mwh": _maximum(frame.energy_end_mwh.to_numpy()[:-1] - frame.energy_start_mwh.to_numpy()[1:])
            if len(frame) > 1 else 0.0,
        "grid_balance_mw": _maximum(frame.pv_ac_mw - frame.curtailment_mw + frame.discharge_mw - frame.charge_mw - frame.grid_mw),
        "commitment_mwh": _maximum(reconstructed_position - frame.net_position_mwh),
        "auction_revenue_eur": _maximum(reconstructed_revenue - frame.auction_revenue_eur),
        "imbalance_mwh": _maximum(frame.grid_mw * duration - frame.net_position_mwh - frame.imbalance_mwh),
        "settlement_eur": _maximum(expected_settlement - frame.imbalance_cashflow_eur),
        "profit_eur": _maximum(expected_profit - frame.net_profit_eur),
        "summary_profit_eur": abs(float(frame.net_profit_eur.sum()) - summary["realized_operating_profit_eur"]),
        "summary_absolute_imbalance_mwh": abs(float(frame.imbalance_mwh.abs().sum()) - summary["absolute_imbalance_mwh"]),
        "soc_lower_mwh": float(np.maximum(battery.energy_min_mwh - frame[["energy_start_mwh", "energy_end_mwh"]].to_numpy(), 0).max()),
        "soc_upper_mwh": float(np.maximum(frame[["energy_start_mwh", "energy_end_mwh"]].to_numpy() - battery.maximum_energy, 0).max()),
        "power_lower_mw": float(np.maximum(-frame[["charge_mw", "discharge_mw"]].to_numpy(), 0).max()),
        "power_upper_mw": float(np.maximum(frame[["charge_mw", "discharge_mw"]].to_numpy() - battery.max_power_mw, 0).max()),
        "simultaneous_charge_discharge_mw": float(np.minimum(frame.charge_mw, frame.discharge_mw).max()),
        "poi_import_mw": float(np.maximum(-config["plant"]["poi_import_mw"] - frame.grid_mw, 0).max()),
        "poi_export_mw": float(np.maximum(frame.grid_mw - config["plant"]["poi_export_mw"], 0).max()),
        "curtailment_lower_mw": float(np.maximum(-frame.curtailment_mw, 0).max()),
        "curtailment_upper_mw": float(np.maximum(frame.curtailment_mw - frame.pv_ac_mw, 0).max()),
        "publication_cash_eur": abs(sum(row["booked_cashflow_eur"] for row in publications) - sum(revenues.values()))}
    violations = [name for name, value in residuals.items() if value is None or not np.isfinite(value) or value > tolerance]
    if frame.interval_id.duplicated().any():
        violations.append("duplicate_intervals")
    if len(set(trade_ids)) != len(trade_ids):
        violations.append("duplicate_trade_ids")
    publication_ids = [(row["delivery_day"], row["auction"]) for row in publications]
    if len(set(publication_ids)) != len(publication_ids):
        violations.append("duplicate_price_publications")
    if sum(row["booked_trades"] for row in publications) != len(ledger):
        violations.append("publication_trade_count")
    refinement_failures = []
    oracle_provenance_conflicts = 0
    oracle_cap_violations = 0
    oracle_cap_details = []
    cap_tolerance = 2 * (config.get("solver", {}).get("feasibility_tolerance") or 1e-6)
    for solve in solves:
        if solve.get("status") == "oracle_refinement_rejected":
            refinement_failures.append({"interval_id": solve["interval_id"], "reason": solve.get("failure_reason"),
                "component_residuals": solve.get("component_residuals"),
                "tracking_fallback": solve.get("fallback_applied", False)})
        if solve.get("price_oracle_enabled"):
            metadata = (solve.get("oracle_provenance") or {}).get("settlement", {})
            oracle_provenance_conflicts += metadata.get("supplied_to_optimizer") is False
            cap = solve.get("oracle_imbalance_cap_mwh")
            value = solve.get("components", {}).get("absolute_imbalance_mwh")
            if cap is not None and value is not None and value > cap + cap_tolerance:
                oracle_cap_violations += 1
                oracle_cap_details.append({"interval_id": solve["interval_id"], "cap_mwh": cap,
                    "reconstructed_absolute_imbalance_mwh": value, "excess_mwh": value - cap,
                    "allowed_numerical_tolerance_mwh": cap_tolerance,
                    "oracle_stage2_used": solve.get("oracle_stage2_used", False)})
    if oracle_cap_violations:
        violations.append("accepted_oracle_plan_exceeds_its_recorded_imbalance_cap")
    conditional = audit_conditional_horizons(directory, config, frame, ledger, solves, tolerance)
    if conditional["violations"]:
        violations.append("conditional_horizon_physics_or_nonanticipativity")
    return {"controller": summary["controller"], "summary": summary, "interval_count": len(frame),
        "ledger_trades": len(ledger), "price_publications": len(publications), "residuals": residuals,
        "violations": violations, "solve_statuses": dict(Counter(row["status"] for row in solves)),
        "oracle_refinement_rejections": refinement_failures,
        "oracle_nested_provenance_conflicts": int(oracle_provenance_conflicts),
        "oracle_cap_violations": oracle_cap_violations,
        "oracle_cap_details": oracle_cap_details,
        "oracle_stage2_solves": sum(bool(row.get("oracle_stage2_used")) for row in solves),
        "conditional_horizon_audit": conditional,
        "solver_wall_seconds": float(sum(row.get("solve_wall_seconds", 0) for row in solves)),
        "solver_runtime_seconds": float(sum(row.get("runtime_seconds", 0) for row in solves)),
        "first_action_prediction_error_mwh": summary.get("realized_minus_expected_absolute_imbalance_mwh")}, frame


def audit_batch(batch, *, allow_partial=False):
    """Pin one manifest version and inspect only jobs explicitly marked complete."""
    batch = Path(batch).resolve()
    manifest_bytes = (batch / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if manifest.get("status") != "complete" and not allow_partial:
        raise ValueError("The batch is not complete; use --allow-partial only for explicitly provisional inspection")
    jobs, violations = [], []
    tracking_baselines = {}
    baseline_differences = []
    for item in manifest.get("results", []):
        if item["status"] != "complete":
            violations.append({"job_id": item["job_id"], "reason": "job_not_complete", "status": item["status"]})
            continue
        config = _read_json(batch / f"config__{item['job_id']}.json")
        directory = Path(item["result"]["output_dir"])
        # Recorded absolute paths may be relocated together with their batch.
        if not directory.is_dir():
            directory = batch / "runs" / f"farm_{item['farm_id']}" / item["job_id"] / "continuous"
        controller_results = []
        frames = {}
        for controller in ("track_committed_grid", item["controller"]):
            result, frame = audit_controller(directory / "controllers" / controller, config)
            controller_results.append(result)
            frames[controller] = frame
            violations.extend({"job_id": item["job_id"], "controller": controller, "reason": reason}
                for reason in result["violations"])
        target, tracking = frames[item["controller"]], frames["track_committed_grid"]
        if not target.interval_id.equals(tracking.interval_id) or _maximum(target.pv_ac_mw - tracking.pv_ac_mw) > 1e-12:
            violations.append({"job_id": item["job_id"], "reason": "different_actual_pv_or_product_grid"})
        key = (str(item["farm_id"]), item["window_id"])
        metrics = ["net_profit_eur", "net_position_mwh", "energy_end_mwh"]
        if key in tracking_baselines:
            previous_case, previous = tracking_baselines[key]
            baseline_differences.append({"farm_id": key[0], "window_id": key[1], "case_id": item["case_id"],
                "reference_case": previous_case, "max_interval_differences": {
                    metric: _maximum(tracking[metric] - previous[metric]) for metric in metrics}})
        else:
            tracking_baselines[key] = (item["case_id"], tracking)
        target_summary = controller_results[1]["summary"]
        reference_summary = controller_results[0]["summary"]
        jobs.append({"job_id": item["job_id"], "case_id": item["case_id"], "farm_id": str(item["farm_id"]),
            "window_id": item["window_id"], "controller": item["controller"],
            "profit_difference_eur": target_summary["realized_operating_profit_eur"] - reference_summary["realized_operating_profit_eur"],
            "imbalance_difference_mwh": target_summary["absolute_imbalance_mwh"] - reference_summary["absolute_imbalance_mwh"],
            "auction_revenue_difference_eur": target_summary["auction_revenue_eur"] - reference_summary["auction_revenue_eur"],
            "wear_difference_eur": target_summary["degradation_eur"] - reference_summary["degradation_eur"],
            "final_inventory_difference_mwh": target_summary["final_energy_mwh"] - reference_summary["final_energy_mwh"],
            "position_change_mwh": float((target.net_position_mwh - tracking.net_position_mwh).abs().sum()),
            "oracle_relative_imbalance_budget": config.get("realtime_mpc", {}).get("oracle_relative_imbalance_budget", 0),
            "oracle_absolute_imbalance_budget_mwh": config.get("realtime_mpc", {}).get("oracle_absolute_imbalance_budget_mwh", 0),
            "controllers": controller_results})
    expected = manifest.get("plan", {}).get("job_count")
    if manifest.get("status") == "complete" and expected != len(jobs):
        violations.append({"reason": "completed_job_count_differs_from_plan", "expected": expected, "actual": len(jobs)})
    return {"schema": "continuous-mpc-independent-audit-v1", "batch": str(batch),
        "batch_status": manifest.get("status"), "provisional": manifest.get("status") != "complete",
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(), "expected_jobs": expected,
        "completed_jobs_audited": len(jobs), "jobs": jobs, "violations": violations,
        "tracking_baseline_comparisons": baseline_differences}


def markdown_audit(audit):
    """Render factual outcomes while separating numerical/reporting issues from physics."""
    rows = []
    for job in audit["jobs"]:
        target = job["controllers"][1]
        summary = target["summary"]
        rows.append({"case": job["case_id"], "farm": job["farm_id"], "window": job["window_id"],
            "profit": job["profit_difference_eur"], "imbalance": job["imbalance_difference_mwh"],
            "wear": job["wear_difference_eur"], "inventory": job["final_inventory_difference_mwh"],
            "fallback": summary["mpc_fallbacks"], "timeouts": summary["mpc_timeouts"],
            "safety": summary["physical_safety_overrides"],
            "refinement": len(target["oracle_refinement_rejections"]), "solves": summary["mpc_solves"]})
    frame = pd.DataFrame(rows)
    lines = ["# Continuous MPC validation", "",
        f"Source batch: `{audit['batch']}`.",
        f"Pinned manifest SHA-256: `{audit['manifest_sha256']}`.",
        f"Batch status: **{audit['batch_status']}**; {audit['completed_jobs_audited']}/{audit['expected_jobs']} jobs audited.", ""]
    if audit["provisional"]:
        lines += ["**Provisional inspection. The batch is still incomplete; totals cover completed jobs only.**", ""]
    if frame.empty:
        return "\n".join(lines + ["No completed jobs.", ""])
    lines += ["Differences below are target MPC minus its own continuous grid-tracking baseline. Positive profit is higher realized operating profit; positive imbalance is more absolute imbalance. These are retrospective development windows, not an untouched test set.", "",
        "| Window | Farm | Controller case | Profit difference EUR | Imbalance difference MWh | Wear difference EUR | Final inventory difference MWh | Tracking fallbacks | Timeouts | Safety overrides | Rejected oracle refinements |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in frame.sort_values(["window", "farm", "case"]).to_dict("records"):
        lines.append(f"| {row['window']} | {row['farm']} | {row['case']} | {row['profit']:.3f} | {row['imbalance']:.6f} | {row['wear']:.3f} | {row['inventory']:.6f} | {row['fallback']} | {row['timeouts']} | {row['safety']} | {row['refinement']} |")
    lines += ["", "## Matched controller totals", "",
        "| Case | Completed jobs | Profit difference EUR | Imbalance difference MWh | MPC solves | Tracking fallbacks | Timeouts | Rejected refinements |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for case, selected in frame.groupby("case", sort=True):
        sums = selected.sum(numeric_only=True)
        lines.append(f"| {case} | {len(selected)} | {sums.profit:.3f} | {sums.imbalance:.6f} | {int(sums.solves)} | {int(sums.fallback)} | {int(sums.timeouts)} | {int(sums.refinement)} |")
    controllers = [controller for job in audit["jobs"] for controller in job["controllers"]]
    max_residuals = {name: max(controller["residuals"][name] or 0 for controller in controllers)
        for name in controllers[0]["residuals"]}
    lines += ["", "## Independent checks", "",
        f"The auditor found **{len(audit['violations'])} physical/accounting/artifact violations** at a `1e-6` reconstruction tolerance. It checks SOC and power bounds, charge/discharge exclusivity, POI bounds, energy transition/continuity, PV balance, immutable ledger positions, single price publications, signed settlement, operating profit and recorded oracle imbalance caps.", "",
        "| Reconstructed residual | Maximum |", "|---|---:|"]
    for name, value in max_residuals.items():
        lines.append(f"| {name} | {value:.8g} |")
    failures = [(job["job_id"], failure) for job in audit["jobs"]
        for failure in job["controllers"][1]["oracle_refinement_rejections"]]
    lines += ["", "## Numerical and provenance findings", ""]
    if failures:
        lines += ["Some oracle economic refinements were rejected by independent objective accounting checks. Their already validated minimum-imbalance stage-one plans were retained. These are distinct from grid-tracking fallbacks.", ""]
        for job_id, failure in failures:
            lines.append(f"- `{job_id}`, `{failure['interval_id']}`: `{failure['reason']}`. Tracking fallback: `{failure['tracking_fallback']}`.")
    conflicts = sum(controller["oracle_nested_provenance_conflicts"] for controller in controllers)
    cap_details = [(job["job_id"], detail) for job in audit["jobs"]
        for detail in job["controllers"][1]["oracle_cap_details"]]
    lines += ["", f"{len(cap_details)} accepted oracle plans exceed their recorded reconstructed-MWh cap beyond the configured `2 * FeasibilityTol` allowance.", ""]
    for job_id, detail in cap_details:
        lines.append(f"- `{job_id}`, `{detail['interval_id']}`: cap `{detail['cap_mwh']:.12g}` MWh; reconstructed `{detail['reconstructed_absolute_imbalance_mwh']:.12g}` MWh; excess `{detail['excess_mwh']:.8g}` MWh; allowance `{detail['allowed_numerical_tolerance_mwh']:.8g}` MWh.")
    lines += ["", f"{conflicts} explicit oracle solves retain nested source metadata saying settlement was supplied only for ex-post accounting. This is a provenance-label contradiction; the top-level oracle disclosure is explicit.", "",
        "## Interpretation", "",
        "Grid tracking uses actual interval PV feedback; these MPC runs use forecast current PV. Perfect-foresight auction prices are common across members. Oracle physical controllers additionally see realized settlement prices, but retain the configured minimum-imbalance budget; an oracle controller is not a guaranteed upper bound on realized profit under uncertain PV.", "",
        "The continuous comparisons include changed battery states and consequent later auction submissions. Profit differences therefore include auction cash-flow changes as well as physical dispatch. Final stored energy can differ, and reported profit has no inventory-value adjustment. These runs impose a soft 50% terminal-SOC penalty inside MPC while tracking has no equivalent soft terminal objective and the shared hard terminal target is null. Cash-profit rankings therefore do not isolate the controller from its terminal-inventory tradeoff. An increase in profit alongside more imbalance is possible under signed settlement; it is not by itself evidence of a forecasting improvement or a controller bug.", "",
        "The short, selected January/August windows and shared prices across farms do not support precise extreme-tail estimates or independent-farm significance claims. Broader deterministic testing should retain paired tracking baselines, begin with the configured October 2025 week, and then include the full May 2026 month; these windows require no absent ensemble archive.", ""]
    return "\n".join(lines)


def main(argv=None):
    """Write audit artifacts outside the immutable source batch when requested."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--output", type=Path, help="Destination Markdown report; JSON is written beside it")
    args = parser.parse_args(argv)
    audit = audit_batch(args.batch, allow_partial=args.allow_partial)
    if args.output:
        output = args.output.resolve()
        if output.is_relative_to(args.batch.resolve()):
            parser.error("Audit output must be outside the source batch")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(markdown_audit(audit))
        output.with_suffix(".json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    print(json.dumps({name: audit[name] for name in ("batch_status", "provisional", "expected_jobs", "completed_jobs_audited", "violations")}, indent=2))
    return 1 if audit["violations"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
