"""The independent artifact auditor detects cash, physics and oracle-cap corruption."""

import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest
from types import SimpleNamespace

from bess_pv_trading.reporting.mpc_history import archive_completed_history


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("energy_research_audit", ROOT / "scripts/audit_energy_research_results.py")
audit = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(audit)


def fixture_controller(tmp_path):
    """Two lossless intervals provide an independently hand-calculated ledger."""
    (tmp_path / "csv").mkdir()
    (tmp_path / "json").mkdir()
    starts = pd.date_range("2026-01-15T12:00:00Z", periods=2, freq="15min")
    frame = pd.DataFrame({"interval_id": [stamp.isoformat() for stamp in starts],
        "start": starts, "end": starts + pd.Timedelta(minutes=15), "pv_ac_mw": [0.2, 0.4],
        "curtailment_mw": [0, 0], "charge_mw": [0, 0.4], "discharge_mw": [0.4, 0],
        "grid_mw": [0.6, 0], "energy_start_mwh": [0.5, 0.4], "energy_end_mwh": [0.4, 0.5],
        "net_position_mwh": [0.1, 0], "imbalance_mwh": [0.05, 0],
        "imbalance_price_eur_mwh": [100, -100], "imbalance_cashflow_eur": [5, 0],
        "auction_revenue_eur": [5, 0], "degradation_eur": [1, 1], "transaction_fee_eur": [0, 0],
        "import_fee_eur": [0, 0], "net_profit_eur": [9, -1]})
    frame.to_csv(tmp_path / "csv/controller_intervals.csv", index=False)
    summary = {"controller": "ensemble_mpc_price_oracle", "realized_operating_profit_eur": 8,
        "absolute_imbalance_mwh": 0.05}
    ledger = [{"interval_id": product, "quantity_mwh": quantity, "price_eur_per_mwh": 50,
        "trade_id": f"DA:{product}", "submission_time": "2026-01-14T11:00:00Z"}
        for product, quantity in zip(frame.interval_id, [0.1, 0])]
    publications = [{"delivery_day": "2026-01-15", "auction": "DA", "booked_cashflow_eur": 5, "booked_trades": 2}]
    for name, value in (("controller_kpis", summary), ("accepted_trades", ledger), ("price_publications", publications)):
        (tmp_path / "json" / f"{name}.json").write_text(json.dumps(value))
    config = {"battery": {"capacity_mwh": 1, "max_power_mw": 1, "soc_min_percent": 0,
        "soc_max_percent": 100, "converter_charge_efficiency": 1, "converter_discharge_efficiency": 1},
        "plant": {"poi_import_mw": 1, "poi_export_mw": 1}, "solver": {"feasibility_tolerance": 1e-8}}
    return config, frame


def test_valid_interval_physics_and_ledger_reconcile(tmp_path):
    config, _ = fixture_controller(tmp_path)
    result, _ = audit.audit_controller(tmp_path, config)
    assert not result["violations"]
    assert max(result["residuals"].values()) < 1e-12


def test_cash_and_energy_corruption_are_detected_independently(tmp_path):
    config, frame = fixture_controller(tmp_path)
    frame.loc[0, "energy_end_mwh"] += 0.01
    frame.loc[0, "imbalance_cashflow_eur"] += 1
    frame.loc[0, "net_profit_eur"] += 1
    frame.to_csv(tmp_path / "csv/controller_intervals.csv", index=False)
    result, _ = audit.audit_controller(tmp_path, config)
    assert {"energy_transition_mwh", "energy_continuity_mwh", "settlement_eur", "profit_eur"}.issubset(result["violations"])


def test_tiny_oracle_budget_violation_uses_declared_solver_tolerance(tmp_path):
    config, _ = fixture_controller(tmp_path)
    solve = {"interval_id": "2026-01-15T12:00:00Z", "status": "optimal", "price_oracle_enabled": True,
        "oracle_provenance": {"settlement": {"supplied_to_optimizer": False}},
        "oracle_imbalance_cap_mwh": 0.1, "components": {"absolute_imbalance_mwh": 0.10000003},
        "oracle_stage2_used": True}
    (tmp_path / "json/mpc_solves.json").write_text(json.dumps([solve]))
    result, _ = audit.audit_controller(tmp_path, config)
    assert result["oracle_cap_violations"] == 1
    assert result["oracle_cap_details"][0]["allowed_numerical_tolerance_mwh"] == 2e-8
    assert result["oracle_nested_provenance_conflicts"] == 1


def test_audit_refuses_final_report_for_unfinished_batch(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "running", "plan": {"job_count": 1}, "results": []}))
    with pytest.raises(ValueError, match="not complete"):
        audit.audit_batch(tmp_path)
    partial = audit.audit_batch(tmp_path, allow_partial=True)
    assert partial["provisional"]
    assert partial["completed_jobs_audited"] == 0


def test_batch_report_uses_each_jobs_terminal_and_forecast_settings(tmp_path, monkeypatch):
    """The report must not substitute the historical study's assumptions."""
    (tmp_path / "controller").mkdir()
    config, frame = fixture_controller(tmp_path / "controller")
    config.update({
        "state": {"terminal_soc_percent": 35, "terminal_soc_min_percent": 10},
        "realtime_mpc": {"terminal_inventory_mode": "none", "terminal_soc_target_percent": 42,
                         "current_pv_information": "realized_interval_average"},
        "energy_price_forecast": {"auction_enabled": True},
    })
    item = {"status": "complete", "job_id": "case", "farm_id": 1, "window_id": "window",
            "case_id": "case", "controller": "deterministic_mpc", "result": {"output_dir": str(tmp_path)}}
    (tmp_path / "manifest.json").write_text(json.dumps({"status": "complete", "plan": {"job_count": 1}, "results": [item]}))
    (tmp_path / "config__case.json").write_text(json.dumps(config))
    # Both controllers have identical valid data; this test targets reporting, not solver behavior.
    frame["net_position_mwh"] = 0.0
    summary = {"realized_operating_profit_eur": 0, "absolute_imbalance_mwh": 0,
               "auction_revenue_eur": 0, "degradation_eur": 0, "final_energy_mwh": 0.5}
    monkeypatch.setattr(audit, "audit_controller", lambda directory, config: ({"summary": summary, "violations": []}, frame))
    settings = audit.audit_batch(tmp_path)["jobs"][0]["comparison_configuration"]
    assert settings["terminal_soc_percent"] == 35
    assert settings["terminal_soc_min_percent"] == 10
    assert settings["mpc_terminal_inventory_mode"] == "none"
    assert settings["mpc_current_pv_information"] == "realized_interval_average"
    assert settings["auction_price_forecasts_enabled"] is True


@pytest.mark.parametrize("storage", ["csv", "daily_parquet"])
def test_conditional_horizons_stream_and_check_every_scenario(tmp_path, storage):
    config, frame = fixture_controller(tmp_path)
    ledger = json.loads((tmp_path / "json/accepted_trades.json").read_text())
    solve_id = "ensemble_mpc_adaptive:2026-01-15T12:00:00+00:00"
    diagnostic = {"solve_id": solve_id, "scenario_ids": ["a", "b"], "horizon_intervals": 2,
        "cutoff_utc": "2026-01-15T12:00:00Z", "controller": "ensemble_mpc_adaptive"}
    rows = []
    for index, interval in frame.iterrows():
        for scenario in ["a", "b"]:
            rows.append({"delivery_day": "2026-01-15", "solve_id": solve_id,
                "scenario_id": scenario, "weight": 0.5, "control_node": index,
                "pv_utilization_fraction": 1.0, **{name: interval[name] for name in (
                    "interval_id", "pv_ac_mw", "charge_mw", "discharge_mw", "energy_start_mwh",
                    "energy_end_mwh", "grid_mw", "imbalance_mwh")}})
    if storage == "daily_parquet":
        archive_completed_history(tmp_path, SimpleNamespace(plans=[], scenarios=rows), [], ["2026-01-15"])
    else:
        pd.DataFrame(rows).to_csv(tmp_path / "csv/mpc_scenarios.csv", index=False)
    result = audit.audit_conditional_horizons(tmp_path, config, frame, ledger, [diagnostic])
    assert result["horizons_checked"] == 1
    assert result["violations"] == []


def test_conditional_audit_detects_anticipative_root_commands_across_chunk_boundary(tmp_path, monkeypatch):
    config, frame = fixture_controller(tmp_path)
    ledger = json.loads((tmp_path / "json/accepted_trades.json").read_text())
    solve_id = "ensemble_mpc_adaptive:2026-01-15T12:00:00+00:00"
    diagnostic = {"solve_id": solve_id, "scenario_ids": ["a", "b"], "horizon_intervals": 2,
        "cutoff_utc": "2026-01-15T12:00:00Z", "controller": "ensemble_mpc_adaptive"}
    rows = []
    for index, interval in frame.iterrows():
        for scenario in ["a", "b"]:
            rows.append({"solve_id": solve_id, "scenario_id": scenario, "weight": 0.5,
                "control_node": index, "pv_utilization_fraction": 0.5 if index == 0 and scenario == "b" else 1.0,
                **{name: interval[name] for name in ("interval_id", "pv_ac_mw", "charge_mw", "discharge_mw",
                    "energy_start_mwh", "energy_end_mwh", "grid_mw", "imbalance_mwh")}})
    monkeypatch.setattr(audit, "_scenario_batches", lambda directory: iter([pd.DataFrame(rows[:1]), pd.DataFrame(rows[1:])]))
    result = audit.audit_conditional_horizons(tmp_path, config, frame, ledger, [diagnostic])
    assert result["horizons_checked"] == 1
    assert any(row["reason"] == "nonanticipativity_mw" for row in result["violations"])
