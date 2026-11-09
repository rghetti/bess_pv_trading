"""Verify the combined K25 summary uses reconciled completed trajectories."""

import json
from pathlib import Path

import pandas as pd
import pytest

from bess_pv_trading.reporting.k25_batch_summary import write_k25_batch_summary


def _case(batch, case, controller, *, reuse=False):
    """Write a small complete trajectory with explicit daily and cash totals."""
    job_id = f"window__farm_1__{case}"
    root = batch / "runs" / "farm_1" / job_id / "continuous"
    root.mkdir(parents=True)
    config = {"battery": {"soc_min_percent": 7, "capacity_mwh": 1.0}, "state": {"terminal_soc_percent": 50},
        "market": {"timezone": "Europe/Rome"},
        "realtime_mpc_comparison": {"controllers": [controller]},
        "realtime_mpc": {"first_interval_priority": "weighted", "first_interval_weight": 25}}
    (root / "resolved_config.json").write_text(json.dumps(config))
    job = {"job_id": job_id, "farm_id": "1", "case_id": case, "controllers": [controller]}
    result = {"job_id": job_id, "controller": controller, "status": "complete"}
    if reuse:
        result["reused_from_job_id"] = "source"
        return job, result
    daily = [{"delivery_day": day, "net_profit_eur": 9, "absolute_imbalance_mwh": 1,
              "pv_realized_energy_mwh": 10, "curtailment_mwh": 2, "final_soc_percent": soc}
             for day, soc in (("2026-01-01", 50), ("2026-01-02", 45))]
    components = {"auction_revenue_eur": 20, "imbalance_cashflow_eur": -1,
        "degradation_eur": 1, "import_fee_eur": 0, "transaction_fee_eur": 0}
    metrics = {"positive_imbalance_mwh": 0.5, "negative_imbalance_mwh": 1.5,
        "positive_deviation_cashflow_eur": 0.5, "negative_deviation_cashflow_eur": -1.5,
        "absolute_settlement_exposure_eur": 2, "mpc_solves": 2, "mpc_fallbacks": 1}
    summary = {"daily_kpis": daily, "components": components, "mpc_metrics": metrics, "input_fingerprint": "same",
        "net_profit_eur": 18, "absolute_imbalance_mwh": 2, "signed_imbalance_mwh": -1,
        "pv_realized_energy_mwh": 20, "curtailment_mwh": 4, "charge_mwh": 2,
        "discharge_mwh": 1, "final_soc_percent": 45, "fallback_count": 0,
        "first_delivery_day": "2026-01-01", "last_delivery_day": "2026-01-02", "delivery_days": 2}
    (root / f"summary__{controller}.json").write_text(json.dumps(summary))
    csv_dir = root / "controllers" / controller / "csv"
    csv_dir.mkdir(parents=True)
    pd.DataFrame([{"auction": "DA", "cashflow_eur": 10, "interval_id": "first", "quantity_mwh": 0.2, "price_eur_per_mwh": 50},
                  {"auction": "DA", "cashflow_eur": 10, "interval_id": "second", "quantity_mwh": -0.1, "price_eur_per_mwh": 60}]
        ).to_csv(csv_dir / "accepted_trades.csv", index=False)
    pd.DataFrame([{"delivery_day": "2026-01-01", "mpc_fallbacks": 0},
                  {"delivery_day": "2026-01-02", "mpc_fallbacks": 1}]).to_csv(
        csv_dir / "controller_daily_kpis.csv", index=False)
    pd.DataFrame({"status": ["optimal", "imbalance_tiebreak_rejected"]}).to_csv(csv_dir / "mpc_solves.csv", index=False)
    pd.DataFrame([{"interval_id": "first", "start": "2025-12-31T23:00:00Z", "delivery_day": "2026-01-01",
                   "duration_hours": 0.25, "curtailment_mw": 8, "imbalance_mwh": -1, "net_position_mwh": 0.2,
                   "grid_mw": -3.2, "charge_mw": 1, "discharge_mw": 0, "energy_start_mwh": 0.5, "imbalance_price_eur_mwh": 40},
                  {"interval_id": "second", "start": "2026-01-01T23:00:00Z", "delivery_day": "2026-01-02",
                   "duration_hours": 0.25, "curtailment_mw": 8, "imbalance_mwh": 1, "net_position_mwh": -0.1,
                   "grid_mw": 3.6, "charge_mw": 0, "discharge_mw": 1, "energy_start_mwh": 0.45, "imbalance_price_eur_mwh": 45}]
        ).to_csv(csv_dir / "realized_dispatch.csv", index=False)
    return job, result


def test_combined_summary_reports_percentages_fallback_and_daily_target(tmp_path):
    """The batch summary makes missed SOC targets and failed refinements visible."""
    job, result = _case(tmp_path, "deterministic_weighted_k25", "deterministic_mpc")
    reused_job, reused_result = _case(tmp_path, "reused", "track_committed_grid", reuse=True)
    path = write_k25_batch_summary(tmp_path, [job, reused_job], [result, reused_result], plots=False)
    text = path.read_text()
    assert "MPC K=25" in text
    assert "10.00%" in text
    assert "20.00%" in text
    assert "1/2" in text
    assert "| 45.00% | 1 | 1/1 | 1 | 0 |" in text
    assert "Grid tracker" not in text


def test_summary_rejects_nonreconciling_auction_cash(tmp_path):
    """A misleading report is not written if ledger and summary disagree."""
    job, result = _case(tmp_path, "deterministic_weighted_k25", "deterministic_mpc")
    ledger = (tmp_path / "runs/farm_1" / job["job_id"] /
              "continuous/controllers/deterministic_mpc/csv/accepted_trades.csv")
    pd.DataFrame([{"auction": "DA", "cashflow_eur": 19}]).to_csv(ledger, index=False)
    with pytest.raises(ValueError, match="ledger does not reconcile"):
        write_k25_batch_summary(tmp_path, [job], [result], plots=False)
    assert not (tmp_path / "summary.md").exists()


def test_intraday_profiles_reject_mismatched_net_commitments(tmp_path):
    """A dispatch position must equal purchases plus sales in the executed ledger."""
    job, result = _case(tmp_path, "deterministic_weighted_k25", "deterministic_mpc")
    ledger = (tmp_path / "runs/farm_1" / job["job_id"] /
              "continuous/controllers/deterministic_mpc/csv/accepted_trades.csv")
    trades = pd.read_csv(ledger)
    trades.loc[0, "quantity_mwh"] = 0.7
    trades.to_csv(ledger, index=False)
    with pytest.raises(ValueError, match="Net commitments do not reconcile"):
        write_k25_batch_summary(tmp_path, [job], [result], plots=False)


def test_comparison_requires_matching_input_fingerprints(tmp_path):
    """Case differences cannot silently compare unmatched prepared inputs."""
    base, base_result = _case(tmp_path, "deterministic_original", "deterministic_mpc")
    case, case_result = _case(tmp_path, "deterministic_weighted_k25", "deterministic_mpc")
    case["baseline_case"] = "deterministic_original"
    path = write_k25_batch_summary(tmp_path, [base, case], [base_result, case_result], plots=False)
    assert "## Controller case differences" in path.read_text()
    path.unlink()
    summary_path = (tmp_path / "runs/farm_1" / case["job_id"] /
                    "continuous/summary__deterministic_mpc.json")
    summary = json.loads(summary_path.read_text())
    summary["input_fingerprint"] = "different"
    summary_path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="inputs or delivery dates differ"):
        write_k25_batch_summary(tmp_path, [base, case], [base_result, case_result], plots=False)
    assert not path.exists()
