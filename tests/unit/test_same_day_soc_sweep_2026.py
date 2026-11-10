"""Verify the 2026 same-day SOC sweep without starting an experiment."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from bess_pv_trading.reporting import same_day_soc_sweep as report


@pytest.fixture
def launcher():
    """Load the standalone launcher for plan and cache-isolation checks."""
    path = Path(__file__).resolve().parents[2] / "scripts/run_same_day_soc_sweep_2026.py"
    spec = importlib.util.spec_from_file_location("same_day_soc_sweep_launcher", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_sets_same_day_horizon_and_eight_workers(launcher, capsys):
    """Check the actual configured grid, dates and resources before execution."""
    assert launcher.main(["--run-id", "same_day_unit_plan"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["delivery_period"] == ["2026-01-01", "2026-08-29"]
    assert plan["soc_targets_percent"] == [7, *range(10, 91, 5), 93]
    assert plan["trajectory_count"] == 76
    assert plan["workers"] == 8
    assert plan["solver_threads_per_job"] == 2
    assert plan["maximum_solver_threads"] == 16
    assert plan["auction_submissions_upper_bound"] == 76 * 241 * 4
    assert plan["solves_started"] == 0
    assert not Path(plan["output_dir"]).exists()
    jobs = launcher.make_jobs(["1"], [7, 93], Path(plan["output_dir"]), 2)
    for job in jobs:
        settings = job["config"]["continuous_optimization"]
        assert settings["ida3_next_day_horizon"] is False
        assert settings["pin_tracker_forecast_replacement"] is True
        assert settings["terminal_soc_scope"] == "each_delivery_day"
        assert job["config"]["state"]["terminal_soc_percent"] == job["target_soc_percent"]


def test_prepared_inputs_isolate_target_and_do_not_change_shared_record(launcher, monkeypatch, tmp_path):
    """Each run gets its own target while reusing a canonical farm input snapshot."""
    canonical = {"plant": SimpleNamespace(battery=SimpleNamespace(capacity_mwh=2.0)),
                 "terminal_energy_mwh": 1.0, "provenance": {
                     "asset_input_settings": {"state": {"terminal_soc_percent": 50}}, "weather": "same"}}
    seen = []
    def shared(config, batch, prepare, progress):
        seen.append(config["state"]["terminal_soc_percent"])
        return deepcopy(canonical)

    monkeypatch.setattr(launcher, "prepare_shared", shared)
    monkeypatch.setattr(launcher, "preparation_key", lambda config: "same")
    jobs = launcher.make_jobs(["1"], [7, 93], tmp_path, 2)
    low, key_low = launcher.prepare_for_target(jobs[0]["config"], tmp_path, None)
    high, key_high = launcher.prepare_for_target(jobs[1]["config"], tmp_path, None)
    assert seen == [50.0, 50.0]
    assert key_low == key_high == "same"
    assert low["terminal_energy_mwh"] == pytest.approx(0.14)
    assert high["terminal_energy_mwh"] == pytest.approx(1.86)
    assert low["input_fingerprint"] != high["input_fingerprint"]
    assert canonical["provenance"]["asset_input_settings"]["state"]["terminal_soc_percent"] == 50


def test_report_uses_full_period_and_labels_partial_results(tmp_path, monkeypatch):
    """The report validates all 241 delivery days and labels incomplete targets."""
    values = {"net_profit_eur": 100.0, "absolute_imbalance_mwh": 2.0,
        "pv_realized_energy_mwh": 100.0, "signed_imbalance_mwh": -2.0,
        "curtailment_mwh": 3.0, "charge_mwh": 4.0, "discharge_mwh": 3.0,
        "auction_revenue_eur": 120.0, "imbalance_cashflow_eur": -10.0,
        "degradation_eur": 10.0, "import_fee_eur": 0.0, "transaction_fee_eur": 0.0,
        "final_soc_percent": 30.0, "mean_midnight_soc_percent": 40.0,
        "auction_fallbacks": 2}
    daily = pd.DataFrame({"initial_soc_percent": [50.0] * 241, "final_soc_percent": [30.0] * 241})
    monkeypatch.setattr(report, "_daily", lambda root, start, end: daily)
    monkeypatch.setattr(report, "_metrics", lambda root, values: dict(values_fixture))
    values_fixture = dict(values)
    monkeypatch.setattr(report, "_plot_farm", lambda frame, farm, path: path.write_bytes(b"fixture"))
    directory, root = report._trajectory_root(tmp_path, "1", 10)
    root.mkdir(parents=True)
    config = {"state": {"terminal_soc_percent": 10, "terminal_soc_min_percent": None},
              "execution": {"dispatch_policy": "track_committed_grid"},
              "continuous_optimization": {"ida3_next_day_horizon": False,
                  "start_date": "2026-01-01", "end_date": "2026-08-29"}}
    (directory / "resolved_config.json").write_text(json.dumps(config))
    summary = {**values, "first_delivery_day": "2026-01-01", "last_delivery_day": "2026-08-29",
               "delivery_days": 241, "components": {key: values[key] for key in (
                   "auction_revenue_eur", "imbalance_cashflow_eur", "degradation_eur", "import_fee_eur",
                   "transaction_fee_eur")}}
    (directory / "summary.json").write_text(json.dumps(summary))
    result = [{"farm": "1", "target_soc_percent": 10, "status": "complete"}]
    path = report.write_soc_sweep_report(tmp_path, result, ["1"], [10, 20])
    assert "2026-01-01 to 2026-08-29" in path.read_text()
    assert "Completed 1 of 2" in path.read_text()
    assert "Farm 1, 20%: not started" in path.read_text()
    assert "/241" in (tmp_path / "farm_1_summary.md").read_text()


def test_failed_preflight_prevents_any_optimization(launcher, monkeypatch, tmp_path):
    """A missing farm input blocks every trajectory before the sweep starts."""
    from concurrent.futures import Future

    submitted = []
    class Pool:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def submit(self, function, job, batch, **kwargs):
            submitted.append(kwargs.get("check_only", False))
            future = Future()
            future.set_result({"farm": job["farm"], "target_soc_percent": job["target_soc_percent"],
                               "status": "failed", "error": "missing weather coverage"})
            return future

    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    monkeypatch.setattr(launcher, "source_checksums", lambda: {})
    monkeypatch.setattr(launcher, "_artifact_requirements", lambda config: [])
    monkeypatch.setattr(launcher, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *args, **kwargs: "revision\n")
    assert launcher.main(["--run-id", "failed_coverage", "--farm-ids", "1", "--run"]) == 1
    assert submitted == [True]
    manifest = json.loads((tmp_path / "outputs/energy_research/failed_coverage/manifest.json").read_text())
    assert manifest["status"] == "failed_preflight"
    assert manifest["results"] == []


def test_product_input_checks_are_not_fixed_to_96_intervals(tmp_path):
    """Compare matching products through DST without assuming fixed local-day length."""
    roots = [tmp_path / "old", tmp_path / "new"]
    for root in roots:
        (root / "csv").mkdir(parents=True)
        pd.DataFrame({"interval_id": range(92), "pv_ac_mw": [0.0] * 92,
                      "imbalance_price_eur_mwh": [40.0] * 92}).to_csv(
                          root / "csv/realized_dispatch.csv", index=False)
        pd.DataFrame({"auction": ["DA"] * 92, "interval_id": range(92),
                      "price_eur_per_mwh": [45.0] * 92}).to_csv(
                          root / "csv/accepted_trades.csv", index=False)
    assert report._check_common_inputs(*roots) == {
        "realized_pv_and_settlement_products": 92, "auction_products": 92}
    dispatch = pd.read_csv(roots[1] / "csv/realized_dispatch.csv")
    dispatch.loc[0, "pv_ac_mw"] = 1.0
    dispatch.to_csv(roots[1] / "csv/realized_dispatch.csv", index=False)
    with pytest.raises(ValueError, match="Common PV"):
        report._check_common_inputs(*roots)
