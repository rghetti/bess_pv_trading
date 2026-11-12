"""Verify the requested queue and a matched result comparison without solving."""
from pathlib import Path
import json
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
import run_linear_soh_farm_comparison as study
from bess_pv_trading.optimization.degradation import degradation_settings_from_config


def test_intraday_analysis_uses_only_complete_pairs(tmp_path, monkeypatch):
    """Forward both matching cases to the existing intraday reporting workflow."""
    from bess_pv_trading.reporting import k25_intraday_profiles
    captured = []
    def report(batch, rows, *, plots):
        captured.extend(rows)
        assert plots is True
        return batch / "intraday_analysis/report.md"
    monkeypatch.setattr(k25_intraday_profiles, "write_intraday_profiles", report)
    for mode in study.CASES:
        root = tmp_path / "runs/farm_1" / mode
        (root / "json").mkdir(parents=True)
        (root / "json/study_config.json").write_text("{}")
        (root / "strategies/integrated/json").mkdir(parents=True)
        (root / "strategies/integrated/json/kpis.json").write_text("{}")
    results = [{"farm": 1, "mode": mode, "status": "complete"} for mode in study.CASES]
    results.append({"farm": 345, "mode": "linear_soh", "status": "complete"})
    assert study.write_intraday_analysis(tmp_path, results).endswith("intraday_analysis/report.md")
    assert {(row["farm"], row["case"]) for row in captured} == {(1, mode) for mode in study.CASES}


def test_preflight_session_releases_raw_caches_on_exit():
    """Closing preflight clears transient references, including on errors."""
    from bess_pv_trading.simulation.gme_oracle import _ForecastPreflightSession
    class Dataset:
        closed = False
        def close(self):
            self.closed = True
    dataset = Dataset()
    with pytest.raises(RuntimeError):
        with _ForecastPreflightSession({}, None) as session:
            session.dataset = dataset
            session.metadata = object()
            session.vintage_cache["x"] = object()
            session.pv_cache["x"] = object()
            raise RuntimeError("test")
    assert dataset.closed and session.dataset is None and session.metadata is None
    assert not session.vintage_cache and not session.pv_cache


def test_eight_jobs_have_identical_operational_settings_and_current_pack_price(tmp_path):
    """Both cases use the same cycle-count limit and three Gurobi threads."""
    jobs = study.resolve_jobs(tmp_path)
    assert len(jobs) == 8
    for farm in study.FARMS:
        selected = {job["mode"]: job["config"] for job in jobs if job["farm"] == farm}
        assert set(selected) == set(study.CASES)
        left, right = selected["linear_soh"], selected["throughput"]
        assert left["solver"] == right["solver"]
        assert left["solver"]["threads"] == 3
        assert left["battery"] == right["battery"]
        assert left["battery"]["daily_cycle_limit_efc"] == 2
        assert left["execution"] == right["execution"]
        assert left["continuous_optimization"]["planning_modes"] == ["integrated"]
        assert left["continuous_optimization"]["end_date"] == "2026-08-29"
        assert degradation_settings_from_config(left).replacement_cost_eur == pytest.approx(
            degradation_settings_from_config(right).replacement_cost_eur)
        assert left["degradation"]["replacement_economics"]["pack_price_usd_per_kwh"] == 70


def test_comparison_checks_matching_inputs_and_separates_cash_and_wear(tmp_path):
    """Cash never subtracts noncash wear, and differing provenance is rejected."""
    results = []
    provenance = {"days": {"2026-01-01": {"realized_pv_sha256": "pv", "auction_price_sha256": "price",
        "skipped_auctions": [], "price_provenance": {}}}, "settlement_values_sha256": "settlement",
        "forecast_model_artifacts": {"model": "model", "checksums": {}}, "weather_forecasts": []}
    configs = {job["mode"]: job["config"] for job in study.resolve_jobs(tmp_path) if job["farm"] == 1}
    for mode in study.CASES:
        root = tmp_path / "runs/farm_1" / mode
        (root / "json").mkdir(parents=True)
        (root / "json/input_provenance.json").write_text(json.dumps(provenance))
        (root / "json/study_config.json").write_text(json.dumps(configs[mode], default=str))
        output = root / "strategies/integrated/csv"
        output.mkdir(parents=True)
        data = {"interval_id": ["t"], "pv_ac_mw": [1.], "delivery_day": ["2026-01-01"],
            "auction_revenue_eur": [10.], "imbalance_cashflow_eur": [-2.], "import_fee_eur": [1.],
            "transaction_fee_eur": [0.], "degradation_eur": [3.], "throughput_mwh": [.4],
            "charge_mw": [.4], "discharge_mw": [0.], "duration_hours": [1.]}
        if mode == "linear_soh":
            data["battery_soh_start"] = [1.]
            data["battery_soh_end"] = [.99]
        pd.DataFrame(data).to_csv(output / "realized_dispatch.csv", index=False)
        pd.DataFrame({"interval_id": ["t", "t"], "auction": ["DA", "IDA1"],
                      "cashflow_eur": [20., -10.], "purchased_mwh": [0., .1],
                      "sold_mwh": [.2, 0.], "transaction_fee_eur": [0., 0.]}).to_csv(
                          output / "accepted_trades.csv", index=False)
        results.append({"farm": 1, "mode": mode, "status": "complete"})
    output, count = study.paired_summary(tmp_path, results)
    result = pd.read_csv(output)
    assert count == 2
    assert json.loads((tmp_path / "summary.json").read_text())["status"] == "partial"
    markdown = (tmp_path / "comparison.md").read_text()
    assert "Matched battery-aging comparison" in markdown
    assert "Excluded incomplete or unmatched farms" in markdown
    assert "Booked auction cash flows" in markdown
    auctions = pd.read_csv(tmp_path / "auction_comparison.csv")
    assert auctions[(auctions["mode"] == "linear_soh") & (auctions.auction == "IDA1")].cashflow_eur.iloc[0] == -10.
    assert (result.market_cash_eur == 7.).all()
    assert (result.cash_minus_model_wear_eur == 4.).all()
    assert result.loc[result["mode"] == "linear_soh", "final_soh"].iloc[0] == pytest.approx(.99)
    expected_internal_throughput = .4 * .9486832980505138
    expected_calendar_loss = .01 * 3600 / 31557600
    expected_cycle_loss = .17 * expected_internal_throughput / (2 * 6000 * .75)
    expected_total_loss = expected_calendar_loss + expected_cycle_loss
    expected_wear = (70 / 1.1637 * .75 * 1000 / .2) * expected_total_loss
    assert result.common_proxy_internal_throughput_mwh.tolist() == pytest.approx([expected_internal_throughput] * 2)
    assert result.common_proxy_total_soh_loss_fraction.tolist() == pytest.approx([expected_total_loss] * 2)
    assert result.common_proxy_noncash_wear_eur.tolist() == pytest.approx([expected_wear] * 2)
    assert result.market_cash_minus_common_proxy_wear_eur.tolist() == pytest.approx([7 - expected_wear] * 2)
    for name in ("market_cash_eur.png", "market_cash_minus_common_proxy_wear_eur.png", "throughput_mwh.png", "daily_soh.png"):
        assert (tmp_path / "figures" / name).stat().st_size > 0
    provenance["settlement_values_sha256"] = "changed"
    (tmp_path / "runs/farm_1/throughput/json/input_provenance.json").write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="imbalance prices"):
        study.paired_summary(tmp_path, results)


def test_resume_preserves_completed_and_recovers_worker_completion(tmp_path, monkeypatch):
    """Only unfinished jobs resume; completed run identities must match."""
    import bess_pv_trading.simulation.continuous_inputs as inputs

    monkeypatch.setattr(inputs, "_replacement_evidence", lambda config: {"checksums": {"model.pt": "abc"}})
    jobs = study.resolve_jobs(tmp_path)
    fingerprints = {study.job_key(job): study._configuration_fingerprint(job["config"]) for job in jobs}
    first, second = jobs[:2]
    for job in (first, second):
        path = tmp_path / "runs" / f"farm_{job['farm']}" / job["mode"] / "json"
        path.mkdir(parents=True)
        (path / "manifest.json").write_text(json.dumps({"status": "complete",
            "config_fingerprint": fingerprints[study.job_key(job)]}))
        (path / "input_provenance.json").write_text(json.dumps({
            "forecast_model_artifacts": {"checksums": {"model.pt": "abc"}}}))
    batch = {"job_fingerprints": fingerprints,
             "results": [{"farm": first["farm"], "mode": first["mode"], "status": "complete"}]}
    pending, completed, recovered = study.resume_jobs(tmp_path, jobs, batch)
    assert len(pending) == 6
    assert len(completed) == 2
    assert recovered[0]["farm"] == second["farm"]
    monkeypatch.setattr(inputs, "_replacement_evidence", lambda config: {"checksums": {"model.pt": "changed"}})
    with pytest.raises(ValueError, match="forecast model differs"):
        study.resume_jobs(tmp_path, jobs, batch)
    batch["job_fingerprints"][study.job_key(first)] = "changed"
    with pytest.raises(ValueError, match="configuration has changed"):
        study.resume_jobs(tmp_path, jobs, batch)


def test_common_proxy_uses_internal_energy_for_charge_and_discharge(tmp_path):
    """Revaluation applies efficiency in opposite directions to executed AC flows."""
    from bess_pv_trading.domain import battery_parameters_from_config

    config = study.resolve_jobs(tmp_path)[0]["config"]
    battery = battery_parameters_from_config(config["battery"])
    settings = degradation_settings_from_config(config)
    frame = pd.DataFrame({"charge_mw": [.4, 0.], "discharge_mw": [0., .3],
                          "duration_hours": [.5, 2.]})
    result = study.common_linear_wear_proxy(frame, battery, settings)
    expected = .4 * .5 * battery.eta_charge + .3 * 2. / battery.eta_discharge
    assert result["common_proxy_internal_throughput_mwh"] == pytest.approx(expected)
    assert result["common_proxy_cycle_loss_fraction"] == pytest.approx(.17 * expected / (2 * 6000 * .75))


def test_job_rejects_forecast_bundle_changed_during_execution(tmp_path, monkeypatch):
    """A completed trajectory cannot be reported if its model files changed."""
    import bess_pv_trading.simulation.continuous as continuous
    import bess_pv_trading.simulation.continuous_inputs as inputs

    (tmp_path / "logs").mkdir()
    job = study.resolve_jobs(tmp_path)[0]
    states = iter([{"status": "pinned", "checksums": {"model.pt": "old"}},
                   {"status": "pinned", "checksums": {"model.pt": "new"}}])
    monkeypatch.setattr(inputs, "_replacement_evidence", lambda config: next(states))
    monkeypatch.setattr(continuous, "run_continuous_experiment", lambda config, progress: {})
    result = study.run_job(job, tmp_path)
    assert result["status"] == "failed"
    assert "model files changed" in result["error"]


def test_resume_preserves_explicit_end_of_life_stop(tmp_path):
    """An EOL stop is terminal and cannot be silently restarted."""
    jobs = study.resolve_jobs(tmp_path)
    fingerprints = {study.job_key(job): study._configuration_fingerprint(job["config"]) for job in jobs}
    job = jobs[0]
    folder = tmp_path / "runs" / f"farm_{job['farm']}" / job["mode"] / "json"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"status": "stopped_at_battery_end_of_life",
        "config_fingerprint": fingerprints[study.job_key(job)]}))
    batch = {"job_fingerprints": fingerprints,
             "results": [{"farm": job["farm"], "mode": job["mode"], "status": "stopped_at_battery_end_of_life"}]}
    pending, finished, _ = study.resume_jobs(tmp_path, jobs, batch)
    assert len(pending) == 7
    assert finished[(job["farm"], job["mode"])] == "stopped_at_battery_end_of_life"
