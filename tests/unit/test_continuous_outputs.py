"""Continuous reports keep midnight state, cash attribution and solver timing."""
from copy import deepcopy
import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.reporting.continuous_outputs import (
    _projection_plot_rows, save_continuous_comparison, save_continuous_projection_plots,
    save_continuous_strategy_outputs,
)


@pytest.fixture
def report_inputs():
    """Use shared YAML assets and existing fixture clocks/runtimes for sample inputs."""
    config = load_config("tests/fixtures/configs/optimization_timing_cases.yaml")
    config["experiment_outputs"]["plots"] = False
    case = config["timing_cases"]
    timezone = config["market"]["timezone"]
    start = pd.Timestamp(case["days"][0], tz=timezone)
    stop = pd.Timestamp(case["days"][-1], tz=timezone) + pd.Timedelta(days=1)
    step = pd.Timedelta(minutes=config["gme_oracle_experiment"]["decision_resolution_minutes"])
    starts = pd.date_range(start, stop, freq=step, inclusive="left").tz_convert("UTC")
    capacity = config["battery"]["capacity_mwh"]
    initial = capacity * config["state"]["initial_soc_percent"] / 100
    increments = capacity * (config["battery"]["soc_max_percent"]
        - config["state"]["initial_soc_percent"]) / 100 / len(starts)
    boundaries = initial + np.arange(len(starts) + 1) * increments
    pv = config["battery"]["max_power_mw"]
    duration = step / pd.Timedelta(hours=1)
    frame = pd.DataFrame({"interval_id": starts.astype(str), "start": starts,
        "end": starts + step, "delivery_day": starts.tz_convert(timezone).strftime("%Y-%m-%d"),
        "energy_start_mwh": boundaries[:-1], "energy_end_mwh": boundaries[1:],
        "pv_ac_mw": pv, "pv_ac_forecast_mw": pv,
        "charge_mw": 0.0, "discharge_mw": 0.0, "curtailment_mw": 0.0,
        "grid_mw": pv, "net_position_mwh": pv * duration, "imbalance_mwh": 0.0,
        "imbalance_cashflow_eur": case["solves"][0]["build_seconds"],
        "degradation_eur": case["solves"][1]["build_seconds"],
        "import_fee_eur": 0.0, "transaction_fee_eur": case["solves"][0]["build_seconds"],
        "auction_revenue_eur": case["solves"][0]["runtime_seconds"]})
    solves = [{**row, "delivery_day": day, "current_auction": row["auction"],
        "fallback_applied": row["status"] == "infeasible"}
        for day, row in zip(case["days"], case["solves"])]
    ledger = frame[["interval_id", "delivery_day", "transaction_fee_eur"]].assign(
        cashflow_eur=frame.auction_revenue_eur,
        submission_time=start - pd.Timedelta(days=1)).to_dict("records")
    return config, frame, solves, ledger


def test_continuous_report_attributes_pre_delivery_trades_and_carries_inventory(report_inputs, tmp_path):
    config, frame, solves, ledger = report_inputs
    expected = {day: row["runtime_seconds"] for day, row in
        zip(config["timing_cases"]["days"], config["timing_cases"]["solves"])}
    projections = [{"delivery_day": frame.delivery_day.iloc[0],
        "predicted_initial_energy_mwh": frame.energy_start_mwh.iloc[0],
        "actual_initial_energy_mwh": frame.energy_start_mwh.iloc[0], "prediction_error_mwh": 0.0}]
    summary = save_continuous_strategy_outputs(config, tmp_path, "integrated", frame,
        pd.DataFrame(), ledger, solves, projections,
        {"initial_expected_profit_by_day": expected})
    daily = pd.read_csv(tmp_path / "csv/daily_kpis.csv")
    profit = (frame.auction_revenue_eur + frame.imbalance_cashflow_eur
        - frame.degradation_eur - frame.transaction_fee_eur).sum()
    assert summary["total_gain_eur"] == pytest.approx(profit)
    assert daily.net_profit_eur.sum() == pytest.approx(profit)
    assert daily.initial_energy_mwh.iloc[-1] == pytest.approx(daily.final_energy_mwh.iloc[0])
    assert summary["final_energy_mwh"] == pytest.approx(frame.energy_end_mwh.iloc[-1])
    np.testing.assert_allclose(daily.realized_minus_expected_profit_eur,
        daily.net_profit_eur - daily.delivery_day.map(expected))
    timing = json.loads((tmp_path / "json/optimization_timing.json").read_text())
    samples = [row["runtime_seconds"] for row in solves]
    assert timing["gurobi_runtime_seconds"]["mean"] == pytest.approx(np.mean(samples))
    assert timing["daily_total_runtime_seconds"]["std"] == pytest.approx(np.std(samples))
    assert summary["fallback_count"] == sum(row["fallback_applied"] for row in solves)
    assert json.loads((tmp_path / "json/state_projections.json").read_text()) == projections
    allocation = pd.read_csv(tmp_path / "csv/pv_power_allocation.csv")
    np.testing.assert_allclose(allocation.pv_to_grid_mw + allocation.pv_to_battery_mw
        + allocation.pv_curtailed_mw, allocation.pv_ac_mw)
    assert not (tmp_path / "plots").exists()
    assert pd.read_csv(tmp_path / "csv/expected_dispatch_plans.csv").empty


def test_continuous_comparison_pairs_days_and_keeps_reference_timings_separate(report_inputs, tmp_path):
    config, frame, solves, _ = report_inputs
    modes = config["timing_cases"]["mode_runtimes"]
    ratio = sum(modes["current_auction_only"]) / sum(modes["integrated"])
    summaries = {}
    for strategy in ("integrated", "current_auction_only", "perfect_pv"):
        execution = frame.copy()
        if strategy == "current_auction_only":
            execution["auction_revenue_eur"] *= ratio
        summaries[strategy] = save_continuous_strategy_outputs(config, tmp_path / strategy,
            strategy, execution, pd.DataFrame(), [], solves, [])
        assert pd.read_csv(tmp_path / strategy / "csv/accepted_trades.csv").empty
        assert pd.read_csv(tmp_path / strategy / "csv/state_projections.csv").empty
    report = save_continuous_comparison(config, tmp_path / "comparison", summaries)
    assert report["same_horizon_perfect_pv_profit_eur"] == summaries["perfect_pv"]["total_gain_eur"]
    assert report["common_prices_weather_settlement_and_assets_verified"] is False
    comparison = pd.read_csv(tmp_path / "comparison/csv/strategy_comparison.csv").set_index("strategy")
    expected = summaries["current_auction_only"]["total_gain_eur"] - summaries["integrated"]["total_gain_eur"]
    assert comparison.loc["current_auction_only", "profit_minus_integrated_eur"] == pytest.approx(expected)
    differences = pd.read_csv(tmp_path / "comparison/csv/daily_profit_differences.csv")
    assert differences.loc[differences.strategy.eq("current_auction_only"), "profit_minus_integrated_eur"].sum() == pytest.approx(expected)
    timing = json.loads((tmp_path / "comparison/json/optimization_timing.json").read_text())
    assert set(timing["by_category"]) == {"replay", "perfect_pv_reference"}
    assert timing["by_category"]["perfect_pv_reference"]["solve_count"] == len(solves)
    sequential_only = {key: summaries[key] for key in ("current_auction_only", "perfect_pv")}
    single = save_continuous_comparison(config, tmp_path / "single_mode", sequential_only)
    assert single["reference_strategy"] == "current_auction_only"
    assert "profit_minus_integrated_eur" not in single["daily"][0]
    assert single["strategies"][0]["profit_minus_reference_eur"] == 0
    verified = deepcopy(summaries)
    fingerprint = config["timing_cases"]["profiles"][0]
    for summary in verified.values():
        summary["input_fingerprint"] = fingerprint
    matched = save_continuous_comparison(config, tmp_path / "verified", verified)
    assert matched["common_prices_weather_settlement_and_assets_verified"] is True
    assert matched["common_input_fingerprint"] == fingerprint
    verified["perfect_pv"].pop("input_fingerprint")
    with pytest.raises(ValueError, match="identical complete input fingerprints"):
        save_continuous_comparison(config, tmp_path / "partial_fingerprints", verified)
    verified["perfect_pv"]["input_fingerprint"] = config["timing_cases"]["profiles"][-1]
    with pytest.raises(ValueError, match="identical complete input fingerprints"):
        save_continuous_comparison(config, tmp_path / "mismatched_fingerprints", verified)
    damaged = deepcopy(summaries)
    damaged["current_auction_only"]["daily_kpis"] = damaged["current_auction_only"]["daily_kpis"][:-1]
    with pytest.raises(ValueError, match="identical delivery dates"):
        save_continuous_comparison(config, tmp_path / "mismatched", damaged)
    path = tmp_path / "current_auction_only/csv/realized_dispatch.csv"
    changed = pd.read_csv(path)
    changed["pv_ac_mw"] *= ratio
    changed.to_csv(path, index=False)
    with pytest.raises(ValueError, match="same delivery grid and realized PV"):
        save_continuous_comparison(config, tmp_path / "wrong_pv", summaries)


def test_continuous_reporting_rejects_missing_settlement_and_state_resets(report_inputs, tmp_path):
    config, frame, solves, ledger = report_inputs
    missing = frame.copy()
    missing.loc[missing.index[0], "imbalance_cashflow_eur"] = np.nan
    with pytest.raises(ValueError, match="finite execution and settlement"):
        save_continuous_strategy_outputs(config, tmp_path / "missing", "integrated", missing,
            pd.DataFrame(), [], solves, [])
    reset = frame.copy()
    boundary = reset.index[reset.delivery_day.ne(reset.delivery_day.iloc[0])][0]
    reset.loc[boundary, "energy_start_mwh"] = config["battery"]["capacity_mwh"]
    with pytest.raises(ValueError, match="carry across every interval and midnight"):
        save_continuous_strategy_outputs(config, tmp_path / "reset", "integrated", reset,
            pd.DataFrame(), [], solves, [])
    bad_ledger = deepcopy(ledger)
    bad_ledger.pop()
    with pytest.raises(ValueError, match="reconcile"):
        save_continuous_strategy_outputs(config, tmp_path / "ledger", "integrated", frame,
            pd.DataFrame(), bad_ledger, solves, [])
    shifted_ledger = deepcopy(ledger)
    shifted_ledger[0]["delivery_day"] = frame.delivery_day.iloc[-1]
    with pytest.raises(ValueError, match="reconcile by delivery day"):
        save_continuous_strategy_outputs(config, tmp_path / "shifted", "integrated", frame,
            pd.DataFrame(), shifted_ledger, solves, [])


def test_resume_preserves_discarded_solver_attempts_without_counting_them_as_decisions(report_inputs, tmp_path):
    config, frame, solves, _ = report_inputs
    discarded = {**solves[-1], "rolled_back_for_resume": True,
        "projection_records": [{"input": config["timing_cases"]["solves"]}]}
    attempts = [solves[0], discarded, solves[-1]]
    summary = save_continuous_strategy_outputs(config, tmp_path, "integrated", frame,
        pd.DataFrame(), [], solves, [], {"optimization_attempts": attempts})
    assert summary["auction_solves"] == len(solves)
    assert summary["optimization_timing"]["solve_count"] == len(attempts)
    assert summary["optimization_timing"]["gurobi_runtime_seconds"]["total"] == pytest.approx(
        sum(row["runtime_seconds"] for row in attempts))
    assert "optimization_attempts" not in summary
    assert json.loads((tmp_path / "json/solves.json").read_text()) == solves
    assert json.loads((tmp_path / "json/optimization_attempts.json").read_text()) == attempts
    assert sum(row["rolled_back_for_resume"] for row in
        summary["optimization_timing"]["solves"]) == sum(bool(row.get("rolled_back_for_resume"))
            for row in attempts)


def test_continuous_plots_use_shared_market_time_and_save_daily_views(report_inputs, tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    config, frame, solves, _ = report_inputs
    config["experiment_outputs"]["plots"] = True
    frame["discharge_mw"] = np.finfo(float).eps * config["battery"]["max_power_mw"]
    frame["requested_charge_mw"] = frame["requested_discharge_mw"] = 0.0
    daily_axes = []
    original_save = Figure.savefig

    def capture_daily(figure, filename, *args, **kwargs):
        if filename.parent.name == "daily":
            daily_axes.append((figure.axes[1].get_ylim(), figure.axes[1].get_legend_handles_labels()[1]))
        return original_save(figure, filename, *args, **kwargs)

    monkeypatch.setattr(Figure, "savefig", capture_daily)
    projections = []
    for day, daily in frame.groupby("delivery_day", sort=True):
        noon = pd.Timestamp(f"{day} {config['market']['delivery_start_times']['IDA3']}",
            tz=config["market"]["timezone"]).tz_convert("UTC")
        for auction, kind, boundary in (("DA", "delivery_midnight", daily.start.iloc[0]),
                ("IDA3", "ida3_noon", noon)):
            actual = float(daily.loc[daily.start.eq(boundary), "energy_start_mwh"].iloc[0])
            projections.append({"delivery_day": day, "auction": auction, "boundary_kind": kind,
                "planning_start_utc": boundary.isoformat(),
                "predicted_initial_energy_mwh": actual - config["timing_cases"]["solves"][0]["build_seconds"],
                "actual_initial_energy_mwh": actual})
    summary = save_continuous_strategy_outputs(config, tmp_path, "integrated", frame,
        pd.DataFrame(), [], solves, projections)
    for name in ("continuous_soc.png", "continuous_dispatch.png", "cumulative_profit.png"):
        assert (tmp_path / "plots" / name).stat().st_size > 0
    assert sorted(path.stem for path in (tmp_path / "plots/daily").glob("*.png")) == config["timing_cases"]["days"]
    for limits, labels in daily_axes:
        assert limits == (-config["battery"]["max_power_mw"], config["battery"]["max_power_mw"])
        assert "Scheduled battery (active plan)" in labels
    midnight = _projection_plot_rows(config, projections, "delivery_midnight")
    noon = _projection_plot_rows(config, projections, "ida3_noon")
    assert midnight.auction.eq("DA").all()
    assert noon.auction.eq("IDA3").all()
    assert len(midnight) == len(config["timing_cases"]["days"]) == len(noon)
    np.testing.assert_allclose(midnight.prediction_error_mwh,
        config["timing_cases"]["solves"][0]["build_seconds"])
    missing = {**projections[0], "actual_initial_energy_mwh": None}
    assert _projection_plot_rows(config, [missing], "delivery_midnight").empty
    save_continuous_comparison(config, tmp_path / "comparison", [summary])
    plot_names = ("midnight_soc_prediction_errors", "ida3_noon_soc_prediction_errors")
    for name in plot_names:
        assert (tmp_path / "plots" / f"{name}.png").stat().st_size > 0
        assert (tmp_path / "comparison/plots" / f"{name}_integrated.png").stat().st_size > 0
        assert not (tmp_path / "comparison/plots" / f"{name}.png").exists()
        assert not (tmp_path / "comparison/plots" / f"{name}_perfect_pv.png").exists()
    # A partial refresh preserves superseded combined figures until all
    # actually solved strategies have rendered their own replacements.
    from bess_pv_trading.reporting import continuous_outputs
    comparison_root = tmp_path / "comparison"
    for name in plot_names:
        (comparison_root / "plots" / f"{name}.png").write_bytes(
            (tmp_path / "plots" / f"{name}.png").read_bytes())
    summaries = [{**summary, "strategy": key} for key in
        ("integrated", "current_auction_only", "perfect_pv")]
    original = continuous_outputs._plot_projection_errors
    def fail_second_strategy(*args, **kwargs):
        if kwargs["strategy_label"] == "current_auction_only":
            raise RuntimeError("synthetic rendering failure")
        return original(*args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(continuous_outputs, "_plot_projection_errors", fail_second_strategy)
        with pytest.raises(RuntimeError, match="synthetic rendering failure"):
            save_continuous_projection_plots(config, comparison_root, summaries)
    assert all((comparison_root / "plots" / f"{name}.png").exists() for name in plot_names)
    references = {item["strategy"]: {"output_dir": item["output_dir"]} for item in summaries}
    generated = save_continuous_projection_plots(config, comparison_root, references)
    assert len(generated) == len(plot_names) * len(summaries)
    for name in plot_names:
        assert not (comparison_root / "plots" / f"{name}.png").exists()
        for item in summaries:
            assert (comparison_root / "plots" / f"{name}_{item['strategy']}.png").stat().st_size > 0
