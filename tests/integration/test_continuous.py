"""Synthetic acceptance coverage for one causal, consecutive operating trajectory."""
from copy import deepcopy
from dataclasses import replace
import importlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.data import ForecastBundle
from bess_pv_trading.domain import DeliveryGrid, OptimizationResult, PhysicalState
from bess_pv_trading.markets import AuctionCalendar
from bess_pv_trading.simulation import synthetic
from bess_pv_trading.simulation.continuous_inputs import _fingerprint


CASES = Path(__file__).resolve().parents[1] / "fixtures/configs/continuous_cases.yaml"


@pytest.fixture
def continuous_config():
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model():
            pass
    except gp.GurobiError as error:
        pytest.skip(str(error))
    config = load_config(CASES)
    # The synthetic oracle below uses its supplied PV curve directly. Keep the
    # continuous provider on that same curve for these cross-replay checks;
    # seasonal clipping has separate dedicated coverage.
    config["policy"]["forecast_pv_night"] = {"enabled": False}
    config["state"].update(
        initial_energy_mwh=config["battery"]["capacity_mwh"] * config["state"]["initial_soc_percent"] / 100,
        terminal_target_mwh=config["battery"]["capacity_mwh"] * config["state"]["terminal_soc_percent"] / 100)
    config["decision_resolution_minutes"] = config["gme_oracle_experiment"]["decision_resolution_minutes"]
    config["native_interval_minutes_by_auction"] = {
        name: item["native_interval_minutes"] or 60
        for name, item in config["gme_oracle_experiment"]["auction_datasets"].items()}
    return config


def _prepared(config, *, one_day=False, cancel=False):
    """Create absolute-time prices/PV and a causal provider without source readers."""
    section, reference = config["continuous_optimization"], config["test_case"]
    days = tuple(stamp.date().isoformat() for stamp in pd.date_range(
        section["start_date"], section["end_date"], freq="D"))
    if one_day:
        days = days[:1]
    calendar = AuctionCalendar.from_config(config)
    data, fixtures = {}, {}
    for day in days:
        daily_config = deepcopy(config)
        daily_config["synthetic"]["delivery_day"] = day
        fixture = synthetic.synthetic_inputs(daily_config)
        grid = fixture["grid"]
        # A native hourly DA curve is constant over its four physical quarters.
        # UTC keeps the two repeated autumn local hours distinct and unambiguous.
        hours = pd.DatetimeIndex(grid.starts).floor("h")
        da = fixture["actual_prices"]["DA"]
        if config["native_interval_minutes_by_auction"]["DA"] == 60:
            da = pd.Series(da, index=hours).groupby(level=0).transform("mean").to_numpy()
        fixture["actual_prices"]["DA"] = da
        fixture["base"] = da.copy()
        fixture["kwargs"]["base_prices"] = da.copy()
        fixture["realized_pv"] = fixture["pv"] * reference["realized_pv_factor"]
        if cancel and day == days[reference["cancelled_day_index"]]:
            name = reference["cancelled_auction"]
            fixture["schedule"] = tuple(event for event in fixture["schedule"] if event.auction != name)
            fixture["actual_prices"].pop(name)
        data[day] = {"grid": grid, "schedule": fixture["schedule"],
            "prices": fixture["actual_prices"], "realized_pv": fixture["realized_pv"],
            "initial_forecast": fixture["pv"].copy(), "skipped": [],
            "price_provenance": {"source": "synthetic_test"},
            "pv_provenance": {"realized_power_missing_fallback_intervals": 0},
            "daily_config": daily_config,
            "native_interval_minutes_by_auction": {
                name: config["native_interval_minutes_by_auction"][name] for name in fixture["actual_prices"]}}
        fixtures[day] = fixture
    grids = [data[day]["grid"] for day in days]
    full = DeliveryGrid(tuple(value for grid in grids for value in grid.interval_ids),
        tuple(value for grid in grids for value in grid.starts),
        tuple(value for grid in grids for value in grid.ends))
    base_pv = pd.Series(np.concatenate([fixtures[day]["pv"] for day in days]),
                        index=pd.DatetimeIndex(full.starts))
    event_names = {event.submission_at: event.auction for fixture in fixtures.values()
                   for event in fixture["schedule"]}

    def provider(cutoff, starts):
        starts = pd.DatetimeIndex(starts)
        name = event_names[pd.Timestamp(cutoff)]
        values = base_pv.reindex(starts).to_numpy() * reference["factors_by_auction"][name]
        assert np.isfinite(values).all(), "Synthetic provider request exceeds the study grid"
        return ForecastBundle(config["plant"]["farm_id"],
            pd.Timestamp(cutoff) - pd.Timedelta(hours=config["source"]["availability_lag_hours"]),
            pd.Timestamp(cutoff), pd.DataFrame({"pv_ac_mw": values}, index=starts),
            {"pv_ac_mw": "MW_AC"})

    for day, item in data.items():
        first = item["schedule"][0].submission_at
        bundle = provider(first, item["grid"].starts)
        item["initial_pv_bundle"] = bundle
        item["pv_metadata"] = {"pv_forecast_vintage_utc": bundle.forecast_vintage.isoformat(),
                               "pv_forecast_available_at_utc": bundle.available_at.isoformat()}
        fixture = fixtures[day]
        fixture.update(pv_forecast_provider=provider,
            forecast_available_at=bundle.available_at, pv_forecast_vintage=bundle.forecast_vintage,
            pv_forecast_policy=config["forecast"]["reoptimization_pv_policy"],
            realized_imbalance_prices_eur_per_mwh=np.full(len(item["grid"].interval_ids),
                reference["settlement_price_eur_per_mwh"]))
    prepared = {"calendar": calendar, "days": days, "days_data": data,
        "plant": fixtures[days[0]]["plant"], "full_grid": full,
        "initial_energy_mwh": config["state"]["initial_energy_mwh"],
        "terminal_energy_mwh": config["state"]["terminal_target_mwh"],
        "terminal_minimum_energy_mwh": None,
        "full_realized_pv": np.concatenate([data[day]["realized_pv"] for day in days]),
        "forecast_provider": provider,
        "imbalance_prices_eur_per_mwh": np.full(len(full.interval_ids), reference["settlement_price_eur_per_mwh"]),
        "settlement_frame": None, "provenance": {"source": "synthetic_test", "offline": True}}
    prepared["input_fingerprint"] = _fingerprint({"days": days,
        "actual_pv": prepared["full_realized_pv"],
        "prices": {day: item["prices"] for day, item in data.items()},
        "config": config})
    return prepared, fixtures


def _module():
    return importlib.import_module("bess_pv_trading.simulation.continuous")


def _run(config, prepared, mode, destination, *, resume=False):
    replay = _module().ContinuousReplay(config, prepared, mode, destination, resume=resume)
    summary = replay.run()
    return replay, summary


def _read(destination, filename):
    return json.loads((destination / "json" / filename).read_text())


def test_ida3_extends_next_day_and_keeps_future_orders_provisional(continuous_config, tmp_path):
    """One real-time replay trades context gates but scores the requested day only."""
    config = continuous_config
    prepared, _ = _prepared(config)
    first, following = prepared["days"][:2]
    first_grid = prepared["days_data"][first]["grid"]
    count = len(first_grid.interval_ids)
    prepared.update(days=(first,), context_days=(following,), full_grid=first_grid,
        full_realized_pv=prepared["full_realized_pv"][:count],
        imbalance_prices_eur_per_mwh=prepared["imbalance_prices_eur_per_mwh"][:count],
        terminal_energy_mwh=None, terminal_minimum_energy_mwh=0.4)
    config["continuous_optimization"].update(end_date=first, ida3_next_day_horizon=True,
                                              terminal_soc_scope="each_delivery_day")
    config["state"].update(terminal_soc_percent=None, terminal_soc_min_percent=40)
    replay, summary = _run(config, prepared, "integrated", tmp_path / "extended")
    ida3 = next(row for row in replay.solves if row["delivery_day"] == first and row["current_auction"] == "IDA3")
    assert ida3["extended_ida3"]
    assert ida3["physical_planning_intervals"] == 144
    assert ida3["minimum_energy_mwh_by_boundary"] == {48: 0.4, 144: 0.4}
    assert set(ida3["modeled_auctions"]) == {"IDA3", *(f"{following}:{name}" for name in ("DA", "IDA1", "IDA2", "IDA3"))}
    assert set(replay.submitted[following]) == {"DA", "IDA1", "IDA2"}
    assert "previous_day_IDA3" in replay.plan_snapshots[following]
    submissions = [row for row in replay.event_trace if row["kind"] == "AUCTION_SUBMISSION"]
    first_auction = submissions[0]
    assert first_auction["auction"] == "DA"
    assert first_auction["delivery_day"] == first
    assert pd.Timestamp(first_auction["at"]).tz_convert("Europe/Rome").date() == (
        pd.Timestamp(first).date() - pd.Timedelta(days=1))
    ida3_days = [row["delivery_day"] for row in submissions if row["auction"] == "IDA3"]
    assert ida3_days == [first]
    assert summary["delivery_days"] == 1
    ledger = pd.read_csv(tmp_path / "extended/csv/accepted_trades.csv")
    assert set(ledger.delivery_day) == {first}
    context = pd.read_csv(tmp_path / "extended/csv/context_accepted_trades.csv")
    assert set(context.delivery_day) == {following}
    context_plans = pd.read_csv(tmp_path / "extended/csv/context_dispatch_plans.csv")
    assert set(context_plans.delivery_day) == {following}
    assert "provisional_previous_day_ida3" in set(context_plans.plan_status)


def test_extended_ida3_enforces_equalities_at_both_future_midnights(continuous_config, tmp_path):
    """A two-day plan links energy and fixes both future midnight SOC values."""
    config = continuous_config
    prepared, _ = _prepared(config)
    first, following = prepared["days"][:2]
    first_grid = prepared["days_data"][first]["grid"]
    count = len(first_grid.interval_ids)
    prepared.update(days=(first,), context_days=(following,), full_grid=first_grid,
        full_realized_pv=prepared["full_realized_pv"][:count],
        imbalance_prices_eur_per_mwh=prepared["imbalance_prices_eur_per_mwh"][:count],
        initial_energy_mwh=0.5, terminal_energy_mwh=0.5, terminal_minimum_energy_mwh=None)
    config["continuous_optimization"].update(end_date=first, ida3_next_day_horizon=True,
                                              terminal_soc_scope="each_delivery_day")
    config["state"].update(terminal_soc_percent=50, terminal_soc_min_percent=None)
    replay, _ = _run(config, prepared, "integrated", tmp_path / "extended_equalities")
    ida3 = next(row for row in replay.solves if row["delivery_day"] == first and row["current_auction"] == "IDA3")
    assert ida3["status"] == "optimal"
    assert ida3["target_energy_mwh_by_boundary"] == {48: 0.5, 144: 0.5}
    assert list(ida3["planned_midnight_energy_mwh"].values()) == pytest.approx([0.5, 0.5])
    assert not ida3["minimum_energy_mwh_by_boundary"]


def test_single_day_equality_does_not_rewrite_projected_initial_soc(continuous_config, tmp_path):
    """The future midnight is targeted without requiring the past projection to equal it."""
    config = continuous_config
    prepared, _ = _prepared(config)
    first, following = prepared["days"][:2]
    first_grid = prepared["days_data"][first]["grid"]
    count = len(first_grid.interval_ids)
    prepared.update(days=(first,), context_days=(following,), full_grid=first_grid,
        full_realized_pv=prepared["full_realized_pv"][:count],
        imbalance_prices_eur_per_mwh=prepared["imbalance_prices_eur_per_mwh"][:count],
        initial_energy_mwh=0.1, terminal_energy_mwh=0.5, terminal_minimum_energy_mwh=None)
    config["continuous_optimization"].update(end_date=first, ida3_next_day_horizon=True,
                                              terminal_soc_scope="each_delivery_day")
    config["state"].update(terminal_soc_percent=50, terminal_soc_min_percent=None)
    replay = _module().ContinuousReplay(config, prepared, "integrated", tmp_path / "initial_state")
    event = next(item for item in replay.events if getattr(item.payload, "delivery_day", None) == first
                 and item.payload.event.auction == "DA" and item.kind.name == "AUCTION_SUBMISSION")
    inputs, _ = replay._planning_inputs(event.payload, event.at)
    assert inputs.state.energy_mwh == pytest.approx(0.1)
    assert inputs.target_energy_mwh_by_boundary == {96: 0.5}
    replay._submit(event)
    assert replay.solves[-1]["status"] == "optimal"
    assert replay.solves[-1]["planned_midnight_energy_mwh"]


@pytest.mark.parametrize("day,following,first_count,future_count", [
    ("2025-03-30", "2025-03-31", 92, 96),
    ("2025-10-25", "2025-10-26", 96, 100),
])
def test_extended_ida3_midnights_follow_local_dst_calendar(
        continuous_config, tmp_path, day, following, first_count, future_count):
    """A two-day horizon uses local midnights rather than fixed 24-hour offsets."""
    config = continuous_config
    config["continuous_optimization"].update(start_date=day, end_date=following)
    config["market"].update(valid_from=day, valid_to=following)
    prepared, _ = _prepared(config)
    first_grid = prepared["days_data"][day]["grid"]
    assert len(first_grid.interval_ids) == first_count
    assert len(prepared["days_data"][following]["grid"].interval_ids) == future_count
    prepared.update(days=(day,), context_days=(following,), full_grid=first_grid,
        full_realized_pv=prepared["full_realized_pv"][:first_count],
        imbalance_prices_eur_per_mwh=prepared["imbalance_prices_eur_per_mwh"][:first_count],
        terminal_energy_mwh=None, terminal_minimum_energy_mwh=0.4)
    config["continuous_optimization"].update(end_date=day, ida3_next_day_horizon=True,
                                              terminal_soc_scope="each_delivery_day")
    config["state"].update(terminal_soc_percent=None, terminal_soc_min_percent=40)
    replay = _module().ContinuousReplay(config, prepared, "integrated", tmp_path / day)
    replay.known[day] = ["DA", "IDA1", "IDA2"]
    event = next(item for item in prepared["days_data"][day]["schedule"] if item.auction == "IDA3")
    inputs, _ = replay._planning_inputs(_module().ContinuousAuction(day, event), event.submission_at)
    assert inputs.minimum_energy_mwh_by_boundary == {48: 0.4, 48 + future_count: 0.4}
    for boundary in inputs.minimum_energy_mwh_by_boundary:
        assert pd.Timestamp(inputs.grid.ends[boundary - 1]).tz_convert("Europe/Rome").hour == 0


def test_extended_auction_rejects_unreachable_initial_midnight_floor(continuous_config, tmp_path):
    """A projected state below the floor is not reset or silently accepted."""
    config = continuous_config
    prepared, _ = _prepared(config)
    first, following = prepared["days"][:2]
    first_grid = prepared["days_data"][first]["grid"]
    count = len(first_grid.interval_ids)
    prepared.update(days=(first,), context_days=(following,), full_grid=first_grid,
        full_realized_pv=prepared["full_realized_pv"][:count],
        imbalance_prices_eur_per_mwh=prepared["imbalance_prices_eur_per_mwh"][:count],
        initial_energy_mwh=0.1, terminal_energy_mwh=None, terminal_minimum_energy_mwh=0.4)
    config["continuous_optimization"].update(end_date=first, ida3_next_day_horizon=True)
    config["state"].update(terminal_soc_percent=None, terminal_soc_min_percent=40)
    replay = _module().ContinuousReplay(config, prepared, "integrated", tmp_path / "infeasible")
    event = next(item for item in replay.events if getattr(item.payload, "delivery_day", None) == first
                 and item.payload.event.auction == "DA" and item.kind.name == "AUCTION_SUBMISSION")
    with pytest.raises(RuntimeError, match="infeasible"):
        replay._submit(event)
    assert replay.state.energy_mwh == pytest.approx(0.1)
    assert not replay.submitted[first]


def test_extended_da_falls_back_when_projected_midnight_is_below_floor(continuous_config, tmp_path):
    """Preserve the prior provisional DA bid if updated physical state breaks the floor."""
    config = continuous_config
    prepared, _ = _prepared(config)
    first, following = prepared["days"][:2]
    first_grid = prepared["days_data"][first]["grid"]
    count = len(first_grid.interval_ids)
    original_provider = prepared["forecast_provider"]
    def zero_provider(cutoff, starts):
        bundle = original_provider(cutoff, starts)
        values = bundle.values.copy()
        values["pv_ac_mw"] = 0.0
        return replace(bundle, values=values)
    prepared.update(days=(first,), context_days=(following,), full_grid=first_grid,
        full_realized_pv=prepared["full_realized_pv"][:count],
        imbalance_prices_eur_per_mwh=prepared["imbalance_prices_eur_per_mwh"][:count],
        forecast_provider=zero_provider, terminal_energy_mwh=None, terminal_minimum_energy_mwh=0.4)
    config["continuous_optimization"].update(end_date=first, ida3_next_day_horizon=True)
    config["state"].update(terminal_soc_percent=None, terminal_soc_min_percent=40)
    replay = _module().ContinuousReplay(config, prepared, "integrated", tmp_path / "fallback_floor")
    event = next(item for item in replay.events if getattr(item.payload, "delivery_day", None) == following
                 and item.payload.event.auction == "DA" and item.kind.name == "AUCTION_SUBMISSION")
    replay.state = PhysicalState(event.at.floor("15min").to_pydatetime(), 0.1)
    future_grid = prepared["days_data"][following]["grid"]
    replay.plan_snapshots[following]["previous_day_IDA3"] = [
        {"interval_id": product, "interval_start_utc": str(start), "planned_trade_DA_mwh": 0.0}
        for product, start in zip(future_grid.interval_ids, future_grid.starts)]
    replay._submit(event)
    assert replay.solves[-1]["fallback_applied"]
    assert replay.solves[-1]["plan_source_auction"] == "previous_day_IDA3"
    assert len(replay.submitted[following]["DA"]) == len(future_grid.interval_ids)


def test_every_auction_plan_has_a_daily_hard_terminal_target(continuous_config, tmp_path, monkeypatch):
    """Each day's market model plans to end at the configured SOC target."""
    continuous_config["continuous_optimization"]["terminal_soc_scope"] = "each_delivery_day"
    continuous_config["continuous_optimization"]["validation_diagnostics"] = "failures_only"
    prepared, _ = _prepared(continuous_config)
    module = _module()
    original = module.build_model
    captured = []

    def capture(inputs):
        captured.append(inputs)
        return original(inputs)

    monkeypatch.setattr(module, "build_model", capture)
    replay, summary = _run(continuous_config, prepared, "integrated", tmp_path / "daily_target")
    assert len(prepared["days"]) > 1
    assert len(captured) == sum(len(item["schedule"]) for item in prepared["days_data"].values())
    assert all(inputs.terminal_target_mwh == prepared["terminal_energy_mwh"] for inputs in captured)
    assert all(row["terminal_target_mwh"] == prepared["terminal_energy_mwh"] for row in replay.solves)
    assert all(row["validation"]["valid"] and "max_errors" not in row["validation"] for row in replay.solves)
    for snapshots in replay.plan_snapshots.values():
        for rows in snapshots.values():
            if rows and rows[-1].get("plan_status") == "optimized":
                assert rows[-1]["soc_end_mwh"] == pytest.approx(prepared["terminal_energy_mwh"], abs=1e-6)
    assert summary["auction_terminal_soc_scope"] == "each_delivery_day"


def test_failed_auction_validation_persists_residuals(continuous_config, tmp_path, monkeypatch):
    """The attempt file retains full validation evidence before aborting a bad solve."""
    from bess_pv_trading.domain import ValidationReport
    from bess_pv_trading.simulation.events import EventKind

    continuous_config["continuous_optimization"]["validation_diagnostics"] = "failures_only"
    prepared, _ = _prepared(continuous_config, one_day=True)
    module = _module()
    replay = module.ContinuousReplay(continuous_config, prepared, "integrated", tmp_path / "failed_validation")
    report = ValidationReport(False, {"energy_balance": 0.1}, 0.0, violations=("energy_balance: 0.1",))
    monkeypatch.setattr(module, "validate_solution", lambda *args, **kwargs: report)
    event = next(event for event in replay.events if event.kind == EventKind.AUCTION_SUBMISSION)
    with pytest.raises(RuntimeError, match="solution validation failed"):
        replay._handle(event)
    attempts = json.loads((replay.destination / "json/optimization_attempts.json").read_text())
    assert attempts[-1]["validation"]["max_errors"] == {"energy_balance": 0.1}
    assert attempts[-1]["validation"]["violations"] == ["energy_balance: 0.1"]


def test_scenario_planning_uses_frozen_members_and_shared_decisions(continuous_config, tmp_path):
    """A cutoff combines the current deterministic update with frozen PV members."""
    config = continuous_config
    config["continuous_optimization"]["pv_ensemble"]["enabled"] = True
    prepared, _ = _prepared(config, one_day=True)
    day = prepared["days"][0]
    daily = prepared["days_data"][day]
    daily["ensemble_members"] = ("1", "2")
    daily["ensemble_pv"] = np.stack([daily["initial_forecast"] * 0.8, daily["initial_forecast"] * 1.2])
    first = daily["schedule"][0]
    prepared["provenance"]["pv_ensemble"] = {day: {"forecast_vintage_utc": str(first.submission_at),
        "available_at_utc": str(first.submission_at)}}
    replay = _module().ContinuousReplay(config, prepared, "integrated", tmp_path / "replay")
    instance = _module().ContinuousAuction(day, first)
    inputs, diagnostic = replay._planning_inputs(instance, first.submission_at)
    assert inputs.scenarios.scenario_ids == ("deterministic", "weather_1", "weather_2")
    np.testing.assert_allclose(inputs.scenarios.probabilities, 1 / 3)
    assert diagnostic["pv_scenario_count"] == 3
    assert all(len(set(owners)) == 1 for owners in inputs.tree.bid_nodes.values())
    assert all(len({row[t] for row in inputs.tree.control_nodes}) == 1
               for t in range(len(inputs.grid.interval_ids)))
    np.testing.assert_allclose(inputs.scenarios.pv_ac_mw[1:], daily["ensemble_pv"])


def test_scenario_replay_carries_committed_orders_and_settles_without_penalty_charge(continuous_config, tmp_path):
    """Run a small synthetic replay and reconcile common orders and actual cash."""
    config = continuous_config
    config["continuous_optimization"]["pv_ensemble"]["enabled"] = True
    config["trading"].update(optimize_imbalances=True, imbalance_penalty_eur_per_mwh=100.0)
    prepared, _ = _prepared(config, one_day=True)
    day = prepared["days"][0]
    daily = prepared["days_data"][day]
    daily["ensemble_members"] = ("1", "2")
    daily["ensemble_pv"] = np.stack([daily["initial_forecast"] * 0.8, daily["initial_forecast"] * 1.2])
    first = daily["schedule"][0]
    prepared["provenance"]["pv_ensemble"] = {day: {"forecast_vintage_utc": str(first.submission_at),
        "available_at_utc": str(first.submission_at)}}
    _, summary = _run(config, prepared, "integrated", tmp_path / "scenario")
    dispatch = pd.read_csv(tmp_path / "scenario/csv/realized_dispatch.csv")
    solves = _read(tmp_path / "scenario", "solves.json")
    assert len(solves) == len(daily["schedule"])
    assert all(row["pv_scenario_count"] == 3 for row in solves)
    assert all(row["pv_scenario_formulation"] == "unreduced" for row in solves)
    assert np.isclose(dispatch.imbalance_penalty_eur.sum(), 0.0)
    assert summary["components"]["illustrative_imbalance_penalty_eur"] == pytest.approx(
        100.0 * dispatch.imbalance_mwh.abs().sum())
    assert summary["net_profit_eur"] == pytest.approx((dispatch.auction_revenue_eur +
        dispatch.imbalance_cashflow_eur - dispatch.degradation_eur - dispatch.import_fee_eur -
        dispatch.transaction_fee_eur).sum())


def test_exact_scenario_replay_reconstructs_deterministic_curtailment(continuous_config, tmp_path):
    config = continuous_config
    config["continuous_optimization"]["pv_ensemble"]["enabled"] = True
    prepared, _ = _prepared(config, one_day=True)
    day = prepared["days"][0]
    daily = prepared["days_data"][day]
    daily["ensemble_members"] = ("1", "2")
    daily["ensemble_pv"] = np.stack([daily["initial_forecast"] * 0.8, daily["initial_forecast"] * 1.2])
    first = daily["schedule"][0]
    prepared["provenance"]["pv_ensemble"] = {day: {"forecast_vintage_utc": str(first.submission_at),
        "available_at_utc": str(first.submission_at)}}
    _, summary = _run(config, prepared, "integrated", tmp_path / "exact")
    solves = _read(tmp_path / "exact", "solves.json")
    assert len(solves) == len(daily["schedule"])
    assert all(row["pv_scenario_formulation"] == "pointwise_minimum" for row in solves)
    assert all(row["original_pv_scenario_count"] == 3 and row["modeled_pv_scenario_count"] == 1
               for row in solves)
    plans = pd.read_csv(tmp_path / "exact/csv/expected_dispatch_plans.csv")
    injection = plans.expected_grid_mw - plans.discharge_mw + plans.charge_mw
    np.testing.assert_allclose(plans.pv_ac_forecast_mw - plans.curtailment_mw, injection, atol=1e-6)
    assert summary["fallback_count"] == 0


def test_no_battery_replay_has_no_battery_flows_or_soc(continuous_config, tmp_path):
    """A zero-capacity plant remains a valid continuous execution and report."""
    config = continuous_config
    config["battery"].update(capacity_mwh=0.0, max_power_mw=0.0)
    config["state"].update(initial_energy_mwh=0.0, terminal_target_mwh=None,
                           terminal_soc_percent=None, terminal_soc_min_percent=None)
    config["degradation"].update(replacement_economics=None, throughput_lifetime=None,
                                 cost_eur_per_mwh_throughput=0.0)
    config["experiment_outputs"]["plots"] = True
    prepared, _ = _prepared(config, one_day=True)
    replay = _module().ContinuousReplay(config, prepared, "integrated", tmp_path / "no_battery",
        progress=lambda message: None)
    summary = replay.run()
    dispatch = pd.read_csv(tmp_path / "no_battery/csv/realized_dispatch.csv")
    assert summary["initial_soc_percent"] is None
    assert summary["final_soc_percent"] is None
    np.testing.assert_allclose(dispatch[["charge_mw", "discharge_mw", "energy_end_mwh", "degradation_eur"]], 0)
    assert (tmp_path / "no_battery/plots/cumulative_profit.png").is_file()
    assert not (tmp_path / "no_battery/plots/continuous_soc.png").exists()


def test_no_battery_scenario_planning_uses_all_members(continuous_config, tmp_path):
    """A PV-only plant can form an exact-delivery shared-scenario plan."""
    config = continuous_config
    config["battery"].update(capacity_mwh=0.0, max_power_mw=0.0)
    config["state"].update(initial_energy_mwh=0.0, terminal_target_mwh=None,
                           terminal_soc_percent=None, terminal_soc_min_percent=None)
    config["degradation"].update(replacement_economics=None, throughput_lifetime=None,
                                 cost_eur_per_mwh_throughput=0.0)
    config["continuous_optimization"]["pv_ensemble"]["enabled"] = True
    prepared, _ = _prepared(config, one_day=True)
    day = prepared["days"][0]
    daily = prepared["days_data"][day]
    daily["ensemble_members"] = ("1", "2")
    daily["ensemble_pv"] = np.stack([daily["initial_forecast"] * 0.8, daily["initial_forecast"] * 1.2])
    first = daily["schedule"][0]
    prepared["provenance"]["pv_ensemble"] = {day: {"forecast_vintage_utc": str(first.submission_at),
        "available_at_utc": str(first.submission_at)}}
    replay = _module().ContinuousReplay(config, prepared, "integrated", tmp_path / "pv_only")
    inputs, _ = replay._planning_inputs(_module().ContinuousAuction(day, first), first.submission_at)
    assert inputs.scenarios.pv_ac_mw.shape[0] == 3
    assert inputs.plant.battery.capacity_mwh == 0
    assert not inputs.optimize_imbalances
    built = _module().build_model(inputs)
    try:
        result = _module().solve_model(built)
    finally:
        built.model.dispose()
    assert result.has_incumbent
    assert _module().validate_solution(inputs, result).valid
    assert result.diagnostics["pv_scenario_formulation"] == "pointwise_minimum"
    assert result.diagnostics["original_pv_scenario_count"] == 3
    assert result.diagnostics["modeled_pv_scenario_count"] == 1
    np.testing.assert_allclose(result.arrays["curtailment_mw"] + result.arrays["pv_injection_mw"],
                               inputs.scenarios.pv_ac_mw, atol=1e-6)
    np.testing.assert_allclose(result.arrays["charge_mw"], 0, atol=1e-8)
    np.testing.assert_allclose(result.arrays["discharge_mw"], 0, atol=1e-8)


@pytest.mark.parametrize("mode", ["integrated", "current_auction_only"])
def test_one_day_matches_existing_price_oracle(continuous_config, tmp_path, mode):
    prepared, fixtures = _prepared(continuous_config, one_day=True)
    fixture = fixtures[prepared["days"][0]]
    fixture["auction_planning_mode"] = mode
    old = tmp_path / "independent"
    independent = synthetic._replay_strategy(deepcopy(continuous_config), fixture, "oracle", old)
    destination = tmp_path / "continuous"
    replay, summary = _run(continuous_config, prepared, mode, destination)
    old_solves = json.loads((old / "solves.json").read_text())
    solves = _read(destination, "solves.json")
    assert [row["current_auction"] for row in solves] == [row["current_auction"] for row in old_solves]
    np.testing.assert_allclose([row["objective_eur"] for row in solves],
        [row["objective_eur"] for row in old_solves], atol=continuous_config["test_case"]["comparison_atol"])
    assert replay.state.energy_mwh == pytest.approx(independent["final_state"]["energy_mwh"],
        abs=continuous_config["test_case"]["comparison_atol"])
    assert summary["net_profit_eur"] == pytest.approx(independent["realized_profit_after_imbalance_eur"],
        abs=continuous_config["test_case"]["comparison_atol"])


@pytest.mark.parametrize("mode", ["integrated", "current_auction_only"])
def test_consecutive_states_commitments_scope_and_final_only_target(continuous_config, tmp_path, monkeypatch, mode):
    tmp_path = tmp_path / "replay"
    prepared, _ = _prepared(continuous_config)
    module, captured = _module(), []
    build = module.build_model

    def capture(inputs):
        captured.append(inputs)
        return build(inputs)

    monkeypatch.setattr(module, "build_model", capture)
    replay, summary = _run(continuous_config, prepared, mode, tmp_path)
    dispatch = pd.read_csv(tmp_path / "csv/realized_dispatch.csv")
    ledger = pd.read_csv(tmp_path / "csv/accepted_trades.csv")
    solves = _read(tmp_path, "solves.json")
    np.testing.assert_allclose(dispatch.energy_start_mwh.iloc[1:], dispatch.energy_end_mwh.iloc[:-1])
    assert dispatch.energy_start_mwh.iloc[0] == pytest.approx(prepared["initial_energy_mwh"])
    assert pd.Timestamp(dispatch.start.iloc[0]) == prepared["full_grid"].starts[0]
    assert pd.Timestamp(dispatch.end.iloc[-1]) == prepared["full_grid"].ends[-1]
    assert len(solves) == sum(len(item["schedule"]) for item in prepared["days_data"].values())
    assert len(captured) == len(solves)
    assert solves[0]["initial_state_is_supplied_boundary"]
    assert pd.Timestamp(solves[0]["cutoff"]) < prepared["full_grid"].starts[0]
    for diagnostic, inputs in zip(solves, captured):
        day, auction = diagnostic["delivery_day"], diagnostic["current_auction"]
        item = prepared["days_data"][day]
        event = next(event for event in item["schedule"] if event.auction == auction)
        eligible_start = item["grid"].starts[min(event.eligible_intervals)]
        assert inputs.grid.starts[0] == eligible_start
        assert inputs.grid.ends[-1] == item["grid"].ends[-1]
        expected_target = prepared["terminal_energy_mwh"] if day == prepared["days"][-1] else None
        assert inputs.terminal_target_mwh == expected_target
        assert diagnostic["terminal_target_mwh"] == expected_target
        expected_pv = prepared["forecast_provider"](event.submission_at, inputs.grid.starts)
        np.testing.assert_allclose(inputs.scenarios.pv_ac_mw[0], expected_pv.values.pv_ac_mw)
        assert pd.Timestamp(diagnostic["pv_forecast_available_at_utc"]) <= event.submission_at
        committed = ledger.loc[pd.to_datetime(ledger.submission_time, utc=True) < event.submission_at]
        expected_positions = committed.groupby("interval_id").quantity_mwh.sum().reindex(inputs.grid.interval_ids, fill_value=0)
        np.testing.assert_allclose(inputs.market.positions(inputs.grid) + np.asarray(inputs.pending_position_mwh),
                                   expected_positions, atol=continuous_config["test_case"]["comparison_atol"])
        if mode == "current_auction_only":
            assert set(inputs.auction_eligibility) == {auction}
            assert set(inputs.tree.bid_nodes) == {auction}
        if auction == "IDA3":
            assert len(inputs.grid.interval_ids) == continuous_config["test_case"]["expected_ida3_intervals"]
    # All actual intervals keep the current delivery day's plan; tomorrow's DA
    # cannot replace today's IDA3 physical commands even though it closes today.
    for day in prepared["days"][:-1]:
        grid = prepared["days_data"][day]["grid"]
        event = next(event for event in prepared["days_data"][day]["schedule"] if event.auction == "IDA3")
        after_noon = dispatch.loc[(pd.to_datetime(dispatch.start, utc=True) >= grid.starts[min(event.eligible_intervals)])
            & (pd.to_datetime(dispatch.start, utc=True) < grid.ends[-1])]
        assert after_noon.active_plan_auction.eq("IDA3").all()
    assert replay.state.cumulative_throughput_mwh == pytest.approx(dispatch.throughput_mwh.sum())
    assert replay.state.cumulative_degradation_eur == pytest.approx(dispatch.degradation_eur.sum())
    fee_state = replay.market.transaction_fee_state
    assert fee_state.cumulative_fees_eur == pytest.approx(ledger.quantity_mwh.abs().sum()
        * continuous_config["transaction_fees"]["flat_eur_per_mwh"])
    months = pd.to_datetime(ledger.submission_time, utc=True).dt.tz_convert(
        continuous_config["market"]["timezone"]).dt.strftime("%Y-%m")
    assert set(fee_state.monthly_volume_mwh) == set(months)
    for month, rows in ledger.groupby(months):
        assert fee_state.monthly_volume_mwh[month] == pytest.approx(rows.quantity_mwh.abs().sum())
    expected_settlement = dispatch.imbalance_mwh * continuous_config["test_case"]["settlement_price_eur_per_mwh"]
    np.testing.assert_allclose(dispatch.imbalance_cashflow_eur, expected_settlement)
    assert summary["net_profit_eur"] == pytest.approx(ledger.cashflow_eur.sum() + expected_settlement.sum()
        - dispatch.degradation_eur.sum() - dispatch.import_fee_eur.sum()
        - fee_state.cumulative_fees_eur)
    da = ledger.loc[ledger.auction == "DA"].copy()
    da["native_hour"] = pd.to_datetime(da.interval_id, utc=True).dt.floor("h")
    assert da.groupby("native_hour").quantity_mwh.agg(np.ptp).max() <= continuous_config["test_case"]["comparison_atol"]


def test_next_day_midnight_projection_never_reads_future_realized_pv(continuous_config, tmp_path, monkeypatch):
    prepared, _ = _prepared(continuous_config)
    target_day = prepared["days"][1]
    event = next(event for event in prepared["days_data"][target_day]["schedule"] if event.auction == "DA")
    changed = deepcopy(prepared)
    starts = pd.DatetimeIndex(changed["full_grid"].starts)
    unseen = starts >= event.submission_at.floor("15min")
    changed["full_realized_pv"][unseen] *= continuous_config["test_case"]["future_realized_pv_factor"]
    offset = 0
    for day in changed["days"]:
        count = len(changed["days_data"][day]["grid"].interval_ids)
        changed["days_data"][day]["realized_pv"] = changed["full_realized_pv"][offset:offset + count].copy()
        offset += count
    selected = []
    module, build = _module(), _module().build_model

    def capture(inputs):
        if inputs.tree.nodes[0].cutoff == event.submission_at:
            selected.append(inputs)
        return build(inputs)

    monkeypatch.setattr(module, "build_model", capture)
    for label, inputs in (("original", prepared), ("changed", changed)):
        _run(continuous_config, inputs, "current_auction_only", tmp_path / label)
    assert len(selected) == 2
    assert selected[0].state.energy_mwh == pytest.approx(selected[1].state.energy_mwh)
    np.testing.assert_allclose(selected[0].scenarios.pv_ac_mw, selected[1].scenarios.pv_ac_mw)
    assert selected[0].state.timestamp == prepared["days_data"][target_day]["grid"].starts[0]


def test_cancelled_auction_has_no_solve_or_submission(continuous_config, tmp_path):
    tmp_path = tmp_path / "replay"
    prepared, _ = _prepared(continuous_config, cancel=True)
    _run(continuous_config, prepared, "integrated", tmp_path)
    day = prepared["days"][continuous_config["test_case"]["cancelled_day_index"]]
    auction = continuous_config["test_case"]["cancelled_auction"]
    assert not any(row["delivery_day"] == day and row["current_auction"] == auction
                   for row in _read(tmp_path, "solves.json"))


def test_sequential_infeasible_later_auction_keeps_plan_and_submits_zero(continuous_config, tmp_path, monkeypatch):
    tmp_path = tmp_path / "replay"
    prepared, _ = _prepared(continuous_config)
    day = prepared["days"][continuous_config["test_case"]["fallback_day_index"]]
    event = next(event for event in prepared["days_data"][day]["schedule"]
        if event.auction == continuous_config["test_case"]["fallback_auction"])
    module, solve = _module(), _module().solve_model

    def fail_selected(built, settings):
        if (set(built.inputs.auction_eligibility) == {event.auction}
                and built.inputs.grid.starts[0] == prepared["days_data"][day]["grid"].starts[min(event.eligible_intervals)]):
            return OptimizationResult("infeasible", False)
        return solve(built, settings)

    monkeypatch.setattr(module, "solve_model", fail_selected)
    replay, summary = _run(continuous_config, prepared, "current_auction_only", tmp_path)
    submitted = replay.submitted[day][event.auction]
    np.testing.assert_allclose(list(submitted.values()), 0)
    failed = next(row for row in replay.solves if row["delivery_day"] == day
                  and row["current_auction"] == event.auction)
    assert failed["fallback_applied"]
    assert failed["fallback_order_policy"] == "zero_new_trades"
    retained = replay.plan_snapshots[day][event.auction]
    prior = {row["interval_id"]: row for row in replay.plan_snapshots[day][failed["plan_source_auction"]]}
    for row in retained:
        for field in ("charge_mw", "discharge_mw", "curtailment_mw"):
            assert row[field] == pytest.approx(prior[row["interval_id"]][field])
    assert replay.state.timestamp == prepared["full_grid"].ends[-1]


def test_first_target_day_failure_cannot_borrow_another_days_plan(continuous_config, tmp_path, monkeypatch):
    prepared, _ = _prepared(continuous_config)
    day = prepared["days"][continuous_config["test_case"]["fallback_day_index"]]
    module, solve = _module(), _module().solve_model

    def fail_tomorrows_da(built, settings):
        if (set(built.inputs.auction_eligibility) == {"DA"}
                and built.inputs.grid.starts[0] == prepared["days_data"][day]["grid"].starts[0]):
            return OptimizationResult("infeasible", False)
        return solve(built, settings)

    monkeypatch.setattr(module, "solve_model", fail_tomorrows_da)
    with pytest.raises(RuntimeError, match=f"{day}/DA: infeasible"):
        _run(continuous_config, prepared, "current_auction_only", tmp_path / "failure")


@pytest.mark.parametrize("reoptimize", [False, True])
def test_perfect_pv_reference_preserves_daily_horizon_and_chronology(continuous_config, tmp_path, monkeypatch, reoptimize):
    continuous_config["continuous_optimization"]["perfect_pv_reoptimize_at_each_auction"] = reoptimize
    prepared, _ = _prepared(continuous_config, cancel=True)
    module, build, captured = _module(), _module().build_model, []

    def no_operational_forecast(*args):
        pytest.fail("Perfect-PV reference must use its explicitly declared perfect production")

    def capture(inputs):
        captured.append(inputs)
        return build(inputs)

    prepared["forecast_provider"] = no_operational_forecast
    monkeypatch.setattr(module, "build_model", capture)
    replay, summary = _run(continuous_config, prepared, "perfect_pv", tmp_path / "perfect")
    expected_events = [(day, event) for day in prepared["days"]
        for event in (prepared["days_data"][day]["schedule"] if reoptimize
            else prepared["days_data"][day]["schedule"][:1])]
    assert [(row["delivery_day"], row["current_auction"]) for row in replay.solves] == [
        (day, event.auction) for day, event in expected_events]
    assert len(captured) == len(expected_events)
    for inputs in captured:
        indices = pd.DatetimeIndex(prepared["full_grid"].starts).get_indexer(inputs.grid.starts)
        np.testing.assert_allclose(inputs.scenarios.pv_ac_mw[0], prepared["full_realized_pv"][indices])
        local_day = pd.Timestamp(inputs.grid.starts[0]).tz_convert(prepared["calendar"].timezone).date().isoformat()
        assert inputs.grid.ends[-1] == prepared["days_data"][local_day]["grid"].ends[-1]
    assert replay.state.energy_mwh == pytest.approx(prepared["terminal_energy_mwh"],
        abs=continuous_config["test_case"]["comparison_atol"])
    assert summary["absolute_imbalance_mwh"] <= continuous_config["test_case"]["comparison_atol"]
    dispatch = pd.read_csv(tmp_path / "perfect/csv/realized_dispatch.csv")
    tolerance = continuous_config["test_case"]["comparison_atol"]
    np.testing.assert_allclose(dispatch.discharge_mw - dispatch.charge_mw,
        dispatch.requested_discharge_mw - dispatch.requested_charge_mw, atol=tolerance, rtol=0)
    trace = _read(tmp_path / "perfect", "event_trace.json")
    retained = [row for row in trace if row.get("optimization_performed") is False]
    assert summary["retained_auction_submission_count"] == len(retained)
    assert summary["perfect_pv_reoptimize_at_each_auction"] == reoptimize
    if reoptimize:
        assert not retained
        return
    assert len(retained) == sum(len(item["schedule"]) - 1 for item in prepared["days_data"].values())
    for day, daily in prepared["days_data"].items():
        initial = daily["schedule"][0].auction
        assert list(replay.plan_snapshots[day]) == [initial]
        by_product = {row["interval_id"]: row for row in replay.plan_snapshots[day][initial]}
        for product, row in by_product.items():
            assert replay.control_plan[product] == row
        for event in daily["schedule"]:
            products = [daily["grid"].interval_ids[index] for index in event.eligible_intervals]
            assert set(replay.submitted[day][event.auction]) == set(products)
            for product in products:
                assert replay.submitted[day][event.auction][product] == pytest.approx(
                    by_product[product][f"planned_trade_{event.auction}_mwh"], abs=tolerance)
        selected = dispatch.loc[dispatch.delivery_day.eq(day)]
        assert selected.active_plan_auction.eq(initial).all()
    daily_kpis = pd.read_csv(tmp_path / "perfect/csv/daily_kpis.csv")
    np.testing.assert_allclose(daily_kpis.realized_minus_expected_profit_eur, 0, atol=tolerance, rtol=0)


@pytest.mark.parametrize("mode", ["current_auction_only", "perfect_pv"])
def test_checkpoint_resume_matches_uninterrupted_execution(continuous_config, tmp_path, monkeypatch, mode):
    prepared, _ = _prepared(continuous_config)
    module = _module()
    uninterrupted = tmp_path / "uninterrupted"
    _run(continuous_config, prepared, mode, uninterrupted)
    resumed = tmp_path / "resumed"
    save = module.ContinuousReplay.save_checkpoint

    def interrupt_after_midnight(self, event):
        save(self, event)
        raise RuntimeError("synthetic interruption after durable checkpoint")

    monkeypatch.setattr(module.ContinuousReplay, "save_checkpoint", interrupt_after_midnight)
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        _run(continuous_config, prepared, mode, resumed)
    assert (resumed / "json/checkpoint.json").is_file()
    checkpoint = _read(resumed, "checkpoint.json")
    following_day = prepared["days"][1]
    pre_delivery = {event.auction for event in prepared["days_data"][following_day]["schedule"]
                    if event.submission_at < prepared["days_data"][following_day]["grid"].starts[0]}
    assert set(checkpoint["submitted"][following_day]) == pre_delivery
    monkeypatch.setattr(module.ContinuousReplay, "save_checkpoint", save)
    _run(continuous_config, prepared, mode, resumed, resume=True)
    expected = pd.read_csv(uninterrupted / "csv/realized_dispatch.csv")
    actual = pd.read_csv(resumed / "csv/realized_dispatch.csv")
    pd.testing.assert_frame_equal(expected, actual, check_exact=False,
        atol=continuous_config["test_case"]["comparison_atol"], rtol=0)
    for name in ("accepted_trades.csv",):
        pd.testing.assert_frame_equal(pd.read_csv(uninterrupted / "csv" / name),
                                      pd.read_csv(resumed / "csv" / name))
    assert _read(resumed, "event_trace.json") == _read(uninterrupted, "event_trace.json")


@pytest.mark.parametrize("case", load_config(CASES)["test_case"]["dst_cases"])
def test_dst_day_lengths_utc_identity_and_native_products(continuous_config, tmp_path, monkeypatch, case):
    day = case["delivery_day"]
    continuous_config["continuous_optimization"].update(start_date=day, end_date=day)
    continuous_config["market"].update(valid_from=day, valid_to=day)
    continuous_config["native_interval_minutes_by_auction"]["DA"] = case["native_da_minutes"]
    continuous_config["gme_oracle_experiment"]["auction_datasets"]["DA"]["native_interval_minutes"] = case["native_da_minutes"]
    prepared, _ = _prepared(continuous_config)
    grid = prepared["full_grid"]
    assert len(grid.interval_ids) == case["expected_physical_intervals"]
    assert len(set(grid.interval_ids)) == len(grid.interval_ids)
    assert tuple(grid.starts[1:]) == tuple(grid.ends[:-1])
    assert all(pd.Timestamp(product) == start for product, start in zip(grid.interval_ids, grid.starts))
    assert np.all(np.diff(pd.DatetimeIndex(grid.starts).asi8) > 0)
    from bess_pv_trading.optimization.market_regime import configured_auction_product_groups
    groups = configured_auction_product_groups(grid, "DA", case["native_da_minutes"], prepared["calendar"].timezone)
    assert len(groups) == case["expected_da_products"]
    module, build, captured = _module(), _module().build_model, []

    def capture(inputs):
        captured.append(inputs)
        return build(inputs)

    monkeypatch.setattr(module, "build_model", capture)
    _run(continuous_config, prepared, "current_auction_only", tmp_path / "dst")
    ida3 = next(inputs for inputs in captured if set(inputs.auction_eligibility) == {"IDA3"})
    assert len(ida3.grid.interval_ids) == continuous_config["test_case"]["expected_ida3_intervals"]
    assert pd.Timestamp(ida3.grid.starts[0]).tz_convert(prepared["calendar"].timezone).strftime("%H:%M") == continuous_config["market"]["delivery_start_times"]["IDA3"]
    da = next(inputs for inputs in captured if set(inputs.auction_eligibility) == {"DA"})
    assert da.native_interval_minutes_by_auction["DA"] == case["native_da_minutes"]


def test_checkpoint_rejects_changed_config_and_input_fingerprints(continuous_config, tmp_path, monkeypatch):
    prepared, _ = _prepared(continuous_config)
    module = _module()
    destination = tmp_path / "checkpointed"
    save = module.ContinuousReplay.save_checkpoint

    def interrupt(self, event):
        save(self, event)
        raise RuntimeError("durable checkpoint")

    monkeypatch.setattr(module.ContinuousReplay, "save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="durable checkpoint"):
        _run(continuous_config, prepared, "integrated", destination)
    changed_config = deepcopy(continuous_config)
    changed_config["state"]["terminal_soc_percent"] = changed_config["state"]["initial_soc_percent"]
    with pytest.raises(ValueError, match="different configuration, inputs or policy"):
        module.ContinuousReplay(changed_config, prepared, "integrated", destination, resume=True)
    changed_input = dict(prepared, input_fingerprint=_fingerprint({"changed": prepared["input_fingerprint"]}))
    with pytest.raises(ValueError, match="different configuration, inputs or policy"):
        module.ContinuousReplay(continuous_config, changed_input, "integrated", destination, resume=True)


def test_future_prices_cannot_change_current_sequential_order(continuous_config, tmp_path):
    prepared, _ = _prepared(continuous_config, one_day=True)
    changed = deepcopy(prepared)
    day = prepared["days"][0]
    first = prepared["days_data"][day]["schedule"][0].auction
    for auction, prices in changed["days_data"][day]["prices"].items():
        if auction != first:
            prices *= continuous_config["test_case"]["future_price_multiplier"]
    original, _ = _run(continuous_config, prepared, "current_auction_only", tmp_path / "original")
    perturbed, _ = _run(continuous_config, changed, "current_auction_only", tmp_path / "perturbed")
    assert original.submitted[day][first] == pytest.approx(perturbed.submitted[day][first],
        abs=continuous_config["test_case"]["comparison_atol"])
    columns = ["charge_mw", "discharge_mw", "curtailment_mw", "soc_end_mwh"]
    np.testing.assert_allclose(pd.DataFrame(original.plan_snapshots[day][first])[columns],
        pd.DataFrame(perturbed.plan_snapshots[day][first])[columns],
        atol=continuous_config["test_case"]["comparison_atol"])


def test_continuous_participation_carries_state_and_filters_gates(continuous_config, tmp_path):
    """A fixed subset retains its own physical state across the day boundary."""
    from bess_pv_trading.simulation.continuous_participation import _profile_inputs

    config = continuous_config
    config["experiment_outputs"]["plots"] = False
    prepared, _ = _prepared(config)
    selected = _profile_inputs(prepared, ("DA",))
    assert all(all(event.auction == "DA" for event in daily["schedule"])
               for daily in selected["days_data"].values())
    assert any(event.auction != "DA" for event in prepared["days_data"][prepared["days"][0]]["schedule"])
    replay, summary = _run(config, selected, "integrated", tmp_path / "da")
    daily = summary["daily_kpis"]
    assert len(daily) >= 2
    assert daily[1]["initial_energy_mwh"] == pytest.approx(daily[0]["final_energy_mwh"])
    assert {row["current_auction"] for row in replay.solves} == {"DA"}
    assert all(row["auction"] == "DA" for row in replay.projections)


@pytest.mark.parametrize("policy,include_det", [("deterministic", True), ("ensemble_mean", True),
                                                ("ensemble_stochastic", True), ("ensemble_stochastic", False)])
def test_extended_pv_policy_information_and_scenarios(continuous_config, tmp_path, policy, include_det):
    """Planning uses available forecasts, explicit members and no realized settlement prices."""
    config = continuous_config
    config["continuous_optimization"]["pv_ensemble"].update(enabled=True, planning_policy=policy,
        include_deterministic_member=include_det, planning_imbalance_penalty_eur_per_mwh=500.0,
        cvar_confidence=0.9, cvar_weight_eur_per_mwh=2.0)
    prepared, _ = _prepared(config, one_day=True)
    day = prepared["days"][0]
    daily = prepared["days_data"][day]
    daily["ensemble_members"] = ("1", "2")
    daily["ensemble_pv"] = np.stack([daily["initial_forecast"] * 0.6, daily["initial_forecast"] * 1.0])
    first = daily["schedule"][0]
    prepared["provenance"]["pv_ensemble"] = {day: {"forecast_vintage_utc": str(first.submission_at),
        "available_at_utc": str(first.submission_at)}}
    replay = _module().ContinuousReplay(config, prepared, "integrated", tmp_path / policy)
    inputs, _ = replay._planning_inputs(_module().ContinuousAuction(day, first), first.submission_at)
    count = 3 if policy == "ensemble_stochastic" and include_det else 2 if policy == "ensemble_stochastic" else 1
    assert len(inputs.scenarios.scenario_ids) == count
    assert np.allclose(inputs.scenarios.probabilities, 1 / count)
    if policy == "ensemble_mean":
        np.testing.assert_allclose(inputs.scenarios.pv_ac_mw[0], daily["ensemble_pv"].mean(axis=0))
    if policy == "ensemble_stochastic":
        assert inputs.optimize_imbalances and inputs.imbalance_penalty_eur_per_mwh == 500.0
        assert inputs.imbalance_cvar_weight_eur_per_mwh == 2.0
        assert np.all(inputs.scenarios.imbalance_prices_eur_per_mwh == 0)
        assert all(len(set(owners)) == 1 for owners in inputs.tree.bid_nodes.values())
        assert all(len({owner[t] for owner in inputs.tree.control_nodes}) == 1 for t in range(len(inputs.grid.interval_ids)))
        built = _module().build_model(inputs)
        try:
            result = _module().solve_model(built)
        finally:
            built.model.dispose()
        assert result.has_incumbent
        assert _module().validate_solution(inputs, result).valid
        assert result.diagnostics["pv_scenario_formulation"] == "unreduced"


def test_matched_pv_only_replays_same_forecasts_and_zero_battery(continuous_config, tmp_path):
    """Counterfactual keeps shared historical inputs but executes without a battery."""
    from bess_pv_trading.simulation.continuous_pv_only import pv_only_inputs
    config = continuous_config
    prepared, _ = _prepared(config, one_day=True)
    child, no_battery = pv_only_inputs(config, prepared)
    assert no_battery["forecast_provider"] is prepared["forecast_provider"]
    assert no_battery["full_realized_pv"] is prepared["full_realized_pv"]
    assert no_battery["plant"].battery.capacity_mwh == 0
    assert no_battery["initial_energy_mwh"] == 0
    _, summary = _run(child, no_battery, "integrated", tmp_path / "matched_pv_only")
    assert summary["initial_energy_mwh"] == summary["final_energy_mwh"] == 0


def test_stochastic_replay_separates_planning_penalty_from_actual_settlement(continuous_config, tmp_path):
    """Full replay uses observed settlement and saves decision-specific ensemble diagnostics."""
    config = continuous_config
    config["experiment_outputs"]["plots"] = False
    config["continuous_optimization"]["pv_ensemble"].update(enabled=True, planning_policy="ensemble_stochastic",
        include_deterministic_member=True, planning_imbalance_penalty_eur_per_mwh=500.0,
        cvar_confidence=0.95, cvar_weight_eur_per_mwh=0.0)
    config["trading"]["optimize_imbalances"] = False
    prepared, _ = _prepared(config, one_day=True)
    day = prepared["days"][0]
    daily = prepared["days_data"][day]
    daily["ensemble_members"] = ("1", "2")
    daily["ensemble_pv"] = np.stack([daily["initial_forecast"] * 0.8, daily["initial_forecast"] * 1.2])
    first = daily["schedule"][0]
    prepared["provenance"]["pv_ensemble"] = {day: {"forecast_vintage_utc": str(first.submission_at),
        "available_at_utc": str(first.submission_at)}}
    _, summary = _run(config, prepared, "integrated", tmp_path / "stochastic")
    dispatch = pd.read_csv(tmp_path / "stochastic/csv/realized_dispatch.csv")
    assert np.allclose(dispatch.imbalance_cashflow_eur, dispatch.imbalance_mwh * config["test_case"]["settlement_price_eur_per_mwh"])
    assert np.allclose(dispatch.imbalance_penalty_eur, 0.0)
    assert summary["total_gain_eur"] == pytest.approx(dispatch.net_profit_eur.sum())
    diagnostics = pd.read_csv(tmp_path / "stochastic/csv/ensemble_diagnostics_by_decision.csv")
    assert set(diagnostics.auction) == {event.auction for event in daily["schedule"]}
    assert (tmp_path / "stochastic/csv/revenue_energy_daily.csv").is_file()


@pytest.mark.parametrize("workers", [1, 2])
def test_participation_driver_pairs_each_profile_with_pv_only(continuous_config, tmp_path, monkeypatch, workers):
    """Every requested auction subset receives its own zero-battery continuous replay."""
    from bess_pv_trading.simulation import continuous_participation
    monkeypatch.setenv("BESS_PARTICIPATION_WORKERS", str(workers))
    config = continuous_config
    config["experiment_outputs"]["plots"] = False
    config["continuous_optimization"].update(output_root=str(tmp_path), run_id="matched", resume=False)
    config["continuous_participation_study"] = {"enabled": True, "candidate_auctions": ["DA", "IDA1", "IDA2", "IDA3"],
        "required_auctions": ["DA"], "profiles": [["DA"], ["DA", "IDA1", "IDA2", "IDA3"]],
        "matched_pv_only": True, "sequential_all_auctions_baseline": False, "perfect_pv_baseline": False, "plots": False}
    prepared, _ = _prepared(config, one_day=True)
    if workers > 1:
        # Mirror production's serializable, preflight-pinned forecast provider.
        from bess_pv_trading.simulation.continuous_inputs import _PreparedForecasts
        provider = prepared["forecast_provider"]
        pinned = _PreparedForecasts()
        starts = pd.DatetimeIndex(prepared["full_grid"].starts)
        for daily in prepared["days_data"].values():
            for event in daily["schedule"]:
                pinned.add(event.submission_at, starts, provider(event.submission_at, starts))
        prepared["forecast_provider"] = pinned
    monkeypatch.setattr(continuous_participation, "prepare_continuous_inputs", lambda config, progress=None: prepared)
    result = continuous_participation.run_continuous_participation(config)
    comparison = pd.read_csv(Path(result["output_dir"]) / "csv/pv_only_comparison_daily.csv")
    assert set(comparison.strategy) == {"DA", "DA+IDA1+IDA2+IDA3"}
    assert np.allclose(comparison.incremental_profit_eur, comparison.pv_bess_profit_eur - comparison.pv_only_profit_eur)
    assert np.allclose(comparison.pv_only_initial_energy_mwh, 0)
    assert np.allclose(comparison.pv_only_final_energy_mwh, 0)


def test_ensemble_mean_projects_each_day_with_its_own_available_cycle(continuous_config, tmp_path):
    """A next-day auction projects the locked prefix with the current day's mean."""
    config = continuous_config
    config["continuous_optimization"]["pv_ensemble"].update(enabled=True, planning_policy="ensemble_mean")
    prepared, _ = _prepared(config)
    provenance = {}
    for offset, day in enumerate(prepared["days"]):
        daily = prepared["days_data"][day]
        daily["ensemble_members"] = ("1", "2")
        daily["ensemble_pv"] = np.stack([daily["initial_forecast"] * (0.3 + offset * 0.2),
                                         daily["initial_forecast"] * (0.7 + offset * 0.2)])
        first = daily["schedule"][0].submission_at
        provenance[day] = {"forecast_vintage_utc": str(first - pd.Timedelta(hours=6)),
                           "available_at_utc": str(first)}
    prepared["provenance"]["pv_ensemble"] = provenance
    next_day = prepared["days"][1]
    event = prepared["days_data"][next_day]["schedule"][0]
    replay = _module().ContinuousReplay(config, prepared, "integrated", tmp_path / "mean")
    inputs, _ = replay._planning_inputs(_module().ContinuousAuction(next_day, event), event.submission_at)
    np.testing.assert_allclose(inputs.scenarios.pv_ac_mw[0], prepared["days_data"][next_day]["ensemble_pv"].mean(axis=0))
    prefix = [row for row in replay.forecast_inputs if row["forecast_role"] == "locked_projection"]
    assert prefix
    assert all(row["pv_forecast_vintage_utc"] == provenance[prepared["days"][0]]["forecast_vintage_utc"] for row in prefix)
    expected = prepared["days_data"][prepared["days"][0]]["ensemble_pv"].mean(axis=0)
    np.testing.assert_allclose([row["pv_ac_forecast_mw"] for row in prefix], expected)
