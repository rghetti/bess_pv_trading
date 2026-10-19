"""Independent AC execution, ledger, and complete synthetic event checks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.domain import AcceptedTrade, DeliveryGrid, MarketState, PhysicalState, PlantParameters, battery_parameters_from_config
from bess_pv_trading.simulation.replay import execute_interval


def reference_cases():
    """Read reference assets and execution traces from the test YAML."""
    return load_config(Path(__file__).resolve().parents[1] / "fixtures/configs/battery_cases.yaml")


def replay_case(name):
    data = reference_cases()
    case = data["replay_cases"][name]
    battery = battery_parameters_from_config(data["batteries"][case["battery"]])
    plant = PlantParameters(**case["plant"], battery=battery)
    state = PhysicalState(pd.Timestamp(case["start"]), case["initial_energy_mwh"])
    return case, plant, state


def execute_reference_step(case, plant, state, step):
    values = dict(step)
    end = state.timestamp + pd.Timedelta(hours=values.pop("duration_hours"))
    return execute_interval(state=state, start=state.timestamp, end=end, plant=plant,
        degradation_eur_per_mwh=case["degradation_eur_per_mwh"], **values)


def test_ledger_offsets_are_financial_and_append_only():
    before = pd.Timestamp("2025-10-01T10:00Z")
    after = before + pd.Timedelta(hours=1)
    grid = DeliveryGrid(("x",),(after,),(after+pd.Timedelta(hours=1),))
    old = MarketState()
    ledger = old.accept(AcceptedTrade("DA",before,after,"x",1,100,"a"))
    ledger = ledger.accept(AcceptedTrade("IDA1",before,after,"x",-.3,80,"b"))
    assert old.accepted_trades == ()
    assert ledger.positions(grid) == pytest.approx([.7])
    assert ledger.cashflow_eur == pytest.approx(76)
    with pytest.raises(ValueError,match="Duplicate"):
        ledger.accept(ledger.accepted_trades[0])


def test_ac_charge_discharge_losses_are_applied_once_and_state_carried():
    case, plant, state = replay_case("cycle")
    records = []
    for step in case["steps"]:
        state, record = execute_reference_step(case, plant, state, step)
        records.append(record)
    expected = case["expected"]
    assert records[0]["energy_end_mwh"] == pytest.approx(expected["first_energy_end_mwh"])
    assert records[-1]["grid_mw"] == pytest.approx(expected["second_grid_mw"])
    assert state.energy_mwh == pytest.approx(expected["final_energy_mwh"], abs=1e-12)
    assert state.cumulative_throughput_mwh == pytest.approx(expected["throughput_mwh"])
    assert state.cumulative_degradation_eur == pytest.approx(expected["degradation_eur"])


def test_direct_pv_is_measured_ac_and_safety_curtails_at_poi():
    case, plant, state = replay_case("pv_safety")
    state, record = execute_reference_step(case, plant, state, case["steps"][0])
    for field, expected in case["expected"].items():
        assert record[field] == pytest.approx(expected)


def test_executor_rejects_invalid_command_and_uncompensated_self_discharge():
    case, plant, state = replay_case("idle_loss")
    step = case["steps"][0]
    with pytest.raises(ValueError,match="energy bounds"):
        execute_reference_step(case, plant, state, step)
    step["requested_charge_mw"] = case["invalid_requested_charge_mw"]
    with pytest.raises(ValueError,match="finite and nonnegative"):
        execute_reference_step(case, plant, state, step)


def test_conditional_view_rejects_unpublished_prices():
    from bess_pv_trading.domain import DegradationSettings, OptimizationInput
    from bess_pv_trading.simulation.replay import condition_and_slice
    from bess_pv_trading.simulation.synthetic import synthetic_inputs
    config = load_config(Path(__file__).resolve().parents[2]/"configs/full_experiments/synthetic_small.yaml")
    fixture = synthetic_inputs(config)
    grid=fixture["grid"]
    inputs=OptimizationInput(grid,fixture["plant"],PhysicalState(grid.starts[0],config["state"]["initial_energy_mwh"]),MarketState(),
        fixture["scenarios"],fixture["tree"],
        {event.auction:tuple(t in event.eligible_intervals for t in range(len(grid.interval_ids)))
            for event in fixture["schedule"]},config["trading"]["trade_limit_mwh"],
        config["trading"]["position_limit_mwh"], config["trading"]["imbalance_penalty_eur_per_mwh"],
        config["state"]["terminal_target_mwh"], DegradationSettings())
    with pytest.raises(ValueError,match="unpublished"):
        condition_and_slice(inputs,first_interval=0,state=inputs.state,market=inputs.market,
            observed_prices={"DA":fixture["actual_prices"]["DA"]},open_auctions=tuple(inputs.tree.bid_nodes),
            decision_cutoff=fixture["schedule"][0].submission_at)


@pytest.mark.solver
@pytest.mark.parametrize("policy", reference_cases()["full_replay"]["policies"], ids=lambda case: case["name"])
def test_all_four_auction_submissions_clear_once_and_reconcile(tmp_path, monkeypatch, policy):
    import gurobipy as gp
    try:
        environment = gp.Env(params={"OutputFlag":0})
        environment.dispose()
    except gp.GurobiError as error:
        pytest.skip(str(error))
    from bess_pv_trading.simulation import synthetic
    from bess_pv_trading.simulation.synthetic import synthetic_inputs, _replay_strategy
    config = load_config(Path(__file__).resolve().parents[2]/"configs/full_experiments/synthetic_small.yaml")
    config["outputs"]["plots"] = False
    if policy["terminal"] == "free":
        config["state"]["terminal_target_mwh"] = None
    terminal_target = config["state"]["terminal_target_mwh"]
    config["execution"] = {"reoptimize_physical_controls": policy["physical_refresh"]}
    config["execution"]["dispatch_policy"] = policy.get("dispatch_policy", "follow_battery_plan")
    config["trading"]["optimize_imbalances"] = policy.get("optimize_imbalances", True)
    fixture = synthetic_inputs(config)
    reference = reference_cases()["full_replay"]
    selected = slice(reference["first_interval"], reference["stop_interval"])
    full = fixture["grid"]
    fixture["grid"] = DeliveryGrid(full.interval_ids[selected],full.starts[selected],full.ends[selected])
    fixture["pv"] = fixture["pv"][selected]
    if policy.get("actual_settlement", False):
        fixture["realized_imbalance_prices_eur_per_mwh"] = np.full(
            len(fixture["grid"].interval_ids), reference["actual_imbalance_price_eur_per_mwh"])
    if "realized_pv_scale" in policy:
        fixture["realized_pv"] = fixture["pv"]*policy["realized_pv_scale"]
    fixture["base"] = fixture["base"][selected]
    fixture["actual_prices"] = {key:value[selected] for key,value in fixture["actual_prices"].items()}
    fixture["schedule"] = tuple(replace(event,eligible_intervals=tuple(t-selected.start
        for t in event.eligible_intervals if selected.start <= t < selected.stop))
        for event in fixture["schedule"])
    # Publishing a price inside a committed control interval never adds a solve.
    fixture["schedule"] = tuple(replace(event, results_at=event.results_at+
        pd.Timedelta(minutes=reference["publication_delay_minutes"]))
        if event.auction == reference["delayed_publication_auction"] else event for event in fixture["schedule"])
    forecast_requests = []
    if policy.get("rolling_forecast", False):
        from bess_pv_trading.data import ForecastBundle
        rolling = reference["rolling_pv"]
        def refreshed_pv(cutoff, starts):
            """Return changing predictions only for the requested unfinished horizon."""
            auction = next(event.auction for event in fixture["schedule"] if event.submission_at == cutoff)
            first = full.interval_ids[selected].index(starts[0].isoformat())
            values = fixture["pv"][first:] * rolling["factors_by_auction"][auction]
            forecast_requests.append((auction, starts))
            return ForecastBundle(fixture["plant"].farm_id,
                cutoff-pd.Timedelta(hours=rolling["availability_lag_hours"]), cutoff,
                pd.DataFrame({"pv_ac_mw": values}, index=starts), {"pv_ac_mw": "MW_AC"})
        fixture["pv_forecast_provider"] = refreshed_pv
    planned = []
    original_build = synthetic.build_model
    def capture_plan(inputs):
        """Inspect actual solver inputs at every refresh without changing decisions."""
        planned.append(inputs)
        return original_build(inputs)
    monkeypatch.setattr(synthetic, "build_model", capture_plan)
    if policy.get("expected_infeasible", False):
        # A persistent production deficit can exhaust the real battery. Strict
        # delivery may then become infeasible; never reset SOC or invent slack.
        with pytest.raises(RuntimeError, match="infeasible"):
            _replay_strategy(config,fixture,"stochastic_integrated",tmp_path/"replay")
        final = planned[-1]
        assert final.state.energy_mwh >= final.plant.battery.energy_min_mwh
        first = fixture["grid"].interval_ids.index(final.grid.interval_ids[0])
        np.testing.assert_allclose(final.scenarios.pv_ac_mw[0],fixture["pv"][first:])
        return
    summary = _replay_strategy(config,fixture,"stochastic_integrated",tmp_path/"replay")
    ledger = pd.read_csv(tmp_path/"replay/ledger.csv")
    dispatch = pd.read_csv(tmp_path/"replay/dispatch.csv")
    auction_count = len(fixture["schedule"])
    control_count = len(fixture["grid"].interval_ids)
    physical_solves = control_count if policy["physical_refresh"] else 0
    assert summary["solves"] == auction_count + physical_solves
    assert len(ledger) == sum(len(event.eligible_intervals) for event in fixture["schedule"])
    assert ledger.trade_id.is_unique
    assert set(ledger.auction) == {event.auction for event in fixture["schedule"]}
    assert summary["terminal_constraint_enabled"] == (terminal_target is not None)
    if terminal_target is None:
        assert summary["terminal_error_mwh"] is None
    else:
        assert summary["terminal_error_mwh"] < 1e-8
    if policy.get("dispatch_policy") != "track_committed_grid":
        assert summary["max_safety_adjustment_mw"] < 1e-8
    assert summary["components"]["auction_revenue_eur"] == pytest.approx(ledger.cashflow_eur.sum())
    assert summary["optimization_imbalances_enabled"] is config["trading"]["optimize_imbalances"]
    if policy.get("actual_settlement", False):
        expected_profit = (ledger.cashflow_eur.sum()-dispatch.degradation_eur.sum()-dispatch.import_fee_eur.sum()
            + (dispatch.imbalance_mwh*reference["actual_imbalance_price_eur_per_mwh"]).sum())
        assert summary["realized_profit_after_imbalance_eur"] == pytest.approx(expected_profit)
        assert summary["realized_minus_expected_profit_eur"] == pytest.approx(
            expected_profit-summary["initial_expected_profit_eur"])
    else:
        assert summary["realized_imbalance_cashflow_eur"] is None
        assert summary["realized_profit_after_imbalance_eur"] is None
        assert summary["realized_minus_expected_profit_eur"] is None
        assert summary["realized_minus_expected_profit_percent"] is None
    if not config["trading"]["optimize_imbalances"] and "realized_pv_scale" not in policy:
        np.testing.assert_allclose(dispatch.imbalance_mwh, 0, atol=1e-8)
    if "realized_pv_scale" in policy:
        assert dispatch.feedback_adjustment_mw.max() > 0
        assert (dispatch.energy_end_mwh-dispatch.planned_energy_end_mwh).abs().max() > 0
        assert summary["execution_dispatch_policy"] == policy["dispatch_policy"]
    import json
    solves=json.loads((tmp_path/"replay/solves.json").read_text())
    saved_summary=json.loads((tmp_path/"replay/summary.json").read_text())
    assert saved_summary["initial_expected_auction_revenue_eur"] == solves[0]["expected_auction_revenue_eur"]
    assert saved_summary["realized_minus_expected_auction_revenue_eur"] == pytest.approx(
        ledger.cashflow_eur.sum()-solves[0]["expected_auction_revenue_eur"])
    publications=json.loads((tmp_path/"replay/price_publications.json").read_text())
    assert len(publications) == auction_count and len(planned) == len(solves)
    assert sum(item["solve_trigger"]=="auction_submission" for item in solves) == auction_count
    assert sum(item["solve_trigger"]=="physical_control" for item in solves) == physical_solves
    assert all(item["solve_trigger"] != "auction_results" for item in solves)
    if policy.get("rolling_forecast", False):
        assert len(forecast_requests) == auction_count
        for item, inputs in zip(solves, planned):
            first = fixture["grid"].interval_ids.index(inputs.grid.interval_ids[0])
            factor = reference["rolling_pv"]["factors_by_auction"][item["current_auction"]]
            np.testing.assert_allclose(inputs.scenarios.pv_ac_mw, np.broadcast_to(
                fixture["pv"][first:]*factor, inputs.scenarios.pv_ac_mw.shape))
            assert pd.Timestamp(item["pv_forecast_available_at_utc"]) <= pd.Timestamp(item["cutoff"])
            assert pd.Timestamp(item["planning_start_utc"]) >= pd.Timestamp(item["cutoff"])
        # Later forecasts cannot rewrite earlier dispatch provenance or battery state.
        ida3_start = next(event.submission_at for event in fixture["schedule"] if event.auction == "IDA3")
        before = pd.to_datetime(dispatch.start, utc=True) < ida3_start
        assert dispatch.loc[before, "active_plan_auction"].ne("IDA3").all()
        assert dispatch.loc[~before, "active_plan_auction"].eq("IDA3").all()
    previous_cash=0.
    for item,event in zip(publications,fixture["schedule"]):
        auction=event.auction
        assert pd.Timestamp(item["published_at"]) == event.results_at
        booked=ledger.loc[ledger.auction.eq(auction)]
        assert item["publication_booked_trades"]==len(booked)
        assert item["publication_quantity_mwh"]==pytest.approx(booked.quantity_mwh.sum())
        assert item["publication_booked_cashflow_eur"]==pytest.approx(booked.cashflow_eur.sum())
        previous_cash+=booked.cashflow_eur.sum()
        assert item["accepted_contractual_cashflow_eur"]==pytest.approx(previous_cash)
    # Each actionable solve sees only published prices and preserves submitted
    # quantities even while their clearing prices are still unknown.
    for item,inputs in zip(solves,planned):
        cutoff = pd.Timestamp(item["cutoff"])
        known = [event.auction for event in fixture["schedule"] if event.results_at <= cutoff]
        assert set(item["known_auctions"]) == set(known)
        assert all(auction not in inputs.tree.bid_nodes for auction in known)
        assert all(item["known_prices_fixed"].values())
        booked = ledger.loc[ledger.auction.isin(known)]
        assert inputs.market.cashflow_eur == pytest.approx(booked.cashflow_eur.sum())
        unpriced = [event.auction for event in fixture["schedule"]
                    if event.submission_at < cutoff < event.results_at]
        pending = ledger.loc[ledger.auction.isin(unpriced)].groupby("interval_id").quantity_mwh.sum()
        np.testing.assert_allclose(inputs.pending_position_mwh,
            pending.reindex(inputs.grid.interval_ids, fill_value=0).to_numpy(), atol=1e-8)
    trace=json.loads((tmp_path/"replay/event_trace.json").read_text())
    assert sum(event["kind"]=="AUCTION_SUBMISSION" for event in trace) == auction_count
    assert sum(event["kind"]=="PHYSICAL_CONTROL" for event in trace) == control_count
    np.testing.assert_allclose(dispatch.energy_start_mwh.iloc[1:],dispatch.energy_end_mwh.iloc[:-1],atol=1e-8)
    ida3=next(event for event in fixture["schedule"] if event.auction=="IDA3")
    submission_index = next(index for index,item in enumerate(solves)
                            if item["solve_trigger"] == "auction_submission" and item["current_auction"] == ida3.auction)
    projected=planned[submission_index]
    prior=fixture["grid"].interval_ids.index(projected.grid.interval_ids[0])-1
    if policy.get("dispatch_policy") == "track_committed_grid":
        assert solves[submission_index]["state_origin"] == "forecast_projected_boundary_under_feedback"
        # The unfinished interval's final PV is unavailable at the auction cutoff.
        battery = fixture["plant"].battery
        row = dispatch.iloc[prior]
        duration = fixture["grid"].delta_hours[prior]
        expected = row.energy_start_mwh*(1-battery.self_discharge_per_hour)**duration
        charge = min(row.requested_charge_mw, max(0,battery.maximum_energy-expected)/(duration*battery.eta_charge))
        discharge = min(row.requested_discharge_mw, max(0,expected-battery.energy_min_mwh)*battery.eta_discharge/duration)
        expected += duration*(battery.eta_charge*charge-discharge/battery.eta_discharge)
        assert projected.state.energy_mwh == pytest.approx(expected, abs=1e-8)
    else:
        assert solves[submission_index]["state_origin"] == "projected_boundary_of_committed_command"
        assert projected.state.energy_mwh==pytest.approx(dispatch.energy_end_mwh.iloc[prior],abs=1e-8)
    np.testing.assert_allclose(dispatch.net_position_mwh,
        ledger.groupby("interval_id",sort=False).quantity_mwh.sum().to_numpy(),atol=1e-8)
