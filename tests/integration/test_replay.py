"""Independent AC execution, ledger, and complete synthetic event checks."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.domain import AcceptedTrade, BatteryParameters, DeliveryGrid, MarketState, PhysicalState, PlantParameters
from bess_pv_trading.simulation.replay import execute_interval


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
    start = pd.Timestamp("2025-10-02T10:00Z")
    plant = PlantParameters(1,2,2,BatteryParameters(2,1,1))
    state = PhysicalState(start,0)
    state, first = execute_interval(state=state,start=start,end=start+pd.Timedelta(hours=1),
        pv_ac_mw=0,requested_charge_mw=1,requested_discharge_mw=0,
        requested_curtailment_mw=0,plant=plant,degradation_eur_per_mwh=2)
    state, second = execute_interval(state=state,start=state.timestamp,end=state.timestamp+pd.Timedelta(hours=1),
        pv_ac_mw=0,requested_charge_mw=0,requested_discharge_mw=.9409,
        requested_curtailment_mw=0,plant=plant,degradation_eur_per_mwh=2)
    assert first["energy_end_mwh"] == pytest.approx(.97)
    assert second["grid_mw"] == pytest.approx(.9409)
    assert state.energy_mwh == pytest.approx(0,abs=1e-12)
    assert state.cumulative_throughput_mwh == pytest.approx(1.9409)
    assert state.cumulative_degradation_eur == pytest.approx(3.8818)


def test_direct_pv_is_measured_ac_and_safety_curtails_at_poi():
    start = pd.Timestamp("2025-10-02T10:00Z")
    state = PhysicalState(start,0)
    plant = PlantParameters(1,0,.75,BatteryParameters(0,0,0))
    state, record = execute_interval(state=state,start=start,end=start+pd.Timedelta(minutes=15),
        pv_ac_mw=1,requested_charge_mw=0,requested_discharge_mw=0,
        requested_curtailment_mw=0,plant=plant,degradation_eur_per_mwh=0)
    assert record["grid_mw"] == .75
    assert record["curtailment_mw"] == .25
    assert record["safety_adjustment_mw"] == .25


def test_executor_rejects_invalid_command_and_uncompensated_self_discharge():
    start = pd.Timestamp("2025-10-02T10:00Z")
    plant = PlantParameters(1,1,1,BatteryParameters(1,1,1,energy_min_mwh=.1,self_discharge_per_hour=.1))
    kwargs = dict(state=PhysicalState(start,.1),start=start,end=start+pd.Timedelta(hours=1),
        pv_ac_mw=0,requested_charge_mw=0,requested_discharge_mw=0,
        requested_curtailment_mw=0,plant=plant,degradation_eur_per_mwh=0)
    with pytest.raises(ValueError,match="energy bounds"):
        execute_interval(**kwargs)
    kwargs["requested_charge_mw"] = float("nan")
    with pytest.raises(ValueError,match="finite and nonnegative"):
        execute_interval(**kwargs)


def test_conditional_view_rejects_unpublished_prices():
    from bess_pv_trading.domain import DegradationSettings, OptimizationInput
    from bess_pv_trading.simulation.replay import condition_and_slice
    from bess_pv_trading.simulation.synthetic import synthetic_inputs
    config = load_config(Path(__file__).resolve().parents[2]/"configs/synthetic_small.yaml")
    fixture = synthetic_inputs(config)
    grid=fixture["grid"]
    inputs=OptimizationInput(grid,fixture["plant"],PhysicalState(grid.starts[0],.5),MarketState(),
        fixture["scenarios"],fixture["tree"],
        {event.auction:tuple(t in event.eligible_intervals for t in range(len(grid.interval_ids)))
            for event in fixture["schedule"]},.25,.25,100,.5,DegradationSettings())
    with pytest.raises(ValueError,match="unpublished"):
        condition_and_slice(inputs,first_interval=0,state=inputs.state,market=inputs.market,
            observed_prices={"DA":fixture["actual_prices"]["DA"]},open_auctions=tuple(inputs.tree.bid_nodes),
            decision_cutoff=fixture["schedule"][0].submission_at)


@pytest.mark.solver
def test_all_four_auction_submissions_clear_once_and_reconcile(tmp_path, monkeypatch):
    import gurobipy as gp
    try:
        environment = gp.Env(params={"OutputFlag":0})
        environment.dispose()
    except gp.GurobiError as error:
        pytest.skip(str(error))
    from bess_pv_trading.simulation import synthetic
    from bess_pv_trading.simulation.synthetic import synthetic_inputs, _replay_strategy
    config = load_config(Path(__file__).resolve().parents[2]/"configs/synthetic_small.yaml")
    config["outputs"]["plots"] = False
    fixture = synthetic_inputs(config)
    selected = slice(40,50)
    full = fixture["grid"]
    fixture["grid"] = DeliveryGrid(full.interval_ids[selected],full.starts[selected],full.ends[selected])
    fixture["pv"] = fixture["pv"][selected]
    fixture["base"] = fixture["base"][selected]
    fixture["actual_prices"] = {key:value[selected] for key,value in fixture["actual_prices"].items()}
    fixture["schedule"] = tuple(replace(event,eligible_intervals=tuple(t-selected.start
        for t in event.eligible_intervals if selected.start <= t < selected.stop))
        for event in fixture["schedule"])
    # An intrainterval publication must preserve the already committed command.
    fixture["schedule"] = tuple(replace(event, results_at=event.results_at+pd.Timedelta(minutes=7))
        if event.auction == "IDA3" else event for event in fixture["schedule"])
    planned = []
    original_build = synthetic.build_model
    def capture_plan(inputs):
        """Inspect actual solver inputs at every refresh without changing decisions."""
        planned.append(inputs)
        return original_build(inputs)
    monkeypatch.setattr(synthetic, "build_model", capture_plan)
    summary = _replay_strategy(config,fixture,"stochastic_integrated",tmp_path/"replay")
    ledger = pd.read_csv(tmp_path/"replay/ledger.csv")
    dispatch = pd.read_csv(tmp_path/"replay/dispatch.csv")
    assert summary["solves"] == 18
    assert len(ledger)==32
    assert ledger.trade_id.is_unique
    assert set(ledger.auction)=={"DA","IDA1","IDA2","IDA3"}
    assert summary["terminal_error_mwh"] < 1e-8
    assert summary["max_safety_adjustment_mw"] < 1e-8
    assert summary["components"]["auction_revenue_eur"] == pytest.approx(ledger.cashflow_eur.sum())
    import json
    solves=json.loads((tmp_path/"replay/solves.json").read_text())
    publications=[(index,item) for index,item in enumerate(solves) if item["solve_trigger"]=="auction_results"]
    assert len(publications)==4 and len(planned)==len(solves)
    assert sum(item["solve_trigger"]=="auction_submission" for item in solves)==4
    assert sum(item["solve_trigger"]=="physical_control" for item in solves)==10
    previous_cash=0.
    order=("DA","IDA1","IDA2","IDA3")
    for (index,item),event in zip(publications,fixture["schedule"]):
        inputs=planned[index]
        auction=event.auction
        assert pd.Timestamp(item["cutoff"])==event.results_at
        assert item["published_auction"]==auction and auction not in inputs.tree.bid_nodes
        assert item["known_auctions"]==list(order[:order.index(auction)+1])
        assert all(item["known_prices_fixed"].values())
        first=fixture["grid"].interval_ids.index(inputs.grid.interval_ids[0])
        for known in item["known_auctions"]:
            np.testing.assert_array_equal(inputs.scenarios.auction_prices_eur_per_mwh[known],
                np.broadcast_to(fixture["actual_prices"][known][first:],inputs.scenarios.pv_ac_mw.shape))
        if auction!="IDA3":
            future=order[order.index(auction)+1]
            innovation_mean=float(np.dot(config["synthetic"]["branch_values"],config["synthetic"]["branch_probabilities"]))
            expected=(fixture["actual_prices"][auction][first:]+config["synthetic"]["stage_spreads"][future]
                      +innovation_mean*config["synthetic"]["residual_scale"][future])
            np.testing.assert_allclose(inputs.scenarios.probabilities @ inputs.scenarios.auction_prices_eur_per_mwh[future],expected)
        booked=ledger.loc[ledger.auction.eq(auction)]
        assert item["publication_booked_trades"]==len(booked)
        assert item["publication_quantity_mwh"]==pytest.approx(booked.quantity_mwh.sum())
        assert item["publication_booked_cashflow_eur"]==pytest.approx(booked.cashflow_eur.sum())
        previous_cash+=booked.cashflow_eur.sum()
        assert item["accepted_contractual_cashflow_eur"]==pytest.approx(previous_cash)
    trace=json.loads((tmp_path/"replay/event_trace.json").read_text())
    assert sum(event["kind"]=="AUCTION_SUBMISSION" for event in trace)==4
    assert sum(event["kind"]=="PHYSICAL_CONTROL" for event in trace)==10
    np.testing.assert_allclose(dispatch.energy_start_mwh.iloc[1:],dispatch.energy_end_mwh.iloc[:-1],atol=1e-8)
    ida3=next(event for event in fixture["schedule"] if event.auction=="IDA3")
    waiting=[item for item in solves if ida3.submission_at < pd.Timestamp(item["cutoff"]) < ida3.results_at]
    assert waiting and any(item["pending_position_abs_mwh"] > 1e-8 for item in waiting)
    publication_index,publication=publications[-1]
    projected=planned[publication_index]
    prior=fixture["grid"].interval_ids.index(projected.grid.interval_ids[0])-1
    assert publication["state_origin"]=="projected_boundary_of_committed_command"
    assert projected.state.energy_mwh==pytest.approx(dispatch.energy_end_mwh.iloc[prior],abs=1e-8)
    np.testing.assert_allclose(dispatch.net_position_mwh,
        ledger.groupby("interval_id",sort=False).quantity_mwh.sum().to_numpy(),atol=1e-8)
