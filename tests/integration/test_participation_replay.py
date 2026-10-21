"""Selected auctions remove orders/solves and support first participation after delivery starts."""
import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.simulation.synthetic import _replay_strategy, synthetic_inputs


@pytest.mark.parametrize("selection", load_config("tests/fixtures/configs/participation_cases.yaml")["selected_profiles"])
def test_oracle_replays_selected_profiles_and_carries_pre_participation_feedback(tmp_path, selection):
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model():
            pass
    except gp.GurobiError as error:
        pytest.skip(str(error))
    config = load_config("configs/full_experiments/synthetic_small.yaml")
    config["execution"] = {"reoptimize_physical_controls": False, "dispatch_policy": "track_committed_grid"}
    config["outputs"]["plots"] = False
    fixture = synthetic_inputs(config)
    fixture["schedule"] = tuple(event for event in fixture["schedule"] if event.auction in selection)
    fixture["actual_prices"] = {auction: curve for auction, curve in fixture["actual_prices"].items() if auction in selection}
    fixture["forecast_available_at"] = min(pd.Timestamp(fixture["grid"].starts[0]), fixture["schedule"][0].submission_at) - pd.Timedelta(hours=1)
    summary = _replay_strategy(config, fixture, "oracle", tmp_path / "replay")
    dispatch = pd.read_csv(tmp_path / "replay/dispatch.csv")
    ledger = pd.read_csv(tmp_path / "replay/ledger.csv")
    assert set(ledger.auction) == set(selection)
    assert summary["auction_decision_solves"] == len(selection)
    assert summary["physical_control_solves"] == 0
    before = pd.to_datetime(dispatch.start, utc=True) < fixture["schedule"][0].submission_at
    assert summary["pre_participation_intervals"] == int(before.sum())
    np.testing.assert_allclose(dispatch.loc[before, "grid_mw"], 0, atol=1e-9)
    np.testing.assert_allclose(dispatch.loc[before, "net_position_mwh"], 0)
    battery = fixture["plant"].battery
    assert dispatch.energy_end_mwh.between(battery.energy_min_mwh - 1e-9, battery.maximum_energy + 1e-9).all()
    assert np.isfinite(dispatch.pv_ac_forecast_mw).all()
    assert summary["initial_expected_profit_eur"] == pytest.approx(summary["initial_residual_objective_eur"] - summary["initial_expected_prefix_cost_eur"])


@pytest.mark.parametrize("policy", load_config("tests/fixtures/configs/participation_cases.yaml")["infeasible_auction_policies"])
def test_infeasible_auction_either_stops_or_submits_previous_planned_orders_and_keeps_controls(tmp_path, monkeypatch, policy):
    from bess_pv_trading.domain import OptimizationResult
    from bess_pv_trading.simulation import synthetic
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model():
            pass
    except gp.GurobiError as error:
        pytest.skip(str(error))
    config = load_config("configs/full_experiments/synthetic_small.yaml")
    config["outputs"]["plots"] = False
    config["execution"] = {"reoptimize_physical_controls": False,
        "dispatch_policy": "track_committed_grid", "infeasible_auction_policy": policy}
    fixture = synthetic.synthetic_inputs(config)
    original = synthetic.solve_model
    calls = []
    def fail_final(built, settings):
        calls.append(built.inputs.state.timestamp)
        if len(calls) == len(fixture["schedule"]):
            return OptimizationResult("infeasible", False)
        return original(built, settings)
    monkeypatch.setattr(synthetic, "solve_model", fail_final)
    if policy == "error":
        with pytest.raises(synthetic.InfeasibleAuctionPlan):
            synthetic._replay_strategy(config, fixture, "oracle", tmp_path / "strict")
        return
    summary = synthetic._replay_strategy(config, fixture, "oracle", tmp_path / "fallback")
    ledger = pd.read_csv(tmp_path / "fallback/ledger.csv")
    plans = pd.read_csv(tmp_path / "fallback/expected_dispatch_plans.csv")
    dispatch = pd.read_csv(tmp_path / "fallback/dispatch.csv")
    final = fixture["schedule"][-1]
    prior = plans.loc[plans.auction == fixture["schedule"][-2].auction].set_index("interval_id")
    trades = ledger.loc[ledger.auction == final.auction].set_index("interval_id")
    np.testing.assert_allclose(trades.quantity_mwh, prior.loc[trades.index, f"planned_trade_{final.auction}_mwh"])
    remaining = dispatch.loc[pd.to_datetime(dispatch.start, utc=True) >= final.submission_at].set_index("interval_id")
    for column in ("charge_mw", "discharge_mw", "curtailment_mw"):
        np.testing.assert_allclose(remaining[f"requested_{column}"], prior.loc[remaining.index, column])
    assert len(calls) == summary["auction_decision_solves"] == len(fixture["schedule"])
    assert summary["successful_auction_decision_solves"] == len(fixture["schedule"])-1
    assert summary["infeasible_auction_decision_count"] == 1
    assert summary["infeasible_auction_decisions"][0]["plan_source_auction"] == fixture["schedule"][-2].auction
    assert summary["transaction_fees"]["validation"]["valid"]
    battery = fixture["plant"].battery
    assert dispatch.energy_end_mwh.between(battery.energy_min_mwh-1e-9, battery.maximum_energy+1e-9).all()


def test_validated_negative_command_roundoff_is_floored_before_physical_execution(tmp_path, monkeypatch):
    from bess_pv_trading.simulation import synthetic
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model():
            pass
    except gp.GurobiError as error:
        pytest.skip(str(error))
    config = load_config("configs/full_experiments/synthetic_small.yaml")
    config["outputs"]["plots"] = False
    config["execution"] = {"reoptimize_physical_controls": False, "dispatch_policy": "track_committed_grid"}
    fixture = synthetic.synthetic_inputs(config)
    original = synthetic.solve_model
    def add_roundoff(built, settings):
        result = original(built, settings)
        for name in ("charge_mw", "discharge_mw", "curtailment_mw"):
            array = result.arrays[name]
            array[array == 0] = -np.finfo(array.dtype).eps
        return result
    monkeypatch.setattr(synthetic, "solve_model", add_roundoff)
    summary = synthetic._replay_strategy(config, fixture, "oracle", tmp_path / "roundoff")
    dispatch = pd.read_csv(tmp_path / "roundoff/dispatch.csv")
    assert dispatch[["requested_charge_mw", "requested_discharge_mw", "requested_curtailment_mw"]].ge(0).all().all()
    assert summary["auction_decision_solves"] == len(fixture["schedule"])
