"""Analytical economic/physics cases; skipped only if a licensed model cannot open."""
from dataclasses import replace

import numpy as np
import pytest

from bess_pv_trading.domain import AcceptedTrade, BatteryParameters, DegradationSettings, MarketState, SolverSettings
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from test_optimization_contracts import example_inputs

pytestmark = pytest.mark.solver


@pytest.fixture(autouse=True)
def licensed_gurobi():
    gp = pytest.importorskip("gurobipy")
    try:
        with gp.Model() as model:
            model.Params.OutputFlag = 0
    except gp.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")


def solve(inputs):
    result = solve_model(build_model(inputs), SolverSettings())
    assert result.status == "optimal", result
    report = validate_solution(inputs, result)
    assert report.valid, report.violations
    return result


def test_direct_pv_revenue_has_no_bess_converter_loss():
    result = solve(example_inputs())
    assert result.objective_eur == pytest.approx(100)
    assert result.arrays["grid_mw"][0, 0] == pytest.approx(1)


def test_grid_charge_discharge_applies_efficiency_once_each_direction():
    inputs = example_inputs(prices=(0, 100), pv=np.zeros((1, 2)), battery=BatteryParameters(1, 1, 1))
    result = solve(inputs)
    assert result.arrays["charge_mw"][0, 0] == pytest.approx(1)
    assert result.arrays["energy_mwh"][0, 1] == pytest.approx(0.97)
    assert result.arrays["discharge_mw"][0, 1] == pytest.approx(0.9409)
    assert result.objective_eur == pytest.approx(94.09)


def test_negative_price_charge_and_curtailment():
    curtailed = solve(example_inputs(prices=(-100,)))
    assert curtailed.arrays["curtailment_mw"][0, 0] == pytest.approx(1)
    assert curtailed.objective_eur == pytest.approx(0)
    inputs = example_inputs(prices=(-20,), pv=np.zeros((1, 1)), battery=BatteryParameters(1, 1, 1))
    charged = solve(replace(inputs, terminal_target_mwh=0.97))
    assert charged.objective_eur == pytest.approx(20)


def test_financial_cross_auction_offset_has_no_physical_cycle():
    inputs = example_inputs(pv=np.zeros((1, 1)), auctions={"DA": np.array([[100.]]), "IDA1": np.array([[80.]])})
    result = solve(replace(inputs, trade_limit_mwh=1, position_limit_mwh=1))
    assert result.trades["DA", "root", "t0"] == pytest.approx(1)
    assert result.trades["IDA1", "root", "t0"] == pytest.approx(-1)
    assert result.objective_eur == pytest.approx(20)
    assert result.arrays["charge_mw"].sum() + result.arrays["discharge_mw"].sum() == 0


def test_unequal_probabilities_and_identical_leaf_split_invariance():
    inputs = example_inputs(prices=((100,), (-100,)), probabilities=(0.75, 0.25))
    result = solve(inputs)
    split = example_inputs(prices=((100,), (100,), (-100,)), probabilities=(0.25, 0.5, 0.25))
    duplicate = solve(split)
    assert result.objective_eur == pytest.approx(50)
    assert duplicate.objective_eur == pytest.approx(result.objective_eur)
    assert duplicate.trades["DA", "root", "t0"] == pytest.approx(result.trades["DA", "root", "t0"])


def test_shared_physical_controls_do_not_see_latent_future_price_path():
    inputs = example_inputs(prices=((0, 100), (100, 0)), probabilities=(0.5, 0.5), pv=np.zeros((2, 2)), battery=BatteryParameters(1, 1, 1))
    result = solve(inputs)
    np.testing.assert_allclose(result.arrays["charge_mw"][0], result.arrays["charge_mw"][1])
    np.testing.assert_allclose(result.arrays["discharge_mw"][0], result.arrays["discharge_mw"][1])
    assert result.objective_eur == pytest.approx(0)
    clairvoyant = solve(example_inputs(prices=(0, 100), pv=np.zeros((1, 2)), battery=BatteryParameters(1, 1, 1)))
    assert 0.5 * clairvoyant.objective_eur > result.objective_eur


def test_shared_node_variables_are_created_once_including_modes():
    inputs = example_inputs(prices=((0, 100), (100, 0)), pv=np.zeros((2, 2)), battery=BatteryParameters(1, 1, 1))
    built = build_model(inputs)
    assert len(built.variables["charge"]) == 2
    assert len(built.variables["mode"]) == 2
    assert len(built.variables["trades"]) == 2


def test_fixed_accepted_trades_remain_in_position_but_sunk_cashflow_not_rebooked():
    inputs = example_inputs(pv=np.full((1, 1), 0.7))
    cutoff = inputs.tree.nodes[0].cutoff
    ledger = MarketState((AcceptedTrade("DA", cutoff, cutoff, "t0", 1, 100), AcceptedTrade("IDA1", cutoff, cutoff, "t0", -0.3, 80)))
    inputs = replace(inputs, market=ledger, tree=replace(inputs.tree, bid_nodes={}), auction_eligibility={})
    result = solve(inputs)
    assert result.arrays["net_position_mwh"][0, 0] == pytest.approx(0.7)
    assert result.objective_eur == pytest.approx(0)
    assert result.diagnostics["accepted_contractual_cashflow_eur"] == pytest.approx(76)


def test_pending_submitted_quantity_is_fixed_before_price_publication_and_can_offset():
    inputs = example_inputs(prices=(-100,), pv=np.zeros((1, 1)))
    no_open_auction = replace(inputs, tree=replace(inputs.tree, bid_nodes={}), auction_eligibility={}, pending_position_mwh=(0.7,))
    committed = solve(no_open_auction)
    assert committed.arrays["net_position_mwh"][0, 0] == pytest.approx(0.7)
    assert committed.arrays["imbalance_mwh"][0, 0] == pytest.approx(-0.7)
    assert committed.objective_eur == pytest.approx(-700)
    assert committed.diagnostics["pending_contractual_cashflow_in_objective"] is False
    # A later financial purchase may offset pending sale quantities without cycling.
    adjusted = solve(replace(inputs, pending_position_mwh=(0.7,)))
    assert adjusted.trades["DA", "root", "t0"] == pytest.approx(-0.7)
    assert adjusted.arrays["net_position_mwh"][0, 0] == pytest.approx(0)
    assert adjusted.objective_eur == pytest.approx(70)


def test_signed_settlement_binary_prevents_opposite_direction_arbitrage():
    inputs = example_inputs(prices=(0,), pv=np.zeros((1, 1)))
    scenarios = replace(inputs.scenarios, imbalance_prices_eur_per_mwh=np.array([[100.]]), imbalance_negative_prices_eur_per_mwh=np.array([[-100.]]))
    inputs = replace(inputs, scenarios=scenarios, imbalance_penalty_eur_per_mwh=0, trade_limit_mwh=0)
    result = solve(inputs)
    assert result.objective_eur == pytest.approx(0)
    assert result.arrays["imbalance_positive_mwh"].sum() == 0
    assert result.arrays["imbalance_negative_mwh"].sum() == 0


def test_imbalance_cap_and_no_incumbent_infeasibility():
    inputs = example_inputs(pv=np.zeros((1, 1)))
    cutoff = inputs.tree.nodes[0].cutoff
    ledger = MarketState((AcceptedTrade("DA", cutoff, cutoff, "t0", 1, 100),))
    inputs = replace(inputs, market=ledger, trade_limit_mwh=0, imbalance_limit_mwh=0)
    result = solve_model(build_model(inputs), SolverSettings())
    assert result.status == "infeasible"
    assert not result.has_incumbent
    assert not result.arrays


def test_time_limit_without_incumbent_has_no_arrays():
    result = solve_model(build_model(example_inputs()), SolverSettings(time_limit_seconds=0))
    assert result.status in ("time_limit_no_incumbent", "optimal")
    if not result.has_incumbent:
        assert result.arrays == {}


def test_throughput_aging_discourages_uneconomic_battery_cycle():
    inputs = example_inputs(prices=(0, 10), pv=np.zeros((1, 2)), battery=BatteryParameters(1, 1, 1))
    result = solve(replace(inputs, degradation=DegradationSettings(20)))
    assert result.objective_eur == pytest.approx(0)
    assert result.components["battery_degradation_cost"][0] == pytest.approx(0)


def test_export_limit_curtails_pv_and_full_battery_cannot_overcharge():
    inputs = example_inputs()
    restricted = solve(replace(inputs, plant=replace(inputs.plant, poi_export_mw=0.2)))
    assert restricted.arrays["curtailment_mw"][0, 0] == pytest.approx(0.8)
    assert restricted.objective_eur == pytest.approx(20)
    inputs = example_inputs(prices=(-100,), pv=np.zeros((1, 1)), battery=BatteryParameters(1, 1, 1))
    full = replace(inputs, state=replace(inputs.state, energy_mwh=1), terminal_target_mwh=1)
    result = solve(full)
    assert result.arrays["charge_mw"].sum() == pytest.approx(0)
    assert result.objective_eur == pytest.approx(0)


def test_initial_energy_discharge_and_import_fee_use_ac_quantities():
    inputs = example_inputs(prices=(100,), pv=np.zeros((1, 1)), battery=BatteryParameters(1, 1, 1))
    result = solve(replace(inputs, state=replace(inputs.state, energy_mwh=1)))
    assert result.arrays["discharge_mw"][0, 0] == pytest.approx(0.97)
    assert result.objective_eur == pytest.approx(97)
    inputs = example_inputs(prices=(-20,), pv=np.zeros((1, 1)), battery=BatteryParameters(1, 1, 1))
    charged = solve(replace(inputs, terminal_target_mwh=0.97, grid_import_fee_eur_per_mwh=3))
    assert charged.components["other_costs"][0] == pytest.approx(3)
    assert charged.objective_eur == pytest.approx(17)


def test_infeasible_iis_and_lp_export_are_opt_in(tmp_path):
    inputs = example_inputs(pv=np.zeros((1, 1)))
    cutoff = inputs.tree.nodes[0].cutoff
    inputs = replace(inputs, market=MarketState((AcceptedTrade("DA", cutoff, cutoff, "t0", 1, 100),)), trade_limit_mwh=0, imbalance_limit_mwh=0)
    exported, iis = tmp_path / "small.lp", tmp_path / "small.ilp"
    result = solve_model(build_model(inputs), SolverSettings(export_path=str(exported), iis_path=str(iis)))
    assert result.status == "infeasible"
    assert exported.exists() and iis.exists()
