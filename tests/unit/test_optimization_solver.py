"""Analytical economic/physics cases; skipped only if a licensed model cannot open."""
from dataclasses import replace

import numpy as np
import pytest

from bess_pv_trading.domain import AcceptedTrade, BatteryParameters, DegradationSettings, MarketState, SolverSettings
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from test_optimization_contracts import example_inputs, without_new_trades
from battery_case_config import case_config, configured_battery
from bess_pv_trading.optimization.degradation import degradation_settings_from_config
from bess_pv_trading.config import load_config
from pathlib import Path

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


def configured_solver_inputs(name):
    """Use the configured reference battery, prices and independently tabulated answer."""
    case = case_config("solver_cases")[name]
    inputs = example_inputs(prices=case["prices"], pv=np.atleast_2d(case["pv_ac_mw"]),
                            hours=case.get("interval_hours"))
    inputs = replace(inputs, plant=replace(inputs.plant, battery=configured_battery(case["battery"])),
        state=replace(inputs.state, energy_mwh=case["initial_energy_mwh"]),
        terminal_target_mwh=case["terminal_target_mwh"],
        imbalance_penalty_eur_per_mwh=case["imbalance_penalty_eur_per_mwh"],
        planned_delivery_balance_resolution_minutes=case.get("planned_delivery_balance_resolution_minutes"),
        native_interval_minutes_by_auction=case.get("native_interval_minutes_by_auction", {}),
        degradation=degradation_settings_from_config(case["degradation"]))
    return inputs, case["expected"]


def assert_configured_physics(result, expected):
    for field in ("charge_mw", "discharge_mw", "energy_mwh"):
        np.testing.assert_allclose(result.arrays[field][0], expected[field], atol=1e-9)
    assert result.objective_eur == pytest.approx(expected["objective_eur"])


def test_direct_pv_revenue_has_no_bess_converter_loss():
    result = solve(example_inputs())
    assert result.objective_eur == pytest.approx(100)
    assert result.arrays["grid_mw"][0, 0] == pytest.approx(1)


@pytest.mark.parametrize("day,hours", [("2026-03-29", 23), ("2025-10-26", 25)])
def test_dst_battery_soc_and_cashflow_use_every_actual_quarter(day, hours):
    from bess_pv_trading.markets import AuctionCalendar
    grid = AuctionCalendar().delivery_grid(day).to_domain()
    count = len(grid.interval_ids)
    battery = configured_battery("half_power")
    prices = np.r_[np.zeros(count//2), np.full(count-count//2, 100.)]
    inputs = example_inputs(prices=prices, pv=np.zeros((1, count)), battery=battery,
                            hours=.25, start=grid.starts[0].isoformat())
    initial = battery.capacity_mwh/2
    inputs = replace(inputs, grid=grid, state=replace(inputs.state, energy_mwh=initial),
                     terminal_target_mwh=initial, optimize_imbalances=False)
    result = solve(inputs)
    delta = grid.delta_hours
    assert len(result.arrays["energy_mwh"][0]) == hours*4+1
    assert result.arrays["energy_mwh"][0, -1] == pytest.approx(initial)
    charge = np.dot(result.arrays["charge_mw"][0], delta)
    discharge = np.dot(result.arrays["discharge_mw"][0], delta)
    assert charge > 0 and discharge > 0
    assert charge*battery.eta_charge == pytest.approx(discharge/battery.eta_discharge)
    assert result.objective_eur == pytest.approx((battery.maximum_energy-initial)*battery.eta_discharge*100)
    assert result.arrays["net_position_mwh"][0].sum() == pytest.approx(discharge-charge)


def test_grid_charge_discharge_applies_efficiency_once_each_direction():
    inputs, expected = configured_solver_inputs("grid_cycle")
    result = solve(inputs)
    assert_configured_physics(result, expected)


def test_null_terminal_target_omits_constraint_and_allows_initial_energy_sale():
    inputs, expected = configured_solver_inputs("initial_energy_sale")
    built = build_model(inputs)
    try:
        assert not any(c.ConstrName.startswith("terminal_energy[") for c in built.model.getConstrs())
    finally:
        built.model.dispose()
    result = solve(inputs)
    assert_configured_physics(result, expected)
    np.testing.assert_allclose(result.arrays["grid_mw"][0], expected["discharge_mw"])


def test_free_imbalances_remove_battery_value_despite_free_terminal_soc():
    inputs, expected = configured_solver_inputs("free_imbalances")
    result = solve(inputs)
    assert_configured_physics(result, expected)


def test_planned_delivery_restores_physical_battery_value_without_imbalance_cost():
    inputs, expected = configured_solver_inputs("backed_delivery")
    result = solve(inputs)
    assert_configured_physics(result, expected)
    np.testing.assert_allclose(result.arrays["grid_mw"][0], result.arrays["net_position_mwh"][0] / inputs.grid.delta_hours)


def test_native_hour_balance_preserves_quarter_hour_pv_shape():
    inputs, expected = configured_solver_inputs("hourly_pv_profile")
    result = solve(inputs)
    assert_configured_physics(result, expected)
    np.testing.assert_allclose(result.arrays["grid_mw"][0], expected["grid_mw"])
    np.testing.assert_allclose(result.arrays["imbalance_mwh"][0], expected["imbalance_mwh"])


def test_disabled_imbalances_remove_variables_and_ignore_settlement_prices():
    case = case_config("imbalance_policy")
    inputs, expected = configured_solver_inputs(case["reference_solver_case"])
    inputs = replace(inputs, optimize_imbalances=False,
                     planned_delivery_balance_resolution_minutes=None)
    built = build_model(inputs)
    legacy = build_model(replace(inputs, optimize_imbalances=True))
    try:
        assert not any(v.VarName.startswith("imbalance_") for v in built.model.getVars())
        assert legacy.model.NumVars - built.model.NumVars == (
            case["imbalance_variables_per_interval"] * inputs.scenarios.pv_ac_mw.size)
    finally:
        built.model.dispose()
        legacy.model.dispose()
    for price in case["settlement_price_cases"]:
        scenarios = replace(inputs.scenarios,
            imbalance_prices_eur_per_mwh=np.full_like(inputs.scenarios.pv_ac_mw, price))
        result = solve(replace(inputs, scenarios=scenarios))
        assert_configured_physics(result, expected)
        np.testing.assert_allclose(result.arrays["imbalance_mwh"], 0, atol=1e-9)
        np.testing.assert_allclose(result.components["imbalance_cashflow"], 0)
        np.testing.assert_allclose(result.components["imbalance_penalty"], 0)


def test_disabled_imbalances_do_not_net_quarter_hour_errors_within_hour():
    case = case_config("imbalance_policy")
    inputs, _ = configured_solver_inputs(case["reference_hourly_case"])
    result = solve(replace(inputs, optimize_imbalances=False))
    for field in ("grid_mw", "curtailment_mw"):
        np.testing.assert_allclose(result.arrays[field][0], case["strict_hourly_expected"][field])
    assert result.objective_eur == pytest.approx(case["strict_hourly_expected"]["objective_eur"])


def test_disabled_imbalances_reject_unbacked_pending_commitments():
    case = case_config("imbalance_policy")
    inputs, _ = configured_solver_inputs(case["reference_solver_case"])
    result = solve_model(build_model(replace(without_new_trades(inputs), optimize_imbalances=False,
        pending_position_mwh=(case["unbacked_commitment_mwh"],))))
    assert result.status == "infeasible"
    assert not result.has_incumbent


def test_negative_price_charge_and_curtailment():
    curtailed = solve(example_inputs(prices=(-100,)))
    assert curtailed.arrays["curtailment_mw"][0, 0] == pytest.approx(1)
    assert curtailed.objective_eur == pytest.approx(0)
    inputs = example_inputs(prices=(-20,), pv=np.zeros((1, 1)), battery=configured_battery())
    charged = solve(replace(inputs, terminal_target_mwh=0.97))
    assert charged.objective_eur == pytest.approx(20)


def test_import_commitment_is_absorbed_by_bess_and_curtailment_reduces_pv_only():
    case = case_config("poi_trading")
    battery = configured_battery("unit")
    hours = case["import_interval_hours"]
    import_mw = case["import_power_mw"]
    inputs = example_inputs(prices=(0,), pv=np.array([[case["pv_ac_mw"]]]),
        hours=hours, battery=battery)
    cutoff = inputs.tree.nodes[0].cutoff
    accepted_import = AcceptedTrade("DA", cutoff, cutoff, "t0", -import_mw * hours, 0.0)
    inputs = replace(without_new_trades(inputs),
        plant=replace(inputs.plant, poi_import_mw=case["poi_import_mw"],
            poi_export_mw=case["poi_export_mw"]),
        market=MarketState((accepted_import,)), optimize_imbalances=False,
        terminal_target_mwh=None)
    result = solve(inputs)
    pv_to_poi = result.arrays["pv_injection_mw"][0, 0]
    curtailed_pv = result.arrays["curtailment_mw"][0, 0]
    charge = result.arrays["charge_mw"][0, 0]
    assert result.arrays["grid_mw"][0, 0] == pytest.approx(-import_mw)
    assert charge >= import_mw - 1e-8
    assert pv_to_poi + curtailed_pv == pytest.approx(case["pv_ac_mw"])
    assert result.arrays["grid_mw"][0, 0] == pytest.approx(
        pv_to_poi + result.arrays["discharge_mw"][0, 0] - charge)
    expected_energy = inputs.state.energy_mwh + battery.eta_charge * charge * hours
    assert result.arrays["energy_mwh"][0, 1] == pytest.approx(expected_energy)


def test_financial_cross_auction_offset_has_no_physical_cycle():
    inputs = example_inputs(pv=np.zeros((1, 1)), auctions={"DA": np.array([[100.]]), "IDA1": np.array([[80.]])})
    result = solve(inputs)
    maximum_sale = inputs.plant.poi_export_mw * inputs.grid.delta_hours[0]
    assert result.trades["DA", "root", "t0"] == pytest.approx(maximum_sale)
    assert result.trades["IDA1", "root", "t0"] == pytest.approx(-maximum_sale)
    prices = inputs.scenarios.auction_prices_eur_per_mwh
    assert result.objective_eur == pytest.approx(maximum_sale * (prices["DA"][0, 0] - prices["IDA1"][0, 0]))
    assert result.arrays["charge_mw"].sum() + result.arrays["discharge_mw"].sum() == 0


@pytest.mark.parametrize("hours", case_config("poi_trading")["delivery_hours"])
def test_trade_energy_scales_with_poi_and_delivery_duration(hours):
    """A large PV plant may trade its full POI output at any grid resolution."""
    case = case_config("poi_trading")
    inputs = example_inputs(prices=(case["sale_price_eur_per_mwh"],),
        pv=np.array([[case["pv_ac_mw"]]]), hours=hours)
    inputs = replace(inputs, plant=replace(inputs.plant, poi_import_mw=case["poi_import_mw"],
        poi_export_mw=case["poi_export_mw"]), optimize_imbalances=False)
    result = solve(inputs)
    energy = case["poi_export_mw"] * hours
    assert result.trades["DA", "root", "t0"] == pytest.approx(energy)
    assert result.objective_eur == pytest.approx(energy * case["sale_price_eur_per_mwh"])
    assert result.arrays["curtailment_mw"][0, 0] == pytest.approx(case["pv_ac_mw"] - case["poi_export_mw"])


def test_native_hourly_trade_can_use_full_poi_in_each_quarter():
    """Product equalities preserve power without a separate hourly volume cap."""
    case = case_config("poi_trading")
    count = case["quarter_hours_per_product"]
    hours = case["native_product_minutes"] / (60 * count)
    inputs = example_inputs(prices=np.full(count, case["sale_price_eur_per_mwh"]),
        pv=np.full((1, count), case["pv_ac_mw"]), hours=hours)
    inputs = replace(inputs, plant=replace(inputs.plant, poi_import_mw=case["poi_import_mw"],
        poi_export_mw=case["poi_export_mw"]), optimize_imbalances=False,
        native_interval_minutes_by_auction={"DA": case["native_product_minutes"]})
    result = solve(inputs)
    np.testing.assert_allclose(list(result.trades.values()), case["poi_export_mw"] * hours)
    assert result.objective_eur == pytest.approx(case["poi_export_mw"] * hours * count * case["sale_price_eur_per_mwh"])


@pytest.mark.parametrize("auction", ["IDA1", "IDA2", "IDA3"])
def test_pre_october_ida_and_battery_can_change_within_hour(auction):
    """Each IDA can capture quarter-hour spreads while hourly controls cannot."""
    case = load_config(Path(__file__).resolve().parents[1] /
        "fixtures/configs/gme_workbook_cases.yaml")["ida_cycle"]
    prices = np.atleast_2d(case["prices_eur_mwh"])
    inputs = example_inputs(prices=prices, pv=np.atleast_2d(case["pv_ac_mw"]),
        battery=configured_battery(case["battery"]), hours=case["interval_hours"],
        auctions={auction: prices}, start=case["start"])
    inputs = replace(inputs, state=replace(inputs.state, energy_mwh=case["initial_energy_mwh"]),
        terminal_target_mwh=case["terminal_energy_mwh"], optimize_imbalances=False,
        native_interval_minutes_by_auction={auction: 15})
    result = solve(inputs)
    assert result.objective_eur == pytest.approx(case["expected_objective_eur"])
    assert np.dot(result.arrays["charge_mw"][0], inputs.grid.delta_hours) == pytest.approx(case["expected_charge_mwh"])
    assert np.dot(result.arrays["discharge_mw"][0], inputs.grid.delta_hours) == pytest.approx(case["expected_discharge_mwh"])
    assert result.diagnostics["product_resolution_constraints"]["dispatch_equalities"] == 0
    assert result.diagnostics["product_resolution_constraints"]["auction_trade_equalities"][auction] == 0
    assert np.ptp(result.arrays["charge_mw"][0]) > 0
    assert np.ptp(list(result.trades.values())) > 0
    # Explicit coarse controls remain configurable, including shared-node controls.
    coarse = solve(replace(inputs, decision_resolution_minutes=60))
    assert coarse.objective_eur == pytest.approx(0)


def test_mixed_native_auctions_group_da_orders_without_grouping_battery_or_idas():
    case = load_config(Path(__file__).resolve().parents[1] /
        "fixtures/configs/gme_workbook_cases.yaml")["ida_cycle"]
    prices = np.atleast_2d(case["prices_eur_mwh"])
    auctions = {"DA": np.full_like(prices, prices.mean()),
                **{auction: prices for auction in ("IDA1", "IDA2", "IDA3")}}
    inputs = example_inputs(prices=prices, pv=np.atleast_2d(case["pv_ac_mw"]),
        hours=case["interval_hours"], auctions=auctions, start=case["start"])
    built = build_model(inputs)
    try:
        constraints = built.variables["product_resolution_constraints"]
        assert constraints["dispatch_equalities"] == 0
        assert constraints["da_trade_equalities"] == len(inputs.grid.interval_ids) - 1
        assert all(constraints["auction_trade_equalities"][auction] == 0
                   for auction in ("IDA1", "IDA2", "IDA3"))
    finally:
        built.model.dispose()


def test_auction_trades_and_reversals_respect_poi_at_each_stage():
    """Bound each auction order and its cumulative position using POI ratings."""
    case = case_config("poi_trading")
    prices = {auction: np.array([[price]]) for auction, price in case["reversal_prices_eur_per_mwh"].items()}
    hours = case["delivery_hours"][0]
    inputs = example_inputs(auctions=prices, pv=np.zeros((1, 1)), hours=hours)
    inputs = replace(inputs, plant=replace(inputs.plant, poi_import_mw=case["poi_import_mw"],
        poi_export_mw=case["poi_export_mw"]), optimize_imbalances=False)
    result = solve(inputs)
    exported, imported = case["poi_export_mw"] * hours, case["poi_import_mw"] * hours
    assert result.trades["DA", "root", "t0"] == pytest.approx(exported)
    assert abs(result.trades["DA", "root", "t0"]) <= exported
    assert -imported <= result.trades["IDA1", "root", "t0"] <= exported
    assert -imported <= result.trades["IDA2", "root", "t0"] <= exported
    assert result.trades["IDA1", "root", "t0"] == pytest.approx(-imported)
    expected = sum(result.trades[auction, "root", "t0"] * price
                   for auction, price in case["reversal_prices_eur_per_mwh"].items())
    assert result.objective_eur == pytest.approx(expected)
    # Preserve the final position and physical dispatch while violating an earlier POI bound.
    tampered = dict(result.trades)
    excess = exported * (case["beyond_poi_factor"] - 1)
    tampered["DA", "root", "t0"] += excess
    tampered["IDA1", "root", "t0"] -= excess
    report = validate_solution(inputs, replace(result, trades=tampered))
    assert not report.valid
    assert report.max_errors["cumulative_poi_export_after_DA_mwh"] == pytest.approx(excess)
    assert report.max_errors["trade_poi_export_DA_mwh"] == pytest.approx(excess)


def test_validator_retains_first_interval_trade_limit_violation():
    """Aggregate per-auction diagnostics across every interval and product."""
    case = case_config("poi_trading")
    duration = case["delivery_hours"][0]
    inputs = example_inputs(prices=np.full(2, case["sale_price_eur_per_mwh"]),
        pv=np.zeros((1, 2)), hours=duration)
    inputs = replace(inputs, plant=replace(inputs.plant,
        poi_import_mw=case["poi_import_mw"], poi_export_mw=case["poi_export_mw"]),
        optimize_imbalances=False)
    result = solve(inputs)
    excessive_trade = case["poi_export_mw"] * duration * case["beyond_poi_factor"]
    product = inputs.grid.interval_ids[0]
    tampered = dict(result.trades)
    tampered["DA", "root", product] = excessive_trade
    report = validate_solution(inputs, replace(result, trades=tampered))
    assert not report.valid
    assert report.max_errors["trade_poi_export_DA_mwh"] == pytest.approx(
        case["poi_export_mw"] * duration * (case["beyond_poi_factor"] - 1))


def test_optional_imbalance_bounds_cover_both_asymmetric_poi_extremes():
    """A full import commitment and full export execution span the whole POI range."""
    case = case_config("poi_trading")
    hours = case["delivery_hours"][0]
    inputs = example_inputs(prices=(case["sale_price_eur_per_mwh"],),
        pv=np.array([[case["pv_ac_mw"]]]), hours=hours)
    inputs = replace(without_new_trades(inputs), plant=replace(inputs.plant,
        poi_import_mw=case["poi_import_mw"], poi_export_mw=case["poi_export_mw"]),
        pending_position_mwh=(-case["poi_import_mw"] * hours,), imbalance_penalty_eur_per_mwh=0)
    inputs = replace(inputs, scenarios=replace(inputs.scenarios,
        imbalance_prices_eur_per_mwh=np.full_like(inputs.scenarios.pv_ac_mw, case["sale_price_eur_per_mwh"])))
    bound = (case["poi_import_mw"] + case["poi_export_mw"]) * hours
    built = build_model(inputs)
    try:
        for direction in ("positive", "negative"):
            assert built.variables[f"imbalance_{direction}"][0, 0].UB == pytest.approx(bound)
    finally:
        built.model.dispose()
    result = solve(inputs)
    assert result.arrays["imbalance_positive_mwh"][0, 0] == pytest.approx(bound)


def test_unequal_probabilities_and_identical_leaf_split_invariance():
    inputs = example_inputs(prices=((100,), (-100,)), probabilities=(0.75, 0.25))
    result = solve(inputs)
    split = example_inputs(prices=((100,), (100,), (-100,)), probabilities=(0.25, 0.5, 0.25))
    duplicate = solve(split)
    assert result.objective_eur == pytest.approx(50)
    assert duplicate.objective_eur == pytest.approx(result.objective_eur)
    assert duplicate.trades["DA", "root", "t0"] == pytest.approx(result.trades["DA", "root", "t0"])


def test_shared_physical_controls_do_not_see_latent_future_price_path():
    inputs = example_inputs(prices=((0, 100), (100, 0)), probabilities=(0.5, 0.5), pv=np.zeros((2, 2)), battery=configured_battery())
    result = solve(inputs)
    np.testing.assert_allclose(result.arrays["charge_mw"][0], result.arrays["charge_mw"][1])
    np.testing.assert_allclose(result.arrays["discharge_mw"][0], result.arrays["discharge_mw"][1])
    assert result.objective_eur == pytest.approx(0)
    clairvoyant = solve(example_inputs(prices=(0, 100), pv=np.zeros((1, 2)), battery=configured_battery()))
    assert 0.5 * clairvoyant.objective_eur > result.objective_eur


def test_shared_node_variables_are_created_once_including_modes():
    inputs = example_inputs(prices=((0, 100), (100, 0)), pv=np.zeros((2, 2)), battery=configured_battery())
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
    inputs = replace(without_new_trades(inputs), scenarios=scenarios, imbalance_penalty_eur_per_mwh=0)
    result = solve(inputs)
    assert result.objective_eur == pytest.approx(0)
    assert result.arrays["imbalance_positive_mwh"].sum() == 0
    assert result.arrays["imbalance_negative_mwh"].sum() == 0


def test_imbalance_cap_and_no_incumbent_infeasibility():
    inputs = example_inputs(pv=np.zeros((1, 1)))
    cutoff = inputs.tree.nodes[0].cutoff
    ledger = MarketState((AcceptedTrade("DA", cutoff, cutoff, "t0", 1, 100),))
    inputs = replace(without_new_trades(inputs), market=ledger, imbalance_limit_mwh=0)
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
    inputs = example_inputs(prices=(0, 10), pv=np.zeros((1, 2)), battery=configured_battery())
    result = solve(replace(inputs, degradation=DegradationSettings(20)))
    assert result.objective_eur == pytest.approx(0)
    assert result.components["battery_degradation_cost"][0] == pytest.approx(0)


def test_export_limit_curtails_pv_and_full_battery_cannot_overcharge():
    inputs = example_inputs()
    restricted = solve(replace(inputs, plant=replace(inputs.plant, poi_export_mw=0.2)))
    assert restricted.arrays["curtailment_mw"][0, 0] == pytest.approx(0.8)
    assert restricted.objective_eur == pytest.approx(20)
    inputs = example_inputs(prices=(-100,), pv=np.zeros((1, 1)), battery=configured_battery())
    full = replace(inputs, state=replace(inputs.state, energy_mwh=1), terminal_target_mwh=1)
    result = solve(full)
    assert result.arrays["charge_mw"].sum() == pytest.approx(0)
    assert result.objective_eur == pytest.approx(0)


def test_initial_energy_discharge_and_import_fee_use_ac_quantities():
    inputs = example_inputs(prices=(100,), pv=np.zeros((1, 1)), battery=configured_battery())
    result = solve(replace(inputs, state=replace(inputs.state, energy_mwh=1)))
    assert result.arrays["discharge_mw"][0, 0] == pytest.approx(0.97)
    assert result.objective_eur == pytest.approx(97)
    inputs = example_inputs(prices=(-20,), pv=np.zeros((1, 1)), battery=configured_battery())
    charged = solve(replace(inputs, terminal_target_mwh=0.97, grid_import_fee_eur_per_mwh=3))
    assert charged.components["other_costs"][0] == pytest.approx(3)
    assert charged.objective_eur == pytest.approx(17)


def test_infeasible_iis_and_lp_export_are_opt_in(tmp_path):
    inputs = example_inputs(pv=np.zeros((1, 1)))
    cutoff = inputs.tree.nodes[0].cutoff
    inputs = replace(without_new_trades(inputs), market=MarketState((AcceptedTrade("DA", cutoff, cutoff, "t0", 1, 100),)), imbalance_limit_mwh=0)
    exported, iis = tmp_path / "small.lp", tmp_path / "small.ilp"
    result = solve_model(build_model(inputs), SolverSettings(export_path=str(exported), iis_path=str(iis)))
    assert result.status == "infeasible"
    assert exported.exists() and iis.exists()
