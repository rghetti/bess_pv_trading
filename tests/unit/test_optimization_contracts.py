"""Solver-independent contract, causality and accounting regression tests."""
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from bess_pv_trading.domain import AcceptedTrade, BatteryParameters, DegradationSettings, DeliveryGrid, InformationNode, InformationTree, MarketState, OptimizationInput, OptimizationResult, PhysicalState, PlantParameters, ScenarioBundle, SolverSettings, battery_parameters_from_config
from bess_pv_trading.optimization import classify_status, evaluate_components, validate_solution
from bess_pv_trading.config import load_config
from pathlib import Path
from battery_case_config import battery_config, case_config, configured_battery


def example_inputs(prices=None, pv=None, probabilities=None, battery=None, hours=None, auctions=None):
    """Hand-built shared root with unresolved outcome leaves, not path revelation."""
    defaults = case_config("optimizer_defaults")
    values = np.atleast_2d(np.array(defaults["prices"] if prices is None else prices, dtype=float))
    hours = defaults["interval_hours"] if hours is None else hours
    if pv is None:
        pv = np.full_like(values, defaults["pv_ac_mw"])
    count, intervals = values.shape
    probabilities = np.array(probabilities if probabilities is not None else np.full(count, 1 / count))
    start = datetime.fromisoformat(defaults["start"])
    starts = tuple(start + timedelta(hours=t * hours) for t in range(intervals))
    ends = tuple(value + timedelta(hours=hours) for value in starts)
    grid = DeliveryGrid(tuple(f"t{t}" for t in range(intervals)), starts, ends)
    root = InformationNode("root", None, start - timedelta(hours=defaults["root_lead_hours"]))
    leaves = tuple(InformationNode(f"leaf{s}", "root", ends[-1], float(probabilities[s])) for s in range(count))
    auction_prices = auctions or {"DA": values}
    scenarios = ScenarioBundle(tuple(f"s{s}" for s in range(count)), probabilities, np.asarray(pv, dtype=float), auction_prices, np.zeros_like(values))
    tree = InformationTree((root,) + leaves, tuple(node.node_id for node in leaves), {auction: ("root",) * count for auction in auction_prices}, tuple(("root",) * intervals for _ in range(count)))
    plant = PlantParameters(defaults["farm_id"], defaults["poi_import_mw"], defaults["poi_export_mw"],
                            battery or configured_battery("no_bess"))
    return OptimizationInput(grid, plant, PhysicalState(start, defaults["initial_energy_mwh"]), MarketState(),
        scenarios, tree, {auction: (True,) * intervals for auction in auction_prices},
        defaults["imbalance_penalty_eur_per_mwh"], terminal_target_mwh=defaults["terminal_target_mwh"])


def without_new_trades(inputs):
    """Close every auction while retaining its declared information ownership."""
    return replace(inputs, auction_eligibility={auction: (False,) * len(inputs.grid.interval_ids)
                   for auction in inputs.auction_eligibility})


def test_ac_converter_efficiency_and_combined_convention():
    battery = configured_battery()
    values = battery_config("unit")
    assert battery.eta_charge * battery.eta_discharge == pytest.approx(
        values["combined_charge_efficiency"] * values["combined_discharge_efficiency"])
    values = battery_config("asymmetric_efficiencies")
    combined = battery_parameters_from_config(values)
    assert combined.eta_charge == values["combined_charge_efficiency"]
    values["efficiency_convention"] = "converter_and_cell"
    with pytest.raises(ValueError, match="double count"):
        battery_parameters_from_config(values)


def test_configured_soc_box_scales_when_nominal_capacity_changes():
    config = load_config(Path(__file__).resolve().parents[2] /
        "configs/full_experiments/gme_oracle_2025-01-01_farm910_throughput.yaml")["battery"]
    cases = case_config("resizing")
    for capacity in cases["capacities_mwh"]:
        config["capacity_mwh"] = capacity
        battery = battery_parameters_from_config(config)
        assert battery.max_power_mw == pytest.approx(config["max_power_mw"])
        assert battery.energy_min_mwh / capacity == pytest.approx(config["soc_min_percent"] / 100)
        assert battery.maximum_energy / capacity == pytest.approx(config["soc_max_percent"] / 100)
    invalid_reference = battery_config("soc_box")
    for overrides in cases["invalid_overrides"]:
        with pytest.raises(ValueError):
            battery_parameters_from_config({**invalid_reference, **overrides})


def test_grid_requires_aware_contiguous_boundaries():
    inputs = example_inputs()
    with pytest.raises(ValueError, match="timezone-aware"):
        replace(inputs.grid, starts=(datetime(2026, 1, 2),))
    with pytest.raises(ValueError, match="contiguous"):
        DeliveryGrid(("a", "b"), (inputs.grid.starts[0], inputs.grid.ends[0] + timedelta(hours=1)), (inputs.grid.ends[0], inputs.grid.ends[0] + timedelta(hours=2)))


def test_ledger_cashflow_and_additive_position_without_energy_losses():
    inputs = example_inputs()
    cutoff = inputs.tree.nodes[0].cutoff
    sold = AcceptedTrade("DA", cutoff, cutoff + timedelta(minutes=20), "t0", 1, 100, "sell")
    bought = AcceptedTrade("IDA1", cutoff + timedelta(hours=3), cutoff + timedelta(hours=3, minutes=20), "t0", -0.3, 80, "buy")
    original = MarketState()
    ledger = original.accept(sold).accept(bought)
    assert original.accepted_trades == ()
    assert ledger.positions(inputs.grid)[0] == pytest.approx(0.7)
    assert ledger.cashflow_eur == pytest.approx(76)
    with pytest.raises(ValueError, match="Duplicate"):
        ledger.accept(sold)


def test_state_timestamp_and_terminal_reachability_are_checked():
    inputs = example_inputs(battery=configured_battery("slow"))
    with pytest.raises(ValueError, match="timestamp"):
        replace(inputs, state=replace(inputs.state, timestamp=inputs.state.timestamp - timedelta(hours=1)))
    with pytest.raises(ValueError, match="unreachable"):
        replace(inputs, terminal_target_mwh=1)


def test_probabilities_and_tree_mass_are_validated():
    inputs = example_inputs(prices=((100,), (0,)), probabilities=(0.7, 0.3))
    with pytest.raises(ValueError, match="sum to one"):
        replace(inputs.scenarios, probabilities=np.array([0.7, 0.7]))
    with pytest.raises(ValueError, match="Leaf mass"):
        replace(inputs, scenarios=replace(inputs.scenarios, probabilities=np.array([0.3, 0.7])))
    bad_child = replace(inputs.tree.nodes[1], conditional_probability=0.9)
    with pytest.raises(ValueError, match="sum to one"):
        replace(inputs, tree=replace(inputs.tree, nodes=(inputs.tree.nodes[0], bad_child, inputs.tree.nodes[2])))


def test_future_observations_and_own_price_in_bid_are_rejected():
    inputs = example_inputs()
    root = inputs.tree.nodes[0]
    with pytest.raises(ValueError, match="unpublished"):
        replace(root, observation_available_at={"auction:DA": root.cutoff + timedelta(seconds=1)})
    with pytest.raises(ValueError, match="own unpublished"):
        replace(inputs, tree=replace(inputs.tree, nodes=(replace(root, observations=("auction:DA",)), *inputs.tree.nodes[1:])))
    with pytest.raises(ValueError, match="before interval end"):
        replace(inputs, tree=replace(inputs.tree, nodes=(replace(root, observations=("pv:t0",)), *inputs.tree.nodes[1:])))


def test_observed_price_cannot_disagree_within_information_node():
    inputs = example_inputs(prices=((100,), (200,)))
    root = replace(inputs.tree.nodes[0], observations=("auction:IDA1",))
    prices = dict(inputs.scenarios.auction_prices_eur_per_mwh, IDA1=np.array([[10.], [20.]]))
    with pytest.raises(ValueError, match="Observed auction"):
        replace(inputs, scenarios=replace(inputs.scenarios, auction_prices_eur_per_mwh=prices), tree=replace(inputs.tree, nodes=(root, *inputs.tree.nodes[1:])))


def test_future_physical_control_owner_is_rejected():
    inputs = example_inputs()
    with pytest.raises(ValueError, match="future information"):
        replace(inputs, tree=replace(inputs.tree, control_nodes=(("leaf0",),)))


def test_degradation_mode_and_solver_settings_validation():
    with pytest.raises(ValueError, match="Xu"):
        DegradationSettings(mode="xu")
    with pytest.raises(ValueError):
        SolverSettings(threads=0)
    with pytest.raises(ValueError):
        DegradationSettings(cost_eur_per_mwh_throughput=-1)


@pytest.mark.parametrize("code,count,expected", [(2, 1, "optimal"), (3, 0, "infeasible"), (5, 0, "unbounded"), (4, 0, "infeasible_or_unbounded"), (9, 1, "feasible_time_limit"), (9, 0, "time_limit_no_incumbent"), (11, 0, "no_incumbent")])
def test_solver_status_classification(code, count, expected):
    assert classify_status(code, count) == expected


def test_numerical_validator_detects_independent_tampering_and_signed_settlement():
    inputs = example_inputs()
    arrays = {"charge_mw": np.zeros((1, 1)), "discharge_mw": np.zeros((1, 1)), "mode_charge": np.zeros((1, 1)), "curtailment_mw": np.zeros((1, 1)), "grid_mw": np.ones((1, 1)), "energy_mwh": np.zeros((1, 2)), "net_position_mwh": np.ones((1, 1)), "imbalance_mwh": np.zeros((1, 1)), "imbalance_positive_mwh": np.zeros((1, 1)), "imbalance_negative_mwh": np.zeros((1, 1))}
    result = OptimizationResult("optimal", True, 100, arrays, {("DA", "root", "t0"): 1})
    assert validate_solution(inputs, result).valid
    result.arrays["grid_mw"] = np.array([[0.5]])
    assert not validate_solution(inputs, result).valid
    changed = replace(inputs, scenarios=replace(inputs.scenarios, imbalance_prices_eur_per_mwh=np.array([[-20.]]), imbalance_negative_prices_eur_per_mwh=np.array([[-40.]])), imbalance_penalty_eur_per_mwh=7)
    component = evaluate_components(changed, arrays, result.trades)
    assert component["imbalance_cashflow"][0] == pytest.approx(20)
    assert component["imbalance_penalty"][0] == pytest.approx(3.5)


def test_aging_accounting_is_carried_without_reset_or_double_booking():
    inputs = example_inputs(battery=configured_battery())
    inputs = replace(inputs, state=replace(inputs.state, cumulative_throughput_mwh=12, cumulative_degradation_eur=30), degradation=DegradationSettings(2))
    arrays = {"charge_mw": np.array([[0.5]]), "discharge_mw": np.zeros((1, 1)), "grid_mw": np.array([[-0.5]]), "net_position_mwh": np.array([[-0.5]])}
    values = evaluate_components(inputs, arrays, {("DA", "root", "t0"): -0.5})
    assert values["battery_degradation_cost"][0] == 1
    assert values["cumulative_degradation_eur"][0] == 31
    assert values["cumulative_throughput_mwh"][0] == 12.5


def test_optimization_reports_license_error_without_solution(monkeypatch):
    import bess_pv_trading.optimization.model as module

    def unavailable(_inputs):
        raise module.SolverUnavailableError("test license failure")

    monkeypatch.setattr(module, "build_model", unavailable)
    result = module.optimize(example_inputs())
    assert result.status == "license_or_environment_error"
    assert not result.has_incumbent
    assert result.arrays == {}


def test_pending_submitted_positions_have_shape_and_poi_bounds():
    inputs = example_inputs()
    with pytest.raises(ValueError, match="finite delivery-length"):
        replace(inputs, pending_position_mwh=(1, 2))
    with pytest.raises(ValueError, match="POI energy bounds"):
        replace(inputs, pending_position_mwh=(3,))


@pytest.mark.parametrize("direction", ["import", "export"])
def test_fixed_and_pending_positions_use_directional_poi_and_interval_duration(direction):
    case = case_config("poi_trading")
    hours = case["delivery_hours"][0]
    inputs = example_inputs(hours=hours)
    inputs = replace(inputs, plant=replace(inputs.plant,
        poi_import_mw=case["poi_import_mw"], poi_export_mw=case["poi_export_mw"]))
    signed_bound = case[f"poi_{direction}_mw"] * hours * (-1 if direction == "import" else 1)
    cutoff = inputs.tree.nodes[0].cutoff
    ledger = MarketState((AcceptedTrade("DA", cutoff, cutoff, "t0", signed_bound,
        case["sale_price_eur_per_mwh"]),))
    assert replace(inputs, market=ledger).market == ledger
    assert replace(inputs, pending_position_mwh=(signed_bound,)).pending_position_mwh == (signed_bound,)
    with pytest.raises(ValueError, match="POI energy bounds"):
        replace(inputs, market=ledger, pending_position_mwh=(signed_bound * (case["beyond_poi_factor"] - 1),))


@pytest.mark.parametrize("field", ["trade_limit_mwh", "position_limit_mwh",
    "trade_limit_mwh_per_native_product", "position_limit_mwh_per_quarter_hour"])
def test_obsolete_commercial_trading_caps_are_rejected(field):
    from bess_pv_trading.config import ConfigurationError, validate_config
    with pytest.raises(ConfigurationError, match="plant.poi_import_mw and plant.poi_export_mw"):
        validate_config({"trading": {field: None}})


def test_imbalance_policy_requires_boolean_in_domain_and_yaml():
    from bess_pv_trading.config import ConfigurationError, validate_config
    for value in case_config("imbalance_policy")["invalid_boolean_values"]:
        with pytest.raises(ValueError, match="boolean"):
            replace(example_inputs(), optimize_imbalances=value)
        with pytest.raises(ConfigurationError, match="boolean"):
            validate_config({"trading": {"optimize_imbalances": value}})


@pytest.mark.parametrize("enabled", [False, True])
def test_synthetic_cli_adapter_preserves_imbalance_policy(enabled):
    from bess_pv_trading.cli import _optimization_inputs
    config = load_config(Path(__file__).resolve().parents[2] / "configs/full_experiments/synthetic_small.yaml")
    config["trading"]["optimize_imbalances"] = enabled
    assert _optimization_inputs(config).optimize_imbalances is enabled
