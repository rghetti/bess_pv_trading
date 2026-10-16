"""Realized-price benchmark information, PV members and physical optimization."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.domain import (
    BatteryParameters, DeliveryGrid, MarketState, OptimizationInput, PhysicalState,
    PlantParameters, SolverSettings,
)
from bess_pv_trading.forecasting.price_oracle import build_perfect_foresight_scenarios
from bess_pv_trading.markets import AuctionCalendar
from bess_pv_trading.optimization import build_model, solve_model, validate_solution


def oracle_fixture(pv=None, **kwargs):
    """Use two native products from an otherwise ordinary delivery day."""
    calendar = AuctionCalendar()
    full = calendar.delivery_grid("2025-11-01")
    grid = DeliveryGrid(("p0", "p1"), tuple(full.starts[40:42]), tuple(full.ends[40:42]))
    event = replace(calendar.schedule("2025-11-01")[0], eligible_intervals=(0, 1))
    arguments = {"decision_cutoff": event.submission_at,
                 "imbalance_prices_eur_per_mwh": 0.0}
    arguments.update(kwargs)
    bundle, tree = build_perfect_foresight_scenarios(grid,
        np.ones((1, 2)) if pv is None else pv, {"DA": np.array([-50., 100.])}, (event,), **arguments)
    return grid, event, bundle, tree


def test_single_member_oracle_has_actual_prices_no_noise_and_explicit_information_relaxation():
    _, _, bundle, tree = oracle_fixture()
    assert bundle.scenario_ids == ("oracle_pv_member_0",)
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["DA"], [[-50., 100.]])
    assert np.array_equal(bundle.probabilities, [1.])
    assert set(tree.bid_nodes) == {"DA"}
    assert "oracle" in bundle.origin and "perfect-foresight" in bundle.origin
    assert bundle.provenance["executable"] is False
    assert bundle.provenance["oracle_prices"] is True
    assert "none" in bundle.provenance["price_sampling"]
    assert not bundle.provenance["operational_eligible"]
    nodes = {node.node_id: node for node in tree.nodes}
    bid = tree.bid_nodes["DA"][0]
    paths = tree.paths()
    assert all("auction:DA" not in nodes[ancestor].observations for ancestor in paths[bid])


def test_member_dimension_preserved_and_future_pv_is_not_revealed_to_bids_or_battery():
    grid, _, bundle, tree = oracle_fixture([[.4, .2], [.8, .6]],
        member_probabilities=[.2, .8], member_ids=["cloudy", "sunny"])
    assert bundle.pv_ac_mw.shape == (2, 2)
    assert np.array_equal(bundle.probabilities, [.2, .8])
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["DA"][0], bundle.auction_prices_eur_per_mwh["DA"][1])
    assert len(set(tree.bid_nodes["DA"])) == 1
    assert tree.control_nodes[0][0] == tree.control_nodes[1][0]
    assert tree.control_nodes[0][1] != tree.control_nodes[1][1]
    assert tree.curtailment_nodes[0][0] != tree.curtailment_nodes[1][0]
    assert all("member" not in value for node in tree.nodes for value in node.observations)
    nodes = {node.node_id: node for node in tree.nodes}
    for member, leaf in enumerate(tree.leaf_nodes):
        assert np.prod([nodes[node].conditional_probability for node in tree.paths()[leaf]]) == pytest.approx(bundle.probabilities[member])
    assert all(node.cutoff >= grid.ends[int(value[-1])]
               for node in tree.nodes for value in node.observations if value.startswith("pv:p"))


def test_identical_observed_prefix_does_not_disclose_later_member_values():
    _, _, bundle, tree = oracle_fixture([[.4, .2], [.4, .8], [.4, .2]])
    assert len(bundle.scenario_ids) == 3
    assert len(set(tree.leaf_nodes)) == 3
    assert tree.control_nodes[0] == tree.control_nodes[1] == tree.control_nodes[2]
    assert tree.curtailment_nodes[0][0] == tree.curtailment_nodes[1][0]
    assert tree.curtailment_nodes[0][1] != tree.curtailment_nodes[1][1]
    assert tree.curtailment_nodes[0] == tree.curtailment_nodes[2]


def test_delayed_pv_observations_keep_commands_shared_until_the_explicit_clock():
    grid, _, _, _ = oracle_fixture()
    delayed = pd.DatetimeIndex(grid.ends) + pd.Timedelta(minutes=15)
    _, _, _, tree = oracle_fixture([[.4, .2], [.8, .6]], pv_observation_times=delayed)
    assert tree.control_nodes[0] == tree.control_nodes[1]
    assert tree.curtailment_nodes[0][0] == tree.curtailment_nodes[1][0]
    assert tree.curtailment_nodes[0][1] != tree.curtailment_nodes[1][1]


def test_actual_dataframes_align_native_boundaries_and_negative_prices_exactly():
    grid, event, _, _ = oracle_fixture()
    frame = pd.DataFrame({"product_start": grid.starts, "product_end": grid.ends,
                          "price_eur_mwh": [-50., 100.], "unit": "EUR/MWh"})
    bundle, _ = build_perfect_foresight_scenarios(grid, [[1., 1.]], {"DA": frame}, (event,),
        decision_cutoff=event.submission_at, imbalance_prices_eur_per_mwh=0)
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["DA"], [[-50., 100.]])
    assert bundle.provenance["actual_price_alignment"]["DA"] == "exact native product_start/product_end match"
    hourly = frame.copy()
    hourly["product_end"] = hourly.product_start + pd.Timedelta(hours=1)
    with pytest.raises(ValueError, match="no interpolation"):
        build_perfect_foresight_scenarios(grid, [[1., 1.]], {"DA": hourly}, (event,),
            decision_cutoff=event.submission_at, imbalance_prices_eur_per_mwh=0)
    wrong_unit = frame.assign(unit="EUR/kWh")
    with pytest.raises(ValueError, match="EUR/MWh"):
        build_perfect_foresight_scenarios(grid, [[1., 1.]], {"DA": wrong_unit}, (event,),
            decision_cutoff=event.submission_at, imbalance_prices_eur_per_mwh=0)
    with pytest.raises(ValueError, match="aliases conflict"):
        build_perfect_foresight_scenarios(grid, [[1., 1.]], {"DA": frame.assign(actual=[-50., 101.])}, (event,),
            decision_cutoff=event.submission_at, imbalance_prices_eur_per_mwh=0)


@pytest.mark.parametrize("as_frame", [False, True])
def test_ida3_requires_actual_afternoon_products_only_with_audited_untradable_placeholders(as_frame):
    calendar = AuctionCalendar()
    native = calendar.delivery_grid("2025-11-01")
    grid = native.to_domain()
    event = calendar.schedule("2025-11-01")[3]
    cutoff = calendar.schedule("2025-11-01")[0].submission_at
    eligible = np.array(event.eligible_intervals)
    actual = np.linspace(-20., 100., len(eligible))
    value = (pd.DataFrame({"product_start": native.starts[eligible], "product_end": native.ends[eligible],
                          "price_eur_mwh": actual}) if as_frame else actual)
    bundle, tree = build_perfect_foresight_scenarios(grid, np.ones(96), {"IDA3": value}, (event,),
        decision_cutoff=cutoff, imbalance_prices_eur_per_mwh=0)
    assert set(tree.bid_nodes) == {"IDA3"}
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["IDA3"][0, eligible], actual)
    assert np.array_equal(bundle.auction_prices_eur_per_mwh["IDA3"][0, :48], np.zeros(48))
    assert bundle.provenance["auction_eligible_product_indices"]["IDA3"] == list(range(48, 96))
    assert bundle.provenance["untradable_price_placeholders"]["IDA3"]["product_indices"] == list(range(48))
    assert "no observed labels" in bundle.provenance["untradable_price_placeholders"]["IDA3"]["policy"]
    inputs = OptimizationInput(grid, PlantParameters(1, 0., 1., BatteryParameters(0., 0., 0.)),
        PhysicalState(grid.starts[0], 0.), MarketState(), bundle, tree,
        {"IDA3": tuple(index in event.eligible_intervals for index in range(96))}, .25, .25, 1000., 0.)
    assert not any(inputs.auction_eligibility["IDA3"][:48])
    assert all(inputs.auction_eligibility["IDA3"][48:])
    with pytest.raises(ValueError, match="outside the declared actual price products"):
        replace(inputs, auction_eligibility={"IDA3": (True,) * 96})
    with pytest.raises(ValueError, match="auction-eligible"):
        build_perfect_foresight_scenarios(grid, np.ones(96), {"IDA3": np.ones(96)}, (event,),
            decision_cutoff=cutoff, imbalance_prices_eur_per_mwh=0)
    if as_frame:
        mixed = value.copy()
        mixed.loc[mixed.index[0], "product_end"] += pd.Timedelta(minutes=45)
        with pytest.raises(ValueError, match="no interpolation"):
            build_perfect_foresight_scenarios(grid, np.ones(96), {"IDA3": mixed}, (event,),
                decision_cutoff=cutoff, imbalance_prices_eur_per_mwh=0)


def test_no_missing_intraday_prices_or_future_pv_observations_are_invented():
    grid, event, _, _ = oracle_fixture()
    full_schedule = AuctionCalendar().schedule("2025-11-01")
    with pytest.raises(ValueError, match="missing IDA prices"):
        build_perfect_foresight_scenarios(grid, [[1., 1.]], {"DA": [-50., 100.]}, full_schedule,
            decision_cutoff=event.submission_at, imbalance_prices_eur_per_mwh=0)
    with pytest.raises(ValueError, match="before its interval end"):
        oracle_fixture(pv_observation_times=grid.starts)
    with pytest.raises(ValueError, match="chronological"):
        oracle_fixture(pv_observation_times=[grid.ends[1] + pd.Timedelta(hours=1), grid.ends[1]])
    with pytest.raises(ValueError, match="residual delivery start"):
        oracle_fixture(decision_cutoff=grid.ends[0])


@pytest.mark.parametrize("pv,kwargs", [
    ([[-1., 1.]], {}), ([[np.nan, 1.]], {}), ([[1., 1.]], {"member_probabilities": [0.]}),
    ([[1., 1.], [1., 1.]], {"member_probabilities": [.5, .8]}),
    ([[1., 1.], [1., 1.]], {"member_ids": ["duplicate", "duplicate"]}),
])
def test_invalid_member_inputs_are_rejected(pv, kwargs):
    with pytest.raises(ValueError):
        oracle_fixture(pv, **kwargs)


@pytest.mark.solver
@pytest.mark.parametrize("pv,expected_oracle,expected_policy", [
    ([[1., 1.]], 25., 12.5),
    ([[.5, .5], [1., 1.]], 12.5, 6.25),
])
def test_oracle_upper_bound_uses_same_constraints_and_actual_price_evaluation(pv, expected_oracle, expected_policy):
    """Compare optimized price information with a feasible fixed common bid policy."""
    gurobi = pytest.importorskip("gurobipy")
    try:
        with gurobi.Model() as model:
            model.Params.OutputFlag = 0
    except gurobi.GurobiError as error:
        pytest.skip(f"Gurobi license unavailable: {error}")
    grid, _, bundle, tree = oracle_fixture(pv)
    plant = PlantParameters(1, 0., 1., BatteryParameters(0., 0., 0.))
    inputs = OptimizationInput(grid, plant, PhysicalState(grid.starts[0], 0.), MarketState(),
        bundle, tree, {"DA": (True, True)}, .25, .25, 1000., 0.)
    settings = SolverSettings()
    oracle_model = build_model(inputs)
    oracle = solve_model(oracle_model, settings)
    assert oracle.status == "optimal"
    assert validate_solution(inputs, oracle).valid
    assert oracle.objective_eur == pytest.approx(expected_oracle)
    # A simple fixed policy sells the smaller member's PV in each native product.
    # Fix only decision bounds, preserving every physical/market constraint and
    # evaluating the identical realized-price coefficients used by the oracle.
    fixed_policy_model = build_model(inputs)
    quantity = float(np.min(pv)) * .25
    for variable in fixed_policy_model.variables["trades"].values():
        variable.LB = quantity
        variable.UB = quantity
    policy = solve_model(fixed_policy_model, settings)
    assert policy.status == "optimal"
    assert validate_solution(inputs, policy).valid
    assert policy.objective_eur == pytest.approx(expected_policy)
    assert oracle.objective_eur > policy.objective_eur
    assert oracle_model.model.Params.MIPGap == pytest.approx(1e-4)
    oracle_model.model.dispose()
    fixed_policy_model.model.dispose()
