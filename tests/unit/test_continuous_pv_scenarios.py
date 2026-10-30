"""Exercise shared decisions, PV feedback and scenario-weighted economics."""
from dataclasses import replace

import numpy as np
import pytest

from bess_pv_trading.domain import SolverSettings
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.scenarios import shared_dispatch_pv_tree
from test_optimization_contracts import example_inputs, without_new_trades


def scenario_inputs(*, exact):
    """Use a no-battery one-hour example with one low and two high PV members."""
    original = example_inputs(prices=[[40.0]], pv=np.array([[1.0]]), hours=1.0)
    bundle, tree = shared_dispatch_pv_tree(original.scenarios, original.tree, original.grid,
        np.array([[0.0], [1.0], [1.0]]), ("deterministic", "weather_1", "weather_2"))
    return replace(original, scenarios=bundle, tree=tree, optimize_imbalances=not exact,
                   imbalance_penalty_eur_per_mwh=100.0)


def solve_checked(inputs):
    """Verify the incumbent independently from the solver's own model."""
    pytest.importorskip("gurobipy")
    built = build_model(inputs)
    try:
        result = solve_model(built, SolverSettings())
    finally:
        built.model.dispose()
    assert result.has_incumbent, result
    assert validate_solution(inputs, result).valid
    return result


def test_scenario_probabilities_and_common_decisions():
    inputs = scenario_inputs(exact=True)
    np.testing.assert_allclose(inputs.scenarios.probabilities, np.full(3, 1 / 3))
    assert len(set(inputs.tree.bid_nodes["DA"])) == 1
    assert len({owners[0] for owners in inputs.tree.control_nodes}) == 1
    np.testing.assert_allclose(inputs.scenarios.auction_prices_eur_per_mwh["DA"], 40.0)
    result = solve_checked(inputs)
    np.testing.assert_allclose(result.arrays["imbalance_mwh"], 0.0, atol=1e-8)
    np.testing.assert_allclose(result.arrays["curtailment_mw"][:, 0], [0, 1, 1], atol=1e-8)
    assert result.objective_eur == pytest.approx(0.0, abs=1e-8)


def test_imbalance_penalty_enters_average_objective():
    inputs = scenario_inputs(exact=False)
    result = solve_checked(inputs)
    np.testing.assert_allclose(result.arrays["net_position_mwh"][:, 0], 1.0, atol=1e-8)
    np.testing.assert_allclose(result.arrays["imbalance_mwh"][:, 0], [-1, 0, 0], atol=1e-8)
    assert result.objective_eur == pytest.approx(40.0 - 100.0 / 3)


def test_exact_delivery_detects_infeasible_locked_commitment():
    pytest.importorskip("gurobipy")
    inputs = replace(without_new_trades(scenario_inputs(exact=True)), pending_position_mwh=(0.5,))
    built = build_model(inputs)
    try:
        result = solve_model(built, SolverSettings())
    finally:
        built.model.dispose()
    assert result.status == "infeasible"
