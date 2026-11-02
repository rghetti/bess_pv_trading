"""Exact-PV envelope reduction keeps the original scenario result contract."""
from dataclasses import replace

import numpy as np
import pytest

from battery_case_config import configured_battery
from bess_pv_trading.domain import InformationNode, SolverSettings
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.optimization.pv_envelope import reduce_exact_pv
from bess_pv_trading.scenarios import shared_dispatch_pv_tree
from test_optimization_contracts import example_inputs
from test_transaction_fees import fee_settings
from test_xu_solver import settings as xu_settings


def sample():
    base = example_inputs(prices=[[40., -10., 80.]], pv=np.ones((1, 3)),
                          battery=configured_battery('unit'), hours=1.)
    pv = np.array([[0.3, 0.7, 0.6], [0.6, 0.2, 0.5], [0.4, 0.4, 0.1]])
    bundle, tree = shared_dispatch_pv_tree(base.scenarios, base.tree, base.grid, pv, ('det', 'w1', 'w2'))
    return replace(base, scenarios=bundle, tree=tree, state=replace(base.state, energy_mwh=0.5),
                   optimize_imbalances=False, terminal_target_mwh=None, terminal_minimum_mwh=0.2,
                   auction_eligibility={'DA': (False, True, True)}, pending_position_mwh=(0.2, 0., 0.))


def test_safe_reduction_preserves_physical_and_market_inputs():
    original = sample()
    reduced = reduce_exact_pv(original)
    assert reduced is not None
    np.testing.assert_allclose(reduced.scenarios.pv_ac_mw, [[0.3, 0.2, 0.1]])
    assert reduced.state == original.state
    assert reduced.market == original.market
    assert reduced.pending_position_mwh == original.pending_position_mwh
    assert reduced.auction_eligibility == original.auction_eligibility
    np.testing.assert_array_equal(reduced.scenarios.auction_prices_eur_per_mwh['DA'], [[40, -10, 80]])


def test_inapplicable_formulations_remain_unreduced():
    original = sample()
    assert reduce_exact_pv(replace(original, optimize_imbalances=True)) is None
    different_prices = original.scenarios.auction_prices_eur_per_mwh['DA'].copy()
    different_prices[1, 0] += 1
    bundle = replace(original.scenarios, auction_prices_eur_per_mwh={'DA': different_prices})
    assert reduce_exact_pv(replace(original, scenarios=bundle)) is None
    # Shared curtailment across different PV levels adds constraints the
    # one-trajectory projection does not capture.
    tree = replace(original.tree, curtailment_nodes=original.tree.control_nodes)
    assert reduce_exact_pv(replace(original, tree=tree)) is None
    divergent_bids = replace(original.tree, bid_nodes={'DA': ('root', *original.tree.leaf_nodes[1:])})
    assert reduce_exact_pv(replace(original, tree=divergent_bids)) is None
    base = example_inputs()
    assert reduce_exact_pv(base) is None


def test_zero_probability_member_still_limits_envelope():
    original = sample()
    probabilities = np.array([0., 0.5, 0.5])
    bundle = replace(original.scenarios, probabilities=probabilities)
    nodes = (original.tree.nodes[0], *(replace(node, conditional_probability=probability)
             for node, probability in zip(original.tree.nodes[1:], probabilities)))
    tree = replace(original.tree, nodes=nodes)
    reduced = reduce_exact_pv(replace(original, scenarios=bundle, tree=tree))
    assert reduced.scenarios.pv_ac_mw[0, 0] == pytest.approx(0.3)


def test_shared_nonroot_decision_owners_are_preserved(licensed):
    original = sample()
    root = original.tree.nodes[0]
    shared = InformationNode('shared', root.node_id, root.cutoff)
    leaves = tuple(replace(node, parent_id=shared.node_id) for node in original.tree.nodes[1:])
    tree = replace(original.tree, nodes=(root, shared, *leaves),
                   bid_nodes={'DA': ('shared',) * 3},
                   control_nodes=(('shared',) * 3,) * 3)
    original = replace(original, tree=tree)
    reduced = reduce_exact_pv(original)
    assert reduced.tree.bid_nodes['DA'] == ('shared',)
    assert reduced.tree.control_nodes == (('shared',) * 3,)
    assert {node.node_id for node in reduced.tree.nodes} == {'root', 'shared', tree.leaf_nodes[0]}
    objectives = []
    for reduction in (False, True):
        built = build_model(original, reduce_exact_pv_scenarios=reduction)
        try:
            result = solve_model(built, SolverSettings(threads=1, output_flag=False))
        finally:
            built.model.dispose()
        assert result.status == 'optimal'
        assert validate_solution(original, result).valid
        assert all(node == 'shared' for _, node, _ in result.trades)
        objectives.append(result.objective_eur)
    assert objectives[0] == pytest.approx(objectives[1], abs=1e-6)


@pytest.fixture
def licensed():
    gp = pytest.importorskip('gurobipy')
    try:
        with gp.Model():
            pass
    except gp.GurobiError as error:
        pytest.skip(str(error))


@pytest.mark.parametrize('kind', ['throughput', 'xu', 'fee'])
def test_lifted_solution_matches_full_objective_and_validates(licensed, kind):
    original = sample()
    if kind == 'xu':
        original = replace(original, degradation=xu_settings())
    if kind == 'fee':
        cutoff = original.tree.nodes[0].cutoff
        fees = fee_settings('flat', as_of=cutoff.isoformat())
        original = replace(original, transaction_fees=fees, auction_submission_times={'DA': cutoff})
    results = []
    for reduction in (False, True):
        built = build_model(original, reduce_exact_pv_scenarios=reduction)
        try:
            result = solve_model(built, SolverSettings(threads=1, output_flag=False))
        finally:
            built.model.dispose()
        assert result.status == 'optimal'
        report = validate_solution(original, result)
        assert report.valid, report.violations
        results.append(result)
    full, reduced = results
    assert reduced.objective_eur == pytest.approx(full.objective_eur, abs=1e-6)
    assert reduced.diagnostics['pv_scenario_formulation'] == 'pointwise_minimum'
    assert reduced.diagnostics['original_pv_scenario_count'] == 3
    assert reduced.diagnostics['modeled_pv_scenario_count'] == 1
    assert full.diagnostics['modeled_pv_scenario_count'] == 3
    np.testing.assert_allclose(reduced.arrays['curtailment_mw'] + reduced.arrays['pv_injection_mw'],
                               original.scenarios.pv_ac_mw, atol=1e-7)
    np.testing.assert_allclose(reduced.arrays['pv_injection_mw'],
                               np.broadcast_to(reduced.arrays['pv_injection_mw'][0], (3, 3)))
    if kind == 'fee':
        assert len(reduced.diagnostics['transaction_fees_by_scenario']) == 3


def test_infeasible_fixed_position_stays_infeasible(licensed):
    original = sample()
    original = replace(original, plant=replace(original.plant, battery=configured_battery('no_bess')),
                       state=replace(original.state, energy_mwh=0.), terminal_minimum_mwh=None,
                       auction_eligibility={'DA': (False, False, False)},
                       pending_position_mwh=(1., 0., 0.))
    for reduction in (False, True):
        built = build_model(original, reduce_exact_pv_scenarios=reduction)
        try:
            result = solve_model(built, SolverSettings(threads=1, output_flag=False))
        finally:
            built.model.dispose()
        assert result.status == 'infeasible'
        assert not result.has_incumbent
        assert result.diagnostics['original_pv_scenario_count'] == 3
