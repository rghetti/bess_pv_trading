"""Safe pointwise PV reduction for exact, shared-dispatch formulations."""
from dataclasses import replace

import numpy as np

from ..domain import InformationTree, OptimizationInput, OptimizationResult
from ..transaction_fees import scenario_fee_details


def reduce_exact_pv(inputs: OptimizationInput) -> OptimizationInput | None:
    """Return an equivalent envelope input, or None when equivalence is not assured."""
    bundle, tree = inputs.scenarios, inputs.tree
    pv = np.asarray(bundle.pv_ac_mw)
    count, intervals = pv.shape
    if count < 2 or inputs.optimize_imbalances:
        return None
    if any(len(set(owners)) != 1 for owners in tree.bid_nodes.values()):
        return None
    if any(len({owners[t] for owners in tree.control_nodes}) != 1 for t in range(intervals)):
        return None
    if any(not np.array_equal(prices, np.broadcast_to(prices[0], prices.shape))
           for prices in bundle.auction_prices_eur_per_mwh.values()):
        return None
    # When one curtailment decision is shared, it must not be assigned to
    # different observed PV levels. Otherwise the original feasible set can
    # impose extra equalities absent from the minimum-envelope formulation.
    owners = tree.curtailment_nodes or tree.control_nodes
    for t in range(intervals):
        observed = {}
        for s in range(count):
            owner, value = owners[s][t], pv[s, t]
            if owner in observed and observed[owner] != value:
                return None
            observed[owner] = value
    reduced = replace(bundle, scenario_ids=('pointwise_minimum',), probabilities=np.ones(1),
        pv_ac_mw=pv.min(axis=0, keepdims=True), auction_prices_eur_per_mwh={key: prices[:1]
            for key, prices in bundle.auction_prices_eur_per_mwh.items()},
        imbalance_prices_eur_per_mwh=bundle.imbalance_prices_eur_per_mwh[:1],
        imbalance_negative_prices_eur_per_mwh=(None if bundle.imbalance_negative_prices_eur_per_mwh is None
            else bundle.imbalance_negative_prices_eur_per_mwh[:1]))
    # Preserve the common decision owners and their original keys even when
    # they lie below the root. A single leaf uses its entire ancestor path.
    leaf = tree.leaf_nodes[0]
    path = set(tree.paths()[leaf])
    nodes = tuple(replace(node, conditional_probability=1.0) for node in tree.nodes if node.node_id in path)
    reduced_tree = InformationTree(nodes, (leaf,),
        {auction: (owners[0],) for auction, owners in tree.bid_nodes.items()},
        (tree.control_nodes[0],),
        None if tree.curtailment_nodes is None else (tree.curtailment_nodes[0],),
        execution_convention=tree.execution_convention)
    return replace(inputs, scenarios=reduced, tree=reduced_tree)


def lift_envelope_result(original: OptimizationInput, result: OptimizationResult) -> None:
    """Restore each scenario's physical curtailment and existing public result shapes."""
    count = len(original.scenarios.scenario_ids)
    result.arrays = {key: np.repeat(values, count, axis=0) for key, values in result.arrays.items()}
    result.arrays['curtailment_mw'] = original.scenarios.pv_ac_mw - result.arrays['pv_injection_mw']
    result.components = {key: np.repeat(values, count, axis=0) for key, values in result.components.items()}
    if original.transaction_fees.mode != 'none':
        result.diagnostics['transaction_fees_by_scenario'] = scenario_fee_details(original, result.trades)
