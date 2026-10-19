"""Causal residual planning, immutable trade accounting, and AC execution."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

import numpy as np

from bess_pv_trading.domain import (
    DegradationSettings, DeliveryGrid, InformationTree, MarketState, OptimizationInput,
    PhysicalState, ScenarioBundle,
)
from bess_pv_trading.optimization.degradation import advance_degradation_state


def condition_and_slice(inputs: OptimizationInput, *, first_interval: int,
                        state: PhysicalState, market: MarketState,
                        observed_prices: Mapping[str, np.ndarray],
                        open_auctions: tuple[str, ...], decision_cutoff=None,
                        oracle_leaf: int | None = None,
                        ) -> OptimizationInput:
    """Condition an explicit finite tree using only already published prices.

    The synthetic tree branches on observable result vectors. Conditioning removes
    inconsistent histories and renormalizes each surviving subtree's mass; it
    never creates information from latent future paths.
    """
    scenario = inputs.scenarios
    if observed_prices:
        if decision_cutoff is None or decision_cutoff.tzinfo is None:
            raise ValueError("Conditioning observations requires an aware decision_cutoff")
        for auction in observed_prices:
            key = f"auction:{auction}"
            available = [node.observation_available_at[key] for node in inputs.tree.nodes
                         if key in node.observation_available_at]
            if not available or min(available) > decision_cutoff:
                raise ValueError(f"Cannot condition on unpublished {auction} prices")
    keep = np.ones(len(scenario.scenario_ids), dtype=bool)
    for auction, observed in observed_prices.items():
        values = scenario.auction_prices_eur_per_mwh[auction]
        keep &= np.all(np.isclose(values, np.asarray(observed)[None, :], atol=1e-9), axis=1)
    if oracle_leaf is not None:
        keep &= np.arange(len(keep)) == oracle_leaf
    indexes = np.flatnonzero(keep)
    if not len(indexes):
        raise ValueError("Observed results lie outside the supplied finite scenario tree")
    weights = scenario.probabilities[indexes].copy()
    if weights.sum() <= 0:
        raise ValueError("Observed history has zero probability in the supplied tree")
    weights /= weights.sum()
    paths = inputs.tree.paths()
    leaf_ids = tuple(inputs.tree.leaf_nodes[s] for s in indexes)
    surviving = set(item for leaf in leaf_ids for item in paths[leaf])
    mass = {node: sum(weight for leaf, weight in zip(leaf_ids, weights) if node in paths[leaf])
            for node in surviving}
    nodes = tuple(replace(node, conditional_probability=(1.0 if node.parent_id is None
                    else mass[node.node_id] / mass[node.parent_id]))
                  for node in inputs.tree.nodes if node.node_id in surviving)
    controls = tuple(inputs.tree.control_nodes[s][first_interval:] for s in indexes)
    curtail = (tuple(inputs.tree.curtailment_nodes[s][first_interval:] for s in indexes)
               if inputs.tree.curtailment_nodes is not None else None)
    tree = InformationTree(nodes, leaf_ids,
            {a: tuple(inputs.tree.bid_nodes[a][s] for s in indexes) for a in open_auctions},
            controls, curtail, inputs.tree.execution_convention)
    selection = np.ix_(indexes, np.arange(first_interval, len(inputs.grid.interval_ids)))
    prices = {a: values[selection] for a, values in scenario.auction_prices_eur_per_mwh.items()}
    conditional = ScenarioBundle(tuple(scenario.scenario_ids[s] for s in indexes), weights,
        scenario.pv_ac_mw[selection], prices, scenario.imbalance_prices_eur_per_mwh[selection],
        (scenario.imbalance_negative_prices_eur_per_mwh[selection]
         if scenario.imbalance_negative_prices_eur_per_mwh is not None else None),
        scenario.origin, {**scenario.provenance, "conditioned_on": list(observed_prices)})
    grid = DeliveryGrid(inputs.grid.interval_ids[first_interval:],
                        inputs.grid.starts[first_interval:], inputs.grid.ends[first_interval:])
    return replace(inputs, grid=grid, state=state, market=market, tree=tree,
                   scenarios=conditional,
                   pending_position_mwh=(tuple(inputs.pending_position_mwh[first_interval:])
                                         if inputs.pending_position_mwh is not None else None),
                   auction_eligibility={a: inputs.auction_eligibility[a][first_interval:]
                                        for a in open_auctions})


def execute_interval(*, state: PhysicalState, start, end, pv_ac_mw: float,
                     requested_charge_mw: float, requested_discharge_mw: float,
                     requested_curtailment_mw: float, plant,
                     degradation_eur_per_mwh: float = 0.0,
                     degradation_settings: DegradationSettings | None = None,
                     ) -> tuple[PhysicalState, dict[str, Any]]:
    """Execute pre-interval commands with declared interval-average protection.

    The final average PV is used only by the physical safety response and ex-post
    settlement. It is never given to the planner as a pre-interval observation.
    This is an idealized interval simulator, not a hardware telemetry model.
    """
    if state.timestamp != start:
        raise ValueError("Execution state timestamp differs from interval start")
    if end <= start:
        raise ValueError("Execution interval must have positive duration")
    magnitudes = (requested_charge_mw, requested_discharge_mw,
                  requested_curtailment_mw, degradation_eur_per_mwh)
    if any(not np.isfinite(value) or value < 0 for value in magnitudes):
        raise ValueError("Execution commands and degradation must be finite and nonnegative")
    if not np.isfinite(pv_ac_mw) or pv_ac_mw < 0:
        raise ValueError("Execution needs eligible nonnegative AC availability")
    if degradation_settings is not None and degradation_eur_per_mwh:
        raise ValueError("Specify degradation_settings or a legacy throughput coefficient, not both")
    aging_settings = degradation_settings or DegradationSettings(degradation_eur_per_mwh)
    battery = plant.battery
    if not battery.energy_min_mwh <= state.energy_mwh <= battery.maximum_energy:
        raise ValueError("Execution starts outside battery energy bounds")
    duration = (end - start).total_seconds() / 3600
    retained = state.energy_mwh * (1 - battery.self_discharge_per_hour) ** duration
    charge = max(0.0, min(requested_charge_mw, battery.max_power_mw,
        max(0.0, battery.maximum_energy - retained) / (duration * battery.eta_charge)))
    discharge = max(0.0, min(requested_discharge_mw, battery.max_power_mw,
        max(0.0, retained - battery.energy_min_mwh) * battery.eta_discharge / duration))
    if charge > 1e-8 and discharge > 1e-8:
        raise ValueError("Execution cannot charge and discharge simultaneously")
    curtailment = min(pv_ac_mw, max(0.0, requested_curtailment_mw))
    grid = pv_ac_mw - curtailment + discharge - charge
    # Reduce a command before curtailing/exporting beyond the connection.
    if grid < -plant.poi_import_mw:
        reduction = min(charge, -plant.poi_import_mw - grid)
        charge -= reduction
        grid += reduction
    if grid > plant.poi_export_mw:
        reduction = min(discharge, grid - plant.poi_export_mw)
        discharge -= reduction
        grid -= reduction
        curtailment += max(0.0, grid - plant.poi_export_mw)
        grid = min(grid, plant.poi_export_mw)
    energy = retained + battery.eta_charge * charge * duration - discharge * duration / battery.eta_discharge
    if energy < battery.energy_min_mwh-1e-8 or energy > battery.maximum_energy+1e-8:
        raise ValueError("Execution would violate battery energy bounds")
    energy = min(battery.maximum_energy, max(battery.energy_min_mwh, energy))
    throughput = (charge + discharge) * duration
    next_state, aging = advance_degradation_state(state=state, battery=battery,
        settings=aging_settings, end=end, energy_mwh=energy,
        charge_mw=charge, discharge_mw=discharge)
    record = {"pv_ac_mw": pv_ac_mw, "charge_mw": charge,
        "discharge_mw": discharge, "curtailment_mw": curtailment,
        "grid_mw": grid, "energy_start_mwh": state.energy_mwh,
        "energy_end_mwh": energy, "throughput_mwh": throughput,
        "degradation_eur": aging,
        "safety_adjustment_mw": abs(charge-requested_charge_mw)
            + abs(discharge-requested_discharge_mw)
            + abs(curtailment-requested_curtailment_mw)}
    if next_state.cycle_segment_energy_mwh is not None:
        record["cycle_segment_energy_end_mwh"] = next_state.cycle_segment_energy_mwh
    return next_state, record
