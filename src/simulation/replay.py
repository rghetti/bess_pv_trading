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
    # Condition an explicit finite tree using only already published prices when `observed_prices`.
    if observed_prices:
        # Enforce the input contract: Conditioning observations requires an aware decision_cutoff.
        if decision_cutoff is None or decision_cutoff.tzinfo is None:
            raise ValueError("Conditioning observations requires an aware decision_cutoff")
        # Condition an explicit finite tree using only already published prices for each `auction` in `observed_prices`.
        for auction in observed_prices:
            key = f"auction:{auction}"
            available = [node.observation_available_at[key] for node in inputs.tree.nodes
                         if key in node.observation_available_at]
            # Enforce the input contract: Cannot condition on unpublished `auction` prices.
            if not available or min(available) > decision_cutoff:
                raise ValueError(f"Cannot condition on unpublished {auction} prices")
    keep = np.ones(len(scenario.scenario_ids), dtype=bool)
    # Condition an explicit finite tree using only already published prices for each `(auction, observed)` in `observed_prices.items()`.
    for auction, observed in observed_prices.items():
        values = scenario.auction_prices_eur_per_mwh[auction]
        keep &= np.all(np.isclose(values, np.asarray(observed)[None, :], atol=1e-9), axis=1)
    # When `oracle_leaf is not None`, set `keep` using `np.arange(len(keep)) == oracle_leaf`.
    if oracle_leaf is not None:
        keep &= np.arange(len(keep)) == oracle_leaf
    indexes = np.flatnonzero(keep)
    # Enforce the input contract: Observed results lie outside the supplied finite scenario tree.
    if not len(indexes):
        raise ValueError("Observed results lie outside the supplied finite scenario tree")
    weights = scenario.probabilities[indexes].copy()
    # Enforce the input contract: Observed history has zero probability in the supplied tree.
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
                     dispatch_policy: str = "follow_battery_plan",
                     grid_target_mw: float | None = None,
                     ) -> tuple[PhysicalState, dict[str, Any]]:
    """Execute pre-interval commands with declared interval-average protection.

    The final average PV is used only by physical feedback/protection and ex-post
    settlement. It is never given to the planner as a pre-interval observation.
    This is an idealized interval simulator, not a hardware telemetry model.
    """
    # Enforce the input contract: Execution state timestamp differs from interval start.
    if state.timestamp != start:
        raise ValueError("Execution state timestamp differs from interval start")
    # Enforce the input contract: Execution interval must have positive duration.
    if end <= start:
        raise ValueError("Execution interval must have positive duration")
    # Enforce the input contract: Unknown execution dispatch_policy.
    if dispatch_policy not in {"follow_battery_plan", "track_committed_grid"}:
        raise ValueError("Unknown execution dispatch_policy")
    # Enforce the input contract: Execution grid target must be finite.
    if grid_target_mw is not None and not np.isfinite(grid_target_mw):
        raise ValueError("Execution grid target must be finite")
    # Enforce the input contract: Grid tracking requires a committed grid_target_mw.
    if dispatch_policy == "track_committed_grid" and grid_target_mw is None:
        raise ValueError("Grid tracking requires a committed grid_target_mw")
    magnitudes = (requested_charge_mw, requested_discharge_mw,
                  requested_curtailment_mw, degradation_eur_per_mwh)
    # Enforce the input contract: Execution commands and degradation must be finite and nonnegative.
    if any(not np.isfinite(value) or value < 0 for value in magnitudes):
        raise ValueError("Execution commands and degradation must be finite and nonnegative")
    # Enforce the input contract: Execution needs eligible nonnegative AC availability.
    if not np.isfinite(pv_ac_mw) or pv_ac_mw < 0:
        raise ValueError("Execution needs eligible nonnegative AC availability")
    # Enforce the input contract: Specify degradation_settings or a legacy throughput coefficient, not both.
    if degradation_settings is not None and degradation_eur_per_mwh:
        raise ValueError("Specify degradation_settings or a legacy throughput coefficient, not both")
    aging_settings = degradation_settings or DegradationSettings(degradation_eur_per_mwh)
    battery = plant.battery
    # Enforce the input contract: Execution starts outside battery energy bounds.
    if not battery.energy_min_mwh <= state.energy_mwh <= battery.maximum_energy:
        raise ValueError("Execution starts outside battery energy bounds")
    duration = (end - start).total_seconds() / 3600
    retained = state.energy_mwh * (1 - battery.self_discharge_per_hour) ** duration
    charge = max(0.0, min(requested_charge_mw, battery.max_power_mw,
        max(0.0, battery.maximum_energy - retained) / (duration * battery.eta_charge)))
    discharge = max(0.0, min(requested_discharge_mw, battery.max_power_mw,
        max(0.0, retained - battery.energy_min_mwh) * battery.eta_discharge / duration))
    # Enforce the input contract: Execution cannot charge and discharge simultaneously.
    if charge > 1e-8 and discharge > 1e-8:
        raise ValueError("Execution cannot charge and discharge simultaneously")
    curtailment = min(pv_ac_mw, max(0.0, requested_curtailment_mw))
    safety_adjustment = (abs(charge-requested_charge_mw) + abs(discharge-requested_discharge_mw)
                         + abs(curtailment-requested_curtailment_mw))
    bounded = (charge, discharge, curtailment)
    # When `dispatch_policy == 'track_committed_grid'`, set `available_grid` using `pv_ac_mw - curtailment + discharge - charge`.
    if dispatch_policy == "track_committed_grid":
        # Release scheduled curtailment before spending additional stored energy.
        available_grid = pv_ac_mw - curtailment + discharge - charge
        curtailment -= min(curtailment, max(0.0, grid_target_mw-available_grid))
        required_battery = grid_target_mw - pv_ac_mw + curtailment
        charge_limit = min(battery.max_power_mw,
            max(0.0, battery.maximum_energy-retained) / (duration*battery.eta_charge))
        discharge_limit = min(battery.max_power_mw,
            max(0.0, retained-battery.energy_min_mwh)*battery.eta_discharge / duration)
        actual_battery = float(np.clip(required_battery, -charge_limit, discharge_limit))
        charge, discharge = max(0.0, -actual_battery), max(0.0, actual_battery)
        # Surplus PV first changes battery power, then spills if charging is limited.
        excess = pv_ac_mw-curtailment+actual_battery-grid_target_mw
        curtailment += min(pv_ac_mw-curtailment, max(0.0, excess))
    feedback_adjustment = sum(abs(a-b) for a,b in zip((charge,discharge,curtailment),bounded))
    before_poi = (charge, discharge, curtailment)
    grid = pv_ac_mw - curtailment + discharge - charge
    # Reduce a command before curtailing/exporting beyond the connection.
    if grid < -plant.poi_import_mw:
        reduction = min(charge, -plant.poi_import_mw - grid)
        charge -= reduction
        grid += reduction
    # When `grid > plant.poi_export_mw`, set `reduction` using `min(discharge, grid - plant.poi_export_mw)`.
    if grid > plant.poi_export_mw:
        reduction = min(discharge, grid - plant.poi_export_mw)
        discharge -= reduction
        grid -= reduction
        curtailment += max(0.0, grid - plant.poi_export_mw)
        grid = min(grid, plant.poi_export_mw)
    safety_adjustment += sum(abs(a-b) for a,b in zip((charge,discharge,curtailment),before_poi))
    energy = retained + battery.eta_charge * charge * duration - discharge * duration / battery.eta_discharge
    # Enforce the input contract: Execution would violate battery energy bounds.
    if energy < battery.energy_min_mwh-1e-8 or energy > battery.maximum_energy+1e-8:
        raise ValueError("Execution would violate battery energy bounds")
    energy = min(battery.maximum_energy, max(battery.energy_min_mwh, energy))
    throughput = (charge + discharge) * duration
    next_state, aging = advance_degradation_state(state=state, battery=battery,
        settings=aging_settings, end=end, energy_mwh=energy,
        charge_mw=charge, discharge_mw=discharge)
    record = {"pv_ac_mw": pv_ac_mw, "pv_injection_mw": pv_ac_mw - curtailment,
        "charge_mw": charge,
        "discharge_mw": discharge, "curtailment_mw": curtailment,
        "grid_mw": grid, "energy_start_mwh": state.energy_mwh,
        "energy_end_mwh": energy, "throughput_mwh": throughput,
        "degradation_eur": aging,
        "requested_charge_mw": requested_charge_mw,
        "requested_discharge_mw": requested_discharge_mw,
        "requested_curtailment_mw": requested_curtailment_mw,
        "dispatch_policy": dispatch_policy, "grid_target_mw": grid_target_mw,
        "grid_tracking_error_mw": None if grid_target_mw is None else grid-grid_target_mw,
        "feedback_adjustment_mw": feedback_adjustment,
        "safety_adjustment_mw": safety_adjustment}
    # When `next_state.cycle_segment_energy_mwh is not None`, set `record['cycle_segment_energy_end_mwh']` using `next_state.cycle_segment_energy_mwh`.
    if next_state.cycle_segment_energy_mwh is not None:
        record["cycle_segment_energy_end_mwh"] = next_state.cycle_segment_energy_mwh
    return next_state, record
