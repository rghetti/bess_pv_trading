"""Xu et al. segmented cycle aging: Eqs. (5), (7), (8)--(15), (37)--(40).

Reference: https://arxiv.org/html/1707.04567v2 . Segments track cycle-depth
memory, not fixed positions in the battery's absolute state-of-charge range.
Both charging and discharging prioritize the shallowest available segment.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Mapping

import numpy as np

from ..domain import (BatteryParameters, DegradationSettings, OptimizationInput,
                      PhysicalState, battery_parameters_from_config)


def degradation_settings_from_config(config: Mapping[str, Any]) -> DegradationSettings:
    """Parse either a complete application config or its degradation section."""
    section = dict(config.get("degradation", config))
    model = section.pop("model", "throughput")
    mode = {"throughput": "constant_throughput", "constant_throughput": "constant_throughput",
            "xu_segmented": "xu_segmented"}.get(model)
    if mode is None:
        raise ValueError(f"Unknown degradation model {model!r}")
    # Published pack prices and reference cycle life can determine the linear
    # wear rate. Keep their source metadata in YAML rather than solver settings.
    economics = section.pop("replacement_economics", None)
    lifetime = section.pop("throughput_lifetime", None)
    if economics is not None or lifetime is not None:
        if "battery" not in config:
            raise ValueError("Replacement/lifetime degradation requires the full battery configuration")
        battery = battery_parameters_from_config(config["battery"])
        if battery.capacity_mwh <= 0:
            raise ValueError("Replacement/lifetime degradation requires positive battery capacity")
    if economics is not None:
        if section.get("replacement_cost_eur") is not None:
            raise ValueError("Specify replacement_economics or replacement_cost_eur, not both")
        price = float(economics["pack_price_usd_per_kwh"])
        fx = float(economics["usd_per_eur"])
        additional = float(economics["additional_replacement_eur_per_kwh"])
        salvage = float(economics["salvage_fraction"])
        if (not all(np.isfinite(value) for value in (price, fx, additional, salvage))
                or price <= 0 or fx <= 0 or additional < 0 or not 0 <= salvage < 1):
            raise ValueError("Invalid replacement pack price, FX, additional cost or salvage fraction")
        section["replacement_cost_eur"] = (
            (price / fx + additional) * battery.capacity_mwh * 1000 * (1 - salvage))
    if lifetime is not None:
        if mode != "constant_throughput":
            raise ValueError("throughput_lifetime is only supported by throughput degradation")
        if section.get("cost_eur_per_mwh_throughput") is not None:
            raise ValueError("Specify throughput_lifetime or a direct throughput cost, not both")
        cycles = float(lifetime["cycle_life"])
        depth = float(lifetime["reference_depth_of_discharge"])
        replacement = section.get("replacement_cost_eur")
        if (not np.isfinite(cycles) or cycles <= 0 or not np.isfinite(depth)
                or not 0 < depth <= 1 or replacement is None
                or not np.isfinite(replacement) or replacement <= 0):
            raise ValueError("Throughput lifetime requires positive cycles/replacement cost and 0<DoD<=1")
        # A reference internal-energy cycle consumes E*DoD/eta_charge AC MWh
        # and delivers E*DoD*eta_discharge AC MWh; charge both throughputs once.
        lifetime_ac_mwh = (cycles * depth * battery.capacity_mwh
                           * (1 / battery.eta_charge + battery.eta_discharge))
        section["cost_eur_per_mwh_throughput"] = float(replacement) / lifetime_ac_mwh
    return DegradationSettings(mode=mode, **section)


def xu_segment_parameters(battery: BatteryParameters, settings: DegradationSettings) -> tuple[np.ndarray, np.ndarray]:
    """Return internal MWh capacities and Eq. (5) EUR per AC discharged MWh.

    E_rate is nameplate internal energy, including energy outside operational SoC
    limits. Eta_dis belongs in both the AC marginal coefficient and energy balance.
    """
    if settings.mode != "xu_segmented" or battery.capacity_mwh <= 0:
        raise ValueError("Xu segment parameters require Xu mode and positive battery capacity")
    if battery.self_discharge_per_hour:
        raise ValueError("Xu segmented model currently requires zero self discharge")
    depth = np.linspace(0.0, 1.0, settings.segment_count + 1)
    stress = settings.stress_coefficient * depth ** settings.stress_exponent
    capacities = np.full(settings.segment_count, battery.capacity_mwh / settings.segment_count)
    marginal = float(settings.replacement_cost_eur) * np.diff(stress) / (battery.eta_discharge * capacities)
    if not np.all(np.isfinite(marginal)) or np.any(np.diff(marginal) < -1e-10):
        raise ValueError("Xu marginal cycle aging costs must be finite and nondecreasing")
    return capacities, marginal


def initial_segment_energy(state: PhysicalState, battery: BatteryParameters,
                           settings: DegradationSettings) -> np.ndarray:
    """Use carried segment memory, or initialize once by Xu appendix Eq. (39)."""
    capacities, _ = xu_segment_parameters(battery, settings)
    if state.cycle_segment_energy_mwh is not None:
        values = np.asarray(state.cycle_segment_energy_mwh, dtype=float)
        if values.shape != capacities.shape or np.any(values > capacities + 1e-8):
            raise ValueError("Xu segment energy shape or capacity differs from configured segments")
        return values.copy()
    if state.cumulative_throughput_mwh or state.cumulative_degradation_eur or state.origin != "known_delivery_start_assumption":
        raise ValueError("A carried Xu state requires cycle_segment_energy_mwh; aging history cannot reset")
    # Initial charge is assigned to shallowest segments; subsequent discharge
    # empties these first, leaving higher-index segments occupied.
    return np.clip(state.energy_mwh - np.r_[0.0, np.cumsum(capacities[:-1])], 0.0, capacities)


def evaluate_xu_dispatch(*, battery: BatteryParameters, settings: DegradationSettings,
                        state: PhysicalState, charge_mw: np.ndarray,
                        discharge_mw: np.ndarray, delta_hours: np.ndarray) -> dict[str, Any]:
    """Replay the appendix's greedy policy numerically, without a solver.

    Accept one actual trajectory. Charge-only half cycles incur zero cycle aging;
    discharge half cycles incur the full same-depth cycle cost, as in Eq. (43).
    """
    charge, discharge, duration = (np.asarray(value, dtype=float) for value in
                                  (charge_mw, discharge_mw, delta_hours))
    if charge.ndim != 1 or charge.shape != discharge.shape or charge.shape != duration.shape:
        raise ValueError("Xu dispatch and durations must be matching one-dimensional vectors")
    if not all(np.all(np.isfinite(value)) for value in (charge, discharge, duration)) or np.any(charge < -1e-8) or np.any(discharge < -1e-8) or np.any(duration <= 0):
        raise ValueError("Xu dispatch requires finite nonnegative powers and positive durations")
    if np.any(np.minimum(charge, discharge) > 1e-8):
        raise ValueError("Xu dispatch cannot charge and discharge simultaneously")
    capacities, costs = xu_segment_parameters(battery, settings)
    energy = np.empty((len(charge) + 1, settings.segment_count))
    energy[0] = initial_segment_energy(state, battery, settings)
    segment_charge = np.zeros((len(charge), settings.segment_count))
    segment_discharge = np.zeros_like(segment_charge)
    # Internal energy packets traverse available shallow segments first, matching
    # Eq. (37)--(40). Cost is computed from their AC discharge equivalent.
    for t, hours in enumerate(duration):
        amount_in = max(0.0, charge[t]) * battery.eta_charge * hours
        amount_out = max(0.0, discharge[t]) * hours / battery.eta_discharge
        energy[t + 1] = energy[t]
        for j in range(settings.segment_count):
            stored = min(amount_in, max(0.0, capacities[j] - energy[t, j]))
            removed = min(amount_out, max(0.0, energy[t, j]))
            energy[t + 1, j] += stored - removed
            segment_charge[t, j] = stored / (battery.eta_charge * hours)
            segment_discharge[t, j] = removed * battery.eta_discharge / hours
            amount_in -= stored
            amount_out -= removed
        if max(amount_in, amount_out) > 1e-7:
            raise ValueError("Xu dispatch exceeds available segment energy or headroom")
    interval_cost = np.sum(segment_discharge * costs[None, :] * duration[:, None], axis=1)
    return {"segment_energy_mwh": energy, "segment_charge_mw": segment_charge,
            "segment_discharge_mw": segment_discharge,
            "interval_degradation_eur": interval_cost,
            "degradation_eur": float(interval_cost.sum())}


def rainflow_discharge_depths(energy_mwh: np.ndarray, capacity_mwh: float) -> np.ndarray:
    """Independently count full and discharge-residue depths by paper Section II-C.

    This benchmark uses only turning points of aggregate energy, with no segment
    state. It neglects pre-history, so use it for an explicitly initialized trace.
    """
    values = np.asarray(energy_mwh, dtype=float)
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)) or not np.isfinite(capacity_mwh) or capacity_mwh <= 0:
        raise ValueError("Rainflow input requires finite one-dimensional energy and positive capacity")
    points = values[np.r_[True, np.diff(values) != 0]]
    if len(points) > 2:
        keep = np.r_[True, np.diff(points)[:-1] * np.diff(points)[1:] < 0, True]
        points = points[keep]
    residue = list(points)
    depths = []
    index = 0
    while index + 3 < len(residue):
        left, middle, right = (abs(residue[index + j + 1] - residue[index + j]) for j in range(3))
        if middle <= left and middle <= right:
            depths.append(middle / capacity_mwh)
            del residue[index + 1:index + 3]
            index = max(0, index - 2)
        else:
            index += 1
    depths.extend((left - right) / capacity_mwh for left, right in zip(residue, residue[1:]) if left > right)
    return np.asarray(depths)


def advance_degradation_state(*, state: PhysicalState, battery: BatteryParameters,
                              settings: DegradationSettings, end: datetime,
                              energy_mwh: float, charge_mw: float, discharge_mw: float,
                              origin: str = "simulated_carried_state") -> tuple[PhysicalState, float]:
    """Carry realized or committed-command aging without rebuilding cycle history."""
    hours = (end - state.timestamp).total_seconds() / 3600
    if hours <= 0:
        raise ValueError("Degradation state must advance over a positive interval")
    throughput = hours * (charge_mw + discharge_mw)
    segments = None
    if settings.mode == "xu_segmented":
        trajectory = evaluate_xu_dispatch(battery=battery, settings=settings, state=state,
            charge_mw=np.array([charge_mw]), discharge_mw=np.array([discharge_mw]),
            delta_hours=np.array([hours]))
        segments = tuple(float(value) for value in trajectory["segment_energy_mwh"][-1])
        aging = trajectory["degradation_eur"]
    else:
        if state.cycle_segment_energy_mwh is not None:
            raise ValueError("Xu cycle segment history cannot be silently discarded by a throughput model")
        aging = throughput * settings.cost_eur_per_mwh_throughput
    return PhysicalState(end, energy_mwh, state.cumulative_throughput_mwh + throughput,
                         state.cumulative_degradation_eur + aging, origin, segments), aging


def build_xu_degradation(model: Any, inputs: OptimizationInput,
                         variables: dict[str, Any], gp: Any) -> dict[int, Any]:
    """Build node-owned segment flows and canonical priority constraints.

    Eqs. (8)--(15) provide the segmented balance. Explicit priority binaries select
    the optimal greedy representative in Theorem 1, including when future cost is
    indifferent. This makes state propagation across residual solves unambiguous.
    """
    battery, settings = inputs.plant.battery, inputs.degradation
    capacity, marginal = xu_segment_parameters(battery, settings)
    initial = initial_segment_energy(inputs.state, battery, settings)
    count, intervals = np.shape(inputs.scenarios.pv_ac_mw)
    charge, discharge, energy = {}, {}, {}
    charge_priority, discharge_priority = {}, {}
    for t in range(intervals):
        for owner in sorted({row[t] for row in inputs.tree.control_nodes}):
            for j in range(settings.segment_count):
                suffix = f"n{owner},t{t},j{j}"
                charge[t, owner, j] = model.addVar(ub=battery.max_power_mw, name=f"xu_charge[{suffix}]")
                discharge[t, owner, j] = model.addVar(ub=battery.max_power_mw, name=f"xu_discharge[{suffix}]")
            model.addConstr(gp.quicksum(charge[t, owner, j] for j in range(settings.segment_count)) == variables["charge"][t, owner], name=f"xu_charge_sum[n{owner},t{t}]")
            model.addConstr(gp.quicksum(discharge[t, owner, j] for j in range(settings.segment_count)) == variables["discharge"][t, owner], name=f"xu_discharge_sum[n{owner},t{t}]")
            for j in range(settings.segment_count - 1):
                suffix = f"n{owner},t{t},j{j}"
                charge_priority[t, owner, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"xu_charge_deeper[{suffix}]")
                discharge_priority[t, owner, j] = model.addVar(vtype=gp.GRB.BINARY, name=f"xu_discharge_deeper[{suffix}]")
                # If any deeper segment is used, this segment must finish full
                # when charging, or empty when discharging. Tails prevent skips.
                model.addConstr(gp.quicksum(charge[t, owner, k] for k in range(j + 1, settings.segment_count)) <= battery.max_power_mw * charge_priority[t, owner, j], name=f"xu_charge_priority[{suffix}]")
                model.addConstr(gp.quicksum(discharge[t, owner, k] for k in range(j + 1, settings.segment_count)) <= battery.max_power_mw * discharge_priority[t, owner, j], name=f"xu_discharge_priority[{suffix}]")
    for s in range(count):
        for t in range(intervals + 1):
            for j in range(settings.segment_count):
                energy[s, t, j] = model.addVar(ub=float(capacity[j]), name=f"xu_energy[s{s},boundary{t},j{j}]")
            model.addConstr(gp.quicksum(energy[s, t, j] for j in range(settings.segment_count)) == variables["energy"][s, t], name=f"xu_energy_sum[s{s},boundary{t}]")
        for j in range(settings.segment_count):
            model.addConstr(energy[s, 0, j] == float(initial[j]), name=f"xu_initial[s{s},j{j}]")
        for t, duration in enumerate(inputs.grid.delta_hours):
            owner = inputs.tree.control_nodes[s][t]
            for j in range(settings.segment_count):
                suffix = f"s{s},t{t},j{j}"
                model.addConstr(energy[s, t + 1, j] == energy[s, t, j] + duration * (battery.eta_charge * charge[t, owner, j] - discharge[t, owner, j] / battery.eta_discharge), name=f"xu_balance[{suffix}]")
                if j < settings.segment_count - 1:
                    model.addConstr(energy[s, t + 1, j] >= float(capacity[j]) * charge_priority[t, owner, j], name=f"xu_fill_first[{suffix}]")
                    model.addConstr(energy[s, t + 1, j] <= float(capacity[j]) * (1 - discharge_priority[t, owner, j]), name=f"xu_empty_first[{suffix}]")
    variables.update(xu_segment_charge=charge, xu_segment_discharge=discharge, xu_segment_energy=energy)
    return {s: gp.quicksum(float(marginal[j]) * discharge[t, inputs.tree.control_nodes[s][t], j] * duration
                          for t, duration in enumerate(inputs.grid.delta_hours)
                          for j in range(settings.segment_count)) for s in range(count)}
