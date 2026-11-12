"""Linear cycle-throughput and calendar aging with daily committed health."""
from dataclasses import asdict, replace
from datetime import date, time
import hashlib
import json
from zoneinfo import ZoneInfo
import numpy as np

from ..domain import LinearHealthState, PhysicalState


def calibration_identity(settings):
    """Bind restored state to every physical aging calibration parameter."""
    raw = json.dumps(asdict(settings.linear_soh), sort_keys=True).encode()
    return settings.linear_soh.calibration_id + ":" + hashlib.sha256(raw).hexdigest()[:20]


def initialize_linear_health(state, battery, settings):
    """Initialize only fresh batteries; never infer used-battery history."""
    if battery.capacity_mwh <= 0 or battery.self_discharge_per_hour:
        raise ValueError("Linear SOH requires positive capacity and zero self-discharge")
    health = state.battery_health
    local_day = state.timestamp.astimezone(ZoneInfo(settings.linear_soh.timezone)).date().isoformat()
    if health is None:
        if state.cumulative_throughput_mwh or state.cumulative_degradation_eur or state.origin != "known_delivery_start_assumption":
            raise ValueError("Used linear SOH batteries require explicit health history")
        health = LinearHealthState(0., 0., 1., local_day, calibration_identity(settings))
    if not isinstance(health, LinearHealthState) or health.calibration_id != calibration_identity(settings):
        raise ValueError("Linear SOH history and calibration identity do not match")
    if health.active_day != local_day:
        raise ValueError("Linear SOH active day differs from state timestamp")
    if (state.timestamp.astimezone(ZoneInfo(settings.linear_soh.timezone)).time() == time.min
            and abs(health.committed_soh - (1 - health.calendar_loss - health.cycle_loss)) > 1e-8):
        raise ValueError("Linear SOH midnight state differs from accumulated losses")
    return health


def cycle_loss_per_internal_mwh(battery, settings):
    """Return fractional capacity loss per internal charge-plus-discharge MWh."""
    p = settings.linear_soh
    return p.cycle_capacity_loss_at_reference_life / (2 * p.cycle_life * p.reference_depth_of_discharge * battery.capacity_mwh)


def linear_wear_value(settings):
    """Value a fraction of original capacity using the configured EOL threshold."""
    return settings.replacement_cost_eur / (1 - settings.linear_soh.end_of_life_soh)


def linear_forecast(inputs, charge, discharge):
    """Independently reconstruct forecast capacity and wear from controls."""
    battery, settings = inputs.plant.battery, inputs.degradation
    health = initialize_linear_health(inputs.state, battery, settings)
    p = settings.linear_soh
    count, intervals = charge.shape
    capacity = np.empty((count, intervals + 1))
    capacity[:, 0] = battery.capacity_mwh * health.committed_soh
    cycle = np.zeros(count)
    calendar = 0.
    for t, duration in enumerate(inputs.grid.delta_hours):
        cycle += cycle_loss_per_internal_mwh(battery, settings) * duration * (
            battery.eta_charge * charge[:, t] + discharge[:, t] / battery.eta_discharge)
        calendar += p.calendar_loss_per_year * duration * 3600 / p.calendar_year_seconds
        capacity[:, t + 1] = capacity[:, t]
        if inputs.grid.ends[t].astimezone(ZoneInfo(p.timezone)).time() == time.min:
            capacity[:, t + 1] = battery.capacity_mwh * (1 - health.calendar_loss
                - health.cycle_loss - calendar - cycle)
    return capacity, linear_wear_value(settings) * (calendar + cycle)


def advance_linear_health(*, state, battery, settings, end, energy_mwh, charge_mw, discharge_mw,
                          origin="simulated_carried_state"):
    """Accumulate actual losses and commit SOH only at a real local midnight."""
    health = initialize_linear_health(state, battery, settings)
    p = settings.linear_soh
    seconds = (end - state.timestamp).total_seconds()
    if seconds <= 0 or min(charge_mw, discharge_mw) < 0 or min(charge_mw, discharge_mw) > 1e-8:
        raise ValueError("Invalid linear health execution interval")
    local_end = end.astimezone(ZoneInfo(p.timezone))
    end_day = local_end.date().isoformat()
    if end_day != health.active_day and (local_end.time() != time.min
            or (date.fromisoformat(end_day) - date.fromisoformat(health.active_day)).days != 1):
        raise ValueError("Linear health execution intervals must be split at local midnight")
    calendar = p.calendar_loss_per_year * seconds / p.calendar_year_seconds
    throughput = (charge_mw * battery.eta_charge + discharge_mw / battery.eta_discharge) * seconds / 3600
    cycle = cycle_loss_per_internal_mwh(battery, settings) * throughput
    q_calendar, q_cycle = health.calendar_loss + calendar, health.cycle_loss + cycle
    soh = health.committed_soh if end_day == health.active_day else 1 - q_calendar - q_cycle
    next_health = LinearHealthState(q_calendar, q_cycle, soh, end_day, health.calibration_id)
    wear = linear_wear_value(settings) * (calendar + cycle)
    return PhysicalState(end, energy_mwh,
        state.cumulative_throughput_mwh + (charge_mw + discharge_mw) * seconds / 3600,
        state.cumulative_degradation_eur + wear, origin, battery_health=next_health), wear


def build_linear_health(model, inputs, variables, gp):
    """Forecast linear exposure and constrain capacity at chronological midnight."""
    battery, settings = inputs.plant.battery, inputs.degradation
    health = initialize_linear_health(inputs.state, battery, settings)
    p = settings.linear_soh
    capacity = {}
    costs = {}
    count, intervals = len(inputs.scenarios.scenario_ids), len(inputs.grid.interval_ids)
    factor = cycle_loss_per_internal_mwh(battery, settings)
    zone = ZoneInfo(p.timezone)
    for s in range(count):
        calendar = gp.LinExpr(health.calendar_loss)
        cycle = gp.LinExpr(health.cycle_loss)
        current = gp.LinExpr(battery.capacity_mwh * health.committed_soh)
        start_total = health.calendar_loss + health.cycle_loss
        for t, duration in enumerate(inputs.grid.delta_hours):
            capacity[s, t] = current
            owner = inputs.tree.control_nodes[s][t]
            internal = (battery.eta_charge * variables["charge"][t, owner]
                        + variables["discharge"][t, owner] / battery.eta_discharge) * duration
            calendar += p.calendar_loss_per_year * duration * 3600 / p.calendar_year_seconds
            cycle += factor * internal
            boundary = inputs.grid.ends[t].astimezone(zone)
            if boundary.time() == time.min:
                current = battery.capacity_mwh * (1 - calendar - cycle)
            model.addConstr(variables["energy"][s, t + 1] <= battery.maximum_energy / battery.capacity_mwh * current,
                            name=f"linear_soh_energy_upper[s{s},t{t}]")
            model.addConstr(variables["energy"][s, t + 1] >= battery.energy_min_mwh / battery.capacity_mwh * current,
                            name=f"linear_soh_energy_lower[s{s},t{t}]")
        capacity[s, intervals] = current
        costs[s] = linear_wear_value(settings) * (calendar + cycle - start_total)
    variables["linear_capacity_mwh"] = capacity
    return costs
