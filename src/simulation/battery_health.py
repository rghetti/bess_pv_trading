"""Carry the optional Naumann physical health through replay boundaries."""

from __future__ import annotations

import ast
from dataclasses import asdict, replace
from datetime import time
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np


class BatteryEndOfLifeReached(RuntimeError):
    """Stop a completed delivery day without changing outstanding commitments."""

    def __init__(self, state):
        self.state = state
        self.status = "stopped_at_battery_end_of_life"
        super().__init__(
            f"Battery reached its configured end of life at {state.timestamp.isoformat()}; "
            "the completed day's state and accepted commitments were retained")


def restore_battery_health(value):
    """Restore typed health and both cycle memories from JSON or a CSV cell."""
    if value is None:
        return None
    from ..domain import BatteryHealthState, LinearHealthState

    if isinstance(value, (BatteryHealthState, LinearHealthState)):
        return value
    # CSV snapshots contain only literal scalar values and segment tuples.
    values = ast.literal_eval(value) if isinstance(value, str) else dict(value)
    if "calendar_loss" in values:
        return LinearHealthState(**values)
    for name in ("discharge_segment_energy_mwh", "charge_segment_energy_mwh"):
        if values.get(name) is not None:
            values[name] = tuple(values[name])
    return BatteryHealthState(**values)


def initialize_physical_health(state, battery, settings):
    """Initialize once; an existing health state remains authoritative."""
    if settings.mode == "linear_soh":
        from ..optimization.linear_soh import initialize_linear_health
        return replace(state, battery_health=initialize_linear_health(state, battery, settings))
    if settings.mode != "naumann_lfp":
        return state
    from ..optimization.naumann import initialize_health

    return replace(state, battery_health=initialize_health(state, battery, settings))


def relative_terminal_soc(state_settings, degradation_mode):
    """Keep available-capacity percentages distinct from explicit MWh overrides."""
    if degradation_mode not in {"naumann_lfp", "linear_soh"}:
        return None, None
    basis = state_settings.get("terminal_soc_basis",
        state_settings.get("initial_soc_basis", "available_capacity"))
    if basis != "available_capacity":
        return None, None
    target = (state_settings.get("terminal_soc_percent")
              if "terminal_target_mwh" not in state_settings else None)
    minimum = (state_settings.get("terminal_soc_min_percent")
               if "terminal_minimum_mwh" not in state_settings else None)
    return (None if target is None else float(target),
            None if minimum is None else float(minimum))


def health_execution_record(state, next_state, battery, settings):
    """Expose physical loss separately from its economic valuation and cash."""
    if settings.mode == "linear_soh":
        from ..optimization.linear_soh import linear_wear_value
        record = {}
        for suffix, boundary in (("start", state), ("end", next_state)):
            h = boundary.battery_health
            capacity = battery.capacity_mwh * h.committed_soh
            record.update({f"battery_health_{suffix}": asdict(h), f"battery_soh_{suffix}": h.committed_soh,
                f"available_capacity_mwh_{suffix}": capacity,
                f"nominal_soc_{suffix}_percent": 100 * boundary.energy_mwh / battery.capacity_mwh,
                f"available_soc_{suffix}_percent": 100 * boundary.energy_mwh / capacity,
                f"linear_active_day_{suffix}": h.active_day,
                f"linear_calendar_loss_fraction_{suffix}": h.calendar_loss,
                f"linear_cycle_loss_fraction_{suffix}": h.cycle_loss})
        for mode in ("calendar", "cycle"):
            loss = record[f"linear_{mode}_loss_fraction_end"] - record[f"linear_{mode}_loss_fraction_start"]
            record[f"{mode}_capacity_loss_fraction"] = loss
            record[f"{mode}_degradation_eur"] = linear_wear_value(settings) * loss
        return record
    if settings.mode != "naumann_lfp":
        return {}
    from ..optimization.naumann import health_diagnostics, wear_value

    record: dict[str, Any] = {}
    for suffix, boundary in (("start", state), ("end", next_state)):
        diagnostics = health_diagnostics(boundary, battery, settings)
        record.update({f"{key}_{suffix}": value for key, value in diagnostics.items()})
        record[f"battery_health_{suffix}"] = asdict(boundary.battery_health)
        record[f"battery_soh_{suffix}"] = boundary.battery_health.committed_soh
        record[f"available_capacity_mwh_{suffix}"] = (
            battery.capacity_mwh * boundary.battery_health.committed_soh)
        record[f"nominal_soc_{suffix}_percent"] = 100 * boundary.energy_mwh / battery.capacity_mwh
        record[f"available_soc_{suffix}_percent"] = (100 * boundary.energy_mwh
            / record[f"available_capacity_mwh_{suffix}"])
    # Loss attribution and EUR valuation use exactly the same executed exposure
    # increments, so neither component books a second degradation penalty.
    for mode in ("calendar", "cycle"):
        loss = (record[f"naumann_{mode}_loss_fraction_end"]
                - record[f"naumann_{mode}_loss_fraction_start"])
        record[f"{mode}_capacity_loss_fraction"] = loss
        record[f"{mode}_degradation_eur"] = wear_value(settings) * loss
    return record


def at_battery_end_of_life(state, settings):
    """Check the committed physical health, never an optimizer's projection."""
    if settings is None or settings.mode not in {"naumann_lfp", "linear_soh"} or state is None or state.battery_health is None:
        return False
    threshold = settings.linear_soh.end_of_life_soh if settings.mode == "linear_soh" else settings.naumann.end_of_life_soh
    return state.battery_health.committed_soh <= threshold


def health_summary(state, battery, settings):
    """Keep calibration assumptions and final health in persisted results."""
    if settings.mode == "linear_soh":
        return {"battery_health": asdict(state.battery_health), "battery_health_calibration": asdict(settings.linear_soh),
                "degradation_cost_accounting": "noncash wear valuation; excluded from market cash flow"}
    if settings.mode != "naumann_lfp":
        return {}
    from ..optimization.naumann import approximation_diagnostics, health_diagnostics

    return {"battery_health": health_diagnostics(state, battery, settings),
            "battery_health_calibration": asdict(settings.naumann),
            "battery_health_approximation": approximation_diagnostics(settings),
            "degradation_cost_accounting": "noncash wear valuation; excluded from market cash flow"}


def forecast_health_record(arrays, interval, scenario, battery, settings, *, energy_start, energy_end):
    """Label optimizer health forecasts separately from committed physical health."""
    if settings.mode == "linear_soh" and arrays:
        capacity = np.asarray(arrays["linear_capacity_mwh"])
        owner = 0 if len(capacity) == 1 else scenario
        return {"forecast_available_capacity_start_mwh": float(capacity[owner, interval]),
                "forecast_available_capacity_end_mwh": float(capacity[owner, interval + 1]),
                "battery_health_role": "forecast_only_linear_throughput"}
    if settings.mode != "naumann_lfp" or not arrays:
        return {}
    from ..optimization.naumann import wear_value

    # Common controls have one physical health path; adaptive recourse has one
    # path per scenario. Scenario selection never turns a forecast into actual state.
    capacities = np.asarray(arrays["naumann_capacity_mwh"])
    owner = 0 if len(capacities) == 1 else scenario
    capacity_start = float(capacities[owner, interval])
    capacity_end = float(capacities[owner, interval + 1])
    record = {"forecast_available_capacity_start_mwh": capacity_start,
        "forecast_available_capacity_end_mwh": capacity_end,
        "forecast_health_lower_start": float(arrays["naumann_health_lower"][owner, interval]),
        "forecast_health_lower_end": float(arrays["naumann_health_lower"][owner, interval + 1]),
        "nominal_soc_start_percent": 100 * energy_start / battery.capacity_mwh,
        "nominal_soc_end_percent": 100 * energy_end / battery.capacity_mwh,
        "forecast_available_soc_start_percent": 100 * energy_start / capacity_start,
        "forecast_available_soc_end_percent": 100 * energy_end / capacity_end,
        "battery_health_role": "forecast_only_conservative_milp_surrogate"}
    for mode in ("calendar", "cycle"):
        losses = arrays[f"naumann_{mode}_loss_pwl"]
        start = float(losses[owner, interval])
        end = float(losses[owner, interval + 1])
        record[f"forecast_{mode}_loss_start_fraction"] = start
        record[f"forecast_{mode}_loss_end_fraction"] = end
        record[f"forecast_{mode}_degradation_eur"] = wear_value(settings) * (end - start)
    if "naumann_calendar_guard_loss_pwl" in arrays:
        guard = arrays["naumann_calendar_guard_loss_pwl"]
        record.update(forecast_calendar_capacity_guard_loss_start_fraction=float(guard[owner, interval]),
            forecast_calendar_capacity_guard_loss_end_fraction=float(guard[owner, interval + 1]),
            forecast_calendar_capacity_guard_role="capacity_protection_only; excluded from wear cost")
    return record


def protect_midnight_health(*, state, battery, settings, end, charge_mw, discharge_mw,
                            curtailment_mw, pv_ac_mw, poi_export_mw, numerical_tolerance):
    """Adjust the executed command to fit capacity after a real midnight fade.

    Forecast error can make grid tracking fill the old SOC ceiling despite the
    planner's health guard. Previewing immutable candidate states allows the
    executor to reduce charging or discharge, accounting for every physical MWh.
    """
    if settings.mode not in {"naumann_lfp", "linear_soh"}:
        return charge_mw, discharge_mw, curtailment_mw, 0.0
    parameters = settings.linear_soh if settings.mode == "linear_soh" else settings.naumann
    if end.astimezone(ZoneInfo(parameters.timezone)).time() != time.min:
        return charge_mw, discharge_mw, curtailment_mw, 0.0
    from ..optimization.naumann import power_limits, preview_naumann_state
    from ..optimization.linear_soh import advance_linear_health

    hours = (end - state.timestamp).total_seconds() / 3600

    def preview(net_discharge):
        """Evaluate a bounded AC command without committing exposure or SOH."""
        charge = max(0.0, -net_discharge)
        discharge = max(0.0, net_discharge)
        energy = state.energy_mwh + hours * (
            battery.eta_charge * charge - discharge / battery.eta_discharge)
        if settings.mode == "linear_soh":
            projected, _ = advance_linear_health(state=state, battery=battery, settings=settings,
                end=end, energy_mwh=energy, charge_mw=charge, discharge_mw=discharge)
        else:
            projected, _ = preview_naumann_state(state=state, battery=battery, settings=settings,
                end=end, energy_mwh=energy, charge_mw=charge, discharge_mw=discharge)
        upper = battery.maximum_energy * projected.battery_health.committed_soh
        return energy <= upper, charge, discharge

    net = discharge_mw - charge_mw
    safe, _, _ = preview(net)
    if safe:
        return charge_mw, discharge_mw, curtailment_mw, 0.0
    discharge_limit = (battery.max_power_mw if settings.mode == "linear_soh"
                       else power_limits(state, battery, settings)[1])
    floor = battery.energy_min_mwh * state.battery_health.committed_soh
    available_discharge = max(0.0, state.energy_mwh - floor) * battery.eta_discharge / hours
    # Additional PV curtailment can create export headroom, but battery discharge
    # itself must fit the POI even when all PV has been curtailed.
    high = min(discharge_limit, available_discharge, poi_export_mw)
    safe, _, _ = preview(high)
    if not safe or high < net:
        raise ValueError("No bounded battery command can protect the midnight faded-capacity limit")
    low = net
    energy_per_power = hours * max(battery.eta_charge, 1 / battery.eta_discharge)
    power_tolerance = max(np.finfo(float).eps * max(1.0, abs(high), abs(low)),
                          numerical_tolerance / (4 * energy_per_power))
    # Keep a feasible upper endpoint throughout, including at discontinuous
    # conservative C-rate bin changes; only the final actual command is committed.
    while high - low > power_tolerance:
        middle = (low + high) / 2
        if middle == low or middle == high:
            break
        safe, _, _ = preview(middle)
        if safe:
            high = middle
        else:
            low = middle
    safe, charge, discharge = preview(high)
    if not safe:
        raise ValueError("Midnight health protection did not produce a feasible physical command")
    curtailment = max(curtailment_mw, min(pv_ac_mw, pv_ac_mw + high - poi_export_mw))
    adjustment = (abs(charge - charge_mw) + abs(discharge - discharge_mw)
                  + abs(curtailment - curtailment_mw))
    return charge, discharge, curtailment, adjustment
