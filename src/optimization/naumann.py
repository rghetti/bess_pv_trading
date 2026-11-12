"""Naumann-calibrated LFP exposures, exact chronological replay and linearization data.

Capacity fractions are dimensionless, calendar time is seconds, depth and fitted
C-rate use original nominal capacity. Independent exposure accumulation is our
reduced variable-condition model, not a reproduction of the ambiguous combined
virtual-FEC equation in the 2020 paper.
"""

from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime, time, timedelta, timezone
from functools import lru_cache
import hashlib
import json
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
from numpy.polynomial import Polynomial

from ..domain import BatteryHealthState, BatteryParameters, DegradationSettings, NaumannSettings, PhysicalState


def _settings(settings: DegradationSettings | NaumannSettings) -> NaumannSettings:
    """Accept a full degradation contract at shared physical-model boundaries."""
    value = settings.naumann if isinstance(settings, DegradationSettings) else settings
    if not isinstance(value, NaumannSettings):
        raise ValueError("Naumann calculations require naumann_lfp settings")
    return value


def calibration_identity(settings: DegradationSettings | NaumannSettings) -> str:
    """Bind carried memory to its physical calibration and numerical conventions."""
    parameters = _settings(settings)
    identity = asdict(parameters)
    # Preserve existing checkpoint fingerprints when the new option is unused.
    if identity["fixed_cycle_rate_c"] is None:
        del identity["fixed_cycle_rate_c"]
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    checksum = hashlib.sha256(encoded.encode()).hexdigest()[:20]
    return f"{parameters.calibration_id}:{checksum}"


def calendar_coefficient(soc, settings):
    """Return fractional capacity loss per square-root second at fixed 25 C."""
    parameters = _settings(settings)
    soc = np.asarray(soc, dtype=float)
    value = parameters.calendar_reference_coefficient * (
        parameters.calendar_soc_cubic * (soc - .5) ** 3 + parameters.calendar_soc_intercept)
    if np.any(~np.isfinite(value)) or np.any(value <= 0):
        raise ValueError("Calendar calibration must remain positive over its SOC domain")
    return value


def depth_factor(depth, settings):
    """Evaluate the published nominal-depth factor without imposing convexity."""
    parameters = _settings(settings)
    depth = np.asarray(depth, dtype=float)
    return (parameters.cycle_depth_cubic * (depth - parameters.cycle_depth_offset) ** 3
            + parameters.cycle_depth_intercept)


def depth_exposure(depth, settings):
    """Return per-cycle squared-loss exposure before the separable rate factor."""
    depth = np.asarray(depth, dtype=float)
    return depth * (depth_factor(depth, settings) / 100) ** 2


def cycle_rate_factor(rate, settings):
    """Use nominal current-rate calibration; internal MW/MWh is its voltage proxy."""
    parameters = _settings(settings)
    return parameters.cycle_rate_slope * np.asarray(rate) + parameters.cycle_rate_intercept


def naumann_segment_parameters(battery, settings):
    """Return nonuniform nominal MWh widths and positive depth-dose/MWh slopes."""
    parameters = _settings(settings)
    if battery.capacity_mwh <= 0 or battery.self_discharge_per_hour:
        raise ValueError("Naumann aging requires positive capacity and zero self-discharge")
    knots = np.asarray(parameters.depth_knots)
    if float(depth_factor(0., settings)) <= 0:
        raise ValueError("Naumann depth factor must be positive over nominal depth [0,1]")
    capacities = battery.capacity_mwh * np.diff(knots)
    marginal = np.diff(depth_exposure(knots, settings)) / capacities
    if np.any(~np.isfinite(marginal)) or np.any(marginal < 0):
        raise ValueError("Naumann depth exposure must have nonnegative increments")
    return capacities, marginal


def _shallow_initial(energy, capacities):
    """Initialize a new-study bank once; never infer carried history from SOC."""
    preceding = np.r_[0., np.cumsum(capacities[:-1])]
    return np.clip(energy - preceding, 0, capacities)


def initialize_health(state, battery, settings):
    """Validate carried health or explicitly initialize a fresh hypothetical battery."""
    parameters = _settings(settings)
    capacities, _ = naumann_segment_parameters(battery, settings)
    day = state.timestamp.astimezone(ZoneInfo(parameters.timezone)).date().isoformat()
    health = state.battery_health
    if state.cycle_segment_energy_mwh is not None:
        raise ValueError("Legacy Xu history cannot initialize two LFP health banks")
    if health is None:
        if (state.origin != "known_delivery_start_assumption" or state.cumulative_throughput_mwh
                or state.cumulative_degradation_eur):
            raise ValueError("Carried or used LFP batteries require explicit health and both banks")
        if not 0 <= state.energy_mwh <= battery.capacity_mwh:
            raise ValueError("Initial energy exceeds original nominal capacity")
        health = BatteryHealthState(0., 0., 1., day,
            tuple(_shallow_initial(state.energy_mwh, capacities)),
            tuple(_shallow_initial(battery.capacity_mwh - state.energy_mwh, capacities)),
            calibration_identity(settings))
    if health.calibration_id != calibration_identity(settings):
        raise ValueError("Carried Naumann calibration or numerical convention has changed")
    if health.active_day != day:
        raise ValueError("Stale daily health: chronological midnight must be executed explicitly")
    for name, expected in (("discharge_segment_energy_mwh", state.energy_mwh),
                           ("charge_segment_energy_mwh", battery.capacity_mwh - state.energy_mwh)):
        values = np.asarray(getattr(health, name))
        if (values.shape != capacities.shape or np.any(values > capacities + 1e-8)
                or not np.isclose(values.sum(), expected, rtol=0, atol=1e-8)):
            raise ValueError("Naumann bank geometry or aggregate energy is inconsistent")
    if max(health.calendar_exposure, health.cycle_exposure) > parameters.max_loss_fraction ** 2:
        raise ValueError("Health exposure exceeds the configured square-root graph domain")
    inferred = 1 - np.sqrt(health.calendar_exposure) - np.sqrt(health.cycle_exposure)
    if inferred > health.committed_soh + 1e-10:
        raise ValueError("Carried exposure implies healing since committed daily SOH")
    return health


# Keep the name explicit for numerical and solver callers sharing the same initialization.
initial_naumann_state = initialize_health


def health_state_from_config(config, state, battery, settings):
    """Restore explicit used-battery history, or attach the declared fresh-study state."""
    if settings.mode != "naumann_lfp":
        if config.get("state", {}).get("battery_health") is not None:
            raise ValueError("Battery health history requires naumann_lfp mode")
        return state
    supplied = config.get("state", {}).get("battery_health")
    if supplied is not None and state.battery_health is None:
        state = replace(state, battery_health=BatteryHealthState(**supplied))
    return replace(state, battery_health=initialize_health(state, battery, settings))


def available_capacity_mwh(state, battery, settings):
    """Return the day's committed capacity without advancing the health clock."""
    return battery.capacity_mwh * initialize_health(state, battery, settings).committed_soh


def effective_battery(state, battery, settings):
    """Scale accessible energy bounds while preserving E0, efficiencies and inverter rating."""
    health = initialize_health(state, battery, settings)
    return replace(battery, energy_min_mwh=battery.energy_min_mwh * health.committed_soh,
                   energy_max_mwh=battery.maximum_energy * health.committed_soh)


def power_limits(state, battery, settings):
    """Return directional AC caps from inverter, nominal-fit and effective C-rate limits."""
    parameters = _settings(settings)
    capacity = available_capacity_mwh(state, battery, settings)
    internal = min(parameters.effective_c_rate_limit * capacity,
                   parameters.nominal_c_rate_max * battery.capacity_mwh)
    return (min(battery.max_power_mw, internal / battery.eta_charge),
            min(battery.max_power_mw, internal * battery.eta_discharge))


def wear_value(settings):
    """Value a unit capacity fraction once, independently of physical calibration."""
    if not isinstance(settings, DegradationSettings) or settings.replacement_cost_eur is None:
        raise ValueError("Health wear valuation requires net replacement economics")
    return settings.replacement_cost_eur / (1 - _settings(settings).end_of_life_soh)


def rate_bin_parameters(settings):
    """Return nominal rate-bin edges and conservative squared upper-edge factors."""
    parameters = _settings(settings)
    if parameters.fixed_cycle_rate_c is not None:
        return (np.asarray([0., parameters.nominal_c_rate_max]),
                np.asarray([float(cycle_rate_factor(parameters.fixed_cycle_rate_c, settings) ** 2)]))
    count = int(np.ceil(parameters.nominal_c_rate_max / parameters.c_rate_bin_width))
    edges = np.minimum(np.arange(count + 1) * parameters.c_rate_bin_width,
                       parameters.nominal_c_rate_max)
    return edges, cycle_rate_factor(edges[1:], settings) ** 2


def rate_bin(rate, settings):
    """Use a deterministic lower-bin tie rule at a shared upper edge."""
    edges, multipliers = rate_bin_parameters(settings)
    if not np.isfinite(rate) or rate < -1e-10 or rate > edges[-1] + 1e-8:
        raise ValueError("Nominal C-rate exceeds the configured calibration-bin domain")
    index = int(np.clip(np.searchsorted(edges, max(0., rate) - 1e-10, side="left") - 1,
                        0, len(multipliers) - 1))
    return index, float(multipliers[index])


def _polynomial_extrema(polynomial, left, right):
    """Evaluate all real stationary points and endpoints of a low-degree polynomial."""
    roots = polynomial.deriv().roots()
    candidates = [left, right]
    candidates.extend(float(root.real) for root in roots
                      if abs(root.imag) < 1e-9 and left < root.real < right)
    return polynomial(np.asarray(candidates))


@lru_cache(maxsize=32)
def _calendar_graph_cached(coefficient, cubic, intercept, initial_segments,
                           max_points, year_seconds, error_budget):
    """Certify a globally shifted chord graph above the degree-six calendar stress."""
    polynomial = (coefficient * (cubic * Polynomial([-.5, 1]) ** 3 + intercept)) ** 2
    minimum = float(np.min(_polynomial_extrema(polynomial, 0., 1.)))
    if minimum <= 0:
        raise ValueError("Calendar stress must be positive on SOC [0,1]")
    segments = initial_segments
    while True:
        x = np.linspace(0., 1., segments + 1)
        values = polynomial(x)
        under, over = 0., 0.
        # Bound every segment, including concave portions that ordinary chords understate.
        for index in range(segments):
            slope = (values[index + 1] - values[index]) / (x[index + 1] - x[index])
            difference = polynomial - Polynomial([values[index] - slope * x[index], slope])
            extrema = _polynomial_extrema(difference, x[index], x[index + 1])
            under = max(under, float(np.max(extrema)))
            over = max(over, float(-np.min(extrema)))
        rounding = 64 * np.finfo(float).eps * max(
            float(np.sum(np.abs(polynomial.coef))), float(np.max(values)), 1e-30)
        shift = max(0., under) + rounding
        excess = shift + max(0., over) + rounding
        annual_error = np.sqrt(year_seconds * (minimum + excess)) - np.sqrt(year_seconds * minimum)
        if annual_error <= error_budget:
            return x, values + shift, excess
        segments *= 2
        if segments + 1 > max_points:
            raise ValueError("Calendar interpolation budget exceeds max_graph_points")


def calendar_graph(settings):
    """Build reusable SOS2 data with a certified annual calendar-SOH error budget."""
    parameters = _settings(settings)
    return _calendar_graph_cached(parameters.calendar_reference_coefficient,
        parameters.calendar_soc_cubic, parameters.calendar_soc_intercept,
        parameters.calendar_initial_soc_segments, parameters.max_graph_points,
        parameters.calendar_year_seconds, parameters.calendar_annual_soh_error_budget)


@lru_cache(maxsize=32)
def _sqrt_graph_cached(maximum_loss, budget, max_points):
    """Refine concave square-root chords at their analytically known worst errors."""
    x = np.array([0., maximum_loss ** 2])
    while True:
        y = np.sqrt(x)
        errors = np.diff(y) ** 2 / (4 * (y[1:] + y[:-1]))
        bad = errors > budget
        if not np.any(bad):
            return x, y, float(np.max(errors)) + 8 * np.finfo(float).eps
        split = ((y[1:] + y[:-1]) / 2) ** 2
        x = np.sort(np.r_[x, split[bad]])
        if len(x) > max_points:
            raise ValueError("Square-root interpolation budget exceeds max_graph_points")


def sqrt_graph(settings):
    """Return exposure knots, fractional-loss ordinates and maximum underestimation."""
    parameters = _settings(settings)
    return _sqrt_graph_cached(parameters.max_loss_fraction, parameters.sqrt_loss_error_budget,
                              parameters.max_graph_points)


_GAUSS_NODES, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(4)
GAUSS_FRACTIONS = (_GAUSS_NODES + 1) / 2
GAUSS_WEIGHTS = _GAUSS_WEIGHTS / 2


def calendar_increment(start_soc, end_soc, seconds, settings):
    """Integrate stress throughout an affine transition, including the idle case."""
    if (not np.isfinite(seconds) or seconds <= 0
            or min(start_soc, end_soc) < -1e-8 or max(start_soc, end_soc) > 1 + 1e-8):
        raise ValueError("Calendar integration requires positive time and SOC in [0,1]")
    samples = start_soc + GAUSS_FRACTIONS * (end_soc - start_soc)
    return float(seconds * np.dot(GAUSS_WEIGHTS, calendar_coefficient(samples, settings) ** 2))


def _advance_bank(values, capacities, energy_in, energy_out):
    """Apply canonical shallow-first movements independently of marginal-slope order."""
    updated = np.asarray(values).copy()
    removed = np.zeros(len(capacities))
    for index, capacity in enumerate(capacities):
        stored = min(energy_in, max(0., capacity - updated[index]))
        removed[index] = min(energy_out, max(0., updated[index]))
        updated[index] += stored - removed[index]
        energy_in -= stored
        energy_out -= removed[index]
    if max(energy_in, energy_out) > 1e-8:
        raise ValueError("Naumann dispatch exceeds nominal bank energy or headroom")
    return updated, removed


def _next_midnight(timestamp, parameters):
    """Advance the local calendar date, allowing 23-hour and 25-hour physical days."""
    zone = ZoneInfo(parameters.timezone)
    tomorrow = timestamp.astimezone(zone).date() + timedelta(days=1)
    return datetime.combine(tomorrow, time(), zone).astimezone(timezone.utc)


def advance_naumann_state(*, state, battery, settings, end, energy_mwh,
                          charge_mw, discharge_mw, origin="simulated_carried_state",
                          calendar_reference_capacity_mwh=None, enforce_midnight_bounds=True):
    """Accrue executed exposure; commit capacity exactly at each chronological midnight."""
    parameters = _settings(settings)
    health = initialize_health(state, battery, settings)
    if end.tzinfo is None or end <= state.timestamp:
        raise ValueError("Naumann execution requires an increasing timezone-aware clock")
    if (min(charge_mw, discharge_mw) < 0 or not np.isfinite(charge_mw + discharge_mw)
            or min(charge_mw, discharge_mw) > 1e-8):
        raise ValueError("Naumann commands must be finite, nonnegative and exclusive")
    duration = (end - state.timestamp).total_seconds() / 3600
    internal_charge = battery.eta_charge * charge_mw
    internal_discharge = discharge_mw / battery.eta_discharge
    expected_energy = state.energy_mwh + duration * (internal_charge - internal_discharge)
    if not np.isclose(expected_energy, energy_mwh, rtol=0, atol=1e-8):
        raise ValueError("Naumann energy and executed commands disagree")
    capacities, marginal = naumann_segment_parameters(battery, settings)
    _, multiplier = rate_bin((internal_charge + internal_discharge) / battery.capacity_mwh, settings)
    before = np.sqrt(health.calendar_exposure) + np.sqrt(health.cycle_exposure)
    cursor = state.timestamp
    energy = state.energy_mwh
    # Split only at real calendar boundaries, never at an auction's scheduling boundary.
    while cursor < end:
        midnight = _next_midnight(cursor, parameters)
        boundary = min(end, midnight)
        seconds = (boundary - cursor).total_seconds()
        hours = seconds / 3600
        next_energy = energy + hours * (internal_charge - internal_discharge)
        available = (battery.capacity_mwh * health.committed_soh
                     if calendar_reference_capacity_mwh is None else calendar_reference_capacity_mwh)
        if not np.isfinite(available) or available <= 0:
            raise ValueError("Calendar capacity reference must be positive")
        calendar = health.calendar_exposure + calendar_increment(
            energy / available, next_energy / available, seconds, settings)
        forward, counted_discharge = _advance_bank(health.discharge_segment_energy_mwh,
            capacities, internal_charge * hours, internal_discharge * hours)
        complement, counted_charge = _advance_bank(health.charge_segment_energy_mwh,
            capacities, internal_discharge * hours, internal_charge * hours)
        cycle = health.cycle_exposure + .5 * multiplier * float(
            marginal @ (counted_discharge + counted_charge))
        if max(calendar, cycle) > parameters.max_loss_fraction ** 2:
            raise ValueError("Executed health exposure exceeds configured graph domain")
        soh = health.committed_soh
        day = health.active_day
        if boundary == midnight:
            soh = float(1 - np.sqrt(calendar) - np.sqrt(cycle))
            if soh <= 0:
                raise ValueError("Executed aging has exhausted battery capacity")
            day = midnight.astimezone(ZoneInfo(parameters.timezone)).date().isoformat()
            if enforce_midnight_bounds and (next_energy > battery.maximum_energy * soh + 1e-8
                    or next_energy < battery.energy_min_mwh * soh - 1e-8):
                raise ValueError("Midnight fade violates accessible energy bounds; energy was not clipped")
        health = BatteryHealthState(calendar, cycle, soh, day, tuple(forward),
                                    tuple(complement), health.calibration_id)
        cursor = boundary
        energy = next_energy
    after = np.sqrt(health.calendar_exposure) + np.sqrt(health.cycle_exposure)
    cost = wear_value(settings) * max(0., float(after - before))
    return replace(state, timestamp=end, energy_mwh=energy_mwh,
        cumulative_throughput_mwh=state.cumulative_throughput_mwh + duration * (charge_mw + discharge_mw),
        cumulative_degradation_eur=state.cumulative_degradation_eur + cost,
        origin=origin, battery_health=health), cost


def preview_naumann_state(**kwargs):
    """Forecast a candidate command on immutable copies for midnight safety protection."""
    return advance_naumann_state(**kwargs, enforce_midnight_bounds=False)


def evaluate_naumann_dispatch(*, battery, settings, state, charge_mw, discharge_mw,
                             delta_hours, starts=None, calendar_reference_capacity_mwh=None):
    """Independently replay one path without Gurobi or resetting carried exposures."""
    charge, discharge, durations = (np.asarray(value, dtype=float)
                                   for value in (charge_mw, discharge_mw, delta_hours))
    if (charge.ndim != 1 or charge.shape != discharge.shape or charge.shape != durations.shape
            or np.any(~np.isfinite(durations)) or np.any(durations <= 0)):
        raise ValueError("Naumann dispatch arrays must have equal one-dimensional positive-time shapes")
    current = replace(state, battery_health=initialize_health(state, battery, settings))
    states = [current]
    costs = []
    for index, hours in enumerate(durations):
        if starts is not None and starts[index] != current.timestamp:
            raise ValueError("Naumann replay clocks are not contiguous")
        energy = current.energy_mwh + hours * (battery.eta_charge * charge[index]
                                               - discharge[index] / battery.eta_discharge)
        current, cost = advance_naumann_state(state=current, battery=battery, settings=settings,
            end=current.timestamp + timedelta(hours=float(hours)), energy_mwh=float(energy),
            charge_mw=float(charge[index]), discharge_mw=float(discharge[index]),
            calendar_reference_capacity_mwh=calendar_reference_capacity_mwh)
        states.append(current)
        costs.append(cost)
    result = {name: np.array([getattr(item.battery_health, name) for item in states])
              for name in ("calendar_exposure", "cycle_exposure", "committed_soh",
                           "discharge_segment_energy_mwh", "charge_segment_energy_mwh")}
    result.update(energy_mwh=np.array([item.energy_mwh for item in states]),
        available_capacity_mwh=battery.capacity_mwh * result["committed_soh"],
        interval_degradation_eur=np.asarray(costs), degradation_eur=float(sum(costs)), final_state=current)
    return result


def health_diagnostics(state, battery, settings):
    """Expose daily physical capacity separately from continuously accrued damage."""
    if settings.mode != "naumann_lfp":
        return {}
    parameters = _settings(settings)
    health = initialize_health(state, battery, settings)
    return {"naumann_committed_soh": health.committed_soh,
        "naumann_available_capacity_mwh": battery.capacity_mwh * health.committed_soh,
        "naumann_active_day": health.active_day,
        "naumann_calendar_exposure": health.calendar_exposure,
        "naumann_cycle_exposure": health.cycle_exposure,
        "naumann_calendar_loss_fraction": float(np.sqrt(health.calendar_exposure)),
        "naumann_cycle_loss_fraction": float(np.sqrt(health.cycle_exposure)),
        "naumann_end_of_life": health.committed_soh <= parameters.end_of_life_soh,
        "naumann_calibration_id": health.calibration_id}


def approximation_diagnostics(settings):
    """Separate depth-curve, calendar, square-root and rate-quantization errors."""
    parameters = _settings(settings)
    polynomial = Polynomial([-parameters.cycle_depth_offset, 1])
    factor = parameters.cycle_depth_cubic * polynomial ** 3 + parameters.cycle_depth_intercept
    dose = Polynomial([0, 1]) * (factor / 100) ** 2
    knots = np.asarray(parameters.depth_knots)
    values = dose(knots)
    maximum_depth_error = 0.
    # The source does not characterize depths below 1%; report absolute curve
    # interpolation error instead of an unstable relative error near zero.
    for index in range(len(knots) - 1):
        slope = (values[index + 1] - values[index]) / (knots[index + 1] - knots[index])
        difference = dose - Polynomial([values[index] - slope * knots[index], slope])
        extrema = _polynomial_extrema(difference, knots[index], knots[index + 1])
        maximum_depth_error = max(maximum_depth_error, float(np.max(np.abs(extrema))))
    edges, factors = rate_bin_parameters(settings)
    lower = cycle_rate_factor(edges[:-1], settings) ** 2
    _, _, calendar_error = calendar_graph(settings)
    _, _, loss_error = sqrt_graph(settings)
    minimum = float(calendar_coefficient(0., settings) ** 2)
    annual_calendar_bound = (np.sqrt(parameters.calendar_year_seconds * (minimum + calendar_error))
                             - np.sqrt(parameters.calendar_year_seconds * minimum))
    return {
        "calendar_upper_stress_excess_per_second": calendar_error,
        "calendar_annual_soh_error_bound": float(annual_calendar_bound),
        "sqrt_component_loss_error_bound": loss_error,
        "depth_curve_max_absolute_exposure_error": maximum_depth_error,
        "rate_max_relative_exposure_overestimate": (float(np.max(factors / lower - 1))
            if parameters.fixed_cycle_rate_c is None else None),
        "rate_max_relative_cycle_loss_overestimate": (float(np.max(np.sqrt(factors / lower) - 1))
            if parameters.fixed_cycle_rate_c is None else None),
        "fixed_cycle_rate_c": parameters.fixed_cycle_rate_c,
        "rate_below_0_2c_is_extrapolation": True,
        "depth_factor_fit_range": [0.05, 1.0],
        "depth_below_0_05_extrapolates_fitted_factor": True,
        "depth_below_0_01_is_extrapolation": True,
        "variable_rate_attribution": ("local half-cycle dose, not completed-cycle mean-rate aging"
            if parameters.fixed_cycle_rate_c is None else "fixed reference rate; actual rate does not affect cycling aging"),
        "future_calendar_soc_normalization": "capacity known at optimization cutoff",
    }
