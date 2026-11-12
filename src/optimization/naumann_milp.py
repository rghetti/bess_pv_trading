"""Linear Naumann LFP exposure model with carried, symmetric Xu depth memory.

Power decisions remain on the AC side.  Virtual bank transfers are internal MWh,
calendar exposure uses seconds, and both health losses are fractions of the
original capacity.  Future local midnights update physical capacity bounds; the
forecast SOC denominator stays at the capacity known at the current cutoff.
A separate calendar guard uses maximum operating SOC after the first midnight,
so physical capacity bounds remain conservative when actual SOC is renormalized.
"""
from __future__ import annotations

from typing import Any
from zoneinfo import ZoneInfo

import numpy as np

from ..domain import OptimizationInput
from .naumann import (calendar_graph, initialize_health,
                      naumann_segment_parameters, rate_bin_parameters,
                      sqrt_graph, wear_value)


# Four positive quadrature weights integrate the analytic degree-six calendar
# exposure stress exactly along a constant-power, zero-self-discharge SOC path.
_GAUSS_ABSCISSA, _GAUSS_WEIGHTS = np.polynomial.legendre.leggauss(4)
_GAUSS_FRACTIONS = (_GAUSS_ABSCISSA + 1.0) / 2.0
_GAUSS_WEIGHTS = _GAUSS_WEIGHTS / 2.0
# Exposures are tiny squared capacity fractions.  Express solver variables in
# micro-units and calendar stress in pico-units so feasibility tolerances cannot
# silently erase an interval's aging.  Public state and arrays retain SI units.
_EXPOSURE_SCALE = 1e6
_CALENDAR_STRESS_SCALE = 1e12


def _value(expression: Any) -> float:
    """Read a solved variable, linear expression, or known constant uniformly."""
    if hasattr(expression, "X"):
        return float(expression.X)
    if hasattr(expression, "getValue"):
        return float(expression.getValue())
    return float(expression)


def _graph(model: Any, argument: Any, x: np.ndarray, y: np.ndarray,
           name: str) -> Any:
    """Enforce a complete PWL graph rather than a one-sided aging epigraph."""
    abscissa = model.addVar(lb=float(x[0]), ub=float(x[-1]), name=f"{name}_x")
    ordinate = model.addVar(lb=float(np.min(y)), ub=float(np.max(y)), name=name)
    model.addConstr(abscissa == argument, name=f"{name}_argument")
    model.addGenConstrPWL(abscissa, ordinate, x.tolist(), y.tolist(),
                         name=f"{name}_graph")
    return ordinate


def _initial(inputs: OptimizationInput):
    """Use the one declared study initialization or the complete carried state."""
    return initialize_health(state=inputs.state, battery=inputs.plant.battery,
                             settings=inputs.degradation)


def _boundary_days(inputs: OptimizationInput) -> tuple[str, ...]:
    """Identify physical capacity updates in the configured local calendar."""
    zone = ZoneInfo(inputs.degradation.naumann.timezone)
    boundaries = (inputs.grid.starts[0],) + inputs.grid.ends
    # An affine power transition must not hide a local midnight inside itself.
    for start, end in zip(inputs.grid.starts, inputs.grid.ends):
        start_local = start.astimezone(zone)
        end_local = end.astimezone(zone)
        if start_local.date() != end_local.date() and any(
                (end_local.hour, end_local.minute, end_local.second,
                 end_local.microsecond)):
            raise ValueError("Naumann delivery intervals must be split at local midnight")
    return tuple(boundary.astimezone(zone).date().isoformat()
                 for boundary in boundaries)


def _build_bank_commands(model, inputs, variables, gp, capacities, marginal):
    """Share depth-memory transfers, priorities and scalar rate doses by node."""
    battery = inputs.plant.battery
    _, intervals = np.shape(inputs.scenarios.pv_ac_mw)
    edges, rate_factors = rate_bin_parameters(inputs.degradation)
    bank_in = {}
    bank_out = {}
    fill = {}
    empty = {}
    bins = {}
    depth_dose = {}
    cycle_increment = {}
    # Each information owner gets one physical rate choice for all virtual depth
    # segments.  The complement bank swaps physical charge and discharge.
    for t, duration in enumerate(inputs.grid.delta_hours):
        maximum_transfer = battery.max_power_mw * duration / battery.eta_discharge
        maximum_dose = 0.5 * min(float(np.dot(marginal, capacities)),
                                 float(np.max(marginal)) * maximum_transfer)
        for owner in sorted({row[t] for row in inputs.tree.control_nodes}):
            charge = variables["charge"][t, owner]
            discharge = variables["discharge"][t, owner]
            internal_power = battery.eta_charge * charge + discharge / battery.eta_discharge
            incoming = duration * battery.eta_charge * charge
            outgoing = duration * discharge / battery.eta_discharge
            suffix = f"n{owner},t{t}"
            # No accounting bank is a separate physical module: both use the
            # entire battery command and their own independent depth history.
            for bank in range(2):
                for j, width in enumerate(capacities):
                    key = (t, owner, bank, j)
                    bank_in[key] = model.addVar(ub=float(width),
                        name=f"naumann_bank_in[{suffix},b{bank},j{j}]")
                    bank_out[key] = model.addVar(ub=float(width),
                        name=f"naumann_bank_out[{suffix},b{bank},j{j}]")
                model.addConstr(gp.quicksum(bank_in[t, owner, bank, j]
                    for j in range(len(capacities))) == (incoming if bank == 0 else outgoing),
                    name=f"naumann_bank_in_sum[{suffix},b{bank}]")
                model.addConstr(gp.quicksum(bank_out[t, owner, bank, j]
                    for j in range(len(capacities))) == (outgoing if bank == 0 else incoming),
                    name=f"naumann_bank_out_sum[{suffix},b{bank}]")
                # Explicit priorities are required: Naumann's increasing depth
                # dose need not have increasing marginal slopes.
                for j in range(len(capacities) - 1):
                    key = (t, owner, bank, j)
                    fill[key] = model.addVar(vtype=gp.GRB.BINARY,
                        name=f"naumann_fill_deeper[{suffix},b{bank},j{j}]")
                    empty[key] = model.addVar(vtype=gp.GRB.BINARY,
                        name=f"naumann_empty_deeper[{suffix},b{bank},j{j}]")
                    model.addConstr(gp.quicksum(bank_in[t, owner, bank, k]
                        for k in range(j + 1, len(capacities))) <= maximum_transfer * fill[key],
                        name=f"naumann_fill_priority[{suffix},b{bank},j{j}]")
                    model.addConstr(gp.quicksum(bank_out[t, owner, bank, k]
                        for k in range(j + 1, len(capacities))) <= maximum_transfer * empty[key],
                        name=f"naumann_empty_priority[{suffix},b{bank},j{j}]")
            depth_dose[t, owner] = model.addVar(ub=maximum_dose * _EXPOSURE_SCALE,
                name=f"naumann_depth_dose[{suffix}]")
            model.addConstr(depth_dose[t, owner] == 0.5 * _EXPOSURE_SCALE * gp.quicksum(
                float(marginal[j]) * bank_out[t, owner, bank, j]
                for bank in range(2) for j in range(len(capacities))),
                name=f"naumann_depth_dose_sum[{suffix}]")
            allocated = {}
            # Disaggregating one scalar avoids a segment-by-rate variable array.
            for k, multiplier in enumerate(rate_factors):
                bins[t, owner, k] = model.addVar(vtype=gp.GRB.BINARY,
                    name=f"naumann_rate_bin[{suffix},k{k}]")
                allocated[k] = model.addVar(ub=maximum_dose * _EXPOSURE_SCALE,
                    name=f"naumann_rate_dose[{suffix},k{k}]")
                model.addConstr(allocated[k] <= maximum_dose * _EXPOSURE_SCALE * bins[t, owner, k],
                    name=f"naumann_rate_dose_active[{suffix},k{k}]")
            model.addConstr(gp.quicksum(bins[t, owner, k]
                for k in range(len(rate_factors))) == 1,
                name=f"naumann_rate_one_bin[{suffix}]")
            model.addConstr(internal_power >= battery.capacity_mwh * gp.quicksum(
                float(edges[k]) * bins[t, owner, k] for k in range(len(rate_factors))),
                name=f"naumann_rate_lower[{suffix}]")
            model.addConstr(internal_power <= battery.capacity_mwh * gp.quicksum(
                float(edges[k + 1]) * bins[t, owner, k] for k in range(len(rate_factors))),
                name=f"naumann_rate_upper[{suffix}]")
            model.addConstr(gp.quicksum(allocated.values()) == depth_dose[t, owner],
                name=f"naumann_rate_dose_sum[{suffix}]")
            cycle_increment[t, owner] = gp.quicksum(float(rate_factors[k]) * allocated[k]
                for k in range(len(rate_factors)))
    variables.update(naumann_bank_in=bank_in, naumann_bank_out=bank_out,
                     naumann_bank_fill=fill, naumann_bank_empty=empty,
                     naumann_rate_bins=bins, naumann_depth_dose=depth_dose,
                     naumann_cycle_increment=cycle_increment)


def build_naumann_degradation(model: Any, inputs: OptimizationInput,
                              variables: dict[str, Any], gp: Any) -> dict[int, Any]:
    """Build future health loss, conservative daily capacity and its euro value."""
    battery = inputs.plant.battery
    settings = inputs.degradation
    health = _initial(inputs)
    capacities, marginal = naumann_segment_parameters(battery, settings)
    calendar_x, calendar_upper, _ = calendar_graph(settings)
    exposure_x, loss_y, sqrt_error = sqrt_graph(settings)
    count, intervals = np.shape(inputs.scenarios.pv_ac_mw)
    days = _boundary_days(inputs)
    reference_capacity = battery.capacity_mwh * health.committed_soh
    if reference_capacity <= 0:
        raise ValueError("Naumann optimization requires positive committed capacity")
    _build_bank_commands(model, inputs, variables, gp, capacities, marginal)
    bank_energy = {}
    calendar_exposure = {}
    cycle_exposure = {}
    calendar_loss = {}
    cycle_loss = {}
    calendar_increment = {}
    calendar_guard_exposure = {}
    calendar_guard_loss = {}
    health_lower = {}
    daily_capacity = {}
    maximum_operating_soc = battery.maximum_energy / battery.capacity_mwh
    future_guard_stress = float(np.interp(maximum_operating_soc, calendar_x, calendar_upper))
    starting_banks = (health.discharge_segment_energy_mwh,
                      health.charge_segment_energy_mwh)
    # Continuous exposure states and PWL losses are scenario states; physical
    # transfers and their priorities belong to the existing information owners.
    for s in range(count):
        for boundary in range(intervals + 1):
            suffix = f"s{s},boundary{boundary}"
            calendar_exposure[s, boundary] = model.addVar(ub=float(exposure_x[-1]) * _EXPOSURE_SCALE,
                name=f"naumann_calendar_exposure[{suffix}]")
            cycle_exposure[s, boundary] = model.addVar(ub=float(exposure_x[-1]) * _EXPOSURE_SCALE,
                name=f"naumann_cycle_exposure[{suffix}]")
            # Only endpoints price wear; midnight boundaries also set capacity.
            # Intermediate loss readouts are reconstructed after optimization.
            needs_loss = (boundary in (0, intervals) or
                          days[boundary] != days[boundary - 1])
            calendar_loss[s, boundary] = (_graph(model, calendar_exposure[s, boundary],
                exposure_x * _EXPOSURE_SCALE, loss_y, f"naumann_calendar_loss[{suffix}]")
                if needs_loss else None)
            cycle_loss[s, boundary] = (_graph(model, cycle_exposure[s, boundary],
                exposure_x * _EXPOSURE_SCALE, loss_y, f"naumann_cycle_loss[{suffix}]")
                if needs_loss else None)
            # During the cutoff's current day the forecast and physical guard
            # have identical exposures, so alias them instead of duplicating an
            # entire PWL graph.  The first midnight still closes that same day.
            if boundary == 0 or days[boundary - 1] == days[0]:
                calendar_guard_exposure[s, boundary] = calendar_exposure[s, boundary]
                calendar_guard_loss[s, boundary] = calendar_loss[s, boundary]
            else:
                calendar_guard_exposure[s, boundary] = model.addVar(
                    ub=float(exposure_x[-1]) * _EXPOSURE_SCALE,
                    name=f"naumann_calendar_guard_exposure[{suffix}]")
                calendar_guard_loss[s, boundary] = _graph(model,
                    calendar_guard_exposure[s, boundary], exposure_x * _EXPOSURE_SCALE,
                    loss_y, f"naumann_calendar_guard_loss[{suffix}]") if needs_loss else None
            health_lower[s, boundary] = ((1 - calendar_guard_loss[s, boundary]
                - cycle_loss[s, boundary] - 2 * sqrt_error) if needs_loss else None)
            for bank in range(2):
                for j, width in enumerate(capacities):
                    bank_energy[s, boundary, bank, j] = model.addVar(ub=float(width),
                        name=f"naumann_bank_energy[{suffix},b{bank},j{j}]")
                total = variables["energy"][s, boundary]
                model.addConstr(gp.quicksum(bank_energy[s, boundary, bank, j]
                    for j in range(len(capacities))) == (total if bank == 0 else battery.capacity_mwh - total),
                    name=f"naumann_bank_energy_sum[{suffix},b{bank}]")
            # Capacity is committed only at a local midnight.  The current day
            # uses its known value; future days use a conservative health bound.
            if boundary == 0:
                active_capacity = reference_capacity
            elif days[boundary] != days[boundary - 1]:
                active_capacity = battery.capacity_mwh * health_lower[s, boundary]
            daily_capacity[s, boundary] = active_capacity
            model.addConstr(variables["energy"][s, boundary] >= battery.energy_min_mwh * health.committed_soh,
                name=f"naumann_energy_lower[{suffix}]")
            model.addConstr(variables["energy"][s, boundary] <= battery.maximum_energy / battery.capacity_mwh * active_capacity,
                name=f"naumann_energy_upper[{suffix}]")
        model.addConstr(calendar_exposure[s, 0] == health.calendar_exposure * _EXPOSURE_SCALE,
            name=f"naumann_initial_calendar[s{s}]")
        model.addConstr(cycle_exposure[s, 0] == health.cycle_exposure * _EXPOSURE_SCALE,
            name=f"naumann_initial_cycle[s{s}]")
        for bank in range(2):
            for j in range(len(capacities)):
                model.addConstr(bank_energy[s, 0, bank, j] == starting_banks[bank][j],
                    name=f"naumann_initial_bank[s{s},b{bank},j{j}]")
        # An interval has an affine internal-energy path.  Positive quadrature
        # weights preserve the calendar graph's certified upper bound.
        for t, duration in enumerate(inputs.grid.delta_hours):
            owner = inputs.tree.control_nodes[s][t]
            stress = []
            for r, fraction in enumerate(_GAUSS_FRACTIONS):
                soc = ((1 - float(fraction)) * variables["energy"][s, t]
                       + float(fraction) * variables["energy"][s, t + 1]) / reference_capacity
                stress.append(_graph(model, soc, calendar_x,
                    calendar_upper * _CALENDAR_STRESS_SCALE,
                    f"naumann_calendar_stress[s{s},t{t},r{r}]"))
            calendar_increment[s, t] = duration * 3600 * gp.quicksum(
                float(weight) * ordinate for weight, ordinate in zip(_GAUSS_WEIGHTS, stress)) / _CALENDAR_STRESS_SCALE
            model.addConstr(calendar_exposure[s, t + 1] == calendar_exposure[s, t]
                + _EXPOSURE_SCALE * calendar_increment[s, t],
                name=f"naumann_calendar_balance[s{s},t{t}]")
            # Actual future-day SOC uses its newly faded capacity.  Its physical
            # SOC ceiling bounds that unknown normalization without multiplying
            # or dividing continuous decisions.  This affects only capacity
            # protection; the SOC-aware calendar forecast still prices wear.
            if days[t] != days[0]:
                model.addConstr(calendar_guard_exposure[s, t + 1]
                    == calendar_guard_exposure[s, t]
                    + duration * 3600 * future_guard_stress * _EXPOSURE_SCALE,
                    name=f"naumann_calendar_guard_balance[s{s},t{t}]")
            model.addConstr(cycle_exposure[s, t + 1] == cycle_exposure[s, t] + variables["naumann_cycle_increment"][t, owner],
                name=f"naumann_cycle_balance[s{s},t{t}]")
            internal_power = (battery.eta_charge * variables["charge"][t, owner]
                              + variables["discharge"][t, owner] / battery.eta_discharge)
            model.addConstr(internal_power <= settings.naumann.effective_c_rate_limit * daily_capacity[s, t],
                name=f"naumann_effective_rate[s{s},t{t}]")
            for bank in range(2):
                for j, width in enumerate(capacities):
                    key = (t, owner, bank, j)
                    model.addConstr(bank_energy[s, t + 1, bank, j] == bank_energy[s, t, bank, j]
                        + variables["naumann_bank_in"][key] - variables["naumann_bank_out"][key],
                        name=f"naumann_bank_balance[s{s},t{t},b{bank},j{j}]")
                    if j < len(capacities) - 1:
                        model.addConstr(bank_energy[s, t + 1, bank, j] >= float(width) * variables["naumann_bank_fill"][key],
                            name=f"naumann_bank_fill[s{s},t{t},b{bank},j{j}]")
                        model.addConstr(bank_energy[s, t + 1, bank, j] <= float(width) * (1 - variables["naumann_bank_empty"][key]),
                            name=f"naumann_bank_empty[s{s},t{t},b{bank},j{j}]")
    variables.update(naumann_bank_energy=bank_energy,
                     naumann_calendar_exposure=calendar_exposure,
                     naumann_cycle_exposure=cycle_exposure,
                     naumann_calendar_loss=calendar_loss,
                     naumann_cycle_loss=cycle_loss,
                     naumann_calendar_increment=calendar_increment,
                     naumann_calendar_guard_exposure=calendar_guard_exposure,
                     naumann_calendar_guard_loss=calendar_guard_loss,
                     naumann_health_lower=health_lower,
                     naumann_capacity_mwh=daily_capacity)
    return {s: wear_value(settings) * (calendar_loss[s, intervals] + cycle_loss[s, intervals]
        - calendar_loss[s, 0] - cycle_loss[s, 0]) for s in range(count)}


def extract_naumann_arrays(inputs: OptimizationInput, variables: dict[str, Any]) -> dict[str, np.ndarray]:
    """Expose solved aging states with the same leading scenario axis everywhere."""
    count, intervals = np.shape(inputs.scenarios.pv_ac_mw)
    segments = len(inputs.degradation.naumann.depth_knots) - 1
    _, multipliers = rate_bin_parameters(inputs.degradation)
    arrays = {}
    boundary_maps = {"naumann_calendar_exposure": "naumann_calendar_exposure",
                     "naumann_cycle_exposure": "naumann_cycle_exposure",
                     "naumann_calendar_loss_pwl": "naumann_calendar_loss",
                     "naumann_cycle_loss_pwl": "naumann_cycle_loss",
                     "naumann_calendar_guard_exposure": "naumann_calendar_guard_exposure",
                     "naumann_calendar_guard_loss_pwl": "naumann_calendar_guard_loss",
                     "naumann_health_lower": "naumann_health_lower",
                     "naumann_capacity_mwh": "naumann_capacity_mwh"}
    for output, source in boundary_maps.items():
        arrays[output] = np.array([[_value(variables[source][s, t])
            if variables[source][s, t] is not None else np.nan
            for t in range(intervals + 1)] for s in range(count)], dtype=float)
        if output in ("naumann_calendar_exposure", "naumann_cycle_exposure",
                      "naumann_calendar_guard_exposure"):
            arrays[output] /= _EXPOSURE_SCALE
    # Fill diagnostic-only loss values using the identical piecewise graph.
    exposure_x, loss_y, sqrt_error = sqrt_graph(inputs.degradation)
    for loss, exposure in (("naumann_calendar_loss_pwl", "naumann_calendar_exposure"),
                           ("naumann_cycle_loss_pwl", "naumann_cycle_exposure"),
                           ("naumann_calendar_guard_loss_pwl", "naumann_calendar_guard_exposure")):
        missing = np.isnan(arrays[loss])
        arrays[loss][missing] = np.interp(arrays[exposure][missing], exposure_x, loss_y)
    missing = np.isnan(arrays["naumann_health_lower"])
    reconstructed = (1 - arrays["naumann_calendar_guard_loss_pwl"]
                     - arrays["naumann_cycle_loss_pwl"] - 2 * sqrt_error)
    arrays["naumann_health_lower"][missing] = reconstructed[missing]
    arrays["naumann_bank_energy_mwh"] = np.array([[[[
        _value(variables["naumann_bank_energy"][s, t, bank, j])
        for j in range(segments)] for bank in range(2)]
        for t in range(intervals + 1)] for s in range(count)])
    for output, source in (("naumann_bank_in_mwh", "naumann_bank_in"),
                           ("naumann_bank_out_mwh", "naumann_bank_out")):
        arrays[output] = np.array([[[[_value(variables[source][t, inputs.tree.control_nodes[s][t], bank, j])
            for j in range(segments)] for bank in range(2)]
            for t in range(intervals)] for s in range(count)])
    for name in ("naumann_depth_dose", "naumann_cycle_increment"):
        arrays[name] = np.array([[_value(variables[name][t, inputs.tree.control_nodes[s][t]])
            for t in range(intervals)] for s in range(count)]) / _EXPOSURE_SCALE
    arrays["naumann_calendar_increment"] = np.array([[
        _value(variables["naumann_calendar_increment"][s, t])
        for t in range(intervals)] for s in range(count)])
    arrays["naumann_rate_bin_weight"] = np.array([[[_value(
        variables["naumann_rate_bins"][t, inputs.tree.control_nodes[s][t], k])
        for k in range(len(multipliers))] for t in range(intervals)] for s in range(count)])
    arrays["naumann_rate_bin"] = np.argmax(arrays["naumann_rate_bin_weight"], axis=2)
    return arrays


def naumann_degradation_from_arrays(inputs: OptimizationInput,
                                  arrays: dict[str, np.ndarray]) -> np.ndarray:
    """Recompute forecast wear from new PWL losses; historical wear stays sunk."""
    loss = arrays["naumann_calendar_loss_pwl"] + arrays["naumann_cycle_loss_pwl"]
    return wear_value(inputs.degradation) * (loss[:, -1] - loss[:, 0])


# Keep the explicit evaluate spelling available to component-accounting callers.
evaluate_naumann_degradation_from_arrays = naumann_degradation_from_arrays


def _bank_reference(initial, incoming, outgoing, capacities):
    """Replay shallow-first transfers without using solved bank allocations."""
    states = np.empty((len(incoming) + 1, len(capacities)))
    admitted = np.zeros((len(incoming), len(capacities)))
    removed = np.zeros_like(admitted)
    states[0] = initial
    for t in range(len(incoming)):
        remaining_in = float(incoming[t])
        remaining_out = float(outgoing[t])
        states[t + 1] = states[t]
        for j, width in enumerate(capacities):
            admitted[t, j] = min(remaining_in, max(0.0, width - states[t, j]))
            removed[t, j] = min(remaining_out, max(0.0, states[t, j]))
            states[t + 1, j] += admitted[t, j] - removed[t, j]
            remaining_in -= admitted[t, j]
            remaining_out -= removed[t, j]
        if max(remaining_in, remaining_out) > 1e-7:
            raise ValueError("Naumann dispatch exceeds its depth-memory bank")
    return states, admitted, removed


def validate_naumann_arrays(inputs: OptimizationInput, arrays: dict[str, np.ndarray],
                           tolerance: float = 1e-6) -> dict[str, float]:
    """Independently check all planner aging equations and carried memory."""
    battery = inputs.plant.battery
    settings = inputs.degradation
    health = _initial(inputs)
    capacities, marginal = naumann_segment_parameters(battery, settings)
    calendar_x, calendar_upper, _ = calendar_graph(settings)
    exposure_x, loss_y, sqrt_error = sqrt_graph(settings)
    edges, multipliers = rate_bin_parameters(settings)
    count, intervals = np.shape(inputs.scenarios.pv_ac_mw)
    shapes = {"naumann_bank_energy_mwh": (count, intervals + 1, 2, len(capacities)),
              "naumann_bank_in_mwh": (count, intervals, 2, len(capacities)),
              "naumann_bank_out_mwh": (count, intervals, 2, len(capacities)),
              "naumann_rate_bin_weight": (count, intervals, len(multipliers)),
              "naumann_rate_bin": (count, intervals),
              "naumann_depth_dose": (count, intervals),
              "naumann_calendar_increment": (count, intervals),
              "naumann_cycle_increment": (count, intervals)}
    for name in ("naumann_calendar_exposure", "naumann_cycle_exposure",
                 "naumann_calendar_loss_pwl", "naumann_cycle_loss_pwl",
                 "naumann_calendar_guard_exposure", "naumann_calendar_guard_loss_pwl",
                 "naumann_health_lower", "naumann_capacity_mwh"):
        shapes[name] = (count, intervals + 1)
    if any(name not in arrays or np.shape(arrays[name]) != shape
           or not np.all(np.isfinite(arrays[name])) for name, shape in shapes.items()):
        return {"naumann_arrays": float("inf")}
    errors = {}
    def equality(name, difference):
        errors[name] = float(np.max(np.abs(difference), initial=0.0))
    def upper(name, excess):
        errors[name] = float(max(0.0, np.max(excess, initial=0.0)))
    def exposure_equality(name, difference):
        # The generic validator uses an ordinary feasibility tolerance.  Measure
        # tiny squared-loss residuals in the same micro-units as the solver rows,
        # rather than allowing that tolerance to swallow a complete aging step.
        equality(name, _EXPOSURE_SCALE * difference)
    def exposure_upper(name, excess):
        upper(name, _EXPOSURE_SCALE * excess)
    duration = np.asarray(inputs.grid.delta_hours)
    incoming = arrays["charge_mw"] * battery.eta_charge * duration
    outgoing = arrays["discharge_mw"] * duration / battery.eta_discharge
    bank_energy = arrays["naumann_bank_energy_mwh"]
    bank_in = arrays["naumann_bank_in_mwh"]
    bank_out = arrays["naumann_bank_out_mwh"]
    reference_capacity = battery.capacity_mwh * health.committed_soh
    equality("naumann_bank_discharge_sum", bank_energy[:, :, 0].sum(axis=2) - arrays["energy_mwh"])
    equality("naumann_bank_charge_sum", bank_energy[:, :, 1].sum(axis=2) - battery.capacity_mwh + arrays["energy_mwh"])
    equality("naumann_bank_balance", np.diff(bank_energy, axis=1) - bank_in + bank_out)
    upper("naumann_bank_capacity", bank_energy - capacities)
    upper("naumann_bank_nonnegative", -bank_energy)
    # Compare each bank against a fresh numerical implementation of the forced
    # priority policy, starting from its supplied historical vector.
    try:
        for s in range(count):
            for bank, initial in enumerate((health.discharge_segment_energy_mwh,
                                            health.charge_segment_energy_mwh)):
                source_in = incoming[s] if bank == 0 else outgoing[s]
                source_out = outgoing[s] if bank == 0 else incoming[s]
                reference = _bank_reference(initial, source_in, source_out, capacities)
                for name, actual, expected in zip(("energy", "in", "out"),
                        (bank_energy[s, :, bank], bank_in[s, :, bank], bank_out[s, :, bank]), reference):
                    equality(f"naumann_bank_{name}_s{s}_b{bank}", actual - expected)
    except ValueError:
        errors["naumann_bank_evaluable"] = float("inf")
    weights = arrays["naumann_rate_bin_weight"]
    equality("naumann_rate_integral", weights - np.round(weights))
    equality("naumann_rate_one_bin", weights.sum(axis=2) - 1)
    upper("naumann_rate_nonnegative", -weights)
    upper("naumann_rate_weight_upper", weights - 1)
    equality("naumann_rate_selected", arrays["naumann_rate_bin"] - np.argmax(weights, axis=2))
    nominal_rate = (incoming + outgoing) / duration / battery.capacity_mwh
    upper("naumann_rate_lower", np.sum(weights * edges[:-1], axis=2) - nominal_rate)
    upper("naumann_rate_upper", nominal_rate - np.sum(weights * edges[1:], axis=2))
    dose = 0.5 * np.sum(bank_out * marginal, axis=(2, 3))
    exposure_equality("naumann_depth_dose", arrays["naumann_depth_dose"] - dose)
    selected = np.argmax(weights, axis=2)
    cycle_increment = dose * multipliers[selected]
    exposure_equality("naumann_cycle_increment", arrays["naumann_cycle_increment"] - cycle_increment)
    exposure_equality("naumann_cycle_initial", arrays["naumann_cycle_exposure"][:, 0] - health.cycle_exposure)
    exposure_equality("naumann_cycle_balance", np.diff(arrays["naumann_cycle_exposure"], axis=1) - cycle_increment)
    soc = arrays["energy_mwh"] / reference_capacity
    quadrature_soc = ((1 - _GAUSS_FRACTIONS) * soc[:, :-1, None]
                      + _GAUSS_FRACTIONS * soc[:, 1:, None])
    calendar_increment = duration * 3600 * np.sum(
        np.interp(quadrature_soc, calendar_x, calendar_upper) * _GAUSS_WEIGHTS, axis=2)
    exposure_equality("naumann_calendar_increment", arrays["naumann_calendar_increment"] - calendar_increment)
    exposure_equality("naumann_calendar_initial", arrays["naumann_calendar_exposure"][:, 0] - health.calendar_exposure)
    exposure_equality("naumann_calendar_balance", np.diff(arrays["naumann_calendar_exposure"], axis=1) - calendar_increment)
    days = _boundary_days(inputs)
    future_guard_stress = float(np.interp(battery.maximum_energy / battery.capacity_mwh,
                                         calendar_x, calendar_upper))
    guard_increment = calendar_increment.copy()
    for t in range(intervals):
        if days[t] != days[0]:
            guard_increment[:, t] = duration[t] * 3600 * future_guard_stress
    guard_exposure = arrays["naumann_calendar_guard_exposure"]
    exposure_equality("naumann_calendar_guard_initial", guard_exposure[:, 0] - health.calendar_exposure)
    exposure_equality("naumann_calendar_guard_balance", np.diff(guard_exposure, axis=1) - guard_increment)
    equality("naumann_calendar_guard_loss_graph", arrays["naumann_calendar_guard_loss_pwl"]
             - np.interp(guard_exposure, exposure_x, loss_y))
    exposure_upper("naumann_calendar_guard_exposure_lower", -guard_exposure)
    exposure_upper("naumann_calendar_guard_exposure_upper", guard_exposure - exposure_x[-1])
    for mode in ("calendar", "cycle"):
        exposure = arrays[f"naumann_{mode}_exposure"]
        equality(f"naumann_{mode}_loss_graph", arrays[f"naumann_{mode}_loss_pwl"]
                 - np.interp(exposure, exposure_x, loss_y))
        exposure_upper(f"naumann_{mode}_exposure_lower", -exposure)
        exposure_upper(f"naumann_{mode}_exposure_upper", exposure - exposure_x[-1])
    expected_health = (1 - arrays["naumann_calendar_guard_loss_pwl"]
                       - arrays["naumann_cycle_loss_pwl"] - 2 * sqrt_error)
    equality("naumann_health_lower", arrays["naumann_health_lower"] - expected_health)
    expected_capacity = np.full((count, intervals + 1), reference_capacity)
    for boundary in range(1, intervals + 1):
        expected_capacity[:, boundary] = expected_capacity[:, boundary - 1]
        if days[boundary] != days[boundary - 1]:
            expected_capacity[:, boundary] = battery.capacity_mwh * expected_health[:, boundary]
    equality("naumann_daily_capacity", arrays["naumann_capacity_mwh"] - expected_capacity)
    upper("naumann_energy_upper", arrays["energy_mwh"]
          - battery.maximum_energy / battery.capacity_mwh * expected_capacity)
    upper("naumann_energy_lower", battery.energy_min_mwh * health.committed_soh - arrays["energy_mwh"])
    upper("naumann_effective_rate", nominal_rate * battery.capacity_mwh
          - settings.naumann.effective_c_rate_limit * expected_capacity[:, :-1])
    # Every additional accounting action must share its physical command's
    # information owner, including paths with zero probability.
    for t in range(intervals):
        for owner in {row[t] for row in inputs.tree.control_nodes}:
            descendants = [s for s in range(count) if inputs.tree.control_nodes[s][t] == owner]
            for name in ("naumann_bank_in_mwh", "naumann_bank_out_mwh",
                         "naumann_rate_bin_weight", "naumann_depth_dose"):
                equality(f"{name}_shared_t{t}_{owner}",
                         arrays[name][descendants, t] - arrays[name][descendants[0], t])
    return errors


def naumann_planner_error_bounds(inputs: OptimizationInput,
                                 arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Bound graph/rate approximation for the same fixed calendar denominator.

    Actual future-day replay refreshes its SOC denominator at midnight.  That
    separate rolling-horizon approximation is deliberately not hidden inside
    these interpolation and rate-quantization bounds.
    """
    _, _, calendar_error = calendar_graph(inputs.degradation)
    _, _, loss_error = sqrt_graph(inputs.degradation)
    edges, factors = rate_bin_parameters(inputs.degradation)
    rate_settings = inputs.degradation.naumann
    lower_factors = (rate_settings.cycle_rate_slope * edges[:-1]
                     + rate_settings.cycle_rate_intercept) ** 2
    selected = np.asarray(arrays["naumann_rate_bin"], dtype=int)
    calendar_bound = np.full(len(inputs.scenarios.scenario_ids),
        float(np.sum(inputs.grid.delta_hours)) * 3600 * calendar_error)
    cycle_bound = np.sum(arrays["naumann_depth_dose"]
        * (factors[selected] - lower_factors[selected]), axis=1)
    # The square root is 1/2-Hölder even at a pristine battery, so this remains
    # valid without assuming a strictly positive historical exposure.
    health_bound = np.sqrt(np.maximum(calendar_bound, 0)) + np.sqrt(np.maximum(cycle_bound, 0)) + 4 * loss_error
    return {"calendar_exposure_bound": calendar_bound,
            "cycle_exposure_bound": cycle_bound,
            "wear_health_bound": health_bound,
            "wear_eur_bound": health_bound * wear_value(inputs.degradation),
            "wear_eur": health_bound * wear_value(inputs.degradation)}
