"""Calendar-day throughput-equivalent cycles based on AC energy and daily capacity."""
from datetime import time
from zoneinfo import ZoneInfo


def daily_cycle_capacity(state, battery):
    """Use the day's committed health, or nameplate capacity without health."""
    health = state.battery_health
    return battery.capacity_mwh * (health.committed_soh if health is not None else 1.)


def daily_cycle_budget(state, battery):
    """Return the active day's remaining AC-throughput budget."""
    if battery.daily_cycle_limit_efc is None:
        return None
    day = state.timestamp.astimezone(ZoneInfo(battery.daily_cycle_timezone)).date().isoformat()
    if state.daily_cycle_day is None:
        if state.origin != "known_delivery_start_assumption":
            raise ValueError("A cycle-limited carried state requires its daily throughput history")
        used = 0.0
    elif state.daily_cycle_day != day:
        raise ValueError("Carried daily throughput belongs to a different local day")
    else:
        used = state.daily_ac_throughput_mwh
    limit = 2 * daily_cycle_capacity(state, battery) * battery.daily_cycle_limit_efc
    if used > limit + 1e-8:
        raise ValueError("Carried daily throughput exceeds the configured cycle limit")
    return max(0.0, limit - used)


def cycle_interval_day(start, end, timezone):
    """Require one real local calendar day per decision interval."""
    zone = ZoneInfo(timezone)
    local_start = start.astimezone(zone)
    local_end = end.astimezone(zone)
    days_crossed = (local_end.date() - local_start.date()).days
    if days_crossed > 1 or days_crossed < 0 or (days_crossed == 1 and local_end.time() != time.min):
        raise ValueError("Cycle-limited intervals must be split at local midnight")
    return local_start.date().isoformat()


def add_daily_cycle_constraints(model, inputs, variables, gp):
    """Constrain each scenario's AC throughput by its real calendar day."""
    battery = inputs.plant.battery
    if battery.daily_cycle_limit_efc is None:
        return
    remaining = daily_cycle_budget(inputs.state, battery)
    groups = {}
    for index, (start, end) in enumerate(zip(inputs.grid.starts, inputs.grid.ends)):
        day = cycle_interval_day(start, end, battery.daily_cycle_timezone)
        groups.setdefault(day, []).append(index)
    future_capacity = variables.get("linear_capacity_mwh", variables.get("naumann_capacity_mwh"))
    if inputs.degradation.mode in {"linear_soh", "naumann_lfp"} and future_capacity is None:
        raise ValueError("Cycle-limited health planning requires forecast capacity")
    for scenario in range(len(inputs.scenarios.scenario_ids)):
        for group_index, (day, intervals) in enumerate(groups.items()):
            # The current day is fixed by actual SOH; future days use that
            # scenario's forecast capacity at the corresponding local midnight.
            if group_index == 0:
                budget = remaining
            elif future_capacity is None:
                budget = 2 * battery.capacity_mwh * battery.daily_cycle_limit_efc
            else:
                budget = 2 * battery.daily_cycle_limit_efc * future_capacity[scenario, intervals[0]]
            terms = []
            for interval in intervals:
                owner = inputs.tree.control_nodes[scenario][interval]
                duration = inputs.grid.delta_hours[interval]
                terms.append(duration * (variables["charge"][interval, owner]
                    + variables["discharge"][interval, owner]))
            model.addConstr(gp.quicksum(terms) <= budget,
                            name=f"daily_cycle_limit[s{scenario},day{day}]")


def daily_cycle_excess(inputs, charge, discharge, arrays):
    """Recompute the largest daily AC-energy budget excess in MWh."""
    battery = inputs.plant.battery
    if battery.daily_cycle_limit_efc is None:
        return 0.0
    remaining = daily_cycle_budget(inputs.state, battery)
    grouped = {}
    for index, (start, end) in enumerate(zip(inputs.grid.starts, inputs.grid.ends)):
        day = cycle_interval_day(start, end, battery.daily_cycle_timezone)
        grouped.setdefault(day, []).append(index)
    capacity = arrays.get("linear_capacity_mwh", arrays.get("naumann_capacity_mwh"))
    if inputs.degradation.mode in {"linear_soh", "naumann_lfp"} and capacity is None:
        raise ValueError("Cycle-limited health validation requires forecast capacity")
    excess = 0.0
    for day_index, indices in enumerate(grouped.values()):
        for scenario in range(len(inputs.scenarios.scenario_ids)):
            if day_index == 0:
                budget = remaining
            elif capacity is None:
                budget = 2 * battery.capacity_mwh * battery.daily_cycle_limit_efc
            else:
                budget = 2 * battery.daily_cycle_limit_efc * capacity[scenario, indices[0]]
            used = sum(inputs.grid.delta_hours[t] * (charge[scenario, t]
                       + discharge[scenario, t]) for t in indices)
            excess = max(excess, float(used - budget))
    return excess
