"""Shared AC battery equations used by auction planning, MPC and execution."""

from __future__ import annotations


def retained_energy(energy, battery, duration):
    """Apply the configured hourly retention to a boundary energy or expression."""
    return energy * (1 - battery.self_discharge_per_hour) ** duration


def energy_transition(energy, charge, discharge, battery, duration):
    """Advance internal MWh from AC MW without applying PV conversion losses."""
    return (retained_energy(energy, battery, duration)
            + battery.eta_charge * charge * duration
            - discharge * duration / battery.eta_discharge)


def add_battery_controls(model, battery, gp, suffix):
    """Create one physically exclusive charge/discharge command pair."""
    charge = model.addVar(ub=battery.max_power_mw, name=f"charge[{suffix}]")
    discharge = model.addVar(ub=battery.max_power_mw, name=f"discharge[{suffix}]")
    mode = model.addVar(vtype=gp.GRB.BINARY, name=f"charge_mode[{suffix}]")
    model.addConstr(charge <= battery.max_power_mw * mode,
                    name=f"charge_mode_limit[{suffix}]")
    model.addConstr(discharge <= battery.max_power_mw * (1 - mode),
                    name=f"discharge_mode_limit[{suffix}]")
    return charge, discharge, mode


def build_battery_degradation(model, inputs, variables, gp):
    """Cost new common/node-owned dispatch using the existing degradation model."""
    from .degradation import build_xu_degradation

    if inputs.degradation.mode == "linear_soh":
        from .linear_soh import build_linear_health
        return build_linear_health(model, inputs, variables, gp)
    if inputs.degradation.mode == "linear_soh_fixed":
        from .linear_soh import cycle_loss_per_internal_mwh, linear_wear_value
        battery = inputs.plant.battery
        coefficient = cycle_loss_per_internal_mwh(battery, inputs.degradation) * linear_wear_value(inputs.degradation)
        return {s: coefficient * gp.quicksum(
            (battery.eta_charge * variables["charge"][t, inputs.tree.control_nodes[s][t]]
             + variables["discharge"][t, inputs.tree.control_nodes[s][t]] / battery.eta_discharge) * duration
            for t, duration in enumerate(inputs.grid.delta_hours))
            for s in range(len(inputs.scenarios.scenario_ids))}
    if inputs.degradation.mode == "naumann_lfp":
        from .naumann_milp import build_naumann_degradation

        return build_naumann_degradation(model, inputs, variables, gp)
    if inputs.degradation.mode == "xu_segmented":
        return build_xu_degradation(model, inputs, variables, gp)
    # Ownership defines sharing; scenario mass is applied by the caller, once.
    return {s: inputs.degradation.cost_eur_per_mwh_throughput * gp.quicksum(
        (variables["charge"][t, inputs.tree.control_nodes[s][t]]
         + variables["discharge"][t, inputs.tree.control_nodes[s][t]]) * duration
        for t, duration in enumerate(inputs.grid.delta_hours))
        for s in range(len(inputs.scenarios.scenario_ids))}
