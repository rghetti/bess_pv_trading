"""Causal forecast economics and delayed-PV-observation recourse with fixed trades.

Only the common first command is issued. Later commands form a conditional
policy on modeled PV histories; the representative path is a shadow projection,
never an executed promise or an observation of the future scenario identity.
"""

from __future__ import annotations

from time import perf_counter
from types import SimpleNamespace

import numpy as np

from ..domain import SolverSettings
from .degradation import evaluate_xu_dispatch
from .model import classify_status
from .physical import add_battery_controls, build_battery_degradation, energy_transition
from .mpc import MpcAction, MpcPlan, MpcResult


def control_history_nodes(inputs):
    """Partition future controls using disclosed PV prefixes, excluding price labels."""
    if inputs.policy != "ensemble_mpc_adaptive":
        return np.zeros(inputs.pv_ac_mw.shape, dtype=int)
    from ..scenario_research import observable_history_nodes

    settings = inputs.settings
    # Branches outside this shortened remaining-day solve are simply absent.
    stages = settings.recourse_stages
    if stages is not None:
        stages = tuple(t for t in stages if t < inputs.pv_ac_mw.shape[1])
    return observable_history_nodes(inputs.pv_ac_mw,
        revelation_lag_intervals=settings.recourse_observation_lag_intervals,
        branch_intervals=stages,
        observation_quantum_mw=settings.recourse_pv_bin_width_mw or None)


def weighted_loss_cvar(losses, weights, alpha):
    """Integrate the worst probability mass, including a fractional quantile atom."""
    values = np.asarray(losses, dtype=float)
    probability = np.asarray(weights, dtype=float)
    if (values.ndim != 1 or values.shape != probability.shape or not len(values)
            or not np.isfinite(values).all() or not np.isfinite(probability).all()
            or np.any(probability < 0) or not np.isclose(probability.sum(), 1)
            or not np.isfinite(alpha) or not 0 < alpha < 1):
        raise ValueError("CVaR requires finite losses, normalized weights and 0 < alpha < 1")
    order = np.argsort(-values, kind="stable")
    mass_before = np.r_[0.0, np.cumsum(probability[order])[:-1]]
    tail_mass = np.minimum(probability[order], np.maximum(1 - alpha - mass_before, 0))
    return float(np.dot(tail_mass, values[order]) / (1 - alpha))


def _terminal_cost(model, inputs, terminal_energy, scenario, gp):
    """Represent each path's configured inventory cost exactly, in EUR."""
    settings = inputs.settings
    mode = settings.terminal_inventory_mode
    if mode == "none":
        return gp.LinExpr()
    if mode == "linear_value":
        return -settings.terminal_inventory_value_eur_per_mwh * terminal_energy
    if mode == "soft_target":
        target = inputs.plant.battery.capacity_mwh * settings.terminal_soc_target_percent / 100
        deviation = model.addVar(name=f"terminal_deviation[s{scenario}]")
        model.addConstr(deviation >= terminal_energy - target)
        model.addConstr(deviation >= target - terminal_energy)
        return settings.terminal_inventory_penalty_eur_per_mwh * deviation
    # A general PWL constraint preserves nonconcave conditional value functions.
    value = model.addVar(lb=-gp.GRB.INFINITY, name=f"continuation_value[s{scenario}]")
    model.addGenConstrPWL(terminal_energy, value, inputs.terminal_value.energy_mwh.tolist(),
        inputs.terminal_value.value_eur.tolist(), name=f"continuation_curve[s{scenario}]")
    return -value


def build_research_model(inputs):
    """Build node-owned physical controls, pathwise losses and monetary CVaR."""
    import gurobipy as gp

    model = gp.Model("causal_adaptive_mpc")
    model.setParam("OutputFlag", 0)
    battery = inputs.plant.battery
    count, length = inputs.pv_ac_mw.shape
    nodes = control_history_nodes(inputs)
    charge, discharge, mode, utilization, energy = {}, {}, {}, {}, {}
    # Shared variables enforce nonanticipativity directly, without pairwise rows.
    for t in range(length):
        for owner in np.unique(nodes[:, t]):
            key = (t, int(owner))
            charge[key], discharge[key], mode[key] = add_battery_controls(model, battery, gp,
                f"t{t},n{owner}")
            utilization[key] = model.addVar(ub=1, name=f"pv_utilization[t{t},n{owner}]")
            if np.all(inputs.pv_ac_mw[nodes[:, t] == owner, t] == 0):
                model.addConstr(utilization[key] == 1)
    # Battery state depends only on the actual initial state and the history of
    # shared controls. Scenarios in one node therefore have exactly the same
    # next SOC. Alias that state rather than creating duplicate path equations.
    initial = model.addVar(lb=battery.energy_min_mwh, ub=battery.maximum_energy, name="energy[initial]")
    model.addConstr(initial == inputs.state.energy_mwh)
    for s in range(count):
        energy[s, 0] = initial
    for t, duration in enumerate(inputs.grid.delta_hours):
        for owner in np.unique(nodes[:, t]):
            members = np.flatnonzero(nodes[:, t] == owner)
            representative = int(members[0])
            # A node must never recombine different prior control histories.
            if any(energy[int(s), t] is not energy[representative, t] for s in members):
                raise ValueError("Adaptive control nodes merge distinct battery histories")
            next_energy = model.addVar(lb=battery.energy_min_mwh, ub=battery.maximum_energy,
                name=f"energy[t{t + 1},n{owner}]")
            model.addConstr(next_energy == energy_transition(energy[representative, t],
                charge[t, int(owner)], discharge[t, int(owner)], battery, duration))
            for s in members:
                energy[int(s), t + 1] = next_energy
            if t == length - 1:
                if inputs.hard_terminal_target_mwh is not None:
                    model.addConstr(next_energy == inputs.hard_terminal_target_mwh)
                if inputs.hard_terminal_minimum_mwh is not None:
                    model.addConstr(next_energy >= inputs.hard_terminal_minimum_mwh)
    variables = {"charge": charge, "discharge": discharge, "energy": energy,
        "utilization": utilization, "mode": mode, "nodes": nodes}
    context = SimpleNamespace(plant=inputs.plant, state=inputs.state, grid=inputs.grid,
        degradation=inputs.degradation, tree=SimpleNamespace(control_nodes=nodes),
        scenarios=SimpleNamespace(scenario_ids=inputs.scenario_ids, pv_ac_mw=inputs.pv_ac_mw))
    wear = build_battery_degradation(model, context, variables, gp)
    forecasts = inputs.price_forecast
    price_positive = None if forecasts is None else forecasts.positive_eur_per_mwh
    price_negative = None if forecasts is None else (
        forecasts.positive_eur_per_mwh if forecasts.negative_eur_per_mwh is None
        else forecasts.negative_eur_per_mwh)
    grid, positive, negative = {}, {}, {}
    absolute, imports, settlement, terminal, losses = {}, {}, {}, {}, {}
    # Settlement quantities are deviations from fixed commitments, never new bids.
    for s in range(count):
        absolute[s], imports[s], settlement[s] = gp.LinExpr(), gp.LinExpr(), gp.LinExpr()
        for t, duration in enumerate(inputs.grid.delta_hours):
            owner = int(nodes[s, t])
            grid[s, t] = model.addVar(lb=-inputs.plant.poi_import_mw, ub=inputs.plant.poi_export_mw,
                name=f"grid[s{s},t{t}]")
            model.addConstr(grid[s, t] == inputs.pv_ac_mw[s, t] * utilization[t, owner]
                + discharge[t, owner] - charge[t, owner])
            upper = max(0.0, inputs.plant.poi_export_mw * duration - inputs.committed_mwh[t])
            lower = max(0.0, inputs.plant.poi_import_mw * duration + inputs.committed_mwh[t])
            positive[s, t] = model.addVar(ub=upper, name=f"excess[s{s},t{t}]")
            negative[s, t] = model.addVar(ub=lower, name=f"shortfall[s{s},t{t}]")
            model.addConstr(positive[s, t] - negative[s, t]
                == grid[s, t] * duration - inputs.committed_mwh[t])
            # Equal directional prices cancel any split slack algebraically. Distinct
            # prices need a binary to prevent artificial simultaneous cash receipts.
            if forecasts is not None and price_positive[s, t] != price_negative[s, t]:
                sign = model.addVar(vtype=gp.GRB.BINARY, name=f"settlement_sign[s{s},t{t}]")
                model.addConstr(positive[s, t] <= upper * sign)
                model.addConstr(negative[s, t] <= lower * (1 - sign))
            absolute[s] += positive[s, t] + negative[s, t]
            if inputs.import_fee_eur_per_mwh:
                imported = model.addVar(ub=inputs.plant.poi_import_mw, name=f"import[s{s},t{t}]")
                model.addConstr(imported >= -grid[s, t])
                imports[s] += inputs.import_fee_eur_per_mwh * duration * imported
            if forecasts is not None:
                settlement[s] += price_positive[s, t] * positive[s, t] - price_negative[s, t] * negative[s, t]
        terminal[s] = _terminal_cost(model, inputs, energy[s, length], s, gp)
        # Risk concerns pathwise monetary operating loss. Artificial imbalance
        # penalties and terminal salvage are deliberately excluded from this score.
        losses[s] = wear[s] + imports[s] - settlement[s]
    expressions = {name: gp.quicksum(float(inputs.weights[s]) * values[s] for s in range(count))
        for name, values in (("absolute_imbalance_mwh", absolute), ("degradation_eur", wear),
            ("import_cost_eur", imports), ("terminal_cost_eur", terminal), ("settlement_eur", settlement))}
    expressions["monetary_cvar_eur"] = gp.LinExpr()
    if inputs.settings.monetary_cvar_weight:
        threshold = model.addVar(lb=-gp.GRB.INFINITY, name="monetary_loss_var_eur")
        tail = {}
        for s in range(count):
            tail[s] = model.addVar(name=f"monetary_tail[s{s}]")
            model.addConstr(tail[s] >= losses[s] - threshold)
        expressions["monetary_cvar_eur"] = threshold + gp.quicksum(
            float(inputs.weights[s]) * tail[s] for s in range(count)) / (1 - inputs.settings.monetary_cvar_alpha)
    variables.update(grid=grid, positive=positive, negative=negative, path_losses=losses)
    model.update()
    return model, variables, expressions


def _extract_plan(inputs, variables, expressions):
    """Retain conditional controls explicitly and label the representative path."""
    count, length = inputs.pv_ac_mw.shape
    nodes = variables["nodes"]
    commands = np.empty((count, length, 3))
    # Commands at the root are literally the same model variables for every path.
    for s in range(count):
        for t in range(length):
            key = (t, int(nodes[s, t]))
            commands[s, t] = (max(0.0, variables["charge"][key].X),
                max(0.0, variables["discharge"][key].X),
                float(np.clip(variables["utilization"][key].X, 0, 1)))
    energy = np.array([[variables["energy"][s, t].X for t in range(length + 1)] for s in range(count)])
    grid = np.array([[variables["grid"][s, t].X for t in range(length)] for s in range(count)])
    imbalance = grid * inputs.grid.delta_hours - inputs.committed_mwh
    actions = [MpcAction(product, inputs.cutoff.isoformat(), *commands[0, t])
        for t, product in enumerate(inputs.grid.interval_ids)]
    return MpcPlan(inputs.cutoff.isoformat(), actions, energy[0].copy(), grid, imbalance,
        inputs.weights @ np.abs(imbalance), inputs.weights @ np.maximum(-imbalance, 0),
        {name: float(value.getValue()) for name, value in expressions.items()},
        scenario_actions=commands, scenario_energy_mwh=energy,
        control_nodes=tuple(tuple(int(n) for n in row) for row in nodes), representative_path_index=0)


def validate_research_plan(inputs, plan, tolerance=1e-6):
    """Reconstruct every conditional path, nonanticipativity and monetary accounting."""
    count, length = inputs.pv_ac_mw.shape
    commands = np.asarray(plan.scenario_actions, dtype=float)
    energy = np.asarray(plan.scenario_energy_mwh, dtype=float)
    if (commands.shape != (count, length, 3) or energy.shape != (count, length + 1)
            or not np.isfinite(commands).all() or not np.isfinite(energy).all()
            or len(plan.actions) != length or np.shape(plan.energy_mwh) != (length + 1,)):
        raise ValueError("Research MPC plan needs complete finite scenario controls and states")
    nodes = control_history_nodes(inputs)
    if np.shape(plan.control_nodes) != nodes.shape or not np.array_equal(plan.control_nodes, nodes):
        raise ValueError("Research MPC control nodes differ from the admissible observation history")
    if plan.representative_path_index != 0:
        raise ValueError("Research MPC representative path must be explicitly indexed as zero")
    representative = np.array([[a.charge_mw, a.discharge_mw, a.pv_utilization_fraction] for a in plan.actions])
    if any(a.interval_id != product or a.issued_at != inputs.cutoff.isoformat()
           for a, product in zip(plan.actions, inputs.grid.interval_ids)):
        raise ValueError("Research MPC action identity differs from the decision grid")
    battery = inputs.plant.battery
    charge, discharge, utilization = commands[:, :, 0], commands[:, :, 1], commands[:, :, 2]
    predicted = energy_transition(energy[:, :-1], charge, discharge, battery, inputs.grid.delta_hours)
    grid = inputs.pv_ac_mw * utilization + discharge - charge
    imbalance = grid * inputs.grid.delta_hours - inputs.committed_mwh
    residuals = {
        "initial_energy": np.max(np.abs(energy[:, 0] - inputs.state.energy_mwh)),
        "energy_transition": np.max(np.abs(predicted - energy[:, 1:])),
        "energy_lower": np.max(battery.energy_min_mwh - energy),
        "energy_upper": np.max(energy - battery.maximum_energy),
        "power_lower": np.max(-commands[:, :, :2]),
        "power_upper": np.max(commands[:, :, :2] - battery.max_power_mw),
        "exclusivity": np.max(np.minimum(charge, discharge)),
        "utilization_lower": np.max(-utilization), "utilization_upper": np.max(utilization - 1),
        "poi_import": np.max(-grid - inputs.plant.poi_import_mw),
        "poi_export": np.max(grid - inputs.plant.poi_export_mw),
        "grid": np.max(np.abs(grid - plan.grid_mw)),
        "imbalance": np.max(np.abs(imbalance - plan.imbalance_mwh)),
        "representative_controls": np.max(np.abs(representative - commands[0])),
        "representative_energy": np.max(np.abs(plan.energy_mwh - energy[0])),
        "expected_absolute": np.max(np.abs(inputs.weights @ np.abs(imbalance)
            - plan.expected_absolute_imbalance_by_interval_mwh)),
        "expected_shortfall": np.max(np.abs(inputs.weights @ np.maximum(-imbalance, 0)
            - plan.expected_shortfall_by_interval_mwh)),
        "nonanticipativity": 0.0}
    # Equal observed prefixes require equal complete physical commands.
    for t in range(length):
        for owner in np.unique(nodes[:, t]):
            values = commands[nodes[:, t] == owner, t]
            residuals["nonanticipativity"] = max(residuals["nonanticipativity"],
                float(np.max(np.abs(values - values[0]))))
    if inputs.hard_terminal_target_mwh is not None:
        residuals["terminal_target"] = np.max(np.abs(energy[:, -1] - inputs.hard_terminal_target_mwh))
    if inputs.hard_terminal_minimum_mwh is not None:
        residuals["terminal_floor"] = np.max(inputs.hard_terminal_minimum_mwh - energy[:, -1])
    if not all(np.isfinite(value) for value in residuals.values()) or max(residuals.values()) > tolerance:
        raise ValueError(f"Research MPC physical validation failed: {residuals}")
    wear = inputs.degradation.cost_eur_per_mwh_throughput * np.sum(
        (charge + discharge) * inputs.grid.delta_hours, axis=1)
    if inputs.degradation.mode == "xu_segmented":
        wear = np.array([evaluate_xu_dispatch(battery=battery, settings=inputs.degradation,
            state=inputs.state, charge_mw=charge[s], discharge_mw=discharge[s],
            delta_hours=inputs.grid.delta_hours)["degradation_eur"] for s in range(count)])
    imports = inputs.import_fee_eur_per_mwh * np.sum(np.maximum(-grid, 0) * inputs.grid.delta_hours, axis=1)
    settlement = np.zeros(count)
    if inputs.price_forecast is not None:
        positive = inputs.price_forecast.positive_eur_per_mwh
        negative = inputs.price_forecast.negative_eur_per_mwh
        negative = positive if negative is None else negative
        settlement = np.sum(np.maximum(imbalance, 0) * positive - np.maximum(-imbalance, 0) * negative, axis=1)
    settings = inputs.settings
    terminal = np.zeros(count)
    if settings.terminal_inventory_mode == "soft_target":
        target = battery.capacity_mwh * settings.terminal_soc_target_percent / 100
        terminal = settings.terminal_inventory_penalty_eur_per_mwh * np.abs(energy[:, -1] - target)
    elif settings.terminal_inventory_mode == "linear_value":
        terminal = -settings.terminal_inventory_value_eur_per_mwh * energy[:, -1]
    elif settings.terminal_inventory_mode == "piecewise_linear":
        terminal = -np.interp(energy[:, -1], inputs.terminal_value.energy_mwh, inputs.terminal_value.value_eur)
    cvar = weighted_loss_cvar(wear + imports - settlement, inputs.weights, settings.monetary_cvar_alpha)
    components = {"absolute_imbalance_mwh": float(inputs.weights @ np.abs(imbalance).sum(axis=1)),
        "degradation_eur": float(inputs.weights @ wear), "import_cost_eur": float(inputs.weights @ imports),
        "terminal_cost_eur": float(inputs.weights @ terminal), "settlement_eur": float(inputs.weights @ settlement),
        "monetary_cvar_eur": cvar}
    return components, {name: float(value) for name, value in residuals.items()}


def solve_research_mpc(inputs, solver=None):
    """Solve forecast economics with one time budget and validate before issuance."""
    solver = solver or SolverSettings()
    started = perf_counter()
    settings = inputs.settings
    diagnostics = {"cutoff_utc": inputs.cutoff.isoformat(), "controller": inputs.policy,
        "price_oracle_enabled": False, "price_forecast_enabled": inputs.price_forecast is not None,
        "forecast_provenance": dict(inputs.forecast_provenance),
        "price_forecast_provenance": None if inputs.price_forecast is None else dict(inputs.price_forecast.provenance),
        "price_forecast_available_at_utc": None if inputs.price_forecast is None else inputs.price_forecast.available_at.isoformat(),
        "terminal_value_provenance": None if inputs.terminal_value is None else dict(inputs.terminal_value.provenance),
        "scenario_ids": list(inputs.scenario_ids), "scenario_weights": inputs.weights.tolist(),
        "scenario_count": len(inputs.weights), "horizon_intervals": len(inputs.grid.interval_ids),
        "horizon_start_utc": str(inputs.grid.starts[0]), "horizon_end_utc": str(inputs.grid.ends[-1]),
        "horizon_hours": float(inputs.grid.delta_hours.sum()),
        "current_pv_information": settings.current_pv_information,
        "current_interval_pv_oracle_enabled": inputs.current_realized_pv_mw is not None,
        "current_realized_pv_mw": inputs.current_realized_pv_mw,
        "recourse_observation_features": "pv_only_no_settlement_price_disclosure",
        "recourse_observation_lag_intervals": settings.recourse_observation_lag_intervals,
        "representative_path_index": 0, "representative_path_role": "shadow_projection_only",
        "monetary_cvar_alpha": settings.monetary_cvar_alpha, "monetary_cvar_weight": settings.monetary_cvar_weight,
        "monetary_loss_definition": "degradation_plus_import_cost_minus_settlement_excludes_terminal_and_penalty",
        "absolute_mip_gap_configured": False, "planning_objective_units": "EUR", "stages": [],
        "fallback_applied": False}
    result = MpcResult("solver_error", diagnostics=diagnostics)
    model = None
    try:
        import gurobipy as gp

        model, variables, expressions = build_research_model(inputs)
        diagnostics["build_seconds"] = perf_counter() - started
        diagnostics["model_variables"] = model.NumVars
        diagnostics["model_binary_variables"] = model.NumBinVars
        diagnostics["model_constraints"] = model.NumConstrs
        nodes = variables["nodes"]
        diagnostics["control_node_count"] = sum(len(np.unique(nodes[:, t])) for t in range(nodes.shape[1]))
        diagnostics["adaptive_recourse_enabled"] = inputs.policy == "ensemble_mpc_adaptive"
        for name, value in {"OutputFlag": int(solver.output_flag), "Threads": solver.threads,
                "MIPGap": solver.mip_gap, "Seed": solver.seed,
                "FeasibilityTol": solver.feasibility_tolerance, "IntFeasTol": solver.integrality_tolerance}.items():
            if value is not None:
                model.setParam(name, value)
        penalty = (settings.economic_imbalance_penalty_eur_per_mwh if inputs.price_forecast is not None
                   else settings.planning_imbalance_penalty_eur_per_mwh)
        objective = (penalty * expressions["absolute_imbalance_mwh"] + expressions["degradation_eur"]
            + expressions["import_cost_eur"] + expressions["terminal_cost_eur"] - expressions["settlement_eur"]
            + settings.monetary_cvar_weight * expressions["monetary_cvar_eur"])
        model.setObjective(objective, gp.GRB.MINIMIZE)
        remaining = settings.solver_time_limit_seconds - (perf_counter() - started)
        if remaining <= 0:
            result.status = "build_time_limit"
            return result
        model.setParam("TimeLimit", remaining)
        model.optimize()
        result.status = classify_status(model.Status, model.SolCount)
        diagnostics["stages"].append({"stage": "causal_economics" if inputs.price_forecast is not None else "adaptive_price_blind",
            "status": result.status, "runtime_seconds": float(model.Runtime),
            "objective": float(model.ObjVal) if model.SolCount else None,
            "bound": float(model.ObjBound) if model.SolCount else None,
            "gap": float(model.MIPGap) if model.SolCount and np.isfinite(model.MIPGap) else None})
        if not model.SolCount:
            return result
        plan = _extract_plan(inputs, variables, expressions)
        components, residuals = validate_research_plan(inputs, plan, max(1e-7, 2 * model.Params.FeasibilityTol))
        recomputed = (penalty * components["absolute_imbalance_mwh"] + components["degradation_eur"]
            + components["import_cost_eur"] + components["terminal_cost_eur"] - components["settlement_eur"]
            + settings.monetary_cvar_weight * components["monetary_cvar_eur"])
        # Check the complete monetary objective; inactive absolute-value and risk
        # epigraphs may be slack, so validate actual physics-based totals instead.
        scale = max(1.0, abs(recomputed), abs(float(model.ObjVal)))
        allowance = max(1e-6, 10 * model.Params.FeasibilityTol * scale)
        if abs(recomputed - model.ObjVal) > allowance:
            raise ValueError(f"Research MPC objective differs from reconstructed cash and wear: {model.ObjVal} vs {recomputed}")
        plan.components = components
        result.plan = plan
        diagnostics.update(physical_residuals=residuals, components=components,
            recomputed_components=components, planning_objective_value=float(recomputed),
            accepted_action_source="causal_forecast_economics" if inputs.price_forecast is not None else "adaptive_price_blind")
    except Exception as error:
        result.status = "invalid_or_failed_solve"
        diagnostics["failure_reason"] = str(error)
        result.plan = None
    finally:
        diagnostics.update(solve_wall_seconds=perf_counter() - started, status=result.status,
            runtime_seconds=sum(row["runtime_seconds"] for row in diagnostics["stages"]),
            fallback_applied=result.plan is None)
        if model is not None:
            model.dispose()
    return result
