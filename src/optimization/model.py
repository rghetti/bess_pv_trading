"""One deterministic-equivalent Gurobi MILP with node-owned decisions."""
from __future__ import annotations

from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from ..domain import BuiltModel, OptimizationInput, OptimizationResult, SolverSettings
from .degradation import build_xu_degradation
from .market_regime import configured_auction_product_groups, configured_dispatch_groups, market_product_groups
from .transaction_fees import build_transaction_fees
from ..transaction_fees import scenario_fee_details


class SolverUnavailableError(RuntimeError):
    """An installed solver cannot create a licensed optimization environment."""


def _physical(model: Any, inputs: OptimizationInput, gp: Any) -> dict[str, Any]:
    """Share battery commands and mode binaries at exactly their information node."""
    battery = inputs.plant.battery
    scenario_count, intervals = np.shape(inputs.scenarios.pv_ac_mw)
    owners = inputs.tree.control_nodes
    curtailment_owners = inputs.tree.curtailment_nodes or owners
    charge, discharge, mode, curtailment = {}, {}, {}, {}
    # Share battery commands and binary modes among scenarios with the same information.
    for t in range(intervals):
        # Each information node receives one mutually exclusive charge/discharge choice.
        for node in sorted({owners[s][t] for s in range(scenario_count)}):
            suffix = f"f{inputs.plant.farm_id},n{node},t{inputs.grid.interval_ids[t]}"
            charge[t, node] = model.addVar(ub=battery.max_power_mw, name=f"charge[{suffix}]")
            discharge[t, node] = model.addVar(ub=battery.max_power_mw, name=f"discharge[{suffix}]")
            mode[t, node] = model.addVar(vtype=gp.GRB.BINARY, name=f"charge_mode[{suffix}]")
            model.addConstr(charge[t, node] <= battery.max_power_mw * mode[t, node], name=f"charge_mode_limit[{suffix}]")
            model.addConstr(discharge[t, node] <= battery.max_power_mw * (1 - mode[t, node]), name=f"discharge_mode_limit[{suffix}]")
        # Shared curtailment cannot exceed the least PV available at this node.
        for node in sorted({curtailment_owners[s][t] for s in range(scenario_count)}):
            available = min(inputs.scenarios.pv_ac_mw[s, t] for s in range(scenario_count) if curtailment_owners[s][t] == node)
            curtailment[t, node] = model.addVar(ub=float(available), name=f"curtailment[n{node},t{inputs.grid.interval_ids[t]}]")
    energy, grid, pv_injection = {}, {}, {}
    # Track scenario-specific energy and injection while reusing the node-owned controls.
    for s, scenario in enumerate(inputs.scenarios.scenario_ids):
        # Include one extra energy boundary for the end of the delivery horizon.
        for t in range(intervals + 1):
            energy[s, t] = model.addVar(lb=battery.energy_min_mwh, ub=battery.maximum_energy, name=f"energy[s{scenario},boundary{t}]")
        model.addConstr(energy[s, 0] == inputs.state.energy_mwh, name=f"initial_energy[s{scenario}]")
        # Enforce an exact terminal target only when the study requests one.
        if inputs.terminal_target_mwh is not None:
            model.addConstr(energy[s, intervals] == inputs.terminal_target_mwh, name=f"terminal_energy[s{scenario}]")
        # A minimum terminal target is a separate, potentially weaker requirement.
        if inputs.terminal_minimum_mwh is not None:
            model.addConstr(energy[s, intervals] >= inputs.terminal_minimum_mwh,
                name=f"terminal_minimum_energy[s{scenario}]")
        # Balance battery energy, PV generation, curtailment and grid exchange each interval.
        for t, duration in enumerate(inputs.grid.delta_hours):
            suffix = f"s{scenario},t{inputs.grid.interval_ids[t]}"
            owner = owners[s][t]
            retention = (1 - battery.self_discharge_per_hour) ** duration
            model.addConstr(energy[s, t + 1] == retention * energy[s, t] + battery.eta_charge * charge[t, owner] * duration - discharge[t, owner] * duration / battery.eta_discharge, name=f"battery_balance[{suffix}]")
            grid[s, t] = model.addVar(lb=-inputs.plant.poi_import_mw, ub=inputs.plant.poi_export_mw, name=f"grid[{suffix}]")
            pv_injection[s, t] = model.addVar(lb=0.0,
                ub=float(inputs.scenarios.pv_ac_mw[s, t]), name=f"pv_injection[{suffix}]")
            model.addConstr(pv_injection[s, t] + curtailment[t, curtailment_owners[s][t]]
                == float(inputs.scenarios.pv_ac_mw[s, t]), name=f"pv_plant_balance[{suffix}]")
            model.addConstr(grid[s, t] == pv_injection[s, t] + discharge[t, owner]
                - charge[t, owner], name=f"poi_balance[{suffix}]")
    return {"charge": charge, "discharge": discharge, "mode": mode,
        "curtailment": curtailment, "pv_injection": pv_injection,
        "energy": energy, "grid": grid}


def _trading(model: Any, inputs: OptimizationInput, gp: Any) -> dict[str, Any]:
    """Apply POI-derived limits to each auction trade and each cumulative position."""
    count, intervals = np.shape(inputs.scenarios.pv_ac_mw)
    trades, positions, auction_revenue = {}, {}, {}
    auction_order = sorted(inputs.auction_eligibility, key=lambda a: ({"DA": 0, "IDA1": 1, "IDA2": 2, "IDA3": 3}.get(a, 4), a))
    # Create the trade variables for `auction` from `auction_order`.
    for auction in auction_order:
        # Create the trade variables for `node` from `sorted(set(inputs.tree.bid_nodes[auction]))`.
        for node in sorted(set(inputs.tree.bid_nodes[auction])):
            # Create the trade variables for `(t, eligible)` from `enumerate(inputs.auction_eligibility[auction])`.
            for t, eligible in enumerate(inputs.auction_eligibility[auction]):
                # Create trade variables when `eligible`.
                if eligible:
                    duration = inputs.grid.delta_hours[t]
                    lower = -inputs.plant.poi_import_mw * duration
                    upper = inputs.plant.poi_export_mw * duration
                    trades[auction, node, t] = model.addVar(lb=lower, ub=upper,
                        name=f"trade[{auction},n{node},t{inputs.grid.interval_ids[t]}]")
    accepted = inputs.market.positions(inputs.grid)
    # When `inputs.pending_position_mwh is not None`, set `accepted` using `accepted + np.asarray(inputs.pending_position_mwh)`.
    if inputs.pending_position_mwh is not None:
        accepted = accepted + np.asarray(inputs.pending_position_mwh)
    # Build the position sum, position upper, position lower constraints for `s` from `range(count)`.
    for s in range(count):
        auction_revenue[s] = gp.LinExpr(0.0)
        # Build the position upper, position lower, position sum constraints for `(t, duration)` from `enumerate(inputs.grid.delta_hours)`.
        for t, duration in enumerate(inputs.grid.delta_hours):
            lower = -inputs.plant.poi_import_mw * duration
            upper = inputs.plant.poi_export_mw * duration
            expression = gp.LinExpr(float(accepted[t]))
            # Build the position upper, position lower constraints for `auction` from `auction_order`.
            for auction in auction_order:
                key = (auction, inputs.tree.bid_nodes[auction][s], t)
                # Apply the position upper, position lower constraint when `key in trades`.
                if key in trades:
                    expression = expression + trades[key]
                    auction_revenue[s] += float(inputs.scenarios.auction_prices_eur_per_mwh[auction][s, t]) * trades[key]
                    model.addConstr(expression <= upper, name=f"position_upper[{auction},s{s},t{t}]")
                    model.addConstr(expression >= lower, name=f"position_lower[{auction},s{s},t{t}]")
            positions[s, t] = model.addVar(lb=lower, ub=upper, name=f"net_position[s{s},t{t}]")
            model.addConstr(positions[s, t] == expression, name=f"position_sum[s{s},t{t}]")
    return {"trades": trades, "positions": positions, "auction_revenue": auction_revenue}


def _product_resolution_equalities(model: Any, inputs: OptimizationInput,
                                   variables: dict[str, Any]) -> dict[str, Any]:
    """Group controls only when requested; tie each auction's native products."""
    groups = configured_dispatch_groups(inputs.grid, inputs.decision_resolution_minutes,
                                        inputs.market_timezone)
    count = len(inputs.scenarios.scenario_ids)
    controls = inputs.tree.control_nodes
    trades = variables.get("trades", {})
    control_equalities = 0
    trade_equalities: dict[str, int] = {auction: 0 for auction in inputs.auction_eligibility}
    # Build the configured dispatch `label` g`group index` t`left` `right` n`left node` `right node` constraints for `(group_index, group)` from `enumerate(groups)`.
    for group_index, group in enumerate(groups):
        # Build the configured dispatch `label` g`group index` t`left` `right` n`left node` `right node` constraints for `(left, right)` from `zip(group, group[1:])`.
        for left, right in zip(group, group[1:]):
            control_pairs = {(controls[s][left], controls[s][right]) for s in range(count)}
            # Build the configured dispatch `label` g`group index` t`left` `right` n`left node` `right node` constraints for `(left_node, right_node)` from `sorted(control_pairs)`.
            for left_node, right_node in sorted(control_pairs):
                # Build the configured dispatch `label` g`group index` t`left` `right` n`left node` `right node` constraints for `(name, label)` from `(('charge', 'charge'), ('discharge', 'discharge'), ('mode', 'mode'))`.
                for name, label in (("charge", "charge"), ("discharge", "discharge"), ("mode", "mode")):
                    model.addConstr(variables[name][left, left_node] == variables[name][right, right_node],
                                    name=f"configured_dispatch_{label}_g{group_index}_t{left}_{right}_n{left_node}_{right_node}")
                    control_equalities += 1

    # Build the native product `auction` trade g`product index` t`left` `right` n`owner` constraints for `auction` from `inputs.auction_eligibility`.
    for auction in inputs.auction_eligibility:
        eligibility = inputs.auction_eligibility[auction]
        bid_nodes = inputs.tree.bid_nodes[auction]
        prices = inputs.scenarios.auction_prices_eur_per_mwh[auction]
        # Process each `(product_index, product_group)` in this collection.
        for product_index, product_group in enumerate(
                configured_auction_product_groups(inputs.grid, auction,
                    inputs.native_interval_minutes_by_auction.get(auction), inputs.market_timezone)):
            # Skip the current item when `len(product_group) <= 1 or not all((eligibility[index] for index in product_group))`.
            if len(product_group) <= 1 or not all(eligibility[index] for index in product_group):
                continue
            # Enforce the input contract: Native `auction` prices must be equal across represented subintervals.
            if any(not np.allclose(prices[:, product_group[0]], prices[:, index],
                                   rtol=0, atol=1e-9) for index in product_group[1:]):
                raise ValueError(f"Native {auction} prices must be equal across represented subintervals")
            # Build the native product `auction` trade g`product index` t`left` `right` n`owner` constraints for `(left, right)` from `zip(product_group, product_group[1:])`.
            for left, right in zip(product_group, product_group[1:]):
                # Build the native product `auction` trade g`product index` t`left` `right` n`owner` constraints for `owner` from `sorted(set(bid_nodes))`.
                for owner in sorted(set(bid_nodes)):
                    left_key, right_key = (auction, owner, left), (auction, owner, right)
                    # Skip the current item when `left_key not in trades or right_key not in trades`.
                    if left_key not in trades or right_key not in trades:
                        continue
                    model.addConstr(trades[left_key] == trades[right_key],
                                    name=f"native_product_{auction}_trade_g{product_index}_t{left}_{right}_n{owner}")
                    trade_equalities[auction] = trade_equalities.get(auction, 0) + 1
    return {"dispatch_groups": len(groups), "dispatch_equalities": control_equalities,
            "decision_resolution_minutes": inputs.decision_resolution_minutes,
            "da_trade_equalities": trade_equalities.get("DA", 0),
            "auction_trade_equalities": trade_equalities}


def _settlement(model: Any, inputs: OptimizationInput, variables: dict[str, Any], gp: Any) -> dict[str, Any]:
    """Optionally optimize deviations; otherwise require interval delivery exactly."""
    count, intervals = np.shape(inputs.scenarios.pv_ac_mw)
    positive, negative, sign, imported = {}, {}, {}, {}
    cashflow, penalty, other_costs = {}, {}, {}
    negative_prices = inputs.scenarios.imbalance_negative_prices_eur_per_mwh
    # When `negative_prices is None`, set `negative_prices` using `inputs.scenarios.imbalance_prices_eur_per_mwh`.
    if negative_prices is None:
        negative_prices = inputs.scenarios.imbalance_prices_eur_per_mwh
    # Build the contracted delivery constraints for `s` from `range(count)`.
    for s in range(count):
        cashflow[s], penalty[s], other_costs[s] = gp.LinExpr(), gp.LinExpr(), gp.LinExpr()
        # Build the contracted delivery constraints for `(t, duration)` from `enumerate(inputs.grid.delta_hours)`.
        for t, duration in enumerate(inputs.grid.delta_hours):
            # Apply the contracted delivery constraint when `not inputs.optimize_imbalances`.
            if not inputs.optimize_imbalances:
                model.addConstr(variables["grid"][s, t] * duration == variables["positions"][s, t],
                                name=f"contracted_delivery[s{s},t{t}]")
            else:
                _imbalance_interval(model, inputs, variables, gp, s, t, duration,
                    positive, negative, sign, cashflow, penalty, negative_prices)
            # Import fees remain physical operating costs in either mode.
            if inputs.grid_import_fee_eur_per_mwh:
                imported[s, t] = model.addVar(ub=inputs.plant.poi_import_mw, name=f"grid_import[s{s},t{t}]")
                direction = model.addVar(vtype=gp.GRB.BINARY, name=f"grid_import_direction[s{s},t{t}]")
                model.addConstr(imported[s, t] >= -variables["grid"][s, t])
                model.addConstr(imported[s, t] <= inputs.plant.poi_import_mw * direction)
                model.addConstr(imported[s, t] <= -variables["grid"][s, t] + inputs.plant.poi_export_mw * (1 - direction))
                other_costs[s] += inputs.grid_import_fee_eur_per_mwh * imported[s, t] * duration
    return {"imbalance_positive": positive, "imbalance_negative": negative, "imbalance_sign": sign, "imbalance_cashflow": cashflow, "imbalance_penalty": penalty, "other_costs": other_costs}


def _imbalance_interval(model, inputs, variables, gp, s, t, duration,
                        positive, negative, sign, cashflow, penalty, negative_prices):
    """Build the optional signed deviation split for one scenario and interval."""
    # Grid energy and contracted energy share the same asymmetric POI range.
    positive_bound = negative_bound = (inputs.plant.poi_export_mw + inputs.plant.poi_import_mw) * duration
    # When `inputs.imbalance_limit_mwh is not None`, set `positive_bound` using `min(positive_bound, inputs.imbalance_limit_mwh)`.
    if inputs.imbalance_limit_mwh is not None:
        positive_bound = min(positive_bound, inputs.imbalance_limit_mwh)
        negative_bound = min(negative_bound, inputs.imbalance_limit_mwh)
    positive[s, t] = model.addVar(ub=positive_bound, name=f"imbalance_positive[s{s},t{t}]")
    negative[s, t] = model.addVar(ub=negative_bound, name=f"imbalance_negative[s{s},t{t}]")
    sign[s, t] = model.addVar(vtype=gp.GRB.BINARY, name=f"imbalance_sign[s{s},t{t}]")
    model.addConstr(positive[s, t] <= positive_bound * sign[s, t], name=f"imbalance_positive_only[s{s},t{t}]")
    model.addConstr(negative[s, t] <= negative_bound * (1 - sign[s, t]), name=f"imbalance_negative_only[s{s},t{t}]")
    model.addConstr(positive[s, t] - negative[s, t] == variables["grid"][s, t] * duration - variables["positions"][s, t], name=f"settlement_balance[s{s},t{t}]")
    cashflow[s] += float(inputs.scenarios.imbalance_prices_eur_per_mwh[s, t]) * positive[s, t] - float(negative_prices[s, t]) * negative[s, t]
    penalty[s] += inputs.imbalance_penalty_eur_per_mwh * (positive[s, t] + negative[s, t])


def _degradation(model: Any, inputs: OptimizationInput, variables: dict[str, Any], gp: Any) -> dict[int, Any]:
    """Cost new throughput or depth-dependent discharge; carried aging is sunk."""
    # Return `build_xu_degradation(model, inputs, variables, gp)` when `inputs.degradation.mode == 'xu_segmented'`.
    if inputs.degradation.mode == "xu_segmented":
        return build_xu_degradation(model, inputs, variables, gp)
    return {s: inputs.degradation.cost_eur_per_mwh_throughput * gp.quicksum((variables["charge"][t, inputs.tree.control_nodes[s][t]] + variables["discharge"][t, inputs.tree.control_nodes[s][t]]) * duration for t, duration in enumerate(inputs.grid.delta_hours)) for s in range(len(inputs.scenarios.scenario_ids))}


def build_model(inputs: OptimizationInput) -> BuiltModel:
    """Build one stochastic program; no data reads, fitting or plotting occur here."""
    started = perf_counter()
    try:
        import gurobipy as gp
        model = gp.Model(f"pv_bess_farm_{inputs.plant.farm_id}")
    except ImportError as error:
        raise SolverUnavailableError("gurobipy is not installed in the active environment") from error
    except Exception as error:
        raise SolverUnavailableError(f"Gurobi environment/license unavailable: {error}") from error
    variables = _physical(model, inputs, gp)
    commercial = _trading(model, inputs, gp)
    variables.update({key: commercial[key] for key in ("trades", "positions")})
    variables["product_resolution_constraints"] = _product_resolution_equalities(model, inputs, variables)
    # Contracts must be backed by planned net AC energy when this policy is
    # enabled. Native-hour balancing preserves natural quarter-hour PV variation.
    resolution = inputs.planned_delivery_balance_resolution_minutes
    # Apply the planned delivery balance constraint when `resolution is not None`.
    if resolution is not None:
        groups = market_product_groups(inputs.grid, resolution, inputs.market_timezone)
        # Build the planned delivery balance constraints for `s` from `range(len(inputs.scenarios.scenario_ids))`.
        for s in range(len(inputs.scenarios.scenario_ids)):
            # Build the planned delivery balance constraints for `(group_index, group)` from `enumerate(groups)`.
            for group_index, group in enumerate(groups):
                model.addConstr(gp.quicksum(variables["grid"][s, t] * inputs.grid.delta_hours[t]
                    - variables["positions"][s, t] for t in group) == 0,
                    name=f"planned_delivery_balance[s{s},g{group_index}]")
    settlement = _settlement(model, inputs, variables, gp)
    variables.update({key: settlement[key] for key in ("imbalance_positive", "imbalance_negative", "imbalance_sign")})
    components = {"auction_revenue": commercial["auction_revenue"], "imbalance_cashflow": settlement["imbalance_cashflow"], "imbalance_penalty": settlement["imbalance_penalty"], "battery_degradation_cost": _degradation(model, inputs, variables, gp), "other_costs": settlement["other_costs"],
        "transaction_fees": build_transaction_fees(model, inputs, variables, gp)}
    profit = {s: components["auction_revenue"][s] + components["imbalance_cashflow"][s] - components["imbalance_penalty"][s] - components["battery_degradation_cost"][s] - components["other_costs"][s] - components["transaction_fees"][s] for s in range(len(inputs.scenarios.scenario_ids))}
    components["operating_profit"] = profit
    model.setObjective(gp.quicksum(float(probability) * profit[s] for s, probability in enumerate(inputs.scenarios.probabilities)), gp.GRB.MAXIMIZE)
    model.update() #imports the changes to the model
    return BuiltModel(inputs, model, variables, components, perf_counter() - started)


def classify_status(status: int, solution_count: int) -> str:
    """Separate termination cause from presence of a usable incumbent."""
    # Return `'optimal' if solution_count else 'no_incumbent'` when `status == 2`.
    if status == 2:
        return "optimal" if solution_count else "no_incumbent"
    # Return `'infeasible'` when `status == 3`.
    if status == 3:
        return "infeasible"
    # Return `'unbounded'` when `status == 5`.
    if status == 5:
        return "unbounded"
    # Return `'infeasible_or_unbounded'` when `status == 4`.
    if status == 4:
        return "infeasible_or_unbounded"
    # Return `'feasible_time_limit' if solution_count else 'time_limit_no_incumbent'` when `status == 9`.
    if status == 9:
        return "feasible_time_limit" if solution_count else "time_limit_no_incumbent"
    return "feasible_terminated" if solution_count else "no_incumbent"


def solve_model(built: BuiltModel, settings: SolverSettings | None = None) -> OptimizationResult:
    """Apply settings, solve, and extract only a confirmed incumbent."""
    settings = settings or SolverSettings()
    model, inputs = built.model, built.inputs
    model.setParam("OutputFlag", int(settings.output_flag))
    # Process each `(name, value)` in this collection.
    for name, value in {"Threads": settings.threads, "TimeLimit": settings.time_limit_seconds, "MIPGap": settings.mip_gap, "Seed": settings.seed}.items():
        model.setParam(name, value)
    # Process each `(name, value)` in this collection.
    for name, value in (("FeasibilityTol", settings.feasibility_tolerance),
                        ("IntFeasTol", settings.integrality_tolerance)):
        # Apply settings, solve, and extract only a confirmed incumbent when `value is not None`.
        if value is not None:
            model.setParam(name, value)
    # Apply settings, solve, and extract only a confirmed incumbent when `settings.export_path`.
    if settings.export_path:
        Path(settings.export_path).parent.mkdir(parents=True, exist_ok=True)
        model.write(settings.export_path)
    started = perf_counter()
    optimize_runtimes = []

    def optimize_once():
        """Retain every native Runtime, including the ambiguity-resolution retry."""
        try:
            model.optimize()
        except Exception:
            optimize_runtimes.append(None)
            raise
        optimize_runtimes.append(float(model.Runtime))

    try:
        optimize_once()
        # Apply settings, solve, and extract only a confirmed incumbent when `model.Status == 4`.
        if model.Status == 4:
            model.setParam("DualReductions", 0)
            optimize_once()
    except Exception as error:
        return OptimizationResult("solver_error", False, diagnostics={"build_seconds": built.build_seconds,
            "solve_seconds": perf_counter() - started, "runtime_seconds": None,
            "optimize_call_runtimes_seconds": optimize_runtimes}, message=str(error))
    diagnostics = {"status_code": model.Status, "solution_count": model.SolCount, "build_seconds": built.build_seconds, "solve_seconds": perf_counter() - started, "runtime_seconds": sum(optimize_runtimes), "optimize_call_runtimes_seconds": optimize_runtimes, "variables": model.NumVars, "binary_variables": model.NumBinVars, "constraints": model.NumConstrs, "nodes": len(inputs.tree.nodes), "leaves": len(inputs.scenarios.scenario_ids), "accepted_contractual_cashflow_eur": inputs.market.cashflow_eur, "pending_contractual_cashflow_in_objective": False, "pending_contractual_cashflow_status": "Unpublished-price contractual constant excluded; submitted quantities remain fixed positions"}
    diagnostics["relative_mip_gap_tolerance"] = settings.mip_gap
    diagnostics["feasibility_tolerance"] = model.Params.FeasibilityTol
    diagnostics["integrality_tolerance"] = model.Params.IntFeasTol
    diagnostics["absolute_mip_gap_configured"] = False
    diagnostics["degradation_mode"] = inputs.degradation.mode
    diagnostics["transaction_fee_formulation"] = built.variables["transaction_fee_formulation"]
    diagnostics["general_constraints"] = model.NumGenConstrs
    diagnostics["sunk_transaction_fees_eur"] = (
        inputs.market.transaction_fee_state or inputs.transaction_fees.initial_state).cumulative_fees_eur
    diagnostics["optimization_imbalances_enabled"] = inputs.optimize_imbalances
    diagnostics["product_resolution_constraints"] = built.variables.get(
        "product_resolution_constraints", {"dispatch_groups": 0, "dispatch_equalities": 0,
            "da_trade_equalities": 0, "auction_trade_equalities": {}})
    # Process each `(attribute, key)` in this collection.
    for attribute, key in (("ObjBound", "objective_bound_eur"), ("MIPGap", "mip_gap"), ("MaxVio", "max_violation")):
        try:
            value = float(getattr(model, attribute))
            diagnostics[key] = value if np.isfinite(value) else None
        except Exception:
            diagnostics[key] = None
    result = OptimizationResult(classify_status(model.Status, model.SolCount), model.SolCount > 0, diagnostics=diagnostics)
    # Apply settings, solve, and extract only a confirmed incumbent when `model.Status == 3 and settings.iis_path`.
    if model.Status == 3 and settings.iis_path:
        Path(settings.iis_path).parent.mkdir(parents=True, exist_ok=True)
        model.computeIIS()
        model.write(settings.iis_path)
        diagnostics["iis_path"] = settings.iis_path
    # Return `result` when `not result.has_incumbent`.
    if not result.has_incumbent:
        return result
    count, intervals = np.shape(inputs.scenarios.pv_ac_mw)
    variable = built.variables
    curtailment_owners = inputs.tree.curtailment_nodes or inputs.tree.control_nodes
    # Extract shared variables into leaf arrays for independent numerical validation.
    for source, destination in (("charge", "charge_mw"), ("discharge", "discharge_mw"), ("mode", "mode_charge")):
        result.arrays[destination] = np.array([[variable[source][t, inputs.tree.control_nodes[s][t]].X for t in range(intervals)] for s in range(count)])
    result.arrays["curtailment_mw"] = np.array([[variable["curtailment"][t, curtailment_owners[s][t]].X for t in range(intervals)] for s in range(count)])
    result.arrays["pv_injection_mw"] = np.array([[variable["pv_injection"][s, t].X
        for t in range(intervals)] for s in range(count)])
    # Process each `(source, destination)` in this collection.
    for source, destination in (("energy", "energy_mwh"), ("grid", "grid_mw"), ("positions", "net_position_mwh")):
        length = intervals + 1 if source == "energy" else intervals
        result.arrays[destination] = np.array([[variable[source][s, t].X for t in range(length)] for s in range(count)])
    # When `inputs.degradation.mode == 'xu_segmented'`, set `segments` using `inputs.degradation.segment_count`.
    if inputs.degradation.mode == "xu_segmented":
        segments = inputs.degradation.segment_count
        result.arrays["xu_segment_energy_mwh"] = np.array([[[variable["xu_segment_energy"][s, t, j].X for j in range(segments)] for t in range(intervals + 1)] for s in range(count)])
        # Process each `(source, destination)` in this collection.
        for source, destination in (("xu_segment_charge", "xu_segment_charge_mw"), ("xu_segment_discharge", "xu_segment_discharge_mw")):
            result.arrays[destination] = np.array([[[variable[source][t, inputs.tree.control_nodes[s][t], j].X for j in range(segments)] for t in range(intervals)] for s in range(count)])
    result.arrays["imbalance_mwh"] = result.arrays["grid_mw"] * inputs.grid.delta_hours - result.arrays["net_position_mwh"]
    # Process each `(source, destination, direction)` in this collection.
    for source, destination, direction in (("imbalance_positive", "imbalance_positive_mwh", 1), ("imbalance_negative", "imbalance_negative_mwh", -1)):
        result.arrays[destination] = (np.array([[variable[source][s, t].X for t in range(intervals)] for s in range(count)])
            if inputs.optimize_imbalances else np.maximum(direction * result.arrays["imbalance_mwh"], 0))
    result.trades = {(auction, node, inputs.grid.interval_ids[t]): item.X for (auction, node, t), item in variable["trades"].items()}
    # When `inputs.transaction_fees.mode != 'none'`, set `diagnostics['transaction_fees_by_scenario']` using `scenario_fee_details(inputs, result.trades)`.
    if inputs.transaction_fees.mode != "none":
        diagnostics["transaction_fees_by_scenario"] = scenario_fee_details(inputs, result.trades)
    result.components = {name: np.array([expression[s].getValue() for s in range(count)]) for name, expression in built.components.items()}
    result.objective_eur = float(model.ObjVal)
    return result


def optimize(inputs: OptimizationInput, settings: SolverSettings | None = None) -> OptimizationResult:
    """Convenience entry point reporting environment/license failures explicitly."""
    try:
        return solve_model(build_model(inputs), settings)
    except SolverUnavailableError as error:
        return OptimizationResult("license_or_environment_error", False, message=str(error))
