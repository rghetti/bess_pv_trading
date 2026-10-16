"""Solver-independent physical and contractual recomputation from result arrays."""
from __future__ import annotations

import numpy as np

from ..domain import OptimizationInput, OptimizationResult, ValidationReport
from .degradation import evaluate_xu_dispatch, initial_segment_energy, xu_segment_parameters


def evaluate_components(inputs: OptimizationInput, arrays: dict[str, np.ndarray], trades: dict[tuple[str, str, str], float]) -> dict[str, np.ndarray]:
    """Evaluate actual signed settlement and AC throughput without solver expressions."""
    count = len(inputs.scenarios.scenario_ids)
    revenue = np.zeros(count)
    for auction, eligible in inputs.auction_eligibility.items():
        for s in range(count):
            owner = inputs.tree.bid_nodes[auction][s]
            for t, product in enumerate(inputs.grid.interval_ids):
                if eligible[t]:
                    revenue[s] += trades[auction, owner, product] * inputs.scenarios.auction_prices_eur_per_mwh[auction][s, t]
    imbalance = arrays["grid_mw"] * inputs.grid.delta_hours - arrays["net_position_mwh"]
    negative_prices = inputs.scenarios.imbalance_negative_prices_eur_per_mwh
    if negative_prices is None:
        negative_prices = inputs.scenarios.imbalance_prices_eur_per_mwh
    settlement = np.sum(np.maximum(imbalance, 0) * inputs.scenarios.imbalance_prices_eur_per_mwh + np.minimum(imbalance, 0) * negative_prices, axis=1)
    penalty = np.abs(imbalance).sum(axis=1) * inputs.imbalance_penalty_eur_per_mwh
    throughput = np.sum((arrays["charge_mw"] + arrays["discharge_mw"]) * inputs.grid.delta_hours, axis=1)
    aging = throughput * inputs.degradation.cost_eur_per_mwh_throughput
    if inputs.degradation.mode == "xu_segmented":
        aging = np.array([evaluate_xu_dispatch(battery=inputs.plant.battery,
            settings=inputs.degradation, state=inputs.state, charge_mw=arrays["charge_mw"][s],
            discharge_mw=arrays["discharge_mw"][s], delta_hours=inputs.grid.delta_hours)["degradation_eur"]
            for s in range(count)])
    fees = np.sum(np.maximum(-arrays["grid_mw"], 0) * inputs.grid.delta_hours, axis=1) * inputs.grid_import_fee_eur_per_mwh
    return {"auction_revenue": revenue, "imbalance_cashflow": settlement, "imbalance_penalty": penalty, "battery_degradation_cost": aging, "other_costs": fees, "operating_profit": revenue + settlement - penalty - aging - fees, "cumulative_throughput_mwh": inputs.state.cumulative_throughput_mwh + throughput, "cumulative_degradation_eur": inputs.state.cumulative_degradation_eur + aging}


def validate_solution(inputs: OptimizationInput, result: OptimizationResult, tolerance: float = 1e-6) -> ValidationReport:
    """Recompute balances, signs, sharing and accounting independently of Gurobi."""
    if not result.has_incumbent:
        return ValidationReport(False, {}, None, (f"No incumbent: {result.status}",))
    count, intervals = np.shape(inputs.scenarios.pv_ac_mw)
    required = {"charge_mw": (count, intervals), "discharge_mw": (count, intervals), "mode_charge": (count, intervals), "curtailment_mw": (count, intervals), "grid_mw": (count, intervals), "net_position_mwh": (count, intervals), "energy_mwh": (count, intervals + 1), "imbalance_mwh": (count, intervals), "imbalance_positive_mwh": (count, intervals), "imbalance_negative_mwh": (count, intervals)}
    if inputs.degradation.mode == "xu_segmented":
        segments = inputs.degradation.segment_count
        required.update(xu_segment_energy_mwh=(count, intervals + 1, segments),
                        xu_segment_charge_mw=(count, intervals, segments),
                        xu_segment_discharge_mw=(count, intervals, segments))
    invalid = [name for name, shape in required.items() if name not in result.arrays or np.shape(result.arrays[name]) != shape or not np.all(np.isfinite(result.arrays[name]))]
    if invalid:
        return ValidationReport(False, {}, None, (f"Missing, nonfinite or incorrectly shaped result arrays: {invalid}",))
    arrays, battery = result.arrays, inputs.plant.battery
    errors: dict[str, float] = {}

    def equality(name: str, value: np.ndarray | float) -> None:
        errors[name] = float(np.max(np.abs(value), initial=0)) if np.all(np.isfinite(value)) else float("inf")

    def upper(name: str, value: np.ndarray | float) -> None:
        errors[name] = float(np.max(value, initial=0)) if np.all(np.isfinite(value)) else float("inf")

    # Check AC bus power and the battery-internal energy transition independently.
    charge, discharge = arrays["charge_mw"], arrays["discharge_mw"]
    curtailment, energy, grid = arrays["curtailment_mw"], arrays["energy_mwh"], arrays["grid_mw"]
    delta = inputs.grid.delta_hours
    equality("ac_bus_balance_mw", inputs.scenarios.pv_ac_mw + discharge - charge - curtailment - grid)
    for s in range(count):
        predicted = np.empty(intervals)
        for t, duration in enumerate(delta):
            retained = energy[s, t] * (1 - battery.self_discharge_per_hour) ** duration
            predicted[t] = retained + charge[s, t] * duration * battery.eta_charge - discharge[s, t] * duration / battery.eta_discharge
        equality(f"battery_energy_balance_mwh_s{s}", predicted - energy[s, 1:])
    equality("initial_energy_mwh", energy[:, 0] - inputs.state.energy_mwh)
    if inputs.terminal_target_mwh is not None:
        equality("terminal_energy_mwh", energy[:, -1] - inputs.terminal_target_mwh)
    upper("energy_upper_mwh", energy - battery.maximum_energy)
    upper("energy_lower_mwh", battery.energy_min_mwh - energy)
    upper("charge_lower_mw", -charge)
    upper("discharge_lower_mw", -discharge)
    upper("charge_power_mw", charge - battery.max_charge_mw)
    upper("discharge_power_mw", discharge - battery.max_discharge_mw)
    equality("mode_integrality", arrays["mode_charge"] - np.round(arrays["mode_charge"]))
    upper("mode_lower", -arrays["mode_charge"])
    upper("mode_upper", arrays["mode_charge"] - 1)
    upper("charge_mode_mw", charge - battery.max_charge_mw * arrays["mode_charge"])
    upper("discharge_mode_mw", discharge - battery.max_discharge_mw * (1 - arrays["mode_charge"]))
    upper("simultaneous_battery_mw", np.minimum(charge, discharge))
    upper("curtailment_lower_mw", -curtailment)
    upper("curtailment_upper_mw", curtailment - inputs.scenarios.pv_ac_mw)
    upper("poi_export_mw", grid - inputs.plant.poi_export_mw)
    upper("poi_import_mw", -grid - inputs.plant.poi_import_mw)
    if inputs.degradation.mode == "xu_segmented":
        capacity, _ = xu_segment_parameters(battery, inputs.degradation)
        segment_energy = arrays["xu_segment_energy_mwh"]
        segment_charge = arrays["xu_segment_charge_mw"]
        segment_discharge = arrays["xu_segment_discharge_mw"]
        equality("xu_segment_initial_mwh", segment_energy[:, 0] - initial_segment_energy(inputs.state, battery, inputs.degradation))
        equality("xu_segment_energy_sum_mwh", segment_energy.sum(axis=2) - energy)
        equality("xu_segment_charge_sum_mw", segment_charge.sum(axis=2) - charge)
        equality("xu_segment_discharge_sum_mw", segment_discharge.sum(axis=2) - discharge)
        equality("xu_segment_energy_balance_mwh", np.diff(segment_energy, axis=1) - delta[None, :, None] * (battery.eta_charge * segment_charge - segment_discharge / battery.eta_discharge))
        upper("xu_segment_capacity_mwh", segment_energy - capacity[None, None, :])
        upper("xu_segment_lower_mwh", -segment_energy)
        upper("xu_segment_charge_lower_mw", -segment_charge)
        upper("xu_segment_discharge_lower_mw", -segment_discharge)
        # Reconstruct canonical cycle memory numerically from aggregate commands,
        # independent of the model's priorities or extracted segment variables.
        try:
            for s in range(count):
                reference = evaluate_xu_dispatch(battery=battery, settings=inputs.degradation,
                    state=inputs.state, charge_mw=charge[s], discharge_mw=discharge[s], delta_hours=delta)
                for name in ("energy_mwh", "charge_mw", "discharge_mw"):
                    equality(f"xu_priority_{name}_s{s}", arrays[f"xu_segment_{name}"][s] - reference[f"segment_{name}"])
        except ValueError:
            errors["xu_dispatch_evaluable"] = float("inf")
    # Rebuild contractual positions from the immutable accepted-trade ledger.
    expected_keys = {(auction, owner, product) for auction, eligible in inputs.auction_eligibility.items() for owner in inputs.tree.bid_nodes[auction] for t, product in enumerate(inputs.grid.interval_ids) if eligible[t]}
    if set(result.trades) != expected_keys or any(not np.isfinite(value) for value in result.trades.values()):
        return ValidationReport(False, errors, None, ("Trade keys differ from actionable node/products or contain nonfinite values",))
    upper("trade_limit_mwh", np.array([abs(value) - inputs.trade_limit_mwh for value in result.trades.values()]))
    positions = np.repeat(inputs.market.positions(inputs.grid)[None, :], count, axis=0)
    if inputs.pending_position_mwh is not None:
        positions += np.asarray(inputs.pending_position_mwh)[None, :]
    auction_order = sorted(inputs.auction_eligibility, key=lambda a: ({"DA": 0, "IDA1": 1, "IDA2": 2, "IDA3": 3}.get(a, 4), a))
    for auction in auction_order:
        for s in range(count):
            for t, product in enumerate(inputs.grid.interval_ids):
                if inputs.auction_eligibility[auction][t]:
                    positions[s, t] += result.trades[auction, inputs.tree.bid_nodes[auction][s], product]
        upper(f"cumulative_position_after_{auction}_mwh", np.abs(positions) - inputs.position_limit_mwh)
    equality("contractual_position_mwh", positions - arrays["net_position_mwh"])
    upper("final_position_limit_mwh", np.abs(positions) - inputs.position_limit_mwh)
    imbalance = grid * delta - positions
    equality("signed_imbalance_mwh", imbalance - arrays["imbalance_mwh"])
    equality("positive_imbalance_mwh", arrays["imbalance_positive_mwh"] - np.maximum(imbalance, 0))
    equality("negative_imbalance_mwh", arrays["imbalance_negative_mwh"] - np.maximum(-imbalance, 0))
    if inputs.imbalance_limit_mwh is not None:
        upper("imbalance_limit_mwh", np.abs(imbalance) - inputs.imbalance_limit_mwh)
    # A node's physical controls and binaries must agree for every descendant leaf.
    for t in range(intervals):
        for owner in set(row[t] for row in inputs.tree.control_nodes):
            descendants = [s for s, row in enumerate(inputs.tree.control_nodes) if row[t] == owner]
            physical_names = ("charge_mw", "discharge_mw", "mode_charge")
            if inputs.degradation.mode == "xu_segmented":
                physical_names += ("xu_segment_charge_mw", "xu_segment_discharge_mw")
            for name in physical_names:
                equality(f"nonanticipativity_{name}_{t}_{owner}", arrays[name][descendants, t] - arrays[name][descendants[0], t])
        curtailment_owners = inputs.tree.curtailment_nodes or inputs.tree.control_nodes
        for owner in set(row[t] for row in curtailment_owners):
            descendants = [s for s, row in enumerate(curtailment_owners) if row[t] == owner]
            equality(f"nonanticipativity_curtailment_{t}_{owner}", curtailment[descendants, t] - curtailment[descendants[0], t])
    try:
        components = evaluate_components(inputs, arrays, result.trades)
    except ValueError as error:
        return ValidationReport(False, errors, None, (f"Cannot evaluate degradation: {error}",))
    for name, values in result.components.items():
        if name in components:
            equality(f"component_{name}_eur", np.asarray(values) - components[name])
    objective = float(np.dot(inputs.scenarios.probabilities, components["operating_profit"]))
    equality("expected_objective_eur", np.inf if result.objective_eur is None else result.objective_eur - objective)
    violations = tuple(f"{name}: {error:.6g}" for name, error in errors.items() if error > tolerance)
    return ValidationReport(not violations, errors, objective, violations)
