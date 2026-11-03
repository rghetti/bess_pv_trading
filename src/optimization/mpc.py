"""Price-blind and explicitly noncausal economic MPC with fixed commitments.

This interface deliberately contains no auction prices, bids or mutable ledger.
Battery and PV-utilization decisions are common across the entire scenario set.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..domain import DeliveryGrid, PhysicalState, PlantParameters, DegradationSettings, SolverSettings
from .degradation import evaluate_xu_dispatch, initial_segment_energy
from .model import classify_status
from .physical import add_battery_controls, energy_transition, build_battery_degradation


MPC_POLICIES = frozenset({"deterministic_mpc", "deterministic_mpc_price_oracle",
                          "ensemble_mpc", "ensemble_mpc_price_oracle"})
CAUSAL_TRACKER = "causal_track_committed_grid"


def uses_realtime_control(config):
    """Identify forecast needs without changing the auction PV planning policy."""
    policies = set(config.get("realtime_mpc_comparison", {}).get("controllers", []))
    policies.add(config.get("execution", {}).get("dispatch_policy", "track_committed_grid"))
    return bool(policies & (MPC_POLICIES | {CAUSAL_TRACKER}))


def needs_mpc_ensemble(config):
    """Request raw members independently of auction ensemble enablement."""
    policies = set(config.get("realtime_mpc_comparison", {}).get("controllers", []))
    policies.add(config.get("execution", {}).get("dispatch_policy", "track_committed_grid"))
    return any(policy.startswith("ensemble_mpc") for policy in policies)


@dataclass(frozen=True)
class MpcSettings:
    """Controller-only settings; shared assets, wear and solver settings stay external."""
    planning_imbalance_penalty_eur_per_mwh: float = 500.0
    terminal_inventory_mode: str = "soft_target"
    terminal_soc_target_percent: float = 50.0
    terminal_inventory_penalty_eur_per_mwh: float = 50.0
    terminal_inventory_value_eur_per_mwh: float | None = None
    oracle_relative_imbalance_budget: float = 0.0
    oracle_absolute_imbalance_budget_mwh: float = 0.0
    ensemble_include_deterministic_member: bool = True
    scenario_weights: Mapping[str, float] | None = None
    solver_time_limit_seconds: float = 10.0
    horizon: str = "remaining_delivery_day"
    reoptimization_minutes: int = 15
    projection_mode: str = "deterministic_open_loop"
    fallback_controller: str = "track_committed_grid"
    current_pv_information: str = "forecast"

    def __post_init__(self):
        # Reject booleans and nonfinite numeric values rather than coercing YAML.
        for name in ("planning_imbalance_penalty_eur_per_mwh", "terminal_soc_target_percent",
                     "terminal_inventory_penalty_eur_per_mwh", "oracle_relative_imbalance_budget",
                     "oracle_absolute_imbalance_budget_mwh", "solver_time_limit_seconds"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"realtime_mpc.{name} must be finite and nonnegative")
        if self.solver_time_limit_seconds <= 0 or not 0 <= self.terminal_soc_target_percent <= 100:
            raise ValueError("MPC needs positive solve time and a terminal target in [0,100]")
        if self.terminal_inventory_mode not in {"none", "soft_target", "linear_value"}:
            raise ValueError("Unknown MPC terminal_inventory_mode")
        if self.current_pv_information not in {"forecast", "realized_interval_average"}:
            raise ValueError("MPC current_pv_information must be forecast or realized_interval_average")
        value = self.terminal_inventory_value_eur_per_mwh
        if value is not None and (isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value)):
            raise ValueError("MPC terminal energy value must be finite")
        if self.terminal_inventory_mode == "linear_value" and value is None:
            raise ValueError("linear_value requires terminal_inventory_value_eur_per_mwh")
        if not isinstance(self.ensemble_include_deterministic_member, bool):
            raise ValueError("MPC ensemble_include_deterministic_member must be a YAML boolean")
        if (self.horizon != "remaining_delivery_day" or self.reoptimization_minutes != 15
                or isinstance(self.reoptimization_minutes, bool)
                or self.projection_mode != "deterministic_open_loop"
                or self.fallback_controller != "track_committed_grid"):
            raise ValueError("MPC v1 requires a 15-minute remaining-day horizon, deterministic open-loop projection and tracking fallback")
        if self.scenario_weights is not None:
            values = list(self.scenario_weights.values()) if isinstance(self.scenario_weights, Mapping) else []
            if (not values or any(isinstance(v, bool) or not isinstance(v, (int, float))
                    or not np.isfinite(v) or v < 0 for v in values)
                    or not np.isclose(sum(values), 1.0, rtol=0, atol=1e-10)):
                raise ValueError("MPC scenario_weights must be a normalized nonnegative mapping")

    @classmethod
    def from_config(cls, config):
        """An absent block preserves the legacy execution selector and defaults."""
        values = config.get("realtime_mpc", {})
        if not isinstance(values, dict) or set(values) - {item.name for item in fields(cls)}:
            raise ValueError("Unknown or invalid realtime_mpc settings")
        return cls(**values)


@dataclass(frozen=True)
class OraclePrices:
    """Actual future prices deliberately disclosed for a noncausal experiment."""
    positive_eur_per_mwh: np.ndarray
    negative_eur_per_mwh: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)
    noncausal: bool = True

    def __post_init__(self):
        # Oracle disclosure is explicit and immutable, just like forecast inputs.
        for name in ("positive_eur_per_mwh", "negative_eur_per_mwh"):
            values = getattr(self, name)
            if values is not None:
                curve = np.array(values, dtype=float, copy=True)
                if curve.ndim != 1 or not np.isfinite(curve).all():
                    raise ValueError("Oracle price curves must be finite vectors")
                curve.setflags(write=False)
                object.__setattr__(self, name, curve)


@dataclass(frozen=True)
class MpcInput:
    """Physical inputs with separately disclosed current-PV and price oracles."""
    cutoff: Any
    grid: DeliveryGrid
    plant: PlantParameters
    state: PhysicalState
    committed_mwh: np.ndarray
    pv_ac_mw: np.ndarray
    scenario_ids: tuple[str, ...]
    weights: np.ndarray
    degradation: DegradationSettings
    policy: str = "deterministic_mpc"
    settings: MpcSettings = field(default_factory=MpcSettings)
    import_fee_eur_per_mwh: float = 0.0
    hard_terminal_target_mwh: float | None = None
    hard_terminal_minimum_mwh: float | None = None
    forecast_provenance: Mapping[str, Any] = field(default_factory=dict)
    oracle: OraclePrices | None = None
    current_realized_pv_mw: float | None = None

    def __post_init__(self):
        count, length = len(self.scenario_ids), len(self.grid.interval_ids)
        cutoff = pd.Timestamp(self.cutoff)
        if self.policy not in MPC_POLICIES or cutoff.tzinfo is None or cutoff != self.state.timestamp:
            raise ValueError("MPC requires an aware interval-start cutoff and a supported policy")
        object.__setattr__(self, "cutoff", cutoff.tz_convert("UTC"))
        if self.state.timestamp != self.grid.starts[0] or not np.allclose(self.grid.delta_hours, 0.25):
            raise ValueError("MPC state/grid must start at the decision and use quarter-hours")
        # Freeze caller-owned arrays: neither a ledger nor a cached forecast may mutate a solve.
        for name, shape in (("committed_mwh", (length,)), ("pv_ac_mw", (count, length)), ("weights", (count,))):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError(f"Invalid MPC {name} shape or values")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if not count or len(set(self.scenario_ids)) != count or np.any(self.pv_ac_mw < 0):
            raise ValueError("MPC needs unique scenarios and nonnegative PV")
        if np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1, rtol=0, atol=1e-10):
            raise ValueError("MPC scenario weights must sum to one")
        if self.policy.startswith("deterministic") and count != 1:
            raise ValueError("Deterministic MPC requires one trajectory")
        # This deliberately disclosed scalar can reveal only the current interval.
        known_current = self.settings.current_pv_information == "realized_interval_average"
        if known_current != (self.current_realized_pv_mw is not None):
            raise ValueError("Realized current PV requires the explicit current_pv_information setting")
        if known_current and (not np.isfinite(self.current_realized_pv_mw) or self.current_realized_pv_mw < 0
                or not np.all(self.pv_ac_mw[:, 0] == self.current_realized_pv_mw)):
            raise ValueError("Every scenario must use the disclosed realized PV in its first interval")
        battery = self.plant.battery
        if not battery.energy_min_mwh <= self.state.energy_mwh <= battery.maximum_energy:
            raise ValueError("MPC initial energy lies outside the battery box")
        if self.degradation.mode == "xu_segmented":
            if battery.self_discharge_per_hour or battery.capacity_mwh <= 0:
                raise ValueError("Xu MPC requires positive capacity and zero self discharge")
            initial_segment_energy(self.state, battery, self.degradation)
        elif self.state.cycle_segment_energy_mwh is not None:
            raise ValueError("MPC cannot discard Xu segment memory")
        for name in ("hard_terminal_target_mwh", "hard_terminal_minimum_mwh"):
            value = getattr(self, name)
            if value is not None and (not np.isfinite(value) or not battery.energy_min_mwh <= value <= battery.maximum_energy):
                raise ValueError(f"Invalid MPC {name}")
        if not np.isfinite(self.import_fee_eur_per_mwh) or self.import_fee_eur_per_mwh < 0:
            raise ValueError("MPC import fee must be finite and nonnegative")
        # Publication clocks remain enforceable even for cached/in-memory providers.
        for name, value in self.forecast_provenance.items():
            # A retrospectively selected model is a separately disclosed input;
            # this exception never admits a weather cycle published after cutoff.
            if (name == "pv_forecast_model_available_at_utc"
                    and self.forecast_provenance.get("pv_forecast_model_retrospective") is True):
                continue
            if name.endswith(("available_at_utc", "vintage_utc")) and value is not None and pd.Timestamp(value) > cutoff:
                raise ValueError(f"MPC forecast {name} is after the cutoff")
        oracle_policy = self.policy.endswith("price_oracle")
        if oracle_policy != (self.oracle is not None):
            raise ValueError("Future realized settlement prices require an explicit price-oracle policy")
        if self.oracle is not None:
            if self.oracle.noncausal is not True:
                raise ValueError("Oracle prices must be explicitly marked noncausal")
            for curve in (self.oracle.positive_eur_per_mwh, self.oracle.negative_eur_per_mwh):
                if curve is not None and (np.shape(curve) != (length,) or not np.isfinite(curve).all()):
                    raise ValueError("Oracle prices must be finite and aligned")


@dataclass(frozen=True)
class MpcAction:
    """A fixed battery command and implementable PV utilization fraction."""
    interval_id: str
    issued_at: str
    charge_mw: float
    discharge_mw: float
    pv_utilization_fraction: float


@dataclass
class MpcPlan:
    """Complete open-loop controls, common energy and scenario delivery projections."""
    solve_id: str
    actions: list[MpcAction]
    energy_mwh: np.ndarray
    grid_mw: np.ndarray
    imbalance_mwh: np.ndarray
    expected_absolute_imbalance_by_interval_mwh: np.ndarray
    expected_shortfall_by_interval_mwh: np.ndarray
    components: dict[str, float]


@dataclass
class MpcResult:
    """A validated plan or a reason to invoke the declared tracking fallback."""
    status: str
    plan: MpcPlan | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class MpcControllerState:
    """Serializable control history, separate from market orders and actual SOC."""
    latest_plan: dict[str, dict] = field(default_factory=dict)
    outstanding_action: dict | None = None
    solves: list[dict] = field(default_factory=list)
    plans: list[dict] = field(default_factory=list)
    scenarios: list[dict] = field(default_factory=list)
    issued_actions: list[dict] = field(default_factory=list)


def build_mpc_model(inputs: MpcInput):
    """Build only physical controls and deviations; there are no market orders."""
    import gurobipy as gp

    model = gp.Model("realtime_economic_mpc")
    model.setParam("OutputFlag", 0)
    battery, settings = inputs.plant.battery, inputs.settings
    count, length = inputs.pv_ac_mw.shape
    charge, discharge, mode, utilization, energy = {}, {}, {}, {}, {}
    # One control owner for every member, for the entire open-loop horizon.
    for t in range(length):
        charge[t, "mpc"], discharge[t, "mpc"], mode[t] = add_battery_controls(model, battery, gp, f"mpc,t{t}")
        utilization[t] = model.addVar(ub=1.0, name=f"pv_utilization[t{t}]")
        # With no forecast PV this setpoint is unidentified. Pass any unexpected
        # generation through protection instead of arbitrarily commanding zero PV.
        if np.all(inputs.pv_ac_mw[:, t] == 0):
            model.addConstr(utilization[t] == 1, name=f"zero_forecast_utilization[t{t}]")
    for t in range(length + 1):
        energy[0, t] = model.addVar(lb=battery.energy_min_mwh, ub=battery.maximum_energy, name=f"energy[t{t}]")
    model.addConstr(energy[0, 0] == inputs.state.energy_mwh, name="observed_initial_energy")
    for t, duration in enumerate(inputs.grid.delta_hours):
        model.addConstr(energy[0, t + 1] == energy_transition(energy[0, t], charge[t, "mpc"],
            discharge[t, "mpc"], battery, duration), name=f"battery_balance[t{t}]")
    if inputs.hard_terminal_target_mwh is not None:
        model.addConstr(energy[0, length] == inputs.hard_terminal_target_mwh, name="study_terminal_target")
    if inputs.hard_terminal_minimum_mwh is not None:
        model.addConstr(energy[0, length] >= inputs.hard_terminal_minimum_mwh, name="terminal_floor")
    # Reuse Xu's canonical segment ordering on the one common battery trajectory.
    context = SimpleNamespace(plant=inputs.plant, state=inputs.state, grid=inputs.grid,
        degradation=inputs.degradation,
        tree=SimpleNamespace(control_nodes=(tuple("mpc" for _ in range(length)),)),
        scenarios=SimpleNamespace(scenario_ids=("common",), pv_ac_mw=inputs.pv_ac_mw[:1]))
    variables = {"charge": charge, "discharge": discharge, "energy": energy}
    degradation = build_battery_degradation(model, context, variables, gp)[0]
    grid, positive, negative = {}, {}, {}
    absolute, imports, settlement = gp.LinExpr(), gp.LinExpr(), gp.LinExpr()
    # Single-price settlement depends only on the signed difference. Splits are
    # exact under the L1 objective; in stage 2 their sum is a conservative budget.
    # Different directional prices require exclusivity to prevent artificial cash.
    directional = (inputs.oracle is not None and inputs.oracle.negative_eur_per_mwh is not None
                   and not np.array_equal(inputs.oracle.positive_eur_per_mwh, inputs.oracle.negative_eur_per_mwh))
    for s in range(count):
        for t, duration in enumerate(inputs.grid.delta_hours):
            grid[s, t] = model.addVar(lb=-inputs.plant.poi_import_mw, ub=inputs.plant.poi_export_mw,
                                     name=f"grid[s{s},t{t}]")
            model.addConstr(grid[s, t] == float(inputs.pv_ac_mw[s, t]) * utilization[t]
                + discharge[t, "mpc"] - charge[t, "mpc"], name=f"poi_balance[s{s},t{t}]")
            upper = max(0.0, inputs.plant.poi_export_mw * duration - inputs.committed_mwh[t])
            lower = max(0.0, inputs.plant.poi_import_mw * duration + inputs.committed_mwh[t])
            positive[s, t] = model.addVar(ub=upper, name=f"excess[s{s},t{t}]")
            negative[s, t] = model.addVar(ub=lower, name=f"shortfall[s{s},t{t}]")
            if directional:
                direction = model.addVar(vtype=gp.GRB.BINARY, name=f"imbalance_sign[s{s},t{t}]")
                model.addConstr(positive[s, t] <= upper * direction)
                model.addConstr(negative[s, t] <= lower * (1 - direction))
            model.addConstr(positive[s, t] - negative[s, t] == grid[s, t] * duration
                            - float(inputs.committed_mwh[t]), name=f"fixed_commitment[s{s},t{t}]")
            weight = float(inputs.weights[s])
            absolute += weight * (positive[s, t] + negative[s, t])
            if inputs.import_fee_eur_per_mwh:
                imported = model.addVar(ub=inputs.plant.poi_import_mw, name=f"import[s{s},t{t}]")
                # A minimization cost makes this epigraph exact in both economic objectives.
                model.addConstr(imported >= -grid[s, t])
                imports += weight * inputs.import_fee_eur_per_mwh * duration * imported
            if inputs.oracle is not None:
                negative_curve = (inputs.oracle.positive_eur_per_mwh if inputs.oracle.negative_eur_per_mwh is None
                                  else inputs.oracle.negative_eur_per_mwh)
                settlement += weight * (float(inputs.oracle.positive_eur_per_mwh[t]) * positive[s, t]
                                       - float(negative_curve[t]) * negative[s, t])
    terminal = gp.LinExpr()
    if settings.terminal_inventory_mode == "soft_target":
        target = battery.capacity_mwh * settings.terminal_soc_target_percent / 100
        deviation = model.addVar(name="terminal_inventory_deviation_mwh")
        model.addConstr(deviation >= energy[0, length] - target)
        model.addConstr(deviation >= target - energy[0, length])
        terminal += settings.terminal_inventory_penalty_eur_per_mwh * deviation
    elif settings.terminal_inventory_mode == "linear_value":
        terminal -= settings.terminal_inventory_value_eur_per_mwh * energy[0, length]
    expressions = {"absolute_imbalance_mwh": absolute, "degradation_eur": degradation,
                   "import_cost_eur": imports, "terminal_cost_eur": terminal, "settlement_eur": settlement}
    variables.update(grid=grid, positive=positive, negative=negative, mode=mode, utilization=utilization)
    model.update()
    return model, variables, expressions


def _plan_from_solution(inputs, variables, expressions):
    """Extract one shared trajectory and retain every scenario's projected delivery."""
    count, length = inputs.pv_ac_mw.shape
    cutoff = pd.Timestamp(inputs.cutoff).isoformat()
    actions = [MpcAction(product, cutoff, max(0.0, variables["charge"][t, "mpc"].X),
        max(0.0, variables["discharge"][t, "mpc"].X), float(np.clip(variables["utilization"][t].X, 0, 1)))
        for t, product in enumerate(inputs.grid.interval_ids)]
    grid = np.array([[variables["grid"][s, t].X for t in range(length)] for s in range(count)])
    imbalance = grid * inputs.grid.delta_hours - inputs.committed_mwh
    return MpcPlan(cutoff, actions, np.array([variables["energy"][0, t].X for t in range(length + 1)]),
        grid, imbalance, inputs.weights @ np.abs(imbalance), inputs.weights @ np.maximum(-imbalance, 0),
        {name: float(expression.getValue()) for name, expression in expressions.items()})


def validate_mpc_plan(inputs: MpcInput, plan: MpcPlan, tolerance=1e-6):
    """Independently reconstruct physics and objective components from controls."""
    battery = inputs.plant.battery
    length = len(inputs.grid.interval_ids)
    if len(plan.actions) != length or plan.energy_mwh.shape != (length + 1,):
        raise ValueError("MPC plan does not cover its input horizon")
    commands = np.array([[a.charge_mw, a.discharge_mw, a.pv_utilization_fraction] for a in plan.actions])
    if not np.isfinite(commands).all() or not np.isfinite(plan.energy_mwh).all():
        raise ValueError("Nonfinite MPC commands")
    charge, discharge, utilization = commands.T
    if any(a.interval_id != product or pd.Timestamp(a.issued_at) != inputs.cutoff
           for a, product in zip(plan.actions, inputs.grid.interval_ids)):
        raise ValueError("MPC action product identity mismatch")
    predicted = energy_transition(plan.energy_mwh[:-1], charge, discharge, battery, inputs.grid.delta_hours)
    grid = inputs.pv_ac_mw * utilization + discharge - charge
    imbalance = grid * inputs.grid.delta_hours - inputs.committed_mwh
    residuals = {
        "initial_energy": abs(plan.energy_mwh[0] - inputs.state.energy_mwh),
        "energy_transition": np.max(np.abs(predicted - plan.energy_mwh[1:])),
        "energy_lower": np.max(battery.energy_min_mwh - plan.energy_mwh),
        "energy_upper": np.max(plan.energy_mwh - battery.maximum_energy),
        "charge_discharge_lower": np.max(-commands[:, :2]),
        "power": np.max(commands[:, :2] - battery.max_power_mw),
        "exclusivity": np.max(np.minimum(charge, discharge)),
        "utilization_lower": np.max(-utilization), "utilization_upper": np.max(utilization - 1),
        "poi_import": np.max(-grid - inputs.plant.poi_import_mw),
        "poi_export": np.max(grid - inputs.plant.poi_export_mw),
        "grid": np.max(np.abs(grid - plan.grid_mw)),
        "imbalance": np.max(np.abs(imbalance - plan.imbalance_mwh)),
        "expected_absolute": np.max(np.abs(inputs.weights @ np.abs(imbalance)
            - plan.expected_absolute_imbalance_by_interval_mwh)),
        "expected_shortfall": np.max(np.abs(inputs.weights @ np.maximum(-imbalance, 0)
            - plan.expected_shortfall_by_interval_mwh))}
    if inputs.hard_terminal_target_mwh is not None:
        residuals["study_terminal"] = abs(plan.energy_mwh[-1] - inputs.hard_terminal_target_mwh)
    if inputs.hard_terminal_minimum_mwh is not None:
        residuals["terminal_floor"] = inputs.hard_terminal_minimum_mwh - plan.energy_mwh[-1]
    if max(residuals.values()) > tolerance:
        raise ValueError(f"MPC physical validation failed: {residuals}")
    degradation = inputs.degradation.cost_eur_per_mwh_throughput * np.sum((charge + discharge) * inputs.grid.delta_hours)
    if inputs.degradation.mode == "xu_segmented":
        degradation = evaluate_xu_dispatch(battery=battery, settings=inputs.degradation, state=inputs.state,
            charge_mw=charge, discharge_mw=discharge, delta_hours=inputs.grid.delta_hours)["degradation_eur"]
    settings = inputs.settings
    terminal = 0.0
    if settings.terminal_inventory_mode == "soft_target":
        terminal = settings.terminal_inventory_penalty_eur_per_mwh * abs(plan.energy_mwh[-1]
            - battery.capacity_mwh * settings.terminal_soc_target_percent / 100)
    elif settings.terminal_inventory_mode == "linear_value":
        terminal = -settings.terminal_inventory_value_eur_per_mwh * plan.energy_mwh[-1]
    settlement = 0.0
    if inputs.oracle is not None:
        negative = inputs.oracle.negative_eur_per_mwh
        negative = inputs.oracle.positive_eur_per_mwh if negative is None else negative
        settlement = inputs.weights @ np.sum(np.maximum(imbalance, 0) * inputs.oracle.positive_eur_per_mwh
            - np.maximum(-imbalance, 0) * negative, axis=1)
    components = {"absolute_imbalance_mwh": float(inputs.weights @ np.abs(imbalance).sum(axis=1)),
        "degradation_eur": float(degradation), "terminal_cost_eur": float(terminal),
        "import_cost_eur": float(inputs.import_fee_eur_per_mwh * (inputs.weights @
            np.sum(np.maximum(-grid, 0) * inputs.grid.delta_hours, axis=1))), "settlement_eur": float(settlement)}
    return components, residuals


def solve_mpc(inputs: MpcInput, solver: SolverSettings | None = None):
    """Solve within one total budget, validating every action before exposing it."""
    solver = solver or SolverSettings()
    started = perf_counter()
    diagnostics = {"cutoff_utc": pd.Timestamp(inputs.cutoff).isoformat(), "controller": inputs.policy,
        "price_oracle_enabled": inputs.oracle is not None, "scenario_ids": list(inputs.scenario_ids),
        "current_pv_information": inputs.settings.current_pv_information,
        "current_interval_pv_oracle_enabled": inputs.current_realized_pv_mw is not None,
        "current_realized_pv_mw": inputs.current_realized_pv_mw,
        "scenario_weights": inputs.weights.tolist(), "scenario_count": len(inputs.weights),
        "horizon_intervals": len(inputs.grid.interval_ids), "forecast_provenance": dict(inputs.forecast_provenance),
        "horizon_start_utc": str(inputs.grid.starts[0]), "horizon_end_utc": str(inputs.grid.ends[-1]),
        "horizon_hours": float(np.sum(inputs.grid.delta_hours)),
        "oracle_provenance": None if inputs.oracle is None else dict(inputs.oracle.provenance),
        "stages": [], "fallback_applied": False, "absolute_mip_gap_configured": False}
    model = None
    first_plan = None
    stage1_valid = False
    result = MpcResult("solver_error", diagnostics=diagnostics)
    try:
        import gurobipy as gp

        model, variables, expressions = build_mpc_model(inputs)
        diagnostics["build_seconds"] = perf_counter() - started
        for key, value in {"OutputFlag": int(solver.output_flag), "Threads": solver.threads,
                           "MIPGap": solver.mip_gap, "Seed": solver.seed,
                           "FeasibilityTol": solver.feasibility_tolerance,
                           "IntFeasTol": solver.integrality_tolerance}.items():
            if value is not None:
                model.setParam(key, value)
        settings = inputs.settings
        costs = expressions["degradation_eur"] + expressions["import_cost_eur"] + expressions["terminal_cost_eur"]
        oracle = inputs.oracle is not None
        objective = expressions["absolute_imbalance_mwh"] if oracle else (
            settings.planning_imbalance_penalty_eur_per_mwh * expressions["absolute_imbalance_mwh"] + costs)
        model.setObjective(objective, gp.GRB.MINIMIZE)

        def optimize_stage(name):
            remaining = max(0.0, settings.solver_time_limit_seconds - (perf_counter() - started))
            model.setParam("TimeLimit", remaining)
            model.optimize()
            row = {"stage": name, "status": classify_status(model.Status, model.SolCount),
                   "runtime_seconds": float(model.Runtime), "objective": None, "bound": None, "gap": None}
            if model.SolCount:
                row.update(objective=float(model.ObjVal), bound=float(model.ObjBound),
                           gap=float(model.MIPGap) if np.isfinite(model.MIPGap) else None)
            diagnostics["stages"].append(row)
            return model.SolCount > 0

        if not optimize_stage("minimum_absolute_imbalance" if oracle else "price_blind"):
            result.status = diagnostics["stages"][-1]["status"]
            return result
        # Keep the certified/feasible first-stage plan before replacing its objective.
        first_plan = _plan_from_solution(inputs, variables, expressions)
        stage1_components, _ = validate_mpc_plan(inputs, first_plan, max(1e-7, 2 * model.Params.FeasibilityTol))
        stage1_valid = True
        result.plan = first_plan
        certified = model.Status == gp.GRB.OPTIMAL
        diagnostics["stage1_minimum_certified"] = bool(oracle and certified)
        if oracle and certified:
            minimum = stage1_components["absolute_imbalance_mwh"]
            budget = minimum * settings.oracle_relative_imbalance_budget + settings.oracle_absolute_imbalance_budget_mwh
            diagnostics.update(stage1_l_min_mwh=minimum, oracle_additional_budget_mwh=budget,
                               oracle_imbalance_cap_mwh=minimum + budget)
            model.addConstr(expressions["absolute_imbalance_mwh"] <= minimum + budget, name="oracle_imbalance_budget")
            model.setObjective(expressions["settlement_eur"] - costs, gp.GRB.MAXIMIZE)
            if perf_counter() - started < settings.solver_time_limit_seconds and optimize_stage("signed_settlement_economics"):
                candidate = _plan_from_solution(inputs, variables, expressions)
                if candidate.components["absolute_imbalance_mwh"] <= minimum + budget + 2 * model.Params.FeasibilityTol:
                    result.plan = candidate
        diagnostics["oracle_stage2_used"] = bool(oracle and result.plan is not first_plan)
        components, residuals = validate_mpc_plan(inputs, result.plan, max(1e-7, 2 * model.Params.FeasibilityTol))
        diagnostics.update(physical_residuals=residuals, recomputed_components=components)
        # Stage 1 ignores economic epigraphs: validate their tightness only in an economic stage.
        economic_stage = not oracle or diagnostics["oracle_stage2_used"]
        if economic_stage:
            errors = {key: abs(result.plan.components[key] - value) for key, value in components.items()
                      if key != "absolute_imbalance_mwh" or (not oracle and settings.planning_imbalance_penalty_eur_per_mwh > 0)}
            diagnostics["component_residuals"] = errors
            allowance = max(1e-6, 4 * model.Params.FeasibilityTol * max(
                1, settings.planning_imbalance_penalty_eur_per_mwh,
                settings.terminal_inventory_penalty_eur_per_mwh,
                float(np.max(np.abs(inputs.oracle.positive_eur_per_mwh))) if oracle else 0))
            if max(errors.values()) > allowance:
                raise ValueError(f"MPC objective component validation failed: {errors}")
        result.plan.components = components
        diagnostics["components"] = components
        if oracle and certified:
            diagnostics["oracle_additional_imbalance_used_mwh"] = max(0.0,
                components["absolute_imbalance_mwh"] - diagnostics["stage1_l_min_mwh"])
        diagnostics["planning_objective_value"] = (components["settlement_eur"] - sum(components[k] for k in
            ("degradation_eur", "import_cost_eur", "terminal_cost_eur")) if diagnostics.get("oracle_stage2_used") else
            components["absolute_imbalance_mwh"] if oracle else
            settings.planning_imbalance_penalty_eur_per_mwh * components["absolute_imbalance_mwh"]
            + sum(components[k] for k in ("degradation_eur", "import_cost_eur", "terminal_cost_eur")))
        result.status = diagnostics["stages"][-1]["status"]
        diagnostics["accepted_action_source"] = "economic" if economic_stage else "price_blind_stage1"
    except Exception as error:
        diagnostics["failure_reason"] = str(error)
        # Economic refinement cannot discard an already validated stage-1 action.
        if stage1_valid and inputs.oracle is not None:
            result.plan = first_plan
            result.plan.components, residuals = validate_mpc_plan(inputs, first_plan,
                max(1e-7, 2 * model.Params.FeasibilityTol))
            result.status = "oracle_refinement_rejected"
            diagnostics.update(oracle_stage2_used=False, accepted_action_source="price_blind_stage1",
                physical_residuals=residuals, components=result.plan.components,
                planning_objective_value=result.plan.components["absolute_imbalance_mwh"])
        else:
            result.plan = None
            result.status = "invalid_or_failed_solve"
    finally:
        diagnostics["solve_wall_seconds"] = perf_counter() - started
        diagnostics["runtime_seconds"] = sum(row["runtime_seconds"] for row in diagnostics["stages"])
        diagnostics["status"] = result.status
        diagnostics["planning_objective_units"] = ("MWh" if inputs.oracle is not None
            and not diagnostics.get("oracle_stage2_used", False) else "EUR")
        if result.plan is not None and "stage1_l_min_mwh" in diagnostics:
            diagnostics["oracle_additional_imbalance_used_mwh"] = max(0.0,
                result.plan.components["absolute_imbalance_mwh"] - diagnostics["stage1_l_min_mwh"])
        if result.plan is None:
            diagnostics["fallback_applied"] = True
        if model is not None:
            model.dispose()
    return result
