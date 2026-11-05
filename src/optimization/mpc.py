"""Physical MPC with fixed commitments and explicit price-information contracts.

Legacy controllers share future controls. Research controllers can use eligible
settlement forecasts and delayed-observation recourse, without market orders.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from time import perf_counter
from types import SimpleNamespace
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..domain import DeliveryGrid, PhysicalState, PlantParameters, DegradationSettings, SolverSettings
from .degradation import evaluate_xu_dispatch, initial_segment_energy, xu_segment_parameters
from .model import classify_status
from .physical import add_battery_controls, energy_transition, build_battery_degradation


MPC_POLICIES = frozenset({"deterministic_mpc", "deterministic_mpc_price_oracle",
                          "ensemble_mpc", "ensemble_mpc_price_oracle",
                          "deterministic_mpc_economic", "ensemble_mpc_economic",
                          "ensemble_mpc_adaptive"})
RESEARCH_MPC_POLICIES = frozenset({"deterministic_mpc_economic",
    "ensemble_mpc_economic", "ensemble_mpc_adaptive"})
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
    planning_objective: str = "imbalance_and_costs"
    first_interval_priority: str = "none"
    first_interval_weight: float = 1.0
    first_interval_priority_tolerance_mwh: float = 1e-7
    terminal_inventory_mode: str = "soft_target"
    terminal_soc_target_percent: float = 50.0
    terminal_inventory_penalty_eur_per_mwh: float = 50.0
    terminal_inventory_value_eur_per_mwh: float | None = None
    oracle_relative_imbalance_budget: float = 0.0
    oracle_absolute_imbalance_budget_mwh: float = 0.0
    oracle_fallback_max_relative_gap: float | None = None
    oracle_fallback_time_reserve_fraction: float = 0.5
    oracle_stage2_time_limit_seconds: float | None = None
    oracle_retry_numeric_focus: int | None = None
    oracle_retry_feasibility_tolerance: float | None = None
    ensemble_include_deterministic_member: bool = True
    scenario_weights: Mapping[str, float] | None = None
    solver_time_limit_seconds: float = 10.0
    horizon: str = "remaining_delivery_day"
    reoptimization_minutes: int = 15
    projection_mode: str = "deterministic_open_loop"
    fallback_controller: str = "track_committed_grid"
    current_pv_information: str = "forecast"
    provisional_ida3_mode: str = "ida2_plan"
    economic_imbalance_penalty_eur_per_mwh: float = 0.0
    monetary_cvar_alpha: float = 0.95
    monetary_cvar_weight: float = 0.0
    recourse_stages: tuple[int, ...] | None = None
    recourse_observation_lag_intervals: int = 1
    recourse_pv_bin_width_mw: float = 0.0
    recourse_observation_features: str = "pv"
    recourse_mode: str = "observed_history"

    def __post_init__(self):
        if not isinstance(self.planning_objective, str) or self.planning_objective not in {"imbalance_and_costs", "imbalance_only"}:
            raise ValueError("MPC planning_objective must be imbalance_and_costs or imbalance_only")
        if self.first_interval_priority not in {"none", "weighted", "lexicographic"}:
            raise ValueError("Unknown MPC first_interval_priority")
        weight = self.first_interval_weight
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not np.isfinite(weight) or weight < 1:
            raise ValueError("MPC first_interval_weight must be finite and at least one")
        if self.first_interval_priority != "weighted" and weight != 1:
            raise ValueError("first_interval_weight other than one requires weighted priority")
        tolerance = self.first_interval_priority_tolerance_mwh
        if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance) or tolerance < 0:
            raise ValueError("MPC first_interval_priority_tolerance_mwh must be finite and nonnegative")
        if not isinstance(self.recourse_mode, str) or self.recourse_mode not in {"observed_history", "first_action_only"}:
            raise ValueError("MPC recourse_mode must be observed_history or first_action_only")
        # Reject booleans and nonfinite numeric values rather than coercing YAML.
        for name in ("planning_imbalance_penalty_eur_per_mwh", "terminal_soc_target_percent",
                     "terminal_inventory_penalty_eur_per_mwh", "oracle_relative_imbalance_budget",
                     "oracle_absolute_imbalance_budget_mwh", "solver_time_limit_seconds",
                     "economic_imbalance_penalty_eur_per_mwh", "monetary_cvar_weight",
                     "recourse_pv_bin_width_mw"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value < 0:
                raise ValueError(f"realtime_mpc.{name} must be finite and nonnegative")
        if self.solver_time_limit_seconds <= 0 or not 0 <= self.terminal_soc_target_percent <= 100:
            raise ValueError("MPC needs positive solve time and a terminal target in [0,100]")
        gap = self.oracle_fallback_max_relative_gap
        if gap is not None and (isinstance(gap, bool) or not isinstance(gap, (int, float))
                                or not np.isfinite(gap) or not 0 <= gap < 1):
            raise ValueError("Oracle fallback maximum relative gap must be null or in [0,1)")
        reserve = self.oracle_fallback_time_reserve_fraction
        if (isinstance(reserve, bool) or not isinstance(reserve, (int, float))
                or not np.isfinite(reserve) or not 0 < reserve < 1):
            raise ValueError("Oracle fallback time reserve fraction must be strictly between zero and one")
        stage2_time = self.oracle_stage2_time_limit_seconds
        if stage2_time is not None and (isinstance(stage2_time, bool) or not isinstance(stage2_time, (int, float))
                                        or not np.isfinite(stage2_time) or stage2_time <= 0):
            raise ValueError("Oracle stage-two time limit must be null or a positive number of seconds")
        focus = self.oracle_retry_numeric_focus
        if focus is not None and (isinstance(focus, bool) or not isinstance(focus, int) or focus not in (1, 2, 3)):
            raise ValueError("Oracle retry numerical focus must be null or an integer from 1 to 3")
        tolerance = self.oracle_retry_feasibility_tolerance
        if tolerance is not None and (focus is None or isinstance(tolerance, bool)
                or not isinstance(tolerance, (int, float)) or not np.isfinite(tolerance)
                or not 1e-9 <= tolerance <= 1e-2):
            raise ValueError("Oracle retry feasibility tolerance requires a focus and must be in [1e-9, 1e-2]")
        if self.terminal_inventory_mode not in {"none", "soft_target", "linear_value", "piecewise_linear"}:
            raise ValueError("Unknown MPC terminal_inventory_mode")
        if self.current_pv_information not in {"forecast", "realized_interval_average"}:
            raise ValueError("MPC current_pv_information must be forecast or realized_interval_average")
        if self.provisional_ida3_mode not in {"ida2_plan", "submitted_only"}:
            raise ValueError("MPC provisional_ida3_mode must be ida2_plan or submitted_only")
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
        # Future observations may only distinguish histories after their disclosure.
        lag = self.recourse_observation_lag_intervals
        if isinstance(lag, bool) or not isinstance(lag, int) or lag < 1:
            raise ValueError("MPC recourse observation lag must be a positive integer")
        if self.recourse_observation_features != "pv":
            raise ValueError("MPC recourse currently observes past PV only, not settlement prices")
        stages = self.recourse_stages
        if stages is not None:
            if (not isinstance(stages, (tuple, list)) or any(isinstance(t, bool)
                    or not isinstance(t, int) or t < 1 for t in stages)
                    or tuple(stages) != tuple(sorted(set(stages)))):
                raise ValueError("MPC recourse_stages must contain increasing positive interval indices")
            object.__setattr__(self, "recourse_stages", tuple(stages))
        alpha = self.monetary_cvar_alpha
        if isinstance(alpha, bool) or not isinstance(alpha, (float, int)) or not np.isfinite(alpha) or not 0 < alpha < 1:
            raise ValueError("MPC monetary_cvar_alpha must lie strictly between zero and one")

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
class ForecastPrices:
    """Eligible joint settlement scenarios; rows align with the MPC PV members.

    ``available_at`` is the latest source/model availability used to construct
    the forecast, not retrieval time of a retrospectively exported file. The
    caller retains source snapshot and feature-vintage evidence in provenance.
    """
    positive_eur_per_mwh: np.ndarray
    available_at: Any
    negative_eur_per_mwh: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        stamp = pd.Timestamp(self.available_at)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("Settlement forecast availability must be timezone aware")
        object.__setattr__(self, "available_at", stamp.tz_convert("UTC"))
        # An owned read-only copy cannot change between validation and optimization.
        for name in ("positive_eur_per_mwh", "negative_eur_per_mwh"):
            raw = getattr(self, name)
            if raw is not None:
                values = np.array(raw, dtype=float, copy=True)
                if values.ndim != 2 or not values.size or not np.isfinite(values).all():
                    raise ValueError("Settlement forecasts must be finite scenario-by-time matrices")
                values.setflags(write=False)
                object.__setattr__(self, name, values)
        if self.negative_eur_per_mwh is not None and self.negative_eur_per_mwh.shape != self.positive_eur_per_mwh.shape:
            raise ValueError("Directional settlement forecasts must share their shape")


@dataclass(frozen=True)
class TerminalValueCurve:
    """A cutoff-eligible piecewise-linear continuation value in EUR versus MWh."""
    energy_mwh: np.ndarray
    value_eur: np.ndarray
    available_at: Any
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        stamp = pd.Timestamp(self.available_at)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError("Continuation value availability must be timezone aware")
        object.__setattr__(self, "available_at", stamp.tz_convert("UTC"))
        # Explicit knots allow nonconcave learned values without changing their shape.
        for name in ("energy_mwh", "value_eur"):
            values = np.array(getattr(self, name), dtype=float, copy=True)
            if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
                raise ValueError("Continuation curves require at least two finite knots")
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        if self.value_eur.shape != self.energy_mwh.shape or np.any(np.diff(self.energy_mwh) <= 0):
            raise ValueError("Continuation energy knots must increase and match their values")


@dataclass(frozen=True)
class MpcInput:
    """Submitted positions plus separately identified provisional planning quantities."""
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
    price_forecast: ForecastPrices | None = None
    terminal_value: TerminalValueCurve | None = None
    provisional_ida3_mwh: np.ndarray | None = None
    planning_position_mwh: np.ndarray = field(init=False, repr=False)

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
        # Forecast IDA3 orders affect planning, while submitted positions retain
        # their binding-ledger meaning. The executable interval has no forecast order.
        provisional = (np.zeros(length) if self.provisional_ida3_mwh is None
                       else np.array(self.provisional_ida3_mwh, dtype=float, copy=True))
        if provisional.shape != (length,) or not np.isfinite(provisional).all():
            raise ValueError("Invalid MPC provisional_ida3_mwh shape or values")
        if provisional[0] != 0 or (self.settings.provisional_ida3_mode == "submitted_only" and np.any(provisional)):
            raise ValueError("Provisional IDA3 cannot change current or submitted-only commitments")
        provisional.setflags(write=False)
        planned = self.committed_mwh + provisional
        planned.setflags(write=False)
        object.__setattr__(self, "provisional_ida3_mwh", provisional)
        object.__setattr__(self, "planning_position_mwh", planned)
        if not count or len(set(self.scenario_ids)) != count or np.any(self.pv_ac_mw < 0):
            raise ValueError("MPC needs unique scenarios and nonnegative PV")
        if np.any(self.weights < 0) or not np.isclose(self.weights.sum(), 1, rtol=0, atol=1e-10):
            raise ValueError("MPC scenario weights must sum to one")
        if self.policy.startswith("deterministic") and count != 1:
            raise ValueError("Deterministic MPC requires one trajectory")
        if self.settings.recourse_mode == "first_action_only" and self.policy != "ensemble_mpc_adaptive":
            raise ValueError("first_action_only requires the ensemble_mpc_adaptive policy")
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
        # First-step priority is currently a price-blind deterministic study,
        # not an implicit change to oracle or scenario-recourse objectives.
        if self.settings.first_interval_priority != "none" and (
                self.policy != "deterministic_mpc" or self.settings.planning_objective != "imbalance_and_costs"):
            raise ValueError("First-interval priority requires deterministic price-blind imbalance-and-costs MPC")
        oracle_policy = self.policy.endswith("price_oracle")
        if oracle_policy != (self.oracle is not None):
            raise ValueError("Future realized settlement prices require an explicit price-oracle policy")
        if self.oracle is not None:
            if self.oracle.noncausal is not True:
                raise ValueError("Oracle prices must be explicitly marked noncausal")
            for curve in (self.oracle.positive_eur_per_mwh, self.oracle.negative_eur_per_mwh):
                if curve is not None and (np.shape(curve) != (length,) or not np.isfinite(curve).all()):
                    raise ValueError("Oracle prices must be finite and aligned")
        # Forecast economics and oracle economics are separate, explicit pathways.
        requires_forecast = self.policy.endswith("_economic")
        if requires_forecast and self.price_forecast is None:
            raise ValueError("Economic MPC requires an eligible settlement price_forecast")
        if self.price_forecast is not None:
            if self.policy not in RESEARCH_MPC_POLICIES or self.oracle is not None:
                raise ValueError("Settlement price forecasts require a research economic/adaptive policy")
            if self.price_forecast.available_at > cutoff:
                raise ValueError("Settlement price forecast is after the cutoff")
            if self.price_forecast.positive_eur_per_mwh.shape != (count, length):
                raise ValueError("Settlement price forecast must align with every PV scenario and product")
        if self.settings.planning_objective == "imbalance_only" and (
                self.oracle is not None or self.price_forecast is not None or self.settings.monetary_cvar_weight):
            raise ValueError("imbalance_only excludes settlement-price objectives and monetary risk")
        if self.settings.monetary_cvar_weight and self.price_forecast is None:
            raise ValueError("Monetary loss CVaR requires eligible settlement forecasts")
        if self.settings.terminal_inventory_mode == "piecewise_linear":
            if self.terminal_value is None:
                raise ValueError("piecewise_linear requires an eligible terminal_value curve")
            if self.terminal_value.available_at > cutoff:
                raise ValueError("Continuation value is after the cutoff")
            if (self.terminal_value.energy_mwh[0] > battery.energy_min_mwh
                    or self.terminal_value.energy_mwh[-1] < battery.maximum_energy):
                raise ValueError("Continuation curve must cover the entire battery operating range")
        elif self.terminal_value is not None:
            raise ValueError("A terminal_value curve requires piecewise_linear terminal mode")


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
    """Issued root control, shadow trajectory and scenario delivery projections.

    Research policies additionally retain ``scenario_actions`` as scenario by
    interval by (charge MW, discharge MW, PV fraction), and scenario boundary
    energy in ``scenario_energy_mwh``. The plain ``actions`` and ``energy_mwh``
    arrays select ``representative_path_index`` solely for shadow projections.
    Legacy policies continue to use one common future control trajectory.
    """
    solve_id: str
    actions: list[MpcAction]
    energy_mwh: np.ndarray
    grid_mw: np.ndarray
    imbalance_mwh: np.ndarray
    expected_absolute_imbalance_by_interval_mwh: np.ndarray
    expected_shortfall_by_interval_mwh: np.ndarray
    components: dict[str, float]
    scenario_actions: np.ndarray | None = None
    scenario_energy_mwh: np.ndarray | None = None
    control_nodes: tuple[tuple[int, ...], ...] | None = None
    representative_path_index: int = 0


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
    if inputs.policy in RESEARCH_MPC_POLICIES:
        from .adaptive_mpc import build_research_model

        return build_research_model(inputs)
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
    grid, positive, negative, imported_power = {}, {}, {}, {}
    interval_support = {}
    absolute, first_absolute, imports, settlement = gp.LinExpr(), gp.LinExpr(), gp.LinExpr(), gp.LinExpr()
    # Single-price settlement depends only on the signed difference. Splits are
    # exact under the L1 objective; in stage 2 their sum is a conservative budget.
    # Different directional prices require exclusivity to prevent artificial cash.
    directional = (inputs.oracle is not None and inputs.oracle.negative_eur_per_mwh is not None
                   and not np.array_equal(inputs.oracle.positive_eur_per_mwh, inputs.oracle.negative_eur_per_mwh))
    for s in range(count):
        for t, duration in enumerate(inputs.grid.delta_hours):
            support_key = (t, float(inputs.pv_ac_mw[s, t]))
            representative = interval_support.get(support_key)
            if representative is not None:
                # Common controls and common oracle curves make equal PV values
                # identical interval supports. Alias exactly, without rounding or
                # dropping zero-probability members from the robust POI envelope.
                grid[s, t] = grid[representative, t]
                positive[s, t] = positive[representative, t]
                negative[s, t] = negative[representative, t]
                if inputs.import_fee_eur_per_mwh:
                    imported_power[s, t] = imported_power[representative, t]
            else:
                interval_support[support_key] = s
                grid[s, t] = model.addVar(lb=-inputs.plant.poi_import_mw, ub=inputs.plant.poi_export_mw,
                                         name=f"grid[s{s},t{t}]")
                model.addConstr(grid[s, t] == float(inputs.pv_ac_mw[s, t]) * utilization[t]
                    + discharge[t, "mpc"] - charge[t, "mpc"], name=f"poi_balance[s{s},t{t}]")
                upper = max(0.0, inputs.plant.poi_export_mw * duration - inputs.planning_position_mwh[t])
                lower = max(0.0, inputs.plant.poi_import_mw * duration + inputs.planning_position_mwh[t])
                positive[s, t] = model.addVar(ub=upper, name=f"excess[s{s},t{t}]")
                negative[s, t] = model.addVar(ub=lower, name=f"shortfall[s{s},t{t}]")
                if directional:
                    direction = model.addVar(vtype=gp.GRB.BINARY, name=f"imbalance_sign[s{s},t{t}]")
                    model.addConstr(positive[s, t] <= upper * direction)
                    model.addConstr(negative[s, t] <= lower * (1 - direction))
                model.addConstr(positive[s, t] - negative[s, t] == grid[s, t] * duration
                                - float(inputs.planning_position_mwh[t]), name=f"fixed_commitment[s{s},t{t}]")
                if inputs.import_fee_eur_per_mwh:
                    imported_power[s, t] = model.addVar(ub=inputs.plant.poi_import_mw,
                        name=f"import[s{s},t{t}]")
                    # A minimization cost makes the import epigraph exact in both
                    # economic objectives, with probability mass counted below.
                    model.addConstr(imported_power[s, t] >= -grid[s, t])
            weight = float(inputs.weights[s])
            absolute += weight * (positive[s, t] + negative[s, t])
            if t == 0:
                first_absolute += weight * (positive[s, t] + negative[s, t])
            if inputs.import_fee_eur_per_mwh:
                imports += weight * inputs.import_fee_eur_per_mwh * duration * imported_power[s, t]
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
    elif settings.terminal_inventory_mode == "piecewise_linear":
        value = model.addVar(lb=-gp.GRB.INFINITY, name="terminal_continuation_value_eur")
        model.addGenConstrPWL(energy[0, length], value,
            inputs.terminal_value.energy_mwh.tolist(), inputs.terminal_value.value_eur.tolist(),
            name="terminal_continuation_curve")
        terminal -= value
    expressions = {"absolute_imbalance_mwh": absolute, "degradation_eur": degradation,
                   "import_cost_eur": imports, "terminal_cost_eur": terminal, "settlement_eur": settlement}
    variables.update(first_interval_absolute_imbalance=first_absolute,
        grid=grid, positive=positive, negative=negative, mode=mode, utilization=utilization,
        interval_support_count=len(interval_support), scenario_interval_count=count * length)
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
    imbalance = grid * inputs.grid.delta_hours - inputs.planning_position_mwh
    return MpcPlan(cutoff, actions, np.array([variables["energy"][0, t].X for t in range(length + 1)]),
        grid, imbalance, inputs.weights @ np.abs(imbalance), inputs.weights @ np.maximum(-imbalance, 0),
        {name: float(expression.getValue()) for name, expression in expressions.items()})


def validate_mpc_plan(inputs: MpcInput, plan: MpcPlan, tolerance=1e-6):
    """Independently reconstruct physics and objective components from controls."""
    if inputs.policy in RESEARCH_MPC_POLICIES:
        from .adaptive_mpc import validate_research_plan

        return validate_research_plan(inputs, plan, tolerance)
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
    imbalance = grid * inputs.grid.delta_hours - inputs.planning_position_mwh
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
    elif settings.terminal_inventory_mode == "piecewise_linear":
        terminal = -np.interp(plan.energy_mwh[-1], inputs.terminal_value.energy_mwh,
                              inputs.terminal_value.value_eur)
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


def _component_validation_allowances(inputs, physical_residuals, feasibility_tolerance,
                                     integrality_tolerance):
    """Accumulate constraint tolerances in each component's own units.

    A horizon cash total contains one signed-deviation equality per product.
    Multiplying a single-row tolerance by only the largest price understates
    its possible accumulated numerical error. Probability weights prevent the
    number of otherwise identical scenarios from artificially inflating bounds.
    Physical validation remains separate and must pass before these are used.
    """
    duration = np.asarray(inputs.grid.delta_hours, dtype=float)
    length = len(duration)
    # Retain the existing four-row numerical margin, but apply actual coefficients
    # and horizon lengths rather than treating MWh and EUR as interchangeable.
    row_tolerance = 4 * feasibility_tolerance
    grid_error = float(physical_residuals.get("grid", 0.0))
    deviation_error = row_tolerance + duration * grid_error
    allowances = {"absolute_imbalance_mwh": float(np.sum(2 * row_tolerance + duration * grid_error)),
        "import_cost_eur": float(inputs.import_fee_eur_per_mwh * np.sum(
            duration * (row_tolerance + grid_error))),
        "settlement_eur": 0.0, "terminal_cost_eur": 0.0}
    # AC throughput changes by at most the clipped lower-bound errors of the two
    # control variables per interval. Xu additionally propagates segment-state
    # balance error through the remaining ordered segment flows.
    wear = inputs.degradation.cost_eur_per_mwh_throughput * 2 * row_tolerance * duration.sum()
    if inputs.degradation.mode == "xu_segmented":
        _, marginal = xu_segment_parameters(inputs.plant.battery, inputs.degradation)
        propagated_energy_error = (2 * row_tolerance * inputs.degradation.segment_count
            * np.arange(1, length + 1).sum())
        control_error = 2 * duration.sum() * (row_tolerance
            + inputs.plant.battery.max_power_mw * integrality_tolerance)
        wear = float(np.max(np.abs(marginal))) * (propagated_energy_error + control_error)
    allowances["degradation_eur"] = float(wear)
    settings = inputs.settings
    if settings.terminal_inventory_mode == "soft_target":
        allowances["terminal_cost_eur"] = row_tolerance * settings.terminal_inventory_penalty_eur_per_mwh
    elif settings.terminal_inventory_mode == "linear_value":
        allowances["terminal_cost_eur"] = row_tolerance * abs(settings.terminal_inventory_value_eur_per_mwh)
    elif settings.terminal_inventory_mode == "piecewise_linear":
        slopes = np.diff(inputs.terminal_value.value_eur) / np.diff(inputs.terminal_value.energy_mwh)
        allowances["terminal_cost_eur"] = row_tolerance * (1 + float(np.max(np.abs(slopes))))
    if inputs.oracle is not None:
        positive = inputs.oracle.positive_eur_per_mwh
        negative = inputs.oracle.negative_eur_per_mwh
        negative = positive if negative is None else negative
        coefficient = np.maximum(np.abs(positive), np.abs(negative))
        error = coefficient * deviation_error
        if not np.array_equal(positive, negative):
            # Near-integral sign indicators can leave a tiny split overlap. Its
            # effect depends on the directional spread and that interval's MWh M.
            upper = np.maximum(0, inputs.plant.poi_export_mw * duration - inputs.planning_position_mwh)
            lower = np.maximum(0, inputs.plant.poi_import_mw * duration + inputs.planning_position_mwh)
            overlap = row_tolerance + np.maximum(upper, lower) * integrality_tolerance
            error += np.abs(positive - negative) * overlap
        allowances["settlement_eur"] = float(np.sum(inputs.weights) * np.sum(error))
    return {name: max(1e-6, value) for name, value in allowances.items()}


def _oracle_incumbent_gap(incumbent_mwh, lower_bound_mwh):
    """Compute a conservative relative gap from independently reconstructed MWh."""
    if not np.isfinite(incumbent_mwh) or not np.isfinite(lower_bound_mwh) or incumbent_mwh < 0:
        return None
    difference = max(0.0, incumbent_mwh - lower_bound_mwh)
    if incumbent_mwh == 0:
        return 0.0 if difference == 0 else None
    return difference / incumbent_mwh


def solve_mpc(inputs: MpcInput, solver: SolverSettings | None = None):
    """Solve within one total budget, validating every action before exposing it."""
    if inputs.policy in RESEARCH_MPC_POLICIES:
        from .adaptive_mpc import solve_research_mpc

        return solve_research_mpc(inputs, solver)
    solver = solver or SolverSettings()
    started = perf_counter()
    diagnostics = {"cutoff_utc": pd.Timestamp(inputs.cutoff).isoformat(), "controller": inputs.policy,
        "planning_objective": inputs.settings.planning_objective,
        "first_interval_priority": inputs.settings.first_interval_priority,
        "first_interval_weight": inputs.settings.first_interval_weight,
        "first_interval_priority_tolerance_mwh": inputs.settings.first_interval_priority_tolerance_mwh,
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
        diagnostics["interval_support_count"] = variables["interval_support_count"]
        diagnostics["scenario_interval_count"] = variables["scenario_interval_count"]
        diagnostics["model_variables"] = model.NumVars
        diagnostics["model_binary_variables"] = model.NumBinVars
        diagnostics["model_constraints"] = model.NumConstrs
        for key, value in {"OutputFlag": int(solver.output_flag), "Threads": solver.threads,
                           "MIPGap": solver.mip_gap, "Seed": solver.seed,
                           "FeasibilityTol": solver.feasibility_tolerance,
                           "IntFeasTol": solver.integrality_tolerance}.items():
            if value is not None:
                model.setParam(key, value)
        settings = inputs.settings
        costs = expressions["degradation_eur"] + expressions["import_cost_eur"] + expressions["terminal_cost_eur"]
        oracle = inputs.oracle is not None
        imbalance_only = settings.planning_objective == "imbalance_only"
        objective = expressions["absolute_imbalance_mwh"] if oracle or imbalance_only else (
            settings.planning_imbalance_penalty_eur_per_mwh * expressions["absolute_imbalance_mwh"] + costs)
        priority = settings.first_interval_priority
        lexicographic = priority == "lexicographic"
        if priority == "weighted":
            objective += (settings.planning_imbalance_penalty_eur_per_mwh
                * (settings.first_interval_weight - 1) * variables["first_interval_absolute_imbalance"])
        model.setObjective(variables["first_interval_absolute_imbalance"] if lexicographic else objective, gp.GRB.MINIMIZE)

        def optimize_stage(name, maximum_seconds=None, *, independent_budget=False):
            remaining = max(0.0, settings.solver_time_limit_seconds - (perf_counter() - started))
            time_limit = maximum_seconds if independent_budget else (
                remaining if maximum_seconds is None else min(remaining, maximum_seconds))
            model.setParam("TimeLimit", time_limit)
            model.optimize()
            row = {"stage": name, "status": classify_status(model.Status, model.SolCount),
                   "runtime_seconds": float(model.Runtime), "time_limit_seconds": float(time_limit),
                   "objective": None, "bound": None, "gap": None}
            if model.SolCount:
                row.update(objective=float(model.ObjVal), bound=float(model.ObjBound),
                           gap=float(model.MIPGap) if np.isfinite(model.MIPGap) else None)
            diagnostics["stages"].append(row)
            return model.SolCount > 0

        fallback_gap = settings.oracle_fallback_max_relative_gap if oracle else None
        independent_first = oracle and settings.oracle_stage2_time_limit_seconds is not None
        first_budget = settings.solver_time_limit_seconds if independent_first else None
        if fallback_gap is not None and not independent_first:
            first_budget = max(0.0, settings.solver_time_limit_seconds - (perf_counter() - started))
            first_budget *= 1 - settings.oracle_fallback_time_reserve_fraction
        first_stage_name = ("minimum_first_interval_imbalance" if lexicographic else
            "minimum_absolute_imbalance" if oracle or imbalance_only else "price_blind")
        found = optimize_stage(first_stage_name, first_budget,
                               independent_budget=independent_first)
        # If the reserved first stage has not met the requested fallback gap,
        # spend the remaining time improving it rather than accept a poor bound.
        if fallback_gap is not None and model.Status == gp.GRB.TIME_LIMIT:
            first_gap = None
            if found:
                preliminary = _plan_from_solution(inputs, variables, expressions)
                preliminary_components, _ = validate_mpc_plan(inputs, preliminary,
                    max(1e-7, 2 * model.Params.FeasibilityTol))
                first_gap = _oracle_incumbent_gap(preliminary_components["absolute_imbalance_mwh"], model.ObjBound)
            if (first_gap is None or first_gap > fallback_gap) and not independent_first:
                if perf_counter() - started < settings.solver_time_limit_seconds:
                    found = optimize_stage("minimum_absolute_imbalance_continued")
        if not found:
            result.status = diagnostics["stages"][-1]["status"]
            return result
        # Keep the certified/feasible first-stage plan before replacing its objective.
        first_plan = _plan_from_solution(inputs, variables, expressions)
        stage1_components, _ = validate_mpc_plan(inputs, first_plan, max(1e-7, 2 * model.Params.FeasibilityTol))
        stage1_valid = True
        result.plan = first_plan
        certified = model.Status == gp.GRB.OPTIMAL
        diagnostics["stage1_minimum_certified"] = bool((oracle or lexicographic) and certified)
        diagnostics["first_interval_refinement_used"] = False
        if lexicographic:
            # A time-limited incumbent is usable, but does not certify a minimum.
            first_minimum = float(first_plan.expected_absolute_imbalance_by_interval_mwh[0])
            diagnostics.update(first_interval_incumbent_mwh=first_minimum,
                first_interval_lower_bound_mwh=float(model.ObjBound) if np.isfinite(model.ObjBound) else None,
                first_interval_minimum_certified=certified,
                first_interval_cap_mwh=first_minimum + settings.first_interval_priority_tolerance_mwh if certified else None)
            if certified and perf_counter() - started < settings.solver_time_limit_seconds:
                model.addConstr(variables["first_interval_absolute_imbalance"] <=
                    first_minimum + settings.first_interval_priority_tolerance_mwh,
                    name="first_interval_priority_cap")
                model.setObjective(objective, gp.GRB.MINIMIZE)
                if optimize_stage("horizon_given_first_interval_priority"):
                    candidate = _plan_from_solution(inputs, variables, expressions)
                    validate_mpc_plan(inputs, candidate, max(1e-7, 2 * model.Params.FeasibilityTol))
                    actual_first = float(candidate.expected_absolute_imbalance_by_interval_mwh[0])
                    if actual_first > first_minimum + settings.first_interval_priority_tolerance_mwh + 2 * model.Params.FeasibilityTol:
                        raise ValueError("Current-imbalance refinement exceeds the certified first-stage cap")
                    result.plan = candidate
                    diagnostics["first_interval_refinement_used"] = True
        first_gap = _oracle_incumbent_gap(stage1_components["absolute_imbalance_mwh"], model.ObjBound) if oracle else None
        qualified = bool(oracle and model.Status == gp.GRB.TIME_LIMIT and fallback_gap is not None
                         and first_gap is not None and first_gap <= fallback_gap)
        if oracle:
            bound = float(model.ObjBound)
            diagnostics.update(stage1_incumbent_mwh=stage1_components["absolute_imbalance_mwh"],
                stage1_lower_bound_mwh=bound if np.isfinite(bound) else None, stage1_relative_gap_reconstructed=first_gap,
                oracle_fallback_qualified=qualified, oracle_fallback_max_relative_gap=fallback_gap)
        if oracle and (certified or qualified):
            minimum = stage1_components["absolute_imbalance_mwh"]
            budget = minimum * settings.oracle_relative_imbalance_budget + settings.oracle_absolute_imbalance_budget_mwh
            diagnostics.update(oracle_cap_reference="certified_incumbent" if certified else "gap_qualified_incumbent",
                               oracle_additional_budget_mwh=budget, oracle_imbalance_cap_mwh=minimum + budget)
            if certified:
                diagnostics["stage1_l_min_mwh"] = minimum
            model.addConstr(expressions["absolute_imbalance_mwh"] <= minimum + budget, name="oracle_imbalance_budget")
            model.setObjective(expressions["settlement_eur"] - costs, gp.GRB.MAXIMIZE)
            separate_budget = settings.oracle_stage2_time_limit_seconds is not None
            can_refine = separate_budget or perf_counter() - started < settings.solver_time_limit_seconds
            refinement_started = perf_counter()
            if can_refine and optimize_stage("signed_settlement_economics",
                    settings.oracle_stage2_time_limit_seconds, independent_budget=separate_budget):
                # Retry only a candidate rejected by physical or cap validation.
                # The optional second solve consumes the remaining stage budget.
                for attempt in range(2):
                    try:
                        candidate = _plan_from_solution(inputs, variables, expressions)
                        candidate_components, _ = validate_mpc_plan(inputs, candidate,
                            max(1e-7, 2 * model.Params.FeasibilityTol))
                        reconstructed = candidate_components["absolute_imbalance_mwh"]
                        diagnostics["oracle_candidate_reconstructed_imbalance_mwh"] = reconstructed
                        # Check the physical total independently of model splits.
                        if reconstructed > minimum + budget + 2 * model.Params.FeasibilityTol:
                            raise ValueError("Oracle refinement exceeds reconstructed imbalance cap: "
                                f"{reconstructed} > {minimum + budget}")
                        if candidate.components["absolute_imbalance_mwh"] > minimum + budget + 2 * model.Params.FeasibilityTol:
                            raise ValueError("Oracle refinement exceeds modeled imbalance cap")
                    except ValueError as error:
                        if attempt or settings.oracle_retry_numeric_focus is None:
                            raise
                        remaining = (settings.oracle_stage2_time_limit_seconds - (perf_counter() - refinement_started)
                            if separate_budget else settings.solver_time_limit_seconds - (perf_counter() - started))
                        if remaining <= 0:
                            raise
                        diagnostics["oracle_retry_initial_rejection"] = str(error)
                        diagnostics["oracle_retry_attempted"] = True
                        model.setParam("NumericFocus", settings.oracle_retry_numeric_focus)
                        if settings.oracle_retry_feasibility_tolerance is not None:
                            model.setParam("FeasibilityTol", min(model.Params.FeasibilityTol,
                                settings.oracle_retry_feasibility_tolerance))
                        diagnostics["oracle_retry_numeric_focus"] = model.Params.NumericFocus
                        diagnostics["oracle_retry_feasibility_tolerance"] = model.Params.FeasibilityTol
                        # Discard the numerically invalid incumbent before re-solving.
                        model.reset()
                        remaining = (settings.oracle_stage2_time_limit_seconds - (perf_counter() - refinement_started)
                            if separate_budget else settings.solver_time_limit_seconds - (perf_counter() - started))
                        if remaining <= 0:
                            raise ValueError("Oracle numerical retry exhausted the economic-stage budget") from error
                        if not optimize_stage("signed_settlement_economics_numeric_retry", remaining,
                                              independent_budget=separate_budget):
                            raise ValueError("Oracle numerical retry returned no feasible incumbent") from error
                    else:
                        result.plan = candidate
                        break
        diagnostics["oracle_stage2_used"] = bool(oracle and result.plan is not first_plan)
        components, residuals = validate_mpc_plan(inputs, result.plan, max(1e-7, 2 * model.Params.FeasibilityTol))
        diagnostics.update(physical_residuals=residuals, recomputed_components=components)
        # Stage 1 ignores economic epigraphs: validate their tightness only in an economic stage.
        economic_stage = ((not oracle and not imbalance_only and
            (not lexicographic or diagnostics["first_interval_refinement_used"])) or diagnostics["oracle_stage2_used"])
        if imbalance_only:
            error = abs(result.plan.components["absolute_imbalance_mwh"] - components["absolute_imbalance_mwh"])
            if error > max(1e-6, 2 * model.Params.FeasibilityTol * len(inputs.grid.interval_ids)):
                raise ValueError("Pure-imbalance objective differs from reconstructed physical deviations")
        if economic_stage:
            errors = {key: abs(result.plan.components[key] - value) for key, value in components.items()
                      if key != "absolute_imbalance_mwh" or (not oracle and settings.planning_imbalance_penalty_eur_per_mwh > 0)}
            diagnostics["component_residuals"] = errors
            allowances = _component_validation_allowances(inputs, residuals,
                model.Params.FeasibilityTol, model.Params.IntFeasTol)
            diagnostics["component_validation_allowances"] = allowances
            if any(error > allowances[key] for key, error in errors.items()):
                raise ValueError(f"MPC objective component validation failed: {errors}")
        result.plan.components = components
        diagnostics["components"] = components
        if oracle and (certified or qualified):
            diagnostics["oracle_additional_imbalance_used_mwh"] = max(0.0,
                components["absolute_imbalance_mwh"] - diagnostics["stage1_incumbent_mwh"])
        diagnostics["planning_objective_value"] = (components["settlement_eur"] - sum(components[k] for k in
            ("degradation_eur", "import_cost_eur", "terminal_cost_eur")) if diagnostics.get("oracle_stage2_used") else
            components["absolute_imbalance_mwh"] if oracle or imbalance_only else
            settings.planning_imbalance_penalty_eur_per_mwh * (components["absolute_imbalance_mwh"]
                + (settings.first_interval_weight - 1) * float(result.plan.expected_absolute_imbalance_by_interval_mwh[0]))
            + sum(components[k] for k in ("degradation_eur", "import_cost_eur", "terminal_cost_eur")))
        if lexicographic and not diagnostics["first_interval_refinement_used"]:
            diagnostics["planning_objective_value"] = float(result.plan.expected_absolute_imbalance_by_interval_mwh[0])
        result.status = diagnostics["stages"][-1]["status"]
        if lexicographic and not diagnostics["first_interval_refinement_used"]:
            result.status = "first_interval_stage1_only"
        if qualified:
            # Stage-two optimality is conditional on an approximate first-stage
            # cap; do not label the combined lexicographic policy optimal.
            result.status = ("feasible_gap_qualified_refined" if diagnostics["oracle_stage2_used"]
                             else "feasible_gap_qualified_unrefined")
        diagnostics["accepted_action_source"] = ("first_interval_stage1" if lexicographic and not diagnostics["first_interval_refinement_used"]
            else "first_interval_refined" if lexicographic else "minimum_expected_absolute_imbalance" if imbalance_only
            else "economic" if economic_stage else "price_blind_stage1")
    except Exception as error:
        diagnostics["failure_reason"] = str(error)
        # Economic refinement cannot discard an already validated stage-1 action.
        if stage1_valid and (inputs.oracle is not None or inputs.settings.first_interval_priority == "lexicographic"):
            result.plan = first_plan
            result.plan.components, residuals = validate_mpc_plan(inputs, first_plan,
                max(1e-7, 2 * model.Params.FeasibilityTol))
            lexicographic = inputs.settings.first_interval_priority == "lexicographic"
            result.status = "first_interval_refinement_rejected" if lexicographic else "oracle_refinement_rejected"
            diagnostics.update(oracle_stage2_used=False, first_interval_refinement_used=False,
                accepted_action_source="first_interval_stage1" if lexicographic else "price_blind_stage1",
                physical_residuals=residuals, components=result.plan.components,
                planning_objective_value=float(first_plan.expected_absolute_imbalance_by_interval_mwh[0])
                    if lexicographic else result.plan.components["absolute_imbalance_mwh"])
        else:
            result.plan = None
            result.status = "invalid_or_failed_solve"
    finally:
        diagnostics["solve_wall_seconds"] = perf_counter() - started
        diagnostics["runtime_seconds"] = sum(row["runtime_seconds"] for row in diagnostics["stages"])
        diagnostics["status"] = result.status
        if result.plan is not None:
            diagnostics["first_interval_expected_imbalance_mwh"] = float(
                result.plan.expected_absolute_imbalance_by_interval_mwh[0])
            if diagnostics.get("first_interval_minimum_certified"):
                diagnostics["first_interval_increase_over_incumbent_mwh"] = max(0.0,
                    diagnostics["first_interval_expected_imbalance_mwh"] - diagnostics["first_interval_incumbent_mwh"])
        diagnostics["planning_objective_units"] = ("MWh" if inputs.settings.planning_objective == "imbalance_only"
            or inputs.oracle is not None and not diagnostics.get("oracle_stage2_used", False)
            or inputs.settings.first_interval_priority == "lexicographic" and not diagnostics.get("first_interval_refinement_used", False)
            else "EUR")
        if result.plan is not None and "oracle_imbalance_cap_mwh" in diagnostics:
            diagnostics["oracle_additional_imbalance_used_mwh"] = max(0.0,
                result.plan.components["absolute_imbalance_mwh"] - diagnostics["stage1_incumbent_mwh"])
        if result.plan is None:
            diagnostics["fallback_applied"] = True
        if model is not None:
            model.dispose()
    return result
