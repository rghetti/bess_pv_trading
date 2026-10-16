"""Validated in-memory contracts; optimizer quantities use MW, MWh and EUR."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping

import numpy as np


def aware(value: datetime, label: str) -> None:
    """Reject ambiguous naive clocks at application boundaries."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


def finite_nonnegative(value: float, label: str) -> None:
    """Check physical magnitudes without clipping invalid input."""
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be finite and nonnegative")


@dataclass(frozen=True)
class DeliveryGrid:
    """Contiguous delivery intervals with explicit UTC-compatible boundaries."""
    interval_ids: tuple[str, ...]
    starts: tuple[datetime, ...]
    ends: tuple[datetime, ...]

    def __post_init__(self) -> None:
        if not self.interval_ids or len(set(self.interval_ids)) != len(self.interval_ids):
            raise ValueError("Delivery interval IDs must be nonempty and unique")
        if len(self.starts) != len(self.interval_ids) or len(self.ends) != len(self.starts):
            raise ValueError("Delivery boundaries and IDs must have matching lengths")
        for start, end in zip(self.starts, self.ends):
            aware(start, "interval start")
            aware(end, "interval end")
        object.__setattr__(self, "starts", tuple(value.astimezone(timezone.utc) for value in self.starts))
        object.__setattr__(self, "ends", tuple(value.astimezone(timezone.utc) for value in self.ends))
        for t, (start, end) in enumerate(zip(self.starts, self.ends)):
            if end <= start or (t and start != self.ends[t - 1]):
                raise ValueError("Delivery intervals must be positive and contiguous")

    @property
    def delta_hours(self) -> np.ndarray:
        return np.array([(end - start).total_seconds() / 3600 for start, end in zip(self.starts, self.ends)])


@dataclass(frozen=True)
class FarmMetadata:
    """Confirmed farm identity/nameplate plus separately declared assumptions."""
    farm_id: int
    installed_dc_mw: float
    tracking_type_id: int
    zone: str = "NORD"
    confirmed: Mapping[str, Any] = field(default_factory=dict)
    assumptions: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ForecastBundle:
    """A forecast vintage is distinct from a probabilistic member."""
    source_cycle: datetime
    available_at: datetime
    values: Mapping[str, Any]
    units: Mapping[str, str]
    lead_time: Any = None
    interval_semantics: Mapping[str, str] = field(default_factory=dict)
    member_origin: str = "deterministic_weather_vintage"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        aware(self.source_cycle, "forecast source cycle")
        aware(self.available_at, "forecast availability")
        if self.available_at < self.source_cycle:
            raise ValueError("Forecast cannot be available before its source cycle")


@dataclass(frozen=True)
class ObservationBundle:
    """Raw and policy-treated observations retain evidence clocks and flags."""
    grid: DeliveryGrid
    raw_values: Mapping[str, Any]
    treated_values: Mapping[str, Any]
    available_at: Any
    evidence_available_at: Any
    flags: Mapping[str, Any] = field(default_factory=dict)
    review_status: Any = None
    provenance: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BatteryParameters:
    """AC-side powers and internal stored energy; choose one efficiency convention."""
    capacity_mwh: float
    max_charge_mw: float
    max_discharge_mw: float
    energy_min_mwh: float = 0.0
    energy_max_mwh: float | None = None
    converter_charge_efficiency: float = 0.97
    converter_discharge_efficiency: float = 0.97
    cell_charge_efficiency: float = 1.0
    cell_discharge_efficiency: float = 1.0
    efficiency_convention: str = "converter_and_cell"
    combined_charge_efficiency: float | None = None
    combined_discharge_efficiency: float | None = None
    inverter_charge_mw: float | None = None
    inverter_discharge_mw: float | None = None
    self_discharge_per_hour: float = 0.0

    def __post_init__(self) -> None:
        for key in ("capacity_mwh", "max_charge_mw", "max_discharge_mw", "energy_min_mwh", "self_discharge_per_hour"):
            finite_nonnegative(getattr(self, key), key)
        if not self.energy_min_mwh <= self.maximum_energy <= self.capacity_mwh:
            raise ValueError("Battery energy bounds must lie within capacity")
        if self.self_discharge_per_hour >= 1:
            raise ValueError("Hourly self discharge must be below one")
        if self.capacity_mwh == 0 and (self.max_charge_mw or self.max_discharge_mw):
            raise ValueError("A no-BESS plant must have zero battery power")
        if self.efficiency_convention == "converter_and_cell":
            if self.combined_charge_efficiency is not None or self.combined_discharge_efficiency is not None:
                raise ValueError("Combined efficiencies would double count converter losses")
            efficiency_values = (self.converter_charge_efficiency, self.converter_discharge_efficiency, self.cell_charge_efficiency, self.cell_discharge_efficiency)
        elif self.efficiency_convention == "combined":
            if self.combined_charge_efficiency is None or self.combined_discharge_efficiency is None:
                raise ValueError("Combined convention requires both combined efficiencies")
            if self.cell_charge_efficiency != 1 or self.cell_discharge_efficiency != 1:
                raise ValueError("Combined convention cannot also specify cell losses")
            efficiency_values = (self.combined_charge_efficiency, self.combined_discharge_efficiency)
        else:
            raise ValueError("Unknown battery efficiency convention")
        if any(not np.isfinite(value) or not 0 < value <= 1 for value in efficiency_values):
            raise ValueError("Efficiencies must be finite and in (0,1]")
        for limit, power in ((self.inverter_charge_mw, self.max_charge_mw), (self.inverter_discharge_mw, self.max_discharge_mw)):
            if limit is not None:
                finite_nonnegative(limit, "inverter limit")
                if power > limit:
                    raise ValueError("Battery AC power limit exceeds inverter AC rating")

    @property
    def maximum_energy(self) -> float:
        return self.capacity_mwh if self.energy_max_mwh is None else self.energy_max_mwh

    @property
    def eta_charge(self) -> float:
        return float(self.combined_charge_efficiency) if self.efficiency_convention == "combined" else self.converter_charge_efficiency * self.cell_charge_efficiency

    @property
    def eta_discharge(self) -> float:
        return float(self.combined_discharge_efficiency) if self.efficiency_convention == "combined" else self.converter_discharge_efficiency * self.cell_discharge_efficiency


@dataclass(frozen=True)
class PlantParameters:
    """One physical plant with a shared AC point of interconnection."""
    farm_id: int
    poi_import_mw: float
    poi_export_mw: float
    battery: BatteryParameters

    def __post_init__(self) -> None:
        finite_nonnegative(self.poi_import_mw, "POI import")
        finite_nonnegative(self.poi_export_mw, "POI export")


@dataclass(frozen=True)
class PhysicalState:
    """Timestamped boundary state and carried cumulative degradation accounting."""
    timestamp: datetime
    energy_mwh: float
    cumulative_throughput_mwh: float = 0.0
    cumulative_degradation_eur: float = 0.0
    origin: str = "known_delivery_start_assumption"
    cycle_segment_energy_mwh: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        aware(self.timestamp, "physical state")
        object.__setattr__(self, "timestamp", self.timestamp.astimezone(timezone.utc))
        for key in ("energy_mwh", "cumulative_throughput_mwh", "cumulative_degradation_eur"):
            finite_nonnegative(getattr(self, key), key)
        if self.cycle_segment_energy_mwh is not None:
            values = tuple(float(value) for value in self.cycle_segment_energy_mwh)
            if not values or any(not np.isfinite(value) or value < 0 for value in values):
                raise ValueError("Cycle segment energy must be a finite nonnegative vector")
            if not np.isclose(sum(values), self.energy_mwh, rtol=0, atol=1e-8):
                raise ValueError("Cycle segment energies must sum to physical battery energy")
            object.__setattr__(self, "cycle_segment_energy_mwh", values)


@dataclass(frozen=True)
class AcceptedTrade:
    """Immutable accepted contractual adjustment; positive quantities are sales."""
    auction: str
    submission_time: datetime
    publication_time: datetime
    interval_id: str
    quantity_mwh: float
    price_eur_per_mwh: float
    trade_id: str = ""

    def __post_init__(self) -> None:
        aware(self.submission_time, "trade submission")
        aware(self.publication_time, "trade publication")
        if self.publication_time < self.submission_time:
            raise ValueError("Trade publication precedes submission")
        if not np.isfinite(self.quantity_mwh) or not np.isfinite(self.price_eur_per_mwh):
            raise ValueError("Accepted trade must have finite quantity and price")

    @property
    def cashflow_eur(self) -> float:
        return self.quantity_mwh * self.price_eur_per_mwh


@dataclass(frozen=True)
class MarketState:
    """Append-only accepted trades are fixed inputs to residual optimization."""
    accepted_trades: tuple[AcceptedTrade, ...] = ()

    def accept(self, trade: AcceptedTrade) -> MarketState:
        if trade.trade_id and any(item.trade_id == trade.trade_id for item in self.accepted_trades):
            raise ValueError("Duplicate accepted trade ID")
        return MarketState(self.accepted_trades + (trade,))

    def positions(self, grid: DeliveryGrid) -> np.ndarray:
        totals = {key: 0.0 for key in grid.interval_ids}
        for trade in self.accepted_trades:
            if trade.interval_id in totals:
                totals[trade.interval_id] += trade.quantity_mwh
        return np.array([totals[key] for key in grid.interval_ids])

    @property
    def cashflow_eur(self) -> float:
        return sum(trade.cashflow_eur for trade in self.accepted_trades)


@dataclass(frozen=True)
class ScenarioBundle:
    """Joint leaf trajectories, with explicit normalized leaf probability mass."""
    scenario_ids: tuple[str, ...]
    probabilities: np.ndarray
    pv_ac_mw: np.ndarray
    auction_prices_eur_per_mwh: Mapping[str, np.ndarray]
    imbalance_prices_eur_per_mwh: np.ndarray
    imbalance_negative_prices_eur_per_mwh: np.ndarray | None = None
    origin: str = "synthetic"
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        shape = np.shape(self.pv_ac_mw)
        if len(shape) != 2 or shape[0] != len(self.scenario_ids) or not shape[1]:
            raise ValueError("PV scenarios must have shape (scenario, interval)")
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("Scenario IDs must be unique")
        probabilities = np.asarray(self.probabilities)
        if probabilities.shape != (shape[0],) or not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0) or not np.isclose(probabilities.sum(), 1.0, rtol=0, atol=1e-10):
            raise ValueError("Scenario probabilities must be nonnegative and sum to one")
        arrays = [self.pv_ac_mw, self.imbalance_prices_eur_per_mwh, *self.auction_prices_eur_per_mwh.values()]
        if self.imbalance_negative_prices_eur_per_mwh is not None:
            arrays.append(self.imbalance_negative_prices_eur_per_mwh)
        if any(np.shape(array) != shape or not np.all(np.isfinite(array)) for array in arrays):
            raise ValueError("Every scenario coefficient must be finite and have matching shape")
        if np.any(np.asarray(self.pv_ac_mw) < 0):
            raise ValueError("PV AC availability cannot be negative")


@dataclass(frozen=True)
class InformationNode:
    """Observable history at a cutoff; child mass is conditional on its parent."""
    node_id: str
    parent_id: str | None
    cutoff: datetime
    conditional_probability: float = 1.0
    observations: tuple[str, ...] = ()
    observation_available_at: Mapping[str, datetime] = field(default_factory=dict)

    def __post_init__(self) -> None:
        aware(self.cutoff, "information cutoff")
        object.__setattr__(self, "cutoff", self.cutoff.astimezone(timezone.utc))
        if not np.isfinite(self.conditional_probability) or not 0 <= self.conditional_probability <= 1:
            raise ValueError("Node conditional probability must lie in [0,1]")
        for key, available in self.observation_available_at.items():
            aware(available, "observation availability")
            if available > self.cutoff:
                raise ValueError(f"Node uses unpublished observation {key}")


@dataclass(frozen=True)
class InformationTree:
    """Explicit history tree plus the owner of every auction and physical action."""
    nodes: tuple[InformationNode, ...]
    leaf_nodes: tuple[str, ...]
    bid_nodes: Mapping[str, tuple[str, ...]]
    control_nodes: tuple[tuple[str, ...], ...]
    curtailment_nodes: tuple[tuple[str, ...], ...] | None = None
    execution_convention: str = "pre_interval_battery_and_curtailment"

    def paths(self) -> dict[str, tuple[str, ...]]:
        by_id = {node.node_id: node for node in self.nodes}
        if len(by_id) != len(self.nodes) or sum(node.parent_id is None for node in self.nodes) != 1:
            raise ValueError("Information tree needs unique nodes and exactly one root")
        paths: dict[str, tuple[str, ...]] = {}
        for node in self.nodes:
            chain: list[str] = []
            current: InformationNode | None = node
            while current is not None:
                if current.node_id in chain:
                    raise ValueError("Information tree contains a cycle")
                chain.append(current.node_id)
                if current.parent_id is None:
                    break
                if current.parent_id not in by_id:
                    raise ValueError("Information node has missing parent")
                parent = by_id[current.parent_id]
                if current.cutoff < parent.cutoff:
                    raise ValueError("Information clock moves backwards")
                current = parent
            paths[node.node_id] = tuple(reversed(chain))
        for node in self.nodes:
            children = [item for item in self.nodes if item.parent_id == node.node_id]
            if children and not np.isclose(sum(item.conditional_probability for item in children), 1.0, rtol=0, atol=1e-10):
                raise ValueError("Child conditional probabilities must sum to one")
            if node.parent_id is None and node.conditional_probability != 1:
                raise ValueError("Root probability must be one")
        return paths


@dataclass(frozen=True)
class DegradationSettings:
    """Select AC throughput or Xu discharge-only cycle-depth accounting.

    Xu's stress defaults reproduce Eq. (24) of the author preprint, rather than
    calibrating a battery chemistry. Replacement cost is the total pack cost in EUR.
    """
    cost_eur_per_mwh_throughput: float = 0.0
    mode: str = "constant_throughput"
    assumptions: str = "illustrative coefficient; no measured battery lifetime data"
    replacement_cost_eur: float | None = None
    stress_coefficient: float = 5.24e-4
    stress_exponent: float = 2.03
    segment_count: int = 16

    def __post_init__(self) -> None:
        finite_nonnegative(self.cost_eur_per_mwh_throughput, "degradation cost")
        if self.mode not in {"constant_throughput", "xu_segmented"}:
            raise ValueError("Unknown degradation mode; use constant_throughput or Xu xu_segmented")
        if not isinstance(self.segment_count, int) or isinstance(self.segment_count, bool) or self.segment_count < 1:
            raise ValueError("Cycle segment_count must be a positive integer")
        if not np.isfinite(self.stress_coefficient) or self.stress_coefficient <= 0:
            raise ValueError("Cycle stress_coefficient must be finite and positive")
        if not np.isfinite(self.stress_exponent) or self.stress_exponent < 1:
            raise ValueError("Cycle stress_exponent must be finite and at least one for convexity")
        if self.replacement_cost_eur is not None:
            finite_nonnegative(self.replacement_cost_eur, "replacement cost")
        if self.mode == "xu_segmented":
            if self.replacement_cost_eur is None or self.replacement_cost_eur <= 0:
                raise ValueError("Xu segmented degradation requires a positive total replacement_cost_eur")
            if self.cost_eur_per_mwh_throughput:
                raise ValueError("Xu segmented aging cannot also charge a throughput coefficient")


@dataclass(frozen=True)
class SolverSettings:
    """Validated conservative Gurobi settings; no environment/license mutation."""
    threads: int = 1
    time_limit_seconds: float = 60.0
    mip_gap: float = 1e-4
    seed: int = 0
    output_flag: bool = False
    export_path: str | None = None
    iis_path: str | None = None

    def __post_init__(self) -> None:
        values = (self.time_limit_seconds, self.mip_gap)
        if self.threads < 1 or not all(np.isfinite(value) and value >= 0 for value in values) or not 0 <= self.seed <= 2_000_000_000:
            raise ValueError("Invalid solver setting")


@dataclass(frozen=True)
class OptimizationInput:
    """Fully in-memory residual optimization; no hidden I/O or forecasting."""
    grid: DeliveryGrid
    plant: PlantParameters
    state: PhysicalState
    market: MarketState
    scenarios: ScenarioBundle
    tree: InformationTree
    auction_eligibility: Mapping[str, tuple[bool, ...]]
    trade_limit_mwh: float
    position_limit_mwh: float
    imbalance_penalty_eur_per_mwh: float
    terminal_target_mwh: float | None = None
    degradation: DegradationSettings = field(default_factory=DegradationSettings)
    grid_import_fee_eur_per_mwh: float = 0.0
    imbalance_limit_mwh: float | None = None
    pending_position_mwh: tuple[float, ...] | np.ndarray | None = None

    def __post_init__(self) -> None:
        for key in ("trade_limit_mwh", "position_limit_mwh", "imbalance_penalty_eur_per_mwh", "grid_import_fee_eur_per_mwh"):
            finite_nonnegative(getattr(self, key), key)
        if self.imbalance_limit_mwh is not None:
            finite_nonnegative(self.imbalance_limit_mwh, "imbalance limit")
        if self.state.timestamp != self.grid.starts[0]:
            raise ValueError("Physical state timestamp must equal residual delivery start")
        battery = self.plant.battery
        if not battery.energy_min_mwh <= self.state.energy_mwh <= battery.maximum_energy:
            raise ValueError("Initial battery energy is outside its bounds")
        if self.degradation.mode == "xu_segmented":
            if battery.capacity_mwh <= 0:
                raise ValueError("Xu cycle aging requires positive battery capacity")
            if battery.self_discharge_per_hour:
                raise ValueError("Xu segmented model currently requires zero self discharge")
            segments = self.state.cycle_segment_energy_mwh
            if segments is None:
                if self.state.cumulative_throughput_mwh or self.state.cumulative_degradation_eur:
                    raise ValueError("A carried Xu state requires cycle_segment_energy_mwh; aging history cannot reset")
                if self.state.origin != "known_delivery_start_assumption":
                    raise ValueError("Xu initialization without segment history requires known_delivery_start_assumption")
            elif len(segments) != self.degradation.segment_count or max(segments) > battery.capacity_mwh / self.degradation.segment_count + 1e-8:
                raise ValueError("Xu segment energy shape or capacity differs from configured segments")
        elif self.state.cycle_segment_energy_mwh is not None:
            raise ValueError("Xu cycle segment history cannot be silently discarded by a throughput model")
        count, intervals = np.shape(self.scenarios.pv_ac_mw)
        if intervals != len(self.grid.interval_ids):
            raise ValueError("Scenario and delivery lengths differ")
        if self.pending_position_mwh is not None:
            pending = np.asarray(self.pending_position_mwh, dtype=float)
            if pending.shape != (intervals,) or not np.all(np.isfinite(pending)):
                raise ValueError("Pending submitted positions must be a finite delivery-length vector")
        fixed_positions = self.market.positions(self.grid)
        if self.pending_position_mwh is not None:
            fixed_positions = fixed_positions + np.asarray(self.pending_position_mwh)
        if np.any(np.abs(fixed_positions) > self.position_limit_mwh + 1e-8):
            raise ValueError("Accepted plus pending submitted positions exceed the commercial limit")
        if set(self.auction_eligibility) != set(self.tree.bid_nodes):
            raise ValueError("Auction masks and bid ownership must match")
        if any(len(mask) != intervals for mask in self.auction_eligibility.values()):
            raise ValueError("Auction eligibility mask has wrong length")
        if not set(self.auction_eligibility).issubset(self.scenarios.auction_prices_eur_per_mwh):
            raise ValueError("Auction prices are missing")
        # Oracle arrays retain rectangular shapes with declared placeholders
        # outside actual auction products. Those coefficients cannot be traded.
        declared_products = self.scenarios.provenance.get("auction_eligible_product_indices")
        if declared_products is not None:
            for auction, mask in self.auction_eligibility.items():
                if auction not in declared_products:
                    raise ValueError("Scenario provenance is missing auction product eligibility")
                allowed = set(declared_products[auction])
                if any(enabled and index not in allowed for index, enabled in enumerate(mask)):
                    raise ValueError("Auction mask permits trading outside the declared actual price products")
        paths = self.tree.paths()
        nodes = {node.node_id: node for node in self.tree.nodes}
        if len(self.tree.leaf_nodes) != count or len(set(self.tree.leaf_nodes)) != count:
            raise ValueError("Each scenario needs its own outcome leaf")
        if np.shape(self.tree.control_nodes) != (count, intervals):
            raise ValueError("Control ownership shape differs from scenarios")
        curtailment = self.tree.curtailment_nodes or self.tree.control_nodes
        if np.shape(curtailment) != (count, intervals):
            raise ValueError("Curtailment ownership shape differs from scenarios")
        if any(len(owners) != count for owners in self.tree.bid_nodes.values()):
            raise ValueError("Bid ownership shape differs from scenarios")
        for auction, owners in self.tree.bid_nodes.items():
            for owner in owners:
                if owner not in paths:
                    raise ValueError("Unknown bid information owner")
                observed = {item for ancestor in paths[owner] for item in nodes[ancestor].observations}
                if f"auction:{auction}" in observed:
                    raise ValueError("An auction bid cannot observe its own unpublished clearing price")
        for node in self.tree.nodes:
            descendants = [s for s, leaf in enumerate(self.tree.leaf_nodes) if leaf in paths and node.node_id in paths[leaf]]
            for observed in node.observations:
                if observed.startswith("auction:"):
                    auction = observed.split(":", 1)[1]
                    if auction in self.scenarios.auction_prices_eur_per_mwh and descendants:
                        prices = np.asarray(self.scenarios.auction_prices_eur_per_mwh[auction])[descendants]
                        if not np.allclose(prices, prices[0], rtol=0, atol=1e-9):
                            raise ValueError("Observed auction prices differ within one information history")
                if observed.startswith("pv:"):
                    product = observed.split(":", 1)[1]
                    if product in self.grid.interval_ids and node.cutoff < self.grid.ends[self.grid.interval_ids.index(product)]:
                        raise ValueError("Completed interval PV cannot be observed before interval end")
        for s, leaf in enumerate(self.tree.leaf_nodes):
            if leaf not in paths or any(node.parent_id == leaf for node in self.tree.nodes):
                raise ValueError("Scenario outcome must be a terminal tree leaf")
            mass = np.prod([nodes[node].conditional_probability for node in paths[leaf]])
            if not np.isclose(mass, self.scenarios.probabilities[s], rtol=0, atol=1e-10):
                raise ValueError("Leaf mass disagrees with scenario probability")
            previous: str | None = None
            for t in range(intervals):
                for owner, deadline in ((self.tree.control_nodes[s][t], self.grid.starts[t]), (curtailment[s][t], self.grid.ends[t] if self.tree.curtailment_nodes else self.grid.starts[t])):
                    if owner not in paths[leaf] or nodes[owner].cutoff > deadline:
                        raise ValueError("Control owner uses unavailable/future information")
                current = self.tree.control_nodes[s][t]
                if previous is not None and previous not in paths[current]:
                    raise ValueError("Successive controls erase observation history")
                previous = current
            for owners in self.tree.bid_nodes.values():
                if owners[s] not in paths[leaf]:
                    raise ValueError("Bid node is not an ancestor of its scenario")
        if self.terminal_target_mwh is not None:
            target = self.terminal_target_mwh
            if not battery.energy_min_mwh <= target <= battery.maximum_energy:
                raise ValueError("Terminal target outside battery bounds")
            lower = upper = self.state.energy_mwh
            for duration in self.grid.delta_hours:
                retention = (1 - battery.self_discharge_per_hour) ** duration
                lower = max(battery.energy_min_mwh, lower * retention - battery.max_discharge_mw * duration / battery.eta_discharge)
                upper = min(battery.maximum_energy, upper * retention + battery.max_charge_mw * duration * battery.eta_charge)
            if not lower - 1e-9 <= target <= upper + 1e-9:
                raise ValueError("Terminal target is unreachable within battery power/energy limits")


@dataclass
class BuiltModel:
    """Gurobi model plus retained handles, without parsing variable names."""
    inputs: OptimizationInput
    model: Any
    variables: dict[str, Any]
    components: dict[str, Any]
    build_seconds: float


@dataclass
class OptimizationResult:
    """Incumbent arrays and separately identified solver diagnostics."""
    status: str
    has_incumbent: bool
    objective_eur: float | None = None
    arrays: dict[str, np.ndarray] = field(default_factory=dict)
    trades: dict[tuple[str, str, str], float] = field(default_factory=dict)
    components: dict[str, np.ndarray] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)
    message: str = ""


@dataclass(frozen=True)
class ValidationReport:
    """Independent numerical residuals; all magnitudes use original model units."""
    valid: bool
    max_errors: Mapping[str, float]
    expected_objective_eur: float | None
    violations: tuple[str, ...] = ()
