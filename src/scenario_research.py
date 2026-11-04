"""Joint trajectory transformations, history partitions and finite-law scores.

ECC preserves a supplied ensemble rank template. Reduction is a weighted
feature-space medoid heuristic, not a bound on the resulting MILP value loss.
All algorithms operate on forecasts already selected at the decision cutoff;
they never read realized outcomes or acquire data.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np


def _readonly(values, *, dtype=float):
    """Copy array inputs so callers cannot mutate a stored scenario law."""
    result = np.array(values, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _freeze_metadata(value):
    """Freeze nested provenance without retaining caller-owned mutable objects."""
    if isinstance(value, Mapping):
        return MappingProxyType({key: _freeze_metadata(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_metadata(item) for item in value)
    if isinstance(value, np.ndarray):
        return _readonly(value, dtype=value.dtype)
    return value


def _weights(values, count):
    """Require an actual probability vector; never silently renormalize it."""
    result = np.asarray(values, dtype=float)
    if (result.shape != (count,) or not np.isfinite(result).all()
            or np.any(result < 0) or not np.isclose(result.sum(), 1, atol=1e-10, rtol=0)):
        raise ValueError("Scenario weights must be finite, nonnegative and sum to one")
    return result


@dataclass(frozen=True)
class ScenarioTrajectories:
    """One aligned finite law; each row couples PV with both settlement prices."""

    pv_ac_mw: np.ndarray
    weights: np.ndarray
    scenario_ids: tuple[str, ...]
    positive_prices_eur_per_mwh: np.ndarray | None = None
    negative_prices_eur_per_mwh: np.ndarray | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        pv = np.asarray(self.pv_ac_mw, dtype=float)
        identifiers = tuple(self.scenario_ids)
        if (pv.ndim != 2 or min(pv.shape) < 1 or not np.isfinite(pv).all() or np.any(pv < 0)
                or len(identifiers) != pv.shape[0] or len(set(identifiers)) != len(identifiers)
                or any(not isinstance(item, str) or not item for item in identifiers)):
            raise ValueError("Scenarios need finite nonnegative PV and unique aligned identifiers")
        object.__setattr__(self, "pv_ac_mw", _readonly(pv))
        object.__setattr__(self, "weights", _readonly(_weights(self.weights, len(pv))))
        object.__setattr__(self, "scenario_ids", identifiers)
        # A missing negative curve means single-price settlement downstream.
        if self.negative_prices_eur_per_mwh is not None and self.positive_prices_eur_per_mwh is None:
            raise ValueError("A negative settlement curve requires the positive curve")
        for name in ("positive_prices_eur_per_mwh", "negative_prices_eur_per_mwh"):
            values = getattr(self, name)
            if values is not None:
                array = np.asarray(values, dtype=float)
                if array.shape != pv.shape or not np.isfinite(array).all():
                    raise ValueError("Joint settlement trajectories must match the PV matrix")
                object.__setattr__(self, name, _readonly(array))
        object.__setattr__(self, "provenance", _freeze_metadata(self.provenance))


def transform_dependence(bundle: ScenarioTrajectories, mode="raw", *, seed=0,
                         calibrated_pv_marginals=None, reference_pv=None):
    """Keep raw paths, shuffle joint interval samples, or apply PV ensemble copula coupling.

    ECC assigns supplied marginal samples to the rank order of ``reference_pv``
    (the raw PV by default). Prices retain their original row identities, so
    their rank association with PV is retained except where raw PV ties need
    breaking. Ties use a seeded random order. Neither operation estimates a
    calibration model. Rank and shuffle operations require equal weights.
    """
    if mode not in {"raw", "independent_shuffle", "ecc"}:
        raise ValueError("Unknown scenario dependence mode")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("Scenario seed must be a nonnegative integer")
    if mode != "ecc" and (calibrated_pv_marginals is not None or reference_pv is not None):
        raise ValueError("Calibrated marginals and rank templates apply only to ECC")
    if mode == "raw":
        return bundle
    count, length = bundle.pv_ac_mw.shape
    if not np.allclose(bundle.weights, 1 / count, atol=1e-12, rtol=0):
        raise ValueError("ECC and independent shuffle require equal scenario weights")
    random = np.random.default_rng(seed)
    prices = bundle.positive_prices_eur_per_mwh
    negative = bundle.negative_prices_eur_per_mwh
    # Use the same per-column permutation for every physical and price variable.
    if mode == "independent_shuffle":
        indices = np.argsort(random.random((count, length)), axis=0, kind="stable")
        pv = np.take_along_axis(bundle.pv_ac_mw, indices, axis=0)
        prices = None if prices is None else np.take_along_axis(prices, indices, axis=0)
        negative = None if negative is None else np.take_along_axis(negative, indices, axis=0)
        details = {"joint_interval_pairs_preserved": True}
    else:
        if calibrated_pv_marginals is None:
            raise ValueError("ECC requires explicit calibrated PV marginal samples")
        samples = np.asarray(calibrated_pv_marginals, dtype=float)
        template = bundle.pv_ac_mw if reference_pv is None else np.asarray(reference_pv, dtype=float)
        if (samples.shape != (count, length) or template.shape != samples.shape
                or not np.isfinite(samples).all() or not np.isfinite(template).all()
                or np.any(samples < 0) or np.any(template < 0)):
            raise ValueError("ECC marginals and rank template must be aligned nonnegative PV")
        # Lexicographic ranks resolve identical raw forecasts reproducibly.
        order = np.lexsort((random.random(template.shape), template), axis=0)
        ranks = np.argsort(order, axis=0, kind="stable")
        pv = np.take_along_axis(np.sort(samples, axis=0), ranks, axis=0)
        details = {"rank_template": "raw" if reference_pv is None else "supplied",
                   "ties": "seeded_random", "unchanged_pv": bool(np.array_equal(pv, bundle.pv_ac_mw))}
    provenance = {**bundle.provenance, "dependence_transform": {"mode": mode, "seed": int(seed), **details}}
    return ScenarioTrajectories(pv, bundle.weights, bundle.scenario_ids, prices, negative, provenance)


def observable_history_nodes(pv_ac_mw, *, revelation_lag_intervals=1,
                             branch_intervals=None, observation_quantum_mw=None,
                             observed_features=None, feature_quantum=None):
    """Partition scenarios only using observations available before each action.

    A lag of one means interval ``t`` PV becomes visible at ``t+1``. Coarse bins
    deliberately reveal less information; exact histories in a finite ensemble
    may identify a member after its first distinct observation. Empty branch
    intervals disable recourse. Extra features follow the same stated lag;
    callers must not place unobserved settlement prices in these features.
    """
    pv = np.asarray(pv_ac_mw, dtype=float)
    if pv.ndim != 2 or min(pv.shape) < 1 or not np.isfinite(pv).all() or np.any(pv < 0):
        raise ValueError("History partitions require a finite nonnegative PV matrix")
    count, length = pv.shape
    if (isinstance(revelation_lag_intervals, bool)
            or not isinstance(revelation_lag_intervals, (int, np.integer)) or revelation_lag_intervals < 1):
        raise ValueError("History revelation lag must be a positive integer")
    branches = tuple(range(1, length)) if branch_intervals is None else tuple(branch_intervals)
    if (any(isinstance(t, bool) or not isinstance(t, (int, np.integer)) or not 1 <= t < length for t in branches)
            or len(set(branches)) != len(branches)):
        raise ValueError("Branch intervals must be unique horizon indices strictly after the root")
    features = pv[:, :, None]
    quantum = [observation_quantum_mw or 0.0]
    if observed_features is not None:
        extra = np.asarray(observed_features, dtype=float)
        if extra.ndim != 3 or extra.shape[:2] != pv.shape or not np.isfinite(extra).all():
            raise ValueError("Observed features must have shape (scenarios, intervals, features)")
        supplied = np.zeros(extra.shape[2]) if feature_quantum is None else np.asarray(feature_quantum, dtype=float)
        if supplied.shape != (extra.shape[2],):
            raise ValueError("One observation quantum is required per additional feature")
        features = np.concatenate((features, extra), axis=2)
        quantum.extend(supplied)
    elif feature_quantum is not None:
        raise ValueError("Feature quantization requires observed_features")
    quantum = np.asarray(quantum, dtype=float)
    if not np.isfinite(quantum).all() or np.any(quantum < 0) or isinstance(observation_quantum_mw, bool):
        raise ValueError("Observation quantum must be nonnegative and finite")
    # Zero quantum keeps exact values; positive quantum uses fixed-width bins.
    features = features.copy()
    discretized = quantum > 0
    features[:, :, discretized] = np.floor(features[:, :, discretized] / quantum[discretized])
    nodes = np.zeros((count, length), dtype=np.int64)
    next_node = 1
    for t in range(1, length):
        visible_length = max(0, t - revelation_lag_intervals + 1)
        if t not in branches or visible_length == 0:
            nodes[:, t] = nodes[:, t - 1]
            continue
        # Previous ownership prevents branches from merging when bins coincide.
        histories = np.column_stack((nodes[:, t - 1], features[:, :visible_length].reshape(count, -1)))
        _, inverse = np.unique(histories, axis=0, return_inverse=True)
        nodes[:, t] = inverse + next_node
        next_node += int(inverse.max()) + 1
    return _readonly(nodes, dtype=np.int64)


def validate_history_nodes(nodes, pv_ac_mw, **history_settings):
    """Reject anticipative splits and recombination; coarser histories are legal."""
    exact = observable_history_nodes(pv_ac_mw, **history_settings)
    supplied = np.asarray(nodes)
    if supplied.shape != exact.shape or supplied.dtype.kind not in "iu":
        raise ValueError("History node IDs must be an aligned integer matrix")
    if np.any(supplied < 0) or len(np.unique(supplied[:, 0])) != 1:
        raise ValueError("History tree requires one shared root and nonnegative node IDs")
    for t in range(supplied.shape[1]):
        # Indistinguishable observations cannot be assigned different controls.
        for group in np.unique(exact[:, t]):
            if len(np.unique(supplied[exact[:, t] == group, t])) != 1:
                raise ValueError("History nodes split indistinguishable observed histories")
        if t:
            for group in np.unique(supplied[:, t]):
                if len(np.unique(supplied[supplied[:, t] == group, t - 1])) != 1:
                    raise ValueError("History nodes merge previously distinct branches")
    return _readonly(supplied, dtype=np.int64)


@dataclass(frozen=True)
class ReductionSettings:
    """Feature weights and units for deterministic forward medoid selection.

    Each block is divided by its number of coordinates and its physical scale.
    Saturation and POI features are explanatory proxies, not simulated dispatch.
    Missing optional asset/price inputs disable only their associated blocks.
    """

    pv_weight: float = 1.0
    ramp_weight: float = 0.0
    cumulative_deficit_weight: float = 1.0
    cumulative_balance_weight: float = 1.0
    saturation_weight: float = 1.0
    poi_weight: float = 1.0
    settlement_weight: float = 1.0
    price_weight: float = 0.0
    power_scale_mw: float = 1.0
    energy_scale_mwh: float = 1.0
    money_scale_eur: float = 100.0
    price_scale_eur_per_mwh: float = 100.0
    distance_block_size: int = 128

    def __post_init__(self):
        for name, value in asdict(self).items():
            if name == "distance_block_size":
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError("Reduction distance_block_size must be a positive integer")
            elif (isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value)
                  or value < 0 or ("scale" in name and value == 0)):
                raise ValueError(f"Invalid scenario reduction {name}")
        if not any(value > 0 for name, value in asdict(self).items() if name.endswith("weight")):
            raise ValueError("Scenario reduction requires a positive feature weight")


@dataclass(frozen=True)
class ReductionResult:
    """Reduced law and the source-row mapping needed to audit probability transport."""

    bundle: ScenarioTrajectories
    selected_indices: np.ndarray
    assignment: np.ndarray
    weighted_distance: float

    def __post_init__(self):
        object.__setattr__(self, "selected_indices", _readonly(self.selected_indices, dtype=np.int64))
        object.__setattr__(self, "assignment", _readonly(self.assignment, dtype=np.int64))


def _reduction_features(bundle, settings, committed_mwh, delta_hours, battery_capacity_mwh,
                        poi_export_mw, poi_import_mw):
    """Build only features derivable from the forecast law and declared assets."""
    pv = bundle.pv_ac_mw
    count, length = pv.shape
    durations = np.broadcast_to(np.asarray(delta_hours, dtype=float), (length,))
    commitments = np.zeros(length) if committed_mwh is None else np.asarray(committed_mwh, dtype=float)
    if (commitments.shape != (length,) or not np.isfinite(commitments).all()
            or not np.isfinite(durations).all() or np.any(durations <= 0)):
        raise ValueError("Reduction commitments and positive interval durations must align")
    for name, value in (("battery_capacity_mwh", battery_capacity_mwh), ("poi_export_mw", poi_export_mw),
                        ("poi_import_mw", poi_import_mw)):
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value < 0):
            raise ValueError(f"Reduction {name} must be finite and nonnegative")
    blocks, names = [], []

    def add(name, values, weight, scale):
        # Equal block lengths should not accidentally change their chosen weight.
        if weight > 0 and values.shape[1]:
            blocks.append(values * np.sqrt(weight / values.shape[1]) / scale)
            names.append(name)

    balance = pv * durations - commitments
    cumulative = np.cumsum(balance, axis=1)
    add("pv", pv, settings.pv_weight, settings.power_scale_mw)
    add("ramps", np.diff(pv, axis=1), settings.ramp_weight, settings.power_scale_mw)
    add("cumulative_deficit", np.cumsum(np.maximum(-balance, 0), axis=1),
        settings.cumulative_deficit_weight, settings.energy_scale_mwh)
    add("cumulative_balance", cumulative, settings.cumulative_balance_weight, settings.energy_scale_mwh)
    if battery_capacity_mwh is not None:
        saturation = np.concatenate((np.maximum(cumulative - battery_capacity_mwh, 0),
                                     np.maximum(-cumulative - battery_capacity_mwh, 0)), axis=1)
        add("capacity_exceedance_proxy", saturation, settings.saturation_weight, settings.energy_scale_mwh)
    if poi_export_mw is not None:
        add("poi_export_exceedance", np.maximum(pv - poi_export_mw, 0), settings.poi_weight, settings.power_scale_mw)
    if poi_import_mw is not None:
        add("poi_import_shortfall", np.maximum(-balance / durations - poi_import_mw, 0),
            settings.poi_weight, settings.power_scale_mw)
    if bundle.positive_prices_eur_per_mwh is not None:
        positive = bundle.positive_prices_eur_per_mwh
        negative = positive if bundle.negative_prices_eur_per_mwh is None else bundle.negative_prices_eur_per_mwh
        settlement = np.maximum(balance, 0) * positive - np.maximum(-balance, 0) * negative
        add("uncontrolled_settlement", settlement, settings.settlement_weight, settings.money_scale_eur)
        add("prices", np.concatenate((positive, negative), axis=1), settings.price_weight,
            settings.price_scale_eur_per_mwh)
    if not blocks:
        raise ValueError("No reduction features are available for the requested weights")
    return np.concatenate(blocks, axis=1), names


def _distances(features, candidates):
    """Euclidean distances with O(N times candidate-block) temporary memory."""
    squared = np.sum(features * features, axis=1)
    cross = features @ features[candidates].T
    distances = np.sqrt(np.maximum(squared[:, None] + squared[candidates][None, :] - 2 * cross, 0))
    # Cancellation in the Gram identity must not give a member distance to itself.
    distances[candidates, np.arange(len(candidates))] = 0
    return distances


def reduce_scenarios(bundle: ScenarioTrajectories, count, *, settings=None, committed_mwh=None,
                     delta_hours=0.25, battery_capacity_mwh=None, poi_export_mw=None, poi_import_mw=None):
    """Greedily retain source members and transfer each source mass to its nearest medoid.

    ``assignment[i]`` indexes the reduced bundle, while ``selected_indices[j]``
    indexes the original bundle. The algorithm is deterministic and evaluates
    candidate blocks without allocating a scenarios-by-scenarios-by-time cube.
    Its distance is a forecast-feature diagnostic, not an economic regret bound.
    """
    settings = settings or ReductionSettings()
    positive_members = np.flatnonzero(bundle.weights > 0)
    if (isinstance(count, bool) or not isinstance(count, (int, np.integer))
            or not 1 <= count <= len(positive_members)):
        raise ValueError("Reduction count must lie between one and the positive-weight member count")
    features, feature_names = _reduction_features(bundle, settings, committed_mwh, delta_hours,
        battery_capacity_mwh, poi_export_mw, poi_import_mw)
    nearest = np.full(len(bundle.weights), np.inf)
    selected = []
    # Forward selection minimizes weighted nearest-medoid distance at each step.
    for _ in range(count):
        candidates = positive_members[~np.isin(positive_members, selected)]
        best_index, best_cost = None, np.inf
        for offset in range(0, len(candidates), settings.distance_block_size):
            block = candidates[offset:offset + settings.distance_block_size]
            distances = _distances(features, block)
            costs = bundle.weights @ np.minimum(nearest[:, None], distances)
            position = int(np.argmin(costs))
            if costs[position] < best_cost:
                best_index, best_cost = int(block[position]), float(costs[position])
        selected.append(best_index)
        nearest = np.minimum(nearest, _distances(features, np.array([best_index]))[:, 0])
    indices = np.asarray(selected, dtype=int)
    # Assign in blocks too: retaining many members never allocates an N-by-N array.
    assignment = np.zeros(len(bundle.weights), dtype=int)
    nearest = np.full(len(bundle.weights), np.inf)
    for offset in range(0, len(indices), settings.distance_block_size):
        block = indices[offset:offset + settings.distance_block_size]
        distances = _distances(features, block)
        positions = np.argmin(distances, axis=1)
        candidate_distances = distances[np.arange(len(bundle.weights)), positions]
        improved = candidate_distances < nearest
        assignment[improved] = offset + positions[improved]
        nearest[improved] = candidate_distances[improved]
    # Tied duplicate representatives keep their own positive source probability.
    assignment[indices] = np.arange(count)
    nearest[indices] = 0
    weights = np.bincount(assignment, weights=bundle.weights, minlength=count)
    distance = float(bundle.weights @ nearest)
    provenance = {**bundle.provenance, "scenario_reduction": {
        "method": "greedy_weighted_feature_medoids", "source_count": len(bundle.weights),
        "retained_count": int(count), "selected_indices": indices.tolist(),
        "assignment": assignment.tolist(), "weighted_feature_distance": distance,
        "active_features": feature_names, "settings": asdict(settings),
        "commitments_supplied": committed_mwh is not None, "milp_value_loss_bound": False}}
    prices = bundle.positive_prices_eur_per_mwh
    negative = bundle.negative_prices_eur_per_mwh
    reduced = ScenarioTrajectories(bundle.pv_ac_mw[indices], weights,
        tuple(bundle.scenario_ids[index] for index in indices), None if prices is None else prices[indices],
        None if negative is None else negative[indices], provenance)
    return ReductionResult(reduced, indices, assignment, distance)


def _score_inputs(scenarios, observation, weights):
    """Validate finite aligned scores without dropping missing targets silently."""
    values = np.asarray(scenarios, dtype=float)
    truth = np.asarray(observation, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or truth.shape != (values.shape[1],):
        raise ValueError("Scores need (scenarios, coordinates) forecasts and one aligned observation")
    if not np.isfinite(values).all() or not np.isfinite(truth).all():
        raise ValueError("Scores require finite forecasts and observations")
    probabilities = np.full(len(values), 1 / len(values)) if weights is None else _weights(weights, len(values))
    return values, truth, probabilities


def weighted_crps(scenarios, observation, weights=None):
    """Exact CRPS of each coordinate's finite distribution in O(T S log S)."""
    values, truth, probabilities = _score_inputs(scenarios, observation, weights)
    order = np.argsort(values, axis=0, kind="stable")
    sorted_values = np.take_along_axis(values, order, axis=0)
    sorted_weights = probabilities[order]
    before = np.cumsum(sorted_weights, axis=0) - sorted_weights
    half_pairwise = np.sum(sorted_weights * sorted_values * (2 * before + sorted_weights - 1), axis=0)
    return probabilities @ np.abs(values - truth) - half_pairwise


def energy_score(scenarios, observation, weights=None, *, coordinate_weights=None, block_size=128):
    """Exact energy score of the weighted finite joint law, using Euclidean distance."""
    values, truth, probabilities = _score_inputs(scenarios, observation, weights)
    if isinstance(block_size, bool) or not isinstance(block_size, int) or block_size < 1:
        raise ValueError("Energy-score block_size must be a positive integer")
    coordinate = np.ones(values.shape[1]) if coordinate_weights is None else np.asarray(coordinate_weights, dtype=float)
    if (coordinate.shape != truth.shape or not np.isfinite(coordinate).all()
            or np.any(coordinate < 0) or not np.any(coordinate > 0)):
        raise ValueError("Coordinate weights must be nonnegative, finite and not all zero")
    scaled = values * np.sqrt(coordinate)
    first = probabilities @ np.linalg.norm((values - truth) * np.sqrt(coordinate), axis=1)
    second = 0.0
    for start in range(0, len(values), block_size):
        indices = np.arange(start, min(len(values), start + block_size))
        second += probabilities @ _distances(scaled, indices) @ probabilities[indices]
    return float(first - second / 2)


def variogram_score(scenarios, observation, weights=None, *, exponent=0.5, lag_weights=None):
    """Weighted variogram score over each unique time pair, grouped by temporal lag.

    ``lag_weights[k-1]`` weights all pairs separated by ``k`` intervals. The
    unique-pair convention is half a symmetric double sum with identical weights.
    """
    values, truth, probabilities = _score_inputs(scenarios, observation, weights)
    if isinstance(exponent, bool) or not np.isfinite(exponent) or not 0 < exponent < 2:
        raise ValueError("Variogram exponent must lie strictly between zero and two")
    length = values.shape[1]
    lags = np.ones(max(0, length - 1)) if lag_weights is None else np.asarray(lag_weights, dtype=float)
    if lags.shape != (length - 1,) or not np.isfinite(lags).all() or np.any(lags < 0):
        raise ValueError("Variogram lag_weights must align and be finite nonnegative")
    score = 0.0
    for lag in range(1, length):
        predicted = probabilities @ np.abs(values[:, lag:] - values[:, :-lag]) ** exponent
        observed = np.abs(truth[lag:] - truth[:-lag]) ** exponent
        score += lags[lag - 1] * np.sum((observed - predicted) ** 2)
    return float(score)
