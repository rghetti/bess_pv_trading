"""Finite conditional auction trees with explicit information and control events."""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from .domain import DeliveryGrid, InformationNode, InformationTree, ScenarioBundle
from .markets import AuctionEvent


@dataclass
class _History:
    node: str
    probability: float
    prices: dict[str, np.ndarray]
    bid_owners: dict[str, str]
    controls: list[str]
    branches: dict[str, int]


def conditional_price_tree(
    grid: DeliveryGrid, pv_ac_mw: np.ndarray, auction_schedule: Sequence[AuctionEvent], *,
    base_prices: float | np.ndarray = 75.0,
    stage_spreads: Mapping[str, float | np.ndarray] | None = None,
    innovation_sampler: Callable[[str, int, np.random.Generator], np.ndarray] | None = None,
    conditional_mean: Callable[[str, np.ndarray], np.ndarray] | None = None,
    history_mean: Callable[[str, Mapping[str, np.ndarray]], np.ndarray] | None = None,
    branch_probabilities: Sequence[float] = (0.5, 0.5),
    imbalance_price: float | np.ndarray = 60.0,
    known_prices: Mapping[str, np.ndarray] | None = None,
    decision_cutoff: Any | None = None, seed: int = 0,
    origin: str = "stochastic-price / deterministic-PV",
    provenance: Mapping[str, Any] | None = None,
) -> tuple[ScenarioBundle, InformationTree]:
    """Branch *at results*, sharing all decisions until their true observation.

    The chain is DA level followed by successive auction spreads. Independent
    innovations belong to individual conditional equations; there is no second
    cross-auction residual vector. Temporal covariance can be supplied inside the
    sampler. Conditional model features/spreads are frozen for this planning run.
    Known results remain fixed; only unknown future results are generated. Each
    observed result retains independently generated descendants for later auctions.
    history_mean supplies an absolute conditional mean from read-only copies of
    earlier sampled/known auction curves; own and later outcomes are absent.
    """
    intervals = len(grid.interval_ids)
    # Enforce the input contract: Supply either conditional_mean or history_mean, never both.
    if conditional_mean is not None and history_mean is not None:
        raise ValueError("Supply either conditional_mean or history_mean, never both")
    pv = np.asarray(pv_ac_mw, dtype=float)
    # When `pv.shape == (1, intervals)`, set `pv` using `pv[0]`.
    if pv.shape == (1, intervals):
        pv = pv[0]
    # Enforce the input contract: Current store supplies one finite deterministic PV trajectory.
    if pv.shape != (intervals,) or np.any(pv < 0) or not np.all(np.isfinite(pv)):
        raise ValueError("Current store supplies one finite deterministic PV trajectory")
    probabilities = np.asarray(branch_probabilities, dtype=float)
    # Enforce the input contract: Positive conditional branch probabilities must sum to one.
    if probabilities.ndim != 1 or not len(probabilities) or np.any(probabilities <= 0) or not np.isclose(probabilities.sum(), 1):
        raise ValueError("Positive conditional branch probabilities must sum to one")
    known = {key: np.broadcast_to(np.asarray(value, float), (intervals,)).copy() for key, value in (known_prices or {}).items()}
    spread = stage_spreads or {}
    base = np.broadcast_to(np.asarray(base_prices, float), (intervals,))
    imbalance = np.broadcast_to(np.asarray(imbalance_price, float), (intervals,))
    # Enforce the input contract: Price coefficients must be finite; negative prices remain valid.
    if not np.isfinite(base).all() or not np.isfinite(imbalance).all() or any(not np.isfinite(value).all() for value in known.values()):
        raise ValueError("Price coefficients must be finite; negative prices remain valid")
    schedule = tuple(event for event in auction_schedule if event.available and event.results_at is not None)
    # Enforce the input contract: Only one event per auction is supported in a delivery-day tree.
    if len({event.auction for event in schedule}) != len(schedule):
        raise ValueError("Only one event per auction is supported in a delivery-day tree")
    first_event = min([pd.Timestamp(grid.starts[0]), *[event.submission_at for event in schedule]])
    cutoff = first_event if decision_cutoff is None else pd.Timestamp(decision_cutoff)
    # Enforce the input contract: decision_cutoff must be aware and no later than the residual grid start.
    if cutoff.tzinfo is None or cutoff > pd.Timestamp(grid.starts[0]):
        raise ValueError("decision_cutoff must be aware and no later than the residual grid start")
    # Branch *at results*, sharing all decisions until their true observation for each `event` in `schedule`.
    for event in schedule:
        # Enforce the input contract: Known `event.auction` price is not published by decision_cutoff.
        if event.auction in known and event.results_at > cutoff:
            raise ValueError(f"Known {event.auction} price is not published by decision_cutoff")
    # Decision ownership is chronological, including IDA3 interleaving with dispatch.
    events: list[tuple[pd.Timestamp, int, str, Any]] = []
    # Branch *at results*, sharing all decisions until their true observation for each `event` in `schedule`.
    for event in schedule:
        # Branch *at results*, sharing all decisions until their true observation when `event.auction not in known`.
        if event.auction not in known:
            # Enforce the input contract: Previously published `event.auction` price must be supplied in known_prices.
            if event.results_at < cutoff:
                raise ValueError(f"Previously published {event.auction} price must be supplied in known_prices")
            events.append((event.results_at, 0, "result", event))
            # Branch *at results*, sharing all decisions until their true observation when `event.submission_at >= cutoff`.
            if event.submission_at >= cutoff:
                events.append((event.submission_at, 1, "bid", event))
    # Branch *at results*, sharing all decisions until their true observation for each `(t, start)` in `enumerate(grid.starts)`.
    for t, start in enumerate(grid.starts):
        events.append((pd.Timestamp(start), 2, "control", t))
    terminal_at = max([pd.Timestamp(grid.ends[-1]), *[item[0] for item in events]])
    events.append((terminal_at, 3, "terminal", None))
    events.sort(key=lambda item: (item[0], item[1]))
    root_observations = tuple(f"auction:{key}" for key in known)
    nodes = [InformationNode("root", None, cutoff.to_pydatetime(), observations=root_observations,
                             observation_available_at={key: cutoff.to_pydatetime() for key in root_observations})]
    active = [_History("root", 1.0, dict(known), {}, [], {})]
    random = np.random.default_rng(seed)
    generated_bids: list[str] = []
    order = ("DA", "IDA1", "IDA2", "IDA3")
    counter = 0
    # Branch *at results*, sharing all decisions until their true observation for each `(event_at, _, kind, payload)` in `events`.
    for event_at, _, kind, payload in events:
        next_active: list[_History] = []
        # Branch *at results*, sharing all decisions until their true observation when `kind == 'bid'`.
        if kind == "bid":
            generated_bids.append(payload.auction)
        # Branch *at results*, sharing all decisions until their true observation for each `history` in `active`.
        for history in active:
            outcomes = [(0, 1.0, None)]
            # Branch *at results*, sharing all decisions until their true observation when `kind == 'result'`.
            if kind == "result":
                # Enforce the input contract: Unknown prices require an explicit innovation sampler.
                if innovation_sampler is None:
                    raise ValueError("Unknown prices require an explicit innovation sampler")
                innovations = np.asarray(innovation_sampler(payload.auction, intervals, random), float)
                # Enforce the input contract: Innovation sampler must return (conditional branches, intervals).
                if innovations.shape != (len(probabilities), intervals) or not np.isfinite(innovations).all():
                    raise ValueError("Innovation sampler must return (conditional branches, intervals)")
                auction = payload.auction
                preceding = [name for name in order[:order.index(auction)] if name in history.prices]
                previous = history.prices[preceding[-1]] if preceding else base
                mean = base if auction == "DA" else previous + np.broadcast_to(np.asarray(spread.get(auction, 0.0), float), (intervals,))
                # When `history_mean is not None`, set `ancestors` using `{}`.
                if history_mean is not None:
                    ancestors = {}
                    # Branch *at results*, sharing all decisions until their true observation for each `(name, curve)` in `history.prices.items()`.
                    for name, curve in history.prices.items():
                        # When `name in order and order.index(name) < order.index(auction)`, set `protected` using `np.array(curve, dtype=float, copy=True)`.
                        if name in order and order.index(name) < order.index(auction):
                            protected = np.array(curve, dtype=float, copy=True)
                            protected.flags.writeable = False
                            ancestors[name] = protected
                    mean = np.asarray(history_mean(auction, MappingProxyType(ancestors)), float)
                    # Enforce the input contract: History-aware conditional mean must match the finite delivery trajectory.
                    if mean.shape != (intervals,) or not np.isfinite(mean).all():
                        raise ValueError("History-aware conditional mean must match the finite delivery trajectory")
                # When `conditional_mean is not None`, set `mean` using `np.asarray(conditional_mean(auction, previous), float)`.
                elif conditional_mean is not None:
                    mean = np.asarray(conditional_mean(auction, previous), float)
                    # Enforce the input contract: Conditional mean must match the finite delivery trajectory.
                    if mean.shape != (intervals,) or not np.isfinite(mean).all():
                        raise ValueError("Conditional mean must match the finite delivery trajectory")
                # Equal observable vectors are one information outcome. Merge
                # mass before sampling descendants so latent draw labels cannot
                # grant recourse or independently sampled future opportunities.
                distinct: dict[tuple[float, ...], tuple[int, float, np.ndarray]] = {}
                # Branch *at results*, sharing all decisions until their true observation for each `(branch, mass)` in `enumerate(probabilities)`.
                for branch, mass in enumerate(probabilities):
                    price = mean + innovations[branch]
                    key = tuple(price.tolist())
                    # When `key in distinct`, set `(representative, previous_mass, value)` using `distinct[key]`.
                    if key in distinct:
                        representative, previous_mass, value = distinct[key]
                        distinct[key] = (representative, previous_mass + float(mass), value)
                    else:
                        distinct[key] = (branch, float(mass), price)
                outcomes = list(distinct.values())
            # Branch *at results*, sharing all decisions until their true observation for each `(branch, conditional_mass, observed_price)` in `outcomes`.
            for branch, conditional_mass, observed_price in outcomes:
                counter += 1
                node_id = f"n{counter}"
                prices = dict(history.prices)
                bids = dict(history.bid_owners)
                controls = list(history.controls)
                branch_history = dict(history.branches)
                observations: tuple[str, ...] = ()
                availability: dict[str, Any] = {}
                # When `kind == 'result'`, set `auction` using `payload.auction`.
                if kind == "result":
                    auction = payload.auction
                    prices[auction] = observed_price
                    branch_history[auction] = branch
                    observations = (f"auction:{auction}",)
                    availability = {key: event_at.to_pydatetime() for key in observations}
                # When `kind == 'bid'`, set `bids[payload.auction]` using `node_id`.
                elif kind == "bid":
                    bids[payload.auction] = node_id
                # Branch *at results*, sharing all decisions until their true observation when `kind == 'control'`.
                elif kind == "control":
                    controls.append(node_id)
                nodes.append(InformationNode(node_id, history.node, event_at.to_pydatetime(), conditional_mass,
                                             observations, availability))
                next_active.append(_History(node_id, history.probability * conditional_mass, prices, bids, controls, branch_history))
        active = next_active
    auctions = tuple(name for name in order if any(name in history.prices for history in active))
    coefficients = {name: np.stack([history.prices[name] for history in active]) for name in auctions}
    supplied_provenance = dict(provenance or {})
    supplied_provenance.update({"seed": seed, "weather_policy": "freeze_current_vintage",
        "pv_members": 1, "pv_member_origin": "deterministic_weather_vintage",
        "price_chain": ("supplied history-aware absolute means; per-equation innovations" if history_mean is not None
                        else "supplied previous-curve absolute means; per-equation innovations" if conditional_mean is not None
                        else "DA level, successive conditional spreads; per-equation innovations"),
        "known_prices": list(known), "conditional_branch_probabilities": probabilities.tolist(),
        "branch_histories": [history.branches for history in active],
        "duplicate_outcome_policy": "combine equal sibling price-vector masses before generating descendants",
        "imbalance_model": "explicit fixed point estimate; not future realized prices",
        "execution_convention": "pre_interval_battery_and_curtailment; deterministic_PV"})
    bundle = ScenarioBundle(tuple(f"s{s}" for s in range(len(active))), np.array([h.probability for h in active]),
        np.broadcast_to(pv, (len(active), intervals)).copy(), coefficients,
        np.broadcast_to(imbalance, (len(active), intervals)).copy(), origin=origin, provenance=supplied_provenance)
    tree = InformationTree(tuple(nodes), tuple(h.node for h in active),
        {auction: tuple(h.bid_owners[auction] for h in active) for auction in generated_bids},
        tuple(tuple(h.controls) for h in active))
    tree.paths()
    return bundle, tree


def synthetic_price_tree(
    grid: DeliveryGrid, pv_ac_mw: np.ndarray, auction_schedule: Sequence[AuctionEvent], *,
    base_prices: float | np.ndarray = 75.0,
    stage_spreads: Mapping[str, float | np.ndarray] | None = None,
    residual_scale: float | Mapping[str, float] = 10.0,
    branch_values: Sequence[float] = (-1.0, 1.0),
    branch_probabilities: Sequence[float] = (0.5, 0.5),
    imbalance_price: float | np.ndarray = 60.0,
    known_prices: Mapping[str, np.ndarray] | None = None,
    decision_cutoff: Any | None = None, seed: int = 0,
) -> tuple[ScenarioBundle, InformationTree]:
    """Hand-constructed synthetic tree; chosen outcomes carry declared masses.

    A branch has a single whole-day residual level, intentionally a simple
    temporally coherent synthetic model rather than calibrated empirical noise.
    """
    values = np.asarray(branch_values, float)
    # Enforce the input contract: Synthetic branch values/probabilities must have matching lengths.
    if values.shape != np.shape(branch_probabilities) or not np.isfinite(values).all():
        raise ValueError("Synthetic branch values/probabilities must have matching lengths")
    def sampler(auction: str, count: int, random: np.random.Generator) -> np.ndarray:
        scale = residual_scale.get(auction, 0.0) if isinstance(residual_scale, Mapping) else residual_scale
        # Enforce the input contract: Residual scales must be nonnegative.
        if not np.isfinite(scale) or scale < 0:
            raise ValueError("Residual scales must be nonnegative")
        return np.repeat((values * scale)[:, None], count, axis=1)
    return conditional_price_tree(grid, pv_ac_mw, auction_schedule, base_prices=base_prices,
        stage_spreads=stage_spreads, innovation_sampler=sampler, branch_probabilities=branch_probabilities,
        imbalance_price=imbalance_price, known_prices=known_prices, decision_cutoff=decision_cutoff,
        seed=seed, origin="synthetic stochastic-price / deterministic-PV",
        provenance={"residual_method": "declared whole-day level branches; not fitted to prices"})


def fitted_price_tree(grid: DeliveryGrid, pv_ac_mw: np.ndarray, auction_schedule: Sequence[AuctionEvent], *,
                      models: Mapping[str, Any], future_features: Mapping[str, Any],
                      feature_available_at: Mapping[str, Any], residual_distributions: Mapping[str, Any],
                      decision_cutoff: Any, branches: int = 2, residual_method: str = "gaussian",
                      known_prices: Mapping[str, np.ndarray] | None = None,
                      imbalance_price: float | np.ndarray = 60.0, seed: int = 0):
    """Connect eligible trained regressions/OOS residuals to a conditional tree.

    Each auction's residual is its own conditional equation error. Dependence
    across auctions propagates through the previous-price predictor. Whole-vector
    residual draws preserve within-auction temporal covariance. Features apart
    from the conditional predecessor must already be available at this cutoff.
    """
    cutoff = pd.Timestamp(decision_cutoff)
    # Enforce the input contract: Fitted scenario generation needs an aware cutoff and positive branch count.
    if cutoff.tzinfo is None or branches < 1:
        raise ValueError("Fitted scenario generation needs an aware cutoff and positive branch count")
    intervals = len(grid.interval_ids)
    known = known_prices or {}
    # Connect eligible trained regressions/OOS residuals to a conditional tree for each `event` in `auction_schedule`.
    for event in auction_schedule:
        auction = event.auction
        # Skip the current item when `not event.available or auction in known`.
        if not event.available or auction in known:
            continue
        # Enforce the input contract: Missing trained model, OOS residual distribution or features for `auction`.
        if auction not in models or auction not in residual_distributions or auction not in future_features:
            raise ValueError(f"Missing trained model, OOS residual distribution or features for {auction}")
        availability = pd.Timestamp(feature_available_at[auction])
        # Enforce the input contract: Future/unpublished `auction` features are ineligible.
        if availability.tzinfo is None or availability > cutoff:
            raise ValueError(f"Future/unpublished {auction} features are ineligible")
        # Enforce the input contract: `auction` model/calibration uses future data.
        if pd.Timestamp(models[auction].fit_cutoff) > cutoff or pd.Timestamp(residual_distributions[auction].calibration_cutoff) > cutoff:
            raise ValueError(f"{auction} model/calibration uses future data")
        # Enforce the input contract: `auction` residual grid differs; calibrate/map products explicitly, including DST.
        if np.asarray(residual_distributions[auction].vectors).shape[1] != intervals:
            raise ValueError(f"{auction} residual grid differs; calibrate/map products explicitly, including DST")
    def sampler(auction, count, random):
        return residual_distributions[auction].sample(branches, method=residual_method, random=random)
    def mean(auction, previous):
        features = future_features[auction].copy()
        # When `auction != 'DA'`, set `features['previous_auction_price']` using `previous`.
        if auction != "DA":
            features["previous_auction_price"] = previous
        prediction = models[auction].predict(features)
        return prediction if auction == "DA" else previous + prediction
    return conditional_price_tree(grid, pv_ac_mw, auction_schedule, innovation_sampler=sampler,
        conditional_mean=mean, branch_probabilities=np.full(branches, 1 / branches),
        known_prices=known, decision_cutoff=cutoff, imbalance_price=imbalance_price, seed=seed,
        provenance={"residual_method": residual_method, "model_source": "eligible ridge models and rolling-origin OOS residuals",
                    "conditional_dependence": "successive spread regressions; conditional innovations assumed independent across auctions"})
