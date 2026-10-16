"""Perfect-price-information benchmarks with causal PV member observations.

Actual clearing prices intentionally replace uncertain price coefficients only
for a benchmark. PV member identities remain latent until their completed
interval outputs distinguish them. The same physical and market constraints
can therefore be used to compare a feasible forecasting policy with an oracle.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ..domain import DeliveryGrid, InformationNode, InformationTree, ScenarioBundle
from ..markets import AuctionEvent


def _timestamps(values: Sequence[Any], label: str) -> pd.DatetimeIndex:
    """Require explicit aware clocks and normalize pandas timestamp resolutions."""
    timestamps = [pd.Timestamp(value) for value in values]
    if any(pd.isna(value) or value.tzinfo is None for value in timestamps):
        raise ValueError(f"{label} requires finite timezone-aware timestamps")
    return pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True)).as_unit("ns")


def _actual_vector(value: Any, grid: DeliveryGrid, auction: str,
                   eligible: Sequence[int]) -> tuple[np.ndarray, str]:
    """Align real native labels exactly; no hourly-to-quarter-hour expansion."""
    eligible_indices = np.array(sorted(eligible), dtype=int)
    if isinstance(value, pd.DataFrame):
        fields = [name for name in ("price_eur_mwh", "actual") if name in value]
        if not fields or not {"product_start", "product_end"}.issubset(value):
            raise ValueError(f"{auction} actual price rows require native boundaries and price_eur_mwh/actual")
        frame = value.copy()
        frame["product_start"] = _timestamps(frame.product_start, "actual product_start")
        frame["product_end"] = _timestamps(frame.product_end, "actual product_end")
        frame = frame.sort_values("product_start")
        if len(fields) > 1 and not np.array_equal(frame[fields[0]].to_numpy(float), frame[fields[1]].to_numpy(float)):
            raise ValueError(f"{auction} actual and price_eur_mwh aliases conflict; actual prices are ambiguous")
        if "unit" in frame and not frame.unit.astype(str).str.upper().str.replace(" ", "").eq("EUR/MWH").all():
            raise ValueError("Oracle price labels must use EUR/MWh")
        expected_starts = _timestamps(grid.starts, "delivery start")[eligible_indices]
        expected_ends = _timestamps(grid.ends, "delivery end")[eligible_indices]
        if (not np.array_equal(frame.product_start.array.asi8, expected_starts.asi8)
                or not np.array_equal(frame.product_end.array.asi8, expected_ends.asi8)):
            raise ValueError(f"{auction} actual native price products differ from the delivery grid; no interpolation is permitted")
        result = frame[fields[0]].to_numpy(float)
        alignment = ("exact native product_start/product_end match" if len(eligible_indices) == len(grid.interval_ids)
                     else "exact native product_start/product_end match on auction-eligible products only")
    else:
        result = np.asarray(value, float)
        alignment = "caller-supplied array explicitly ordered on auction-eligible native delivery products"
    if result.shape != (len(eligible_indices),) or not np.isfinite(result).all():
        raise ValueError(f"{auction} actual prices must be one finite value per auction-eligible native delivery product")
    # ScenarioBundle uses a rectangular grid, while IDA3 genuinely has no morning
    # products. These zeros are inert coefficients outside the explicit trading
    # mask, never fabricated observed clearing prices or training labels.
    expanded = np.zeros(len(grid.interval_ids), dtype=float)
    expanded[eligible_indices] = result
    return expanded, alignment


def _coefficient_array(value: Any, members: int, intervals: int, label: str) -> np.ndarray:
    """Broadcast an explicit settlement assumption without inventing its value."""
    try:
        result = np.broadcast_to(np.asarray(value, float), (members, intervals)).copy()
    except ValueError as exc:
        raise ValueError(f"{label} must broadcast to (PV member, delivery product)") from exc
    if not np.isfinite(result).all():
        raise ValueError(f"{label} must be finite")
    return result


@dataclass
class _MemberHistory:
    """Keep indistinguishable PV members together until an observable split."""
    node: str
    members: tuple[int, ...]
    bids: dict[str, str]
    controls: dict[int, str]
    curtailments: dict[int, str]


def build_perfect_foresight_scenarios(
    grid: DeliveryGrid, pv_members_ac_mw: np.ndarray, actual_prices: Mapping[str, Any],
    auction_schedule: Sequence[AuctionEvent], *, decision_cutoff: Any,
    imbalance_prices_eur_per_mwh: Any, member_probabilities: Sequence[float] | None = None,
    member_ids: Sequence[str] | None = None, pv_observation_times: Sequence[Any] | None = None,
    imbalance_negative_prices_eur_per_mwh: Any | None = None,
    actual_price_provenance: str = "caller supplied realized clearing prices; benchmark only",
    imbalance_price_provenance: str = "caller supplied explicit settlement prices/assumption",
) -> tuple[ScenarioBundle, InformationTree]:
    """Build one no-noise realized-price leaf per PV member, without revealing it.

    Future clearing prices are deterministic objective coefficients, matching the
    existing synthetic oracle convention. Auction observation labels retain
    their actual scheduled result clocks, so this benchmark cannot masquerade
    as an executable forecast. Separate end-of-interval curtailment ownership is
    supported by the existing optimizer. Delayed PV observations conservatively
    keep curtailment and battery commands shared until information arrives.
    Actual labels cover only each auction's eligible products. Rectangular-array
    zero placeholders outside that mask are audited and must never be tradable.
    """
    cutoff = _timestamps([decision_cutoff], "decision_cutoff")[0]
    if cutoff > pd.Timestamp(grid.starts[0]):
        raise ValueError("Oracle decision_cutoff cannot follow the residual delivery start")
    pv = np.asarray(pv_members_ac_mw, float)
    if pv.ndim == 1:
        pv = pv[None, :]
    if pv.ndim != 2 or pv.shape[1] != len(grid.interval_ids) or not len(pv) or not np.isfinite(pv).all() or (pv < 0).any():
        raise ValueError("PV members must be finite nonnegative (member, delivery product) AC MW values")
    members, intervals = pv.shape
    weights = np.full(members, 1 / members) if member_probabilities is None else np.asarray(member_probabilities, float)
    if weights.shape != (members,) or not np.isfinite(weights).all() or (weights <= 0).any() or not np.isclose(weights.sum(), 1, rtol=0, atol=1e-10):
        raise ValueError("PV member probabilities must be positive and sum to one")
    identifiers = tuple(f"pv_member_{index}" for index in range(members)) if member_ids is None else tuple(member_ids)
    if len(identifiers) != members or len(set(identifiers)) != members or any(not isinstance(value, str) or not value for value in identifiers):
        raise ValueError("PV member IDs must be unique nonempty strings matching member count")
    if not actual_prices or not actual_price_provenance or not imbalance_price_provenance:
        raise ValueError("Oracle actual price inputs and their benchmark provenance must be explicit")
    allowed = {"DA", "IDA1", "IDA2", "IDA3"}
    if set(actual_prices) - allowed or any(event.auction not in allowed for event in auction_schedule):
        raise ValueError("Oracle prices support only DA/IDA1/IDA2/IDA3 actual auction labels")
    schedule = tuple(event for event in auction_schedule if event.available and event.results_at is not None)
    if len({event.auction for event in schedule}) != len(schedule):
        raise ValueError("Only one event per auction is supported")
    requested = {event.auction for event in schedule}
    if set(actual_prices) != requested:
        raise ValueError("Supply actual prices for exactly the available scheduled auctions; missing IDA prices are never fabricated")
    eligible_by_auction = {}
    for event in schedule:
        eligible = tuple(event.eligible_intervals)
        if (len(set(eligible)) != len(eligible)
                or any(isinstance(index, bool) or not isinstance(index, (int, np.integer))
                       or not 0 <= index < intervals for index in eligible)):
            raise ValueError("Auction eligible intervals must be unique valid residual-grid product indices")
        eligible_by_auction[event.auction] = tuple(sorted(eligible))
    observed_at = _timestamps(grid.ends if pv_observation_times is None else pv_observation_times, "pv_observation_times")
    if len(observed_at) != intervals or (observed_at < _timestamps(grid.ends, "delivery end")).any():
        raise ValueError("Completed interval PV cannot be observed before its interval end")
    # Requiring ordered observation times preserves the simple completed-prefix
    # interpretation; arbitrary forecast-issuance timestamps are not substituted.
    if not observed_at.is_monotonic_increasing:
        raise ValueError("PV observation times must be chronological")
    prices, alignment = {}, {}
    for auction, value in actual_prices.items():
        vector, alignment[auction] = _actual_vector(value, grid, auction, eligible_by_auction[auction])
        prices[auction] = np.broadcast_to(vector, (members, intervals)).copy()
    imbalance = _coefficient_array(imbalance_prices_eur_per_mwh, members, intervals, "imbalance prices")
    negative_imbalance = (None if imbalance_negative_prices_eur_per_mwh is None else
        _coefficient_array(imbalance_negative_prices_eur_per_mwh, members, intervals, "negative imbalance prices"))

    root_observations = tuple(f"auction:{event.auction}" for event in schedule if event.results_at <= cutoff)
    nodes = [InformationNode("root", None, cutoff.to_pydatetime(), observations=root_observations,
                             observation_available_at={key: cutoff.to_pydatetime() for key in root_observations})]
    active = [_MemberHistory("root", tuple(range(members)), {}, {}, {})]
    # Same-clock order makes a just-completed PV observation available to the
    # next battery command and to this interval's automatic curtailment action.
    events: list[tuple[pd.Timestamp, int, str, Any]] = []
    for event in schedule:
        if event.results_at > cutoff:
            events.append((event.results_at, 1, "result", event.auction))
        if event.submission_at >= cutoff:
            events.append((event.submission_at, 2, "bid", event.auction))
    for index, (start, end, observation) in enumerate(zip(grid.starts, grid.ends, observed_at)):
        events.append((observation, 0, "pv_observation", index))
        events.append((pd.Timestamp(start), 3, "control", index))
        events.append((pd.Timestamp(end), 4, "curtailment", index))
    events.sort(key=lambda value: (value[0], value[1]))
    counter = 0
    for event_at, _, kind, payload in events:
        following = []
        for history in active:
            groups: dict[float, tuple[int, ...]] = {}
            if kind == "pv_observation":
                for value in np.unique(pv[list(history.members), payload]):
                    groups[float(value)] = tuple(member for member in history.members if pv[member, payload] == value)
            else:
                groups[0.] = history.members
            parent_mass = weights[list(history.members)].sum()
            for group in groups.values():
                counter += 1
                node_id = f"oracle_n{counter}"
                bids, controls, curtailments = dict(history.bids), dict(history.controls), dict(history.curtailments)
                observations = ()
                if kind == "pv_observation":
                    observations = (f"pv:{grid.interval_ids[payload]}",)
                elif kind == "result":
                    observations = (f"auction:{payload}",)
                elif kind == "bid":
                    bids[payload] = node_id
                elif kind == "control":
                    controls[payload] = node_id
                elif kind == "curtailment":
                    curtailments[payload] = node_id
                conditional = float(weights[list(group)].sum() / parent_mass)
                nodes.append(InformationNode(node_id, history.node, event_at.to_pydatetime(), conditional,
                    observations, {key: event_at.to_pydatetime() for key in observations}))
                following.append(_MemberHistory(node_id, group, bids, controls, curtailments))
        active = following
    terminal_at = max([pd.Timestamp(grid.ends[-1]), *[event[0] for event in events]])
    leaves: list[str | None] = [None] * members
    bids_by_member: list[dict[str, str] | None] = [None] * members
    controls_by_member: list[tuple[str, ...] | None] = [None] * members
    curtailments_by_member: list[tuple[str, ...] | None] = [None] * members
    for history in active:
        mass = weights[list(history.members)].sum()
        for member in history.members:
            leaf = f"oracle_leaf_{member}"
            nodes.append(InformationNode(leaf, history.node, terminal_at.to_pydatetime(), float(weights[member] / mass)))
            leaves[member] = leaf
            bids_by_member[member] = history.bids
            controls_by_member[member] = tuple(history.controls[index] for index in range(intervals))
            curtailments_by_member[member] = tuple(history.curtailments[index] for index in range(intervals))
    open_auctions = tuple(event.auction for event in schedule if event.submission_at >= cutoff)
    tree = InformationTree(tuple(nodes), tuple(leaves),
        {auction: tuple(value[auction] for value in bids_by_member) for auction in open_auctions},
        tuple(controls_by_member), tuple(curtailments_by_member),
        execution_convention="pre_interval_battery; end_interval_curtailment; completed_PV_observations")
    tree.paths()
    bundle = ScenarioBundle(tuple(f"oracle_{name}" for name in identifiers), weights.copy(), pv.copy(),
        prices, imbalance, negative_imbalance, origin="oracle / perfect-foresight clearing prices",
        provenance={"executable": False, "oracle_prices": True, "operational_eligible": False,
            "information_relaxation": "future actual clearing prices used as deterministic objective coefficients; benchmark only",
            "price_sampling": "none; exactly one realized price curve per supplied auction, identical across PV members",
            "pv_members": members, "pv_member_ids": list(identifiers), "pv_member_probabilities": weights.tolist(),
            "pv_member_revelation": "identities remain latent; only completed-interval PV values can distinguish histories",
            "pv_observation_times": [value.isoformat() for value in observed_at],
            "pv_observation_policy": "explicit study observation clocks, default interval end; not API forecast publication timestamps",
            "execution_convention": tree.execution_convention,
            "actual_price_provenance": actual_price_provenance, "actual_price_alignment": alignment,
            "auction_eligible_product_indices": {auction: list(indices) for auction, indices in eligible_by_auction.items()},
            "untradable_price_placeholders": {auction: {
                "value_eur_mwh": 0.0,
                "product_indices": [index for index in range(intervals) if index not in indices],
                "policy": "rectangular-array placeholders only; no observed labels exist here; auction trading mask must exclude these products"}
                for auction, indices in eligible_by_auction.items() if len(indices) != intervals},
            "imbalance_price_provenance": imbalance_price_provenance,
            "benchmark_scope": "price-information upper bound for feasible policies evaluated on these same realized prices, PV members, weights and constraints",
            "decision_cutoff": cutoff.isoformat()})
    return bundle, tree
