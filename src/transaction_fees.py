"""Gross market-transaction fees, dated tariffs and causal operator-volume state."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import date, datetime, timedelta
from math import isfinite
from typing import Any, Mapping
from zoneinfo import ZoneInfo


def _clock(value) -> datetime:
    """Accept explicit offset-aware clocks without assuming a local timezone."""
    result = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    # Enforce the input contract: Transaction-fee clocks must be timezone-aware.
    if result.tzinfo is None or result.utcoffset() is None:
        raise ValueError("Transaction-fee clocks must be timezone-aware")
    return result


def _nonnegative(value, label) -> float:
    """Reject invalid financial quantities instead of silently clipping them."""
    result = float(value)
    # Enforce the input contract: `label` must be finite and nonnegative.
    if isinstance(value, bool) or not isfinite(result) or result < 0:
        raise ValueError(f"{label} must be finite and nonnegative")
    return result


@dataclass(frozen=True)
class TransactionFeeState:
    """Known operator gross volume, including pending guaranteed transactions.

    Volumes include any external portfolio activity known at as_of. They exclude
    tentative future trades. Cumulative fees are sunk costs, never reoptimized.
    """
    as_of: datetime | None = None
    monthly_volume_mwh: Mapping[str, float] = field(default_factory=dict)
    cumulative_fees_eur: float = 0.0
    provenance: str = ""

    def __post_init__(self):
        # Normalize the fee-state clock when an accounting timestamp is present.
        if self.as_of is not None:
            object.__setattr__(self, "as_of", _clock(self.as_of))
        # Validate each carried monthly traded volume before applying a fee tier.
        for month, volume in self.monthly_volume_mwh.items():
            # Enforce the input contract: Monthly transaction-volume keys must be YYYY-MM.
            if date.fromisoformat(f"{month}-01").strftime("%Y-%m") != month:
                raise ValueError("Monthly transaction-volume keys must be YYYY-MM")
            _nonnegative(volume, "Monthly gross volume")
        object.__setattr__(self, "monthly_volume_mwh", {
            month: float(volume) for month, volume in self.monthly_volume_mwh.items()})
        object.__setattr__(self, "cumulative_fees_eur", _nonnegative(self.cumulative_fees_eur, "Previously incurred fees"))


@dataclass(frozen=True)
class TransactionFeeTariff:
    """Progressive marginal bands with explicitly bounded applicability."""
    version: str
    valid_from: date
    valid_to: date
    # Thresholds partition cumulative monthly volume; each rate applies to one marginal band.
    thresholds_mwh: tuple[float, ...]
    marginal_eur_per_mwh: tuple[float, ...]
    applicability: str
    source: str

    def __post_init__(self):
        # Enforce the input contract: Fee tariffs require a version, source and complete calendar-month validity.
        if (not self.version or not self.source or self.valid_from > self.valid_to
                or self.valid_from.day != 1 or (self.valid_to + timedelta(days=1)).day != 1):
            raise ValueError("Fee tariffs require a version, source and complete calendar-month validity")
        # Enforce the input contract: Tariff applicability must be verified or assumed_unchanged.
        if self.applicability not in {"verified", "assumed_unchanged"}:
            raise ValueError("Tariff applicability must be verified or assumed_unchanged")
        # Enforce the input contract: A marginal fee is required for every volume band.
        if len(self.marginal_eur_per_mwh) != len(self.thresholds_mwh) + 1:
            raise ValueError("A marginal fee is required for every volume band")
        # Enforce the input contract: Fee thresholds must be finite, positive and strictly increasing.
        if any(not isfinite(x) or x <= 0 for x in self.thresholds_mwh) or any(
                right <= left for left, right in zip(self.thresholds_mwh, self.thresholds_mwh[1:])):
            raise ValueError("Fee thresholds must be finite, positive and strictly increasing")
        # Reject invalid marginal rates in the configured fee schedule.
        for rate in self.marginal_eur_per_mwh:
            _nonnegative(rate, "Marginal transaction fee")

    def increment(self, starting_volume: float, new_volume: float) -> float:
        """Integrate marginal rates over new MWh; no retroactive repricing."""
        start = _nonnegative(starting_volume, "Starting monthly volume")
        volume = _nonnegative(new_volume, "New gross volume")
        stop, lower, cost = start + volume, 0.0, 0.0
        # Integrate marginal rates over new MWh; no retroactive repricing for each `(upper, rate)` in `zip((*self.thresholds_mwh, float('inf')), self.marginal_eur_per_mwh)`.
        for upper, rate in zip((*self.thresholds_mwh, float("inf")), self.marginal_eur_per_mwh):
            cost += max(0.0, min(stop, upper) - max(start, lower)) * rate
            lower = upper
        return cost


@dataclass(frozen=True)
class TransactionFeeSettings:
    """One operator's tariff and explicit starting-volume assumption."""
    mode: str = "none"
    flat_eur_per_mwh: float = 0.0
    timezone: str = "Europe/Rome"
    operator_scope: str = "standalone_plant"
    tariffs: tuple[TransactionFeeTariff, ...] = ()
    initial_state: TransactionFeeState = field(default_factory=TransactionFeeState)

    def __post_init__(self):
        # Enforce the input contract: transaction_fees.mode must be none, flat or gme_real.
        if self.mode not in {"none", "flat", "gme_real"}:
            raise ValueError("transaction_fees.mode must be none, flat or gme_real")
        _nonnegative(self.flat_eur_per_mwh, "Flat transaction fee")
        ZoneInfo(self.timezone)
        # Enforce the input contract: Unknown transaction-fee operator scope.
        if self.operator_scope not in {"standalone_plant", "operator_with_exogenous_volume"}:
            raise ValueError("Unknown transaction-fee operator scope")
        # Enforce the input contract: gme_real requires dated tariffs and an explicit timestamped operator-volume state with provenance.
        if self.mode == "gme_real" and (not self.tariffs or self.initial_state.as_of is None
                or not self.initial_state.provenance):
            raise ValueError("gme_real requires dated tariffs and an explicit timestamped operator-volume state with provenance")
        ordered = sorted(self.tariffs, key=lambda item: item.valid_from)
        # Enforce the input contract: Transaction-fee tariff validity windows overlap.
        if any(left.valid_to >= right.valid_from for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Transaction-fee tariff validity windows overlap")

    def month(self, at) -> str:
        """Use the transaction clock in market time, independently of delivery."""
        return _clock(at).astimezone(ZoneInfo(self.timezone)).strftime("%Y-%m")

    def starting_volume(self, state: TransactionFeeState, month: str) -> float:
        """Reset new months to zero; require known earlier/current GME volume."""
        # Return `float(state.monthly_volume_mwh[month])` when `month in state.monthly_volume_mwh`.
        if month in state.monthly_volume_mwh:
            return float(state.monthly_volume_mwh[month])
        # Return the selected result for this condition.
        if self.mode != "gme_real" or (state.as_of is not None and month > self.month(state.as_of)):
            return 0.0
        raise ValueError(f"Missing known starting operator volume for transaction month {month}")

    def tariff(self, month: str) -> TransactionFeeTariff:
        """Reject unsupported historical dates instead of using today's tariff."""
        day = date.fromisoformat(f"{month}-01")
        matches = [item for item in self.tariffs if item.valid_from <= day <= item.valid_to]
        # Enforce the input contract: No applicable transaction-fee tariff for `month`.
        if len(matches) != 1:
            raise ValueError(f"No applicable transaction-fee tariff for {month}")
        return matches[0]

    def increment(self, month: str, starting_volume: float, gross_volume: float) -> float:
        """Return EUR charged only on the new gross market volume in MWh."""
        volume = _nonnegative(gross_volume, "Gross transaction volume")
        # Return `0.0` when `self.mode == 'none'`.
        if self.mode == "none":
            return 0.0
        # Return `self.flat_eur_per_mwh * volume` when `self.mode == 'flat'`.
        if self.mode == "flat":
            return self.flat_eur_per_mwh * volume
        return self.tariff(month).increment(starting_volume, volume)


def transaction_fees_from_config(config: Mapping[str, Any]) -> TransactionFeeSettings:
    """Read the existing YAML mapping; an absent section preserves fee-free runs."""
    section = dict(config.get("transaction_fees", {}))
    unknown = set(section) - {"mode", "flat_eur_per_mwh", "timezone", "operator_scope", "tariffs", "initial_state"}
    # Enforce the input contract: Unknown transaction-fee settings: `sorted(unknown)`.
    if unknown:
        raise ValueError(f"Unknown transaction-fee settings: {sorted(unknown)}")
    # Enforce the input contract: Flat transaction fees require flat_eur_per_mwh.
    if section.get("mode") == "flat" and "flat_eur_per_mwh" not in section:
        raise ValueError("Flat transaction fees require flat_eur_per_mwh")
    # Enforce the input contract: gme_real requires explicit operator_scope.
    if section.get("mode") == "gme_real" and "operator_scope" not in section:
        raise ValueError("gme_real requires explicit operator_scope")
    tariffs = []
    # Read the existing YAML mapping; an absent section preserves fee-free runs for each `item` in `section.pop('tariffs', ())`.
    for item in section.pop("tariffs", ()):
        item = dict(item)
        # Read the existing YAML mapping; an absent section preserves fee-free runs for each `name` in `('valid_from', 'valid_to')`.
        for name in ("valid_from", "valid_to"):
            item[name] = date.fromisoformat(str(item[name]))
        # Read the existing YAML mapping; an absent section preserves fee-free runs for each `name` in `('thresholds_mwh', 'marginal_eur_per_mwh')`.
        for name in ("thresholds_mwh", "marginal_eur_per_mwh"):
            item[name] = tuple(float(value) for value in item[name])
        tariffs.append(TransactionFeeTariff(**item))
    initial = TransactionFeeState(**section.pop("initial_state", {}))
    return TransactionFeeSettings(**section, tariffs=tuple(tariffs), initial_state=initial)


def validate_fee_context(settings, state, auction_times, cutoff, accepted_trades=()):
    """Check that known monthly volume cannot reveal future executed activity."""
    # Check that known monthly volume cannot reveal future executed activity when `settings.mode == 'none'`.
    if settings.mode == "none":
        return
    cutoff = _clock(cutoff)
    # Enforce the input contract: Transaction-fee volume state is unavailable at the decision cutoff.
    if state.as_of is not None and state.as_of > cutoff:
        raise ValueError("Transaction-fee volume state is unavailable at the decision cutoff")
    # Enforce the input contract: Future executed monthly volume cannot enter a known transaction-fee state.
    if state.as_of is not None and any(month > settings.month(state.as_of) and volume > 0
            for month, volume in state.monthly_volume_mwh.items()):
        raise ValueError("Future executed monthly volume cannot enter a known transaction-fee state")
    # Check that known monthly volume cannot reveal future executed activity for each `at` in `auction_times.values()`.
    for at in auction_times.values():
        # Enforce the input contract: Open transaction-fee decisions cannot precede the decision cutoff.
        if _clock(at) < cutoff:
            raise ValueError("Open transaction-fee decisions cannot precede the decision cutoff")
        month = settings.month(at)
        settings.starting_volume(state, month)
        # Check that known monthly volume cannot reveal future executed activity when `settings.mode == 'gme_real'`.
        if settings.mode == "gme_real":
            settings.tariff(month)
    known = {}
    # Check that known monthly volume cannot reveal future executed activity for each `trade` in `accepted_trades`.
    for trade in accepted_trades:
        # Enforce the input contract: Priced transaction-fee ledger entries require already published results.
        if trade.publication_time > cutoff:
            raise ValueError("Priced transaction-fee ledger entries require already published results")
        # Enforce the input contract: Operator fee state predates an already executed ledger transaction.
        if settings.mode == "gme_real" and trade.submission_time > state.as_of:
            raise ValueError("Operator fee state predates an already executed ledger transaction")
        month = settings.month(trade.submission_time)
        known[month] = known.get(month, 0.0) + abs(trade.quantity_mwh)
    # Enforce the input contract: Operator monthly fee state must already include all previously executed ledger volume.
    if settings.mode == "gme_real" and any(
            settings.starting_volume(state, month) + 1e-8 < volume for month, volume in known.items()):
        raise ValueError("Operator monthly fee state must already include all previously executed ledger volume")


def account_trade_fees(settings: TransactionFeeSettings, initial_state: TransactionFeeState,
                       trades: list[dict]) -> tuple[TransactionFeeState, list[dict], dict]:
    """Price executed transactions chronologically, separately from publication.

    This pure-Python integration is also the independent reference for MILP
    costs. Quarter-hour shares are already MWh: no duration factor is applied.
    Persist final_state and use it as the next run's initial_state exactly once.
    """
    state, records, seen = initial_state, [], set()
    # Process each `row` in this collection.
    for row in sorted(trades, key=lambda item: (_clock(item["submission_time"]), item["trade_id"])):
        at = _clock(row["submission_time"])
        # Enforce the input contract: Cannot charge transactions before the known fee-state cutoff.
        if state.as_of is not None and at < state.as_of:
            raise ValueError("Cannot charge transactions before the known fee-state cutoff")
        # Enforce the input contract: Duplicate transaction fee trade ID.
        if row["trade_id"] in seen:
            raise ValueError("Duplicate transaction fee trade ID")
        seen.add(row["trade_id"])
        quantity = float(row["quantity_mwh"])
        # Enforce the input contract: Executed transaction quantities must be finite.
        if not isfinite(quantity):
            raise ValueError("Executed transaction quantities must be finite")
        month, volume = settings.month(at), abs(quantity)
        before = settings.starting_volume(state, month)
        fee = settings.increment(month, before, volume)
        state = TransactionFeeState(at, {**state.monthly_volume_mwh, month: before + volume},
            state.cumulative_fees_eur + fee, state.provenance)
        records.append({**row, "transaction_month": month, "gross_traded_mwh": volume,
            "purchased_mwh": max(-quantity, 0.0), "sold_mwh": max(quantity, 0.0),
            "transaction_fee_eur": fee, "monthly_volume_before_mwh": before,
            "monthly_volume_after_mwh": before + volume,
            "tariff_version": settings.tariff(month).version if settings.mode == "gme_real" else settings.mode})
    by_auction, by_month = {}, {}
    # Price executed transactions chronologically, separately from publication for each `row` in `records`.
    for row in records:
        # Price executed transactions chronologically, separately from publication for each `(key, groups)` in `((row['auction'], by_auction), (row['transaction_month'], by_month))`.
        for key, groups in ((row["auction"], by_auction), (row["transaction_month"], by_month)):
            group = groups.setdefault(key, {name: 0.0 for name in
                ("gross_traded_mwh", "purchased_mwh", "sold_mwh", "transaction_fee_eur")})
            # Price executed transactions chronologically, separately from publication for each `name` in `group`.
            for name in group:
                group[name] += row[name]
    # Price executed transactions chronologically, separately from publication for each `(month, group)` in `by_month.items()`.
    for month, group in by_month.items():
        group.update(starting_operator_volume_mwh=settings.starting_volume(initial_state, month),
            ending_operator_volume_mwh=state.monthly_volume_mwh[month],
            tariff_version=settings.tariff(month).version if settings.mode == "gme_real" else settings.mode,
            tariff_applicability=settings.tariff(month).applicability if settings.mode == "gme_real" else "configured")
    summary = {"mode": settings.mode, "operator_scope": settings.operator_scope,
        "configuration": asdict(settings), "initial_state": asdict(initial_state), "final_state": asdict(state),
        "by_auction": by_auction, "by_month": by_month,
        **{name: sum(row[name] for row in records) for name in
            ("gross_traded_mwh", "purchased_mwh", "sold_mwh", "transaction_fee_eur")},
        "scope": "Exact incremental fees conditional on known starting operator volume; unknown external/future activity is not modeled"}
    return state, records, summary


def scenario_fee_details(inputs, trades) -> list[dict]:
    """Reconstruct each leaf's fees from actual signed decisions, without Gurobi."""
    settings = inputs.transaction_fees
    initial = inputs.market.transaction_fee_state or settings.initial_state
    details = []
    # Reconstruct each leaf's fees from actual signed decisions, without Gurobi for each `scenario` in `range(len(inputs.scenarios.scenario_ids))`.
    for scenario in range(len(inputs.scenarios.scenario_ids)):
        rows = []
        # Reconstruct each leaf's fees from actual signed decisions, without Gurobi for each `(auction, mask)` in `inputs.auction_eligibility.items()`.
        for auction, mask in inputs.auction_eligibility.items():
            at = inputs.auction_submission_times.get(auction)
            # When `at is None`, set `at` using `inputs.grid.starts[0]`.
            if at is None:  # Fee-free legacy hand-built models have no calendar.
                at = inputs.grid.starts[0]
            owner = inputs.tree.bid_nodes[auction][scenario]
            rows.extend({"auction": auction, "submission_time": at, "interval_id": product,
                "quantity_mwh": trades[auction, owner, product], "trade_id": f"{auction}:{product}"}
                for product, eligible in zip(inputs.grid.interval_ids, mask) if eligible)
        _, _, summary = account_trade_fees(settings, initial, rows)
        details.append(summary)
    return details
