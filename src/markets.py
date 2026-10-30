"""Historically bounded Italian auction calendar with explicit publication events."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Any, Mapping
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

GME_TIMING_SOURCE = "https://www.mercatoelettrico.org/Portals/0/Documents/en-US/20250930DTF03rev13MPE_En.pdf"
GME_TIMING_SOURCE_REV14 = "https://gme.mercatoelettrico.org/Portals/0/Documents/en-US/20260113DTF03rev14MPE_en.pdf"
QH_SOURCE = "https://www.mercatoelettrico.org/en-us/Home/NoticesandAnnouncements/NoticesandAnnouncementsME?id=7101"
AUCTION_DISPATCH_SCOPE = "eligible_delivery_suffix"


@dataclass(frozen=True)
class MarketDeliveryGrid:
    """UTC physical intervals of one local delivery day, including DST days."""
    starts: pd.DatetimeIndex
    ends: pd.DatetimeIndex
    product_ids: tuple[str, ...]
    timezone: str = "Europe/Rome"

    @property
    def delta_hours(self) -> np.ndarray:
        return np.asarray((self.ends - self.starts).total_seconds() / 3600)

    def __len__(self) -> int:
        return len(self.starts)

    def to_domain(self):
        """Convert calendar products into the solver-independent shared contract."""
        from bess_pv_trading.domain import DeliveryGrid
        return DeliveryGrid(self.product_ids, tuple(self.starts.to_pydatetime()), tuple(self.ends.to_pydatetime()))


@dataclass(frozen=True)
class AuctionEvent:
    """Submission and results timestamps are distinct information events."""
    auction: str
    gate_at: pd.Timestamp
    submission_at: pd.Timestamp
    results_at: pd.Timestamp | None
    eligible_intervals: tuple[int, ...]
    result_provenance: str
    available: bool = True
    rule_version: str = "GME-DTF03-rev13"

    @property
    def name(self) -> str:
        return self.auction


@dataclass(frozen=True)
class AuctionCalendar:
    """Quarter-hour physical grid and auction events with bounded rule validity.

    Native DA and IDA product resolutions are applied separately by the optimizer.

    DA final results are *not earlier than* 13:05 in the consulted rule. A default
    13:30 is an explicit conservative study assumption, not a guaranteed deadline.
    Recorded publication timestamps override it; missing/cancelled results can be
    represented with a None override or a cancelled auction.
    """
    timezone: str = "Europe/Rome"
    valid_from: str = "2025-10-01"
    valid_to: str = "2026-08-31"
    decision_buffer_minutes: float = 5.0
    gates: Mapping[str, str] = field(default_factory=lambda: {"DA": "12:00", "IDA1": "15:00", "IDA2": "22:00", "IDA3": "10:00"})
    gate_day_offsets: Mapping[str, int] = field(default_factory=lambda: {"DA": -1, "IDA1": -1, "IDA2": -1, "IDA3": 0})
    result_times: Mapping[str, str] = field(default_factory=lambda: {"DA": "13:30", "IDA1": "15:30", "IDA2": "22:30", "IDA3": "10:30"})
    delivery_start_times: Mapping[str, str | None] = field(default_factory=lambda: {"DA": None, "IDA1": None, "IDA2": None, "IDA3": "12:00"})
    result_provenance: Mapping[str, str] = field(default_factory=lambda: {
        "DA": "study_assumption_13:30; GME final results not earlier than 13:05; recorded timestamps preferred",
        "IDA1": "GME DTF03 rev13: results by 15:30 local",
        "IDA2": "GME DTF03 rev13: results by 22:30 local",
        "IDA3": "GME DTF03 rev13: results by 10:30 local"})

    def __post_init__(self) -> None:
        ZoneInfo(self.timezone)
        # Enforce the input contract: Calendar validity end precedes start.
        if pd.Timestamp(self.valid_to) < pd.Timestamp(self.valid_from):
            raise ValueError("Calendar validity end precedes start")
        # Enforce the input contract: Decision buffer must lie in [0, 60) minutes.
        if self.decision_buffer_minutes < 0 or self.decision_buffer_minutes >= 60:
            raise ValueError("Decision buffer must lie in [0, 60) minutes")
        # Require a complete and parseable timing rule for each auction.
        for auction in ("DA", "IDA1", "IDA2", "IDA3"):
            # Enforce the input contract: Missing explicit market rule/provenance for `auction`.
            if (auction not in self.gates or auction not in self.gate_day_offsets
                    or auction not in self.result_times or auction not in self.result_provenance
                    or auction not in self.delivery_start_times):
                raise ValueError(f"Missing explicit market rule/provenance for {auction}")
            time.fromisoformat(self.gates[auction])
            time.fromisoformat(self.result_times[auction])
            delivery_start = self.delivery_start_times[auction]
            # Validate the optional start time for auctions with restricted delivery products.
            if delivery_start is not None:
                time.fromisoformat(delivery_start)
            # Enforce the input contract: `auction` gate_day_offsets must be an integer number of local days.
            if isinstance(self.gate_day_offsets[auction], bool) or not isinstance(self.gate_day_offsets[auction], int):
                raise ValueError(f"{auction} gate_day_offsets must be an integer number of local days")

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "AuctionCalendar":
        settings = dict(config.get("market", config))
        allowed = ("timezone", "valid_from", "valid_to", "decision_buffer_minutes", "gates",
                   "gate_day_offsets", "result_times", "delivery_start_times", "result_provenance")
        return cls(**{key: settings[key] for key in allowed if key in settings})

    def _date(self, delivery_day: str | date) -> date:
        day = date.fromisoformat(delivery_day) if isinstance(delivery_day, str) else delivery_day
        # Enforce the input contract: Delivery day `day` outside configured auction-calendar validity `self.valid_from`..`self.valid_to`.
        if not date.fromisoformat(self.valid_from) <= day <= date.fromisoformat(self.valid_to):
            raise ValueError(f"Delivery day {day} outside configured auction-calendar validity {self.valid_from}..{self.valid_to}")
        return day

    def _local(self, day: date, clock: str) -> pd.Timestamp:
        return pd.Timestamp(datetime.combine(day, time.fromisoformat(clock)), tz=self.timezone).tz_convert("UTC")

    def delivery_grid(self, delivery_day: str | date) -> MarketDeliveryGrid:
        """Generate between local midnights, never by adding 24 elapsed hours."""
        day = self._date(delivery_day)
        first = self._local(day, "00:00")
        last = self._local(day + timedelta(days=1), "00:00")
        boundaries = pd.date_range(first, last, freq="15min")
        starts, ends = boundaries[:-1], boundaries[1:]
        return MarketDeliveryGrid(starts, ends, tuple(x.isoformat() for x in starts), self.timezone)

    def schedule(self, delivery_day: str | date, *, recorded_results: Mapping[str, Any] | None = None,
                 cancelled: tuple[str, ...] = ()) -> tuple[AuctionEvent, ...]:
        """Create rule-based events, allowing actual missing/delayed publications."""
        day = self._date(delivery_day)
        grid = self.delivery_grid(day)
        events = []
        # Create rule-based events, allowing actual missing/delayed publications for each `auction` in `('DA', 'IDA1', 'IDA2', 'IDA3')`.
        for auction in ("DA", "IDA1", "IDA2", "IDA3"):
            event_day = day + timedelta(days=self.gate_day_offsets[auction])
            gate = self._local(event_day, self.gates[auction])
            result = self._local(event_day, self.result_times[auction])
            version = "GME-DTF03-rev14" if event_day >= date(2026, 1, 13) else "GME-DTF03-rev13"
            provenance = self.result_provenance[auction].replace("rev13", "rev14" if "rev14" in version else "rev13")
            # When `recorded_results is not None and auction in recorded_results`, set `value` using `recorded_results[auction]`.
            if recorded_results is not None and auction in recorded_results:
                value = recorded_results[auction]
                result = None if value is None else pd.Timestamp(value)
                # Create rule-based events, allowing actual missing/delayed publications when `result is not None`.
                if result is not None:
                    # Enforce the input contract: Recorded market publication requires explicit timezone.
                    if result.tzinfo is None:
                        raise ValueError("Recorded market publication requires explicit timezone")
                    result = result.tz_convert("UTC")
                provenance = "recorded_publication" if result is not None else "missing_recorded_results"
            # When `auction in cancelled`, set `result` using `None`.
            if auction in cancelled:
                result = None
                provenance = "cancelled_auction"
            # Enforce the input contract: `auction` results cannot be known before its gate.
            if result is not None and result <= gate:
                raise ValueError(f"{auction} results cannot be known before its gate")
            eligible = tuple(range(len(grid)))
            delivery_start = self.delivery_start_times[auction]
            # Update `eligible` when this condition is met.
            if delivery_start is not None:
                eligible = tuple(np.flatnonzero(
                    grid.starts >= self._local(day, delivery_start)).tolist())
            events.append(AuctionEvent(auction, gate,
                gate - pd.Timedelta(minutes=self.decision_buffer_minutes), result, eligible,
                provenance, result is not None, version))
        return tuple(events)
