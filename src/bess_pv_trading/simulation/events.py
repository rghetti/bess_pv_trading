"""Deterministic event clock with separate publication and decision events."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
import heapq
from typing import Any, Callable, Iterable

import pandas as pd


class EventKind(IntEnum):
    """Tie order: information available at a cutoff may be used at that cutoff."""

    FORECAST_PUBLICATION = 0
    AUCTION_RESULTS = 1
    OBSERVATION_AVAILABLE = 2
    AUCTION_SUBMISSION = 3
    PHYSICAL_CONTROL = 4
    SETTLEMENT = 5


@dataclass(frozen=True, order=True)
class Event:
    """UTC event with a stable sequence for equal timestamps and priorities."""

    at: pd.Timestamp
    kind: EventKind
    sequence: int = 0
    payload: Any = field(default=None, compare=False)

    def __post_init__(self) -> None:
        instant = pd.Timestamp(self.at)
        if instant.tzinfo is None:
            raise ValueError("Event timestamps must be timezone-aware")
        object.__setattr__(self, "at", instant.tz_convert("UTC"))


class EventEngine:
    """Execute handlers in causal order and reject events inserted in the past."""

    def __init__(self, events: Iterable[Event]) -> None:
        self._events = list(events)
        heapq.heapify(self._events)
        self.current: Event | None = None
        self.trace: list[dict[str, Any]] = []

    def schedule(self, event: Event) -> None:
        """Add an event that cannot precede the current information state."""
        if self.current is not None and event < self.current:
            raise ValueError("Cannot schedule an event before the current event")
        heapq.heappush(self._events, event)

    def run(self, handlers: dict[EventKind, Callable[[Event], None]]) -> None:
        """Process all scheduled events; handler exceptions stop the replay."""
        while self._events:
            self.current = heapq.heappop(self._events)
            handler = handlers.get(self.current.kind)
            if handler is None:
                raise ValueError(f"Missing handler for {self.current.kind.name}")
            handler(self.current)
            self.trace.append({"at": self.current.at.isoformat(),
                               "kind": self.current.kind.name,
                               "sequence": self.current.sequence})
