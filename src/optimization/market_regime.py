"""Separate native auction-product resolution from physical control resolution."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from zoneinfo import ZoneInfo

from ..domain import DeliveryGrid


MGP_QUARTER_HOUR_START = date(2025, 10, 1)
MARKET_TIMEZONE = ZoneInfo("Europe/Rome")


def historical_quarter_hour_indices(grid: DeliveryGrid) -> frozenset[int]:
    """Return physical quarter-hours represented by pre-transition MGP hours."""
    return frozenset(index for group in _historical_hour_groups(grid) for index in group)


def market_product_groups(grid: DeliveryGrid, interval_minutes: int,
                          timezone: str = "Europe/Rome") -> tuple[tuple[int, ...], ...]:
    """Group physical intervals by configured local-market product resolution."""
    # Enforce the input contract: Market product resolution must be 15, 30, or 60 minutes.
    if interval_minutes not in (15, 30, 60):
        raise ValueError("Market product resolution must be 15, 30, or 60 minutes")
    # Enforce the input contract: Market product resolution must divide one local hour.
    if 60 % interval_minutes:
        raise ValueError("Market product resolution must divide one local hour")
    zone = ZoneInfo(timezone)
    groups: dict[tuple[date, int, int, int], list[int]] = defaultdict(list)
    # Group physical intervals by configured local-market product resolution for each `(index, start)` in `enumerate(grid.starts)`.
    for index, start in enumerate(grid.starts):
        local = start.astimezone(zone)
        slot = local.minute // interval_minutes
        groups[(local.date(), local.hour, slot, local.fold)].append(index)
    # The input grid is chronological. Sorting local clocks would interleave the
    # two autumn folds for quarter-hour and half-hour products.
    return tuple(tuple(indices) for indices in groups.values())


def configured_dispatch_groups(grid: DeliveryGrid, resolution_minutes: int | None,
                               timezone: str = "Europe/Rome") -> tuple[tuple[int, ...], ...]:
    """Physical controls are free each grid interval unless explicitly grouped.

    Hourly MGP products constrain DA orders only. Quarter-hour IDA orders and
    battery controls remain independent before the October 2025 DA transition.
    """
    # Return `()` when `resolution_minutes is None`.
    if resolution_minutes is None:
        return ()
    return tuple(group for group in market_product_groups(grid, resolution_minutes, timezone)
                 if len(group) > 1)


def configured_auction_product_groups(grid: DeliveryGrid, auction: str,
                                     resolution_minutes: int | None,
                                     timezone: str = "Europe/Rome") -> tuple[tuple[int, ...], ...]:
    """Resolve native auction products, preserving the legacy pre-QH DA rule."""
    # Return `market_product_groups(grid, resolution_minutes, timezone)` when `resolution_minutes is not None`.
    if resolution_minutes is not None:
        return market_product_groups(grid, resolution_minutes, timezone)
    # Return `()` when `auction != 'DA'`.
    if auction != "DA":
        return ()
    historical = historical_quarter_hour_indices(grid)
    return tuple(group for group in market_product_groups(grid, 60, timezone)
                 if any(index in historical for index in group))


def _historical_hour_groups(grid: DeliveryGrid) -> tuple[tuple[int, ...], ...]:
    """Return all historical local-hour groups, including one-quarter residuals."""
    groups: dict[tuple[date, int, int], list[int]] = defaultdict(list)
    quarter = timedelta(minutes=15)
    # Return all historical local-hour groups, including one-quarter residuals for each `(index, (start, end))` in `enumerate(zip(grid.starts, grid.ends))`.
    for index, (start, end) in enumerate(zip(grid.starts, grid.ends)):
        local = start.astimezone(MARKET_TIMEZONE)
        # Skip the current item when `local.date() >= MGP_QUARTER_HOUR_START or end - start != quarter`.
        if local.date() >= MGP_QUARTER_HOUR_START or end - start != quarter:
            continue
        # Skip the current item when `local.minute not in (0, 15, 30, 45) or local.second or local.microsecond`.
        if local.minute not in (0, 15, 30, 45) or local.second or local.microsecond:
            continue
        groups[(local.date(), local.hour, local.fold)].append(index)
    return tuple(tuple(indices) for _, indices in sorted(groups.items()))
