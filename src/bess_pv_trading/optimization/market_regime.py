"""Delivery-product constraints implied by the Italian MGP time regime."""
from __future__ import annotations

from collections import defaultdict
from datetime import date, timedelta
from zoneinfo import ZoneInfo

from ..domain import DeliveryGrid


MGP_QUARTER_HOUR_START = date(2025, 10, 1)
MARKET_TIMEZONE = ZoneInfo("Europe/Rome")


def historical_hourly_dispatch_groups(grid: DeliveryGrid) -> tuple[tuple[int, ...], ...]:
    """Group represented 15-minute controls by pre-transition local market hour.

    The returned groups use interval starts and preserve the autumn DST fold, so
    the two repeated civil hours remain separate hourly products. In a residual
    solve, a group can contain only the still-active quarters of its hour.
    """
    return tuple(group for group in _historical_hour_groups(grid) if len(group) > 1)


def historical_quarter_hour_indices(grid: DeliveryGrid) -> frozenset[int]:
    """Return physical quarter-hours represented by pre-transition MGP hours."""
    return frozenset(index for group in _historical_hour_groups(grid) for index in group)


def market_product_groups(grid: DeliveryGrid, interval_minutes: int,
                          timezone: str = "Europe/Rome") -> tuple[tuple[int, ...], ...]:
    """Group physical intervals by configured local-market product resolution."""
    if interval_minutes not in (15, 30, 60):
        raise ValueError("Market product resolution must be 15, 30, or 60 minutes")
    if 60 % interval_minutes:
        raise ValueError("Market product resolution must divide one local hour")
    zone = ZoneInfo(timezone)
    groups: dict[tuple[date, int, int, int], list[int]] = defaultdict(list)
    for index, start in enumerate(grid.starts):
        local = start.astimezone(zone)
        slot = local.minute // interval_minutes
        groups[(local.date(), local.hour, slot, local.fold)].append(index)
    return tuple(tuple(indices) for _, indices in sorted(groups.items()))


def configured_dispatch_groups(grid: DeliveryGrid, resolution_minutes: int | None,
                               timezone: str = "Europe/Rome") -> tuple[tuple[int, ...], ...]:
    """Use explicit study resolution, retaining the historical rule for old callers."""
    if resolution_minutes is None:
        return historical_hourly_dispatch_groups(grid)
    return tuple(group for group in market_product_groups(grid, resolution_minutes, timezone)
                 if len(group) > 1)


def configured_auction_product_groups(grid: DeliveryGrid, auction: str,
                                     resolution_minutes: int | None,
                                     timezone: str = "Europe/Rome") -> tuple[tuple[int, ...], ...]:
    """Resolve native auction products, preserving the legacy pre-QH DA rule."""
    if resolution_minutes is not None:
        return market_product_groups(grid, resolution_minutes, timezone)
    if auction != "DA":
        return ()
    historical = historical_quarter_hour_indices(grid)
    return tuple(group for group in market_product_groups(grid, 60, timezone)
                 if any(index in historical for index in group))


def _historical_hour_groups(grid: DeliveryGrid) -> tuple[tuple[int, ...], ...]:
    """Return all historical local-hour groups, including one-quarter residuals."""
    groups: dict[tuple[date, int, int], list[int]] = defaultdict(list)
    quarter = timedelta(minutes=15)
    for index, (start, end) in enumerate(zip(grid.starts, grid.ends)):
        local = start.astimezone(MARKET_TIMEZONE)
        if local.date() >= MGP_QUARTER_HOUR_START or end - start != quarter:
            continue
        if local.minute not in (0, 15, 30, 45) or local.second or local.microsecond:
            continue
        groups[(local.date(), local.hour, local.fold)].append(index)
    return tuple(tuple(indices) for _, indices in sorted(groups.items()))
