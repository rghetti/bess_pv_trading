"""Cancellation-aware auction features for the full-year price search."""

from __future__ import annotations

from datetime import timedelta
import json

import duckdb
import numpy as np
import pandas as pd

from .price_feature_data import _lookup, _profiles


def catalog_cancellations(database, start, end, timezone):
    """Collect explicit GME cancellation notices from currently selected snapshots."""
    with duckdb.connect(str(database), read_only=True) as connection:
        rows = connection.execute("""SELECT s.dataset, s.snapshot_id, s.raw_sha256, s.manifest_json
            FROM api_snapshots s JOIN api_current_requests r ON r.snapshot_id = s.snapshot_id
            WHERE s.provider = 'gme' AND s.zone = 'NORD'
              AND s.dataset IN ('day_ahead_prices', 'ida1_prices', 'ida2_prices', 'ida3_prices')
            ORDER BY s.dataset, s.snapshot_id""").fetchall()
    records = []
    for dataset, snapshot_id, raw_sha256, manifest_json in rows:
        metadata = json.loads(manifest_json).get("metadata", {})
        days = metadata.get("cancelled_delivery_days", metadata.get("cancelled_flow_dates", []))
        for day in days:
            date = pd.Timestamp(day).date().isoformat()
            if start <= date < end:
                records.append({"dataset": dataset, "delivery_day": date, "snapshot_id": snapshot_id,
                                "raw_sha256": raw_sha256, "evidence": "GME snapshot cancellation notice"})
    return sorted(records, key=lambda item: (item["dataset"], item["delivery_day"], item["snapshot_id"]))


def cancellation_days(records, auctions, frame, timezone):
    """Reject contradictory prices and retain only explicitly documented days."""
    result = {name: set() for name in auctions}
    datasets = {item["dataset"]: name for name, item in auctions.items()}
    for record in records:
        if record["dataset"] in datasets:
            result[datasets[record["dataset"]]].add(record["delivery_day"])
    local_days = pd.DatetimeIndex(pd.to_datetime(frame.product_start, utc=True)).tz_convert(timezone).strftime("%Y-%m-%d")
    for name, auction in auctions.items():
        conflicts = frame.dataset.eq(auction["dataset"]).to_numpy() & np.isin(local_days, list(result[name]))
        if conflicts.any():
            raise ValueError(f"Cancellation notice conflicts with observed {name} prices")
    return result


def build_cancellation_aware_base(original, histories, starts, settings, group, cancellations, max_fallback_days):
    """Preserve missing gaps while replacing explicitly cancelled auction features."""
    matrix, names, anchor, reference, cutoffs, fallback_count = original(
        histories[group["market"]], histories, starts, settings, group)
    auction = settings["auctions"][group["market"]]
    local = starts.tz_convert(settings["timezone"])
    target_days = np.asarray(local.strftime("%Y-%m-%d"))
    extra, extra_names = [], []
    # Replace a cancelled earlier-auction curve with the nearest earlier cleared curve.
    prior = None
    for market in auction["earlier_markets"]:
        cancelled = np.isin(target_days, list(cancellations[market]))
        values = matrix[:, names.index(f"cleared_{market}")]
        if cancelled.any() and prior is not None:
            for offset in [0, *settings["near_auction_offsets_products"]]:
                column = f"cleared_{market}" if offset == 0 else f"cleared_{market}_neighbor_{offset}"
                previous = f"cleared_{prior}" if offset == 0 else f"cleared_{prior}_neighbor_{offset}"
                matrix[cancelled, names.index(column)] = matrix[cancelled, names.index(previous)]
        extra.append(cancelled.astype(float))
        extra_names.append(f"earlier_{market}_cancelled")
        prior = market
        anchor = matrix[:, names.index(f"cleared_{market}")].copy()
    if auction["earlier_markets"]:
        reference = anchor.copy()
    # A cancelled historical auction falls back to its latest available same-slot price.
    for market in auction["past_other_markets"]:
        for lag in settings["past_other_market_lag_hours"]:
            wanted = starts - pd.Timedelta(hours=lag)
            wanted_days = wanted.tz_convert(settings["timezone"]).strftime("%Y-%m-%d")
            cancelled = np.isin(wanted_days, list(cancellations[market]))
            column = names.index(f"past_{market}_{lag}h")
            unresolved = cancelled & ~np.isfinite(matrix[:, column])
            for offset in range(1, max_fallback_days + 1):
                if not unresolved.any():
                    break
                replacement = _lookup(histories[market], wanted[unresolved] - pd.Timedelta(days=offset), cutoffs[unresolved])
                positions = np.flatnonzero(unresolved)
                valid = np.isfinite(replacement)
                matrix[positions[valid], column] = replacement[valid]
                unresolved[positions[valid]] = False
            extra.append(cancelled.astype(float))
            extra_names.append(f"past_{market}_{lag}h_cancelled")
    # Replace an entirely cancelled own-auction daily profile with a preceding complete profile.
    own_hours = list(range(auction["delivery_start_hour"], auction["delivery_end_hour"]))
    dates = np.asarray(local.date)
    for lag in settings["daily_curve_lag_days"]:
        wanted = np.array([(day - timedelta(days=lag)).isoformat() for day in dates])
        cancelled = np.isin(wanted, list(cancellations[group["market"]]))
        columns = [names.index(f"own_curve_{lag}d_hour_{hour}") for hour in own_hours]
        unresolved = cancelled & ~np.isfinite(matrix[:, columns]).all(axis=1)
        for offset in range(1, max_fallback_days + 1):
            if not unresolved.any():
                break
            positions = np.flatnonzero(unresolved)
            profiles, _ = _profiles(histories[group["market"]], starts[positions], cutoffs[positions],
                [lag + offset], group["native_minutes"], own_hours, settings["timezone"])
            valid = np.isfinite(profiles).all(axis=1)
            matrix[np.ix_(positions[valid], columns)] = profiles[valid]
            unresolved[positions[valid]] = False
        extra.append(cancelled.astype(float))
        extra_names.append(f"own_curve_{lag}d_cancelled")
    return np.column_stack([matrix, *extra]), names + extra_names, anchor, reference, cutoffs, fallback_count
