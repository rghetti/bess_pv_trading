"""Normalize public Energy Charts day-ahead prices without changing provenance."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd


def parse_energy_charts_prices(payload, *, zone, start, end_exclusive, timezone="Europe/Rome"):
    """Parse quarter-hourly prices, preserving gaps and unknown publication times."""
    data = json.loads(payload)
    if data.get("unit") != "EUR / MWh":
        raise ValueError("Unexpected Energy Charts price unit")
    timestamps = data.get("unix_seconds")
    prices = data.get("price")
    if not isinstance(timestamps, list) or not isinstance(prices, list) or len(timestamps) != len(prices):
        raise ValueError("Price timestamps and values must be aligned lists")
    starts = pd.DatetimeIndex(pd.to_datetime(timestamps, unit="s", utc=True)).as_unit("ns")
    if starts.has_duplicates or not starts.is_monotonic_increasing:
        raise ValueError("Price timestamps must be unique and ordered")
    if len(starts) and np.any(starts.asi8 % pd.Timedelta(minutes=15).value):
        raise ValueError("Energy Charts products must be aligned to quarter hours")
    first = pd.Timestamp(start, tz=timezone).tz_convert("UTC")
    stop = pd.Timestamp(end_exclusive, tz=timezone).tz_convert("UTC")
    values = pd.to_numeric(pd.Series(prices), errors="coerce").to_numpy(dtype=float)
    # Preserve missing values as absent intervals; never manufacture target prices.
    mask = (starts >= first) & (starts < stop) & np.isfinite(values)
    chosen = starts[mask]
    frame = pd.DataFrame({"product_start": chosen, "product_end": chosen + pd.Timedelta(minutes=15),
        "value": values[mask], "provider": "energy_charts", "dataset": "day_ahead_prices",
        "zone": zone, "unit": "EUR/MWh", "resolution": "PT15M",
        "published_at": pd.Series(pd.NaT, index=range(len(chosen)), dtype="datetime64[ns, UTC]"),
        "publication_provenance": "unknown; public API retrieval is not historical publication evidence"})
    if frame.empty:
        raise ValueError("Energy Charts returned no finite prices in the requested interval")
    return frame, {"reported_license": data.get("license_info"), "null_prices": int((~np.isfinite(values)).sum()),
                   "source_rows": len(timestamps), "accepted_rows": len(frame)}
