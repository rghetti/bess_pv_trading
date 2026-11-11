"""Acquire weekly Swiss reservoir energy levels published by the federal energy office."""

from __future__ import annotations

from io import BytesIO
import hashlib

import numpy as np
import pandas as pd

from .api import ApiRequest, SecureTransport
from .duckdb_store import DuckDBCache


def parse_reservoir_levels(raw, *, timezone, publication_delay_days):
    """Keep weekly observations and label the configured publication assumption."""
    if isinstance(publication_delay_days, bool) or not isinstance(publication_delay_days, int) or publication_delay_days < 0:
        raise ValueError("Reservoir publication delay must be a nonnegative calendar-day count")
    frame = pd.read_csv(BytesIO(raw))
    fields = {"valais_gwh": "Wallis_speicherinhalt_gwh", "graubunden_gwh": "Graubuenden_speicherinhalt_gwh",
              "ticino_gwh": "Tessin_speicherinhalt_gwh", "other_ch_gwh": "UebrigCH_speicherinhalt_gwh",
              "total_ch_gwh": "TotalCH_speicherinhalt_gwh"}
    required = {"Datum", "TotalCH_max_speicherinhalt_gwh", *fields.values()}
    if not required.issubset(frame.columns):
        raise ValueError(f"Reservoir export lacks fields: {sorted(required - set(frame.columns))}")
    dates = pd.DatetimeIndex(pd.to_datetime(frame.Datum, format="%Y-%m-%d", errors="raise"))
    if dates.has_duplicates or not dates.is_monotonic_increasing or (dates.dayofweek != 0).any():
        raise ValueError("Reservoir reference dates must be unique increasing local Mondays")
    local = dates.tz_localize(timezone, ambiguous="raise", nonexistent="raise")
    starts = local.tz_convert("UTC")
    ends = (local + pd.DateOffset(days=7)).tz_convert("UTC")
    available = (local + pd.DateOffset(days=publication_delay_days)).tz_convert("UTC")
    values = {}
    for name, source in fields.items():
        values[name] = pd.to_numeric(frame[source], errors="raise").to_numpy(float)
    capacity = pd.to_numeric(frame["TotalCH_max_speicherinhalt_gwh"], errors="raise").to_numpy(float)
    if np.any(np.isfinite(capacity) & (capacity <= 0)):
        raise ValueError("Reservoir capacity must be positive when present")
    values["total_ch_percent"] = np.divide(values["total_ch_gwh"] * 100, capacity,
        out=np.full(len(frame), np.nan), where=np.isfinite(capacity) & (capacity > 0))
    output = []
    for name, series in values.items():
        output.append(pd.DataFrame({"provider": "bfe", "dataset": "reservoir_levels", "zone": "CH",
            "product_start": starts, "product_end": ends, "reference_time": starts,
            "scheduled_available_at": available, "historical_available_at": pd.NaT,
            "availability_provenance": "assumed_local_publication_delay_not_verified_historical_release",
            "variable": name, "value": series, "unit": "%" if name.endswith("_percent") else "GWh",
            "source_row": np.arange(len(frame))}))
    return pd.concat(output, ignore_index=True)


def download_reservoir_levels(config, *, transport=None, refresh=False):
    """Cache the complete public weekly CSV without changing prior snapshots."""
    settings = config["swiss_reservoir"]
    identity = {"provider": "bfe", "dataset": "reservoir_levels", "zone": "CH",
        "start": "2000-01-01", "end": "open", "options": {"parser": "bfe-reservoir-v1",
        "timezone": settings["timezone"], "publication_delay_days": settings["publication_delay_days"]}}
    store = DuckDBCache(settings["database_path"], threads=config["api"]["database_threads"])
    cached = None if refresh else store.load(identity)
    if cached is not None:
        return {"status": "cached", "rows": len(cached[0]), "snapshot": cached[1]}
    client = transport or SecureTransport(**config["api"]["http"])
    raw = client.get(ApiRequest(settings["url"]))
    # Unchanged weekly downloads should not create another immutable snapshot.
    if refresh:
        previous = store.load(identity)
        if previous is not None and previous[1].get("raw_sha256") == hashlib.sha256(raw).hexdigest():
            return {"status": "cached", "rows": len(previous[0]), "snapshot": previous[1]}
    frame = parse_reservoir_levels(raw, timezone=settings["timezone"],
                                   publication_delay_days=settings["publication_delay_days"])
    metadata = {"source_kind": "bfe_public_csv", "source_url": settings["url"],
        "reference": "local Monday 00:00, equivalent to Sunday 24:00", "unit_scope": "GWh or percent",
        "availability": "configured local calendar-day delay; historical publication time unverified",
        "publication_delay_days": settings["publication_delay_days"],
        "missing_values": {name: int(frame.loc[frame.variable == name, "value"].isna().sum())
                           for name in frame.variable.unique()}}
    with store.batch_write():
        manifest = store.save(identity, raw, frame, metadata)
        store.verify_native_snapshot(identity, snapshot_id=manifest["snapshot_id"])
    return {"status": "complete", "rows": len(frame), "snapshot": manifest}
