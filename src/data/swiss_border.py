"""Fetch and preserve Swissgrid's public cross-border market schedules and NTC."""

from __future__ import annotations

from html.parser import HTMLParser
from io import BytesIO
from urllib.parse import urljoin, urlsplit
import re

import numpy as np
import pandas as pd

from .api import ApiRequest, SecureTransport
from .duckdb_store import DuckDBCache


class _Links(HTMLParser):
    """Collect page links without executing provider HTML or scripts."""

    def __init__(self):
        super().__init__()
        self.hrefs = []

    def handle_starttag(self, tag, attrs):
        if tag.lower() == "a":
            value = dict(attrs).get("href")
            if value:
                self.hrefs.append(value)


def discover_swiss_csv(page, page_url, year):
    """Select the official annual file; reject other hosts and ambiguous links."""
    parser = _Links()
    parser.feed(page.decode("utf-8"))
    suffix = re.compile(rf"/Grenzfluesse-{int(year)}\.csv$", re.IGNORECASE)
    urls = set()
    for href in parser.hrefs:
        url = urljoin(page_url, href)
        parts = urlsplit(url)
        if parts.scheme == "https" and parts.hostname == "www.swissgrid.ch" and not parts.query and suffix.search(parts.path):
            urls.add(url)
    if len(urls) != 1:
        raise ValueError(f"Expected one official Swissgrid cross-border CSV for {year}; found {len(urls)}")
    return urls.pop()


def parse_swiss_border(raw, *, year, timezone, columns):
    """Keep native quarter-hours, signed source values, missingness and DST folds."""
    frame = pd.read_csv(BytesIO(raw), sep=";", encoding="utf-8-sig")
    if frame.columns[0] != "Date Time" or not columns or len(set(columns.values())) != len(columns):
        raise ValueError("Swissgrid timestamp or configured source columns are invalid")
    missing = set(columns.values()) - set(frame.columns)
    if missing:
        raise ValueError(f"Swissgrid export lacks configured columns: {sorted(missing)}")
    naive = pd.DatetimeIndex(pd.to_datetime(frame["Date Time"], format="%d.%m.%Y %H:%M", errors="raise"))
    # Swissgrid repeats the autumn hour in chronological order and omits the
    # spring gap. Infer both folds from that sequence rather than dropping one.
    starts = naive.tz_localize(timezone, ambiguous="infer", nonexistent="raise").tz_convert("UTC")
    if starts.has_duplicates or not starts.is_monotonic_increasing or (naive.year != int(year)).any():
        raise ValueError("Swissgrid timestamps are duplicated, out of order, or outside the configured year")
    rows = []
    for name, source_column in columns.items():
        raw_values = frame[source_column]
        values = pd.to_numeric(raw_values, errors="coerce")
        if (raw_values.notna() & values.isna()).any():
            raise ValueError(f"Swissgrid field {name} contains a nonnumeric value")
        unit = "EUR/MWh" if "[EUR/MWh]" in source_column else "MW" if "[MW]" in source_column else None
        if unit is None:
            raise ValueError(f"Swissgrid field {name} has an unknown unit")
        # Preserve missing observations instead of silently filling them.
        rows.append(pd.DataFrame({"provider": "swissgrid", "dataset": "border_context", "zone": "CH",
            "product_start": starts, "product_end": starts + pd.Timedelta(minutes=15),
            "variable": name, "value": values.to_numpy(dtype=float), "unit": unit,
            "source_field": source_column, "source_row": np.arange(len(frame)),
            "historical_available_at": pd.NaT,
            "availability_provenance": "weekly_publication_exact_historical_release_unknown"}))
    return pd.concat(rows, ignore_index=True)


def download_swiss_border(config, *, transport=None, refresh=False):
    """Fetch a provider-discovered annual CSV, then publish one verified snapshot."""
    settings = config["swiss_border"]
    year = int(settings["year"])
    identity = {"provider": "swissgrid", "dataset": "border_context", "zone": "CH",
        "start": f"{year}-01-01", "end": f"{year + 1}-01-01",
        "options": {"year": year, "parser": "swissgrid-border-v1", "columns": settings["columns"]}}
    store = DuckDBCache(settings["database_path"], threads=config["api"]["database_threads"])
    cached = None if refresh else store.load(identity)
    if cached is not None:
        return {"status": "cached", "rows": len(cached[0]), "snapshot": cached[1]}
    client = transport or SecureTransport(**config["api"]["http"])
    page = client.get(ApiRequest(settings["page_url"]))
    url = discover_swiss_csv(page, settings["page_url"], year)
    raw = client.get(ApiRequest(url))
    frame = parse_swiss_border(raw, year=year, timezone=settings["timezone"], columns=settings["columns"])
    metadata = {"source_kind": "swissgrid_public_csv", "source_url": url,
        "source_page": settings["page_url"], "native_interval_minutes": 15,
        "timezone": settings["timezone"], "historical_availability": "exact_weekly_release_unknown",
        "series_scope": "intraday NTC and commercial schedules, not measured physical flows",
        "sign_convention": "source values retained; schedules positive for Swiss net exports",
        "missing_values": {name: int(frame.loc[frame.variable == name, "value"].isna().sum())
                           for name in settings["columns"]}}
    with store.batch_write():
        manifest = store.save(identity, raw, frame, metadata)
        store.verify_native_snapshot(identity, snapshot_id=manifest["snapshot_id"])
    return {"status": "complete", "rows": len(frame), "snapshot": manifest}
