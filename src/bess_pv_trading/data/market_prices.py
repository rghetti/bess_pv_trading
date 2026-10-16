"""Strict manual market imports, revision-aware selection, and reproducible caches.

No farm-data correction policy is applied to prices. Verified provider downloads
live in data/api.py; this module handles explicitly mapped manual market files.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import getpass
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

AUCTION_ALIASES = {"MGP": "DA", "MI-A1": "IDA1", "MI-A2": "IDA2", "MI-A3": "IDA3"}
REQUIRED = ("provider", "dataset", "zone", "auction", "product_start", "product_end",
            "price_eur_mwh", "published_at", "publication_provenance", "revision")
IDENTITY = ("provider", "dataset", "zone", "auction", "product_start", "product_end", "revision")


class MarketDataError(ValueError):
    """A market input is unavailable, ambiguous, or inconsistent."""


def _aware_utc(values: pd.Series, name: str) -> pd.Series:
    """Require explicit offsets so ambiguous local DST labels cannot slip in."""
    parsed = []
    for value in values:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise MarketDataError(f"{name} requires an offset-aware ISO timestamp: {value!r}")
        parsed.append(stamp.tz_convert("UTC"))
    return pd.Series(pd.DatetimeIndex(parsed), index=values.index, dtype="datetime64[ns, UTC]")


def validate_market_records(records: pd.DataFrame) -> pd.DataFrame:
    """Normalize canonical records without filling missing values or clipping prices."""
    missing = set(REQUIRED) - set(records.columns)
    if missing:
        raise MarketDataError(f"Missing canonical market columns: {sorted(missing)}")
    result = records.loc[:, list(REQUIRED)].copy()
    for name in ("provider", "dataset", "zone", "auction", "publication_provenance", "revision"):
        if result[name].isna().any() or result[name].astype(str).str.strip().eq("").any():
            raise MarketDataError(f"Missing {name}; supply explicit provenance/revision")
        result[name] = result[name].astype(str).str.strip()
    result["auction"] = result["auction"].replace(AUCTION_ALIASES)
    if not result["auction"].isin(["DA", "IDA1", "IDA2", "IDA3", "IMBALANCE"]).all():
        raise MarketDataError("Unknown auction; continuous indices cannot replace IDA prices")
    for name in ("product_start", "product_end", "published_at"):
        result[name] = _aware_utc(result[name], name)
    result["price_eur_mwh"] = pd.to_numeric(result["price_eur_mwh"], errors="raise")
    if not np.isfinite(result["price_eur_mwh"]).all():
        raise MarketDataError("Missing/nonfinite market prices must not be filled")
    duration = (result.product_end - result.product_start).dt.total_seconds() / 60
    if not duration.isin([15, 30, 60]).all():
        raise MarketDataError("Market products must be explicit 15/30/60 minute intervals")
    if result.duplicated(list(IDENTITY)).any():
        raise MarketDataError("Duplicate market product/revision records")
    return result.sort_values(["product_start", "auction", "published_at", "revision"]).reset_index(drop=True)


def import_market_csv(path: str | Path, *, column_mapping: Mapping[str, str] | None = None,
                      constants: Mapping[str, Any] | None = None, separator: str = ",") -> pd.DataFrame:
    """Import a manually exported GME/provider file through an explicit column map.

    Mapping keys are canonical columns, values are source columns. Constants may
    declare provider, dataset, zone, revision or publication provenance, but never
    infer a historical publication time from today's retrieval timestamp.
    """
    source = Path(path)
    if not source.is_file():
        raise MarketDataError(f"Market file missing: {source}. Supply a manual GME MGP/MI-A CSV export.")
    frame = pd.read_csv(source, sep=separator, dtype={"revision": str})
    if column_mapping:
        frame = frame.rename(columns={raw: canonical for canonical, raw in column_mapping.items()})
    for name, value in (constants or {}).items():
        if name in ("price_eur_mwh", "product_start", "product_end"):
            raise MarketDataError(f"{name} must be supplied per actual product")
        frame[name] = value
    return validate_market_records(frame)


def import_market_xml(path: str | Path, *, record_tag: str, field_mapping: Mapping[str, str],
                      constants: Mapping[str, Any] | None = None) -> pd.DataFrame:
    """Read a manual XML export using user-verified tags (no invented GME schema)."""
    root = ET.parse(path).getroot()
    rows = []
    for record in root.iter(record_tag):
        rows.append({canonical: record.findtext(raw) for canonical, raw in field_mapping.items()})
    if not rows:
        raise MarketDataError(f"No XML records matched {record_tag!r}; verify the export schema")
    frame = pd.DataFrame(rows)
    for name, value in (constants or {}).items():
        if name in ("price_eur_mwh", "product_start", "product_end"):
            raise MarketDataError(f"{name} must be supplied per actual product")
        frame[name] = value
    return validate_market_records(frame)


def available_records(records: pd.DataFrame, as_of: Any) -> pd.DataFrame:
    """Select the most recently published revision actually available at a cutoff."""
    cutoff = pd.Timestamp(as_of)
    if cutoff.tzinfo is None:
        raise MarketDataError("as_of must be timezone-aware")
    eligible = records.loc[records.published_at <= cutoff.tz_convert("UTC")].copy()
    keys = list(IDENTITY[:-1])
    # Equal publication timestamps with differing revisions require a declared order.
    if eligible.duplicated(keys + ["published_at"]).any():
        raise MarketDataError("Ambiguous revision ordering at identical publication timestamps")
    return eligible.sort_values("published_at").drop_duplicates(keys, keep="last").reset_index(drop=True)


def market_completeness(records: pd.DataFrame, expected_starts: pd.DatetimeIndex,
                        auction: str, zone: str = "NORD") -> dict[str, Any]:
    """Report gaps against an externally constructed delivery grid without filling."""
    selected = records.loc[(records.auction == auction) & (records.zone == zone)]
    starts = pd.DatetimeIndex(selected.product_start)
    return {"auction": auction, "zone": zone, "expected": len(expected_starts),
            "records": len(selected), "missing": [x.isoformat() for x in expected_starts.difference(starts)],
            "unexpected": [x.isoformat() for x in starts.difference(expected_starts)]}


def market_coverage_report(records: pd.DataFrame, config: Mapping[str, Any]) -> dict[str, Any]:
    """Compare supplied prices with requested quarter-hour products, listing gaps.

    Native hourly/half-hourly files remain native records. They do not satisfy a
    requested observed quarter-hour product and are counted explicitly.
    """
    from ..markets import AuctionCalendar
    calendar = AuctionCalendar.from_config(config)
    settings = config.get("market_data", {})
    request = settings.get("cache_key", {})
    zone = config.get("market", {}).get("zone", request.get("zone", "NORD"))
    if records.empty:
        return {"records": 0, "missing_products": None, "status": "empty"}
    latest = available_records(records, records.published_at.max())
    latest = latest.loc[latest.zone == zone]
    first = pd.Timestamp(request.get("start", records.product_start.min().tz_convert(calendar.timezone).date())).date()
    end = pd.Timestamp(request.get("end", records.product_start.max().tz_convert(calendar.timezone).date() + pd.Timedelta(days=1))).date()
    first = max(first, pd.Timestamp(calendar.valid_from).date())
    end = min(end, pd.Timestamp(calendar.valid_to).date() + pd.Timedelta(days=1))
    expected_auctions = settings.get("requested_auctions", ["DA", "IDA1", "IDA2", "IDA3"])
    summaries = []
    missing_products = 0
    for day in pd.date_range(first, end, inclusive="left"):
        grid = calendar.delivery_grid(day.date())
        for event in calendar.schedule(day.date()):
            if event.auction not in expected_auctions:
                continue
            expected = grid.starts[list(event.eligible_intervals)]
            prices = latest.loc[(latest.auction == event.auction) & latest.product_start.isin(expected)]
            qh = prices.loc[(prices.product_end - prices.product_start) == pd.Timedelta(minutes=15)]
            missing = expected.difference(pd.DatetimeIndex(qh.product_start))
            missing_products += len(missing)
            summaries.append({"delivery_day": day.date().isoformat(), "auction": event.auction,
                "expected": len(expected), "observed_quarter_hour": len(qh), "missing_count": len(missing),
                "missing_starts": [value.isoformat() for value in missing],
                "native_other_duration_count": len(prices) - len(qh)})
    return {"records": len(records), "zone": zone, "schema": "market-coverage-v1", "missing_products": missing_products,
            "status": "complete" if not missing_products else "gaps", "products": summaries,
            "fill_policy": "none; native non-quarter-hour prices do not imply observed quarter-hour prices"}


@dataclass(frozen=True)
class MarketCacheKey:
    """The entire nonsecret request identity; vintage differs from retrieval time."""
    provider: str
    dataset: str
    zone: str
    product: str
    start: str
    end: str
    vintage_revision: str
    schema_version: str = "market-v1"

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()


class MarketCache:
    """Small CSV caches with a matching request/provenance manifest."""
    def __init__(self, directory: str | Path):
        self.directory = Path(directory)

    def load(self, key: MarketCacheKey) -> pd.DataFrame | None:
        stem = self.directory / key.digest()
        if not stem.with_suffix(".csv").exists() or not stem.with_suffix(".json").exists():
            return None
        metadata = json.loads(stem.with_suffix(".json").read_text())
        if metadata["request"] != asdict(key):
            raise MarketDataError("Cache request manifest does not match key")
        return import_market_csv(stem.with_suffix(".csv"))

    def save(self, key: MarketCacheKey, records: pd.DataFrame) -> Path:
        frame = validate_market_records(records)
        self.directory.mkdir(parents=True, exist_ok=True)
        stem = self.directory / key.digest()
        temporary = stem.with_suffix(".csv.tmp")
        frame.to_csv(temporary, index=False)
        temporary.replace(stem.with_suffix(".csv"))
        metadata = {"request": asdict(key), "retrieved_at": datetime.now(timezone.utc).isoformat(),
                    "records": len(frame), "historical_availability": "published_at; never retrieved_at"}
        stem.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
        return stem.with_suffix(".csv")


def provider_credential(environment_name: str, *, interactive: bool = False) -> str:
    """Read a secret without putting it in logs, URLs, configs or unattended prompts."""
    value = os.environ.get(environment_name)
    if value:
        return value
    if interactive and sys.stdin.isatty():
        return getpass.getpass(f"{environment_name}: ")
    raise MarketDataError(f"Missing credential environment variable {environment_name}; unattended mode does not prompt")


def import_market_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    """CLI boundary: cache-first manual import, with explicit unsupported providers."""
    settings = config.get("market_data", {})
    key_values = settings.get("cache_key")
    cache = MarketCache(settings.get("cache_directory", "data/market/cache"))
    key = MarketCacheKey(**key_values) if key_values else None
    cached = cache.load(key) if key else None
    if cached is not None:
        output = Path(settings.get("output_path", "data/market/canonical_prices.csv"))
        output.parent.mkdir(parents=True, exist_ok=True)
        cached.to_csv(output, index=False)
        coverage = market_coverage_report(cached, config)
        coverage_path = output.with_suffix(".coverage.json")
        coverage_path.write_text(json.dumps(coverage, indent=2) + "\n")
        return {"status": "cached", "records": len(cached), "cache_key": key.digest(), "output": str(output),
                "missing_products": coverage["missing_products"], "coverage_report": str(coverage_path)}
    provider = settings.get("provider", "manual_gme_csv")
    if provider not in ("manual_gme_csv", "manual_gme_xml"):
        raise MarketDataError(f"Provider {provider!r} download is not implemented by import-market-data; use download-data with a verified API configuration or manual GME CSV/XML import")
    path = settings.get("input_path")
    if not path:
        raise MarketDataError("No market_data.input_path: no price labels exist in the farm dataset")
    kwargs = {"constants": settings.get("constants", {})}
    if provider == "manual_gme_xml":
        frame = import_market_xml(path, record_tag=settings["record_tag"], field_mapping=settings["field_mapping"], **kwargs)
    else:
        frame = import_market_csv(path, column_mapping=settings.get("column_mapping"), separator=settings.get("separator", ","), **kwargs)
    output = Path(settings.get("output_path", "data/market/canonical_prices.csv"))
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    if key:
        cache.save(key, frame)
    coverage = market_coverage_report(frame, config)
    coverage_path = output.with_suffix(".coverage.json")
    coverage_path.write_text(json.dumps(coverage, indent=2) + "\n")
    return {"status": "imported", "missing_products": coverage["missing_products"], "coverage_report": str(coverage_path), "records": len(frame), "output": str(output),
            "earliest_product": frame.product_start.min().isoformat() if len(frame) else None,
            "latest_product_end": frame.product_end.max().isoformat() if len(frame) else None}
