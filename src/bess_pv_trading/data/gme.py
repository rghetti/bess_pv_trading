"""Documented GME zonal-price requests and bounded, secret-safe JSON ZIP parsing.

Transport/authentication credentials and cache persistence belong to ``api``.
This module never infers historical publication from retrieval or flow dates.
Synthetic documented-format fixtures verify the parser; live payload coverage
must still be established by the user's authenticated downloads.
"""
from __future__ import annotations

import base64
import binascii
from datetime import date, timedelta
from io import BytesIO
import json
import math
from pathlib import PurePosixPath
import re
import stat
from typing import Any, Mapping, Sequence
import zipfile
import zlib

import pandas as pd

from .market_prices import MarketDataError


API_BASE = "https://api.mercatoelettrico.org/request"
TOKEN_URL = f"{API_BASE}/api/v1/Auth"
DATA_URL = f"{API_BASE}/api/v1/RequestData"
REFERENCE_DOCUMENTATION = "https://gme.mercatoelettrico.org/Portals/0/Documents/en-US/20251015Manuale_tecnico_API_En.pdf"
MI_QUARTER_HOUR_REFERENCE = "https://www.mercatoelettrico.org/Portals/0/Documents/it-IT/20250101DTF25MPE.pdf"
MI_PRODUCT_REFERENCE = "https://www.mercatoelettrico.org/Portals/0/Documenti/Presentazioni/217/20250326_intro15mMGP.pdf"
DAY_AHEAD_QUARTER_HOUR_START = date(2025, 10, 1)
SUPPORTED_HISTORY_START = date(2025, 1, 1)
DATASETS = {
    "day_ahead_prices": {"segment": "MGP", "auction": "DA"},
    "ida1_prices": {"segment": "MI-A1", "auction": "IDA1"},
    "ida2_prices": {"segment": "MI-A2", "auction": "IDA2"},
    "ida3_prices": {"segment": "MI-A3", "auction": "IDA3"},
}
ZONES = {"NORD", "CNOR", "CSUD", "SUD", "CALA", "SICI", "SARD", "PUN"}
_SECRET_FIELD_NAMES = {"login", "password", "token", "accesstoken", "authorization", "clientsecret", "clientid", "apikey", "secret"}


class GMEAPIError(MarketDataError):
    """The GME request or response does not meet the verified provider contract."""


class GMENoDataError(GMEAPIError):
    """The decoded document contains no eligible prices, including cancellation."""


class GMESensitiveResponseError(GMEAPIError):
    """The response may echo credentials; no raw or parsed body may be retained."""


def dataset_metadata(dataset: str) -> dict[str, Any]:
    """Describe public request fields and unverified historical availability."""
    if dataset not in DATASETS:
        raise GMEAPIError("Unsupported GME dataset; only MGP and MI-A1/2/3 zonal clearing prices are supported")
    return {**DATASETS[dataset], "provider": "gme", "dataset": dataset, "unit": "EUR/MWh",
            "endpoint": DATA_URL, "method": "POST", "platform": "PublicMarketResults",
            "data_name": "ME_ZonalPrices", "reference": REFERENCE_DOCUMENTATION,
            "mi_quarter_hour_reference": MI_QUARTER_HOUR_REFERENCE,
            "supported_history_from": SUPPORTED_HISTORY_START.isoformat(),
            "resolution_policy": "minimum native: MGP hourly before 2025-10-01, quarter-hour thereafter; MI-A quarter-hour in supported history",
            "historical_feature_eligible": False,
            "availability_provenance": "not_verified; flow date and retrieval are not historical publication times",
            "historical_coverage": "must be established from actual authenticated responses"}


def _utc(value: Any, name: str) -> pd.Timestamp:
    """Reject naive information and interval timestamps."""
    try:
        result = pd.Timestamp(value)
        if pd.isna(result) or result.tzinfo is None:
            raise ValueError
        return result.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError):
        raise GMEAPIError(f"{name} must be an offset-aware timestamp") from None


def _bounds(start: Any, end: Any) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Validate exclusive local-day bounds before using inclusive API dates."""
    first, last = _utc(start, "start"), _utc(end, "end")
    if first >= last:
        raise GMEAPIError("GME request requires a positive interval")
    local = [value.tz_convert("Europe/Rome") for value in (first, last)]
    if any(value != value.normalize() for value in local):
        raise GMEAPIError("GME requests require complete Europe/Rome calendar days")
    if local[0].date() < SUPPORTED_HISTORY_START:
        raise GMEAPIError("GME native interval semantics are verified only from 2025-01-01 in this adapter")
    return first, last


def _native_minutes(dataset: str, day: date, granularity_type: str | None) -> int:
    """Keep provider aggregations distinct from native observed clearing labels."""
    native = 60 if dataset == "day_ahead_prices" and day < DAY_AHEAD_QUARTER_HOUR_START else 15
    if granularity_type is not None:
        if dataset != "day_ahead_prices" or granularity_type != f"PT{native}":
            raise GMEAPIError("Only native GME granularity is supported; aggregated PT30/PT60 prices cannot replace native quarter-hour labels")
    return native


def build_request(dataset: str, start: Any, end: Any, *,
                  granularity_type: str | None = None) -> tuple[str, dict[str, Any]]:
    """Build a nonsecret JSON POST body; dates are inclusive yyyyMMdd integers."""
    metadata = dataset_metadata(dataset)
    first, last = _bounds(start, end)
    first_day = first.tz_convert("Europe/Rome").date()
    last_day = last.tz_convert("Europe/Rome").date() - timedelta(days=1)
    if granularity_type is not None and first_day < DAY_AHEAD_QUARTER_HOUR_START:
        raise GMEAPIError("GME GranularityType is documented only from 2025-10-01; omit it for earlier native hourly prices")
    _native_minutes(dataset, first_day, granularity_type)
    _native_minutes(dataset, last_day, granularity_type)
    attributes = {} if granularity_type is None else {"GranularityType": granularity_type}
    return DATA_URL, {"Platform": "PublicMarketResults", "Segment": metadata["segment"],
                      "DataName": "ME_ZonalPrices", "IntervalStart": int(first_day.strftime("%Y%m%d")),
                      "IntervalEnd": int(last_day.strftime("%Y%m%d")), "Attributes": attributes}


def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """Reject duplicate JSON object keys rather than silently hiding fields."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise GMEAPIError("GME JSON contains duplicate object fields")
        result[key] = value
    return result


def _json(body: bytes | str) -> Any:
    """Parse strict UTF-8 JSON without reflecting provider text in exceptions."""
    try:
        text = body.decode("utf-8-sig") if isinstance(body, bytes) else body
        return json.loads(text, object_pairs_hook=_object_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(ValueError()))
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
        if isinstance(exc, GMEAPIError):
            raise
        raise GMEAPIError("GME response is not strict UTF-8 JSON") from None


def _check_secret_fields(value: Any) -> None:
    """Reject credential-like fields recursively before any data can be cached."""
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if normalized in _SECRET_FIELD_NAMES:
                raise GMESensitiveResponseError("GME data response contains credential-like fields; response must not be retained")
            _check_secret_fields(child)
    elif isinstance(value, list):
        for child in value:
            _check_secret_fields(child)


def _check_secret_values(body: bytes | str, sensitive_values: Sequence[str]) -> None:
    """Check plain and JSON-escaped in-memory secrets without printing any value."""
    text = body.decode("utf-8-sig", errors="replace") if isinstance(body, bytes) else body
    for value in sensitive_values:
        if not isinstance(value, str) or not value:
            continue
        candidates = {value, json.dumps(value, ensure_ascii=False)[1:-1],
                      json.dumps(value, ensure_ascii=True)[1:-1]}
        # A free-text note can itself contain serialized JSON. Check one further
        # escaping layer as well as the semantically decoded document below.
        candidates.update(json.dumps(candidate, ensure_ascii=True)[1:-1] for candidate in tuple(candidates))
        if any(candidate in text for candidate in candidates):
            raise GMESensitiveResponseError("GME data response may echo a credential; response must not be retained")


def parse_token_response(payload: bytes | str) -> str:
    """Read the authentication token in memory; never include server reasons."""
    document = _json(payload)
    if not isinstance(document, dict) or document.get("Success") is not True:
        raise GMEAPIError("GME authentication failed; verify the assigned API login and password")
    token = document.get("token")
    if not isinstance(token, str) or not token or not re.fullmatch(r"[A-Za-z0-9_.-]+", token):
        raise GMEAPIError("GME authentication response lacks a valid nonempty token")
    return token


def _decoded_response(payload: bytes | str, sensitive_values: Sequence[str], *,
                      max_response_bytes: int = 64 * 1024 * 1024,
                      max_archive_bytes: int = 48 * 1024 * 1024,
                      max_uncompressed_bytes: int = 128 * 1024 * 1024,
                      max_records: int = 1_000_000) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate one bounded JSON ZIP in memory, with no filesystem extraction."""
    if not isinstance(payload, (bytes, str)):
        raise GMEAPIError("GME payload must be JSON bytes or text")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0
           for value in (max_response_bytes, max_archive_bytes, max_uncompressed_bytes, max_records)):
        raise GMEAPIError("GME response safety limits must be positive integers")
    raw = payload.encode("utf-8") if isinstance(payload, str) else payload
    if len(raw) > max_response_bytes:
        raise GMEAPIError("GME response exceeds the configured envelope size limit")
    _check_secret_values(raw, sensitive_values)
    envelope = _json(raw)
    if not isinstance(envelope, dict):
        raise GMEAPIError("GME data response requires an object envelope")
    _check_secret_fields(envelope)
    _check_secret_values(json.dumps(envelope, ensure_ascii=False), sensitive_values)
    content = envelope.get("ContentResponse")
    if envelope.get("FormatType") != ".json.zip" or not isinstance(content, str) or not content:
        raise GMEAPIError("GME request did not return the documented .json.zip data envelope; check credentials, coverage and provider status")
    if len(content) > 4 * ((max_archive_bytes + 2) // 3):
        raise GMEAPIError("GME archive exceeds the configured encoded size limit")
    try:
        zipped = base64.b64decode(content, validate=True)
    except (ValueError, binascii.Error):
        raise GMEAPIError("GME ContentResponse is not valid base64 ZIP data") from None
    if len(zipped) > max_archive_bytes:
        raise GMEAPIError("GME archive exceeds the configured size limit")
    # The original base64 envelope is retained in a cache. ZIP filenames, extra
    # fields and comments can therefore leak a credential even when JSON is safe.
    _check_secret_values(zipped, sensitive_values)
    try:
        with zipfile.ZipFile(BytesIO(zipped)) as archive:
            _check_secret_values(archive.comment, sensitive_values)
            entries = archive.infolist()
            if len(entries) != 1:
                raise GMEAPIError("GME archive must contain exactly one JSON file")
            entry = entries[0]
            _check_secret_values(entry.filename, sensitive_values)
            _check_secret_values(entry.comment, sensitive_values)
            _check_secret_values(entry.extra, sensitive_values)
            path = PurePosixPath(entry.filename)
            if (entry.is_dir() or path.is_absolute() or ".." in path.parts or "\\" in entry.filename
                    or "\x00" in entry.filename or re.match(r"^[A-Za-z]:", entry.filename)
                    or path.suffix.lower() != ".json" or stat.S_ISLNK(entry.external_attr >> 16)):
                raise GMEAPIError("GME archive contains an unsafe or unsupported member path")
            if entry.flag_bits & 1 or entry.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED):
                raise GMEAPIError("GME archive encryption/compression is unsupported")
            if entry.file_size > max_uncompressed_bytes:
                raise GMEAPIError("GME JSON exceeds the configured decompressed size limit")
            with archive.open(entry) as stream:
                decoded = stream.read(max_uncompressed_bytes + 1)
            if len(decoded) > max_uncompressed_bytes:
                raise GMEAPIError("GME JSON exceeds the configured decompressed size limit")
    except (zipfile.BadZipFile, RuntimeError, OSError, EOFError, NotImplementedError, UnicodeDecodeError, zlib.error):
        raise GMEAPIError("GME ContentResponse is not a valid supported ZIP archive") from None
    _check_secret_values(decoded, sensitive_values)
    records = _json(decoded)
    _check_secret_fields(records)
    _check_secret_values(json.dumps(records, ensure_ascii=False), sensitive_values)
    if not isinstance(records, list) or len(records) > max_records or any(not isinstance(row, dict) for row in records):
        raise GMEAPIError("GME zonal-price JSON must be a bounded array of record objects")
    return envelope, records


def validate_sanitized_response(payload: bytes | str, sensitive_values: Sequence[str] = (),
                                **limits: int) -> None:
    """Prove a data envelope and decoded file do not echo the known secrets.

    The API layer may retain an unparsed response only after this succeeds. An
    authentication response is never a valid data envelope and is never cached.
    """
    _decoded_response(payload, sensitive_values, **limits)


def _integer(value: Any, field: str) -> int:
    """Reject fractional, boolean and nonnumeric product identifiers."""
    if isinstance(value, bool) or not re.fullmatch(r"[0-9]+", str(value)):
        raise GMEAPIError(f"GME {field} must be an integer")
    return int(value)


def parse_gme_response(payload: bytes | str, *, dataset: str, zone: str = "NORD",
                       retrieved_at: Any, start: Any | None = None, end: Any | None = None,
                       granularity_type: str | None = None, sensitive_values: Sequence[str] = (),
                       **limits: int) -> pd.DataFrame:
    """Normalize native MGP/MI-A prices while retaining missing/cancelled status."""
    metadata = dataset_metadata(dataset)
    if zone not in ZONES or (zone == "PUN" and dataset != "day_ahead_prices"):
        raise GMEAPIError("GME requested zone must be an explicit supported zonal code; PUN is MGP-only")
    retrieval = _utc(retrieved_at, "retrieved_at")
    if (start is None) != (end is None):
        raise GMEAPIError("Both start and end are required to validate requested date coverage")
    bounds = None if start is None else _bounds(start, end)
    envelope, records = _decoded_response(payload, sensitive_values, **limits)
    if not records:
        raise GMENoDataError("GME returned an empty zonal-price dataset")
    rows = []
    cancelled_days: set[str] = set()
    cancelled_rows = 0
    for record in records:
        if not {"FlowDate", "Market", "Zone", "Notes"}.issubset(record):
            raise GMEAPIError("GME zonal-price record lacks documented required fields")
        if record["Market"] != metadata["segment"]:
            raise GMEAPIError("GME response market does not match the requested auction")
        if record["Zone"] != zone:
            continue
        raw_day = str(record["FlowDate"])
        try:
            if not re.fullmatch(r"[0-9]{8}", raw_day):
                raise ValueError
            day = pd.to_datetime(raw_day, format="%Y%m%d", errors="raise").date()
        except (ValueError, TypeError, OverflowError):
            raise GMEAPIError("GME FlowDate must be a valid yyyyMMdd calendar date") from None
        if day < SUPPORTED_HISTORY_START:
            raise GMEAPIError("GME native interval semantics are verified only from 2025-01-01")
        notes = record["Notes"]
        if notes is not None and not isinstance(notes, str):
            raise GMEAPIError("GME Notes must be a string or null")
        if notes is not None and len(notes) > 4096:
            raise GMEAPIError("GME Notes exceeds the supported length")
        if isinstance(notes, str) and "auction cancelled" in notes.lower():
            if bounds is not None and not (bounds[0].tz_convert("Europe/Rome").date() <= day
                    < bounds[1].tz_convert("Europe/Rome").date()):
                raise GMEAPIError("GME cancellation falls outside the requested date interval")
            # A cancelled auction has no clearing product. Do not require an
            # invented Hour/Period or turn a cancellation notice's zero into price.
            cancelled_days.add(raw_day)
            cancelled_rows += 1
            continue
        if not {"Hour", "Price"}.issubset(record):
            raise GMEAPIError("GME zonal-price record lacks documented required fields")
        minutes = _native_minutes(dataset, day, granularity_type)
        local_start = pd.Timestamp(day).tz_localize("Europe/Rome")
        local_end = pd.Timestamp(day + timedelta(days=1)).tz_localize("Europe/Rome")
        starts = pd.date_range(local_start.tz_convert("UTC"), local_end.tz_convert("UTC"),
                               freq=f"{minutes}min", inclusive="left")
        hour = _integer(record["Hour"], "Hour")
        period_value = record.get("Period")
        if minutes == 15:
            period = _integer(period_value, "Period")
            if not 1 <= period <= len(starts) or hour != (period - 1) // 4 + 1:
                raise GMEAPIError("GME quarter-hour Period/Hour is inconsistent with the native DST product grid")
        else:
            period = hour
            if not 1 <= hour <= len(starts):
                raise GMEAPIError("GME Hour is outside the native hourly DST product grid")
            if period_value not in (None, "") and _integer(period_value, "Period") != hour:
                raise GMEAPIError("GME hourly Period disagrees with its native Hour")
        product_start = starts[period - 1]
        if dataset == "ida3_prices" and product_start.tz_convert("Europe/Rome").hour < 12:
            raise GMEAPIError("GME MI-A3 returned an ineligible morning delivery product")
        product_end = product_start + pd.Timedelta(minutes=minutes)
        if bounds is not None and (product_start < bounds[0] or product_end > bounds[1]):
            raise GMEAPIError("GME response delivery products fall outside the requested date interval")
        value = record["Price"]
        missing = value is None or value == ""
        if missing:
            price = float("nan")
            status = "missing_price"
        else:
            try:
                if isinstance(value, bool):
                    raise ValueError
                price = float(value)
                if not math.isfinite(price):
                    raise ValueError
            except (ValueError, TypeError, OverflowError):
                raise GMEAPIError("GME Price must be finite numeric EUR/MWh, null, or empty") from None
            status = "observed"
        rows.append({"provider": "gme", "dataset": dataset, "zone": zone, "auction": metadata["auction"],
            "product_start": product_start, "product_end": product_end, "value": price,
            "price_eur_mwh": price, "unit": "EUR/MWh", "retrieved_at": retrieval,
            "source_document_time": pd.NaT, "historical_available_at": pd.NaT, "published_at": pd.NaT,
            "historical_feature_eligible": False, "availability_provenance": metadata["availability_provenance"],
            "publication_provenance": metadata["availability_provenance"], "revision": "latest_at_retrieval",
            "record_status": status, "resolution": f"PT{minutes}M", "native_interval_minutes": minutes,
            "source_flow_date": raw_day, "source_hour": hour, "source_period": period,
            "source_market": metadata["segment"], "source_notes": notes,
            "request_id": envelope.get("RequestId"), "interval_provenance": "native chronological flow Period/Hour in Europe/Rome; explicit DST-length civil day"})
    if not rows and cancelled_days:
        raise GMENoDataError("GME reports the requested auction was cancelled; no prices are invented")
    if not rows:
        raise GMENoDataError("GME returned no prices for the requested zone")
    frame = pd.DataFrame(rows).sort_values("product_start").reset_index(drop=True)
    if frame.source_flow_date.isin(cancelled_days).any():
        raise GMEAPIError("GME response mixes cancelled and priced records for the same auction day")
    if frame.duplicated(["product_start", "product_end"]).any():
        raise GMEAPIError("GME response has duplicate native auction/zone products")
    frame.attrs.update({"reference": REFERENCE_DOCUMENTATION, "response_contract": "documented JSON ZIP; live schema requires authenticated verification",
                        "missing_price_rows": int(frame.record_status.eq("missing_price").sum()),
                        "cancelled_price_rows": cancelled_rows, "cancelled_flow_dates": sorted(cancelled_days),
                        "historical_feature_eligible": False})
    return frame
