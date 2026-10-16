"""Verified Terna public-API requests and lossless time-series normalization.

The provider does not infer a product interval or historical publication time
from a reference label or the time a response was downloaded. See docs/terna_api.md.
"""
from __future__ import annotations

from datetime import date, datetime
import json
import math
import re
from typing import Any, Mapping

import pandas as pd

from bess_pv_trading.data.api import ApiRequest
from bess_pv_trading.data.market_prices import MarketDataError

TOKEN_URL = "https://api.terna.it/public-api/access-token"
API_BASE = "https://api.terna.it"
AUTH_FIELDS = ("client_id", "client_secret", "grant_type")
TOKEN_DOCUMENTATION = "https://developer.terna.it/docs/read/Access_Token"
DATASETS = {
    "total_load": ("/load/v2.0/total-load", "total_load", "total_load_MW", "MW"),
    "total_load_forecast": ("/load/v2.0/total-load", "total_load", "forecast_total_load_MW", "MW"),
    "load_forecast": ("/market/v1.0/input/forecast-load", "forecast_load", "load_MW", "MW"),
    "wind_forecast": ("/generation/v2.0/wind-production-forecast", "wind_production_forecast", "forecast", "MW"),
    "renewable_generation": ("/generation/v2.0/renewable-generation", "renewable_generation", "renewable_generation_GWh", "GWh"),
}
DOCUMENTATION = {
    "total_load": "https://developer.terna.it/docs/read/apis_catalog/load/Total_Load",
    "total_load_forecast": "https://developer.terna.it/docs/read/apis_catalog/load/Total_Load",
    "load_forecast": "https://developer.terna.it/docs/read/apis_catalog/market/Load_Forecast",
    "wind_forecast": "https://developer.terna.it/docs/read/apis_catalog/generation/Wind_Production_Forecast",
    "renewable_generation": "https://developer.terna.it/docs/read/apis_catalog/generation/Renewable_Generation",
}
BIDDING_ZONES = {"North", "Centre-North", "South", "Centre-South", "Sardinia", "Sicily", "Calabria", "Italy"}
ENERGY_SOURCES = {"Thermal", "Wind", "Geothermal", "Photovoltaic", "Self-consumption", "Hydro"}
ZONE_CODES = {"North": "NORD", "Nord": "NORD", "NORD": "NORD", "Centre-North": "CNOR", "Centro-Nord": "CNOR", "Centre-South": "CSUD", "Centro-Sud": "CSUD", "South": "SUD", "Sud": "SUD", "Sardinia": "SARD", "Sardegna": "SARD", "Sicily": "SICI", "Sicilia": "SICI", "Calabria": "CALA", "Italy": "IT"}
_COLUMNS = ["provider", "dataset", "zone", "reference_time", "product_start", "product_end", "value", "unit", "retrieved_at", "source_document_time", "historical_available_at", "availability_provenance", "historical_feature_eligible", "interval_provenance", "session_type", "energy_source", "source_publication_date", "source_prevision_date", "source_date_tz", "source_offset", "source_array", "source_field", "unit_provenance", "source_record", "value_mw", "value_gwh"]


# Keep verified capabilities and unresolved semantics beside every cached request.
def dataset_metadata(dataset: str) -> dict[str, Any]:
    """Return nonsecret provenance without claiming tested historical coverage."""
    if dataset not in DATASETS:
        # Reuse the actionable unsupported-dataset message before credentials/network.
        request_spec(dataset, "2000-01-01", "2000-01-01", {})
    path, array_name, value_field, unit = DATASETS[dataset]
    return {
        "provider": "terna", "dataset": dataset, "endpoint": API_BASE + path,
        "documentation": DOCUMENTATION[dataset], "verified_on": "2026-10-16",
        "response_array": array_name, "source_value_field": value_field, "unit": unit,
        "schema_provenance": "observed user-downloaded wind_production_forecast/forecast, with documented forecast_load/load_MW compatibility" if dataset=="wind_forecast" else "published endpoint schema",
        "date_query_semantics": "inclusive Italian local calendar dates, dd/mm/yyyy",
        "interval_semantics": "unverified reference label; no product bounds without explicit configured interval assumption",
        "publication_semantics": "only an explicit publication_date clock establishes historical availability; prevision_date and retrieval time do not",
        "historical_coverage": "not guaranteed by public endpoint documentation",
        "revision_semantics": "latest at retrieval; no documented revision archive or revision identifier",
        "pagination": "not documented for this endpoint; unexpected continuation hints are rejected",
        "series_scope": "national actual energy by source" if dataset == "renewable_generation" else "MSD wind forecast, not a DA solar forecast" if dataset == "wind_forecast" else "source load/forecast series, not auction clearing prices",
        "live_authenticated_verification": False,
    }


# Build only documented routes and query parameters; credentials stay outside specs.
def request_spec(dataset: str, start: Any, end: Any, options: Mapping[str, Any] | None = None) -> ApiRequest:
    """Build an inclusive local-calendar-date query without undocumented pagination."""
    options = dict(options or {})
    if dataset in {"daily_prices", "preliminary_prices", "imbalance_prices"}:
        raise MarketDataError("Terna imbalance-price download is blocked: official Fees documentation conflicts between /fees/v1.0/ and /ifees/v1.0/market/. Verify the authenticated endpoint; use a provenance-bearing manual import meanwhile.")
    if dataset not in DATASETS:
        raise MarketDataError(f"Unsupported Terna dataset {dataset!r}; supported: {', '.join(DATASETS)}. No verified Terna DA/IDA clearing-price or solar-forecast endpoint is configured.")
    start_day, end_day = _day(start), _day(end)
    if end_day < start_day:
        raise MarketDataError("Terna end date must be on or after the start date")
    allowed = {"interval_minutes", "timestamp_label", "zone"}
    if dataset.startswith("total_load"):
        allowed |= {"bidding_zones"}
    elif dataset == "renewable_generation":
        allowed |= {"energy_sources"}
    else:
        allowed |= {"session_type"}
    unknown = set(options) - allowed
    if unknown:
        raise MarketDataError(f"Unknown Terna options: {', '.join(sorted(unknown))}")
    _interval_options(options)
    params: dict[str, Any] = {"dateFrom": start_day.strftime("%d/%m/%Y"), "dateTo": end_day.strftime("%d/%m/%Y")}
    if dataset.startswith("total_load"):
        zones = _string_list(options.get("bidding_zones", []), "bidding_zones")
        if set(zones) - BIDDING_ZONES:
            raise MarketDataError("Terna bidding_zones must use the documented English names, e.g. North; misspelled values can silently return empty bodies")
        if zones:
            params["biddingZone"] = zones
    elif dataset == "renewable_generation":
        sources = _string_list(options.get("energy_sources", []), "energy_sources")
        if set(sources) - ENERGY_SOURCES:
            raise MarketDataError("Terna energy_sources must use documented source names, e.g. Photovoltaic or Wind")
        if sources:
            params["type"] = sources
    else:
        session = options.get("session_type")
        valid = {f"MSD{index}" for index in range(1, 7)}
        if dataset == "load_forecast":
            valid.add("MGP")
        if session not in valid:
            raise MarketDataError(f"Terna {dataset} requires an explicit session_type in {', '.join(sorted(valid))}; wind forecasts are documented for MSD1..MSD6")
        params["sessionType"] = session
    return ApiRequest(url=API_BASE + DATASETS[dataset][0], params=params, headers={"Accept": "application/json"})


# Decode authentication separately so its payload is never sent into the data cache.
def parse_token(payload: bytes | str) -> tuple[str, int]:
    """Validate the documented Bearer token response without echoing its contents."""
    data = _json_object(payload, "token")
    token = data.get("access_token")
    expires = data.get("expires_in")
    if not isinstance(token, str) or not token.strip() or re.search(r"[\s\x00-\x1f\x7f]", token):
        raise MarketDataError("Terna returned an invalid access token")
    if str(data.get("token_type", "")).lower() != "bearer":
        raise MarketDataError("Terna returned an unsupported token type")
    try:
        duration = int(expires)
    except (TypeError, ValueError, OverflowError):
        raise MarketDataError("Terna returned an invalid token expiry") from None
    if isinstance(expires, bool) or duration <= 0 or str(duration) != str(expires):
        raise MarketDataError("Terna returned an invalid token expiry")
    return token, duration


# Retain the provider's source labels, dimensions, and provenance on every row.
def parse_terna_json(payload: bytes | str, dataset: str, options: Mapping[str, Any] | None = None,
                     retrieved_at: Any = None) -> pd.DataFrame:
    """Normalize documented responses; missing publication clocks remain unknown.

    Explicit interval options are study declarations, not facts established by
    the provider documentation. Date-only publication labels never become an
    invented midnight availability timestamp. Ambiguous DST labels are rejected.
    """
    options = dict(options or {})
    if dataset not in DATASETS:
        raise MarketDataError(f"Unsupported Terna dataset {dataset!r}")
    interval_minutes, timestamp_label = _interval_options(options)
    retrieved = _aware_timestamp(retrieved_at, "retrieved_at")
    document = _json_object(payload, "data")
    result = document.get("result")
    if not isinstance(result, dict) or str(result.get("status", "")).lower() != "completed":
        raise MarketDataError("Terna response did not report Completed; inspect the provider request and access permissions")
    # No supported endpoint documents pagination; refuse unexpected truncation hints.
    for name in ("next", "next_page", "nextPage", "continuationToken", "hasMore"):
        if document.get(name) or result.get(name):
            raise MarketDataError("Terna response indicates pagination not covered by the verified schema")
    _, array_name, value_field, unit = DATASETS[dataset]
    if dataset=="wind_forecast":
        # The live endpoint differs from the published load-shaped example.
        # Accept these two explicit contracts; never guess an arbitrary array/value.
        variants={"wind_production_forecast":"forecast","forecast_load":"load_MW"}
        present=[name for name in variants if name in document]
        if len(present)>1:
            raise MarketDataError("Terna wind response contains ambiguous supported arrays")
        if present:
            array_name=present[0]
            value_field=variants[array_name]
    records = document.get(array_name)
    if not isinstance(records, list):
        raise MarketDataError(f"Terna response is missing its supported {array_name} array")
    rows = []
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise MarketDataError(f"Terna record {index} must be an object")
        reference_field = "date" if dataset in {"total_load", "total_load_forecast", "renewable_generation"} else "reference_date"
        reference = _local_timestamp(record, reference_field, require_clock=True)
        source_zone = record.get("bidding_zone", record.get("zone", "Italy" if dataset == "renewable_generation" else None))
        if not isinstance(source_zone, str) or not source_zone.strip():
            raise MarketDataError(f"Terna record {index} lacks a zone")
        zone = ZONE_CODES.get(source_zone, source_zone)
        requested_zone = options.get("zone")
        if requested_zone and zone != ZONE_CODES.get(str(requested_zone), str(requested_zone)):
            continue
        value = _numeric(record.get(value_field), value_field)
        source_session = record.get("session_type")
        if dataset in {"load_forecast", "wind_forecast"}:
            if not isinstance(source_session, str) or not source_session.strip():
                raise MarketDataError(f"Terna record {index} lacks session_type")
            if options.get("session_type") and source_session != options["session_type"]:
                raise MarketDataError("Terna response session_type differs from the requested session")
        source = record.get("energy_source")
        if dataset == "renewable_generation" and source not in ENERGY_SOURCES:
            raise MarketDataError(f"Terna record {index} has an unsupported energy_source")
        publication = record.get("publication_date")
        publication_known = bool(publication) and _has_clock(publication)
        available = _local_timestamp(record, "publication_date", require_clock=True) if publication_known else pd.NaT
        source_document = _local_timestamp(record, "prevision_date", require_clock=True) if record.get("prevision_date") else pd.NaT
        if pd.notna(available) and available > retrieved:
            raise MarketDataError("Terna publication time is later than retrieval time")
        start, end = pd.NaT, pd.NaT
        if interval_minutes is not None:
            duration = pd.Timedelta(minutes=interval_minutes)
            start = reference if timestamp_label == "start" else reference - duration
            end = start + duration
        rows.append({
            "provider": "terna", "dataset": dataset, "zone": zone, "reference_time": reference,
            "product_start": start, "product_end": end, "value": value, "unit": unit,
            "retrieved_at": retrieved, "source_document_time": source_document,
            "historical_available_at": available,
            "availability_provenance": "provider_publication_date" if publication_known else "unknown_date_only_publication" if publication else "unknown_no_publication_timestamp",
            "historical_feature_eligible": bool(publication_known),
            "interval_provenance": "configured_interval_assumption" if interval_minutes is not None else "unresolved_provider_reference_label",
            "session_type": source_session, "energy_source": source,
            "source_publication_date": publication, "source_prevision_date": record.get("prevision_date"),
            "source_date_tz": record.get("date_tz"), "source_offset": record.get("date_offset", record.get("offset")),
            "source_array": array_name, "source_field": value_field,
            "unit_provenance": "endpoint_documentation_MW; observed forecast field has no unit suffix" if dataset=="wind_forecast" and value_field=="forecast" else "documented_endpoint_field",
            "source_record": json.dumps(record, sort_keys=True, separators=(",", ":")),
            "value_mw": value if unit == "MW" else float("nan"),
            "value_gwh": value if unit == "GWh" else float("nan"),
        })
    frame = pd.DataFrame(rows, columns=_COLUMNS)
    for name in ("reference_time", "product_start", "product_end", "retrieved_at", "source_document_time", "historical_available_at"):
        frame[name] = pd.to_datetime(frame[name], utc=True)
    identity = ["dataset", "zone", "reference_time", "session_type", "energy_source", "source_document_time", "source_publication_date"]
    if frame.duplicated(identity).any():
        raise MarketDataError("Duplicate Terna reference/zone/session/source/vintage records; no undocumented correction was applied")
    frame.attrs.update({"documentation": DOCUMENTATION[dataset], "source_array":array_name,"source_value_field":value_field,"historical_coverage": "not guaranteed by the public endpoint documentation", "pagination": "not documented; one request per configured date slice", "product_interval_semantics": "configured assumption" if interval_minutes is not None else "unresolved"})
    return frame.sort_values(["reference_time", "zone"], kind="stable").reset_index(drop=True)


# Apply explicit, validated temporal conventions rather than inferring from cadence.
def _interval_options(options: Mapping[str, Any]) -> tuple[int | None, str | None]:
    interval, label = options.get("interval_minutes"), options.get("timestamp_label")
    if interval is None and label is None:
        return None, None
    if isinstance(interval, bool) or not isinstance(interval, int) or interval <= 0 or label not in {"start", "end"}:
        raise MarketDataError("Terna interval alignment requires both positive integer interval_minutes and timestamp_label start/end; omit both when source semantics are unresolved")
    return interval, label


def _day(value: Any) -> date:
    """Accept only an explicit calendar day; conversion of UTC hours is not implicit."""
    if isinstance(value, datetime):
        if value.time() != datetime.min.time() or value.tzinfo is not None:
            raise MarketDataError("Terna request dates must be local calendar dates, not timezone-aware instants")
        return value.date()
    if isinstance(value, date):
        return value
    try:
        if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
            raise ValueError
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise MarketDataError("Terna request dates require YYYY-MM-DD local calendar dates") from None


def _string_list(value: Any, name: str) -> list[str]:
    """Reject scalar filter typos before Terna can silently return an empty body."""
    if not isinstance(value, (list, tuple)) or any(not isinstance(item, str) for item in value):
        raise MarketDataError(f"Terna {name} must be a list of strings")
    return list(value)


def _json_object(payload: bytes | str, context: str) -> dict[str, Any]:
    """Raise sanitized errors without exposing the provider body or authentication."""
    try:
        document = json.loads(payload)
    except (TypeError, ValueError, UnicodeError):
        raise MarketDataError(f"Terna {context} response is not valid JSON") from None
    if not isinstance(document, dict):
        raise MarketDataError(f"Terna {context} response must be a JSON object")
    return document


def _numeric(value: Any, field: str) -> float:
    """Preserve signed numbers without clipping, filling, or guessed locale formats."""
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError):
        raise MarketDataError(f"Terna contains a missing or invalid numeric {field}") from None
    if isinstance(value, bool) or not math.isfinite(number):
        raise MarketDataError(f"Terna contains a missing or nonfinite numeric {field}")
    return number


def _has_clock(value: Any) -> bool:
    """Do not manufacture midnight from a date-only publication label."""
    return isinstance(value, str) and bool(re.search(r"[T ]\d{2}:\d{2}", value))


def _aware_timestamp(value: Any, field: str) -> pd.Timestamp:
    """Require an explicit retrieval timezone, including in mocked transports."""
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None:
            raise ValueError
        return stamp.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError):
        raise MarketDataError(f"Terna {field} must be a timezone-aware timestamp") from None


def _local_timestamp(record: Mapping[str, Any], field: str, *, require_clock: bool) -> pd.Timestamp:
    """Honor explicit offsets; fail on inconsistent zones, ambiguous or skipped hours."""
    value, zone = record.get(field), record.get("date_tz")
    if not isinstance(zone, str) or not zone or require_clock and not _has_clock(value):
        raise MarketDataError(f"Terna {field} requires a timestamp and date_tz")
    try:
        local = pd.Timestamp(value)
        if pd.isna(local):
            raise ValueError
        offset = record.get("date_offset", record.get("offset"))
        # Reference offsets do not automatically describe a publication on another day.
        if field not in {"date", "reference_date"}:
            offset = None
        if local.tzinfo is not None:
            stamp = local
        elif offset:
            if not isinstance(offset, str) or not re.fullmatch(r"[+-]\d{2}:\d{2}", offset):
                raise ValueError
            stamp = pd.Timestamp(str(value) + offset)
        else:
            stamp = local.tz_localize(zone, ambiguous="raise", nonexistent="raise")
        in_zone = stamp.tz_convert(zone)
        if in_zone.tz_localize(None) != local.tz_localize(None):
            raise ValueError
        if offset and in_zone.strftime("%z") != offset.replace(":", ""):
            raise ValueError
        return stamp.tz_convert("UTC")
    except Exception:
        raise MarketDataError(f"Terna {field} has an invalid, inconsistent, ambiguous, or nonexistent local timestamp; an explicit reference offset is required across DST") from None
