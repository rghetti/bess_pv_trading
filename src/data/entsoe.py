"""Verified ENTSO-E request definitions and strict XML interval decoding.

Transport, credentials and cache policy belong to :mod:`.api`. This module never
reads a token and never treats document creation or retrieval as publication.
"""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any
import xml.etree.ElementTree as ET

import pandas as pd

from .market_prices import MarketDataError


ENTSOE_ENDPOINT = "https://web-api.tp.entsoe.eu/api"
NORD_EIC = "10Y1001A1001A73I"
REFERENCE_DOCUMENTATION = "https://documenter.getpostman.com/view/7009892/2s93JtP3F6"
EXTRACTION_GUIDE = ("https://transparency.entsoe.eu/content/static_content/download?"
                    "path=%2FStatic+content%2Fweb+api%2FIG-for-TP-data-extraction-process.pdf")
DATASETS = {
    "day_ahead_prices": {"document_type": "A44", "unit": "EUR/MWh", "auction": "DA",
                         "article": "12.1.D", "domain_parameters": ("in_Domain", "out_Domain")},
    "load_forecast": {"document_type": "A65", "process_type": "A01", "unit": "MW",
                      "article": "6.1.B", "domain_parameters": ("outBiddingZone_Domain",)},
    "generation_forecast": {"document_type": "A71", "process_type": "A01", "unit": "MW",
                            "article": "14.1.C", "domain_parameters": ("in_Domain",)},
    "solar_forecast": {"document_type": "A69", "process_type": "A01", "psr_type": "B16",
                       "unit": "MW", "article": "14.1.D", "domain_parameters": ("in_Domain",)},
    "wind_onshore_forecast": {"document_type": "A69", "process_type": "A01", "psr_type": "B19",
                              "unit": "MW", "article": "14.1.D", "domain_parameters": ("in_Domain",)},
    "wind_offshore_forecast": {"document_type": "A69", "process_type": "A01", "psr_type": "B18",
                               "unit": "MW", "article": "14.1.D", "domain_parameters": ("in_Domain",)},
}


class EntsoeAPIError(MarketDataError):
    """The provider rejected a request or returned an unsupported document."""


class EntsoeNoDataError(EntsoeAPIError):
    """The provider explicitly reported no matching data; no values are invented."""


def dataset_metadata(dataset: str) -> dict[str, Any]:
    """Describe the verified request and unresolved historical availability."""
    if dataset not in DATASETS:
        raise EntsoeAPIError("Unsupported ENTSO-E dataset; IDA and imbalance coverage are not assumed")
    return {**deepcopy(DATASETS[dataset]), "provider": "entsoe", "dataset": dataset,
            "endpoint": ENTSOE_ENDPOINT, "reference": REFERENCE_DOCUMENTATION,
            "extraction_guide": EXTRACTION_GUIDE,
            "historical_feature_eligible": False,
            "availability_provenance": "not_verified; document creation is not historical publication",
            "historical_coverage": "must be measured from actual responses; no completeness assumed"}


def _utc(value: Any, name: str) -> pd.Timestamp:
    """Require explicit offsets at the provider boundary, including DST days."""
    try:
        timestamp = pd.Timestamp(value)
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise ValueError
        return timestamp.tz_convert("UTC")
    except (TypeError, ValueError, OverflowError) as exc:
        raise EntsoeAPIError(f"{name} must be an offset-aware timestamp") from None


def build_request(dataset: str, zone_eic: str, start: Any, end: Any) -> tuple[str, dict[str, str]]:
    """Build a public request identity; the transport adds securityToken later."""
    specification = dataset_metadata(dataset)
    if not isinstance(zone_eic, str) or len(zone_eic) != 16:
        raise EntsoeAPIError("zone_eic must be a 16-character EIC code")
    if any(not (character.isascii() and (character.isalnum() or character == "-")) for character in zone_eic):
        raise EntsoeAPIError("zone_eic must be a 16-character EIC code")
    first, last = _utc(start, "start"), _utc(end, "end")
    if first >= last or last - first > pd.Timedelta(days=366):
        raise EntsoeAPIError("Request requires a positive interval of at most one year; split longer downloads")
    if any(value.second or value.microsecond or value.nanosecond for value in (first, last)):
        raise EntsoeAPIError("ENTSO-E request boundaries must be exact minutes")
    params = {"documentType": specification["document_type"],
              "periodStart": first.strftime("%Y%m%d%H%M"),
              "periodEnd": last.strftime("%Y%m%d%H%M")}
    params.update({name: zone_eic for name in specification["domain_parameters"]})
    if "process_type" in specification:
        params["processType"] = specification["process_type"]
    if "psr_type" in specification:
        params["psrType"] = specification["psr_type"]
    return ENTSOE_ENDPOINT, params


def _text(element: ET.Element, path: str, *, required: bool = True) -> str | None:
    """Read local XML names after namespace removal without guessing field names."""
    value = element.findtext(path)
    if value is None or not value.strip():
        if required:
            raise EntsoeAPIError(f"ENTSO-E XML is missing required field {path}")
        return None
    return value.strip()


def _period_points(period: ET.Element, curve_type: str, value_field: str,
                   cancelled: bool) -> tuple[list[dict[str, Any]], int]:
    """Decode A01 fixed blocks and A03 repeated blocks at their native resolution."""
    first = _utc(_text(period, "timeInterval/start"), "Period start")
    last = _utc(_text(period, "timeInterval/end"), "Period end")
    resolution = _text(period, "resolution")
    # Explicit native MTUs avoid interpreting an hourly observation as four QHs.
    durations = {"PT15M": pd.Timedelta(minutes=15), "PT30M": pd.Timedelta(minutes=30),
                 "PT60M": pd.Timedelta(hours=1)}
    if resolution not in durations:
        raise EntsoeAPIError("Unsupported ENTSO-E native resolution; expected PT15M/PT30M/PT60M")
    step = durations[resolution]
    if first >= last or (last - first) % step:
        raise EntsoeAPIError("ENTSO-E Period is not a positive whole number of native intervals")
    count = int((last - first) / step)
    points: dict[int, tuple[float, str]] = {}
    for point in period.findall("Point"):
        try:
            position = int(_text(point, "position"))
        except (TypeError, ValueError):
            raise EntsoeAPIError("ENTSO-E Point position must be an integer") from None
        if position < 1 or position > count or position in points:
            raise EntsoeAPIError("ENTSO-E Point has a duplicate or out-of-range position")
        raw_value = _text(point, value_field, required=False)
        if cancelled or raw_value is None:
            value, status = float("nan"), "depublished_or_missing"
        else:
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                raise EntsoeAPIError("ENTSO-E Point value must be numeric") from None
            if not math.isfinite(value):
                raise EntsoeAPIError("ENTSO-E Point value must be finite")
            status = "observed"
        points[position] = value, status
    if not points:
        raise EntsoeAPIError("ENTSO-E Period contains no Points")
    positions = sorted(points)
    if curve_type == "A03" and positions[0] != 1:
        raise EntsoeAPIError("A03 Period must begin at position 1; a missing first block is ambiguous")
    rows = []
    for offset, position in enumerate(positions):
        stop = positions[offset + 1] if curve_type == "A03" and offset + 1 < len(positions) else count + 1
        native_positions = range(position, stop) if curve_type == "A03" else (position,)
        value, status = points[position]
        for native_position in native_positions:
            product_start = first + (native_position - 1) * step
            rows.append({"product_start": product_start, "product_end": product_start + step,
                         "value": value, "record_status": status, "resolution": resolution,
                         "source_position": position, "native_position": native_position,
                         "source_block_repeated": curve_type == "A03" and native_position != position,
                         "source_block_start": first + (position - 1) * step,
                         "source_block_end": first + (stop - 1) * step if curve_type == "A03" else product_start + step})
    return rows, count - len(rows)


def parse_entsoe_xml(payload: bytes | str, *, dataset: str, zone: str = "NORD",
                     zone_eic: str = NORD_EIC, retrieved_at: Any) -> pd.DataFrame:
    """Normalize a verified provider document without inventing publication times.

    Missing A01 positions and disjoint Period gaps remain missing. A03 repetition
    decodes the source's constant blocks, preserving the declared product length.
    Missing values remain explicit depublished/missing records and cannot enter the
    strict market importer. Rows are never made historically eligible by this API.
    """
    specification = dataset_metadata(dataset)
    retrieval = _utc(retrieved_at, "retrieved_at")
    body = payload.encode("utf-8") if isinstance(payload, str) else payload
    if not isinstance(body, bytes):
        raise EntsoeAPIError("ENTSO-E payload must be XML bytes or text")
    if b"<!DOCTYPE" in body.upper() or b"<!ENTITY" in body.upper():
        raise EntsoeAPIError("ENTSO-E XML must not contain DTD/entity declarations")
    try:
        document = ET.fromstring(body)
    except ET.ParseError:
        raise EntsoeAPIError("ENTSO-E response is not valid XML") from None
    for element in document.iter():
        element.tag = element.tag.rsplit("}", 1)[-1]
    if document.tag == "Acknowledgement_MarketDocument":
        reasons = document.findall("Reason")
        no_data = any(_text(reason, "code", required=False) == "999" and
                      "no matching data" in (_text(reason, "text", required=False) or "").lower()
                      for reason in reasons)
        if no_data:
            raise EntsoeNoDataError("ENTSO-E reports no matching data; check coverage, dataset and EIC")
        # Server reason text can echo credentials/query URLs; never expose it.
        raise EntsoeAPIError("ENTSO-E rejected the request; check credentials and documented parameters")
    expected_root = "Publication_MarketDocument" if dataset == "day_ahead_prices" else "GL_MarketDocument"
    if document.tag != expected_root or _text(document, "type") != specification["document_type"]:
        raise EntsoeAPIError("ENTSO-E returned an unexpected document type")
    process_type = _text(document, "process.processType", required=False)
    if "process_type" in specification and process_type != specification["process_type"]:
        raise EntsoeAPIError("ENTSO-E forecast document has an unexpected process type")
    if dataset == "day_ahead_prices" and process_type not in (None, "A01"):
        raise EntsoeAPIError("ENTSO-E price document has a non-day-ahead process type")
    document_id, revision = _text(document, "mRID"), _text(document, "revisionNumber")
    if not revision.isdecimal() or int(revision) < 1:
        raise EntsoeAPIError("ENTSO-E revisionNumber must be a positive integer")
    created = _utc(_text(document, "createdDateTime"), "createdDateTime")
    rows, missing_positions = [], 0
    for series in document.findall("TimeSeries"):
        series_id = _text(series, "mRID")
        curve = _text(series, "curveType", required=False) or "A01"
        if curve not in ("A01", "A03"):
            raise EntsoeAPIError("Only verified ENTSO-E A01/A03 block curves are supported")
        contract = _text(series, "contract_MarketAgreement.type", required=False)
        if dataset == "day_ahead_prices":
            if contract not in (None, "A01"):
                raise EntsoeAPIError("A non-day-ahead contract cannot be imported as DA or IDA")
            if _text(series, "businessType", required=False) not in (None, "A62"):
                raise EntsoeAPIError("Unexpected ENTSO-E day-ahead price business type")
            for domain in ("in_Domain.mRID", "out_Domain.mRID"):
                if _text(series, domain) != zone_eic:
                    raise EntsoeAPIError("ENTSO-E price response EIC does not match the requested zone")
            if (_text(series, "currency_Unit.name") != "EUR" or
                    _text(series, "price_Measure_Unit.name") != "MWH"):
                raise EntsoeAPIError("ENTSO-E price response requires explicit EUR/MWH units")
            value_field = "price.amount"
        else:
            domain = "outBiddingZone_Domain.mRID" if dataset == "load_forecast" else "inBiddingZone_Domain.mRID"
            if _text(series, domain, required=False) != zone_eic:
                # A71 can report consumption as out-domain; keep it distinct.
                if dataset != "generation_forecast" or _text(series, "outBiddingZone_Domain.mRID", required=False) != zone_eic:
                    raise EntsoeAPIError("ENTSO-E fundamental response EIC does not match the requested zone")
                direction = "consumption"
            else:
                direction = "production" if dataset != "load_forecast" else "load"
            if _text(series, "quantity_Measure_Unit.name") != "MAW":
                raise EntsoeAPIError("ENTSO-E forecast response requires explicit MAW (MW) units")
            if "psr_type" in specification and _text(series, "MktPSRType/psrType") != specification["psr_type"]:
                raise EntsoeAPIError("ENTSO-E response production type does not match the requested forecast")
            value_field = "quantity"
        cancelled = _text(series, "cancelledTS", required=False) == "A01"
        periods = series.findall("Period")
        if not periods:
            raise EntsoeAPIError("ENTSO-E TimeSeries contains no Periods")
        for period in periods:
            period_rows, missing = _period_points(period, curve, value_field, cancelled)
            missing_positions += missing
            for row in period_rows:
                row.update({"provider": "entsoe", "dataset": dataset, "zone": zone, "zone_eic": zone_eic,
                            "unit": specification["unit"], "retrieved_at": retrieval,
                            "document_id": document_id, "series_id": series_id, "revision": revision,
                            "source_document_time": created, "historical_available_at": pd.NaT,
                            "availability_provenance": "not_verified; document creation is not historical publication",
                            "historical_feature_eligible": False, "curve_type": curve,
                            "process_type": process_type, "psr_type": specification.get("psr_type"),
                            "business_type": _text(series, "businessType", required=False)})
                if dataset == "day_ahead_prices":
                    row.update({"auction": "DA", "price_eur_mwh": row["value"], "published_at": pd.NaT,
                                "publication_provenance": row["availability_provenance"]})
                else:
                    row.update({"value_mw": row["value"], "direction": direction})
                rows.append(row)
    if not rows:
        raise EntsoeNoDataError("ENTSO-E returned no time-series records")
    frame = pd.DataFrame(rows)
    identity = ["product_start", "product_end", "revision"]
    if dataset == "generation_forecast":
        identity.append("direction")
    if frame.duplicated(identity).any():
        raise EntsoeAPIError("ENTSO-E response has duplicate products; no series was silently selected")
    grouping = ["direction"] if dataset == "generation_forecast" else ["dataset"]
    for _, group in frame.groupby(grouping):
        ordered = group.sort_values("product_start")
        if (ordered.product_start.iloc[1:].reset_index(drop=True) <
                ordered.product_end.iloc[:-1].reset_index(drop=True)).any():
            raise EntsoeAPIError("ENTSO-E response has overlapping products")
    for column in ("product_start", "product_end", "source_block_start", "source_block_end", "retrieved_at",
                   "source_document_time", "historical_available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if "published_at" in frame:
        frame["published_at"] = pd.to_datetime(frame["published_at"], utc=True)
    frame = frame.sort_values(["product_start", "series_id"]).reset_index(drop=True)
    frame.attrs["quality"] = {"missing_a01_positions": missing_positions,
                              "missing_or_depublished_records": int(frame.value.isna().sum()),
                              "native_resolutions": sorted(frame.resolution.unique().tolist()),
                              "historical_availability_verified": False}
    return frame
