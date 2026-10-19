"""Import manual GME XLSX price workbooks as immutable DuckDB snapshots."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
from pathlib import Path
from typing import Any, Mapping
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .duckdb_store import DuckDBCache
from .gme import DATASETS


class GMEWorkbookError(ValueError):
    """A manual GME workbook does not match its explicitly configured schema."""


_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _shared_strings(archive: zipfile.ZipFile) -> list[str]:
    """Read optional shared strings without relying on third-party XLSX code."""
    try:
        root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
    except KeyError:
        return []
    return ["".join(part.text or "" for part in item.findall(".//x:t", _NS))
            for item in root.findall("x:si", _NS)]


def _cell_value(cell: ET.Element, strings: list[str]) -> str:
    """Decode the string forms used by GME exports and reject formulas."""
    kind = cell.get("t")
    if kind == "inlineStr":
        return "".join(part.text or "" for part in cell.findall(".//x:t", _NS))
    if cell.find("x:f", _NS) is not None:
        raise GMEWorkbookError("GME workbook formulas are unsupported; provide source values")
    value = cell.findtext("x:v", default="", namespaces=_NS)
    if kind == "s":
        try:
            return strings[int(value)]
        except (ValueError, IndexError):
            raise GMEWorkbookError("GME workbook has an invalid shared-string index") from None
    return value


def _read_sheet(payload: bytes, sheet_xml: str) -> list[list[str]]:
    """Read one configured worksheet by row/cell order, with no archive extraction."""
    try:
        with zipfile.ZipFile(__import__("io").BytesIO(payload)) as archive:
            if sheet_xml not in archive.namelist():
                raise GMEWorkbookError(f"Configured worksheet is absent: {sheet_xml}")
            strings = _shared_strings(archive)
            root = ET.fromstring(archive.read(sheet_xml))
    except (zipfile.BadZipFile, ET.ParseError, OSError):
        raise GMEWorkbookError("Input is not a readable XLSX workbook") from None
    rows = []
    for row in root.findall(".//x:sheetData/x:row", _NS):
        rows.append([_cell_value(cell, strings) for cell in row.findall("x:c", _NS)])
    if len(rows) < 2:
        raise GMEWorkbookError("GME workbook has no data rows")
    return rows


def _period_start(day: pd.Timestamp, period: int, interval_minutes: int,
                  timezone_name: str) -> pd.Timestamp:
    """Map GME's chronological one-based Period/Hour to the actual local day grid."""
    local_start = day.tz_localize(timezone_name)
    local_end = (day + pd.Timedelta(days=1)).tz_localize(timezone_name)
    starts = pd.date_range(local_start, local_end, freq=f"{interval_minutes}min", inclusive="left")
    if period < 1 or period > len(starts):
        raise GMEWorkbookError(f"Period {period} is outside the {len(starts)}-product local day")
    return starts[period - 1].tz_convert("UTC")


def _price(value: str, decimal_separator: str, thousands_separator: str,
           missing_tokens: set[str]) -> tuple[float, str]:
    """Parse a zonal EUR/MWh cell without treating absent prices as zero."""
    raw = value.strip()
    if raw in missing_tokens:
        return float("nan"), "missing_price"
    normalized = raw
    if thousands_separator:
        normalized = normalized.replace(thousands_separator, "")
    if decimal_separator != ".":
        normalized = normalized.replace(decimal_separator, ".")
    try:
        result = float(normalized)
    except (ValueError, OverflowError):
        raise GMEWorkbookError("GME price cell is not numeric under the configured separators") from None
    if not np.isfinite(result):
        raise GMEWorkbookError("GME price cell must be finite")
    return result, "observed"


def _parse_workbook(payload: bytes, source_file: str, dataset: str,
                    settings: Mapping[str, Any]) -> pd.DataFrame:
    """Convert configured GME rows to native-interval records without filling gaps."""
    metadata = DATASETS.get(dataset)
    if metadata is None:
        raise GMEWorkbookError(f"Unsupported GME dataset: {dataset}")
    required = ("timezone", "native_interval_minutes", "sheet_xml", "columns",
                "date_format", "decimal_separator", "thousands_separator")
    absent = [key for key in required if key not in settings]
    if absent:
        raise GMEWorkbookError(f"Workbook configuration is missing {absent}")
    minutes = settings["native_interval_minutes"]
    if isinstance(minutes, bool) or minutes not in (15, 30, 60):
        raise GMEWorkbookError("native_interval_minutes must be 15, 30, or 60")
    columns = dict(settings["columns"])
    for name in ("delivery_day", "period", "price"):
        if name not in columns:
            raise GMEWorkbookError(f"Workbook columns must map {name}")
    rows = _read_sheet(payload, str(settings["sheet_xml"]))
    header = rows[0]
    index = {name: position for position, name in enumerate(header)}
    # Notes are optional in MGP exports even when MI workbooks expose them.
    missing_headers = [source for name, source in columns.items()
                       if name != "notes" and source not in index]
    if missing_headers:
        raise GMEWorkbookError(f"Workbook headers do not contain configured columns {missing_headers}")
    first = pd.Timestamp(settings["date_start"])
    stop = pd.Timestamp(settings["date_end_exclusive"])
    if stop <= first:
        raise GMEWorkbookError("Configured workbook date bounds must be increasing")
    zone = str(settings["zone"])
    eligible_start = settings.get("eligible_start_clock")
    missing_tokens = set(map(str, settings.get("missing_price_tokens", ["", "-", "—"])))
    output = []
    cancelled_days: set[str] = set()
    for row_number, row in enumerate(rows[1:], start=2):
        if len(row) < len(header):
            row = row + [""] * (len(header) - len(row))
        try:
            day_value = pd.Timestamp(datetime.strptime(row[index[columns["delivery_day"]]],
                                      settings["date_format"]).date())
        except (ValueError, TypeError, IndexError):
            raise GMEWorkbookError(f"Invalid delivery date in worksheet row {row_number}") from None
        if not first <= day_value < stop:
            raise GMEWorkbookError(f"Delivery date outside configured bounds in worksheet row {row_number}")
        note_column = columns.get("notes")
        note = row[index[note_column]].strip() if note_column in index else ""
        raw_period = row[index[columns["period"]]].strip()
        raw_price = row[index[columns["price"]]].strip()
        marker = note.casefold()
        if not raw_period and not raw_price and any(term in marker for term in
                ("asta cancellata", "asta annullata", "auction cancelled", "auction canceled")):
            cancelled_days.add(day_value.date().isoformat())
            continue
        try:
            period = int(raw_period)
        except (ValueError, TypeError, IndexError):
            raise GMEWorkbookError(f"Invalid market period in worksheet row {row_number}") from None
        start = _period_start(day_value, period, minutes, settings["timezone"])
        if eligible_start is not None:
            if start.tz_convert(settings["timezone"]).strftime("%H:%M") < eligible_start:
                raise GMEWorkbookError(f"Ineligible delivery period in worksheet row {row_number}")
        price, status = _price(raw_price, str(settings["decimal_separator"]),
                               str(settings["thousands_separator"]), missing_tokens)
        output.append({"provider": "gme", "dataset": dataset, "zone": zone,
            "auction": metadata["auction"], "product_start": start,
            "product_end": start + pd.Timedelta(minutes=minutes), "value": price,
            "price_eur_mwh": price, "unit": "EUR/MWh", "retrieved_at": pd.Timestamp.now(tz="UTC"),
            "source_document_time": pd.NaT, "historical_available_at": pd.NaT,
            "published_at": pd.NaT, "historical_feature_eligible": False,
            "availability_provenance": "manual GME XLSX import; original publication timestamp not supplied",
            "publication_provenance": "manual GME XLSX import; no historical publication clock",
            "revision": "workbook_sha256:" + hashlib.sha256(payload).hexdigest(),
            "record_status": status, "resolution": f"PT{minutes}M",
            "native_interval_minutes": minutes,
            "source_flow_date": day_value.date().isoformat(), "source_hour": period,
            "source_period": period, "source_market": metadata["segment"],
            "source_notes": note, "source_file": source_file,
            "interval_provenance": "native chronological GME Period/Hour mapped onto Europe/Rome local-day grid"})
    frame = pd.DataFrame(output)
    if frame.empty:
        raise GMEWorkbookError("GME workbook contains no products")
    if frame.duplicated(["product_start", "product_end"]).any():
        raise GMEWorkbookError("GME workbook contains duplicate delivery products")
    frame = frame.sort_values("product_start", kind="stable").reset_index(drop=True)
    frame.attrs["cancelled_delivery_days"] = sorted(cancelled_days)
    return frame


def import_gme_workbooks(config: Mapping[str, Any]) -> dict[str, Any]:
    """Import configured manual XLSX files into immutable, checksummed DuckDB snapshots."""
    settings = config.get("gme_workbook_import", {})
    database = settings.get("database_path")
    workbooks = settings.get("workbooks")
    if not database or not isinstance(workbooks, list) or not workbooks:
        raise GMEWorkbookError("Configure database_path and at least one GME workbook")
    cache = DuckDBCache(database, threads=int(settings["duckdb_threads"]))
    timezone_name = settings["timezone"]
    first = pd.Timestamp(settings["date_start"]).tz_localize(timezone_name)
    stop = pd.Timestamp(settings["date_end_exclusive"]).tz_localize(timezone_name)
    summary = []
    with cache.batch_write():
        for item in workbooks:
            dataset = str(item["dataset"])
            path = Path(item["path"])
            if not path.is_file():
                raise FileNotFoundError(f"Configured GME workbook is missing: {path}")
            payload = path.read_bytes()
            source_file = path.name
            frame = _parse_workbook(payload, source_file, dataset,
                                    {**settings, **item, "timezone": timezone_name})
            cancelled_days = list(frame.attrs.get("cancelled_delivery_days", []))
            identity = {"provider": "gme", "dataset": dataset, "zone": str(item["zone"]),
                "start": first.tz_convert("UTC").isoformat(),
                "end": stop.tz_convert("UTC").isoformat(), "source_kind": "manual_gme_xlsx"}
            raw_sha = hashlib.sha256(payload).hexdigest()
            current = cache.manifest(identity)
            if current is not None and current.get("metadata", {}).get("source_sha256") == raw_sha:
                manifest = current
                status = "cached"
            else:
                metadata = {"source_kind": "manual_gme_xlsx", "source_file": source_file,
                    "source_sha256": raw_sha, "source_path_configured": str(item["path"]),
                    "native_interval_minutes": int(item["native_interval_minutes"]),
                    "timezone": timezone_name,
                    "cancelled_delivery_days": cancelled_days,
                    "historical_availability": "unverified; workbook content date is not publication time",
                    "historical_feature_eligible": False}
                manifest = cache.save(identity, payload, frame, metadata)
                status = "imported"
            summary.append({"dataset": dataset, "auction": DATASETS[dataset]["auction"],
                "zone": item["zone"], "records": int(manifest["records"]),
                "observed_prices": int(frame["price_eur_mwh"].notna().sum()),
                "missing_price_rows": int(frame["price_eur_mwh"].isna().sum()),
                "cancelled_delivery_days": cancelled_days,
                "native_interval_minutes": int(item["native_interval_minutes"]),
                "source_sha256": raw_sha, "snapshot_id": manifest["snapshot_id"],
                "status": status})
    return {"status": "complete", "database": str(database),
            "date_start": str(settings["date_start"]),
            "date_end_exclusive": str(settings["date_end_exclusive"]),
            "datasets": summary}
