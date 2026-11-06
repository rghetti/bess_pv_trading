"""Acquire Swissgrid interval energy statistics as self-contained DuckDB snapshots."""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import json
import os
from pathlib import Path
import posixpath
from typing import Any, Mapping
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .api import ApiRequest, SecureTransport
from .duckdb_store import DuckDBCache
from .gme_workbooks import _shared_strings


_NS = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
_RELATIONSHIP_ID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
PARSER_REVISION = "swissgrid-energy-statistics-v1"


def _cell_text(cell: ET.Element, strings: list[str]) -> str:
    """Decode provider values, including the workbook's cached formula results."""
    if cell.get("t") == "inlineStr":
        return "".join(part.text or "" for part in cell.findall(".//x:t", _NS))
    value = cell.findtext("x:v", default="", namespaces=_NS)
    if cell.get("t") == "s" and value:
        return strings[int(value)]
    return value


def _sheet_path(archive: zipfile.ZipFile, sheet_name: str) -> str:
    """Resolve the configured sheet name through XLSX relationships."""
    workbook = ET.fromstring(archive.read("xl/workbook.xml"))
    relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
    targets = {item.attrib["Id"]: item.attrib["Target"] for item in relationships}
    for sheet in workbook.findall("x:sheets/x:sheet", _NS):
        if sheet.get("name") == sheet_name:
            target = targets[sheet.attrib[_RELATIONSHIP_ID]]
            return target.lstrip("/") if target.startswith("/") else posixpath.normpath(
                posixpath.join("xl", target)
            )
    raise ValueError(f"Swissgrid worksheet {sheet_name!r} is absent")


def _source_rows(payload: bytes, settings: Mapping[str, Any]) -> tuple[list[dict], dict]:
    """Stream only the required worksheet and retain headers, units and source rows."""
    columns = {settings["timestamp_column"], *(field["column"] for field in settings["fields"].values())}
    headers = {}
    units = {}
    rows = []
    formula_cells = 0
    with zipfile.ZipFile(BytesIO(payload)) as archive:
        strings = _shared_strings(archive)
        sheet_path = _sheet_path(archive, settings["sheet_name"])
        with archive.open(sheet_path) as stream:
            for _, element in ET.iterparse(stream, events=("end",)):
                if element.tag != f"{{{_NS['x']}}}row":
                    continue
                row_number = int(element.attrib["r"])
                values = {}
                # Decode selected columns only; canton and formatting cells are irrelevant here.
                for cell in element:
                    column = "".join(character for character in cell.get("r", "") if character.isalpha())
                    if column in columns:
                        values[column] = _cell_text(cell, strings)
                        formula_cells += int(cell.find("x:f", _NS) is not None)
                if row_number == settings["header_row"]:
                    headers = values
                elif row_number == settings["unit_row"]:
                    units = values
                elif row_number >= settings["first_data_row"]:
                    # Blank formatting rows do not represent observations; numeric gaps remain rows.
                    if values.get(settings["timestamp_column"], "").strip():
                        rows.append({"source_row": row_number, **values})
                    elif any(str(value).strip() for value in values.values()):
                        raise ValueError(f"Swissgrid source row {row_number} has values but no timestamp")
                element.clear()
    if not rows:
        raise ValueError("Swissgrid workbook contains no interval rows")
    for name, field in settings["fields"].items():
        header = " ".join(headers.get(field["column"], "").split()).casefold()
        expected = " ".join(field["expected_header"].split()).casefold()
        if expected not in header or units.get(field["column"]) != field["native_unit"]:
            raise ValueError(f"Swissgrid header/unit contract changed for {name}: column {field['column']}")
    return rows, {"source_sheet": settings["sheet_name"], "cached_formula_cells": formula_cells}


def parse_energy_statistics(payload: bytes, settings: Mapping[str, Any], source: Mapping[str, Any]):
    """Normalize native interval energies and prices without interpolation or zero filling."""
    rows, metadata = _source_rows(payload, settings)
    source_frame = pd.DataFrame(rows)
    raw_stamps = source_frame[settings["timestamp_column"]].astype(str)
    numeric_stamps = pd.to_numeric(raw_stamps, errors="coerce")
    wall_clock = pd.to_datetime(raw_stamps.where(numeric_stamps.isna()),
                                format=settings["timestamp_format"], errors="coerce")
    # Excel serial dates have cached floating point values; round only their timestamp precision.
    numeric_mask = numeric_stamps.notna()
    wall_clock.loc[numeric_mask] = (
        pd.Timestamp("1899-12-30") + pd.to_timedelta(numeric_stamps[numeric_mask], unit="D")
    ).dt.round("min")
    if wall_clock.isna().any():
        raise ValueError("Swissgrid workbook contains unparseable timestamps")
    stamps = pd.DatetimeIndex(wall_clock).tz_localize(
        settings["timezone"], ambiguous="infer", nonexistent="raise"
    ).tz_convert("UTC").as_unit("ns")
    interval = pd.Timedelta(minutes=int(settings["interval_minutes"]))
    if source["timestamp_label"] == "interval_end":
        starts = stamps - interval
    elif source["timestamp_label"] == "interval_start":
        starts = stamps
    else:
        raise ValueError("timestamp_label must explicitly be interval_start or interval_end")
    year = int(source["year"])
    first = pd.Timestamp(f"{year}-01-01", tz=settings["timezone"]).tz_convert("UTC")
    stop = pd.Timestamp(f"{year + 1}-01-01", tz=settings["timezone"]).tz_convert("UTC")
    grid = pd.date_range(first, stop, freq=interval, inclusive="left").as_unit("ns")
    if starts.has_duplicates or not starts.is_monotonic_increasing or not starts.isin(grid).all():
        raise ValueError(f"Swissgrid {year} timestamps are duplicated, unordered or outside their source year")
    missing_grid = grid.difference(starts)
    if source.get("require_complete_year", False) and len(missing_grid):
        raise ValueError(f"Swissgrid {year} is missing {len(missing_grid)} interval timestamps")
    frame = pd.DataFrame({
        "provider": "swissgrid", "dataset": settings["dataset"], "zone": "CH",
        "product_start": starts, "product_end": starts + interval, "reference_time": starts,
        "source_row": source_frame["source_row"].to_numpy(), "source_timestamp_text": raw_stamps.to_numpy(),
        "source_file": source["filename"], "source_timestamp_label": source["timestamp_label"],
        "native_interval_minutes": int(settings["interval_minutes"]),
        "historical_available_at": pd.NaT, "scheduled_available_at": pd.NaT,
        "availability_provenance": "monthly_publication_exact_historical_release_unknown",
        "temporal_semantics": "realized_interval_energy_or_control_price",
    })
    conversion = 1000.0 * interval.total_seconds() / 3600.0
    column_units = {}
    # Keep both native energy and mean MW so the unit conversion can be audited independently.
    for name, field in settings["fields"].items():
        raw_values = source_frame.get(field["column"], pd.Series("", index=source_frame.index)).fillna("").astype(str)
        values = pd.to_numeric(raw_values.replace("", np.nan), errors="coerce").astype(float)
        if (raw_values.ne("") & values.isna()).any() or np.isinf(values).any():
            raise ValueError(f"Swissgrid {name} contains a nonnumeric or infinite value")
        if field["native_unit"] == "kWh":
            frame[f"{name}_energy_kwh"] = values.to_numpy()
            frame[f"{name}_mw"] = values.to_numpy() / conversion
            column_units[f"{name}_energy_kwh"] = "kWh"
            column_units[f"{name}_mw"] = "MW"
        elif field["native_unit"] == "Euro/MWh":
            frame[f"{name}_eur_mwh"] = values.to_numpy()
            column_units[f"{name}_eur_mwh"] = "EUR/MWh"
        else:
            raise ValueError(f"Unsupported configured Swissgrid unit for {name}")
    # Net border exports retain Switzerland as the positive direction, with missing legs preserved.
    for border, legs in settings["net_borders"].items():
        frame[f"physical_net_export_{border}_mw"] = (
            frame[f"{legs['export']}_mw"] - frame[f"{legs['import']}_mw"]
        )
        column_units[f"physical_net_export_{border}_mw"] = "MW"
    frame["value"] = frame["consumption_mw"] - frame["generation_mw"]
    frame["value_mw"] = frame["value"]
    frame["unit"] = "MW"
    metadata.update({
        "normalizer_revision": PARSER_REVISION, "source_kind": "swissgrid_public_xlsx",
        "source_file": source["filename"], "source_url": source["url"],
        "source_page": settings["source_page"], "source_year": year,
        "source_timestamp_label": source["timestamp_label"], "native_timezone": settings["timezone"],
        "native_interval_minutes": int(settings["interval_minutes"]), "column_units": column_units,
        "energy_to_mean_mw_divisor": conversion, "timestamp_gaps_full_year": len(missing_grid),
        "first_product_start": starts.min().isoformat(), "last_product_end": (starts.max() + interval).isoformat(),
        "source_rows": len(frame), "missing_values": {name: int(frame[name].isna().sum()) for name in column_units},
        "cross_border_definition": "physical energy over all transmission lines per border, separate directional legs",
        "historical_availability": "initial monthly publication after month end; exact historical revisions not reconstructed",
        "publication_policy": "15th working day after month end; monthly revisions; data reliable from six months onwards",
        "same_delivery_day_feature_use": "retrospective; explicitly authorized by user, not gate-available observations",
        "value_definition": "control-block consumption minus generation in MW",
    })
    return frame, metadata


def _identity(settings: Mapping[str, Any], source: Mapping[str, Any]) -> dict:
    """Key immutable source revisions by year and the complete normalization contract."""
    year = int(source["year"])
    return {
        "provider": "swissgrid", "dataset": settings["dataset"], "zone": "CH",
        "start": f"{year}-01-01", "end": f"{year + 1}-01-01",
        "options": {"parser": PARSER_REVISION, "fields": settings["fields"],
                    "net_borders": settings["net_borders"], "timezone": settings["timezone"],
                    "interval_minutes": settings["interval_minutes"],
                    "sheet_name": settings["sheet_name"], "timestamp_label": source["timestamp_label"]},
    }


def _write_report(path: Path, report: Mapping[str, Any]) -> None:
    """Publish an acquisition report atomically so interruptions leave a readable checkpoint."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")
    temporary.replace(path)


def _plot_source(frame: pd.DataFrame, settings: Mapping[str, Any], year: int, directory: Path) -> list[str]:
    """Group observed physical exchanges and national balance by source year for inspection."""
    if not settings.get("plots", False):
        return []
    cache = Path(settings["plot_cache_directory"])
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache.resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    borders = settings["plot_borders"]
    columns = [f"physical_net_export_{border}_mw" for border in borders]
    columns += ["generation_mw", "consumption_mw"]
    indexed = frame.set_index("product_start")[columns].tz_convert(settings["timezone"])
    daily = indexed.resample("D").mean()
    directory.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True, constrained_layout=True)
    for border in borders:
        axes[0].plot(daily.index, daily[f"physical_net_export_{border}_mw"], label=f"CH → {border.upper()}")
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_ylabel("Daily mean net export (MW)")
    axes[0].legend()
    axes[1].plot(daily.index, daily.generation_mw, label="Generation")
    axes[1].plot(daily.index, daily.consumption_mw, label="Consumption")
    axes[1].set_ylabel("Daily mean power (MW)")
    axes[1].legend()
    figure.suptitle(f"Swissgrid realized physical exchanges and control-block balance — {year}")
    path = directory / "physical_flows_and_balance.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [str(path)]


def acquire_energy_statistics(config: Mapping[str, Any], *, offline: bool = False,
                              refresh: bool = False, prepare_only: bool = False) -> dict:
    """Fetch and parse before opening one short DuckDB write session; retain raw XLSX bytes."""
    settings = config["swissgrid_energy"]
    store = DuckDBCache(settings["database_path"], threads=int(settings["database_threads"]))
    report_path = Path(settings["report_path"])
    report = {"status": "preparing", "database_path": str(store.directory),
              "sources": [], "prepared_at": datetime.now(timezone.utc).isoformat()}
    staged = []
    client = None
    for source in settings["sources"]:
        identity = _identity(settings, source)
        existing = None if refresh else store.load(identity)
        if existing is not None:
            # Cached snapshots contain the exact workbook, normalized records and checksums.
            store.verify_native_snapshot(identity, snapshot_id=existing[1]["snapshot_id"])
            plots = _plot_source(existing[0], settings, int(source["year"]), report_path.parent / str(source["year"]))
            report["sources"].append({"year": source["year"], "status": "cached", "manifest": existing[1], "plots": plots})
            continue
        raw_path = Path(settings["raw_directory"]) / source["filename"]
        if raw_path.exists() and (not refresh or offline):
            payload = raw_path.read_bytes()
        else:
            if offline:
                raise FileNotFoundError(f"Swissgrid workbook is absent in offline mode: {raw_path}")
            if client is None:
                client = SecureTransport(**settings["http"])
            payload = client.get(ApiRequest(source["url"]))
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            raw_path.write_bytes(payload)
        frame, metadata = parse_energy_statistics(payload, settings, source)
        staged.append((identity, payload, frame, metadata, source))
        report["sources"].append({"year": source["year"], "status": "prepared", "metadata": metadata})
        _write_report(report_path, report)
    if prepare_only:
        report["status"] = "prepared"
        _write_report(report_path, report)
        return report
    # Parsing and HTTP have finished; snapshots commit independently while sharing a single writer.
    if staged:
        with store.batch_write():
            for identity, payload, frame, metadata, source in staged:
                manifest = store.save(identity, payload, frame, metadata)
                store.verify_native_snapshot(identity, snapshot_id=manifest["snapshot_id"])
                entry = next(item for item in report["sources"] if item["year"] == source["year"])
                entry.update({"status": "imported", "manifest": manifest})
        # Rendering happens after the writer is released, so plotting cannot prolong database locks.
        for _, _, frame, _, source in staged:
            entry = next(item for item in report["sources"] if item["year"] == source["year"])
            entry["plots"] = _plot_source(frame, settings, int(source["year"]), report_path.parent / str(source["year"]))
    report["status"] = "complete"
    report["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_report(report_path, report)
    return report
