"""Acquire France and Switzerland aggregate electricity-balance features."""
from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
import hashlib
import json
from pathlib import Path
import posixpath
import zipfile
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

from .api import ApiRequest, SecureTransport
from .duckdb_store import DuckDBCache
from .feature_common import write_json
from .gme_workbooks import _cell_value, _shared_strings


def _identity(provider, dataset, zone, start, end, **options):
    """Build stable request identities so retrieval revisions remain traceable."""
    return {"provider": provider, "dataset": dataset, "zone": zone,
            "start": pd.Timestamp(start).isoformat(), "end": pd.Timestamp(end).isoformat(),
            "options": {"parser": "country-balance-v1", **options}}


def _number(value):
    """Convert provider numeric values while preserving nulls instead of zero filling."""
    if value is None or str(value).strip() in {"", "null", "None"}:
        return np.nan
    return float(str(value).replace(",", "."))


def parse_french_export(raw, *, start, end, config):
    """Normalize native half-hour measurements and retain every component and source row."""
    document = json.loads(raw)
    records = document if isinstance(document, list) else document.get("records", document.get("results", []))
    if not isinstance(records, list):
        raise ValueError("RTE export did not contain a JSON record list")
    source = config["france"]
    generation = source["generation_fields"] + [name for name in source["optional_generation_fields"]
                   if records and name in records[0]]
    optional_load = [name for name in source["optional_consumption_fields"]
                     if records and name in records[0]]
    rows = []
    wall_stamps = pd.DatetimeIndex([pd.Timestamp(f"{record[source['date_field']]} {record[source['hour_field']]}")
                                    for record in records])
    utc_stamps = pd.DatetimeIndex(pd.to_datetime(
        [record[source["timestamp_field"]] for record in records], utc=True))
    # Use date_heure as the unambiguous instant, but validate it against the
    # separately reported Paris wall clock. During spring-forward RTE can map
    # impossible local 02:xx records onto the same UTC instants as 03:xx; reject
    # precisely those mismatches. Autumn folds remain distinct by UTC instant.
    mapped_wall = utc_stamps.tz_convert(source["timezone"]).tz_localize(None)
    for ordinal, record in enumerate(records):
        if mapped_wall[ordinal] != wall_stamps[ordinal]:
            continue
        start_utc = utc_stamps[ordinal]
        if not (pd.Timestamp(start) <= start_utc < pd.Timestamp(end)):
            continue
        # RTE exposes a 15-minute row structure, but actual load/generation values
        # are half-hourly. Ignore forecast-only quarter points rather than creating
        # overlapping synthetic actual intervals.
        if start_utc.tz_convert(source["timezone"]).minute % 30:
            continue
        components = {name: _number(record.get(name)) for name in generation}
        load = _number(record.get(source["consumption_field"]))
        pumps = _number(record.get(source["pumping_field"]))
        storage_load = {name: _number(record.get(name)) for name in optional_load}
        # Required source components must all be present; structural optional fields are
        # included only if RTE supplies them in this export schema.
        core_generation = [_number(record.get(name)) for name in source["generation_fields"]]
        generation_mw = float(np.sum(core_generation + [components[n] for n in generation
                                      if n not in source["generation_fields"]])) if all(np.isfinite(core_generation + [components[n] for n in generation if n not in source["generation_fields"]])) else np.nan
        normalized_load = (load - pumps * float(source["pumping_multiplier"]) if np.isfinite(load)
                           and (np.isfinite(pumps) or source["pumping_field"] not in record)
                           else np.nan)
        if np.isfinite(normalized_load) and optional_load:
            normalized_load += sum(storage_load.values()) if all(np.isfinite(list(storage_load.values()))) else np.nan
        value = normalized_load - generation_mw if np.isfinite(normalized_load) and np.isfinite(generation_mw) else np.nan
        rows.append({"product_start": start_utc, "product_end": start_utc + pd.Timedelta(minutes=30),
            "reference_time": start_utc, "value": value, "unit": "MW", "value_mw": value,
            "consumption_reported_mw": load, "pumping_reported_mw": pumps,
            "consumption_normalized_mw": normalized_load, "generation_mw": generation_mw,
            "generation_components_json": json.dumps(components, sort_keys=True),
            "nature": record.get("nature"), "source_record": json.dumps(record, sort_keys=True),
            "source_row": ordinal, "historical_available_at": pd.NaT,
            "scheduled_available_at": pd.NaT, "availability_provenance": "unknown_historical_publication"})
    frame = pd.DataFrame(rows)
    if not frame.empty and frame.product_start.duplicated().any():
        raise ValueError("RTE export has duplicate timestamps; preserve and resolve source revisions explicitly")
    return frame


def _month_bounds(first, stop, timezone):
    """Yield month-sized UTC request bounds, clipped to the configured study window."""
    cursor = pd.Timestamp(first).tz_convert(timezone)
    final = pd.Timestamp(stop).tz_convert(timezone)
    while cursor < final:
        month_end = cursor.normalize().replace(day=1) + pd.offsets.MonthBegin(1)
        bound = min(month_end, final)
        yield cursor.tz_convert("UTC"), bound.tz_convert("UTC")
        cursor = bound


def download_french(config, *, dry_run=False, refresh=False, max_requests=None,
                    transport=None, progress=None):
    """Fetch one RTE JSON export per month and publish immutable normalized snapshots."""
    selected = config["country_balances"]
    france = selected["france"]
    start = pd.Timestamp(selected["start"], tz=france["timezone"]).tz_convert("UTC")
    stop = pd.Timestamp(selected["end_exclusive"], tz=france["timezone"]).tz_convert("UTC")
    store = DuckDBCache(selected["database_path"], threads=config["api"]["database_threads"])
    planned, cached = [], 0
    for first, end in _month_bounds(start, stop, france["timezone"]):
        key = _identity("rte", france["dataset"], "FR", first, end,
                        generation_fields=france["generation_fields"],
                        optional_generation_fields=france["optional_generation_fields"],
                        pumping_multiplier=france["pumping_multiplier"])
        existing = None if refresh else store.load(key)
        if existing is not None:
            # An empty provider response remains a recorded snapshot, but it
            # is not evidence that the requested month is complete forever.
            if existing[0].empty:
                existing = None
        if existing is not None:
            if store.load_unparsed(key) is not None:
                store.clear_unparsed(key)
            _, old_manifest = existing
            if old_manifest.get("metadata", {}).get("normalizer_revision") != "country-balance-v5":
                old_raw = store.load_raw(key)
                revised = parse_french_export(old_raw, start=first, end=end, config=selected)
                revised["provider"], revised["dataset"], revised["zone"] = "rte", france["dataset"], "FR"
                metadata = {**old_manifest.get("metadata", {}), "source_kind": "rte_public_json",
                    "native_interval_minutes": 30, "native_timezone": france["timezone"],
                    "balance_definition": "adjusted_consumption_minus_total_generation",
                    "historical_availability": "unknown", "normalizer_revision": "country-balance-v5",
                    "dst_handling": "date_heure UTC validated against local date+heure; impossible spring wall-clock rows skipped; fall repeated hours remain distinct UTC instants"}
                with store.batch_write():
                    revised_manifest = store.save(key, old_raw, revised, metadata)
                    store.verify_native_snapshot(key, snapshot_id=revised_manifest["snapshot_id"])
                existing = store.load(key)
        planned.append((first, end, key, existing))
        cached += existing is not None
    requests_left = len(planned) - cached
    limit = requests_left if max_requests is None else min(requests_left, max_requests)
    result = {"status": "planned" if dry_run else "partial", "database": selected["database_path"],
              "cached_requests": cached, "pending_requests": requests_left,
              "completed_requests": cached, "errors": [], "months": []}
    if dry_run:
        return result
    client = transport or SecureTransport(**config["api"]["http"])
    used = 0
    for first, end, key, existing in planned:
        if existing is not None:
            continue
        retained = store.load_unparsed(key) if not refresh else None
        if retained is None and used >= limit:
            break
        start_local = first.tz_convert(france["timezone"]).isoformat()
        end_local = end.tz_convert(france["timezone"]).isoformat()
        where = (f"date_heure >= '{start_local}' AND date_heure < '{end_local}'")
        request = ApiRequest(france["endpoint"], params={"where": where, "order_by": france["timestamp_field"]})
        raw = retained[0] if retained is not None else None
        try:
            if raw is None:
                raw = client.get(request)
                used += 1
            frame = parse_french_export(raw, start=first, end=end, config=selected)
            frame["provider"], frame["dataset"], frame["zone"] = "rte", france["dataset"], "FR"
            metadata = {"source_kind": "rte_public_json", "native_interval_minutes": 30,
                "native_timezone": france["timezone"], "balance_definition": "adjusted_consumption_minus_total_generation",
                "consumption_definition": "RTE national consumption; pumping adjustment configurable; excludes pumping and auxiliary own-use in source definition",
                "historical_availability": "unknown", "retrieved_at": datetime.now(timezone.utc).isoformat(),
                "source_record_count": len(frame), "normalizer_revision": "country-balance-v5",
                "dst_handling": "date_heure UTC validated against local date+heure; impossible spring wall-clock rows skipped; fall repeated hours remain distinct UTC instants"}
            with store.batch_write():
                manifest = store.save(key, raw, frame, metadata)
                store.verify_native_snapshot(key, snapshot_id=manifest["snapshot_id"])
            if frame.empty:
                result["errors"].append({"month": first.strftime("%Y-%m"),
                    "error": "Provider returned no usable source records; month remains incomplete"})
                result["months"].append({"start": first.isoformat(), "end": end.isoformat(), "records": 0,
                                         "snapshot_id": manifest["snapshot_id"]})
                continue
            result["completed_requests"] += 1
            result["months"].append({"start": first.isoformat(), "end": end.isoformat(), "records": len(frame),
                                     "snapshot_id": manifest["snapshot_id"]})
            if progress:
                progress(f"RTE national balance {first.tz_convert(france['timezone']):%Y-%m}: {len(frame)} native records")
        except Exception as error:
            if raw is not None:
                store.save_unparsed(key, raw, datetime.now(timezone.utc))
            result["errors"].append({"month": first.strftime("%Y-%m"), "error": str(error)[:300]})
    result["remaining_requests"] = len(planned) - result["completed_requests"]
    result["status"] = "complete" if result["remaining_requests"] == 0 and not result["errors"] else "partial"
    return result


def _swiss_rows(raw, settings):
    """Read only SwissGrid's configured raw worksheet columns and preserve source row numbers."""
    ns = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with zipfile.ZipFile(BytesIO(raw)) as archive:
        strings = _shared_strings(archive)
        if settings.get("sheet_name"):
            workbook = ET.fromstring(archive.read("xl/workbook.xml"))
            relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
            rel_ns = {"r": "http://schemas.openxmlformats.org/package/2006/relationships"}
            targets = {item.attrib["Id"]: item.attrib["Target"]
                       for item in relationships.findall("r:Relationship", rel_ns)}
            sheet_entry = next((item for item in workbook.findall(".//x:sheets/x:sheet", ns)
                                if item.attrib.get("name") == settings["sheet_name"]), None)
            if sheet_entry is None:
                raise ValueError(f"SwissGrid worksheet {settings['sheet_name']} is absent")
            target = targets[sheet_entry.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]]
            sheet_path = target.lstrip("/")
            if not sheet_path.startswith("xl/"):
                sheet_path = posixpath.normpath(posixpath.join("xl", sheet_path))
        else:
            sheet_path = settings["sheet_xml"]
        sheet = ET.fromstring(archive.read(sheet_path))
    records = []
    for row in sheet.findall(".//x:sheetData/x:row", ns):
        index = int(row.attrib["r"])
        if index < int(settings["first_data_row"]):
            continue
        values = {}
        for cell in row.findall("x:c", ns):
            address = cell.attrib.get("r", "")
            column = "".join(c for c in address if c.isalpha()).upper()
            if column not in {settings["timestamp_column"], settings["generation_column"], settings["consumption_column"]}:
                continue
            values[column] = _cell_value(cell, strings)
        if not values.get(settings["timestamp_column"]):
            continue
        timestamp_text = values[settings["timestamp_column"]]
        try:
            timestamp = pd.Timestamp(datetime.strptime(timestamp_text, settings["timestamp_format"]))
        except ValueError:
            # Some annual SwissGrid exports store interval starts as Excel's
            # numeric day serial instead of formatted date strings.
            timestamp = (pd.Timestamp("1899-12-30")
                         + pd.to_timedelta(float(timestamp_text), unit="D")).round("min")
        records.append((index, timestamp, _number(values.get(settings["generation_column"])),
                        _number(values.get(settings["consumption_column"]))))
    return records


def import_swiss_workbooks(config):
    """Import configured SwissGrid workbooks as native quarter-hour DuckDB snapshots."""
    selected = config["country_balances"]
    swiss = selected["switzerland"]
    files = sorted(Path().glob(swiss["path_glob"]))
    if not files:
        return {"status": "awaiting_workbooks", "pattern": swiss["path_glob"], "database": selected["database_path"]}
    store = DuckDBCache(selected["database_path"], threads=config["api"]["database_threads"])
    outputs = []
    for path in files:
        raw = path.read_bytes()
        records = _swiss_rows(raw, swiss)
        naive = pd.DatetimeIndex([item[1] for item in records])
        # Infer the two autumn occurrences from source row order; spring gaps stay absent.
        localized = naive.tz_localize(swiss["timezone"], ambiguous="infer", nonexistent="raise")
        frame = pd.DataFrame([{"source_row": row, "product_start": stamp.tz_convert("UTC"),
            "product_end": stamp.tz_convert("UTC") + pd.Timedelta(minutes=15),
            "reference_time": stamp.tz_convert("UTC"), "generation_energy_kwh": gen,
            "consumption_energy_kwh": load, "generation_mw": gen / 250.0,
            "consumption_mw": load / 250.0, "value": (load - gen) / 250.0,
            "value_mw": (load - gen) / 250.0, "unit": "MW", "historical_available_at": pd.NaT,
            "scheduled_available_at": pd.NaT, "availability_provenance": "unknown_historical_publication"}
            for (row, _, gen, load), stamp in zip(records, localized)])
        frame["provider"], frame["dataset"], frame["zone"] = "swissgrid", swiss["dataset"], "CH"
        digest = hashlib.sha256(raw).hexdigest()
        years = frame.product_start.dt.tz_convert(swiss["timezone"]).dt.year.unique()
        if len(years) != 1:
            raise ValueError(f"SwissGrid workbook {path.name} must contain one calendar year per file")
        year = int(years[0])
        file_start = pd.Timestamp(f"{year}-01-01", tz=swiss["timezone"]).tz_convert("UTC")
        file_end = pd.Timestamp(f"{year+1}-01-01", tz=swiss["timezone"]).tz_convert("UTC")
        key = _identity("swissgrid", swiss["dataset"], "CH", file_start, file_end,
                        source_file=path.name)
        metadata = {"source_kind": "manual_swissgrid_xlsx", "source_file": path.name,
            "source_sha256": digest, "source_sheet": swiss["sheet_name"], "native_interval_minutes": 15,
            "native_timezone": swiss["timezone"], "native_energy_unit": "kWh",
            "balance_definition": "SwissGrid control-block consumption D minus production C; D includes pumping, losses and own-use",
            "historical_availability": "unknown",
            "normalizer_revision": "swissgrid-balance-v5"}
        existing = store.load(key)
        if existing is None or existing[1].get("metadata", {}).get("normalizer_revision") != "swissgrid-balance-v5":
            with store.batch_write():
                manifest = store.save(key, raw, frame, metadata)
                store.verify_native_snapshot(key, snapshot_id=manifest["snapshot_id"])
        else:
            _, manifest = existing
        outputs.append({"file": str(path), "records": len(frame), "snapshot_id": manifest["snapshot_id"],
                        "raw_sha256": digest})
    return {"status": "complete", "database": selected["database_path"], "files": outputs}


def prepare_country_balances(config):
    """Create an auditable 15-minute wide table by interval containment, without filling gaps."""
    selected = config["country_balances"]
    store = DuckDBCache(selected["database_path"], threads=config["api"]["database_threads"])
    with store._connect() as con:
        if con is None:
            raise FileNotFoundError("Country balance DuckDB is absent")
        con.execute("BEGIN TRANSACTION")
        rows = con.execute("""SELECT provider,dataset,zone,product_start,product_end,value,value_mw,
            unit,_snapshot_id,_row_id FROM current_api_records WHERE dataset=? AND zone IN ('FR','CH')
            ORDER BY product_start""", [selected["dataset"]]).fetchdf()
        manifests = con.execute("""SELECT snapshot_id,provider,dataset,zone,raw_sha256,normalized_sha256,manifest_json
            FROM api_snapshots JOIN api_current_requests USING(snapshot_id)
            WHERE dataset=? AND zone IN ('FR','CH') ORDER BY snapshot_id""", [selected["dataset"]]).fetchdf()
        con.execute("COMMIT")
    if rows.empty:
        raise FileNotFoundError("No French or Swiss balance snapshots have been imported")
    rows["product_start"] = pd.to_datetime(rows.product_start, utc=True)
    rows["product_end"] = pd.to_datetime(rows.product_end, utc=True)
    first = pd.Timestamp(selected["start"], tz=selected["timezone"]).tz_convert("UTC")
    stop = pd.Timestamp(selected["end_exclusive"], tz=selected["timezone"]).tz_convert("UTC")
    grid = pd.date_range(first, stop, freq="15min", inclusive="left", tz="UTC")
    table = pd.DataFrame({"interval_start_utc": grid,
        "interval_end_utc": grid + pd.Timedelta(minutes=15)})
    sources = {}
    for provider, zone, output in (("rte", "FR", "french_balance_mw"),
                                   ("swissgrid", "CH", "swiss_balance_mw")):
        source = rows.loc[(rows.provider == provider) & (rows.zone == zone)].copy()
        values = np.full(len(grid), np.nan)
        lineage = np.full(len(grid), None, dtype=object)
        if not source.empty:
            delta_ns = pd.Timedelta(minutes=15).value
            starts_ns = pd.DatetimeIndex(source.product_start).as_unit("ns").asi8
            ends_ns = pd.DatetimeIndex(source.product_end).as_unit("ns").asi8
            if ((starts_ns-first.value) % delta_ns).any() or ((ends_ns-starts_ns) % delta_ns).any():
                raise ValueError(f"{zone} intervals do not align to the configured 15-minute grid")
            assigned = np.zeros(len(grid), dtype=bool)
            for source_index, (_, source_row) in enumerate(source.iterrows()):
                left = int((starts_ns[source_index]-first.value)//delta_ns)
                right = int((ends_ns[source_index]-first.value)//delta_ns)
                left, right = max(left, 0), min(right, len(grid))
                if left >= right:
                    continue
                if assigned[left:right].any():
                    raise ValueError(f"Overlapping {zone} balance products at {grid[left]}")
                values[left:right] = source_row["value_mw"]
                lineage[left:right] = f"{source_row['_snapshot_id']}:{source_row['_row_id']}"
                assigned[left:right] = True
        table[output] = values
        table[output + "_source_row"] = lineage
        local = table.interval_start_utc.dt.tz_convert(selected["timezone"])
        coverage_by_year = pd.DataFrame({"year": local.dt.year, "value": values}).groupby("year").value.apply(
            lambda values: int(np.isfinite(values).sum())).to_dict()
        sources[zone] = {"native_interval_minutes": int((source.product_end-source.product_start).dt.total_seconds().div(60).mode().iloc[0]) if not source.empty else None,
                         "native_rows": len(source),
                         "finite_native_records": int(source.value.notna().sum()),
                         "missing_aligned_intervals": int(np.isnan(values).sum()),
                         "finite_aligned_intervals_by_local_year": {str(year): count for year, count in coverage_by_year.items()},
                         "missing_interval_starts_utc": [stamp.isoformat() for stamp in table.loc[np.isnan(values), "interval_start_utc"]]}
    output_root = Path(selected["output_root"])
    run_id = selected.get("run_id") or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = output_root / f"country-balances-{run_id}"
    directory.mkdir(parents=True, exist_ok=False)
    parquet = directory / "country_balances_15min.parquet"
    table.to_parquet(parquet, index=False)
    table.to_csv(directory / "country_balances_15min.csv", index=False)
    report = {"status": "complete", "timezone": "UTC", "grid_minutes": 15,
        "interval_convention": "[start,end); French 30-minute values map to each contained 15-minute row, without being treated as independent observations",
        "balance_sign": "positive means consumption exceeds generation; negative means generation exceeds consumption",
        "historical_availability": "unknown in source records; no availability override applied",
        "source_summary": sources, "snapshots": manifests.to_dict("records"),
        "first_interval_utc": first.isoformat(), "end_exclusive_utc": stop.isoformat(),
        "rows": len(table), "parquet": str(parquet)}
    write_json(directory / "manifest.json", report)
    return {"status": "complete", "output_dir": str(directory), "rows": len(table), "sources": sources}
