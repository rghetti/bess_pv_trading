"""Import native quarter-hour Terna monthly settlement exports into DuckDB."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from .duckdb_store import DuckDBCache
from .gme_workbooks import _read_sheet


def parse_terna_workbook(payload: bytes, settings: Mapping[str, Any],
                         workbook: Mapping[str, Any]) -> pd.DataFrame:
    """Keep the selected macrozone and reject ambiguous, missing or shifted intervals.

    The value-only XLSX reader is shared with GME imports. Date labels denote
    local interval starts by explicit configuration; prices are never interpolated.
    Ambiguous autumn timestamps require a source with an explicit offset before
    this parser can accept them. Spring nonexistent labels are always rejected.
    """
    if settings["timestamp_label"] != "start" or settings["ambiguous_time_policy"] != "reject":
        raise ValueError("Terna workbook requires explicit start labels and ambiguous_time_policy=reject")
    minutes = settings["native_interval_minutes"]
    if isinstance(minutes, bool) or minutes != 15:
        raise ValueError("Terna settlement workbooks must use native 15-minute intervals")
    rows = _read_sheet(payload, settings["sheet_xml"])
    header_index = int(settings["header_row"]) - 1
    if not 0 <= header_index < len(rows) - 1:
        raise ValueError("Terna workbook header_row does not identify a populated table")
    header, data = rows[header_index], rows[header_index + 1:]
    columns = settings["columns"]
    if len(header) != len(set(header)) or not set(columns.values()).issubset(header):
        raise ValueError("Terna workbook lacks unique configured column headers")
    if any(len(row) != len(header) for row in data):
        raise ValueError("Terna workbook contains incomplete/sparse rows; refusing shifted column values")
    frame = pd.DataFrame(data, columns=header)
    frame["source_row"] = np.arange(header_index + 2, header_index + 2 + len(frame))
    original_rows = len(frame)
    frame = frame.loc[frame[columns["macrozone"]].str.strip().eq(settings["zone"])].copy()
    if frame.empty:
        raise ValueError("Terna workbook has no records for the configured macrozone")
    local = pd.DatetimeIndex(pd.to_datetime(frame[columns["timestamp"]],
        format=settings["date_format"], errors="raise"))
    starts = local.tz_localize(settings["timezone"], ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    first = pd.Timestamp(workbook["date_start"]).tz_localize(settings["timezone"]).tz_convert("UTC")
    stop = pd.Timestamp(workbook["date_end_exclusive"]).tz_localize(settings["timezone"]).tz_convert("UTC")
    expected = pd.date_range(first, stop, freq=f"{minutes}min", inclusive="left")
    if starts.duplicated().any() or not starts.sort_values().equals(expected):
        raise ValueError("Terna workbook does not cover every expected quarter-hour exactly once")
    numbers = {}
    for name in ("aggregate_imbalance_mwh", "exchanges_mwh", "foreign_mwh",
                 "base_price_eur_mwh", "incentive_component_eur_mwh", "price_eur_mwh"):
        values = frame[columns[name]].astype(str)
        if settings["thousands_separator"]:
            values = values.str.replace(settings["thousands_separator"], "", regex=False)
        values = pd.to_numeric(values.str.replace(settings["decimal_separator"], ".", regex=False), errors="raise").to_numpy(float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Terna workbook has nonfinite {name}")
        numbers[name] = values
    residual = np.abs(numbers["base_price_eur_mwh"] + numbers["incentive_component_eur_mwh"] - numbers["price_eur_mwh"])
    if np.any(residual > float(settings["price_component_tolerance_eur_mwh"])):
        raise ValueError("Terna final price differs from base plus incentive beyond configured rounding tolerance")
    signs = frame[columns["aggregate_sign"]].str.strip().to_numpy()
    if not np.isin(signs, ["+", "-", "0"]).all():
        raise ValueError("Terna aggregate-sign column contains unknown labels")
    digest = hashlib.sha256(payload).hexdigest()
    result = pd.DataFrame({"provider": "terna", "dataset": settings["dataset"], "zone": settings["zone"],
        "macrozone": settings["zone"], "product_start": starts,
        "product_end": starts + pd.Timedelta(minutes=minutes), "reference_time": starts,
        "reference_time_local": local.astype(str), "source_date_tz": settings["timezone"],
        **numbers, "value": numbers["price_eur_mwh"], "unit": "EUR/MWh",
        "aggregate_sign": signs, "native_interval_minutes": minutes, "resolution": "PT15M",
        "source_file": Path(workbook["path"]).name, "source_row": frame.source_row.to_numpy(),
        "source_kind": "manual_terna_xlsx", "revision": "workbook_sha256:" + digest,
        "retrieved_at": pd.Timestamp.now(tz="UTC"), "published_at": pd.NaT,
        "historical_available_at": pd.NaT, "historical_feature_eligible": False,
        "availability_provenance": "manual monthly export; historical publication time not supplied",
        "interval_provenance": "explicit local-start labels localized in configured timezone; native quarter-hour grid verified"})
    result = result.sort_values("product_start").reset_index(drop=True)
    result.attrs.update(excluded_other_zone_records=original_rows-len(result),
                        price_component_max_residual_eur_mwh=float(residual.max()))
    return result


def import_terna_workbooks(config: Mapping[str, Any]) -> dict:
    """Save source bytes, NORD-only native records and checksummed monthly snapshots."""
    settings = config["terna_workbook_import"]
    cache = DuckDBCache(settings["database_path"], threads=int(settings["duckdb_threads"]))
    policy = {key: value for key, value in settings.items()
              if key not in {"database_path", "duckdb_threads", "workbooks", "report_path"}}
    policy_digest = hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()
    summaries = []
    with cache.batch_write():
        for workbook in settings["workbooks"]:
            payload = Path(workbook["path"]).read_bytes()
            frame = parse_terna_workbook(payload, settings, workbook)
            digest = hashlib.sha256(payload).hexdigest()
            identity = {"provider": "terna", "dataset": settings["dataset"], "zone": settings["zone"],
                "start": frame.product_start.iloc[0].isoformat(), "end": frame.product_end.iloc[-1].isoformat(),
                "source_kind": "manual_terna_xlsx"}
            current = cache.manifest(identity)
            cached = (current is not None and current.get("metadata", {}).get("source_sha256") == digest
                and current.get("metadata", {}).get("normalization_config_sha256") == policy_digest)
            if cached:
                # Idempotent imports also validate the previously stored bytes.
                cache.load(identity, verify_checksums=True)
                manifest = current
            else:
                manifest = cache.save(identity, payload, frame,
                    {"source_kind": "manual_terna_xlsx", "source_file": Path(workbook["path"]).name,
                     "source_sha256": digest, "normalization_config_sha256": policy_digest,
                     "normalization_policy": policy, "historical_feature_eligible": False,
                     **frame.attrs})
            summaries.append({"month": str(workbook["date_start"])[:7], "records": len(frame),
                "start_utc": identity["start"], "end_utc_exclusive": identity["end"],
                "snapshot_id": manifest["snapshot_id"], "source_sha256": digest,
                "price_min_eur_mwh": float(frame.price_eur_mwh.min()),
                "price_max_eur_mwh": float(frame.price_eur_mwh.max()),
                "zero_price_intervals": int(frame.price_eur_mwh.eq(0).sum()),
                "status": "cached" if cached else "imported", **frame.attrs})
    report = {"status": "complete", "database": settings["database_path"],
        "provider": "terna", "dataset": settings["dataset"], "zone": settings["zone"],
        "records": sum(item["records"] for item in summaries), "months": summaries}
    target = Path(settings["report_path"])
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report
