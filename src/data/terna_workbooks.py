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


def _decode_second_occurrence(local: pd.DatetimeIndex, timezone: str,
                              minutes: int) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex, np.ndarray]:
    """Decode Terna's user-confirmed +1 minute markers only in the repeated hour."""
    markers = local.minute % minutes == 1
    if ((local.minute % minutes != 0) & ~markers).any() or (local.second != 0).any():
        raise ValueError("Terna labels must be quarter-hour starts or repeated-hour +1 minute markers")
    corrected = local - pd.to_timedelta(markers.astype(int), unit="min")
    daylight = corrected.tz_localize(timezone, ambiguous=True, nonexistent="raise")
    standard = corrected.tz_localize(timezone, ambiguous=False, nonexistent="raise")
    ambiguous = daylight.asi8 != standard.asi8
    if (markers & ~ambiguous).any():
        raise ValueError("Terna +1 minute marker outside the repeated autumn hour")
    # Ordinary quarter labels select the first fold; marked labels select the second.
    starts = corrected.tz_localize(timezone, ambiguous=~markers, nonexistent="raise").tz_convert("UTC")
    return corrected, starts, markers


def parse_terna_workbook(payload: bytes, settings: Mapping[str, Any],
                         workbook: Mapping[str, Any]) -> pd.DataFrame:
    """Keep the selected macrozone and reject ambiguous, missing or shifted intervals.

    The value-only XLSX reader is shared with GME imports. Date labels denote
    local interval starts by explicit configuration; prices are never interpolated.
    The explicit second_occurrence_minute_marker policy decodes the confirmed
    Terna export convention. Spring nonexistent labels are always rejected.
    """
    policy = settings["ambiguous_time_policy"]
    if settings["timestamp_label"] != "start" or policy not in {
            "reject", "quarantine_day", "second_occurrence_minute_marker"}:
        raise ValueError("Terna workbook requires start labels and a supported ambiguous_time_policy")
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
    first = pd.Timestamp(workbook["date_start"]).tz_localize(settings["timezone"]).tz_convert("UTC")
    stop = pd.Timestamp(workbook["date_end_exclusive"]).tz_localize(settings["timezone"]).tz_convert("UTC")
    expected = pd.date_range(first, stop, freq=f"{minutes}min", inclusive="left")
    quarantined = []
    quarantined_days = []
    selected_zone_rows = len(frame)
    if policy == "quarantine_day":
        # Reject uncertain autumn labels as a whole day, retaining the exact rows.
        # Export labels such as 02:01 are never rounded into guessed UTC intervals.
        daylight = local.tz_localize(settings["timezone"], ambiguous=True, nonexistent="raise")
        standard = local.tz_localize(settings["timezone"], ambiguous=False, nonexistent="raise")
        uncertain = (daylight.asi8 != standard.asi8) | (local.minute % minutes != 0) | (local.second != 0)
        quarantined_days = sorted(set(local[uncertain].strftime("%Y-%m-%d")))
        for day in quarantined_days:
            left = pd.Timestamp(day, tz=settings["timezone"])
            right = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(settings["timezone"])
            if right - left <= pd.Timedelta(days=1):
                raise ValueError("Off-grid Terna labels outside an autumn clock-change day")
        keep = ~local.strftime("%Y-%m-%d").isin(quarantined_days)
        quarantined = frame.loc[~keep].to_dict("records")
        frame = frame.loc[keep].copy()
        local = local[keep]
        expected = expected[~expected.tz_convert(settings["timezone"]).strftime("%Y-%m-%d").isin(quarantined_days)]
    markers = np.zeros(len(local), dtype=bool)
    if policy == "second_occurrence_minute_marker":
        local, starts, markers = _decode_second_occurrence(local, settings["timezone"], minutes)
    else:
        starts = local.tz_localize(settings["timezone"], ambiguous="raise", nonexistent="raise").tz_convert("UTC")
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
        "source_timestamp_text": frame[columns["timestamp"]].to_numpy(),
        "local_time_fold": np.asarray([stamp.fold for stamp in starts.tz_convert(settings["timezone"])], dtype=np.int8),
        **numbers, "value": numbers["price_eur_mwh"], "unit": "EUR/MWh",
        "aggregate_sign": signs, "native_interval_minutes": minutes, "resolution": "PT15M",
        "source_file": Path(workbook["path"]).name, "source_row": frame.source_row.to_numpy(),
        "source_price_text": frame[columns["price_eur_mwh"]].to_numpy(),
        "source_kind": "manual_terna_xlsx", "revision": "workbook_sha256:" + digest,
        "retrieved_at": pd.Timestamp.now(tz="UTC"), "published_at": pd.NaT,
        "historical_available_at": pd.NaT, "historical_feature_eligible": False,
        "availability_provenance": "manual monthly export; historical publication time not supplied",
        "interval_provenance": (
            "local-start labels; user-confirmed Terna +1 minute marker identifies second autumn occurrence; native UTC grid verified"
            if policy == "second_occurrence_minute_marker" else
            "explicit local-start labels localized in configured timezone; native quarter-hour grid verified")})
    result = result.sort_values("product_start").reset_index(drop=True)
    result.attrs.update(excluded_other_zone_records=original_rows-selected_zone_rows,
                        quarantined_delivery_days=quarantined_days,
                        quarantined_source_rows=quarantined,
                        ambiguous_time_policy=policy,
                        decoded_second_occurrence_records=int(markers.sum()),
                        price_component_max_residual_eur_mwh=float(residual.max()) if len(residual) else None)
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
