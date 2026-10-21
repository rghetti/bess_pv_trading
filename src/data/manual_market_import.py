"""Stage native manual market exports offline and merge only on explicit command."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shlex

import numpy as np
import pandas as pd
import yaml

from ..config import load_config
from .api import _atomic_bytes, _json_bytes
from .duckdb_store import DuckDBCache
from .gme import DATASETS, native_price_resolution_minutes
from .gme_workbooks import _parse_workbook, _read_sheet
from .terna_workbooks import parse_terna_workbook


KEYS = ["provider", "dataset", "zone", "product_start", "product_end"]
SEMANTIC = ["price_eur_mwh", "unit", "native_interval_minutes", "base_price_eur_mwh",
            "incentive_component_eur_mwh", "aggregate_imbalance_mwh", "aggregate_sign",
            "exchanges_mwh", "foreign_mwh"]


def _settings(config):
    """Freeze the requested local-date window without opening any database."""
    settings = dict(config["manual_market_import"])
    if settings["zone"] != "NORD":
        raise ValueError("This manual import requires the explicitly selected NORD zone")
    settings["date_start"] = str(pd.Timestamp(settings["date_start"]).date())
    stop = settings["date_end_exclusive"]
    if stop is None:
        stop = (pd.Timestamp.now(tz=settings["timezone"]).date() + timedelta(days=1)).isoformat()
    settings["date_end_exclusive"] = str(pd.Timestamp(stop).date())
    if settings["date_start"] >= settings["date_end_exclusive"]:
        raise ValueError("Manual import requires increasing local-date bounds")
    return settings


def _unique_records(frame):
    """Remove only identical native products; conflicting values stop the import."""
    if frame.empty:
        return frame.copy()
    if frame[KEYS].isna().any().any():
        raise ValueError("Manual market records require complete provider and interval keys")
    fields = [name for name in SEMANTIC if name in frame]
    counts = frame.groupby(KEYS, dropna=False)[fields].nunique(dropna=False)
    if counts.gt(1).any().any():
        bad = counts.loc[counts.gt(1).any(axis=1)].index[0]
        raise ValueError(f"Conflicting prices or quantities for the same native product: {bad}")
    return frame.drop_duplicates(KEYS, keep="first").sort_values(KEYS, kind="stable").reset_index(drop=True)


def _read_scope(cache, provider, dataset, zone):
    """Read only the requested completed snapshots, retaining exact source evidence."""
    frames, entries = [], cache.entries(provider=provider, dataset=dataset, zone=zone)
    for entry in entries:
        loaded = cache.load_snapshot(entry["identity"], snapshot_id=entry["manifest"]["snapshot_id"])
        if loaded is None:
            raise ValueError("A published manual-market snapshot is missing")
        frame, _ = loaded
        if not frame.empty:
            frames.append(frame)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), entries


def _novel_records(existing, incoming):
    """Check shared values and reject incompatible overlapping product boundaries."""
    if existing.empty or incoming.empty:
        return incoming.copy()
    old = _unique_records(existing)
    _unique_records(pd.concat([old, incoming], ignore_index=True))
    ordered = old.sort_values("product_start")
    starts = pd.DatetimeIndex(ordered.product_start).asi8
    ends = pd.DatetimeIndex(ordered.product_end).asi8
    if len(starts) > 1 and (starts[1:] < ends[:-1]).any():
        raise ValueError("Existing market history contains overlapping native intervals")
    new_starts = pd.DatetimeIndex(incoming.product_start).asi8
    new_ends = pd.DatetimeIndex(incoming.product_end).asi8
    index = np.searchsorted(starts, new_ends, side="left") - 1
    valid = index >= 0
    selected = np.maximum(index, 0)
    overlaps = valid & (ends[selected] > new_starts)
    exact = (starts[selected] == new_starts) & (ends[selected] == new_ends)
    if (overlaps & ~exact).any():
        raise ValueError("Native interval boundaries overlap with different durations; no resampling is allowed")
    known = pd.MultiIndex.from_frame(old[KEYS])
    return incoming.loc[~pd.MultiIndex.from_frame(incoming[KEYS]).isin(known)].copy()


def _prepare_sources(settings, progress):
    """Validate current Excel files using the existing provider schema adapters."""
    root = Path(settings["source_directory"])
    gme = load_config(settings["gme_schema_config"])["gme_workbook_import"]
    terna = load_config(settings["terna_schema_config"])["terna_workbook_import"]
    terna = {**terna, "zone": settings["zone"], "timezone": settings["timezone"],
             "ambiguous_time_policy": settings["terna_ambiguous_time_policy"]}
    prepared = []
    paths = [(path, "gme") for path in sorted(root.glob(settings["gme_glob"]))]
    paths += [(path, "terna") for path in sorted(root.glob(settings["terna_glob"]))]
    if not paths:
        raise ValueError("No configured market workbooks were found")
    segments = {value["segment"]: key for key, value in DATASETS.items()}
    for number, (path, provider) in enumerate(paths, 1):
        payload = path.read_bytes()
        if provider == "gme":
            match = re.fullmatch(r"(\d{8})_(\d{8})_(MGP|MI-A[123])_PrezziZonali_Nord\.xlsx", path.name)
            if match is None:
                raise ValueError(f"Unsupported GME workbook name: {path.name}")
            first = pd.Timestamp(datetime.strptime(match[1], "%Y%m%d").date())
            stop = pd.Timestamp(datetime.strptime(match[2], "%Y%m%d").date()) + pd.Timedelta(days=1)
            dataset = segments[match[3]]
            header = _read_sheet(payload, gme["sheet_xml"])[0]
            minutes = 15 if "Periodo" in header else 60
            columns = {**gme["columns"], "period": "Periodo" if minutes == 15 else "Ora"}
            if minutes == 15:
                columns["hour"] = "Ora"
            policy = {**gme, "columns": columns, "zone": settings["zone"],
                      "timezone": settings["timezone"], "date_start": str(first.date()),
                      "date_end_exclusive": str(stop.date()), "native_interval_minutes": minutes}
            if dataset == "ida3_prices":
                policy["eligible_start_clock"] = settings["ida3_start_clock"]
            frame = _parse_workbook(payload, path.name, dataset, policy)
        else:
            match = re.fullmatch(r"Riepilogo_Mensile_Quarto_Orario_(\d{6})\.xlsx", path.name)
            if match is None:
                raise ValueError(f"Unsupported Terna workbook name: {path.name}")
            first = pd.Timestamp(datetime.strptime(match[1], "%Y%m"))
            stop = first + pd.offsets.MonthBegin(1)
            dataset = terna["dataset"]
            frame = parse_terna_workbook(payload, terna,
                {"path": str(path), "date_start": str(first.date()), "date_end_exclusive": str(stop.date())})
        prepared.append({"path": path, "payload": payload, "frame": frame,
                         "provider": provider, "dataset": dataset,
                         "first": first, "stop": stop, "quality": dict(frame.attrs)})
        if progress:
            progress(f"Validated {number}/{len(paths)} workbooks: {path.name}, {len(frame)} native records")
    return prepared


def _coverage(frame, settings, cancellations):
    """Measure every native product, including DST and the half-day IDA3 session."""
    summaries, daily = [], []
    days = pd.date_range(settings["date_start"], settings["date_end_exclusive"], inclusive="left")
    for specification in settings["datasets"]:
        provider, dataset = specification["provider"], specification["dataset"]
        portion = frame.loc[frame.provider.eq(provider) & frame.dataset.eq(dataset)] if not frame.empty else frame
        if not portion.empty:
            unique = _unique_records(portion)
            local = unique.product_start.dt.tz_convert(settings["timezone"])
            durations = (unique.product_end-unique.product_start).dt.total_seconds()/60
            expected_minutes = np.array([native_price_resolution_minutes(dataset, stamp.date())
                                        if provider == "gme" else 15 for stamp in local])
            if (not np.equal(durations, expected_minutes).all()
                    or (local.dt.minute.to_numpy() % expected_minutes != 0).any()
                    or (local.dt.second != 0).any() or (local.dt.microsecond != 0).any()
                    or (local.dt.nanosecond != 0).any()):
                raise ValueError(f"Native resolution or product alignment is invalid for {provider}/{dataset}")
            if "native_interval_minutes" in unique and not unique.native_interval_minutes.eq(durations).all():
                raise ValueError(f"Declared resolution disagrees with product boundaries for {provider}/{dataset}")
            if not unique.unit.eq("EUR/MWh").all() or not unique.zone.eq(settings["zone"]).all():
                raise ValueError("Market history has incompatible zones or price units")
            if dataset == "ida3_prices" and (local.dt.strftime("%H:%M") < settings["ida3_start_clock"]).any():
                raise ValueError("IDA3 history contains ineligible morning products")
            clocks = unique.product_start.dt.tz_convert(settings["timezone"]).dt.strftime("%Y-%m-%d")
            actual = unique.groupby(clocks).size().to_dict()
            observed = unique.loc[unique.price_eur_mwh.notna()].groupby(clocks).size().to_dict()
            resolutions = {str(int(key)): int(value) for key, value in unique.groupby("native_interval_minutes").size().items()}
        else:
            unique, actual, observed, resolutions = portion, {}, {}, {}
        missing = 0
        for day in days:
            key = str(day.date())
            left = day.tz_localize(settings["timezone"])
            right = (day + pd.Timedelta(days=1)).tz_localize(settings["timezone"])
            minutes = native_price_resolution_minutes(dataset, day.date()) if provider == "gme" else 15
            grid = pd.date_range(left, right, freq=f"{minutes}min", inclusive="left")
            if dataset == "ida3_prices":
                grid = grid[grid.strftime("%H:%M") >= settings["ida3_start_clock"]]
            cancelled = key in cancellations.get((provider, dataset), set())
            count, valued = int(actual.get(key, 0)), int(observed.get(key, 0))
            gap = 0 if cancelled and count == 0 else max(0, len(grid) - valued)
            missing += gap
            daily.append({"provider": provider, "dataset": dataset, "delivery_day": key,
                          "native_interval_minutes": minutes, "expected_products": len(grid),
                          "records": count, "observed_prices": valued, "missing_prices": gap,
                          "status": "cancelled" if cancelled and count == 0 else "complete" if count == valued == len(grid) else "gap"})
        summaries.append({"provider": provider, "dataset": dataset, "records": len(portion),
            "unique_products": len(unique), "duplicate_products": len(portion)-len(unique),
            "resolutions_minutes": resolutions, "missing_prices_in_requested_window": missing,
            "first_start_utc": None if unique.empty else unique.product_start.min().isoformat(),
            "last_end_utc_exclusive": None if unique.empty else unique.product_end.max().isoformat()})
    return summaries, pd.DataFrame(daily)


def _plot_coverage(daily, path):
    """Save monthly observed-price coverage without treating cancellations as prices."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    work = daily.assign(month=daily.delivery_day.str[:7])
    grouped = work.groupby(["dataset", "month"])[["observed_prices", "expected_products"]].sum()
    ratio = (grouped.observed_prices / grouped.expected_products).unstack("month")
    fig, ax = plt.subplots(figsize=(14, 4), layout="constrained")
    artist = ax.imshow(ratio.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap="YlGn")
    ax.set_xticks(range(len(ratio.columns)), ratio.columns, rotation=65)
    ax.set_yticks(range(len(ratio.index)), ratio.index)
    ax.set_title("NORD workbook coverage: observed prices / expected native products")
    fig.colorbar(artist, ax=ax, label="Fraction; cancellations and gaps are not filled")
    fig.savefig(path, dpi=140)
    plt.close(fig)


def stage_market_workbooks(config, progress=None):
    """Create a separate verified DuckDB package without opening the live database."""
    settings = _settings(config)
    prepared = _prepare_sources(settings, progress)
    frames = [source["frame"] for source in prepared if not source["frame"].empty]
    if not frames:
        raise ValueError("All workbook products are unresolved; no usable intervals were staged")
    combined = pd.concat(frames, ignore_index=True)
    # Choose a lossless timestamp unit before Parquet serialization, including NaT-only fields.
    for name in combined:
        if pd.api.types.is_datetime64_any_dtype(combined[name].dtype):
            combined[name] = combined[name].dt.as_unit("ns")
    canonical = _unique_records(combined)
    repeated = combined.loc[combined.duplicated(KEYS, keep=False)]
    duplicates = [{"provider": key[0], "dataset": key[1], "zone": key[2],
                   "product_start": str(key[3]), "product_end": str(key[4]),
                   "sources": group[["source_file", "source_row", "source_price_text"]].to_dict("records")}
                  for key, group in repeated.groupby(KEYS, dropna=False)]
    run = datetime.now(timezone.utc).strftime("workbooks-%Y%m%dT%H%M%S%fZ")
    directory = Path(settings["stage_directory"]) / run
    database = directory / "market.duckdb"
    if database.resolve() == Path(settings["target_database"]).resolve():
        raise ValueError("Staging must use a different database from the live downloader")
    directory.mkdir(parents=True, exist_ok=False)
    cache = DuckDBCache(database, threads=settings["duckdb_threads"])
    sources, cancellations = [], {}
    with cache.atomic_write():
        for source in prepared:
            path, provider, dataset = source["path"], source["provider"], source["dataset"]
            frame = canonical.loc[canonical.source_file.eq(path.name)].reset_index(drop=True)
            frame.attrs = source["quality"]
            cancelled = source["quality"].get("cancelled_delivery_days", [])
            cancellations.setdefault((provider, dataset), set()).update(cancelled)
            identity = {"provider": provider, "dataset": dataset, "zone": settings["zone"],
                "start": source["first"].tz_localize(settings["timezone"]).tz_convert("UTC").isoformat(),
                "end": source["stop"].tz_localize(settings["timezone"]).tz_convert("UTC").isoformat(),
                "source_kind": f"manual_{provider}_xlsx", "source_file": path.name,
                "normalization_schema": "native-workbooks-v2"}
            manifest = cache.save(identity, source["payload"], frame,
                {"source_kind": identity["source_kind"], "source_file": path.name,
                 "source_sha256": hashlib.sha256(source["payload"]).hexdigest(),
                 "source_path_configured": str(path), "timezone": settings["timezone"],
                 "historical_feature_eligible": False, "cancelled_delivery_days": cancelled,
                 "source_records": len(source["frame"]), "canonical_records": len(frame),
                 "quality": source["quality"]})
            restored, _ = cache.load(identity)
            pd.testing.assert_frame_equal(frame, restored, check_exact=True)
            sources.append({"file": str(path), "provider": provider, "dataset": dataset,
                "source_sha256": manifest["raw_sha256"], "snapshot_id": manifest["snapshot_id"],
                "source_records": len(source["frame"]), "canonical_records": len(frame),
                "cancelled_delivery_days": cancelled,
                "excluded_other_zone_records": source["quality"].get("excluded_other_zone_records", 0),
                "quarantined_delivery_days": source["quality"].get("quarantined_delivery_days", []),
                "decoded_second_occurrence_records": source["quality"].get("decoded_second_occurrence_records", 0),
                "quarantined_source_rows": len(source["quality"].get("quarantined_source_rows", []))})
        # Check all native rows after the final schema promotion, before committing.
        for entry in cache.entries():
            cache.verify_native_snapshot(entry["identity"], snapshot_id=entry["manifest"]["snapshot_id"])
    coverage, daily = _coverage(canonical, settings, cancellations)
    daily.to_csv(directory / "daily_coverage.csv", index=False)
    _plot_coverage(daily, directory / "monthly_coverage.png")
    _atomic_bytes(directory / "duplicates.json", _json_bytes(duplicates))
    quarantine = [{"source_file": source["path"].name,
                   "source_sha256": hashlib.sha256(source["payload"]).hexdigest(),
                   "delivery_days": source["quality"].get("quarantined_delivery_days", []),
                   "source_rows": source["quality"].get("quarantined_source_rows", [])}
                  for source in prepared if source["quality"].get("quarantined_source_rows")]
    _atomic_bytes(directory / "quarantine.json", _json_bytes(quarantine))
    generated = {"include": os.path.relpath(Path("configs/standard.yaml").resolve(), directory.resolve()),
                 "manual_market_import": {**settings, "stage_path": str(database.resolve()),
                     "check_report_path": str((directory / "postcheck.json").resolve())}}
    (directory / "import.yaml").write_text(yaml.safe_dump(generated, sort_keys=False))
    quoted_config = shlex.quote(str((directory / "import.yaml").resolve()))
    (directory / "NEXT_STEPS.txt").write_text(
        "Run from the repository root after the downloader has stopped.\n"
        "The first command merges valid rows transactionally; the second audits the entire requested history.\n"
        "A nonzero check exit means gaps, duplicate/conflicting products, or invalid stored data require review.\n\n"
        f"PYTHONPATH=src python -m bess_pv_trading merge-market-workbooks --config {quoted_config}\n"
        f"PYTHONPATH=src python -m bess_pv_trading check-market-history --config {quoted_config}\n")
    report = {"status": "staged", "live_database_opened": False, "database": str(database.resolve()),
        "target_database": settings["target_database"], "import_config": str((directory / "import.yaml").resolve()),
        "date_start": settings["date_start"], "date_end_exclusive": settings["date_end_exclusive"],
        "workbooks": len(prepared), "records": len(canonical),
        "identical_duplicate_rows_removed": len(combined)-len(canonical),
        "final_schema_native_parity_verified_snapshots": len(sources),
        "sources": sources, "datasets": coverage,
        "quarantined_rows": sum(item["quarantined_source_rows"] for item in sources),
        "quarantine_report": str((directory / "quarantine.json").resolve()),
        "coverage_scope": "Excel files only; existing January-March IDA history is checked after the live downloader stops",
        "historical_availability": "unknown; no publication clocks or prices manufactured",
        "report": str((directory / "manifest.json").resolve())}
    _atomic_bytes(directory / "manifest.json", _json_bytes(report))
    return report


def merge_market_workbooks(config):
    """Merge valid staged products atomically; never run alongside the downloader."""
    settings = _settings(config)
    if not settings["stage_path"]:
        raise ValueError("Use the import.yaml generated by stage-market-workbooks")
    stage_path, target_path = Path(settings["stage_path"]), Path(settings["target_database"])
    if stage_path.resolve() == target_path.resolve() or not stage_path.is_file():
        raise ValueError("A separate existing staging database is required")
    if not target_path.is_file():
        raise ValueError("The target database must already exist; refusing a new empty history")
    stage = DuckDBCache(stage_path, threads=settings["duckdb_threads"])
    prepared = []
    with stage.read_session():
        for entry in stage.entries():
            frame, manifest = stage.load_snapshot(entry["identity"], snapshot_id=entry["manifest"]["snapshot_id"])
            stage.verify_native_snapshot(entry["identity"], snapshot_id=manifest["snapshot_id"])
            prepared.append((entry["identity"], frame, manifest, stage.load_raw(entry["identity"])))
    target = DuckDBCache(target_path, threads=settings["duckdb_threads"])
    scopes, imported, reused = {}, 0, 0
    with target.atomic_write():
        allowed = {(item["provider"], item["dataset"], settings["zone"]) for item in settings["datasets"]}
        def protected_snapshots():
            """Keep ENTSO-E and every unselected dataset's current pointers unchanged."""
            return {entry["manifest"]["snapshot_id"] for entry in target.entries()
                    if tuple(entry["identity"][name] for name in ("provider", "dataset", "zone")) not in allowed}
        protected = protected_snapshots()
        for identity, frame, manifest, payload in prepared:
            key = tuple(identity[name] for name in ("provider", "dataset", "zone"))
            if key not in scopes:
                scopes[key] = _read_scope(target, *key)[0]
            old = target.manifest(identity)
            if old is not None and old.get("metadata", {}).get("staged_snapshot_id") == manifest["snapshot_id"]:
                target.load(identity)
                reused += len(frame)
                continue
            novel = _novel_records(scopes[key], frame)
            metadata = {**manifest["metadata"], "staged_snapshot_id": manifest["snapshot_id"],
                        "identical_existing_products_reused": len(frame)-len(novel)}
            saved = target.save(identity, payload, novel, metadata)
            target.verify_native_snapshot(identity, snapshot_id=saved["snapshot_id"])
            imported += len(novel)
            reused += len(frame)-len(novel)
            if not novel.empty:
                scopes[key] = pd.concat([scopes[key], novel], ignore_index=True)
        if protected_snapshots() != protected:
            raise ValueError("An unrelated source snapshot changed; manual merge rolled back")
    return {"status": "merged", "database": str(target_path), "new_records": imported,
            "identical_records_reused": reused, "unrelated_snapshots_preserved": len(protected),
            "postcheck": check_market_history(config)}


def check_market_history(config):
    """Audit full history after acquisition, including the preexisting Q1 IDA data."""
    settings = _settings(config)
    target = DuckDBCache(settings["target_database"], threads=settings["duckdb_threads"])
    if not target.directory.is_file():
        raise ValueError("The market database is missing")
    frames, cancellations, checked = [], {}, []
    with target.read_session():
        for scope in settings["datasets"]:
            provider, dataset = scope["provider"], scope["dataset"]
            frame, entries = _read_scope(target, provider, dataset, settings["zone"])
            if not frame.empty:
                _unique_records(frame)
                frames.append(frame)
            for entry in entries:
                manifest = entry["manifest"]
                target.verify_native_snapshot(entry["identity"], snapshot_id=manifest["snapshot_id"])
                cancellations.setdefault((provider, dataset), set()).update(
                    manifest.get("metadata", {}).get("cancelled_delivery_days", []))
                checked.append(manifest["snapshot_id"])
    combined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    coverage, daily = _coverage(combined, settings, cancellations)
    parity = None
    if settings["stage_path"]:
        stage = DuckDBCache(settings["stage_path"], threads=settings["duckdb_threads"])
        with stage.read_session():
            staged_frames = [_read_scope(stage, scope["provider"], scope["dataset"], settings["zone"])[0]
                             for scope in settings["datasets"]]
        expected = pd.concat([frame for frame in staged_frames if not frame.empty], ignore_index=True)
        for _, frame in expected.groupby(["provider", "dataset", "zone"]):
            first = frame.iloc[0]
            actual = combined.loc[combined.provider.eq(first.provider) & combined.dataset.eq(first.dataset) & combined.zone.eq(first.zone)]
            if not _novel_records(actual, frame).empty:
                raise ValueError("Postcheck: staged price products are absent from the target")
        parity = {"status": "identical", "staged_products": len(expected)}
    gaps = sum(item["missing_prices_in_requested_window"] for item in coverage)
    duplicates = sum(item["duplicate_products"] for item in coverage)
    report = {"status": "passed" if not gaps and not duplicates else "gaps_or_duplicates",
        "database": str(target.directory), "date_start": settings["date_start"],
        "date_end_exclusive": settings["date_end_exclusive"], "datasets": coverage,
        "snapshot_checksums_and_native_rows_verified": len(checked), "snapshot_ids": checked,
        "staged_value_and_interval_parity": parity, "missing_prices": gaps, "duplicate_products": duplicates,
        "focus_delivery_days": daily.loc[daily.delivery_day.isin(settings["audit_delivery_days"])].to_dict("records"),
        "fill_policy": "none; cancellations, missing files and unresolved DST remain explicit"}
    path = Path(settings["check_report_path"])
    path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(path.with_name(path.stem + "_daily_coverage.csv"), index=False)
    _atomic_bytes(path, _json_bytes(report))
    report["report"] = str(path)
    return report
