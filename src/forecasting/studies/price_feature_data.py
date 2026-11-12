"""Prepare native-product price matrices and matched feature-selection cohorts."""

from __future__ import annotations

from datetime import date, timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .price_search_history import PriceForecastSettings, _features, historical_decision_times


def digest(path):
    """Fingerprint a prepared file in bounded chunks."""
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def day_clocks(starts, clock, offset, timezone):
    """Construct one explicit market clock for each local delivery date."""
    days = starts.tz_convert(timezone).date
    mapping = {day: pd.Timestamp(f"{day + timedelta(days=offset)} {clock}", tz=timezone).tz_convert("UTC")
               for day in np.unique(days)}
    return pd.DatetimeIndex([mapping[day] for day in days]).as_unit("ns")


def interval_values(frame, grid, values):
    """Carry an interval mean within its recorded interval, preserving source gaps."""
    values = np.asarray(values)
    integer = np.issubdtype(values.dtype, np.integer)
    result = np.full(len(grid), np.iinfo(np.int64).max if integer else np.nan,
                     dtype=np.int64 if integer else float)
    if frame.empty:
        return result
    starts = pd.DatetimeIndex(pd.to_datetime(frame.product_start, utc=True)).as_unit("ns")
    ends = pd.DatetimeIndex(pd.to_datetime(frame.product_end, utc=True)).as_unit("ns")
    if starts.has_duplicates or not starts.is_monotonic_increasing or (ends <= starts).any():
        raise ValueError("Source intervals must be unique, ordered and positive")
    if len(frame) > 1 and (starts[1:] < ends[:-1]).any():
        raise ValueError("Source intervals overlap; resolve revisions before feature preparation")
    parents = starts.searchsorted(grid, side="right") - 1
    valid = parents >= 0
    valid[valid] &= grid[valid] < ends[parents[valid]]
    result[valid] = values[parents[valid]]
    return result


def italian_holidays(starts, settings):
    """Generate configured national fixed dates and Gregorian Easter holidays."""
    local = starts.tz_convert(settings["timezone"])
    fixed = set(settings["fixed_holidays"])
    holidays = set()
    if settings.get("easter_holidays", True):
        # Gregorian computus is deterministic and needs no downloaded calendar package.
        for year in np.unique(local.year):
            year = int(year)
            a = year % 19
            b, c = divmod(year, 100)
            d, e = divmod(b, 4)
            f = (b + 8) // 25
            g = (b - f + 1) // 3
            h = (19 * a + b - d - g + 15) % 30
            i, k = divmod(c, 4)
            l = (32 + 2 * e + 2 * i - h - k) % 7
            m = (a + 11 * h + 22 * l) // 451
            month, remainder = divmod(h + l - 7 * m + 114, 31)
            easter = date(year, month, remainder + 1)
            holidays.update([easter, easter + timedelta(days=1)])
    return np.array([day.strftime("%m-%d") in fixed or day in holidays for day in local.date], dtype=float)


def _price_sources(frame, settings):
    """Assign declared clearing clocks while retaining every native target interval."""
    sources = {}
    for market, auction in settings["auctions"].items():
        selected = frame.loc[frame.dataset.eq(auction["dataset"])].copy()
        selected = selected.loc[np.isfinite(pd.to_numeric(selected.value, errors="coerce"))]
        selected = selected.sort_values("product_start").reset_index(drop=True)
        if selected.empty or selected.product_start.duplicated().any():
            raise ValueError(f"Missing or duplicate native price products for {market}")
        starts = pd.DatetimeIndex(pd.to_datetime(selected.product_start, utc=True)).as_unit("ns")
        fallback = day_clocks(starts, auction["publication_time"], auction["publication_day_offset"], settings["timezone"])
        recorded = pd.to_datetime(selected.get("published_at", pd.Series(pd.NaT, index=selected.index)), utc=True)
        selected["available_at"] = recorded.where(recorded.notna(), pd.Series(fallback, index=selected.index))
        selected["price_eur_per_mwh"] = selected.value.to_numpy(float)
        sources[market] = selected
    return sources


def _feature_histories(sources, grid, minutes):
    """Align historical prices for lag features; training labels remain unresampled."""
    histories = {}
    step = minutes // 15
    for market, source in sources.items():
        values = interval_values(source, grid, source.price_eur_per_mwh.to_numpy(float))
        clocks = interval_values(source, grid, pd.DatetimeIndex(source.available_at).as_unit("ns").asi8)
        usable = len(grid) // step * step
        grouped = values[:usable].reshape(-1, step)
        finite = np.isfinite(grouped).all(axis=1)
        means = grouped.mean(axis=1)
        availability = clocks[:usable].reshape(-1, step).max(axis=1)
        starts = grid[:usable:step]
        histories[market] = pd.DataFrame({"product_start": starts[finite],
            "product_end": starts[finite] + pd.Timedelta(minutes=minutes),
            "available_at": pd.to_datetime(availability[finite], utc=True),
            "price_eur_per_mwh": means[finite]})
    return histories


def _profiles(history, starts, cutoffs, lag_days, minutes, hours, timezone):
    """Join prior local-day hourly curves, retaining missing and DST-hour profiles."""
    local = pd.DatetimeIndex(history.product_start).tz_convert(timezone)
    grouped = history.assign(day=local.strftime("%Y-%m-%d"), hour=local.hour).groupby(["day", "hour"])
    table = grouped.agg(price=("price_eur_per_mwh", "mean"), count=("price_eur_per_mwh", "size"),
                        publication=("available_at", "max"))
    table.loc[table["count"].ne(60 // minutes), "price"] = np.nan
    prices = table.price.unstack("hour").reindex(columns=hours)
    clocks = table.publication.unstack("hour").reindex(index=prices.index, columns=hours)
    arrays, names = [], []
    for lag in lag_days:
        wanted = [(day - timedelta(days=lag)).isoformat() for day in starts.tz_convert(timezone).date]
        values = prices.reindex(wanted).to_numpy(float, copy=True)
        joined_clocks = clocks.reindex(wanted)
        for column in range(len(hours)):
            available = pd.DatetimeIndex(pd.to_datetime(joined_clocks.iloc[:, column], utc=True)).as_unit("ns")
            values[:, column][available.isna() | (available > cutoffs)] = np.nan
        arrays.append(values)
        names.extend([f"own_curve_{lag}d_hour_{hour}" for hour in hours])
    return np.column_stack(arrays) if arrays else np.empty((len(starts), 0)), names


def _lookup(history, starts, cutoffs):
    """Look up a price only when its own clearing clock precedes the decision."""
    index = pd.DatetimeIndex(history.product_start)
    positions = index.get_indexer(starts)
    valid = positions >= 0
    availability = pd.DatetimeIndex(history.available_at).as_unit("ns")
    valid[valid] &= availability[positions[valid]] <= cutoffs[valid]
    values = np.full(len(starts), np.nan)
    values[valid] = history.price_eur_per_mwh.to_numpy(float)[positions[valid]]
    return values


def _external_grid(frames, grid, settings):
    """Extract configured national features once, then reuse them for every market."""
    arrays, names, indices = [], [], {}
    parsed_json = {}
    for block, specification in settings["feature_blocks"].items():
        indices[block] = []
        for feature in specification["features"]:
            frame = frames[feature["source"]]
            selected = np.ones(len(frame), dtype=bool)
            for key, value in feature.get("selectors", {}).items():
                selected &= frame[key].eq(value).to_numpy()
            source = frame.loc[selected].copy()
            if source.empty:
                raise ValueError(f"Feature {feature['name']} has no selected source rows")
            column = feature["column"]
            if feature.get("component"):
                key = (feature["source"], column)
                if key not in parsed_json:
                    parsed_json[key] = frame[column].map(lambda raw: json.loads(raw) if pd.notna(raw) else {})
                values = pd.to_numeric(parsed_json[key].loc[selected].map(
                    lambda item: item.get(feature["component"], np.nan)), errors="raise").to_numpy(float)
            else:
                values = pd.to_numeric(source[column], errors="raise").to_numpy(float)
            if feature.get("kind") == "daily_gas":
                days = pd.to_datetime(source.gas_delivery_day)
                offset = pd.DateOffset(days=int(feature.get("lag_days", 0)))
                source["product_start"] = pd.DatetimeIndex(days).tz_localize(settings["timezone"]) + offset
                source["product_end"] = pd.DatetimeIndex(days + pd.Timedelta(days=1)).tz_localize(settings["timezone"]) + offset
            else:
                shift = pd.Timedelta(hours=feature.get("lag_hours", 0))
                source["product_start"] = pd.to_datetime(source.product_start, utc=True) + shift
                source["product_end"] = pd.to_datetime(source.product_end, utc=True) + shift
            order = np.argsort(pd.DatetimeIndex(source.product_start).asi8, kind="stable")
            source = source.iloc[order]
            aligned = interval_values(source, grid, values[order])
            indices[block].append(len(arrays))
            arrays.append(aligned)
            names.append(feature["name"])
    if len(names) != len(set(names)):
        raise ValueError("External feature names must be unique")
    return np.column_stack(arrays), names, indices


def _base_matrix(history, histories, starts, settings, group):
    """Build causal price lags, calendar, curves and earlier-auction neighbors."""
    auction = settings["auctions"][group["market"]]
    lags = auction.get("lag_hours", settings["lag_hours"])
    model_settings = PriceForecastSettings(lag_hours=tuple(lags),
        max_lag_age_hours=settings["max_lag_age_hours"], daily_lag_fallback="target_clock",
        interval_minutes=group["native_minutes"], timezone=settings["timezone"],
        decision_local_time=auction["decision_time"], decision_day_offset=auction["decision_day_offset"])
    cutoffs = historical_decision_times(starts, model_settings)
    matrix, fallback_count = _features(history, starts, cutoffs, model_settings)
    names = [f"own_price_lag_{lag}h" for lag in lags]
    names += ["hour_sin", "hour_cos", "weekday_sin", "weekday_cos", "weekend", "annual_sin", "annual_cos"]
    holiday = italian_holidays(starts, settings)
    matrix = np.column_stack([matrix, holiday])
    names.append("italian_public_holiday")
    curves, curve_names = _profiles(history, starts, cutoffs, settings["daily_curve_lag_days"],
        group["native_minutes"], list(range(auction["delivery_start_hour"], auction["delivery_end_hour"])), settings["timezone"])
    matrix = np.column_stack([matrix, curves])
    names.extend(curve_names)
    anchor = np.zeros(len(starts))
    for market in auction["earlier_markets"]:
        values = _lookup(histories[market], starts, cutoffs)
        matrix = np.column_stack([matrix, values])
        names.append(f"cleared_{market}")
        anchor = values.copy()
        for offset in settings["near_auction_offsets_products"]:
            wanted = starts + pd.Timedelta(minutes=offset * group["native_minutes"])
            crossing = wanted.tz_convert(settings["timezone"]).date != starts.tz_convert(settings["timezone"]).date
            wanted = pd.DatetimeIndex(np.where(crossing, starts.asi8, wanted.asi8), tz="UTC")
            matrix = np.column_stack([matrix, _lookup(histories[market], wanted, cutoffs)])
            names.append(f"cleared_{market}_neighbor_{offset}")
    for market in auction["past_other_markets"]:
        for lag in settings["past_other_market_lag_hours"]:
            values = _lookup(histories[market], starts - pd.Timedelta(hours=lag), cutoffs)
            matrix = np.column_stack([matrix, values])
            names.append(f"past_{market}_{lag}h")
    reference = matrix[:, 0].copy() if not auction["earlier_markets"] else anchor.copy()
    return matrix, names, anchor, reference, cutoffs, fallback_count


def prepare_matrices(settings, source_root, destination):
    """Save read-only arrays and explicit matched train/validation/test cohorts."""
    source_root = Path(source_root)
    source_manifest = json.loads((source_root / "manifest.json").read_text())
    selection = source_manifest["selection"]
    first = pd.Timestamp(selection["start"], tz=selection["timezone"]).tz_convert("UTC")
    stop = pd.Timestamp(selection["end_exclusive"], tz=selection["timezone"]).tz_convert("UTC")
    grid = pd.date_range(first, stop, freq="15min", inclusive="left").as_unit("ns")
    frames = {name: pd.read_parquet(source_root / details["path"]) for name, details in source_manifest["tables"].items()}
    external, external_names, block_indices = _external_grid(frames, grid, settings)
    prices = _price_sources(frames["nord_auctions"], settings)
    feature_histories = {}
    groups = {}
    for group_name, group in settings["product_groups"].items():
        minutes = int(group["native_minutes"])
        if minutes not in feature_histories:
            feature_histories[minutes] = _feature_histories(prices, grid, minutes)
        histories = feature_histories[minutes]
        native = prices[group["market"]]
        duration = (native.product_end - native.product_start).dt.total_seconds() / 60
        local = pd.DatetimeIndex(native.product_start).tz_convert(settings["timezone"])
        days = local.strftime("%Y-%m-%d").to_numpy()
        # Retain additional observed targets for proxy export without extending the test split.
        native_end = group.get("native_end_exclusive", group["test_end"])
        selected = duration.eq(minutes).to_numpy() & (days >= group["train_start"]) & (days < native_end)
        native = native.loc[selected].reset_index(drop=True)
        if native.empty:
            raise ValueError(f"No native target rows for {group_name}")
        starts = pd.DatetimeIndex(native.product_start).as_unit("ns")
        days = starts.tz_convert(settings["timezone"]).strftime("%Y-%m-%d").to_numpy()
        matrix, names, anchor, reference, cutoffs, fallback_count = _base_matrix(
            histories[group["market"]], histories, starts, settings, group)
        # Aggregate exogenous interval means to a native product only when all quarters exist.
        source_positions = grid.get_indexer(starts)
        offsets = np.arange(minutes // 15)
        positions = source_positions[:, None] + offsets
        if (source_positions < 0).any() or (positions >= len(grid)).any():
            raise ValueError("Configured source extraction does not cover every native target interval")
        external_native = external[positions].mean(axis=1)
        width = matrix.shape[1]
        matrix = np.column_stack([matrix, external_native]).astype(np.float32)
        names.extend(external_names)
        directory = Path(destination) / "matrices" / group_name
        directory.mkdir(parents=True, exist_ok=True)
        arrays = {"x": matrix, "y": native.value.to_numpy(float), "anchor": anchor,
                  "reference": reference, "starts": starts.asi8, "decisions": cutoffs.asi8,
                  "publication": pd.DatetimeIndex(native.available_at).as_unit("ns").asi8,
                  "days": days.astype("U10")}
        files = {}
        for name, values in arrays.items():
            path = directory / f"{name}.npy"
            np.save(path, values, allow_pickle=False)
            files[name] = {"path": str(path.relative_to(destination)), "sha256": digest(path)}
        mapped_blocks = {name: [width + index for index in indices] for name, indices in block_indices.items()}
        cohorts = {}
        for cohort_name, blocks in settings["cohorts"].items():
            columns = list(range(width)) + [column for block in blocks for column in mapped_blocks[block]]
            complete = np.isfinite(matrix[:, columns]).all(axis=1) & np.isfinite(anchor) & np.isfinite(reference)
            validation = complete & (days >= group["validation_start"]) & (days < group["validation_end"])
            test = complete & (days >= group["test_start"]) & (days < group["test_end"])
            embargo = timedelta(days=int(settings["embargo_days"]))
            training_end = (date.fromisoformat(group["train_end"]) - embargo).isoformat()
            final_end = (date.fromisoformat(group["test_start"]) - embargo).isoformat()
            training = complete & (days >= group["train_start"]) & (days < training_end)
            final_training = complete & (days >= group["train_start"]) & (days < final_end)
            if validation.any():
                training &= arrays["publication"] <= cutoffs[validation].asi8.min()
            if test.any():
                final_training &= arrays["publication"] <= cutoffs[test].asi8.min()
            eligible_days = np.unique(days[training])
            inner_days = max(int(settings["inner_min_days"]), int(math.ceil(len(eligible_days) * settings["inner_validation_fraction"])))
            inner_validation = training & np.isin(days, eligible_days[-inner_days:])
            inner_training = training & ~inner_validation
            counts = {"train": int(training.sum()), "inner_train": int(inner_training.sum()),
                      "inner_validation": int(inner_validation.sum()), "validation": int(validation.sum()),
                      "test": int(test.sum()), "final_train": int(final_training.sum())}
            eligible = (counts["train"] >= settings["minimum_training_rows"]
                        and min(counts["inner_train"], counts["inner_validation"]) >= settings["minimum_inner_rows"]
                        and min(counts["validation"], counts["test"]) >= settings["minimum_evaluation_rows"])
            cohort = {"eligible": eligible, "blocks": blocks, "counts": counts,
                      "exclusion_policy": "complete cases across every block in this cohort; identical rows for all subsets"}
            if eligible:
                masks = {"train": training, "inner_train": inner_training, "inner_validation": inner_validation,
                         "validation": validation, "test": test, "final_train": final_training}
                cohort["indices"] = {}
                for name, mask in masks.items():
                    path = directory / f"{cohort_name}_{name}.npy"
                    np.save(path, np.flatnonzero(mask), allow_pickle=False)
                    cohort["indices"][name] = {"path": str(path.relative_to(destination)), "sha256": digest(path)}
            else:
                cohort["skip_reason"] = "Insufficient matched rows for one or more chronological splits"
            cohorts[cohort_name] = cohort
        if not any(cohort["eligible"] for cohort in cohorts.values()):
            raise ValueError(f"Configured product group {group_name} has no eligible cohort: "
                             f"{ {name: cohort['counts'] for name, cohort in cohorts.items()} }")
        groups[group_name] = {"market": group["market"], "native_minutes": minutes, "split": group,
            "rows": len(native), "base_width": width, "feature_names": names, "block_columns": mapped_blocks,
            "files": files, "cohorts": cohorts, "lag_fallback_count": fallback_count,
            "base_missing_rows": int((~np.isfinite(matrix[:, :width]).all(axis=1)).sum())}
        print(f"Prepared {group_name}: {len(native)} native targets; "
              f"eligible cohorts {[name for name, cohort in cohorts.items() if cohort['eligible']]}", flush=True)
    return groups
