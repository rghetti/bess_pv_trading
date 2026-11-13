"""Gate-aligned daily records for rolling full-curve price forecasting.

Input files are immutable exports of the market catalog. Weather run timestamps
describe initialization, and their configured publication lag is an assumption.
"""

from __future__ import annotations

from bisect import bisect_right
from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import load_config
from .price_feature_data import _external_grid, _price_sources, digest, italian_holidays


def delivery_grid(day, minutes, timezone, first_hour=0):
    """Enumerate real UTC products between local delivery boundaries, including DST."""
    local_day = pd.Timestamp(day)
    first = (local_day + pd.Timedelta(hours=first_hour)).tz_localize(timezone)
    end = (local_day + pd.Timedelta(days=1)).tz_localize(timezone)
    return pd.date_range(first, end, freq=f"{minutes}min", inclusive="left").tz_convert("UTC")


def slot_numbers(starts, timezone):
    """Map local quarter-hours to 96 positions plus four repeated-hour positions."""
    local = pd.DatetimeIndex(starts).tz_convert(timezone)
    used = set()
    slots = []
    for stamp in local:
        position = stamp.hour * 4 + stamp.minute // 15
        if position in used:
            if stamp.hour != 2:
                raise ValueError("Unexpected repeated local quarter hour")
            position = 96 + stamp.minute // 15
        if position in used:
            raise ValueError("Duplicate local product slot")
        used.add(position)
        slots.append(position)
    return np.asarray(slots, dtype=int)


def pad_curve(starts, values, timezone):
    """Place a local-day profile into its fixed 100-slot feature representation."""
    result = np.full(100, np.nan)
    result[slot_numbers(starts, timezone)] = np.asarray(values, dtype=float)
    return result


class PriceIndex:
    """Keep sorted native auction intervals with per-product publication clocks."""

    def __init__(self, frame):
        source = frame.sort_values("product_start")
        self.starts = pd.DatetimeIndex(pd.to_datetime(source.product_start, utc=True)).as_unit("ns").asi8
        self.ends = pd.DatetimeIndex(pd.to_datetime(source.product_end, utc=True)).as_unit("ns").asi8
        self.published = pd.DatetimeIndex(pd.to_datetime(source.available_at, utc=True)).as_unit("ns").asi8
        self.values = source.price_eur_per_mwh.to_numpy(float)
        if len(self.starts) > 1 and (self.starts[1:] < self.ends[:-1]).any():
            raise ValueError("Overlapping price intervals must be resolved before preparing examples")

    def lookup(self, starts, cutoff=None):
        """Repeat hourly input prices within their native interval; mask unpublished data."""
        query = pd.DatetimeIndex(starts).as_unit("ns").asi8
        positions = np.searchsorted(self.starts, query, side="right") - 1
        valid = positions >= 0
        valid[valid] &= query[valid] < self.ends[positions[valid]]
        if cutoff is not None:
            valid[valid] &= self.published[positions[valid]] <= pd.Timestamp(cutoff).value
        values = np.full(len(query), np.nan)
        values[valid] = self.values[positions[valid]]
        return values

    def latest_publication(self, starts, fallback):
        """Use the latest observed target publication; retain schedule if no rows exist."""
        query = pd.DatetimeIndex(starts).as_unit("ns").asi8
        positions = np.searchsorted(self.starts, query, side="right") - 1
        valid = positions >= 0
        valid[valid] &= query[valid] < self.ends[positions[valid]]
        clocks = self.published[positions[valid]]
        clocks = clocks[clocks != pd.NaT.value]
        return max(pd.Timestamp(fallback), pd.Timestamp(int(clocks.max()), tz="UTC")) if len(clocks) else pd.Timestamp(fallback)


def _feature_blocks(frames, grid, base):
    """Align declared national features once, preserving gaps in each block."""
    base = dict(base)
    base["feature_blocks"] = {name: base["feature_blocks"][name] for name in
                              ("nord_forecasts", "gas", "france", "switzerland")}
    matrix, names, indices = _external_grid(frames, grid, base)
    return matrix, names, indices


class WeatherIndex:
    """Index each city, variable and run separately to avoid mixing forecast vintages."""

    def __init__(self, weather, settings):
        self.options = settings
        self.runs = {}
        self.run_times = {}
        self.latest_valid_prefix = {}
        if weather.empty:
            return
        selected = weather.loc[weather.location_id.isin(settings["weather_locations"])
            & weather.variable.isin(settings["weather_variables"])].copy()
        selected["cycle_time"] = pd.to_datetime(selected.cycle_time, utc=True)
        selected["valid_time"] = pd.to_datetime(selected.valid_time, utc=True)
        selected = selected.loc[selected.cycle_time.dt.hour.isin(settings["weather_cycle_hours_utc"])]
        for (city, variable), series in selected.groupby(["location_id", "variable"], sort=False):
            cycles = []
            for cycle, block in series.groupby("cycle_time", sort=True):
                block = block.sort_values("valid_time")
                if block.valid_time.duplicated().any():
                    # Different request configurations can cache the same run.
                    # Collapse identical observations only; conflicting snapshots
                    # require explicit reconciliation before a study can start.
                    duplicates = block.loc[block.valid_time.duplicated(keep=False)]
                    for _, repeated in duplicates.groupby("valid_time"):
                        values = pd.to_numeric(repeated.value, errors="coerce").to_numpy(float)
                        if (not np.allclose(values, values[0], rtol=0, atol=0, equal_nan=True)
                                or repeated.temporal_semantics.nunique(dropna=False) != 1):
                            raise ValueError(f"Conflicting weather timestamps for {city}/{variable}/{cycle}")
                    block = block.drop_duplicates("valid_time", keep="first")
                cycles.append((cycle, pd.DatetimeIndex(block.valid_time).as_unit("ns").asi8,
                    pd.to_numeric(block.value, errors="coerce").to_numpy(float),
                    str(block.temporal_semantics.iloc[0])))
            self.runs[(city, variable)] = cycles
            self.run_times[(city, variable)] = [item[0].value for item in cycles]
            self.latest_valid_prefix[(city, variable)] = np.maximum.accumulate([
                int(item[1][-1]) if len(item[1]) else np.iinfo(np.int64).min for item in cycles
            ])

    def extract(self, starts, cutoff, previous_cutoff, timezone, required_starts=None):
        """Select the newest eligible full-horizon run and expose older-run fallback."""
        profiles, revisions, trace = [], [], {}
        query = pd.DatetimeIndex(starts).as_unit("ns").asi8
        required = query if required_starts is None else pd.DatetimeIndex(required_starts).as_unit("ns").asi8
        required_positions = np.searchsorted(query, required)
        if ((required_positions >= len(query)).any() or
                not np.array_equal(query[required_positions], required)):
            raise ValueError("Weather coverage products must belong to the ordered delivery grid")
        lag = pd.Timedelta(hours=float(self.options["weather_release_lag_hours"]))
        for city in self.options["weather_locations"]:
            for variable in self.options["weather_variables"]:
                runs = self.runs.get((city, variable), [])
                run_times = self.run_times.get((city, variable), [])
                prefix = self.latest_valid_prefix.get((city, variable), [])
                def selected(at):
                    position = bisect_right(run_times, (at - lag).value) - 1
                    if position < 0:
                        return None, np.full(len(starts), np.nan), None, False
                    latest_cycle = runs[position][0]
                    latest_output = np.full(len(starts), np.nan)
                    # Once no earlier run reaches the delivery horizon, stop searching.
                    while position >= 0 and prefix[position] >= required.min():
                        cycle, times, values, semantics = runs[position]
                        valid = np.isfinite(values)
                        out = np.full(len(query), np.nan)
                        if valid.sum() >= 2:
                            times, values = times[valid], values[valid]
                            if semantics == "hourly_mean_ending":
                                pos = np.searchsorted(times, query, side="right")
                                good = (pos < len(times)) & (pos > 0)
                                out[good] = values[pos[good]]
                                good[good] &= query[good] >= times[pos[good]] - 3_600_000_000_000
                                out[~good] = np.nan
                            else:
                                out = np.interp(query, times, values, left=np.nan, right=np.nan)
                                # Never bridge gaps larger than a native hour.
                                pos = np.searchsorted(times, query, side="left")
                                exact = (pos < len(times)) & (times[np.minimum(pos, len(times)-1)] == query)
                                between = (pos > 0) & (pos < len(times))
                                good = exact.copy()
                                good[between] |= (times[pos[between]] - times[pos[between] - 1]) <= 3_600_000_000_000
                                out[~good] = np.nan
                        if cycle == latest_cycle:
                            latest_output = out
                        if np.isfinite(out[required_positions]).all():
                            return cycle, out, latest_cycle, cycle != latest_cycle
                        position -= 1
                    # Preserve the newest partial run for coverage reporting; it
                    # cannot qualify as a complete weather-feature sample.
                    return latest_cycle, latest_output, latest_cycle, False
                current_cycle, current, latest_cycle, fallback = selected(cutoff)
                prior_cycle, prior, prior_latest, prior_fallback = selected(previous_cutoff)
                profiles.extend(pad_curve(starts, current, timezone))
                revisions.extend(pad_curve(starts, current - prior, timezone))
                trace[f"{city}/{variable}"] = {"cycle": None if current_cycle is None else current_cycle.isoformat(),
                    "previous_cycle": None if prior_cycle is None else prior_cycle.isoformat(),
                    "latest_eligible_cycle": None if latest_cycle is None else latest_cycle.isoformat(),
                    "previous_latest_eligible_cycle": None if prior_latest is None else prior_latest.isoformat(),
                    "older_complete_cycle_used": fallback, "previous_older_complete_cycle_used": prior_fallback}
        return np.asarray(profiles), np.asarray(revisions), trace


def build_examples(snapshot_dir, settings):
    """Build native-target examples for DA and each IDA at DA and own gate.

    DA-gate IDA records intentionally expose neither the delivery-day DA clearing
    nor a derived reference. The rolling learner must attach its own earlier
    out-of-sample DA forecast to those records.
    """
    root = Path(snapshot_dir)
    manifest = json.loads((root / "manifest.json").read_text())
    frames = {}
    for name, entry in manifest["tables"].items():
        path = root / entry["path"]
        if digest(path) != entry["sha256"]:
            raise ValueError(f"Pinned source has changed: {path}")
        frames[name] = pd.read_parquet(path)
    base = load_config(settings["base_search_config"])["price_feature_search"]
    timezone = settings["timezone"]
    sources = _price_sources(frames["nord_auctions"], base)
    price = {market: PriceIndex(frame) for market, frame in sources.items()}
    weather = WeatherIndex(frames.get("regional_weather", pd.DataFrame()), settings)
    first = pd.Timestamp(settings["start"], tz=timezone).tz_convert("UTC")
    end = pd.Timestamp(settings["end_exclusive"], tz=timezone).tz_convert("UTC")
    fine_grid = pd.date_range(first, end, freq="15min", inclusive="left")
    external, external_names, external_indices = _feature_blocks(frames, fine_grid, base)
    external_lookup = {stamp: index for index, stamp in enumerate(fine_grid.asi8)}
    notices = {(row["dataset"], row["delivery_day"]) for row in manifest.get("gme_cancellation_notices", [])}
    markets = ("DA", "IDA1", "IDA2", "IDA3")
    results = []
    source_digest = digest(root / "manifest.json")
    for day in pd.date_range(settings["example_start"], settings["end_exclusive"], inclusive="left"):
        day_text = day.strftime("%Y-%m-%d")
        quarter_grid = delivery_grid(day, 15, timezone)
        holiday = bool(italian_holidays(pd.DatetimeIndex([quarter_grid[0]]), base)[0])
        gate_cache = {}
        for market in markets:
            auction = base["auctions"][market]
            minutes = 60 if market == "DA" and day < pd.Timestamp("2025-10-01") else 15
            native_grid = delivery_grid(day, minutes, timezone, auction["delivery_start_hour"])
            target = price[market].lookup(native_grid)
            target_publication = pd.Timestamp(f"{(day + timedelta(days=auction['publication_day_offset'])).date()} {auction['publication_time']}", tz=timezone).tz_convert("UTC")
            target_publication = price[market].latest_publication(native_grid, target_publication)
            for gate in (["DA"] if market == "DA" else ["DA", market]):
                gate_config = base["auctions"][gate]
                cutoff = pd.Timestamp(f"{(day + timedelta(days=gate_config['decision_day_offset'])).date()} {gate_config['decision_time']}", tz=timezone).tz_convert("UTC")
                da_gate = base["auctions"]["DA"]
                da_cutoff = pd.Timestamp(f"{(day + timedelta(days=da_gate['decision_day_offset'])).date()} {da_gate['decision_time']}", tz=timezone).tz_convert("UTC")
                known = [] if gate == "DA" else auction["earlier_markets"]
                if gate not in gate_cache:
                    # Build a gate's inputs once; several future target heads reuse them.
                    core = []
                    historical_cancellations = []
                    for prior_market in markets:
                        for lag_days in (1, 2, 7):
                            previous = delivery_grid(day - pd.Timedelta(days=lag_days), 15, timezone)
                            previous_day = day - pd.Timedelta(days=lag_days)
                            previous_auction = base["auctions"][prior_market]
                            notice_clock = pd.Timestamp(
                                f"{(previous_day + timedelta(days=previous_auction['publication_day_offset'])).date()} {previous_auction['publication_time']}",
                                tz=timezone).tz_convert("UTC")
                            cancelled = ((previous_auction["dataset"], previous_day.strftime("%Y-%m-%d")) in notices
                                         and notice_clock <= cutoff)
                            values = pad_curve(previous, price[prior_market].lookup(previous, cutoff), timezone)
                            if cancelled:
                                # Replace only a documented cancellation with the last published prior curve.
                                for back in range(1, int(settings.get("max_cancellation_fallback_days", 14)) + 1):
                                    replacement_day = previous_day - pd.Timedelta(days=back)
                                    replacement = delivery_grid(replacement_day, 15, timezone)
                                    candidate = pad_curve(replacement, price[prior_market].lookup(replacement, cutoff), timezone)
                                    if np.isfinite(candidate).any():
                                        values = candidate
                                        break
                            core.extend(values)
                            historical_cancellations.append(float(cancelled))
                    for index, previous_market in enumerate(markets[:3]):
                        values = price[previous_market].lookup(quarter_grid, cutoff) if previous_market in known else np.full(len(quarter_grid), np.nan)
                        dataset = base["auctions"][previous_market]["dataset"]
                        cancelled = previous_market in known and (dataset, day_text) in notices
                        if cancelled:
                            # Only a documented cancellation permits an earlier-market fallback.
                            for earlier in reversed(markets[:index]):
                                earlier_dataset = base["auctions"][earlier]["dataset"]
                                if (earlier_dataset, day_text) not in notices:
                                    values = price[earlier].lookup(quarter_grid, cutoff)
                                    break
                        core.extend(pad_curve(quarter_grid, values, timezone))
                        core.append(float(cancelled))
                    weekday = day.weekday()
                    core.extend(float(weekday == position) for position in range(7))
                    core.append(float(holiday))
                    for offset in (-1, 1):
                        neighboring = delivery_grid(day + pd.Timedelta(days=offset), 15, timezone)
                        core.append(float(italian_holidays(pd.DatetimeIndex([neighboring[0]]), base)[0]))
                    core.extend(historical_cancellations)
                    blocks = {"core": np.asarray(core, dtype=np.float32)}
                    positions = np.asarray([external_lookup.get(t, -1) for t in quarter_grid.asi8])
                    for name, columns in external_indices.items():
                        vectors = []
                        for column in columns:
                            values = np.full(len(quarter_grid), np.nan)
                            present = positions >= 0
                            values[present] = external[positions[present], column]
                            vectors.extend(pad_curve(quarter_grid, values, timezone))
                        blocks[name] = np.asarray(vectors, dtype=np.float32)
                    required_weather = (delivery_grid(day, 15, timezone, 12) if gate == "IDA3" else quarter_grid)
                    current_weather, revisions, weather_trace = weather.extract(
                        quarter_grid, cutoff, da_cutoff, timezone, required_weather)
                    blocks["weather"] = np.asarray(np.r_[current_weather, revisions], dtype=np.float32)
                    gate_cache[gate] = blocks, weather_trace
                blocks, weather_trace = gate_cache[gate]
                if market == "DA":
                    # Compare the same local clock on the prior delivery day. On
                    # an autumn repeat, use the available first 02:00 profile
                    # for the second occurrence if yesterday had no second hour.
                    yesterday = delivery_grid(day - pd.Timedelta(days=1), 15, timezone)
                    previous_curve = pad_curve(yesterday, price["DA"].lookup(yesterday, cutoff), timezone)
                    for position in range(4):
                        if not np.isfinite(previous_curve[96 + position]):
                            previous_curve[96 + position] = previous_curve[8 + position]
                    reference = previous_curve[slot_numbers(native_grid, timezone)]
                elif gate == "DA":
                    reference = np.full(len(native_grid), np.nan)
                else:
                    reference = np.full(len(native_grid), np.nan)
                    for prior_market in reversed(known):
                        candidate = price[prior_market].lookup(native_grid, cutoff)
                        dataset = base["auctions"][prior_market]["dataset"]
                        if (dataset, day_text) in notices:
                            continue
                        reference = candidate
                        break
                # Ignore structural padding when evaluating active-product coverage.
                active_slots = slot_numbers(native_grid, timezone)
                coverage = {}
                for block_name, block_values in blocks.items():
                    if block_name == "core":
                        continue
                    profiles = block_values.reshape(-1, 100)
                    active = np.isfinite(profiles[:, active_slots])
                    coverage[block_name] = {"available": int(active.sum()), "required": int(active.size),
                        "complete": bool(active.all())}
                weather_profiles = blocks["weather"].reshape(-1, 100)
                current = np.isfinite(weather_profiles[:len(weather_profiles) // 2, active_slots])
                weather_current = {"available": int(current.sum()), "required": int(current.size),
                    "complete": bool(current.all())}
                results.append({"delivery_day": day_text, "market": market, "gate": gate,
                    "decision_at": cutoff.isoformat(), "target_publication_at": target_publication.isoformat(),
                    "product_starts_utc": [stamp.isoformat() for stamp in native_grid],
                    "target_values": target.tolist(), "price_reference": reference.tolist(),
                    "feature_blocks": blocks,
                    "metadata": {"native_minutes": minutes, "source_manifest_sha256": source_digest,
                        "cancelled": (auction["dataset"], day_text) in notices,
                        "is_holiday": holiday,
                        "reference_dst_policy": "DA previous local clock; repeated autumn hour uses the first occurrence when necessary",
                        "weather_cycle_assumption_hours": settings["weather_release_lag_hours"],
                        "weather_vintages": weather_trace, "external_feature_names": external_names,
                        "feature_block_coverage": coverage, "weather_current_coverage": weather_current,
                        "weather_usable": coverage["weather"]["complete"]}})
    return results
