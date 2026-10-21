"""Prepare and fingerprint shared, offline inputs for consecutive delivery days."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, replace
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from ..data import config_path
from ..data.gme import native_price_resolution_minutes
from ..domain import DeliveryGrid, PlantParameters, battery_parameters_from_config
from ..markets import AuctionCalendar
from .gme_oracle import _load_auction_prices, _load_pv, _load_pv_forecast, _soc_energy_targets
from .settlement import load_realized_imbalance_prices


def _json_value(value: Any) -> Any:
    """Normalize scientific and clock metadata for stable source fingerprints."""
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def _fingerprint(value: Any) -> str:
    """Hash canonical metadata rather than mutable source retrieval locations."""
    payload = json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode()).hexdigest()


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    """Include aligned values, index, column names and dtypes in weather evidence."""
    digest = hashlib.sha256()
    digest.update(json.dumps([(str(key), str(dtype)) for key, dtype in frame.dtypes.items()]).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy(dtype="uint64").tobytes())
    return digest.hexdigest()


def _utc_index(starts) -> pd.DatetimeIndex:
    """Use one explicit UTC precision for request identity and cached slicing."""
    index = pd.DatetimeIndex(starts)
    if index.tz is None or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Continuous forecast starts must be unique increasing timezone-aware timestamps")
    return index.tz_convert("UTC").as_unit("ns")


def _validate_bundle(bundle, cutoff: pd.Timestamp, starts: pd.DatetimeIndex) -> None:
    """Reject unavailable vintages and unresolved or ambiguously aligned forecasts."""
    if pd.Timestamp(bundle.available_at) > cutoff:
        raise ValueError(f"PV forecast available at {bundle.available_at} is after submission {cutoff}")
    if len(bundle.members) != 1:
        raise ValueError("Continuous optimization currently requires one deterministic PV member")
    index = _utc_index(bundle.values.index)
    values = pd.to_numeric(bundle.values["pv_ac_mw"], errors="coerce").to_numpy(float)
    if not index.equals(starts) or values.shape != (len(starts),):
        raise ValueError("Continuous PV forecast does not exactly cover the requested physical grid")
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Continuous PV forecast must be finite and nonnegative")


class _PreparedForecasts:
    """Serve defensive subsets of preflighted vintages without rereading sources."""

    def __init__(self):
        self._bundles: dict[str, Any] = {}
        self.provenance: list[dict[str, Any]] = []

    def add(self, cutoff, starts, bundle) -> None:
        """Freeze the complete weather/PV evidence for a submission cutoff."""
        cutoff = pd.Timestamp(cutoff).tz_convert("UTC")
        starts = _utc_index(starts)
        _validate_bundle(bundle, cutoff, starts)
        self._bundles[cutoff.isoformat()] = replace(bundle, values=bundle.values.copy(deep=True),
                                                    provenance=deepcopy(bundle.provenance))
        self.provenance.append({"cutoff_utc": cutoff.isoformat(),
            "forecast_vintage_utc": pd.Timestamp(bundle.forecast_vintage).isoformat(),
            "available_at_utc": pd.Timestamp(bundle.available_at).isoformat(),
            "first_interval_utc": starts[0].isoformat(), "last_interval_utc": starts[-1].isoformat(),
            "intervals": len(starts), "aligned_weather_pv_sha256": _frame_fingerprint(bundle.values),
            "provenance": deepcopy(bundle.provenance)})

    def __call__(self, cutoff, starts):
        """Return a copied slice; unpreflighted requests cannot alter pinned inputs."""
        cutoff = pd.Timestamp(cutoff)
        if cutoff.tzinfo is None:
            raise ValueError("Continuous forecast cutoff must include a timezone")
        cutoff = cutoff.tz_convert("UTC")
        starts = _utc_index(starts)
        bundle = self._bundles.get(cutoff.isoformat())
        if bundle is None:
            raise ValueError(f"Continuous forecast cutoff {cutoff.isoformat()} was not preflighted")
        positions = _utc_index(bundle.values.index).get_indexer(starts)
        if np.any(positions < 0):
            raise ValueError(f"Continuous forecast request at {cutoff.isoformat()} exceeds preflighted intervals")
        return replace(bundle, values=bundle.values.iloc[positions].copy(deep=True),
                       provenance=deepcopy(bundle.provenance))


def prepare_continuous_inputs(config: dict[str, Any], progress: Callable[[str], Any] | None = None) -> dict[str, Any]:
    """Pin exact prices, common measured PV/fallbacks and every causal forecast.

    Dates are inclusive local delivery dates. Measured production is read only
    within those dates, including when first-day orders close on the prior day.
    All database access uses existing read-only catalog readers; this function
    never acquires data or changes configuration, caches or source snapshots.
    """
    settings = config["continuous_optimization"]
    try:
        first = date.fromisoformat(str(settings["start_date"]))
        last = date.fromisoformat(str(settings["end_date"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("Continuous optimization requires ISO start_date and end_date delivery dates") from None
    if last < first:
        raise ValueError("Continuous optimization end_date must not precede start_date")
    days = tuple(day.date().isoformat() for day in pd.date_range(first, last, freq="D"))
    calendar = AuctionCalendar.from_config(config)
    if calendar.timezone != config["gme_oracle_experiment"]["timezone"]:
        raise ValueError("Continuous market and price experiment timezones must agree")
    if config["forecast"]["reoptimization_pv_policy"] != "latest_available_forecast_at_each_reoptimization":
        raise ValueError("Continuous optimization requires the latest available forecast at each reoptimization")
    # Actual execution state advances only when a quarter closes. A delayed
    # telemetry feed needs a separate hidden physical state and observed state;
    # admitting it here would expose unavailable SOC at subsequent submissions.
    if float(config.get("source", {}).get("telemetry_latency_minutes", 0.0)) != 0.0:
        raise ValueError("Continuous optimization requires source.telemetry_latency_minutes: 0; "
                         "the current controller assumes immediate completed-interval observations")
    battery = battery_parameters_from_config(config["battery"])
    plant = PlantParameters(**{key: config["plant"][key]
                              for key in ("farm_id", "poi_import_mw", "poi_export_mw")}, battery=battery)
    if plant.farm_id != int(config["forecast"]["farm_id"]):
        raise ValueError("Plant and PV forecast must identify the same farm")
    initial_energy, terminal_energy = _soc_energy_targets(config["state"], battery)
    days_data: dict[str, dict[str, Any]] = {}
    all_grids = [calendar.delivery_grid(day).to_domain() for day in days]
    full_grid = DeliveryGrid(tuple(item for grid in all_grids for item in grid.interval_ids),
        tuple(item for grid in all_grids for item in grid.starts),
        tuple(item for grid in all_grids for item in grid.ends))
    full_starts = _utc_index(full_grid.starts)
    initial_requests: dict[str, tuple[pd.DatetimeIndex, Any]] = {}
    forecast_requests: dict[str, pd.DatetimeIndex] = {}
    if progress:
        progress(f"Preflighting continuous delivery inputs for {days[0]} through {days[-1]} (offline)")

    # Read actual production once per day, with one strategy-independent fallback.
    for day, grid in zip(days, all_grids):
        daily = deepcopy(config)
        study = daily["gme_oracle_experiment"]
        study["delivery_day"] = day
        study["price_database"] = str(config_path(config, study["price_database"]))
        for item in study["auction_datasets"].values():
            item["native_interval_minutes"] = native_price_resolution_minutes(item["dataset"], date.fromisoformat(day))
        scheduled = calendar.schedule(day)
        try:
            prices, price_provenance, skipped = _load_auction_prices(daily, calendar, grid, scheduled)
            schedule = tuple(event for event in scheduled if event.auction in prices and event.available)
            first_cutoff = min(pd.Timestamp(grid.starts[0]), *(event.submission_at for event in scheduled))
            initialization = config["forecast"].get("initialization_cutoff") if day == days[0] else None
            if initialization is not None:
                stamp = pd.Timestamp(initialization)
                if stamp.tzinfo is None or stamp > first_cutoff:
                    raise ValueError("forecast.initialization_cutoff must precede the first submission and delivery")
                first_cutoff = stamp
            forecast, actual, bundle, pv_provenance = _load_pv(daily, calendar, grid,
                                                              decision_cutoff=first_cutoff)
            _validate_bundle(bundle, first_cutoff, _utc_index(grid.starts))
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Continuous preflight for {day}: {error}") from error
        initial_requests[pd.Timestamp(first_cutoff).tz_convert("UTC").isoformat()] = (_utc_index(grid.starts), bundle)
        days_data[day] = {"grid": grid, "schedule": schedule, "prices": prices,
            "skipped": skipped, "price_provenance": price_provenance,
            "realized_pv": np.asarray(actual, dtype=float).copy(),
            "initial_forecast": np.asarray(forecast, dtype=float).copy(),
            "initial_pv_bundle": bundle, "pv_provenance": pv_provenance,
            "pv_metadata": {"pv_forecast_vintage_utc": pd.Timestamp(bundle.forecast_vintage).isoformat(),
                "pv_forecast_available_at_utc": pd.Timestamp(bundle.available_at).isoformat()},
            "daily_config": daily,
            "native_interval_minutes_by_auction": {auction: int(item["native_interval_minutes"])
                for auction, item in study["auction_datasets"].items() if auction in prices}}
        for event in schedule:
            cutoff = event.submission_at.tz_convert("UTC")
            # Include the in-progress interval for its forecast-only remainder.
            starts = full_starts[(full_starts >= cutoff.floor("15min"))
                                 & (full_starts < pd.Timestamp(grid.ends[-1]))]
            key = cutoff.isoformat()
            forecast_requests[key] = forecast_requests.get(key, starts).union(starts)
        if progress:
            progress(f"Preflight {day}: {len(schedule)} available auctions, {len(grid.interval_ids)} delivery intervals")

    # Freeze union forecasts covering both current-day projection and next-day plan.
    forecasts = _PreparedForecasts()
    for key, (starts, _) in initial_requests.items():
        forecast_requests[key] = forecast_requests.get(key, starts).union(starts)
    for key, starts in sorted(forecast_requests.items()):
        initial = initial_requests.get(key)
        try:
            bundle = (initial[1] if initial is not None and starts.equals(initial[0]) else
                      _load_pv_forecast(config, calendar, pd.Timestamp(key), starts))
            forecasts.add(pd.Timestamp(key), starts, bundle)
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Continuous weather preflight at {key}: {error}") from error
    settlement, settlement_provenance = load_realized_imbalance_prices(config, full_grid)
    settlement_values = (None if settlement is None else
                         settlement.settlement_price_eur_mwh.to_numpy(float, copy=True))
    full_actual = np.concatenate([days_data[day]["realized_pv"] for day in days])
    provenance = {"schema_version": "continuous-inputs-v1", "offline": True,
        "start_date": days[0], "end_date": days[-1], "delivery_timezone": calendar.timezone,
        "measured_pv_basis": "AC; no conversion losses applied",
        "realized_pv_fallback": "one initial frozen forecast per day shared by all strategies",
        "plant": asdict(plant),
        "asset_input_settings": {key: deepcopy(config.get(key)) for key in
            ("plant", "battery", "state", "pv", "source", "forecast_mapping", "forecast_semantics",
             "forecast", "policy", "degradation", "transaction_fees", "trading", "execution")},
        "days": {day: {"price_provenance": item["price_provenance"],
            "skipped_auctions": item["skipped"], "pv_provenance": item["pv_provenance"],
            "realized_pv_sha256": _fingerprint(item["realized_pv"]),
            "auction_price_sha256": {name: _fingerprint(curve) for name, curve in item["prices"].items()}}
            for day, item in days_data.items()},
        "weather_forecasts": forecasts.provenance,
        "settlement": settlement_provenance,
        "settlement_values_sha256": None if settlement_values is None else _fingerprint(settlement_values)}
    fingerprint = _fingerprint(provenance)
    return {"calendar": calendar, "days": days, "days_data": days_data,
        "plant": plant, "initial_energy_mwh": initial_energy, "terminal_energy_mwh": terminal_energy,
        "full_grid": full_grid, "full_realized_pv": full_actual,
        "forecast_provider": forecasts, "imbalance_prices_eur_per_mwh": settlement_values,
        "settlement_frame": settlement, "input_fingerprint": fingerprint, "provenance": provenance}
