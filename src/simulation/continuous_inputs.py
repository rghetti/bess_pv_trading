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

from ..data import config_path, night_window_mask
from ..data.gme import native_price_resolution_minutes
from ..domain import DeliveryGrid, PlantParameters, battery_parameters_from_config
from ..markets import AuctionCalendar
from ..optimization.prepared import terminal_minimum_energy_mwh
from .gme_oracle import (_forecast_preflight_session, _load_auction_prices, _load_pv,
    _load_pv_forecast, _soc_energy_targets)
from .settlement import load_realized_imbalance_prices
from ..optimization.mpc import uses_realtime_control, needs_mpc_ensemble


def _json_value(value: Any) -> Any:
    """Normalize scientific and clock metadata for stable source fingerprints."""
    # Return `{str(key): _json_value(item) for key, item in value.items()}` when `isinstance(value, Mapping)`.
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    # Return `[_json_value(item) for item in value]` when `isinstance(value, (list, tuple))`.
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    # Return `[_json_value(item) for item in value.tolist()]` when `isinstance(value, np.ndarray)`.
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    # Return `_json_value(value.item())` when `isinstance(value, np.generic)`.
    if isinstance(value, np.generic):
        return _json_value(value.item())
    # Return `value.isoformat()` when `isinstance(value, (datetime, date, pd.Timestamp))`.
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    # Return `str(value)` when `isinstance(value, Path)`.
    if isinstance(value, Path):
        return str(value)
    # Return `str(value)` when `isinstance(value, float) and (not np.isfinite(value))`.
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
    # Enforce the input contract: Continuous forecast starts must be unique increasing timezone-aware timestamps.
    if index.tz is None or index.hasnans or not index.is_unique or not index.is_monotonic_increasing:
        raise ValueError("Continuous forecast starts must be unique increasing timezone-aware timestamps")
    return index.tz_convert("UTC").as_unit("ns")


def _replacement_evidence(config):
    """Pin the selected frozen model and its declared dependencies before replay."""
    settings = config.get("pv", {}).get("forecast_replacement", {})
    if not settings.get("enabled", False):
        return None
    farm = int(config["forecast"]["farm_id"])
    models = settings.get("models_by_farm", {})
    model = models.get(str(farm), models.get(farm))
    if model is None:
        return None
    artifact = config_path(config, model["artifact_path"])
    if not artifact.is_file():
        return {"status": "fallback_artifact_missing", "artifact": str(artifact)}
    metadata = json.loads(artifact.read_text()) if model.get("kind") == "weather_bundle" else {}
    checksums = {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()}
    # A model can change independently of YAML; pin and validate every bundle file.
    for name, expected in metadata.get("checksums", {}).items():
        actual = hashlib.sha256((artifact.parent / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Forecast bundle checksum mismatch: {name}")
        checksums[name] = actual
    if model.get("kind") in {"weather_bundle", "neural"}:
        import torch
        torch.set_num_threads(int(config["solver"]["threads"]))
    return {"status": "pinned", "model": model["name"], "artifact": str(artifact),
        "available_at_utc": model["available_at_utc"], "checksums": checksums,
        "training_period": metadata.get("training_period"), "selection_period": metadata.get("selection_period"),
        "train_last_target_utc": metadata.get("train_last_target_utc"),
        "specification": metadata.get("specification"), "inference_threads": int(config["solver"]["threads"])}


def _replace_initial_forecast(config, day, cutoff, grid, bundle, forecast, actual, provenance, evidence):
    """Use the same selected bundle for initialization and frozen measurement fallback."""
    if evidence is None or evidence.get("status") != "pinned":
        return forecast, actual, bundle, provenance
    settings = config["pv"]["forecast_replacement"]
    models = settings["models_by_farm"]
    farm = int(config["forecast"]["farm_id"])
    model = models.get(str(farm), models.get(farm))
    available = pd.Timestamp(evidence["available_at_utc"])
    if model["kind"] != "weather_bundle" or (available > cutoff and not settings.get("allow_retrospective_artifacts", False)):
        return forecast, actual, bundle, provenance
    from ..forecasting.inference.pv_forecast_bundle import predict_bundle

    frame = bundle.values.copy(deep=True)
    frame["interval_start_utc"] = pd.DatetimeIndex(grid.starts)
    frame["delivery_day"] = day
    # The weather-only network does not consume auction identity. DA is a valid
    # routing label required by the existing sequence packer, not a feature.
    frame["auction"] = "DA"
    frame["forecast_cycle_utc"] = pd.Timestamp(bundle.forecast_vintage)
    policy = config.get("policy", {}).get("forecast_pv_night", {})
    forecast = predict_bundle(frame, Path(evidence["artifact"]).parent, farm_id=farm, night_policy=policy)
    values = bundle.values.copy(deep=True)
    values["pv_ac_mw"] = forecast
    bundle = replace(bundle, values=values)
    actual = np.asarray(actual).copy()
    fallback_ids = set(provenance.get("realized_power_fallback_interval_ids", []))
    for index, product in enumerate(grid.interval_ids):
        if product in fallback_ids:
            actual[index] = forecast[index]
    provenance = {**provenance, "initial_forecast_model": evidence["model"],
        "initial_forecast_model_retrospective": bool(available > cutoff),
        "realized_fallback_model": evidence["model"], "forecast_model_evidence": evidence}
    return forecast, actual, bundle, provenance


def _prepare_pv_ensembles(config, days_data, progress=None):
    """Convert each frozen preceding-day weather ensemble to AC trajectories."""
    import xarray as xr

    from ..data import load_metadata
    from ..forecasting.data.pv_probabilistic_data import align_cycle
    from ..forecasting.models.pv import forecast_pv

    settings = config["continuous_optimization"]["pv_ensemble"]
    policy = config.get("policy", {})
    forecast_night = policy.get("forecast_pv_night", policy.get("night", {}))
    source = config_path(config, settings["zarr"])
    farm_id = int(config["forecast"]["farm_id"])
    replacement = config["pv"].get("forecast_replacement", {})
    models = replacement.get("models_by_farm", {})
    has_model = str(farm_id) in models or farm_id in models
    model = models.get(str(farm_id), models.get(farm_id))
    if replacement.get("enabled") and not has_model:
        raise ValueError("PV ensemble model selection is missing for the selected farm")
    use_bundle = bool(replacement.get("enabled") and model is not None)
    if use_bundle:
        if model.get("kind") != "weather_bundle":
            raise ValueError("PV ensemble model requires a configured weather bundle for the selected farm")
        artifact = config_path(config, model["artifact_path"])
        if not artifact.is_file():
            raise ValueError(f"PV ensemble model artifact is missing: {artifact}")
        available_at = pd.Timestamp(model["available_at_utc"])
        if available_at.tzinfo is None:
            raise ValueError("PV ensemble model availability must be timezone-aware")
    farm = load_metadata(config_path(config, config["inputs"]["metadata_parquet"]), [farm_id])[farm_id]
    provenance = {}
    with xr.open_zarr(source, consolidated=True, chunks=None) as dataset:
        members = tuple(str(value) for value in dataset.member.values)
        if len(members) != int(settings["member_count"]) or len(set(members)) != len(members):
            raise ValueError("PV ensemble archive has an unexpected member axis")
        for day, daily in days_data.items():
            starts = _utc_index(daily["grid"].starts)
            cycle = (pd.Timestamp(day) - pd.Timedelta(days=1)).tz_localize("UTC")
            try:
                weather = align_cycle(dataset, farm_id, cycle, starts, config, ensemble=True)
            except (KeyError, ValueError) as error:
                raise ValueError(f"PV ensemble missing or invalid for {day}, cycle {cycle}: {error}") from error
            first_cutoff = min(event.submission_at for event in daily["schedule"])
            if weather.available_at > first_cutoff:
                raise ValueError(f"PV ensemble for {day} is not available at the first auction cutoff")
            if use_bundle:
                if available_at > pd.Timestamp(first_cutoff) and not replacement.get("allow_retrospective_artifacts", False):
                    raise ValueError(f"PV ensemble model was not available at the first cutoff for {day}")
                from ..forecasting.inference.pv_forecast_bundle import predict_bundle

                # Keep member curves independent for both interpolation and neural sequence packing.
                frame = weather.values.copy().reset_index(drop=False)
                frame["interval_start_utc"] = np.tile(starts, len(members))
                frame["delivery_day"] = day
                frame["auction"] = "DA"
                frame["forecast_cycle_utc"] = cycle
                frame["_scenario_member"] = frame["member"].astype(str)
                predicted = predict_bundle(frame, artifact.parent, farm_id=farm_id,
                                           night_policy=forecast_night)
                values = np.stack([predicted[frame["_scenario_member"].to_numpy() == member]
                                   for member in members])
                conversion = {"model": model.get("name"), "kind": model["kind"],
                    "manifest": str(artifact), "manifest_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "available_at_utc": available_at.isoformat(),
                    "retrospective": bool(available_at > pd.Timestamp(first_cutoff))}
            else:
                converted = forecast_pv(weather, farm, config["pv"], forecast_night)
                values = np.stack([converted.values.loc[converted.values.member == member, "pv_ac_mw"].to_numpy(float)
                                   for member in members])
                conversion = converted.provenance
            # Match the deterministic loader's explicit nighttime-only missing-data policy.
            missing = ~np.isfinite(values)
            night = night_window_mask(starts, forecast_night)
            if missing.any() and config["forecast"]["missing_forecast_policy"] == "zero_only_when_policy_marks_night" and forecast_night.get("enabled", True):
                values[missing & np.broadcast_to(night, values.shape)] = 0.0
            # Apply the same prior to every trajectory, independently of model kind.
            values[:, night] = 0.0
            if values.shape != (len(members), len(starts)) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError(f"PV ensemble for {day} has incomplete or invalid AC trajectories")
            daily["ensemble_pv"] = values
            daily["ensemble_members"] = members
            provenance[day] = {"source": str(source), "forecast_vintage_utc": cycle.isoformat(),
                "available_at_utc": weather.available_at.isoformat(), "availability": "assumed cycle plus configured latency",
                "member_ids": list(members), "values_sha256": _fingerprint(values),
                "nighttime_missing_filled_zero_member_intervals": int(missing.sum()),
                "conversion": conversion, "forecast_pv_night_policy": forecast_night}
            if progress:
                progress(f"Preflight {day}: converted {len(members)} raw PV ensemble members")
    return provenance


def _validate_bundle(bundle, cutoff: pd.Timestamp, starts: pd.DatetimeIndex) -> None:
    """Reject unavailable vintages and unresolved or ambiguously aligned forecasts."""
    # Enforce the input contract: PV forecast available at `bundle.available_at` is after submission `cutoff`.
    if pd.Timestamp(bundle.available_at) > cutoff:
        raise ValueError(f"PV forecast available at {bundle.available_at} is after submission {cutoff}")
    # Enforce the input contract: Continuous optimization currently requires one deterministic PV member.
    if len(bundle.members) != 1:
        raise ValueError("Continuous optimization currently requires one deterministic PV member")
    index = _utc_index(bundle.values.index)
    values = pd.to_numeric(bundle.values["pv_ac_mw"], errors="coerce").to_numpy(float)
    # Enforce the input contract: Continuous PV forecast does not exactly cover the requested physical grid.
    if not index.equals(starts) or values.shape != (len(starts),):
        raise ValueError("Continuous PV forecast does not exactly cover the requested physical grid")
    # Enforce the input contract: Continuous PV forecast must be finite and nonnegative.
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
        # Enforce the input contract: Continuous forecast cutoff must include a timezone.
        if cutoff.tzinfo is None:
            raise ValueError("Continuous forecast cutoff must include a timezone")
        cutoff = cutoff.tz_convert("UTC")
        starts = _utc_index(starts)
        bundle = self._bundles.get(cutoff.isoformat())
        # Enforce the input contract: Continuous forecast cutoff `cutoff.isoformat()` was not preflighted.
        if bundle is None:
            raise ValueError(f"Continuous forecast cutoff {cutoff.isoformat()} was not preflighted")
        positions = _utc_index(bundle.values.index).get_indexer(starts)
        # Enforce the input contract: Continuous forecast request at `cutoff.isoformat()` exceeds preflighted intervals.
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
    # Keep legacy tracking preflight/fingerprints unchanged. MPC workflows pin
    # model artifacts and use the selected model for their initial fallback too.
    replacement_evidence = _replacement_evidence(config) if uses_realtime_control(config) else None
    try:
        first = date.fromisoformat(str(settings["start_date"]))
        last = date.fromisoformat(str(settings["end_date"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("Continuous optimization requires ISO start_date and end_date delivery dates") from None
    # Enforce the input contract: Continuous optimization end_date must not precede start_date.
    if last < first:
        raise ValueError("Continuous optimization end_date must not precede start_date")
    days = tuple(day.date().isoformat() for day in pd.date_range(first, last, freq="D"))
    calendar = AuctionCalendar.from_config(config)
    # Enforce the input contract: Continuous market and price experiment timezones must agree.
    if calendar.timezone != config["gme_oracle_experiment"]["timezone"]:
        raise ValueError("Continuous market and price experiment timezones must agree")
    # Enforce the input contract: Continuous optimization requires the latest available forecast at each reoptimization.
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
    # Enforce the input contract: Plant and PV forecast must identify the same farm.
    if plant.farm_id != int(config["forecast"]["farm_id"]):
        raise ValueError("Plant and PV forecast must identify the same farm")
    initial_energy, terminal_energy = _soc_energy_targets(config["state"], battery)
    terminal_minimum = terminal_minimum_energy_mwh(config["state"], battery)
    days_data: dict[str, dict[str, Any]] = {}
    all_grids = [calendar.delivery_grid(day).to_domain() for day in days]
    full_grid = DeliveryGrid(tuple(item for grid in all_grids for item in grid.interval_ids),
        tuple(item for grid in all_grids for item in grid.starts),
        tuple(item for grid in all_grids for item in grid.ends))
    full_starts = _utc_index(full_grid.starts)
    initial_requests: dict[str, tuple[pd.DatetimeIndex, Any]] = {}
    forecast_requests: dict[str, pd.DatetimeIndex] = {}
    # Pin exact prices, common measured PV/fallbacks and every causal forecast when `progress`.
    if progress:
        progress(f"Preflighting continuous delivery inputs for {days[0]} through {days[-1]} (offline)")

    # Reuse verified workbook snapshots across days, keeping memory bounded and
    # releasing database connections before weather conversion or inference.
    from ..data.api_catalog import SnapshotReadCache
    price_snapshot_cache = SnapshotReadCache(settings.get("price_snapshot_cache_entries", 8))
    # Read actual production once per day, with one strategy-independent fallback.
    for day, grid in zip(days, all_grids):
        daily = deepcopy(config)
        study = daily["gme_oracle_experiment"]
        study["delivery_day"] = day
        study["price_database"] = str(config_path(config, study["price_database"]))
        # Pin exact prices, common measured PV/fallbacks and every causal forecast for each `item` in `study['auction_datasets'].values()`.
        for item in study["auction_datasets"].values():
            item["native_interval_minutes"] = native_price_resolution_minutes(item["dataset"], date.fromisoformat(day))
        scheduled = calendar.schedule(day)
        try:
            prices, price_provenance, skipped = _load_auction_prices(daily, calendar, grid, scheduled,
                snapshot_cache=price_snapshot_cache)
            schedule = tuple(event for event in scheduled if event.auction in prices and event.available)
            first_cutoff = min(pd.Timestamp(grid.starts[0]), *(event.submission_at for event in scheduled))
            initialization = config["forecast"].get("initialization_cutoff") if day == days[0] else None
            # When `initialization is not None`, set `stamp` using `pd.Timestamp(initialization)`.
            if initialization is not None:
                stamp = pd.Timestamp(initialization)
                # Enforce the input contract: forecast.initialization_cutoff must precede the first submission and delivery.
                if stamp.tzinfo is None or stamp > first_cutoff:
                    raise ValueError("forecast.initialization_cutoff must precede the first submission and delivery")
                first_cutoff = stamp
            forecast, actual, bundle, pv_provenance = _load_pv(daily, calendar, grid,
                                                              decision_cutoff=first_cutoff)
            forecast, actual, bundle, pv_provenance = _replace_initial_forecast(
                config, day, first_cutoff, grid, bundle, forecast, actual, pv_provenance, replacement_evidence)
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
        # Pin exact prices, common measured PV/fallbacks and every causal forecast for each `event` in `schedule`.
        for event in schedule:
            cutoff = event.submission_at.tz_convert("UTC")
            # Include the in-progress interval for its forecast-only remainder.
            starts = full_starts[(full_starts >= cutoff.floor("15min"))
                                 & (full_starts < pd.Timestamp(grid.ends[-1]))]
            key = cutoff.isoformat()
            forecast_requests[key] = forecast_requests.get(key, starts).union(starts)
        # Pin exact prices, common measured PV/fallbacks and every causal forecast when `progress`.
        if progress:
            progress(f"Preflight {day}: {len(schedule)} available auctions, {len(grid.interval_ids)} delivery intervals")

        # Pin each interval-start information set; no later vintage can enter an MPC.
        if uses_realtime_control(config):
            day_starts = _utc_index(grid.starts)
            for index, cutoff in enumerate(day_starts):
                key = cutoff.isoformat()
                starts = day_starts[index:]
                forecast_requests[key] = forecast_requests.get(key, starts).union(starts)

    # Freeze union forecasts covering both current-day projection and next-day plan.
    forecasts = _PreparedForecasts()
    # Pin exact prices, common measured PV/fallbacks and every causal forecast for each `(key, (starts, _))` in `initial_requests.items()`.
    for key, (starts, _) in initial_requests.items():
        forecast_requests[key] = forecast_requests.get(key, starts).union(starts)
    # MPC has many more cutoffs than auctions; distinguish this scan from the
    # completed daily checks and report progress during long cached-data reads.
    forecast_request_count = len(forecast_requests)
    show_forecast_progress = progress is not None and uses_realtime_control(config)
    if show_forecast_progress:
        progress(f"Daily input checks complete; preflighting {forecast_request_count} forecast cutoffs")
    with _forecast_preflight_session(config, calendar) as forecast_reader:
        for request_index, (key, starts) in enumerate(sorted(forecast_requests.items()), start=1):
            initial = initial_requests.get(key)
            try:
                bundle = (initial[1] if initial is not None and starts.equals(initial[0]) else
                          _load_pv_forecast(config, calendar, pd.Timestamp(key), starts))
                forecasts.add(pd.Timestamp(key), starts, bundle)
            except (ValueError, RuntimeError) as error:
                raise ValueError(f"Continuous weather preflight at {key}: {error}") from error
            if show_forecast_progress and (request_index % 100 == 0 or request_index == forecast_request_count):
                progress(f"Forecast preflight {request_index}/{forecast_request_count}: {key}")
        forecast_reader_stats = forecast_reader.stats
    if show_forecast_progress:
        progress(f"Cutoff forecast cache: {forecast_reader_stats['zarr_store_opens']} Zarr open, "
            f"{forecast_reader_stats['metadata_reads']} metadata read, "
            f"{forecast_reader_stats['distinct_raw_vintages']} raw vintages, "
            f"{forecast_reader_stats['raw_vintage_cache_hits']} raw cache hits, "
            f"{forecast_reader_stats['distinct_pv_conversions']} PV conversions, "
            f"{forecast_reader_stats['pv_conversion_cache_hits']} PV cache hits")
    ensemble_provenance = None
    if settings.get("pv_ensemble", {}).get("enabled", False) or needs_mpc_ensemble(config):
        ensemble_provenance = _prepare_pv_ensembles(config, days_data, progress)
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
        "weather_forecasts": forecasts.provenance, "pv_ensemble": ensemble_provenance,
        **({"forecast_model_artifacts": replacement_evidence} if replacement_evidence is not None else {}),
        "settlement": settlement_provenance,
        "settlement_values_sha256": None if settlement_values is None else _fingerprint(settlement_values)}
    fingerprint = _fingerprint(provenance)
    return {"calendar": calendar, "days": days, "days_data": days_data,
        "plant": plant, "initial_energy_mwh": initial_energy, "terminal_energy_mwh": terminal_energy,
        "terminal_minimum_energy_mwh": terminal_minimum,
        "full_grid": full_grid, "full_realized_pv": full_actual,
        "forecast_provider": forecasts, "imbalance_prices_eur_per_mwh": settlement_values,
        "settlement_frame": settlement, "input_fingerprint": fingerprint, "provenance": provenance}
