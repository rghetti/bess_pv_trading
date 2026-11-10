"""Prepare and fingerprint shared, offline inputs for consecutive delivery days."""
from __future__ import annotations

from copy import deepcopy
from contextlib import ExitStack
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
    metadata = json.loads(artifact.read_text()) if model.get("kind") in {"weather_bundle", "reviewed_hpo"} else {}
    checksums = {artifact.name: hashlib.sha256(artifact.read_bytes()).hexdigest()}
    # A model can change independently of YAML; pin and validate every bundle file.
    for name, expected in metadata.get("checksums", {}).items():
        if model.get("kind") == "reviewed_hpo" and name == "prediction_mw.npy":
            continue
        actual = hashlib.sha256((artifact.parent / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Forecast bundle checksum mismatch: {name}")
        checksums[name] = actual
    if model.get("kind") in {"weather_bundle", "neural", "reviewed_hpo"}:
        import torch
        torch.set_num_threads(int(config["solver"]["threads"]))
    return {"status": "pinned", "model": model["name"], "artifact": str(artifact),
        "available_at_utc": model["available_at_utc"], "checksums": checksums,
        "training_period": metadata.get("training_period"), "selection_period": metadata.get("selection_period"),
        "train_last_target_utc": metadata.get("train_last_target_utc"),
        "specification": metadata.get("specification"), "inference_threads": int(config["solver"]["threads"])}


def _reviewed_issue_hours(config, evidence):
    """Return the declared complete issue length, if this reviewed bundle needs one."""
    if not evidence or evidence.get("status") != "pinned":
        return None
    models = config["pv"]["forecast_replacement"]["models_by_farm"]
    farm = int(config["forecast"]["farm_id"])
    model = models.get(str(farm), models.get(farm))
    if model.get("kind") != "reviewed_hpo":
        return None
    metadata = json.loads(Path(evidence["artifact"]).read_text())
    hours = metadata.get("input_contract", {}).get("issue_horizon_hours")
    return None if hours is None else int(hours)


def _issue_context_prediction(config, calendar, cutoff, bundle, evidence, cache):
    """Infer a complete eligible weather issue once, then slice requested intervals."""
    from ..forecasting.inference.pv_reviewed_hpo import predict_reviewed_hpo

    hours = _reviewed_issue_hours(config, evidence)
    cycle = pd.Timestamp(bundle.forecast_vintage).tz_convert("UTC")
    key = (cycle.isoformat(), evidence["artifact"])
    starts = _utc_index(bundle.values.index)
    if key not in cache:
        # Selecting again at the same cutoff must preserve the exact issue vintage.
        issue_starts = pd.date_range(cycle, periods=hours * 4, freq="15min")
        full = _load_pv_forecast(config, calendar, cutoff, issue_starts)
        if pd.Timestamp(full.forecast_vintage) != cycle or pd.Timestamp(full.available_at) > pd.Timestamp(cutoff):
            raise ValueError("Full issue weather vintage differs or is unavailable")
        frame = full.values.copy(deep=True)
        frame["interval_start_utc"] = issue_starts
        frame["delivery_day"] = issue_starts.tz_convert(config["market"]["timezone"]).strftime("%Y-%m-%d")
        frame["auction"] = "DA"
        frame["forecast_cycle_utc"] = cycle
        frame["forecast_available_at_utc"] = pd.Timestamp(full.available_at)
        frame["decision_cutoff_utc"] = pd.Timestamp(cutoff)
        predictions = predict_reviewed_hpo(frame, Path(evidence["artifact"]).parent,
            farm_id=int(config["forecast"]["farm_id"]),
            night_policy=config.get("policy", {}).get("forecast_pv_night", {}))
        cache[key] = (pd.Series(predictions, index=issue_starts), _frame_fingerprint(full.values))
    predictions, fingerprint = cache[key]
    positions = predictions.index.get_indexer(starts)
    if np.any(positions < 0):
        raise ValueError("Requested forecast lies outside the model's complete issue context")
    values = bundle.values.copy(deep=True)
    values["pv_ac_mw"] = predictions.iloc[positions].to_numpy()
    provenance = {**bundle.provenance, "reviewed_hpo_model": evidence["model"],
        "model_issue_context_hours": hours, "model_issue_context_weather_sha256": fingerprint}
    return replace(bundle, values=values, provenance=provenance)


def _replace_initial_forecast(config, day, cutoff, grid, bundle, forecast, actual, provenance, evidence, *, calendar=None, issue_cache=None):
    """Use the same selected bundle for initialization and frozen measurement fallback."""
    if evidence is None or evidence.get("status") != "pinned":
        return forecast, actual, bundle, provenance
    settings = config["pv"]["forecast_replacement"]
    models = settings["models_by_farm"]
    farm = int(config["forecast"]["farm_id"])
    model = models.get(str(farm), models.get(farm))
    available = pd.Timestamp(evidence["available_at_utc"])
    if model["kind"] not in {"weather_bundle", "reviewed_hpo"} or (available > cutoff and not settings.get("allow_retrospective_artifacts", False)):
        return forecast, actual, bundle, provenance

    if _reviewed_issue_hours(config, evidence) is not None:
        if calendar is None or issue_cache is None:
            raise ValueError("Complete issue-context inference requires a preflight reader")
        bundle = _issue_context_prediction(config, calendar, cutoff, bundle, evidence, issue_cache)
        forecast = bundle.values["pv_ac_mw"].to_numpy(dtype=float).copy()
        actual = np.asarray(actual).copy()
        fallback_ids = set(provenance.get("realized_power_fallback_interval_ids", []))
        for index, product in enumerate(grid.interval_ids):
            if product in fallback_ids:
                actual[index] = forecast[index]
        provenance = {**provenance, "initial_forecast_model": evidence["model"],
            "initial_forecast_model_retrospective": bool(available > cutoff),
            "realized_fallback_model": evidence["model"], "forecast_model_evidence": evidence,
            "model_issue_context_hours": _reviewed_issue_hours(config, evidence),
            "model_issue_context_weather_sha256": bundle.provenance["model_issue_context_weather_sha256"]}
        return forecast, actual, bundle, provenance

    frame = bundle.values.copy(deep=True)
    frame["interval_start_utc"] = pd.DatetimeIndex(grid.starts)
    frame["delivery_day"] = day
    # The weather-only network does not consume auction identity. DA is a valid
    # routing label required by the existing sequence packer, not a feature.
    frame["auction"] = "DA"
    frame["forecast_cycle_utc"] = pd.Timestamp(bundle.forecast_vintage)
    policy = config.get("policy", {}).get("forecast_pv_night", {})
    if model["kind"] == "reviewed_hpo":
        from ..forecasting.inference.pv_reviewed_hpo import predict_reviewed_hpo

        frame["forecast_available_at_utc"] = pd.Timestamp(bundle.available_at)
        frame["decision_cutoff_utc"] = pd.Timestamp(cutoff)
        forecast = predict_reviewed_hpo(frame, Path(evidence["artifact"]).parent,
                                        farm_id=farm, night_policy=policy)
    else:
        from ..forecasting.inference.pv_forecast_bundle import predict_bundle

        forecast = predict_bundle(frame, Path(evidence["artifact"]).parent, farm_id=farm, night_policy=policy)
    values = bundle.values.copy(deep=True)
    values["pv_ac_mw"] = forecast
    bundle = replace(bundle, values=values, provenance={**bundle.provenance,
        **({"reviewed_hpo_model": model["name"]} if model["kind"] == "reviewed_hpo" else {})})
    actual = np.asarray(actual).copy()
    fallback_ids = set(provenance.get("realized_power_fallback_interval_ids", []))
    for index, product in enumerate(grid.interval_ids):
        if product in fallback_ids:
            actual[index] = forecast[index]
    provenance = {**provenance, "initial_forecast_model": evidence["model"],
        "initial_forecast_model_retrospective": bool(available > cutoff),
        "realized_fallback_model": evidence["model"], "forecast_model_evidence": evidence}
    return forecast, actual, bundle, provenance


def _fill_prior_issue_context(frame, starts, model_starts, *, config, calendar, cutoff, cycle):
    """Fill only missing pre-delivery weather with the same eligible deterministic vintage."""
    columns = [name for name in config["forecast_mapping"] if name in frame]
    before_delivery = pd.DatetimeIndex(frame["interval_start_utc"]) < starts[0]
    missing = frame[columns].isna().to_numpy() & before_delivery[:, None]
    if not missing.any():
        return frame, None
    policy = config["continuous_optimization"]["pv_ensemble"].get("issue_context_missing_policy", "error")
    if policy == "error":
        raise ValueError("Ensemble issue context has missing pre-delivery weather")
    if policy != "same_vintage_deterministic":
        raise ValueError("Unknown ensemble issue-context missing-data policy")
    # The configured fallback cannot fill missing delivery intervals or use a later vintage.
    deterministic = _load_pv_forecast(config, calendar, cutoff, model_starts)
    if pd.Timestamp(deterministic.forecast_vintage) != cycle or pd.Timestamp(deterministic.available_at) > cutoff:
        raise ValueError("Deterministic issue-context fill must use the same eligible weather vintage")
    donor = deterministic.values.reindex(pd.DatetimeIndex(frame["interval_start_utc"]))
    result = frame.copy(deep=True)
    counts = {}
    intervals = set()
    for index, name in enumerate(columns):
        mask = missing[:, index]
        if not mask.any():
            continue
        if name not in donor or not np.isfinite(donor[name].to_numpy(dtype=float)[mask]).all():
            raise ValueError(f"Same-vintage deterministic issue context is missing {name}")
        result.loc[mask, name] = donor[name].to_numpy()[mask]
        counts[name] = int(mask.sum())
        intervals.update(pd.DatetimeIndex(frame.loc[mask, "interval_start_utc"]).astype(str))
    return result, {"policy": policy, "forecast_vintage_utc": cycle.isoformat(),
        "available_at_utc": pd.Timestamp(deterministic.available_at).isoformat(),
        "filled_member_values_by_feature": counts, "filled_interval_ids": sorted(intervals),
        "donor_weather_sha256": _frame_fingerprint(deterministic.values),
        "scope": "missing pre-delivery model context only; delivery weather unchanged"}


def _prepare_pv_ensembles(config, days_data, progress=None, *, cycle_overrides=None):
    """Convert one specified available weather ensemble per delivery day."""
    import xarray as xr

    from ..data import load_metadata
    from ..forecasting.data.pv_probabilistic_data import align_cycle
    from ..forecasting.models.pv import forecast_pv

    settings = config["continuous_optimization"]["pv_ensemble"]
    policy = config.get("policy", {})
    forecast_night = policy.get("forecast_pv_night", policy.get("night", {}))
    farm_id = int(config["forecast"]["farm_id"])
    sources = settings.get("sources")
    source_inventory = None
    if sources:
        from ..data.ensemble_inventory import resolve_ensemble_sources
        source_inventory = resolve_ensemble_sources([config_path(config, path) for path in sources],
            list(days_data), farm_ids=[farm_id], timezone=config["market"]["timezone"],
            availability_lag_hours=float(config["source"]["availability_lag_hours"]),
            first_cutoffs={day: min(event.submission_at for event in daily["schedule"])
                for day, daily in days_data.items()}, overlap_policy=settings.get("overlap_policy", "error"))
    replacement = config["pv"].get("forecast_replacement", {})
    models = replacement.get("models_by_farm", {})
    has_model = str(farm_id) in models or farm_id in models
    model = models.get(str(farm_id), models.get(farm_id))
    if replacement.get("enabled") and not has_model:
        raise ValueError("PV ensemble model selection is missing for the selected farm")
    use_bundle = bool(replacement.get("enabled") and model is not None)
    if use_bundle:
        if model.get("kind") not in {"weather_bundle", "reviewed_hpo"}:
            raise ValueError("PV ensemble model requires a supported weather model bundle")
        artifact = config_path(config, model["artifact_path"])
        if not artifact.is_file():
            raise ValueError(f"PV ensemble model artifact is missing: {artifact}")
        available_at = pd.Timestamp(model["available_at_utc"])
        if available_at.tzinfo is None:
            raise ValueError("PV ensemble model availability must be timezone-aware")
    farm = load_metadata(config_path(config, config["inputs"]["metadata_parquet"]), [farm_id])[farm_id]
    provenance = {}
    # Open each selected archive once, retaining lazy stores rather than materializing
    # a full annual weather cube. Per-day conversion still reads only its own cycle.
    with ExitStack() as stack:
        datasets = {}
        for day, daily in days_data.items():
            source = (Path(source_inventory[day]["source"]) if source_inventory is not None
                else config_path(config, settings["zarr"]))
            if source not in datasets:
                datasets[source] = stack.enter_context(xr.open_zarr(source, consolidated=True, chunks=None))
            dataset = datasets[source]
            members = tuple(str(value) for value in dataset.member.values)
            expected = settings.get("member_count", "auto")
            if (not members or len(set(members)) != len(members)
                    or expected not in (None, "auto") and len(members) != int(expected)):
                raise ValueError("PV ensemble archive has an unexpected member axis")
            starts = _utc_index(daily["grid"].starts)
            cycle = ((pd.Timestamp(day) - pd.Timedelta(days=1)).tz_localize("UTC")
                if cycle_overrides is None else pd.Timestamp(cycle_overrides[day]))
            issue_hours = None
            if use_bundle and model.get("kind") == "reviewed_hpo":
                metadata = json.loads(artifact.read_text())
                issue_hours = metadata.get("input_contract", {}).get("issue_horizon_hours")
            model_starts = (pd.date_range(cycle, periods=int(issue_hours) * 4, freq="15min")
                if issue_hours is not None else starts)
            requested_positions = model_starts.get_indexer(starts)
            if np.any(requested_positions < 0):
                raise ValueError(f"PV ensemble delivery intervals exceed issue context for {day}")
            try:
                weather = align_cycle(dataset, farm_id, cycle, model_starts, config, ensemble=True)
            except (KeyError, ValueError) as error:
                raise ValueError(f"PV ensemble missing or invalid for {day}, cycle {cycle}: {error}") from error
            first_cutoff = min(event.submission_at for event in daily["schedule"])
            if weather.available_at > first_cutoff:
                raise ValueError(f"PV ensemble for {day} is not available at the first auction cutoff")
            if use_bundle:
                if available_at > pd.Timestamp(first_cutoff) and not replacement.get("allow_retrospective_artifacts", False):
                    raise ValueError(f"PV ensemble model was not available at the first cutoff for {day}")
                # Keep member curves independent for both interpolation and neural sequence packing.
                frame = weather.values.copy().reset_index(drop=False)
                frame["interval_start_utc"] = np.tile(model_starts, len(members))
                frame["delivery_day"] = np.tile(model_starts.tz_convert(config["market"]["timezone"]).strftime("%Y-%m-%d"), len(members))
                frame["auction"] = "DA"
                frame["forecast_cycle_utc"] = cycle
                frame["_scenario_member"] = frame["member"].astype(str)
                context_fill = None
                if issue_hours is not None:
                    frame, context_fill = _fill_prior_issue_context(frame, starts, model_starts,
                        config=config, calendar=AuctionCalendar.from_config(config),
                        cutoff=pd.Timestamp(first_cutoff), cycle=cycle)
                if model["kind"] == "reviewed_hpo":
                    from ..forecasting.inference.pv_reviewed_hpo import predict_reviewed_hpo

                    frame["forecast_available_at_utc"] = pd.Timestamp(weather.available_at)
                    frame["decision_cutoff_utc"] = pd.Timestamp(first_cutoff)
                    predicted = np.empty(len(frame), dtype=float)
                    for member in members:
                        member_mask = frame["_scenario_member"].to_numpy() == member
                        predicted[member_mask] = predict_reviewed_hpo(frame.loc[member_mask].copy(),
                            artifact.parent, farm_id=farm_id, night_policy=forecast_night)
                else:
                    from ..forecasting.inference.pv_forecast_bundle import predict_bundle

                    predicted = predict_bundle(frame, artifact.parent, farm_id=farm_id,
                                               night_policy=forecast_night)
                values = np.stack([predicted[frame["_scenario_member"].to_numpy() == member][requested_positions]
                                   for member in members])
                conversion = {"model": model.get("name"), "kind": model["kind"],
                    "manifest": str(artifact), "manifest_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
                    "available_at_utc": available_at.isoformat(),
                    "retrospective": bool(available_at > pd.Timestamp(first_cutoff)),
                    "model_issue_context_hours": issue_hours,
                    "model_issue_context_weather_sha256": _frame_fingerprint(frame),
                    "model_issue_context_fill": context_fill}
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
                "conversion": conversion, "forecast_pv_night_policy": forecast_night,
                **({"source_inventory": source_inventory[day]} if source_inventory is not None else {})}
            if progress:
                progress(f"Preflight {day}: converted {len(members)} raw PV ensemble members")
    return provenance


def _prepare_ensemble_updates(config, days_data, base_provenance, progress=None):
    """Prepare newer available cycles for their covered remaining delivery intervals."""
    import xarray as xr
    from types import SimpleNamespace

    settings = config["continuous_optimization"]["pv_ensemble"]
    paths = settings.get("sources") or [settings["zarr"]]
    paths = [config_path(config, path) for path in paths]
    overlap = settings.get("overlap_policy", "error")
    latency = pd.Timedelta(hours=float(config["source"]["availability_lag_hours"]))
    earliest = min(pd.Timestamp(value["forecast_vintage_utc"]) for value in base_provenance.values())
    latest = max(pd.Timestamp(daily["grid"].starts[-1]) for daily in days_data.values())
    # Read archive coordinates once; model inference is deferred until a cycle
    # can actually be used during one of the requested delivery days.
    candidates = {}
    for path in paths:
        with xr.open_zarr(path, consolidated=True, chunks=None) as dataset:
            cycles = pd.DatetimeIndex(pd.to_datetime(dataset.cycle_time.values, utc=True))
            max_lead = pd.to_timedelta(dataset.lead_time.values).max()
            for cycle in cycles:
                if cycle <= earliest or cycle + latency > latest:
                    continue
                if cycle in candidates and overlap == "error":
                    raise ValueError(f"Multiple ensemble sources contain cycle {cycle}")
                if cycle not in candidates or overlap == "last":
                    candidates[cycle] = (path, max_lead)
    for day, daily in days_data.items():
        starts = _utc_index(daily["grid"].starts)
        first_cutoff = min(pd.Timestamp(event.submission_at) for event in daily["schedule"])
        initial_cycle = pd.Timestamp(base_provenance[day]["forecast_vintage_utc"])
        updates = []
        for cycle, (source, max_lead) in sorted(candidates.items()):
            available = cycle + latency
            if cycle <= initial_cycle or available > starts[-1]:
                continue
            eligible_at = max(first_cutoff, available)
            first = int(starts.searchsorted(eligible_at.ceil("15min")))
            if first >= len(starts) or cycle + max_lead < starts[-1] + pd.Timedelta(hours=1):
                continue
            grid = daily["grid"]
            remaining = DeliveryGrid(grid.interval_ids[first:], grid.starts[first:], grid.ends[first:])
            temporary = {day: {"grid": remaining, "schedule": [SimpleNamespace(submission_at=eligible_at)]}}
            update_config = deepcopy(config)
            update_settings = update_config["continuous_optimization"]["pv_ensemble"]
            update_settings["sources"] = None
            update_settings["zarr"] = str(source)
            info = _prepare_pv_ensembles(update_config, temporary, cycle_overrides={day: cycle})[day]
            updates.append({"starts": starts[first:], "values": temporary[day]["ensemble_pv"],
                            "members": temporary[day]["ensemble_members"], "provenance": info})
        daily["ensemble_updates"] = updates
        base_provenance[day]["available_updates"] = [entry["provenance"] for entry in updates]
        if progress:
            progress(f"Preflight {day}: prepared {len(updates)} newer ensemble cycles")


def ensemble_at_cutoff(daily, info, cutoff, starts):
    """Select the latest eligible ensemble covering every requested interval."""
    cutoff = pd.Timestamp(cutoff)
    starts = _utc_index(starts)
    entries = [{"starts": _utc_index(daily["grid"].starts), "values": daily["ensemble_pv"],
                "members": daily["ensemble_members"], "provenance": info}]
    entries.extend(daily.get("ensemble_updates", []))
    eligible = []
    for entry in entries:
        provenance = entry["provenance"]
        if pd.Timestamp(provenance["available_at_utc"]) > cutoff:
            continue
        positions = entry["starts"].get_indexer(starts)
        if np.all(positions >= 0):
            eligible.append((pd.Timestamp(provenance["forecast_vintage_utc"]), entry, positions))
    if not eligible:
        raise ValueError("No available ensemble cycle covers the requested delivery intervals")
    _, chosen, positions = max(eligible, key=lambda item: item[0])
    return chosen["values"][:, positions], chosen["members"], chosen["provenance"]


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
    extended_horizon = bool(settings.get("ida3_next_day_horizon", False))
    # The same-day tracker may explicitly request complete issue-context neural
    # forecasts, matching the extended IDA3 studies without changing old runs.
    require_replacement = settings.get("pin_tracker_forecast_replacement", False)
    if not isinstance(require_replacement, bool):
        raise ValueError("continuous_optimization.pin_tracker_forecast_replacement must be a boolean")
    replacement_evidence = (_replacement_evidence(config)
                            if uses_realtime_control(config) or extended_horizon or require_replacement else None)
    if require_replacement and (replacement_evidence is None or replacement_evidence.get("status") != "pinned"):
        raise ValueError("Same-day neural tracker requires an available pinned forecast replacement artifact")
    try:
        first = date.fromisoformat(str(settings["start_date"]))
        last = date.fromisoformat(str(settings["end_date"]))
    except (KeyError, TypeError, ValueError):
        raise ValueError("Continuous optimization requires ISO start_date and end_date delivery dates") from None
    # Enforce the input contract: Continuous optimization end_date must not precede start_date.
    if last < first:
        raise ValueError("Continuous optimization end_date must not precede start_date")
    context_end = date.fromisoformat(str(settings.get("context_end_date", last)))
    if extended_horizon and context_end < last + pd.Timedelta(days=1):
        raise ValueError("IDA3 next-day planning requires at least one context delivery day")
    if not extended_horizon and context_end != last:
        raise ValueError("Context delivery days require IDA3 next-day planning")
    days = tuple(day.date().isoformat() for day in pd.date_range(first, last, freq="D"))
    context_days = tuple(day.date().isoformat() for day in pd.date_range(
        last + pd.Timedelta(days=1), context_end, freq="D"))
    input_days = days + context_days
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
    all_grids = [calendar.delivery_grid(day).to_domain() for day in input_days]
    scored_grids = all_grids[:len(days)]
    full_grid = DeliveryGrid(tuple(item for grid in scored_grids for item in grid.interval_ids),
        tuple(item for grid in scored_grids for item in grid.starts),
        tuple(item for grid in scored_grids for item in grid.ends))
    full_starts = _utc_index(tuple(item for grid in all_grids for item in grid.starts))
    issue_cache = {}
    initial_requests: dict[str, tuple[pd.DatetimeIndex, Any]] = {}
    forecast_requests: dict[str, pd.DatetimeIndex] = {}
    # Pin exact prices, common measured PV/fallbacks and every causal forecast when `progress`.
    if progress:
        progress(f"Preflighting continuous delivery inputs for {input_days[0]} through {input_days[-1]} (offline)")

    # Reuse verified workbook snapshots across days, keeping memory bounded and
    # releasing database connections before weather conversion or inference.
    from ..data.api_catalog import SnapshotReadCache
    price_snapshot_cache = SnapshotReadCache(settings.get("price_snapshot_cache_entries", 8))
    # Read actual production once per day, with one strategy-independent fallback.
    for day, grid in zip(input_days, all_grids):
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
                config, day, first_cutoff, grid, bundle, forecast, actual, pv_provenance, replacement_evidence,
                calendar=calendar, issue_cache=issue_cache)
            _validate_bundle(bundle, first_cutoff, _utc_index(grid.starts))
        except (ValueError, RuntimeError) as error:
            raise ValueError(f"Continuous preflight for {day}: {error}") from error
        initial_requests[pd.Timestamp(first_cutoff).tz_convert("UTC").isoformat()] = (_utc_index(grid.starts), bundle)
        days_data[day] = {"grid": grid, "schedule": schedule, "prices": prices,
            "research_initial_cutoff": first_cutoff.isoformat(),
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
            forecast_end = pd.Timestamp(grid.ends[-1])
            if extended_horizon and event.auction == "IDA3" and date.fromisoformat(day) <= last:
                forecast_end = pd.Timestamp(calendar.delivery_grid(
                    (date.fromisoformat(day) + pd.Timedelta(days=1)).isoformat()).ends[-1])
            # Include the in-progress interval for its forecast-only remainder.
            starts = full_starts[(full_starts >= cutoff.floor("15min"))
                                 & (full_starts < forecast_end)]
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
                if config.get("realtime_mpc", {}).get("terminal_inventory_mode") == "piecewise_linear":
                    from .energy_price_inputs import continuation_grid
                    extra = continuation_grid(config, grid.ends[-1])
                    starts = starts.union(_utc_index(extra.starts))
                forecast_requests[key] = forecast_requests.get(key, starts).union(starts)

    # Freeze union forecasts covering both current-day projection and next-day plan.
    forecasts = _PreparedForecasts()
    # Pin exact prices, common measured PV/fallbacks and every causal forecast for each `(key, (starts, _))` in `initial_requests.items()`.
    for key, (starts, _) in initial_requests.items():
        forecast_requests[key] = forecast_requests.get(key, starts).union(starts)
    # MPC has many more cutoffs than auctions; distinguish this scan from the
    # completed daily checks and report progress during long cached-data reads.
    forecast_request_count = len(forecast_requests)
    show_forecast_progress = progress is not None and (
        uses_realtime_control(config) or extended_horizon or require_replacement)
    if show_forecast_progress:
        progress(f"Daily input checks complete; preflighting {forecast_request_count} forecast cutoffs")
    with _forecast_preflight_session(config, calendar) as forecast_reader:
        for request_index, (key, starts) in enumerate(sorted(forecast_requests.items()), start=1):
            initial = initial_requests.get(key)
            try:
                bundle = (initial[1] if initial is not None and starts.equals(initial[0]) else
                          _load_pv_forecast(config, calendar, pd.Timestamp(key), starts))
                if (_reviewed_issue_hours(config, replacement_evidence) is not None
                        and (pd.Timestamp(replacement_evidence["available_at_utc"]) <= pd.Timestamp(key)
                            or config["pv"]["forecast_replacement"].get("allow_retrospective_artifacts", False))
                        and bundle.provenance.get("reviewed_hpo_model") != replacement_evidence["model"]):
                    bundle = _issue_context_prediction(config, calendar, pd.Timestamp(key), bundle,
                        replacement_evidence, issue_cache)
                forecasts.add(pd.Timestamp(key), starts, bundle)
            except (ValueError, RuntimeError) as error:
                raise ValueError(f"Continuous weather preflight at {key}: {error}") from error
            progress_interval = 10 if extended_horizon or require_replacement else 100
            if show_forecast_progress and (request_index % progress_interval == 0 or request_index == forecast_request_count):
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
        if settings.get("pv_ensemble", {}).get("cycle_updates", "frozen_initial") == "latest_available":
            _prepare_ensemble_updates(config, days_data, ensemble_provenance, progress)
    settlement, settlement_provenance = load_realized_imbalance_prices(config, full_grid)
    settlement_values = (None if settlement is None else
                         settlement.settlement_price_eur_mwh.to_numpy(float, copy=True))
    full_actual = np.concatenate([days_data[day]["realized_pv"] for day in days])
    curated = config.get("forecast", {}).get("curated_realized", {}).get("enabled", False)
    provenance = {"schema_version": "continuous-inputs-v1", "offline": True,
        "start_date": days[0], "end_date": days[-1], "delivery_timezone": calendar.timezone,
        "measured_pv_basis": ("reviewed AC MW; may include retrospective pvlib and night-zero counterfactuals"
                              if curated else "AC; no conversion losses applied"),
        "realized_pv_fallback": ("no forecast fallback; retained exclusions use reviewed historical-GHI pvlib and night policy"
                                 if curated else "one initial frozen forecast per day shared by all strategies"),
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
    if extended_horizon:
        provenance["scored_end_date"] = last.isoformat()
        provenance["context_end_date"] = context_end.isoformat()
    prepared = {"calendar": calendar, "days": days, "days_data": days_data,
        "plant": plant, "initial_energy_mwh": initial_energy, "terminal_energy_mwh": terminal_energy,
        "terminal_minimum_energy_mwh": terminal_minimum,
        "full_grid": full_grid, "full_realized_pv": full_actual,
        "forecast_provider": forecasts, "imbalance_prices_eur_per_mwh": settlement_values,
        "settlement_frame": settlement, "provenance": provenance}
    if extended_horizon:
        prepared["context_days"] = context_days
    from .energy_price_inputs import prepare_energy_price_provider
    prepare_energy_price_provider(config, prepared)
    prepared["input_fingerprint"] = _fingerprint(provenance)
    return prepared
