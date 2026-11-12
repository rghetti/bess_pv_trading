"""Read-only price artifacts and causal inputs for the energy-market studies."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from ..data import config_path, night_window_mask
from ..domain import DeliveryGrid
from ..forecasting.energy_prices import (CausalPriceForecaster, prepare_price_history,
    prepare_exogenous_features)
from ..forecasting.continuation import ContinuationSettings, build_continuation_value
from ..optimization.mpc import ForecastPrices
from ..scenario_research import ScenarioTrajectories, ReductionSettings, reduce_scenarios, transform_dependence
from .continuous_inputs import _json_value
from .research_inputs import price_information_cutoff


def validate_energy_research_settings(config):
    """Reject misspelled switches and unusable economic input combinations."""
    settings = config.get("energy_price_forecast", {})
    allowed = {"auction_enabled", "settlement_enabled", "artifacts", "seed", "exogenous_path", "history_paths"}
    if not isinstance(settings, dict) or set(settings) - allowed:
        raise ValueError("Unknown energy_price_forecast settings")
    for name in ("auction_enabled", "settlement_enabled"):
        if not isinstance(settings.get(name, False), bool):
            raise ValueError(f"energy_price_forecast.{name} must be a boolean")
    seed = settings.get("seed", 0)
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("energy_price_forecast.seed must be a nonnegative integer")
    artifacts = settings.get("artifacts", {})
    if not isinstance(artifacts, dict) or set(artifacts) - {"DA", "IDA1", "IDA2", "IDA3", "settlement", "settlement_negative"}:
        raise ValueError("Unknown energy price artifact identifiers")
    if any(not isinstance(path, str) or not path for path in artifacts.values()):
        raise ValueError("Price artifacts must be nonempty paths")
    history_paths = settings.get("history_paths", {})
    if (not isinstance(history_paths, dict) or set(history_paths) - set(artifacts)
            or any(not isinstance(path, str) or not path for path in history_paths.values())):
        raise ValueError("Price history_paths must identify configured model artifacts")
    if settings.get("settlement_enabled") and "settlement" not in artifacts:
        raise ValueError("Causal settlement economics requires a settlement forecast artifact")
    if settings.get("auction_enabled"):
        auctions = config.get("gme_oracle_experiment", {}).get("participating_auctions") or ["DA", "IDA1", "IDA2", "IDA3"]
        if set(auctions) - artifacts.keys():
            raise ValueError("Causal auction planning requires an artifact for every participating auction")
    scenarios = config.get("realtime_scenarios", {})
    if not isinstance(scenarios, dict) or set(scenarios) - {"dependence", "seed", "reduce_to", "reduction", "calibrated_marginals_path"}:
        raise ValueError("Unknown realtime_scenarios settings")
    if scenarios.get("dependence", "raw") not in {"raw", "independent_shuffle", "ecc"}:
        raise ValueError("Unknown realtime scenario dependence mode")
    if scenarios.get("dependence") == "ecc" and not scenarios.get("calibrated_marginals_path"):
        raise ValueError("ECC requires a calibrated marginal artifact with availability evidence")
    count = scenarios.get("reduce_to")
    if count is not None and (isinstance(count, bool) or not isinstance(count, int) or count < 1):
        raise ValueError("Scenario reduce_to must be a positive integer or null")
    scenario_seed = scenarios.get("seed", 0)
    if isinstance(scenario_seed, bool) or not isinstance(scenario_seed, int) or scenario_seed < 0:
        raise ValueError("realtime_scenarios.seed must be a nonnegative integer")
    ReductionSettings(**scenarios.get("reduction", {}))
    continuation = config.get("continuation_value", {})
    if not isinstance(continuation, dict) or set(continuation) - {"horizon_hours", "settings"}:
        raise ValueError("Unknown continuation_value settings")
    ContinuationSettings(**continuation.get("settings", {}))
    if config.get("realtime_mpc", {}).get("terminal_inventory_mode") == "piecewise_linear":
        hours = continuation.get("horizon_hours", 6)
        if isinstance(hours, bool) or not isinstance(hours, (int, float)) or not np.isfinite(hours) or hours <= 0 or hours * 4 != int(hours * 4):
            raise ValueError("Continuation horizon_hours must contain positive whole quarter hours")
        if not settings.get("settlement_enabled", False):
            raise ValueError("Conditional continuation requires eligible settlement forecasts")
        if config.get("research_information", {}).get("pv_updates", "latest") != "latest":
            raise ValueError("Conditional continuation currently requires latest eligible PV forecasts")


class EnergyPriceProvider:
    """Load each fitted model once; prediction never fits or opens DuckDB."""

    def __init__(self, config):
        validate_energy_research_settings(config)
        settings = config.get("energy_price_forecast", {})
        self.seed = settings.get("seed", 0)
        self.models, self.provenance, self.history = {}, {}, {}
        required = set()
        if settings.get("auction_enabled"):
            required.update(config.get("gme_oracle_experiment", {}).get("participating_auctions") or
                ["DA", "IDA1", "IDA2", "IDA3"])
        if settings.get("settlement_enabled"):
            required.add("settlement")
            if "settlement_negative" in settings.get("artifacts", {}):
                required.add("settlement_negative")
        for name, relative in settings.get("artifacts", {}).items():
            if name not in required:
                continue
            path = config_path(config, relative)
            self.models[name] = CausalPriceForecaster.load(path)
            self.provenance[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        # Observation sidecars may include later labels; predict filters their clocks.
        for name, relative in settings.get("history_paths", {}).items():
            if name not in required:
                continue
            path = config_path(config, relative)
            self.history[name] = prepare_price_history(pd.read_parquet(path),
                timezone=self.models[name].settings.timezone)
            self.provenance[name]["observation_history"] = {"path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        self.exogenous = None
        if settings.get("exogenous_path") and any(model.settings.exogenous_features for model in self.models.values()):
            path = config_path(config, settings["exogenous_path"])
            self.exogenous = prepare_exogenous_features(pd.read_parquet(path))
            self.provenance["exogenous"] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def predict(self, name, cutoff, starts, scenario_count=1):
        """Give repeated calls at the same information cutoff reproducible draws."""
        if name not in self.models:
            raise ValueError(f"Missing causal price model for {name}")
        # Reuse each daily residual path as the horizon shrinks during dispatch.
        day = pd.Timestamp(starts[0]).tz_convert(self.models[name].settings.timezone).date().isoformat()
        identity = f"{self.seed}:{name}:{day}"
        seed = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:4], "little")
        predicted = self.models[name].predict(starts, cutoff=cutoff, scenario_count=scenario_count,
            history=self.history.get(name), exogenous=self.exogenous, seed=seed)
        return replace(predicted, provenance={**predicted.provenance,
            "artifact": self.provenance[name], "pv_price_coupling": "independent_residual_sampling_assumption"})


def prepare_energy_price_provider(config, prepared):
    """Attach immutable model evidence before the replay fingerprint is finalized."""
    settings = config.get("energy_price_forecast", {})
    if not (settings.get("auction_enabled") or settings.get("settlement_enabled")):
        return
    provider = EnergyPriceProvider(config)
    prepared["energy_price_provider"] = provider
    prepared["provenance"]["energy_price_forecasts"] = provider.provenance
    checks = 0
    controllers = set(config.get("realtime_mpc_comparison", {}).get("controllers", []))
    controllers.add(config.get("execution", {}).get("dispatch_policy", ""))
    scenario_count = 2 if any(name.startswith("ensemble_mpc") for name in controllers) else 1
    # Verify every requested information set before starting any optimizer.
    # Predictions are cheap array operations on prepared observations, never fits.
    for daily in prepared["days_data"].values():
        starts = pd.DatetimeIndex(daily["grid"].starts)
        if settings.get("auction_enabled"):
            for event in daily["schedule"]:
                cutoff = price_information_cutoff(config, daily, event.submission_at)
                for future in daily["schedule"]:
                    if future.results_at <= event.submission_at:
                        continue
                    eligible = starts >= cutoff
                    if future.auction == "IDA3":
                        eligible &= starts.tz_convert(prepared["calendar"].timezone).hour >= 12
                    provider.predict(future.auction, cutoff, starts[eligible], 1)
                    checks += 1
        if settings.get("settlement_enabled"):
            for index, decision in enumerate(starts):
                cutoff = price_information_cutoff(config, daily, decision)
                # Validate both deterministic and empirical-residual paths; two
                # members suffice to test calibration coverage without big arrays.
                provider.predict("settlement", cutoff, starts[index:], scenario_count)
                checks += 1
                if "settlement_negative" in provider.models:
                    provider.predict("settlement_negative", cutoff, starts[index:], scenario_count)
                    checks += 1
                if config.get("realtime_mpc", {}).get("terminal_inventory_mode") == "piecewise_linear":
                    extra = continuation_grid(config, daily["grid"].ends[-1])
                    provider.predict("settlement", cutoff, pd.DatetimeIndex(extra.starts), 1)
                    checks += 1
                    if "settlement_negative" in provider.models:
                        provider.predict("settlement_negative", cutoff, pd.DatetimeIndex(extra.starts), 1)
                        checks += 1
    prepared["provenance"]["energy_price_prediction_checks"] = checks


def settlement_forecasts(replay, daily, cutoff, starts, count):
    """Build a forecast-only settlement contract, separately from actual prices."""
    if not replay.config.get("energy_price_forecast", {}).get("settlement_enabled", False):
        return None
    provider = replay.prepared.get("energy_price_provider")
    if provider is None:
        raise ValueError("Causal settlement provider was not preflighted")
    selected = price_information_cutoff(replay.config, daily, cutoff)
    positive = provider.predict("settlement", selected, starts, scenario_count=count)
    negative = None
    if "settlement_negative" in provider.models:
        negative = provider.predict("settlement_negative", selected, starts, scenario_count=count)
    available = positive.available_at if negative is None else max(positive.available_at, negative.available_at)
    if available > pd.Timestamp(cutoff):
        raise ValueError("Directional settlement forecast is unavailable at the decision cutoff")
    return ForecastPrices(positive.scenarios_eur_per_mwh, available,
        None if negative is None else negative.scenarios_eur_per_mwh,
        {"positive": positive.provenance, "negative": None if negative is None else negative.provenance,
         "forecast_cutoff_utc": selected.isoformat()})


def _load_calibration_artifact(path, cache):
    """Freeze one owned, checksummed marginal artifact for the lifetime of a replay."""
    key = str(Path(path).resolve())
    if cache is not None and key in cache:
        return cache[key]
    with np.load(path, allow_pickle=False) as archive:
        result = {name: np.array(archive[name], copy=True) for name in archive.files}
    result["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    for value in result.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    if cache is not None:
        cache[key] = result
    return result


def transform_mpc_scenarios(config, grid, cutoff, plant, values, ids, weights, commitments,
                            price_forecast=None, *, artifact_cache=None):
    """Apply coupling before reduction so retained members carry probability mass."""
    settings = config.get("realtime_scenarios", {})
    bundle = ScenarioTrajectories(values, weights, ids,
        None if price_forecast is None else price_forecast.positive_eur_per_mwh,
        None if price_forecast is None else price_forecast.negative_eur_per_mwh)
    mode = settings.get("dependence", "raw")
    samples = None
    if settings.get("calibrated_marginals_path"):
        path = config_path(config, settings["calibrated_marginals_path"])
        artifact = _load_calibration_artifact(path, artifact_cache)
        available = pd.Timestamp(str(artifact["available_at_utc"].item()))
        if pd.isna(available) or available.tzinfo is None or available > cutoff:
            raise ValueError("Calibrated PV marginals are unavailable at this decision")
        kind = str(artifact.get("artifact_kind", artifact.get(
            "artifactkind", np.array("precomputed_marginals"))).item())
        if kind not in {"affine_marginal_calibration", "precomputed_marginals"}:
            raise ValueError("Unknown PV calibration artifact kind")
        if kind == "affine_marginal_calibration":
            # Learned parameters act on the current eligible forecast. This keeps
            # calibration valid when member counts or forecast cycles expand.
            bias = float(artifact["calibration_bias_normalized"].item())
            spread = float(artifact["calibration_spread_scale"].item())
            capacity = float(artifact["ac_nameplate_mw"].item())
            fitted = pd.Timestamp(str(artifact["calibration_fit_cutoff_utc"].item()))
            if pd.isna(fitted) or fitted.tzinfo is None or fitted > available:
                raise ValueError("PV calibration fit cutoff must precede artifact availability")
            if not np.isfinite([bias, spread, capacity]).all() or spread < 0 or capacity <= 0:
                raise ValueError("Invalid affine PV calibration parameters")
            if int(artifact["farm_id"].item()) != int(plant.farm_id):
                raise ValueError("PV calibration artifact belongs to a different farm")
            mean = weights @ values
            samples = np.clip(mean + bias * capacity + spread * (values - mean), 0, capacity)
            policy = config.get("policy", {})
            night = policy.get("forecast_pv_night", policy.get("night", {}))
            samples[:, night_window_mask(pd.DatetimeIndex(grid.starts), night)] = 0.0
        else:
            # Precomputed samples need their own target and publication clocks;
            # the entire artifact need not have been available on its first day.
            starts = pd.DatetimeIndex(artifact["interval_starts_utc"].astype(str))
            positions = starts.get_indexer(pd.DatetimeIndex(grid.starts))
            if np.any(positions < 0):
                raise ValueError("Calibrated PV marginals are unavailable or do not cover this horizon")
            samples = artifact["pv_ac_mw"][:, positions]
            if "forecast_available_at_utc" in artifact:
                clocks = pd.DatetimeIndex(artifact["forecast_available_at_utc"].astype(str))[positions]
                if clocks.tz is None or clocks.hasnans or np.any(clocks > cutoff):
                    raise ValueError("Calibrated marginal forecast vintage is after the cutoff")
        bundle = replace(bundle, provenance={"calibrated_marginals_sha256": artifact["sha256"],
            "calibrated_marginals_available_at_utc": available.isoformat(), "calibration_kind": kind})
    if samples is not None:
        # Couple the same calibrated marginal law before any dependence ablation.
        # The shuffled comparison therefore changes paths, not marginal quality.
        transformed = transform_dependence(bundle, "ecc", seed=settings.get("seed", 0),
            calibrated_pv_marginals=samples)
        if mode == "independent_shuffle":
            transformed = transform_dependence(transformed, mode, seed=settings.get("seed", 0))
    else:
        transformed = transform_dependence(bundle, mode, seed=settings.get("seed", 0))
    selected = np.arange(len(ids))
    count = settings.get("reduce_to")
    if count is not None and count < len(ids):
        result = reduce_scenarios(transformed, count, settings=ReductionSettings(**settings.get("reduction", {})),
            committed_mwh=commitments, delta_hours=grid.delta_hours,
            battery_capacity_mwh=plant.battery.capacity_mwh,
            poi_import_mw=plant.poi_import_mw, poi_export_mw=plant.poi_export_mw)
        transformed, selected = result.bundle, result.selected_indices
    elif count is not None and count > len(ids):
        raise ValueError("Scenario reduction cannot increase the represented ensemble")
    prices = None if price_forecast is None else replace(price_forecast,
        positive_eur_per_mwh=transformed.positive_prices_eur_per_mwh,
        negative_eur_per_mwh=transformed.negative_prices_eur_per_mwh)
    return transformed, prices, selected, _json_value(transformed.provenance)


def continuation_grid(config, horizon_end):
    """Describe forecast-only products beyond the physical MPC delivery book."""
    count = int(config.get("continuation_value", {}).get("horizon_hours", 6) * 4)
    boundaries = pd.date_range(horizon_end, periods=count + 1, freq="15min")
    return DeliveryGrid(tuple(f"continuation:{stamp.isoformat()}" for stamp in boundaries[:-1]),
        tuple(boundaries[:-1].to_pydatetime()), tuple(boundaries[1:].to_pydatetime()))


def conditional_terminal_value(replay, daily, cutoff, horizon_end):
    """Value energy/headroom using available forecasts beyond the dispatch horizon."""
    # An energy-only continuation must not silently assume nameplate capacity.
    if replay.aging.mode == "linear_soh":
        raise ValueError("linear_soh is not yet supported by MPC continuation values")
    if replay.config.get("realtime_mpc", {}).get("terminal_inventory_mode") != "piecewise_linear":
        return None
    grid = continuation_grid(replay.config, horizon_end)
    pv, provenance = replay._forecast(cutoff, pd.DatetimeIndex(grid.starts), "continuation", causal=True)
    forecast = settlement_forecasts(replay, daily, cutoff, pd.DatetimeIndex(grid.starts), 1)
    # Only already submitted quantities enter continuation; future trades stay unknown.
    by_start = dict(zip(replay.grid.starts, replay.grid.interval_ids))
    commitments = np.array([replay._position(by_start[start]) if start in by_start else 0.0 for start in grid.starts])
    settings = dict(replay.config.get("continuation_value", {}).get("settings", {}))
    settings.setdefault("throughput_cost_eur_per_mwh", replay.aging.cost_eur_per_mwh_throughput)
    settings.setdefault("import_fee_eur_per_mwh", replay.trading["grid_import_fee_eur_per_mwh"])
    plant = replay.plant
    health_approximation = {}
    if replay.aging.mode == "naumann_lfp":
        from ..optimization.naumann import available_capacity_mwh, effective_battery, power_limits

        if settings["throughput_cost_eur_per_mwh"] != 0:
            raise ValueError("Naumann energy-only continuation does not add a throughput wear penalty")
        operating_battery = effective_battery(replay.state, plant.battery, replay.aging)
        charge_limit, discharge_limit = power_limits(replay.state, plant.battery, replay.aging)
        operating_battery = replace(operating_battery, max_power_mw=min(charge_limit, discharge_limit))
        plant = replace(plant, battery=operating_battery)
        # This existing continuation DP has energy as its only state. Freeze
        # only observed capacity, disclose omitted tail health and cycle memory,
        # and retain Naumann wear inside the physical optimization horizon.
        health_approximation = {"capacity_reference": "committed_observed_at_cutoff",
            "capacity_mwh": available_capacity_mwh(replay.state, replay.plant.battery, replay.aging),
            "observed_soh": replay.state.battery_health.committed_soh,
            "symmetric_power_limit_mw": operating_battery.max_power_mw,
            "future_capacity_fade_modelled": False, "cycle_memory_modelled": False,
            "calendar_and_cycle_wear_modelled": False,
            "wear_scope": "Naumann wear is charged in the physical horizon; continuation values energy only"}
    return build_continuation_value(cutoff=cutoff, available_at=cutoff, grid=grid, plant=plant,
        positive_eur_per_mwh=forecast.positive_eur_per_mwh[0], pv_ac_mw=pv, committed_mwh=commitments,
        negative_eur_per_mwh=None if forecast.negative_eur_per_mwh is None else forecast.negative_eur_per_mwh[0],
        settings=ContinuationSettings(**settings), provenance={"pv": provenance, "prices": forecast.provenance,
            "actual_degradation_mode": replay.aging.mode, "forecast_is_oracle": False,
            **({"battery_health_approximation": health_approximation} if health_approximation else {})})
