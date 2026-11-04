"""Chronological, two-parameter CRPS calibration of physical PV ensembles.

Only forecast distributions are transformed. Raw AC remains in the audit;
the configured target can be raw measurements or explicitly reviewed,
retrospective counterfactual production with per-interval source labels.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd

from ..scenario_research import ScenarioTrajectories, transform_dependence, weighted_crps


def _aware(value, label):
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone aware")
    return stamp.tz_convert("UTC")


def _array_hash(values):
    """Pin numerical arrays including their shape and native byte representation."""
    array = np.ascontiguousarray(values)
    digest = hashlib.sha256(str((array.dtype.str, array.shape)).encode())
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class CalibrationSettings:
    """One predeclared affine distribution correction, shared across lead times."""

    bias_bounds_normalized: tuple[float, float] = (-0.3, 0.3)
    spread_bounds: tuple[float, float] = (0.25, 3.0)
    max_evaluations: int = 1000
    parameter_tolerance: float = 1e-6
    objective_tolerance: float = 1e-8
    minimum_training_intervals: int = 96
    central_interval_probability: float = 0.8

    def __post_init__(self):
        for name, baseline in (("bias_bounds_normalized", 0.0), ("spread_bounds", 1.0)):
            values = tuple(getattr(self, name))
            if (len(values) != 2 or any(isinstance(value, bool) or not np.isfinite(value) for value in values)
                    or not values[0] < values[1] or not values[0] <= baseline <= values[1]):
                raise ValueError(f"Calibration {name} must be finite ordered bounds containing {baseline}")
            object.__setattr__(self, name, values)
        if self.spread_bounds[0] < 0:
            raise ValueError("Calibration spread scale cannot be negative")
        for name in ("max_evaluations", "minimum_training_intervals"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"Calibration {name} must be a positive integer")
        for name in ("parameter_tolerance", "objective_tolerance"):
            value = getattr(self, name)
            if isinstance(value, bool) or not np.isfinite(value) or value <= 0:
                raise ValueError(f"Calibration {name} must be positive and finite")
        if not 0 < self.central_interval_probability < 1:
            raise ValueError("Central interval probability must lie strictly between zero and one")


def apply_bias_spread(raw_pv_ac_mw, *, bias_normalized, spread_scale, ac_nameplate_mw, night_mask=None):
    """Transform each marginal while preserving raw member ranks and AC bounds."""
    values = np.asarray(raw_pv_ac_mw, dtype=float)
    if values.ndim != 2 or min(values.shape) < 1 or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("PV calibration requires a finite nonnegative scenario-by-time matrix")
    if (not np.isfinite(ac_nameplate_mw) or ac_nameplate_mw <= 0 or not np.isfinite(bias_normalized)
            or not np.isfinite(spread_scale) or spread_scale < 0):
        raise ValueError("PV calibration requires positive AC nameplate and finite nonnegative spread")
    mean = values.mean(axis=0)
    calibrated = np.clip(mean + bias_normalized * ac_nameplate_mw + spread_scale * (values - mean), 0, ac_nameplate_mw)
    if night_mask is not None:
        night = np.asarray(night_mask)
        if night.dtype != bool or night.shape != (values.shape[1],):
            raise ValueError("Forecast-night mask must be an aligned boolean vector")
        calibrated[:, night] = 0.0
    return calibrated


def fit_bias_spread(raw_pv_ac_mw, observed_ac_mw, *, ac_nameplate_mw, interval_starts,
                    observed_available_at, fit_cutoff, eligible=None, night_mask=None, settings=None):
    """Fit normalized empirical CRPS using only labels available by the fit cutoff."""
    from scipy.optimize import minimize

    settings = settings or CalibrationSettings()
    cutoff = _aware(fit_cutoff, "Calibration fit cutoff")
    values = np.asarray(raw_pv_ac_mw, dtype=float)
    # Validate the forecast law and physical scaling before filtering target rows.
    apply_bias_spread(values, bias_normalized=0, spread_scale=1, ac_nameplate_mw=ac_nameplate_mw, night_mask=night_mask)
    starts = pd.DatetimeIndex([_aware(value, "Target interval start") for value in interval_starts])
    available = pd.DatetimeIndex([_aware(value, "Target availability") for value in observed_available_at])
    actual = np.asarray(observed_ac_mw, dtype=float)
    if (len(starts) != values.shape[1] or len(available) != len(starts) or actual.shape != (len(starts),)
            or starts.has_duplicates or not starts.is_monotonic_increasing or np.any(available < starts + pd.Timedelta(minutes=15))):
        raise ValueError("Calibration targets and completed-interval availability must align uniquely")
    valid = np.ones(len(starts), dtype=bool) if eligible is None else np.asarray(eligible)
    if valid.dtype != bool or valid.shape != actual.shape:
        raise ValueError("Calibration eligibility must be an aligned boolean mask")
    valid = valid & np.isfinite(actual) & (actual >= 0) & (available <= cutoff) & (starts < cutoff)
    if int(valid.sum()) < settings.minimum_training_intervals:
        raise ValueError("Insufficient eligible, causally available calibration targets")
    # Positive spread and clipping preserve ranks, so sort raw members only once.
    normalized = values[:, valid] / ac_nameplate_mw
    means = normalized.mean(axis=0)
    deviations = np.sort(normalized, axis=0) - means
    truth = actual[valid] / ac_nameplate_mw
    count = len(values)
    pairwise_coefficients = (2 * np.arange(count) - count + 1)[:, None] / (count * count)
    night = np.zeros(len(truth), dtype=bool) if night_mask is None else np.asarray(night_mask)[valid]

    def objective(parameters):
        # Exact finite-law CRPS: first absolute moment minus half the pairwise term.
        calibrated = np.clip(means + parameters[0] + parameters[1] * deviations, 0, 1)
        calibrated[:, night] = 0
        crps = np.abs(calibrated - truth).mean(axis=0) - np.sum(calibrated * pairwise_coefficients, axis=0)
        return float(crps.mean())

    initial = np.array([0.0, 1.0])
    initial_objective = objective(initial)
    started = perf_counter()
    result = minimize(objective, initial, method="Powell",
        bounds=[settings.bias_bounds_normalized, settings.spread_bounds],
        options={"maxfev": settings.max_evaluations, "xtol": settings.parameter_tolerance,
                 "ftol": settings.objective_tolerance})
    candidate = np.asarray(result.x, dtype=float)
    improved = np.isfinite(candidate).all() and np.isfinite(result.fun) and float(result.fun) <= initial_objective
    selected = candidate if improved else initial
    return {"schema": "pv-affine-marginal-calibration-v1", "bias_normalized": float(selected[0]),
        "spread_scale": float(selected[1]), "ac_nameplate_mw": float(ac_nameplate_mw),
        "fit_cutoff_utc": cutoff.isoformat(), "training_objective": "mean_empirical_crps_normalized_by_ac_nameplate",
        "training_intervals": int(valid.sum()), "excluded_or_unavailable_training_intervals": int((~valid).sum()),
        "last_training_target_start_utc": starts[valid][-1].isoformat(),
        "latest_training_target_available_at_utc": available[valid].max().isoformat(),
        "raw_training_crps_normalized": initial_objective, "calibrated_training_crps_normalized": objective(selected),
        "optimization": {"method": "Powell", "success": bool(result.success), "message": str(result.message),
            "evaluations": int(result.nfev), "retained_raw_parameters": not improved,
            "runtime_seconds": perf_counter() - started, "global_optimum_claimed": False},
        "settings": asdict(settings), "input_scenarios_sha256": _array_hash(values[:, valid]),
        "training_targets_sha256": _array_hash(actual[valid]),
        "computation": "retrospective reconstruction using chronological training labels",
        "historical_model_availability": "assumed available immediately at declared fit cutoff"}


def calibration_scores(scenarios, observed, eligible, *, central_probability=0.8):
    """Evaluate proper marginal CRPS, point MAE and empirical interval coverage."""
    values = np.asarray(scenarios, dtype=float)
    observed = np.asarray(observed, dtype=float)
    eligible = np.asarray(eligible)
    if (values.ndim != 2 or values.shape[1:] != observed.shape or eligible.shape != observed.shape
            or eligible.dtype != bool or not np.isfinite(values).all() or not 0 < central_probability < 1):
        raise ValueError("Calibration scores need aligned finite scenarios, targets and boolean eligibility")
    eligible = eligible & np.isfinite(observed) & (observed >= 0)
    if not eligible.any():
        return {"eligible_intervals": 0, "crps_mw": None, "mae_mean_mw": None, "central_interval_coverage": None}
    selected, truth = values[:, eligible], observed[eligible]
    lower, upper = np.quantile(selected, [(1 - central_probability) / 2, (1 + central_probability) / 2], axis=0)
    return {"eligible_intervals": int(eligible.sum()), "crps_mw": float(weighted_crps(selected, truth).mean()),
        "mae_mean_mw": float(np.abs(selected.mean(axis=0) - truth).mean()),
        "mean_error_mw": float((selected.mean(axis=0) - truth).mean()),
        "central_interval_probability": central_probability,
        "central_interval_coverage": float(((truth >= lower) & (truth <= upper)).mean()),
        "mean_central_interval_width_mw": float((upper - lower).mean())}


def save_calibration_npz(path, *, model, farm_id, interval_starts, raw_scenarios, scenario_ids,
                         forecast_available_at, forecast_vintage, night_mask=None):
    """Save safe numeric/Unicode arrays plus parameters for runtime recalibration.

    The precomputed D-1 marginals support evaluation. Runtime control applies the
    same fitted parameters to its currently eligible raw forecasts, so later
    deterministic-member updates are not silently frozen by the export window.
    """
    path = Path(path)
    starts = pd.DatetimeIndex([_aware(value, "Export start") for value in interval_starts])
    available = pd.DatetimeIndex([_aware(value, "Forecast availability") for value in forecast_available_at])
    vintage = pd.DatetimeIndex([_aware(value, "Forecast vintage") for value in forecast_vintage])
    if len(starts) != len(available) or len(starts) != len(vintage) or starts.has_duplicates or not starts.is_monotonic_increasing:
        raise ValueError("Calibration export clocks must align with unique increasing products")
    if np.any(vintage > available) or np.any(available > starts):
        raise ValueError("Exported forecast vintages must be available by their delivery intervals")
    calibrated = apply_bias_spread(raw_scenarios, bias_normalized=model["bias_normalized"],
        spread_scale=model["spread_scale"], ac_nameplate_mw=model["ac_nameplate_mw"], night_mask=night_mask)
    if calibrated.shape != (len(scenario_ids), len(starts)) or len(set(scenario_ids)) != len(scenario_ids):
        raise ValueError("Calibration scenario identifiers must match the exported matrix")
    fit_cutoff = _aware(model["fit_cutoff_utc"], "Model fit cutoff")
    if fit_cutoff > starts[0]:
        raise ValueError("Calibration model is unavailable at the first evaluation interval")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, artifactkind=np.array("affine_marginal_calibration"), farm_id=np.array(farm_id),
            pv_ac_mw=calibrated, interval_starts_utc=np.array(starts.astype(str), dtype=str),
            available_at_utc=np.array(fit_cutoff.isoformat()), calibration_fit_cutoff_utc=np.array(fit_cutoff.isoformat()),
            forecast_available_at_utc=np.array(available.astype(str), dtype=str),
            forecast_vintage_utc=np.array(vintage.astype(str), dtype=str), scenario_ids=np.array(scenario_ids, dtype=str),
            calibration_bias_normalized=np.array(model["bias_normalized"]),
            calibration_spread_scale=np.array(model["spread_scale"]), ac_nameplate_mw=np.array(model["ac_nameplate_mw"]),
            raw_scenarios_sha256=np.array(_array_hash(raw_scenarios)),
            model_sha256=np.array(hashlib.sha256(json.dumps(model, sort_keys=True, allow_nan=False).encode()).hexdigest()))
    temporary.replace(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "scenario_count": len(scenario_ids), "interval_count": len(starts), "artifactkind": "affine_marginal_calibration"}


def prepare_calibration_data(config, days):
    """Reuse physical conversion and select the explicitly configured target dataset."""
    from ..data import config_path, load_metadata, load_observations, open_store, night_window_mask
    from ..markets import AuctionCalendar
    from ..simulation.continuous_inputs import _json_value, _prepare_pv_ensembles
    from ..simulation.gme_oracle import _ForecastPreflightSession, _load_pv_forecast
    from .data.pv_outages import apply_episode_exclusions

    calendar = AuctionCalendar.from_config(config)
    farm = int(config["farm_id"])
    daily = {}
    with _ForecastPreflightSession(config, calendar):
        for day in days:
            grid = calendar.delivery_grid(day).to_domain()
            schedule = calendar.schedule(day)
            cutoff = min(event.submission_at for event in schedule)
            deterministic = _load_pv_forecast(config, calendar, cutoff, pd.DatetimeIndex(grid.starts).as_unit("ns"))
            daily[day] = {"grid": grid, "schedule": schedule,
                "initial_forecast": deterministic.values.pv_ac_mw.to_numpy(float),
                "deterministic_vintage": deterministic.forecast_vintage,
                "deterministic_available": deterministic.available_at}
    provenance = _prepare_pv_ensembles(config, daily)
    starts = pd.DatetimeIndex([stamp for day in days for stamp in daily[day]["grid"].starts])
    curated = config["forecast"].get("curated_realized", {}).get("enabled", False)
    source_days = []
    if curated:
        from ..simulation.curated_pv import load_curated_realized_pv

        blocks, interval_sources = [], []
        for day in days:
            values, source = load_curated_realized_pv(config, farm, daily[day]["grid"])
            blocks.append(values)
            interval_sources.extend(source.pop("interval_sources"))
            source_days.append({"delivery_day": day, **source})
        targets = np.concatenate(blocks)
        raw = np.array([np.nan if row["raw_metered_ac_mw"] is None else row["raw_metered_ac_mw"]
                        for row in interval_sources], dtype=float)
        audit = pd.DataFrame({"interval_start_utc": starts, "raw_ac_mw": raw, "target_ac_mw": targets,
            "target_source": [row["source"] for row in interval_sources],
            "raw_nonfinite": ~np.isfinite(raw), "raw_negative": np.isfinite(raw) & (raw < 0),
            "initial_forecast_fallback": False, "eligible": np.isfinite(targets) & (targets >= 0),
            "episode_excluded": False, "review_excluded": False})
    else:
        # Raw mode retains its explicit missing and reviewed-episode exclusions.
        with open_store(config) as dataset:
            observed = load_observations(dataset, [farm], variables=(config["forecast"]["realized_power_variable"],),
                start=starts[0], end=starts[-1], telemetry_latency_minutes=config["source"].get("telemetry_latency_minutes", 0))
        frame = observed.frame.set_index("time").reindex(starts)
        targets = frame.raw_value.to_numpy(float) * float(config["forecast"]["realized_power_scale_to_mw"])
        raw = targets.copy()
        audit = pd.DataFrame({"interval_start_utc": starts, "raw_ac_mw": raw, "target_ac_mw": targets,
            "target_source": "metered_ac", "raw_nonfinite": ~np.isfinite(raw),
            "raw_negative": np.isfinite(raw) & (raw < 0),
            "initial_forecast_fallback": ~np.isfinite(targets), "eligible": np.isfinite(targets) & (targets >= 0)})
        audit = apply_episode_exclusions(audit, config, farm_id=farm)
    # Supplied availability remains interval completion plus the configured telemetry lag.
    observed_available = starts + pd.Timedelta(minutes=15 + config["source"].get("telemetry_latency_minutes", 0))
    audit["target_available_at_utc"] = observed_available
    night = night_window_mask(starts, config["policy"].get("forecast_pv_night", config["policy"].get("night", {})))
    measured_night = night_window_mask(starts, config["policy"].get("realized_pv_night", {"enabled": False}))
    audit["initial_forecast_fallback"] = ~np.isfinite(targets) & ~measured_night
    audit["forecast_night_prior"] = night
    include_deterministic = config["pv_scenario_calibration"].get("include_deterministic_member", True)
    blocks, identifiers, available, vintages = [], None, [], []
    for day in days:
        item = daily[day]
        ids = tuple(f"weather_{value}" for value in item["ensemble_members"])
        block = item["ensemble_pv"]
        available_at = pd.Timestamp(provenance[day]["available_at_utc"])
        vintage = pd.Timestamp(provenance[day]["forecast_vintage_utc"])
        if include_deterministic:
            ids = ("deterministic",) + ids
            block = np.vstack((item["initial_forecast"], block))
            available_at = max(available_at, pd.Timestamp(item["deterministic_available"]))
            vintage = max(vintage, pd.Timestamp(item["deterministic_vintage"]))
        if identifiers is not None and identifiers != ids:
            raise ValueError("One calibration/export window requires a consistent member layout; split windows when archives differ")
        identifiers = ids
        blocks.append(block)
        available.extend([available_at] * len(item["grid"].starts))
        vintages.extend([vintage] * len(item["grid"].starts))
    metadata = load_metadata(config_path(config, config["inputs"]["metadata_parquet"]), [farm])[farm]
    exclusion_path = config.get("pv_data_quality", {}).get("exclusions_csv")
    source_evidence = {"forecast_sources": provenance, "measurement_source": (
        config["forecast"]["curated_realized"]["dataset_root"] if curated else config["inputs"]["zarr"]),
        "measurement_units": "AC MW; reviewed synthetic counterfactuals included" if curated else "AC MW from untouched stored AC power",
        "raw_targets_sha256": _array_hash(raw), "selected_targets_sha256": _array_hash(targets),
        "target_source_counts": audit.target_source.value_counts().to_dict(), "curated_day_provenance": source_days,
        "target_availability_assumption": "interval completion for retrospective fitting; reviewed reconstructions were created later" if curated else "interval completion plus telemetry latency",
        "episode_csv_sha256": None if not exclusion_path else hashlib.sha256(config_path(config, exclusion_path).read_bytes()).hexdigest(),
        "fallback_targets_used": 0, "automatic_fault_detection": False}
    return {"starts": starts, "raw_scenarios": np.concatenate(blocks, axis=1), "observed": targets,
        "observed_available_at": observed_available, "eligible": audit.eligible.to_numpy(bool), "audit": audit,
        "night_mask": night, "scenario_ids": identifiers, "forecast_available_at": pd.DatetimeIndex(available),
        "forecast_vintage": pd.DatetimeIndex(vintages), "ac_nameplate_mw": metadata.ac_capacity_mw, "provenance": _json_value(source_evidence)}


def run_farm_calibration(config, *, check_only=False):
    """Fit one frozen farm calibration and evaluate explicitly configured later windows."""
    from ..data import config_path

    settings = config["pv_scenario_calibration"]
    fit = settings["training"]
    training_days = pd.date_range(fit["start_date"], fit["end_date"], freq="D").strftime("%Y-%m-%d").tolist()
    fit_cutoff = _aware(fit["fit_cutoff_utc"], "Fit cutoff")
    parameters = CalibrationSettings(**settings.get("parameters", {}))
    windows = settings["evaluation_windows"]
    for window in windows:
        if pd.Timestamp(window["start_date"]) <= pd.Timestamp(fit["end_date"]):
            raise ValueError("Evaluation windows must follow the complete calibration period")
    training = prepare_calibration_data(config, training_days)
    prepared_windows = [(window, prepare_calibration_data(config,
        pd.date_range(window["start_date"], window["end_date"], freq="D").strftime("%Y-%m-%d").tolist())) for window in windows]
    if check_only:
        return {"farm_id": int(config["farm_id"]), "status": "preflight_passed", "fits_started": 0,
            "training_intervals": int(training["eligible"].sum()), "scenario_count": len(training["scenario_ids"]),
            "evaluation_windows": [window["id"] for window, _ in prepared_windows]}
    model = fit_bias_spread(training["raw_scenarios"], training["observed"], ac_nameplate_mw=training["ac_nameplate_mw"],
        interval_starts=training["starts"], observed_available_at=training["observed_available_at"], fit_cutoff=fit_cutoff,
        eligible=training["eligible"], night_mask=training["night_mask"], settings=parameters)
    model.update(farm_id=int(config["farm_id"]), provenance=training["provenance"],
        exclusion_counts={column: int(training["audit"][column].sum()) for column in
            ("raw_nonfinite", "raw_negative", "initial_forecast_fallback", "episode_excluded", "review_excluded")})
    output = config_path(config, settings["output_directory"])
    folder = output / f"farm_{config['farm_id']}"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "model.json").write_text(json.dumps(model, indent=2, allow_nan=False) + "\n")
    training["audit"].to_csv(folder / "training_targets_audit.csv", index=False)
    results = []
    for window, data in prepared_windows:
        calibrated = apply_bias_spread(data["raw_scenarios"], bias_normalized=model["bias_normalized"],
            spread_scale=model["spread_scale"], ac_nameplate_mw=model["ac_nameplate_mw"], night_mask=data["night_mask"])
        law = ScenarioTrajectories(data["raw_scenarios"], np.full(len(data["scenario_ids"]), 1 / len(data["scenario_ids"])), data["scenario_ids"])
        ecc = transform_dependence(law, "ecc", calibrated_pv_marginals=calibrated, seed=settings["seed"])
        shuffled = transform_dependence(ecc, "independent_shuffle", seed=settings["seed"])
        scores = {name: calibration_scores(values, data["observed"], data["eligible"],
            central_probability=parameters.central_interval_probability) for name, values in
            (("raw", data["raw_scenarios"]), ("calibrated_ecc", ecc.pv_ac_mw), ("calibrated_shuffled", shuffled.pv_ac_mw))}
        artifact = save_calibration_npz(output / f"farm_{config['farm_id']}_{window['start_date']}.npz",
            model=model, farm_id=int(config["farm_id"]), interval_starts=data["starts"], raw_scenarios=data["raw_scenarios"],
            scenario_ids=data["scenario_ids"], forecast_available_at=data["forecast_available_at"],
            forecast_vintage=data["forecast_vintage"], night_mask=data["night_mask"])
        result = {"window": window, "artifact": artifact, "scores": scores, "forecast_provenance": data["provenance"],
            "marginal_crps_difference_ecc_minus_shuffle": None if scores["calibrated_ecc"]["crps_mw"] is None else
                scores["calibrated_ecc"]["crps_mw"] - scores["calibrated_shuffled"]["crps_mw"],
            "selection": "evaluation results never select or refit the frozen two-parameter model"}
        data["audit"].to_csv(folder / f"{window['id']}_targets_audit.csv", index=False)
        if settings.get("plots", True):
            _plot_calibration(folder / f"{window['id']}_forecast.png", data, calibrated, parameters.central_interval_probability)
        results.append(result)
    report = {"farm_id": int(config["farm_id"]), "status": "complete", "model": model, "evaluations": results}
    (folder / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def _plot_calibration(path, data, calibrated, central_probability):
    """Plot raw targets and raw/calibrated means, marking excluded observations."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(11, 4))
    axis.plot(data["starts"], data["observed"], color="black", linewidth=1, label="Selected AC target")
    axis.plot(data["starts"], data["raw_scenarios"].mean(axis=0), label="Raw ensemble mean")
    axis.plot(data["starts"], calibrated.mean(axis=0), label="Calibrated mean")
    lower, upper = np.quantile(calibrated, [(1 - central_probability) / 2, (1 + central_probability) / 2], axis=0)
    axis.fill_between(data["starts"], lower, upper, alpha=0.2, label=f"Calibrated central {100 * central_probability:g}%")
    excluded = ~data["eligible"] & np.isfinite(data["observed"])
    axis.scatter(data["starts"][excluded], data["observed"][excluded], marker="x", color="red", label="Excluded score targets")
    axis.set(xlabel="Delivery time (UTC)", ylabel="AC power (MW)")
    axis.legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
