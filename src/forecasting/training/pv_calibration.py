"""Offline, auditable through-origin calibration of the PV DC-to-AC scale."""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ...data import (
    align_forecast, config_path, load_metadata,
    load_observations, open_store, prepare_observations, select_forecast_vintage,
)
from ..models.pv import forecast_pv
from ..data.pv_outages import modeling_instances


def fit_scale(x: np.ndarray, y: np.ndarray) -> float:
    """Unique MSE-minimizing slope for y = beta*x with zero intercept."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    # Enforce the input contract: Calibration needs matching, finite nonempty vectors.
    if x.shape != y.shape or x.ndim != 1 or not len(x) or not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("Calibration needs matching, finite nonempty vectors")
    denominator = float(np.dot(x, x))
    # Enforce the input contract: Calibration regressor has zero squared norm.
    if denominator <= 0:
        raise ValueError("Calibration regressor has zero squared norm")
    coefficient = float(np.dot(x, y) / denominator)
    # Enforce the input contract: Fitted PV coefficient must be positive and finite.
    if coefficient <= 0 or not np.isfinite(coefficient):
        raise ValueError("Fitted PV coefficient must be positive and finite")
    return coefficient


def _scores(x: np.ndarray, y: np.ndarray, coefficient: float) -> dict[str, float]:
    residual = y - coefficient * x
    return {"mse_mw2": float(np.mean(residual ** 2)), "rmse_mw": float(np.sqrt(np.mean(residual ** 2))),
            "mae_mw": float(np.mean(np.abs(residual))), "bias_mw": float(np.mean(coefficient * x - y))}


def _plots(pairs: pd.DataFrame, fit: dict[str, Any], path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eligible = pairs.loc[pairs.eligible]
    x = eligible.regressor_mw.to_numpy()
    y = eligible.measured_ac_mw.to_numpy()
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), layout="constrained")
    axes[0].scatter(x, y, s=5, alpha=0.3, label="Measured AC")
    line = np.linspace(0, float(np.max(x)), 100)
    axes[0].plot(line, fit["coefficient"] * line, label=f"Fitted {fit['coefficient']:.4f}")
    axes[0].plot(line, fit["reference_coefficient"] * line, label=f"Reference {fit['reference_coefficient']:.4f}")
    axes[0].set(xlabel="PVWatts DC × transformer efficiency (MW)", ylabel="Measured AC at POI (MW)")
    axes[0].legend()
    times = pd.to_datetime(eligible.interval_start_utc, utc=True)
    axes[1].plot(times, y - fit["coefficient"] * x, linewidth=0.7)
    axes[1].axhline(0, color="black", linewidth=0.6)
    axes[1].set(xlabel="Delivery interval (UTC)", ylabel="Measured − fitted (MW)")
    fig.savefig(path, dpi=150)
    plt.close(fig)


def calibrate_pv_output(config: dict[str, Any]) -> dict[str, Any]:
    """Pair latest available weather with measured AC and fit a DC-to-AC scale."""
    settings = config["pv_calibration"]
    # Pair latest available weather with measured AC and fit a DC-to-AC scale for each `key` in `('start_day', 'end_day_exclusive')`.
    for key in ("start_day", "end_day_exclusive"):
        # Enforce the input contract: pv_calibration.`key` is required.
        if not settings.get(key):
            raise ValueError(f"pv_calibration.{key} is required")
    tz = settings["delivery_timezone"]
    start = pd.Timestamp(settings["start_day"]).tz_localize(tz).tz_convert("UTC")
    end = pd.Timestamp(settings["end_day_exclusive"]).tz_localize(tz).tz_convert("UTC")
    # Enforce the input contract: PV calibration end_day_exclusive must follow start_day.
    if end <= start:
        raise ValueError("PV calibration end_day_exclusive must follow start_day")
    starts = pd.date_range(start, end, freq="15min", inclusive="left")
    validation_fraction = float(settings["validation_fraction"])
    minimum_pairs = int(settings["minimum_pairs"])
    minimum_dc_fraction = float(settings["minimum_dc_fraction_of_ac_rating"])
    # Enforce the input contract: Invalid PV calibration split, minimum pairs, or DC threshold.
    if not 0 < validation_fraction < 1 or minimum_pairs < 2 or not 0 <= minimum_dc_fraction < 1:
        raise ValueError("Invalid PV calibration split, minimum pairs, or DC threshold")
    farm_id = int(config["farm_id"])
    farm = load_metadata(config_path(config, config["inputs"]["metadata_parquet"]), [farm_id])[farm_id]
    declared_rating = float(config["farms"][str(farm_id)]["ac_capacity_kw"])
    # Enforce the input contract: Configured and metadata farm AC ratings disagree.
    if not np.isclose(declared_rating, farm.ac_capacity_kw):
        raise ValueError("Configured and metadata farm AC ratings disagree")
    pv_settings = deepcopy(config["pv"])
    pv_settings["output"]["method"] = "explicit_efficiency"  # modeled DC is independent of output mode
    source = config["source"]
    dataset = open_store(config)
    try:
        cycles = pd.DatetimeIndex(pd.to_datetime(dataset.cycle_time.values, utc=True))
        availability_variable = source.get("explicit_availability_variable")
        # When `availability_variable`, set `available` using `dataset[availability_variable]`.
        if availability_variable:
            available = dataset[availability_variable]
            # When `set(available.dims) == {'farm_id', 'cycle_time'}`, set `available` using `available.sel(farm_id=farm_id)`.
            if set(available.dims) == {"farm_id", "cycle_time"}:
                available = available.sel(farm_id=farm_id)
            # Enforce the input contract: Forecast availability metadata must be indexed by cycle_time.
            if available.dims != ("cycle_time",):
                raise ValueError("Forecast availability metadata must be indexed by cycle_time")
            availability = pd.DatetimeIndex(pd.to_datetime(available.values, utc=True))
        else:
            availability = cycles + pd.Timedelta(hours=float(source["availability_lag_hours"]))
        # Search the newest cycle that was available by each delivery interval start.
        selected = np.full(len(starts), -1, dtype=int)
        # Pair latest available weather with measured AC and fit a DC-to-AC scale for each `(i, interval)` in `enumerate(starts)`.
        for i, interval in enumerate(starts):
            eligible_cycles = np.flatnonzero((availability <= interval) & (cycles <= interval))
            # When `len(eligible_cycles)`, set `selected[i]` using `eligible_cycles[np.argmax(cycles.asi8[eligible_cycles])]`.
            if len(eligible_cycles):
                selected[i] = eligible_cycles[np.argmax(cycles.asi8[eligible_cycles])]
        modeled = pd.Series(np.nan, index=starts, dtype=float)
        forecast_valid = pd.Series(False, index=starts, dtype=bool)
        vintages = pd.Series(pd.NaT, index=starts, dtype="datetime64[ns, UTC]")
        published = pd.Series(pd.NaT, index=starts, dtype="datetime64[ns, UTC]")
        # Apply forecast pv for `position` from `np.unique(selected[selected >= 0])`.
        for position in np.unique(selected[selected >= 0]):
            group_starts = starts[selected == position]
            bundle = select_forecast_vintage(dataset, farm_id, group_starts[0], config["forecast_mapping"],
                float(source["availability_lag_hours"]), availability_variable, source.get("numeric_lead_units", "hours"))
            aligned = align_forecast(bundle, group_starts, config["forecast_semantics"], config["policy"].get("nonnegative_forecasts", []))
            pv = forecast_pv(aligned, farm, pv_settings, config["policy"].get("night", {}))
            modeled.loc[group_starts] = pv.values.modeled_dc_mw.to_numpy()
            forecast_valid.loc[group_starts] = pv.values.pv_input_valid.to_numpy()
            vintages.loc[group_starts] = bundle.forecast_vintage
            published.loc[group_starts] = bundle.available_at
        observations = load_observations(dataset, [farm_id], variables=("power",), start=start, end=end - pd.Timedelta(minutes=15))
    finally:
        dataset.close()
    treated = prepare_observations(observations, config["policy"])
    observed = treated.frame.set_index("time").reindex(starts)
    pairs = pd.DataFrame({"interval_start_utc": starts, "forecast_cycle_utc": vintages.to_numpy(),
        "forecast_available_at_utc": published.to_numpy(), "modeled_dc_mw": modeled.to_numpy(),
        "regressor_mw": modeled.to_numpy() * float(config["pv"]["output"]["transformer_efficiency"]),
        "measured_ac_mw": observed.raw_value.to_numpy(dtype=float) * float(config["forecast"]["realized_power_scale_to_mw"]),
        "pv_input_valid": forecast_valid.to_numpy(), "review_excluded": observed.review_excluded.fillna(False).to_numpy(dtype=bool),
        "plateau_zero": observed.plateau_zero.fillna(False).to_numpy(dtype=bool),
        "nighttime_zero": observed.nighttime_zero.fillna(False).to_numpy(dtype=bool),
        "measurement_interpolated": observed.measurement_interpolated.fillna(False).to_numpy(dtype=bool)})
    valid = (pairs.pv_input_valid & np.isfinite(pairs.regressor_mw) & np.isfinite(pairs.measured_ac_mw)
        & (pairs.regressor_mw >= minimum_dc_fraction * farm.ac_capacity_mw)
        & (pairs.measured_ac_mw >= 0) & ~pairs.review_excluded)
    pairs["eligible"] = valid
    pairs["exclusion_reason"] = np.select([
        ~pairs.pv_input_valid | ~np.isfinite(pairs.regressor_mw), ~np.isfinite(pairs.measured_ac_mw),
        pairs.regressor_mw < minimum_dc_fraction * farm.ac_capacity_mw, pairs.measured_ac_mw < 0,
        pairs.review_excluded],
        ["missing_or_invalid_forecast", "missing_measurement", "low_dc", "negative_measurement", "review_excluded"], default="")
    pairs=modeling_instances(pairs,config,farm_id=farm_id)
    valid=pairs.eligible
    pairs.loc[valid, "exclusion_reason"] = ""
    eligible = pairs.loc[valid].sort_values("interval_start_utc")
    # Enforce the input contract: Only `len(eligible)` eligible PV calibration pairs; minimum is `minimum_pairs`.
    if len(eligible) < minimum_pairs:
        raise ValueError(f"Only {len(eligible)} eligible PV calibration pairs; minimum is {minimum_pairs}")
    validation_count = max(1, int(np.ceil(len(eligible) * validation_fraction)))
    train, validation = eligible.iloc[:-validation_count], eligible.iloc[-validation_count:]
    # Enforce the input contract: PV calibration has no training pairs after validation split.
    if len(train) < 1:
        raise ValueError("PV calibration has no training pairs after validation split")
    beta_train = fit_scale(train.regressor_mw.to_numpy(), train.measured_ac_mw.to_numpy())
    beta_all = fit_scale(eligible.regressor_mw.to_numpy(), eligible.measured_ac_mw.to_numpy())
    reference = float(config["pv"]["output"]["pv_output_efficiency"])
    result = {"status": "complete", "farm_id": farm_id, "ac_rating_mw": farm.ac_capacity_mw,
        "dc_reference_capacity_mw": farm.ac_capacity_mw / float(config["pv"]["dc"]["reference_conversion_efficiency"]),
        "coefficient": beta_all, "train_coefficient": beta_train, "reference_coefficient": reference,
        "model": "measured_ac_mw = coefficient * (pvlib_dc_mw * transformer_efficiency); zero_intercept_ols_mse",
        "target": "raw_measured_ac_at_poi_mw", "forecast_selection": "latest_cycle_available_at_each_delivery_interval_start",
        "pairs_total": len(pairs), "pairs_eligible": len(eligible), "train_pairs": len(train), "validation_pairs": len(validation),
        "validation_start_utc": validation.interval_start_utc.iloc[0].isoformat(),
        "validation_end_utc": validation.interval_start_utc.iloc[-1].isoformat(),
        "validation_fitted": _scores(validation.regressor_mw.to_numpy(), validation.measured_ac_mw.to_numpy(), beta_train),
        "validation_reference": _scores(validation.regressor_mw.to_numpy(), validation.measured_ac_mw.to_numpy(), reference),
        "all_pairs_fitted": _scores(eligible.regressor_mw.to_numpy(), eligible.measured_ac_mw.to_numpy(), beta_all),
        "config_override": {"pv": {"output": {"method": "calibrated_scale", "calibrated_coefficient": beta_all}}},
        "interpretation": "Aggregate predictive scale; curtailment, outages and forecast error can affect it. Not an identified physical inverter efficiency."}
    root = config_path(config, settings["output_root"])
    run_id = settings.get("run_id") or datetime.now(timezone.utc).strftime("farm%d_%Y%m%dT%H%M%S%fZ" % farm_id)
    output = root / run_id
    # Pair latest available weather with measured AC and fit a DC-to-AC scale for each `subdir` in `('json', 'csv', 'plots')`.
    for subdir in ("json", "csv", "plots"):
        (output / subdir).mkdir(parents=True, exist_ok=False)
    (output / "json" / "study_config.json").write_text(json.dumps(config, indent=2, default=str) + "\n")
    pairs.to_csv(output / "csv" / "aligned_pairs.csv", index=False)
    _plots(pairs, result, output / "plots" / "calibration.png")
    result["output_dir"] = str(output)
    (output / "json" / "fit.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    return result
