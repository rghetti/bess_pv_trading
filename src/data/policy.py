"""Versioned user-requested derived masks; raw observations are never changed."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .contracts import ObservationBundle, utc_timestamp


# The clock mask uses interval starts, independently of proxy solar position.
def night_window_mask(times: Any, policy: dict[str, Any]) -> np.ndarray:
    """Return local-time half-open night masks, optionally varying by month."""
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if not policy.get("enabled", True):
        return np.zeros(len(index), dtype=bool)
    local = index.tz_convert(policy.get("timezone", "Europe/Rome"))
    minutes = local.hour * 60 + local.minute
    def mask_for(window):
        start_hour, start_minute = map(int, window.get("start", "21:30").split(":"))
        end_hour, end_minute = map(int, window.get("end", "05:00").split(":"))
        if not (0 <= start_hour < 24 and 0 <= end_hour < 24 and
                0 <= start_minute < 60 and 0 <= end_minute < 60):
            raise ValueError("Night-window endpoints must be valid local clock times")
        start, end = start_hour * 60 + start_minute, end_hour * 60 + end_minute
        if start == end:
            raise ValueError("Night-window endpoints must differ")
        return np.asarray((minutes >= start) | (minutes < end) if start > end
                          else (minutes >= start) & (minutes < end))

    monthly = policy.get('monthly_windows')
    if monthly is None:
        return mask_for(policy)
    if set(map(str, monthly)) != {f'{month:02d}' for month in range(1, 13)}:
        raise ValueError('Seasonal night policy must provide keys 01 through 12')
    result = np.zeros(len(index), dtype=bool)
    for month in range(1, 13):
        result |= np.asarray(local.month == month) & mask_for(monthly[f'{month:02d}'])
    return result


# Anchor-based matching avoids treating a gradual ramp as a repeated run.
def plateau_runs(values: Any, times: Any, policy: dict[str, Any]) -> list[np.ndarray]:
    """Identify qualifying finite nonzero native-measurement runs."""
    minimum = int(policy.get("minimum_run", 4))
    if minimum < 4:
        raise ValueError("Requested plateau policy requires at least four native samples")
    atol, rtol = float(policy.get("atol_kw", 0.01)), float(policy.get("rtol", 0.0001))
    if atol < 0 or rtol < 0:
        raise ValueError("Plateau tolerances must be nonnegative")
    values = np.asarray(values, dtype=float)
    index = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    native = pd.Timedelta(minutes=float(policy.get("native_minutes", 15)))
    result: list[np.ndarray] = []
    start = 0
    while start < len(values):
        if not np.isfinite(values[start]) or values[start] == 0:
            start += 1
            continue
        end = start + 1
        while end < len(values) and index[end] - index[end - 1] == native and np.isfinite(values[end]) and values[end] != 0 and np.isclose(values[end], values[start], atol=atol, rtol=rtol):
            end += 1
        if end - start >= minimum:
            result.append(np.arange(start, end))
        start = end
    return result


# Apply a finite-negative floor only to explicitly named nonnegative magnitudes.
def clip_negative_forecasts(frame: pd.DataFrame, allowlist: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Copy forecasts, preserving structural NaNs and all signed variables."""
    treated = frame.copy(deep=True)
    flags = pd.DataFrame(False, index=frame.index, columns=frame.columns)
    for variable in allowlist:
        if variable not in frame:
            continue
        values = np.asarray(frame[variable], dtype=float)
        flag = np.isfinite(values) & (values < 0)
        treated.loc[flag, variable] = 0.0
        flags.loc[flag, variable] = True
    return treated, flags


# Treat each farm/variable independently and retain masks alongside raw values.
def prepare_observations(bundle: ObservationBundle, policy: dict[str, Any]) -> ObservationBundle:
    """Create finalized policy observations and causal evidence timestamps.

    Repeated calls use raw_value, so values, masks and energy are idempotent.
    Missing timestamp rows are not invented. Existing review fields survive.
    """
    frame = bundle.frame.copy(deep=True).sort_values(["farm_id", "variable", "time"]).reset_index(drop=True)
    if frame.duplicated(["farm_id", "variable", "time"]).any():
        raise ValueError("Observation keys must be unique")
    for column in ("time", "interval_end", "available_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if (frame.interval_end <= frame.time).any() or (frame.available_at < frame.interval_end).any():
        raise ValueError("Observation availability cannot precede interval completion")
    frame["treated_value"] = frame.raw_value.astype(float)
    if "review_excluded" not in frame:
        frame["review_excluded"] = False
    frame["derived_review_excluded"] = frame.review_excluded.astype(bool)
    frame["evidence_at"] = frame.available_at
    frame["plateau_evidence_at"] = pd.NaT
    frame["plateau_evidence_at"] = pd.to_datetime(frame.plateau_evidence_at, utc=True)
    for flag in ("plateau_zero", "nighttime_zero", "measurement_interpolated", "night_window"):
        frame[flag] = False
    frame["plateau_run_length"] = 0
    frame["gap_hours"] = 0.0
    interpolation = policy.get("interpolation", {})
    plateau_policy = policy.get("plateau", {})
    maximum_gap = interpolation.get("maximum_gap_hours")
    if maximum_gap is not None and float(maximum_gap) < 0:
        raise ValueError("Maximum interpolation gap must be nonnegative or null")
    # A detected plateau is first knowable when its fourth reading completes.
    for (_, variable), group in frame.groupby(["farm_id", "variable"], sort=False):
        indices = group.index.to_numpy()
        if variable == "power" and plateau_policy.get("enabled", True):
            for run in plateau_runs(group.raw_value, group.time, plateau_policy):
                positions = indices[run]
                threshold_at = group.iloc[run[int(plateau_policy.get("minimum_run", 4)) - 1]].available_at
                evidence = group.iloc[run].available_at.map(lambda value: max(value, threshold_at))
                frame.loc[positions, "treated_value"] = 0.0
                frame.loc[positions, "plateau_zero"] = True
                frame.loc[positions, "plateau_run_length"] = len(run)
                frame.loc[positions, "plateau_evidence_at"] = evidence.to_numpy()
                frame.loc[positions, "evidence_at"] = evidence.to_numpy()
        if variable == "power":
            night = night_window_mask(group.time, policy.get("night", {}))
            frame.loc[indices[night], "night_window"] = True
            changed = night & (frame.loc[indices, "treated_value"].to_numpy() > 0)
            frame.loc[indices[changed], "nighttime_zero"] = True
            frame.loc[indices[changed], "treated_value"] = 0.0
            # The nighttime clock rule may already justify a plateau's final zero.
            already_zero = night & (group.raw_value.to_numpy() > 0)
            frame.loc[indices[already_zero], "evidence_at"] = group.loc[group.index[already_zero], "available_at"].to_numpy()
        if interpolation.get("enabled", True) and variable in interpolation.get("variables", ["power", "ghi"]):
            treated = frame.loc[indices, "treated_value"].to_numpy(copy=True)
            missing = np.isnan(treated)
            position = 0
            while position < len(treated):
                if not missing[position]:
                    position += 1
                    continue
                right = position + 1
                while right < len(treated) and missing[right]:
                    right += 1
                left = position - 1
                if left >= 0 and right < len(treated) and np.isfinite(treated[left]) and np.isfinite(treated[right]):
                    gap_hours = (group.iloc[right].time - group.iloc[left].time).total_seconds() / 3600
                    if maximum_gap is None or gap_hours <= float(maximum_gap):
                        span = (group.iloc[right].time - group.iloc[left].time).total_seconds()
                        targets = indices[position:right]
                        fraction = (group.iloc[position:right].time - group.iloc[left].time).dt.total_seconds() / span
                        frame.loc[targets, "treated_value"] = treated[left] + fraction.to_numpy() * (treated[right] - treated[left])
                        frame.loc[targets, "measurement_interpolated"] = True
                        frame.loc[targets, "gap_hours"] = gap_hours
                        if frame.loc[indices[left], "derived_review_excluded"] or frame.loc[indices[right], "derived_review_excluded"]:
                            frame.loc[targets, "derived_review_excluded"] = True
                        endpoint_evidence = max(frame.loc[indices[left], "evidence_at"], frame.loc[indices[right], "evidence_at"])
                        frame.loc[targets, "evidence_at"] = frame.loc[targets, "available_at"].map(lambda value: max(value, endpoint_evidence)).to_numpy()
                position = right
        if variable == "power":
            # Reapply the clock mask after interpolation, retaining all reasons.
            positive_night = frame.loc[indices, "night_window"].to_numpy() & (frame.loc[indices, "treated_value"].to_numpy() > 0)
            frame.loc[indices[positive_night], "nighttime_zero"] = True
            frame.loc[indices[positive_night], "treated_value"] = 0.0
    for column in ("evidence_at", "plateau_evidence_at"):
        frame[column] = pd.to_datetime(frame[column], utc=True)
    if "review_excluded" not in frame:
        frame["review_excluded"] = False
    frame["policy_version"] = policy.get("version", "requested-v1")
    summary = observation_summary(frame)
    provenance = {**bundle.provenance, "policy": policy, "accounting": "finalized_ex_post_treated_AC"}
    return ObservationBundle(frame, provenance, summary)


# Energy changes compare observed finite pairs; imputed energy is reported separately.
def observation_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Return serializable counts and AC-energy reconciliation by farm/day."""
    power = frame.loc[frame.variable == "power"].copy()
    rows: list[dict[str, Any]] = []
    if len(power):
        power["day_utc"] = power.time.dt.strftime("%Y-%m-%d")
        for (farm, day), group in power.groupby(["farm_id", "day_utc"]):
            duration = (group.interval_end - group.time).dt.total_seconds().to_numpy() / 3600
            raw, derived = group.raw_value.to_numpy(), group.treated_value.to_numpy()
            paired = np.isfinite(raw) & np.isfinite(derived)
            raw_energy = float(np.sum(np.where(np.isfinite(raw), raw, 0) * duration) / 1000)
            treated_energy = float(np.sum(np.where(np.isfinite(derived), derived, 0) * duration) / 1000)
            imputed = ~np.isfinite(raw) & np.isfinite(derived)
            filled_energy = float(np.sum(derived[imputed] * duration[imputed]) / 1000)
            rows.append({"farm_id": int(farm), "day_utc": day, "raw_finite_energy_mwh": raw_energy, "treated_finite_energy_mwh": treated_energy, "paired_adjustment_mwh": float(np.sum((derived[paired] - raw[paired]) * duration[paired]) / 1000), "imputed_energy_mwh": filled_energy, "energy_change_mwh": treated_energy - raw_energy})
    changed = np.isfinite(frame.treated_value) & (~np.isfinite(frame.raw_value) | (frame.treated_value != frame.raw_value))
    return {"rows": len(frame), "adjusted_rows": int(changed.sum()), "adjusted_fraction": float(changed.mean()) if len(frame) else 0.0, "remaining_missing": int(frame.treated_value.isna().sum()), "remaining_nonfinite": int((~np.isfinite(frame.treated_value)).sum()), "flags": {name: int(frame[name].sum()) for name in ("plateau_zero", "nighttime_zero", "measurement_interpolated", "review_excluded", "derived_review_excluded")}, "daily_ac_energy": rows}
