"""Frozen Gaussian policy selection and causal online error-covariance updates."""

from datetime import date, timedelta
import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.gaussian_adaptive import (
    calibrate_for_cutoff, evaluate_adaptive_gaussian, select_gaussian_policy,
)
from bess_pv_trading.forecasting.gaussian_prices import GaussianPriceCalibration


def _rows(start_day="2025-01-01", days=22, minutes=60):
    """Generate complete native days with correlated but nonconstant errors."""
    frames = []
    first = pd.Timestamp(start_day).date()
    for offset in range(days):
        day = first + timedelta(days=offset)
        midnight = pd.Timestamp(day, tz="Europe/Rome")
        following = pd.Timestamp(day + timedelta(days=1), tz="Europe/Rome")
        starts = pd.date_range(midnight, following, inclusive="left", freq=f"{minutes}min").tz_convert("UTC")
        slots = np.arange(len(starts), dtype=float)
        index = (day - date(2025, 1, 1)).days
        prediction = 80 + 10 * np.cos(2 * np.pi * slots / len(slots)) + index / 10
        error = ((index % 5 - 2) * (1 + .3 * np.sin(slots / 4))
                 + .2 * np.cos(slots + index))
        frames.append(pd.DataFrame({"delivery_day": day.isoformat(), "product_start": starts,
            "product_end": starts + pd.Timedelta(minutes=minutes), "native_interval_minutes": minutes,
            "actual": prediction + error, "prediction": prediction,
            "fit_cutoff": pd.Timestamp(day - timedelta(days=2), tz="Europe/Rome").tz_convert("UTC"),
            "forecast_cutoff": (pd.Timestamp(day - timedelta(days=1), tz="Europe/Rome") + pd.Timedelta(hours=12)).tz_convert("UTC"),
            "training_max_delivery_day": (day - timedelta(days=3)).isoformat(),
            "prediction_provenance": "chronological_out_of_sample"}))
    return pd.concat(frames, ignore_index=True)


def _settings(**values):
    return {"window_days_candidates": [7, 14], "covariance_shrinkage_candidates": [.1, .5],
            "minimum_complete_days": 3, "inner_validation_days": 4,
            "minimum_inner_scored_days": 3, "selection_metric": "joint_nll", **values}


def _fixed_policy(**values):
    return {"window_days": 7, "covariance_shrinkage": .1, "minimum_complete_days": 3,
            "timezone": "Europe/Rome", **values}


def test_selection_compares_identical_inner_days_and_returns_frozen_reloadable_policy():
    selected, scores = select_gaussian_policy(_rows(), _settings())
    assert len(scores) == 4 and scores.selected.sum() == 1
    assert scores.scored_days.eq(4).all() and scores.scored_rows.eq(4 * 24).all()
    assert selected["selection_source"] == "calibration_predictions_only"
    assert selected["fixed_after_selection"] is True
    assert selected["selection_inner_days"] == ["2025-01-19", "2025-01-20", "2025-01-21", "2025-01-22"]
    assert selected["selection_score"] == scores.selection_score.min()
    assert selected["selection_metric"] == "mean_joint_nll_per_product"
    assert {"mean_crps_eur_mwh", "coverage_95", "ramp_coverage_95"}.issubset(scores.columns)
    assert json.loads(json.dumps(selected, allow_nan=False)) == selected


def test_future_or_partly_mature_days_cannot_change_an_earlier_artifact(tmp_path):
    rows = _rows()
    cutoff = pd.Timestamp("2025-01-15T12:00+01:00")
    original = calibrate_for_cutoff(rows, cutoff, _fixed_policy())
    perturbed = rows.copy()
    # Jan 15 is only partly delivered at this cutoff. Exclude its entire day,
    # as well as all later days, from the covariance and bias calibration.
    perturbed.loc[perturbed.delivery_day.ge("2025-01-15"), "actual"] += 10000
    changed = calibrate_for_cutoff(perturbed, cutoff, _fixed_policy())
    assert original.to_dict() == changed.to_dict()
    grid = next(iter(original.grids.values()))
    assert grid.first_delivery_day == "2025-01-08" and grid.last_delivery_day == "2025-01-14"
    assert grid.calibration_days == 7
    path = tmp_path / "artifact.json"
    original.save(path)
    assert GaussianPriceCalibration.load(path).to_dict() == original.to_dict()


def test_online_test_updates_change_only_later_matured_distributions_and_never_policy():
    calibration = _rows()
    test = _rows("2025-01-24", 4)
    policy, _ = select_gaussian_policy(calibration, _settings())
    frozen = json.loads(json.dumps(policy))
    original = evaluate_adaptive_gaussian(calibration, test, policy)
    changed_test = test.copy()
    changed_test.loc[changed_test.delivery_day.eq("2025-01-24"), "actual"] += 100
    changed = evaluate_adaptive_gaussian(calibration, changed_test, policy)
    assert policy == frozen and original["policy"] == changed["policy"]
    # Jan 24's complete error becomes mature at Jan 25 midnight: it is absent
    # from Jan 25's D-1-noon forecast and available to Jan 26's forecast.
    for day in ("2025-01-24", "2025-01-25"):
        assert original["daily_calibrations"][day].to_dict() == changed["daily_calibrations"][day].to_dict()
    assert original["daily_calibrations"]["2025-01-26"].to_dict() != changed["daily_calibrations"]["2025-01-26"].to_dict()
    assert original["test_labels_used_for_selection"] is False
    assert original["earlier_mature_test_errors_used_for_updates"] is True
    assert original["metrics"]["scored_rows"] == 96 and original["metrics"]["scored_days"] == 4
    assert original["metrics"]["consecutive_product_ramps"]["rows"] == 92
    assert original["daily_audit"][0]["mature_prior_test_days_available"] == 0
    assert original["daily_audit"][2]["mature_prior_test_days_available"] == 1


def test_perturbing_later_test_actuals_cannot_change_earlier_scored_means_sigmas_or_artifacts():
    calibration, test = _rows(), _rows("2025-01-24", 4)
    policy = _fixed_policy(selection_max_product_end=calibration.product_end.max().isoformat())
    original = evaluate_adaptive_gaussian(calibration, test, policy)
    test.loc[test.delivery_day.eq("2025-01-27"), "actual"] -= 1000
    changed = evaluate_adaptive_gaussian(calibration, test, policy)
    for day in original["daily_calibrations"]:
        assert original["daily_calibrations"][day].to_dict() == changed["daily_calibrations"][day].to_dict()
    names = ["delivery_day", "product_start", "gaussian_mean", "sigma_eur_mwh"]
    pd.testing.assert_frame_equal(original["scored_predictions"][names], changed["scored_predictions"][names])


def test_dst_native_layout_is_excluded_without_borrowing_ordinary_hourly_covariance():
    history = _rows("2025-03-05", 24)
    test = _rows("2025-03-30", 2)
    policy = _fixed_policy(selection_max_product_end=history.product_end.max().isoformat())
    result = evaluate_adaptive_gaussian(history, test, policy)
    assert "2025-03-30" in result["excluded_days"]
    assert list(result["daily_calibrations"]) == ["2025-03-31"]
    assert result["metrics"]["scored_rows"] == 24
    assert result["metrics"]["unscored_rows"] == 23
    assert next(iter(result["daily_calibrations"]["2025-03-31"].grids.values())).slot_labels[2] == "02:00/fold0"


def test_constant_errors_still_produce_positive_definite_covariance_and_zero_mean_noise():
    rows = _rows()
    rows["actual"] = rows.prediction + 2
    artifact = calibrate_for_cutoff(rows, "2025-01-23T12:00+01:00", _fixed_policy())
    grid = next(iter(artifact.grids.values()))
    assert np.linalg.eigvalsh(grid.covariance_eur2_mwh2).min() > 0
    assert np.allclose(grid.bias_eur_mwh, 2)
    assert artifact.to_dict()["innovation_mean_eur_mwh"] == 0


def test_policy_selected_after_forecast_cutoff_and_mixed_native_durations_are_rejected():
    history = _rows()
    policy = _fixed_policy(selection_max_product_end="2025-01-23T00:00+01:00")
    with pytest.raises(ValueError, match="policy was selected using information after"):
        calibrate_for_cutoff(history, "2025-01-20T12:00+01:00", policy)
    mixed = pd.concat([_rows("2025-01-01", 2), _rows("2025-01-03", 2, minutes=15)], ignore_index=True)
    with pytest.raises(ValueError, match="one native duration"):
        calibrate_for_cutoff(mixed, "2025-01-08T12:00+01:00", _fixed_policy())
    with pytest.raises(ValueError, match="timezone-aware"):
        calibrate_for_cutoff(history, "2025-01-23T12:00", _fixed_policy())
    with pytest.raises(ValueError, match="native durations disagree"):
        calibrate_for_cutoff(history, "2025-01-23T12:00+01:00", _fixed_policy(native_interval_minutes=15))


@pytest.mark.parametrize("settings", [
    {"window_days_candidates": []}, {"covariance_shrinkage_candidates": [.1, .1]},
    {"window_days_candidates": [2]}, {"minimum_complete_days": True},
    {"covariance_shrinkage_candidates": [0]}, {"selection_metric": "test_coverage"},
    {"inner_validation_days": 0},
])
def test_invalid_candidate_policies_are_rejected(settings):
    with pytest.raises(ValueError):
        select_gaussian_policy(_rows(), _settings(**settings))
