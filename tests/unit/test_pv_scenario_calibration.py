"""Chronology, raw-target preservation and safe affine-calibration artifacts."""

from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting import pv_scenario_calibration as calibration
from bess_pv_trading.scenario_research import ScenarioTrajectories, transform_dependence, weighted_crps


def synthetic_case(count=24, start="2026-01-01T00:00:00Z"):
    """A known location shift isolates the implemented two-parameter objective."""
    starts = pd.date_range(start, periods=count, freq="15min")
    mean = np.linspace(0.15, 0.6, count)
    raw = np.array([mean - 0.03, mean, mean + 0.03])
    observed = mean + 0.12
    return raw, observed, starts


def fit_case(raw, observed, starts, **extra):
    return calibration.fit_bias_spread(raw, observed, ac_nameplate_mw=1, interval_starts=starts,
        observed_available_at=starts + pd.Timedelta(minutes=15), fit_cutoff=starts[-1] + pd.Timedelta(minutes=15),
        settings=calibration.CalibrationSettings(minimum_training_intervals=4), **extra)


def test_fit_optimizes_exact_empirical_crps_and_recovers_known_bias():
    raw, observed, starts = synthetic_case()
    model = fit_case(raw, observed, starts)
    adjusted = calibration.apply_bias_spread(raw, bias_normalized=model["bias_normalized"],
        spread_scale=model["spread_scale"], ac_nameplate_mw=1)
    assert model["bias_normalized"] == pytest.approx(0.12, abs=2e-4)
    assert model["calibrated_training_crps_normalized"] == pytest.approx(weighted_crps(adjusted, observed).mean())
    assert model["calibrated_training_crps_normalized"] < model["raw_training_crps_normalized"]
    assert model["training_intervals"] == 24
    assert model["optimization"]["global_optimum_claimed"] is False


def test_future_and_excluded_targets_cannot_change_fitted_parameters():
    raw, observed, starts = synthetic_case()
    cutoff = starts[11] + pd.Timedelta(minutes=15)
    eligible = np.ones(len(starts), dtype=bool)
    eligible[2] = False
    settings = calibration.CalibrationSettings(minimum_training_intervals=4)
    kwargs = dict(ac_nameplate_mw=1, interval_starts=starts, observed_available_at=starts + pd.Timedelta(minutes=15),
        fit_cutoff=cutoff, eligible=eligible, settings=settings)
    original = calibration.fit_bias_spread(raw, observed, **kwargs)
    changed = observed.copy()
    changed[12:] = 999
    changed[2] = 999
    alternate = calibration.fit_bias_spread(raw, changed, **kwargs)
    assert original["bias_normalized"] == alternate["bias_normalized"]
    assert original["spread_scale"] == alternate["spread_scale"]
    assert original["training_targets_sha256"] == alternate["training_targets_sha256"]
    assert original["training_intervals"] == 11
    assert pd.Timestamp(original["latest_training_target_available_at_utc"]) <= cutoff


def test_missing_negative_and_flagged_targets_are_excluded_without_mutation():
    raw, observed, starts = synthetic_case()
    observed[0] = np.nan
    observed[1] = -0.1
    eligible = np.ones(len(starts), dtype=bool)
    eligible[2] = False
    before = observed.copy()
    model = fit_case(raw, observed, starts, eligible=eligible)
    assert model["training_intervals"] == 21
    np.testing.assert_array_equal(observed, before)
    assert np.isnan(observed[0])
    assert observed[1] == -0.1


def test_forecast_transform_preserves_ac_bounds_member_ranks_and_night_prior():
    raw = np.array([[0, 0.2, 0.6], [0, 0.3, 0.8], [0, 0.4, 1.0]])
    calibrated = calibration.apply_bias_spread(raw, bias_normalized=0.1, spread_scale=2, ac_nameplate_mw=1,
        night_mask=np.array([True, False, False]))
    assert calibrated.min() >= 0
    assert calibrated.max() <= 1
    np.testing.assert_array_equal(calibrated[:, 0], 0)
    assert np.all(np.diff(calibrated, axis=0) >= 0)
    with pytest.raises(ValueError, match="spread"):
        calibration.apply_bias_spread(raw, bias_normalized=0, spread_scale=-1, ac_nameplate_mw=1)


def test_ecc_and_shuffled_calibrated_laws_have_identical_marginal_scores():
    raw, observed, _ = synthetic_case()
    calibrated = calibration.apply_bias_spread(raw, bias_normalized=0.1, spread_scale=1.5, ac_nameplate_mw=1)
    law = ScenarioTrajectories(raw, np.full(3, 1 / 3), ("a", "b", "c"))
    ecc = transform_dependence(law, "ecc", calibrated_pv_marginals=calibrated)
    shuffled = transform_dependence(ecc, "independent_shuffle", seed=9)
    scores = [calibration.calibration_scores(values, observed, np.ones(len(observed), dtype=bool))
        for values in (ecc.pv_ac_mw, shuffled.pv_ac_mw)]
    for key in ("crps_mw", "mae_mean_mw", "central_interval_coverage", "mean_central_interval_width_mw"):
        assert scores[0][key] == pytest.approx(scores[1][key])


def test_npz_exports_safe_parameters_and_per_interval_forecast_clocks(tmp_path):
    raw, observed, starts = synthetic_case(start="2026-01-15T00:00:00Z")
    model = fit_case(*synthetic_case())
    path = tmp_path / "farm_1_2026-01-15.npz"
    artifact = calibration.save_calibration_npz(path, model=model, farm_id=1, interval_starts=starts,
        raw_scenarios=raw, scenario_ids=("deterministic", "weather_0", "weather_1"),
        forecast_available_at=["2026-01-14T06:00:00Z"] * len(starts),
        forecast_vintage=["2026-01-14T00:00:00Z"] * len(starts))
    assert artifact["scenario_count"] == 3
    assert len(artifact["sha256"]) == 64
    with np.load(path, allow_pickle=False) as saved:
        for name in saved.files:
            saved[name]
        assert saved["artifactkind"].item() == "affine_marginal_calibration"
        assert saved["calibration_bias_normalized"].item() == model["bias_normalized"]
        assert saved["forecast_available_at_utc"].shape == (len(starts),)
        assert saved["scenario_ids"].tolist() == ["deterministic", "weather_0", "weather_1"]
        assert saved["pv_ac_mw"].shape == raw.shape


def test_export_rejects_future_forecasts_and_models(tmp_path):
    raw, _, starts = synthetic_case(start="2026-01-15T00:00:00Z")
    model = fit_case(*synthetic_case())
    kwargs = dict(model=model, farm_id=1, interval_starts=starts, raw_scenarios=raw, scenario_ids=("a", "b", "c"),
        forecast_available_at=["2026-01-16T06:00:00Z"] * len(starts),
        forecast_vintage=["2026-01-14T00:00:00Z"] * len(starts))
    with pytest.raises(ValueError, match="available"):
        calibration.save_calibration_npz(tmp_path / "future.npz", **kwargs)
    kwargs["forecast_available_at"] = ["2026-01-14T06:00:00Z"] * len(starts)
    kwargs["model"] = {**model, "fit_cutoff_utc": "2026-01-16T00:00:00Z"}
    with pytest.raises(ValueError, match="unavailable"):
        calibration.save_calibration_npz(tmp_path / "future.npz", **kwargs)


@pytest.mark.parametrize("settings", [
    {"bias_bounds_normalized": [0.1, 0.2]}, {"spread_bounds": [-1, 2]}, {"max_evaluations": True},
    {"minimum_training_intervals": 0}, {"objective_tolerance": 0}, {"central_interval_probability": 1},
])
def test_invalid_calibration_choices_fail(settings):
    with pytest.raises(ValueError):
        calibration.CalibrationSettings(**settings)


def test_raw_data_preparation_excludes_reviewed_and_missing_rows_without_using_treated_targets(tmp_path, monkeypatch):
    import bess_pv_trading.data as data_module
    from bess_pv_trading.markets import AuctionCalendar
    from bess_pv_trading.simulation import continuous_inputs, gme_oracle
    from bess_pv_trading.domain import DeliveryGrid

    starts = pd.date_range("2026-01-02T12:00:00Z", periods=4, freq="15min")
    grid = DeliveryGrid(tuple(starts.astype(str)), tuple(starts.to_pydatetime()), tuple((starts + pd.Timedelta(minutes=15)).to_pydatetime()))
    schedule = (SimpleNamespace(submission_at=pd.Timestamp("2026-01-01T11:00:00Z")),)
    calendar = SimpleNamespace(delivery_grid=lambda day: SimpleNamespace(to_domain=lambda: grid), schedule=lambda day: schedule)
    monkeypatch.setattr(AuctionCalendar, "from_config", lambda config: calendar)
    monkeypatch.setattr(gme_oracle, "_load_pv_forecast", lambda *args: SimpleNamespace(
        values=pd.DataFrame({"pv_ac_mw": [0.9] * 4}), forecast_vintage=pd.Timestamp("2026-01-01T00:00:00Z"),
        available_at=pd.Timestamp("2026-01-01T06:00:00Z")))

    def ensembles(config, daily):
        daily["2026-01-02"].update(ensemble_members=(0, 1), ensemble_pv=np.ones((2, 4)) * 0.8)
        return {"2026-01-02": {"available_at_utc": "2026-01-01T06:00:00Z", "forecast_vintage_utc": "2026-01-01T00:00:00Z"}}

    monkeypatch.setattr(continuous_inputs, "_prepare_pv_ensembles", ensembles)
    monkeypatch.setattr(data_module, "open_store", lambda config: nullcontext(object()))
    frame = pd.DataFrame({"time": starts, "raw_value": [np.nan, -100, 200, 300],
        "treated_value": [900, 900, 900, 900]})
    monkeypatch.setattr(data_module, "load_observations", lambda *args, **kwargs: SimpleNamespace(frame=frame))
    monkeypatch.setattr(data_module, "load_metadata", lambda *args: {1: SimpleNamespace(ac_capacity_mw=1)})
    exclusions = tmp_path / "excluded.csv"
    pd.DataFrame([{"farm_id": 1, "start_utc": starts[2], "end_utc": starts[2],
        "decision": "exclude", "reason": "reviewed_exclusion"}]).to_csv(exclusions, index=False)
    config = {"farm_id": 1, "forecast": {"realized_power_variable": "power", "realized_power_scale_to_mw": 0.001},
        "source": {"telemetry_latency_minutes": 0}, "inputs": {"zarr": "fake.zarr", "metadata_parquet": "fake.parquet"},
        "pv_data_quality": {"exclusions_csv": str(exclusions)}, "policy": {"night": {"enabled": False}},
        "pv_scenario_calibration": {"include_deterministic_member": True}}
    result = calibration.prepare_calibration_data(config, ["2026-01-02"])
    np.testing.assert_array_equal(result["observed"], [np.nan, -0.1, 0.2, 0.3])
    np.testing.assert_array_equal(result["eligible"], [False, False, False, True])
    assert result["provenance"]["fallback_targets_used"] == 0
    assert result["provenance"]["automatic_fault_detection"] is False
    assert result["raw_scenarios"].shape == (3, 4)


def prepared_fixture(start):
    raw, observed, starts = synthetic_case(start=start)
    audit = pd.DataFrame({"interval_start_utc": starts})
    for column in ("raw_nonfinite", "raw_negative", "initial_forecast_fallback", "episode_excluded", "review_excluded"):
        audit[column] = False
    return {"starts": starts, "raw_scenarios": raw, "observed": observed,
        "observed_available_at": starts + pd.Timedelta(minutes=15), "eligible": np.ones(len(starts), dtype=bool),
        "audit": audit, "night_mask": np.zeros(len(starts), dtype=bool), "scenario_ids": ("a", "b", "c"),
        "forecast_available_at": starts - pd.Timedelta(hours=18), "forecast_vintage": starts - pd.Timedelta(hours=24),
        "ac_nameplate_mw": 1, "provenance": {"source": "synthetic_fixture"}}


def test_workflow_preflight_never_fits_and_evaluation_cannot_select_model(tmp_path, monkeypatch):
    config = {"farm_id": 1, "pv_scenario_calibration": {"training": {"start_date": "2026-01-01", "end_date": "2026-01-02",
        "fit_cutoff_utc": "2026-01-03T00:00:00Z"}, "evaluation_windows": [{"id": "evaluation", "start_date": "2026-01-04", "end_date": "2026-01-04"}],
        "parameters": {"minimum_training_intervals": 4}, "output_directory": str(tmp_path), "plots": False, "seed": 5}}
    monkeypatch.setattr(calibration, "prepare_calibration_data", lambda config, days: prepared_fixture(f"{days[0]}T00:00:00Z"))
    check = calibration.run_farm_calibration(config, check_only=True)
    assert check["fits_started"] == 0
    assert list(tmp_path.iterdir()) == []
    result = calibration.run_farm_calibration(config)
    assert result["status"] == "complete"
    assert (tmp_path / "farm_1_2026-01-04.npz").is_file()
    assert (tmp_path / "farm_1" / "model.json").is_file()
    assert result["evaluations"][0]["marginal_crps_difference_ecc_minus_shuffle"] == pytest.approx(0, abs=1e-12)
    assert "never select or refit" in result["evaluations"][0]["selection"]
