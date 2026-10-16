"""Offline mean-study to Gaussian-path integration and untouched-test checks."""

from datetime import timedelta
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.gaussian_prices import GaussianPriceCalibration
from bess_pv_trading.forecasting.price_pipeline import attach_gaussian_study


def _predictions(start_day, days, *, minutes=60, actual_shift=0.):
    """Make complete OOS native days with correlated slot forecast errors."""
    day_frames = []
    first = pd.Timestamp(start_day).date()
    for offset in range(days):
        day = first + timedelta(days=offset)
        midnight = pd.Timestamp(day, tz="Europe/Rome")
        following = pd.Timestamp(day + timedelta(days=1), tz="Europe/Rome")
        starts = pd.date_range(midnight, following, freq=f"{minutes}min", inclusive="left").tz_convert("UTC")
        slots = np.arange(len(starts))
        prediction = 80 + 8 * np.cos(2 * np.pi * slots / len(slots))
        # Both daily-level and slot-shaped components vary across days; the
        # covariance should consequently retain meaningful off-diagonal terms.
        errors = 1.5 + (offset - 2) * (.8 + .3 * np.sin(slots / 4))
        day_frames.append(pd.DataFrame({
            "delivery_day": day.isoformat(), "product_start": starts,
            "product_end": starts + pd.Timedelta(minutes=minutes),
            "actual": prediction + errors + actual_shift, "prediction": prediction,
            "native_interval_minutes": minutes,
            "fit_cutoff": pd.Timestamp(day - timedelta(days=1), tz="Europe/Rome").tz_convert("UTC"),
            "forecast_cutoff": (pd.Timestamp(day - timedelta(days=1), tz="Europe/Rome")
                                + pd.Timedelta(hours=12)).tz_convert("UTC"),
            "training_max_delivery_day": (day - timedelta(days=2)).isoformat(),
            "prediction_provenance": "chronological_out_of_sample",
        }))
    return pd.concat(day_frames, ignore_index=True)


def _study(directory, *, actual_shift=0.):
    """Save six calibration days, a full gap day, and two untouched test days."""
    directory.mkdir(parents=True)
    _predictions("2025-01-10", 6).to_parquet(directory / "calibration_predictions.parquet", index=False)
    _predictions("2025-01-17", 2, actual_shift=actual_shift).to_parquet(directory / "test_predictions.parquet", index=False)
    manifest = directory / "manifest.json"
    manifest.write_text(json.dumps({"schema": "unit-test-linear-study", "status": "complete"}), encoding="utf-8")
    return {"output_dir": str(directory), "manifest": str(manifest)}


def _config(**settings):
    return {"linear_price_study": {"timezone": "Europe/Rome"}, "price_scenarios": {
        "minimum_complete_days": 3, "sample_count": 100, "seed": 19,
        "covariance_shrinkage": .4, **settings,
    }}


@pytest.fixture(autouse=True)
def _skip_expensive_plot(monkeypatch):
    """Keep integration focused on numerical artifacts rather than rendering."""
    monkeypatch.setattr("bess_pv_trading.forecasting.price_pipeline._plot_paths", lambda *args: None)


def test_gaussian_attachment_saves_reloadable_scores_and_seeded_complete_day_paths(tmp_path):
    study = _study(tmp_path / "study")
    progress = []
    result = attach_gaussian_study(study, _config(), progress=progress.append)
    assert result["status"] == "complete" and result["operational_eligible"] is False
    assert result["variance_source"] == "chronological out-of-sample calibration errors, not raw price variance"
    assert len(result["study_manifest_sha256"]) == 64
    assert len(progress) == 1 and "60-minute" in progress[0]
    regime = result["regimes"][0]
    assert regime["status"] == "evaluated" and regime["sample_count"] == 100
    assert regime["sample_day"] == "2025-01-17"
    assert regime["test_metrics"]["scored_rows"] == 48
    assert regime["test_metrics"]["scored_days"] == 2
    assert regime["test_metrics"]["consecutive_product_ramps"]["rows"] == 46
    artifact_path = Path(regime["artifact"])
    artifact = GaussianPriceCalibration.load(artifact_path)
    grid = next(iter(artifact.grids.values()))
    assert grid.calibration_days == 6
    assert grid.first_delivery_day == "2025-01-10" and grid.last_delivery_day == "2025-01-15"
    assert artifact.operational_eligible is False
    assert artifact.provenance == "chronological_out_of_sample"
    assert np.any(np.abs(grid.covariance_eur2_mwh2 - np.diag(np.diag(grid.covariance_eur2_mwh2))) > 0)
    products = pd.read_parquet(artifact_path.parent / "products_2025-01-17.parquet")
    paths = np.load(regime["sample_paths"])
    assert paths["prices_eur_mwh"].shape == (100, 24)
    assert paths["innovations_eur_mwh"].shape == (100, 24)
    assert np.allclose(paths["probabilities"], .01)
    assert np.allclose(paths["prices_eur_mwh"],
                       paths["point_forecast_eur_mwh"] + paths["bias_eur_mwh"] + paths["innovations_eur_mwh"])
    regenerated = artifact.sample_future(products.prediction.to_numpy(), products, 100,
        decision_cutoff=products.forecast_cutoff.min(), seed=19, require_operational=False)
    assert np.array_equal(paths["prices_eur_mwh"], regenerated)
    scores = pd.read_parquet(artifact_path.parent / "test_scores.parquet")
    assert scores.delivery_day.nunique() == 2 and len(scores) == 48
    assert {"sigma_eur_mwh", "crps_eur_mwh", "covered_95", "gaussian_mean"}.issubset(scores.columns)
    assert json.loads((artifact_path.parent / "metrics.json").read_text())["metrics"]["scored_rows"] == 48
    assert json.loads((Path(result["output_dir"]) / "manifest.json").read_text()) == result


def test_changing_only_held_out_actuals_cannot_change_calibration_or_sampled_paths(tmp_path):
    original = attach_gaussian_study(_study(tmp_path / "original"), _config())
    perturbed = attach_gaussian_study(_study(tmp_path / "perturbed", actual_shift=1000), _config())
    first, second = original["regimes"][0], perturbed["regimes"][0]
    assert Path(first["artifact"]).read_bytes() == Path(second["artifact"]).read_bytes()
    with np.load(first["sample_paths"]) as a, np.load(second["sample_paths"]) as b:
        assert set(a.files) == set(b.files)
        for name in a.files:
            assert np.array_equal(a[name], b[name]), name
    assert second["test_metrics"]["mae_eur_mwh"] > first["test_metrics"]["mae_eur_mwh"] + 900
    assert second["test_metrics"]["mean_crps_eur_mwh"] > first["test_metrics"]["mean_crps_eur_mwh"]


@pytest.mark.parametrize("count", [0, -1, True, 1.5, "100"])
def test_malformed_sample_count_is_rejected_before_artifact_publication(tmp_path, count):
    study = _study(tmp_path / "study")
    with pytest.raises(ValueError, match="sample_count must be a positive integer"):
        attach_gaussian_study(study, _config(sample_count=count))
    assert not (Path(study["output_dir"]) / "uncertainty").exists()


def test_test_forecasts_cannot_be_issued_before_calibration_was_frozen(tmp_path):
    study = _study(tmp_path / "study")
    test_path = Path(study["output_dir"]) / "test_predictions.parquet"
    rows = pd.read_parquet(test_path)
    rows["forecast_cutoff"] = pd.Timestamp("2025-01-12T12:00Z")
    rows["fit_cutoff"] = pd.Timestamp("2025-01-12T00:00Z")
    rows["training_max_delivery_day"] = "2025-01-10"
    rows.to_parquet(test_path, index=False)
    with pytest.raises(ValueError, match="Untouched test forecasts and delivery must follow the calibration cutoff"):
        attach_gaussian_study(study, _config())
    assert not (Path(study["output_dir"]) / "uncertainty").exists()


def test_sample_uses_latest_issuance_cutoff_when_native_products_have_different_forecast_times(tmp_path, monkeypatch):
    study = _study(tmp_path / "study")
    test_path = Path(study["output_dir"]) / "test_predictions.parquet"
    rows = pd.read_parquet(test_path)
    first_day = rows.delivery_day.eq("2025-01-17")
    earlier = pd.Timestamp("2025-01-16T09:00Z")
    later = pd.Timestamp("2025-01-16T14:00Z")
    rows.loc[first_day, "forecast_cutoff"] = earlier
    rows.loc[first_day & rows.product_start.dt.hour.ge(12), "forecast_cutoff"] = later
    rows.to_parquet(test_path, index=False)
    recorded = []
    original = GaussianPriceCalibration.sample_future

    def record_cutoff(self, *args, **kwargs):
        recorded.append(kwargs["decision_cutoff"])
        return original(self, *args, **kwargs)

    monkeypatch.setattr(GaussianPriceCalibration, "sample_future", record_cutoff)
    attach_gaussian_study(study, _config())
    assert recorded == [later]


def test_no_shared_native_regime_cannot_publish_a_complete_uncertainty_manifest(tmp_path):
    study = _study(tmp_path / "study")
    test_path = Path(study["output_dir"]) / "test_predictions.parquet"
    _predictions("2025-01-17", 2, minutes=15).to_parquet(test_path, index=False)
    with pytest.raises(ValueError):
        attach_gaussian_study(study, _config())
    assert not (Path(study["output_dir"]) / "uncertainty").exists()


def test_native_day_paths_cannot_use_forecasts_issued_after_the_first_product_started(tmp_path):
    study = _study(tmp_path / "study")
    test_path = Path(study["output_dir"]) / "test_predictions.parquet"
    rows = pd.read_parquet(test_path)
    later_products = rows.delivery_day.eq("2025-01-17") & rows.product_start.ge(pd.Timestamp("2025-01-17T12:00Z"))
    # Every row remains issued before its own delivery product; together they
    # cannot form a full-day forecast available before the day's first product.
    rows.loc[later_products, "forecast_cutoff"] = pd.Timestamp("2025-01-17T09:00Z")
    rows.to_parquet(test_path, index=False)
    with pytest.raises(ValueError, match="Future price products begin before the decision cutoff"):
        attach_gaussian_study(study, _config())
    assert not (Path(study["output_dir"]) / "uncertainty").exists()


def test_pipeline_requires_both_calibration_and_test_rows(tmp_path):
    study = _study(tmp_path / "study")
    test_path = Path(study["output_dir"]) / "test_predictions.parquet"
    rows = pd.read_parquet(test_path)
    rows.iloc[:0].to_parquet(test_path, index=False)
    with pytest.raises(ValueError, match="nonempty separate calibration and test predictions"):
        attach_gaussian_study(study, _config())
