"""Verify chronological model selection, point artifacts and growing cache views."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.linear_price_study import (
    LinearPriceForecast, _candidates, _common_end, _group_keys, _predict_candidate, _split, _stamp,
    evaluate_linear_price_study, run_linear_price_study)
from bess_pv_trading.forecasting.prices import RidgePriceModel


def _features(days=54, frequency="60min", start="2025-01-01"):
    """Make realistic ordered products with available historical lag features."""
    first = pd.Timestamp(start, tz="Europe/Rome")
    stop = pd.Timestamp((first.date() + pd.Timedelta(days=days)).isoformat(), tz="Europe/Rome")
    starts = pd.date_range(first.tz_convert("UTC"), stop.tz_convert("UTC"), freq=frequency, inclusive="left")
    local = starts.tz_convert("Europe/Rome")
    minute_count = int(pd.Timedelta(frequency).total_seconds() / 60)
    day_number = np.array([(stamp.date() - first.date()).days for stamp in local])
    signal = np.sin(2 * np.pi * local.hour / 24)
    prices = 90 + 3 * day_number + 10 * signal
    return pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(frequency),
        "delivery_day": [stamp.date().isoformat() for stamp in local],
        "native_interval_minutes": minute_count, "price_eur_mwh": prices,
        "lag_1d": prices - 3, "lag_7d": prices - 21, "signal": signal,
        "retrieved_at": pd.Timestamp("2026-10-16T00:00:00Z")})


def _settings(**overrides):
    """Keep fixtures small while retaining all four ordered study periods."""
    return {"timezone": "Europe/Rome", "validation_days": 7, "calibration_days": 6,
        "test_days": 6, "maturity_gap_days": 1, "min_training_days": 14,
        "refit_every_days": 7, "alphas": [1.0, 100.0], "training_windows_days": [28],
        "targets": ["level", "delta_day"], "feature_sets": {"lags": ["lags"]}, **overrides}


def test_selection_and_validation_predictions_ignore_later_labels():
    frame = _features()
    settings = _settings()
    first = evaluate_linear_price_study(frame, {"lags": ["lag_1d", "signal"]}, settings)
    splits = first["regimes"][0]["split"]
    changed = frame.copy()
    changed.loc[changed.delivery_day.ge(splits["calibration_start_day"]), "price_eur_mwh"] += 5000
    second = evaluate_linear_price_study(changed, {"lags": ["lag_1d", "signal"]}, settings)
    assert first["validation_results"] == second["validation_results"]
    assert first["regimes"][0]["selected_candidate"] == second["regimes"][0]["selected_candidate"]
    pd.testing.assert_frame_equal(first["validation_predictions"], second["validation_predictions"])


def test_forecasts_are_chronological_and_coefficients_update_weekly():
    frame = _features()
    candidate = {"id": "ridge", "kind": "ridge", "feature_names": ["lag_1d", "signal"],
        "alpha": 1.0, "training_window_days": 28, "target": "level"}
    days = sorted(frame.delivery_day.unique())[30:46]
    predicted, refits = _predict_candidate(frame, candidate, days, _settings(),
        pd.Series(True, index=frame.index), lambda day: "test")
    assert [item["refit_delivery_day"] for item in refits] == [days[0], days[7], days[14]]
    assert predicted.training_max_delivery_day.lt(predicted.delivery_day).all()
    assert (predicted.fit_cutoff <= predicted.forecast_cutoff).all()
    assert predicted.prediction_provenance.eq("chronological_out_of_sample").all()
    assert not predicted.operational_eligible.any()
    first_model = refits[0]["model"]
    # Altering the final target cannot affect any earlier fitted prediction.
    changed = frame.copy()
    changed.loc[changed.delivery_day.eq(days[-1]), "price_eur_mwh"] = -99999
    repeated, other_refits = _predict_candidate(changed, candidate, days, _settings(),
        pd.Series(True, index=frame.index), lambda day: "test")
    np.testing.assert_allclose(predicted.prediction, repeated.prediction)
    assert first_model == other_refits[0]["model"]


def test_delta_model_roundtrip_keeps_offset_separate_and_retrospective(tmp_path):
    model = RidgePriceModel(("signal",), np.array([0.]), np.array([1.]), np.array([2.]),
        3., 1., "2025-02-01T11:00:00+00:00", 100, "retrospective_DA_delta_day_unverified_vintages")
    artifact = LinearPriceForecast({"kind": "ridge", "id": "delta", "target": "delta_day",
        "feature_names": ["signal"]}, model.fit_cutoff, "2025-02-01", model)
    path = tmp_path / "model.json"
    artifact.save(path)
    restored = LinearPriceForecast.load(path)
    frame = pd.DataFrame({"signal": [1., -1.], "lag_1d": [100., 80.]})
    np.testing.assert_allclose(restored.predict(frame), [105., 81.])
    assert json.loads(path.read_text())["target_offset"] == "lag_1d"
    payload = json.loads(path.read_text())
    payload["operational_eligible"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="operational"):
        LinearPriceForecast.load(path)


@pytest.mark.parametrize("baseline,expected", [("previous_day", [100., 80.]),
    ("previous_week", [90., 110.]), ("day_week_average", [95., 95.])])
def test_baseline_point_artifacts_are_reusable(baseline, expected, tmp_path):
    artifact = LinearPriceForecast({"kind": "baseline", "baseline": baseline},
        "2025-02-01T11:00:00Z", "2025-02-01")
    artifact.save(tmp_path / "point.json")
    model = LinearPriceForecast.load(tmp_path / "point.json")
    np.testing.assert_allclose(model.predict(pd.DataFrame({"lag_1d": [100., 80.], "lag_7d": [90., 110.]})), expected)


def test_a_winning_baseline_is_selected_without_forcing_a_regression():
    frame = _features()
    frame["lag_1d"] = frame.price_eur_mwh
    outcome = evaluate_linear_price_study(frame, {"lags": ["signal"]}, _settings())
    assert outcome["regimes"][0]["selected_candidate"]["baseline"] == "previous_day"
    assert outcome["regimes"][0]["test_metrics"]["mae_eur_mwh"] == 0
    assert outcome["models"]["60min"] == []


def test_maturity_gap_precedes_first_test_forecast_and_validation():
    split = _split("2025-02-28", _settings())
    assert split["validation_end_day"] < split["calibration_start_day"]
    assert split["calibration_end_day"] < split["maturity_gap_start_day"] < split["test_start_day"]
    first_test_cutoff = pd.Timestamp(split["test_start_day"], tz="Europe/Rome") - pd.Timedelta(hours=12)
    assert pd.Timestamp(split["calibration_cutoff"]) <= first_test_cutoff
    with pytest.raises(ValueError, match="maturity_gap_days"):
        _split("2025-02-28", _settings(maturity_gap_days=0))


@pytest.mark.parametrize("day", ["2025-03-31", "2025-10-27"])
def test_da_study_cutoff_stays_at_local_noon_across_dst(day):
    cutoff = _stamp(day, "Europe/Rome", previous_noon=True).tz_convert("Europe/Rome")
    assert cutoff.hour == 12 and cutoff.minute == 0
    assert cutoff.date() == (pd.Timestamp(day).date() - pd.Timedelta(days=1))


def test_short_regime_is_reported_even_when_baselines_can_predict():
    frame = _features(days=25)
    outcome = evaluate_linear_price_study(frame, {"lags": ["lag_1d"]}, _settings())
    assert outcome["predictions"].empty
    assert outcome["regimes"][0]["status"] == "insufficient_data"


def test_common_end_ignores_partial_products_and_missing_forecasts():
    frame = _features(days=3)
    frame["load_forecast_mw"] = 20000.
    frame.loc[frame.index[-1], "load_forecast_mw"] = np.nan
    assert _common_end(frame, ["load_forecast_mw"], "Europe/Rome") == "2025-01-02"
    incomplete = frame.loc[frame.delivery_day.ne("2025-01-03")].drop(index=24)
    assert _common_end(incomplete, ["load_forecast_mw"], "Europe/Rome") == "2025-01-01"


def test_native_regimes_are_never_pooled():
    hourly = _features()
    quarters = _features(frequency="15min", start="2025-10-01")
    outcome = evaluate_linear_price_study(pd.concat([hourly, quarters]), {"lags": ["lag_1d"]}, _settings())
    assert {item["native_interval_minutes"] for item in outcome["regimes"]} == {15, 60}
    for minutes, rows in outcome["predictions"].groupby("native_interval_minutes"):
        assert ((rows.product_end - rows.product_start).dt.total_seconds() / 60).eq(minutes).all()


def test_run_exports_frozen_inputs_contract_and_cache_provenance(tmp_path, monkeypatch):
    from bess_pv_trading.forecasting import linear_price_study
    rows = _features()
    source = [{"manifest": "fixture/manifest.json", "normalized_sha256": "abc", "checksums_verified": True}]
    monkeypatch.setattr(linear_price_study, "load_api_cache_records", lambda *args, **kwargs: (rows, source))
    settings = _settings(output_directory=str(tmp_path / "runs"),
        analysis_directory=str(tmp_path / "analysis"),
        native_product_regimes=[{"start": "2025-01-01", "end": None, "minutes": 60}],
        feature_sets={"lags": ["calendar", "autoregressive"]},
        feature_settings={"lag_days": [1, 7], "daily_summary_lags": [1], "rolling_windows": [7], "daily_profile_lags": [1]})
    result = run_linear_price_study({"linear_price_study": settings})
    assert result["status"] == "completed" and not result["operational_eligible"]
    directory = Path(result["output_dir"])
    assert len(result["frozen_features_sha256"]) == 64
    assert json.loads((directory / "source_caches.json").read_text()) == {"prices": source}
    assert (directory / "validation_candidate_scores.json").is_file()
    assert (directory / "daily_metrics.csv").is_file()
    assert (directory / "test_60min.png").is_file()
    assert (Path(result["derived_monthly_directories"][0]) / "features.parquet").is_file()
    calibration = pd.read_parquet(result["calibration_predictions"])
    tests = pd.read_parquet(result["test_predictions"])
    assert calibration.delivery_day.max() < tests.delivery_day.min()
    assert calibration.product_end.max() <= tests.forecast_cutoff.min()
    artifact = LinearPriceForecast.load(result["regimes"][0]["point_model"])
    assert not artifact.operational_eligible
    assert "not an unseen prospective" in result["prior_evaluation_overlap"]


def _varying_hour_features(days=54, frequency="15min"):
    """Make a truly linear price relation whose slope differs by local hour."""
    frame = _features(days, frequency)
    local = frame.product_start.dt.tz_convert("Europe/Rome")
    day_index = np.array([(stamp.date() - local.iloc[0].date()).days for stamp in local])
    frame["signal"] = np.sin(day_index * .53) + local.dt.minute.to_numpy() / 60
    slopes = np.where(local.dt.hour < 12, 4., -3.)
    frame["price_eur_mwh"] = 100 + 2 * local.dt.hour + slopes * frame.signal
    return frame


def test_grouped_hour_fits_have_independent_coefficients_and_reusable_artifacts(tmp_path):
    frame = _varying_hour_features()
    candidate = {"id": "hourly", "kind": "ridge", "model_structure": "by_hour",
        "feature_names": ["signal"], "alpha": 0., "training_window_days": 28, "target": "level"}
    days = sorted(frame.delivery_day.unique())[30:39]
    prediction, refits = _predict_candidate(frame, candidate, days, _settings(),
        pd.Series(True, index=frame.index), lambda day: "test")
    assert len(prediction) == 9 * 96
    np.testing.assert_allclose(prediction.prediction, prediction.actual, atol=1e-10)
    assert len(refits[0]["group_models"]) == 24
    assert refits[0]["schema_version"] == "linear-price-point-model-v2"
    assert prediction.model_structure.eq("by_hour").all()
    assert prediction.training_max_delivery_day.lt(prediction.delivery_day).all()
    assert (prediction.fit_cutoff <= prediction.forecast_cutoff).all()
    record = dict(refits[0])
    record.pop("refit_delivery_day")
    path = tmp_path / "hour_models.json"
    path.write_text(json.dumps(record))
    restored = LinearPriceForecast.load(path)
    assert restored.model is None and len(restored.group_models) == 24
    selected = frame.loc[frame.delivery_day.eq(days[0])]
    np.testing.assert_allclose(restored.predict(selected), selected.price_eur_mwh, atol=1e-10)
    # Every scaler uses only its hour's earlier rows, never another hour.
    first_model = restored.group_models["15min/hour00/fold0"]
    train = frame.loc[frame.delivery_day.between("2025-01-03", "2025-01-30")]
    midnight = train.loc[train.product_start.dt.tz_convert("Europe/Rome").dt.hour.eq(0)]
    assert first_model.center[0] == pytest.approx(midnight.signal.mean())
    assert first_model.coefficients[0] / first_model.scale[0] == pytest.approx(4.)
    evening_model = restored.group_models["15min/hour18/fold0"]
    assert evening_model.coefficients[0] / evening_model.scale[0] == pytest.approx(-3.)


def test_grouped_selection_and_validation_are_independent_of_later_outcomes():
    frame = _varying_hour_features(frequency="60min")
    settings = _settings(model_structures=["pooled", "by_hour"], alphas=[0.], targets=["level"])
    outcome = evaluate_linear_price_study(frame, {"lags": ["signal"]}, settings)
    winner = outcome["regimes"][0]["selected_candidate"]
    assert winner["model_structure"] == "by_hour"
    boundary = outcome["regimes"][0]["split"]["calibration_start_day"]
    changed = frame.copy()
    changed.loc[changed.delivery_day.ge(boundary), "price_eur_mwh"] += 10000
    repeated = evaluate_linear_price_study(changed, {"lags": ["signal"]}, settings)
    assert outcome["validation_results"] == repeated["validation_results"]
    assert winner == repeated["regimes"][0]["selected_candidate"]
    pd.testing.assert_frame_equal(outcome["validation_predictions"], repeated["validation_predictions"])


def test_grouping_keeps_dst_folds_and_native_durations_separate(tmp_path):
    starts = pd.to_datetime(["2025-10-26T00:00:00Z", "2025-10-26T01:00:00Z"], utc=True)
    frame = pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(minutes=15),
        "native_interval_minutes": 15, "signal": [1., 1.], "lag_1d": [50., 50.]})
    assert list(_group_keys(frame, "by_hour", "Europe/Rome")) == ["15min/hour02/fold0", "15min/hour02/fold1"]
    model = RidgePriceModel(("signal",), np.array([0.]), np.array([1.]), np.array([2.]),
        3., 1., "2025-10-25T10:00:00Z", 100, "retrospective_DA_level_unverified_vintages")
    artifact = LinearPriceForecast({"kind": "ridge", "model_structure": "by_hour", "target": "level",
        "feature_names": ["signal"]}, model.fit_cutoff, "2025-10-24",
        group_models={"15min/hour02/fold0": model})
    assert artifact.predict(frame.iloc[:1])[0] == 5
    with pytest.raises(ValueError, match="fold1"):
        artifact.predict(frame.iloc[1:])
    hourly = frame.iloc[:1].copy()
    hourly["product_end"] = hourly.product_start + pd.Timedelta(hours=1)
    hourly["native_interval_minutes"] = 60
    with pytest.raises(ValueError, match="60min"):
        artifact.predict(hourly)


def test_product_specific_delta_models_preserve_quarter_clock_and_day_offset(tmp_path):
    frame = _varying_hour_features()
    local = frame.product_start.dt.tz_convert("Europe/Rome")
    frame["price_eur_mwh"] = frame.lag_1d + (local.dt.minute / 15 + 1) * frame.signal
    candidate = {"id": "product", "kind": "ridge", "model_structure": "by_product",
        "feature_names": ["signal"], "alpha": 0., "training_window_days": 28, "target": "delta_day"}
    days = sorted(frame.delivery_day.unique())[30:32]
    predicted, refits = _predict_candidate(frame, candidate, days, _settings(),
        pd.Series(True, index=frame.index), lambda day: "test")
    assert len(refits[0]["group_models"]) == 96
    assert "15min/00:15/fold0" in refits[0]["group_models"]
    np.testing.assert_allclose(predicted.prediction, predicted.actual, atol=1e-10)
    record = dict(refits[0])
    record.pop("refit_delivery_day")
    path = tmp_path / "product.json"
    path.write_text(json.dumps(record))
    restored = LinearPriceForecast.load(path)
    assert restored.to_dict()["target_offset"] == "lag_1d"
    selected = frame.loc[frame.delivery_day.eq(days[0])]
    np.testing.assert_allclose(restored.predict(selected), selected.price_eur_mwh, atol=1e-10)


def test_grouped_candidates_reject_insufficient_group_history_without_borrowing():
    frame = _varying_hour_features(frequency="60min")
    # This hour appears just before validation and cannot meet fourteen days.
    local = frame.product_start.dt.tz_convert("Europe/Rome")
    keep = local.dt.hour.ne(18) | frame.delivery_day.ge("2025-02-01")
    frame = frame.loc[keep].reset_index(drop=True)
    outcome = evaluate_linear_price_study(frame, {"lags": ["signal"]},
        _settings(model_structures=["by_hour"], alphas=[0.], targets=["level"]))
    grouped = [record for record in outcome["validation_results"] if record["candidate"]["kind"] == "ridge"]
    assert grouped[0]["status"] == "insufficient_training_or_validation"
    assert outcome["regimes"][0]["status"] == "insufficient_data"


def test_grouped_candidate_grid_is_explicit_and_keeps_pooled_identifiers():
    settings = _settings(model_structures=["pooled", "by_hour"], alphas=[1.], targets=["level"])
    candidates = _candidates({"lags": ["signal"]}, settings)
    assert len(candidates) == 5
    assert {row["id"] for row in candidates if row["kind"] == "ridge"} == {
        "ridge_lags_a1_w28_level", "ridge_lags_a1_w28_level_by_hour"}
    with pytest.raises(ValueError, match="model_structures"):
        _candidates({"lags": ["signal"]}, _settings(model_structures=["by_hour", "by_hour"]))
