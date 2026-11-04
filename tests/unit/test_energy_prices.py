"""Causal publication clocks, chronological calibration and price artifact tests."""

from dataclasses import replace
import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.energy_prices import (
    CausalPriceForecaster, PriceForecastSettings, load_price_history,
    price_forecast_scores, validate_price_history, prepare_price_history,
    prepare_exogenous_features, historical_decision_times,
)


def price_history():
    """Synthetic prices expose daily shape and a changing level without data access."""
    starts = pd.date_range("2025-12-01", "2026-01-06", freq="15min", inclusive="left", tz="UTC")
    hour = starts.hour + starts.minute / 60
    values = 50 + 20 * np.sin(2 * np.pi * hour / 24) + np.arange(len(starts)) / 1000
    return pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(minutes=15),
                         "available_at": starts + pd.Timedelta(hours=4, minutes=15),
                         "price_eur_per_mwh": values})


def fitted(history=None, **settings):
    """Keep test calibration short while preserving a chronological training prefix."""
    return CausalPriceForecaster.fit(price_history() if history is None else history,
        cutoff="2026-01-01T00:00:00Z", settings=PriceForecastSettings(min_training_rows=96,
            calibration_days=3, **settings))


def test_no_future_training_labels_or_lag_values_enter_predictions():
    history = price_history()
    original = fitted(history)
    changed = history.copy()
    changed.loc[changed.available_at > pd.Timestamp("2026-01-01T00:00:00Z"), "price_eur_per_mwh"] += 1e6
    alternate = fitted(changed)
    targets = pd.date_range("2026-01-01T00:00:00Z", periods=96, freq="15min")
    expected = original.predict(targets, cutoff=targets[0])
    actual = alternate.predict(targets, cutoff=targets[0], history=changed)
    np.testing.assert_allclose(expected.point_eur_per_mwh, actual.point_eur_per_mwh)
    assert original.history.available_at.max() <= targets[0]
    assert original.provenance["complete_calibration_days"] >= 1
    assert original.provenance["calibration_rows"] > 0


def test_model_cannot_be_used_before_fit_and_missing_history_fails():
    model = fitted()
    with pytest.raises(ValueError, match="not available"):
        model.predict(["2026-01-01T00:00:00Z"], cutoff="2025-12-31T00:00:00Z")
    with pytest.raises(ValueError, match="causally available"):
        model.predict(["2027-01-01T00:00:00Z"], cutoff="2027-01-01T00:00:00Z")


def test_residual_scenarios_sample_whole_daily_paths_and_repeat_seed():
    model = fitted()
    targets = pd.date_range("2026-01-02", periods=96, freq="15min", tz="Europe/Rome").tz_convert("UTC")
    prediction = model.predict(targets, cutoff="2026-01-01T00:00:00Z", scenario_count=8, seed=3)
    repeated = model.predict(targets, cutoff="2026-01-01T00:00:00Z", scenario_count=8, seed=3)
    np.testing.assert_array_equal(prediction.scenarios_eur_per_mwh, repeated.scenarios_eur_per_mwh)
    residuals = prediction.scenarios_eur_per_mwh - prediction.point_eur_per_mwh
    blocks = {block["day"]: block["residual_eur_per_mwh"] for block in model.residual_blocks}
    selected = prediction.provenance["sampled_blocks"][0]["sampled_residual_days"]
    for row, day in enumerate(selected):
        np.testing.assert_allclose(residuals[row], blocks[day], atol=1e-12)
    assert not prediction.scenarios_eur_per_mwh.flags.writeable


def test_no_calibration_means_no_residual_scenarios():
    model = CausalPriceForecaster.fit(price_history(), cutoff="2026-01-01T00:00:00Z",
        settings=PriceForecastSettings(calibration_days=0, min_training_rows=96))
    with pytest.raises(ValueError, match="complete chronological"):
        model.predict(["2026-01-01T00:00:00Z"], cutoff="2026-01-01T00:00:00Z", scenario_count=2)


def test_seasonal_persistence_uses_published_same_slot_fallback():
    model = fitted(method="seasonal_persistence", lag_hours=(24,))
    targets = pd.DatetimeIndex([pd.Timestamp("2026-01-02T00:00:00Z")])
    prediction = model.predict(targets, cutoff="2026-01-01T00:00:00Z")
    expected = price_history().set_index("product_start").loc["2025-12-31T00:00:00Z", "price_eur_per_mwh"]
    assert prediction.point_eur_per_mwh[0] == pytest.approx(expected)
    assert prediction.provenance["prediction_lag_fallback_count"] == 1


def test_exogenous_vintages_are_selected_at_each_historical_decision():
    history = price_history()
    features = pd.DataFrame({"product_start": history.product_start,
        "available_at": history.product_start - pd.Timedelta(hours=26),
        "feature": "load", "value": np.arange(len(history), dtype=float)})
    future = features.copy()
    future["available_at"] = future.product_start + pd.Timedelta(hours=1)
    future["value"] = 1e9
    settings = PriceForecastSettings(min_training_rows=96, calibration_days=3, exogenous_features=("load",))
    base = CausalPriceForecaster.fit(history, cutoff="2026-01-01T00:00:00Z", settings=settings, exogenous=features)
    changed = CausalPriceForecaster.fit(history, cutoff="2026-01-01T00:00:00Z", settings=settings,
        exogenous=pd.concat([features, future], ignore_index=True))
    np.testing.assert_allclose(base.coefficient, changed.coefficient)
    targets = pd.DatetimeIndex([pd.Timestamp("2026-01-02T00:00:00Z")])
    prediction = changed.predict(targets, cutoff="2026-01-01T00:00:00Z", exogenous=pd.concat([features, future]))
    assert np.isfinite(prediction.point_eur_per_mwh).all()
    with pytest.raises(ValueError, match="Exogenous inputs"):
        changed.predict(targets, cutoff="2026-01-01T00:00:00Z")


def test_artifact_round_trip_and_tamper_detection(tmp_path):
    model = fitted()
    path = tmp_path / "price_model.json"
    model.save(path)
    restored = CausalPriceForecaster.load(path)
    assert restored.artifact_sha256 == model.artifact_sha256
    targets = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="15min")
    np.testing.assert_array_equal(model.predict(targets, cutoff=targets[0]).point_eur_per_mwh,
        restored.predict(targets, cutoff=targets[0]).point_eur_per_mwh)
    document = json.loads(path.read_text())
    document["payload"]["coefficient"][0] += 1
    path.write_text(json.dumps(document))
    with pytest.raises(ValueError, match="checksum"):
        CausalPriceForecaster.load(path)


def test_weighted_crps_matches_independent_pairwise_formula():
    actual = np.array([0.0, 5.0])
    ensemble = np.array([[1.0, 8.0], [3.0, 2.0], [-2.0, 7.0]])
    weights = np.array([0.2, 0.3, 0.5])
    pair = np.sum(weights[:, None, None] * weights[None, :, None]
        * np.abs(ensemble[:, None, :] - ensemble[None, :, :]), axis=(0, 1))
    expected = np.mean(weights @ np.abs(ensemble - actual) - 0.5 * pair)
    assert price_forecast_scores(actual, ensemble, weights)["crps_eur_per_mwh"] == pytest.approx(expected)


@pytest.mark.parametrize("name", ["product_start", "product_end", "available_at"])
def test_naive_history_clocks_rejected(name):
    frame = price_history().head(4).copy()
    frame[name] = frame[name].dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        validate_price_history(frame)


def test_catalog_adapter_preserves_snapshot_provenance_and_native_products(monkeypatch):
    import bess_pv_trading.data.api_catalog as catalog

    records = pd.DataFrame({"product_start": [pd.Timestamp("2026-01-02T12:00:00Z")],
        "product_end": [pd.Timestamp("2026-01-02T13:00:00Z")], "price_eur_mwh": [42.0]})
    evidence = [{"storage": {"snapshot_id": "pinned"}, "normalized_sha256": "checksum"}]
    monkeypatch.setattr(catalog, "load_api_cache_records", lambda *args, **kwargs: (records, evidence))
    frame, provenance = load_price_history("market.duckdb", provider="gme", dataset="day_ahead_prices",
        zone="NORD", publication_local_time="13:00", publication_day_offset=-1)
    assert len(frame) == 4
    assert (frame.price_eur_per_mwh == 42).all()
    assert (frame.available_at == pd.Timestamp("2026-01-01T12:00:00Z")).all()
    assert provenance["snapshot_sources"] == evidence
    assert provenance["availability"]["kind"].startswith("assumed")
    with pytest.raises(ValueError, match="Retrieval time"):
        load_price_history("market.duckdb", provider="gme", dataset="day_ahead_prices", zone="NORD",
                           availability_column="retrieved_at")


def test_catalog_missing_values_and_conflicting_products_are_never_zero_filled(monkeypatch):
    import bess_pv_trading.data.api_catalog as catalog

    records = pd.DataFrame({"product_start": pd.to_datetime(["2026-01-01T00:00:00Z"] * 2),
        "product_end": pd.to_datetime(["2026-01-01T00:15:00Z"] * 2), "price_eur_mwh": [1.0, 2.0]})
    monkeypatch.setattr(catalog, "load_api_cache_records", lambda *args, **kwargs: (records, []))
    with pytest.raises(ValueError, match="unique"):
        load_price_history("market.duckdb", provider="terna", dataset="imbalance_prices", zone="NORD",
                           publication_lag_hours=4)


def test_dst_residuals_require_matching_day_length():
    model = fitted()
    targets = pd.date_range("2026-03-29", "2026-03-30", freq="15min", inclusive="left", tz="Europe/Rome")
    # Supply ordinary causal lags near the DST day without manufacturing calibration days.
    history = price_history().copy()
    shift = pd.Timedelta(days=87)
    for name in ("product_start", "product_end", "available_at"):
        history[name] += shift
    with pytest.raises(ValueError, match="DST day length"):
        model.predict(targets, cutoff="2026-03-28T00:00:00Z", scenario_count=2, history=history)


@pytest.mark.parametrize("settings", [{"lag_hours": []}, {"ridge_alpha": -1},
    {"calibration_days": True}, {"method": "oracle"}, {"interval_minutes": 17}])
def test_invalid_settings_fail(settings):
    with pytest.raises(ValueError):
        PriceForecastSettings(**settings)


def test_prepared_history_matches_uncached_predictions_without_revalidation(monkeypatch):
    import bess_pv_trading.forecasting.energy_prices as module

    model = fitted()
    history = price_history()
    prepared = prepare_price_history(history)
    starts = pd.date_range("2026-01-01T00:00:00Z", periods=8, freq="15min")
    expected = model.predict(starts, cutoff=starts[0], history=history)
    monkeypatch.setattr(module, "validate_price_history", lambda *args: pytest.fail("History must be prepared once"))
    actual = model.predict(starts, cutoff=starts[0], history=prepared)
    np.testing.assert_array_equal(actual.point_eur_per_mwh, expected.point_eur_per_mwh)


def test_auction_gate_cutoff_is_shared_across_all_delivery_products():
    starts = pd.date_range("2026-01-02", periods=96, freq="15min", tz="Europe/Rome")
    settings = PriceForecastSettings(decision_local_time="11:45", decision_day_offset=-1)
    clocks = historical_decision_times(starts, settings)
    assert len(clocks.unique()) == 1
    assert clocks[0] == pd.Timestamp("2026-01-01T10:45:00Z")


def test_half_day_residual_paths_for_ida3():
    history = price_history()
    history = history.loc[history.product_start.dt.tz_convert("Europe/Rome").dt.hour >= 12]
    model = fitted(history, lag_hours=(24, 168), delivery_start_hour=12, delivery_end_hour=24,
                   decision_local_time="09:45", decision_day_offset=0)
    starts = pd.date_range("2026-01-01T12:00:00+01:00", periods=48, freq="15min")
    prediction = model.predict(starts, cutoff="2026-01-01T08:45:00Z", scenario_count=4)
    assert prediction.scenarios_eur_per_mwh.shape == (4, 48)
    assert all(len(block["starts"]) == 48 for block in model.residual_blocks)


def test_explicit_dst_clock_mapping_keeps_all_real_delivery_intervals():
    model = fitted(residual_dst_policy="same_clock_slots")
    targets = pd.date_range("2026-03-29", "2026-03-30", freq="15min", inclusive="left", tz="Europe/Rome")
    history = price_history().copy()
    for name in ("product_start", "product_end", "available_at"):
        history[name] += pd.Timedelta(days=87)
    prediction = model.predict(targets, cutoff="2026-03-28T00:00:00Z", scenario_count=2, history=history)
    assert prediction.scenarios_eur_per_mwh.shape == (2, 92)
    assert prediction.provenance["residual_dst_policy"] == "same_clock_slots"


def test_adding_a_backward_compatible_default_preserves_loaded_artifact_identity(tmp_path):
    from bess_pv_trading.forecasting.energy_prices import _digest

    model = fitted()
    path = tmp_path / "older_model.json"
    model.save(path)
    document = json.loads(path.read_text())
    document["payload"]["settings"].pop("residual_dst_policy")
    document["sha256"] = _digest(document["payload"])
    path.write_text(json.dumps(document))
    loaded = CausalPriceForecaster.load(path)
    assert loaded.artifact_sha256 == document["sha256"]
    assert loaded.settings.residual_dst_policy == "strict_day_length"


def test_half_day_price_lags_fall_back_to_target_clock_after_autumn_dst():
    model = fitted(daily_lag_fallback="target_clock")
    history = price_history().copy()
    for name in ("product_start", "product_end", "available_at"):
        history[name] += pd.Timedelta(days=298)
    history = history.loc[history.product_start.dt.tz_convert("Europe/Rome").dt.hour >= 12]
    targets = pd.date_range("2026-10-25T12:00:00+01:00", periods=48, freq="15min")
    prediction = model.predict(targets, cutoff="2026-10-25T08:45:00Z", history=history)
    assert np.isfinite(prediction.point_eur_per_mwh).all()
    assert prediction.provenance["prediction_lag_fallback_count"] > 0
