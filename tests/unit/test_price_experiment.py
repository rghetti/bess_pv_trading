"""Native-product recovery, DST lags and chronological retrospective evaluation."""
import json

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.price_experiment import (
    FEATURES, build_retrospective_features, evaluate_price_holdout,
    prepare_native_prices, run_price_experiment,
)


def _settings(**overrides):
    settings = {"timezone": "Europe/Rome", "min_training_days": 5, "holdout_days": 5,
                "alpha": 1.0, "native_product_regimes": [
                    {"start": "2025-01-01", "end": "2025-10-01", "minutes": 60},
                    {"start": "2025-10-01", "end": None, "minutes": 15}]}
    return {**settings, **overrides}


def _rows(start, end, frequency="15min", hourly_prices=False):
    stamps = pd.date_range(start, end, freq=frequency, inclusive="left", tz="Europe/Rome").tz_convert("UTC")
    clock = stamps.tz_convert("Europe/Rome")
    elapsed = np.arange(len(stamps), dtype=float)
    price = (60 + clock.hour.to_numpy() * 2 + clock.day.to_numpy() * .5
             + (0 if hourly_prices else elapsed % 4 * .1))
    frame = pd.DataFrame({"product_start": stamps, "product_end": stamps + pd.Timedelta(frequency),
                          "price_eur_mwh": price, "retrieved_at": pd.Timestamp("2026-10-16", tz="UTC"),
                          "historical_available_at": pd.NaT, "historical_feature_eligible": False,
                          "resolution": "PT15M" if frequency == "15min" else "PT60M"})
    return frame


def test_declared_hourly_market_recovers_equal_quarters_but_keeps_quarters_after_transition():
    before = _rows("2025-09-30", "2025-10-01", hourly_prices=True)
    after = _rows("2025-10-01", "2025-10-02", hourly_prices=True)
    # A repeated equal-value A03 block can span multiple native products.
    before["source_block_start"] = before.product_start.iloc[0]
    before["source_block_end"] = before.product_end.iloc[-1]
    native = prepare_native_prices(pd.concat([before, after]), _settings())
    assert len(native.loc[native.delivery_day == "2025-09-30"]) == 24
    assert len(native.loc[native.delivery_day == "2025-10-01"]) == 96
    assert set(native.native_interval_minutes) == {15, 60}
    assert native.loc[native.native_interval_minutes == 60, "provider_grid_rows"].eq(4).all()
    assert native.loc[native.native_interval_minutes == 15, "provider_grid_rows"].eq(1).all()
    assert native.historical_available_at.isna().all()
    assert not native.historical_feature_eligible.any()


def test_hourly_recovery_rejects_mixed_prices_or_incomplete_provider_quarters():
    rows = _rows("2025-09-30", "2025-10-01", hourly_prices=True)
    changed = rows.copy()
    changed.loc[1, "price_eur_mwh"] += 1
    with pytest.raises(ValueError, match="differ within"):
        prepare_native_prices(changed, _settings())
    with pytest.raises(ValueError, match="incomplete provider"):
        prepare_native_prices(rows.drop(index=1), _settings())
    with pytest.raises(ValueError, match="native_product_regimes"):
        prepare_native_prices(rows, _settings(native_product_regimes=[]))


def test_native_lags_never_splice_hourly_prices_into_quarter_hour_targets():
    before = _rows("2025-09-20", "2025-10-01", hourly_prices=True)
    after = _rows("2025-10-01", "2025-10-10")
    features = build_retrospective_features(prepare_native_prices(pd.concat([before, after]), _settings()), "Europe/Rome")
    first = features.loc[features.delivery_day == "2025-10-01"]
    eighth = features.loc[features.delivery_day == "2025-10-08"]
    assert first.lag_1d.isna().all() and first.lag_7d.isna().all()
    assert eighth.lag_1d.notna().all() and eighth.lag_7d.notna().all()


def test_native_lags_keep_repeated_dst_hour_folds_independent():
    rows = _rows("2025-10-18", "2025-10-27")
    features = build_retrospective_features(prepare_native_prices(rows, _settings()), "Europe/Rome")
    repeated = features.loc[(features.delivery_day == "2025-10-26") & (features.local_hour == 2)]
    assert len(repeated) == 2
    first, second = repeated.sort_values("product_start").itertuples(index=False)
    assert first.local_fold == 0 and second.local_fold == 1
    assert np.isfinite(first.lag_1d) and np.isnan(second.lag_1d)
    assert np.isfinite(first.lag_7d) and np.isnan(second.lag_7d)


def test_holdout_labels_do_not_fit_model_and_scaler_uses_training_only():
    settings = _settings()
    rows = _rows("2025-10-01", "2025-11-01")
    native = prepare_native_prices(rows, settings)
    features = build_retrospective_features(native, "Europe/Rome")
    predictions, evaluations, models = evaluate_price_holdout(features, settings)
    altered = rows.copy()
    altered.loc[altered.index[-1], "price_eur_mwh"] += 10000
    changed_features = build_retrospective_features(prepare_native_prices(altered, settings), "Europe/Rome")
    changed, _, changed_models = evaluate_price_holdout(changed_features, settings)
    assert np.allclose(predictions.ridge_prediction, changed.ridge_prediction)
    assert models == changed_models
    training = features.loc[features.delivery_day < "2025-10-27"].dropna(subset=list(FEATURES))
    assert np.allclose(models["15min"]["center"], training[list(FEATURES)].mean().to_numpy())
    evaluation = evaluations[0]
    assert evaluation["holdout_start_day"] == "2025-10-27"
    assert evaluation["holdout_end_day"] == "2025-10-31"
    assert evaluation["holdout_days_observed"] == 5
    assert {scores["rows"] for scores in evaluation["models"].values()} == {len(predictions)}


def test_insufficient_history_is_reported_without_fitting():
    features = build_retrospective_features(prepare_native_prices(_rows("2025-10-01", "2025-10-12"), _settings()), "Europe/Rome")
    predictions, evaluations, models = evaluate_price_holdout(features, _settings(min_training_days=30))
    assert predictions.empty and not models
    assert evaluations[0]["status"] == "insufficient_data"


def test_run_saves_reusable_months_provenance_metrics_and_native_dst_coverage(tmp_path, monkeypatch):
    from bess_pv_trading.data import api_catalog
    rows = _rows("2025-10-01", "2025-11-01")
    source = [{"manifest": "fixture/manifest.json", "normalized_sha256": "fixture", "checksums_verified": True}]
    monkeypatch.setattr(api_catalog, "load_api_cache_records", lambda *args, **kwargs: (rows, source))
    settings = _settings(analysis_directory=str(tmp_path / "analysis"), output_directory=str(tmp_path / "experiments"))
    result = run_price_experiment({"price_experiment": settings})
    assert result["complete_days"] == 31 and result["incomplete_days"] == 0
    assert result["native_products"] == 31 * 96 + 4
    assert result["historical_publication_verified"] is False
    run = __import__("pathlib").Path(result["run_directory"])
    assert (run / "price_history.png").is_file()
    assert (run / "holdout_15min.png").is_file()
    assert json.loads((run / "source_caches.json").read_text()) == source
    monthly = tmp_path / "analysis" / "entsoe" / "day_ahead_prices" / "NORD" / "year=2025" / "month=10"
    derived = pd.read_parquet(monthly / "prices_with_features.parquet")
    assert len(derived) == len(rows)
    assert derived.historical_available_at.isna().all()
    assert (monthly / "daily_quality.csv").is_file()
    assert pd.read_parquet(run / "holdout_predictions.parquet").price_eur_mwh.notna().all()
    assert (run / "prices_with_features.parquet").is_file()
    assert json.loads((monthly / "manifest.json").read_text())["source_caches"] == str((run / "source_caches.json").resolve())


def test_date_filters_use_local_midnights_and_whole_missing_days_are_reported(tmp_path, monkeypatch):
    from bess_pv_trading.data import api_catalog
    rows = _rows("2025-10-01", "2025-10-05")
    rows = rows.loc[rows.product_start.dt.tz_convert("Europe/Rome").dt.day != 3]
    captured = {}

    def load(*args, **kwargs):
        captured.update(kwargs)
        return rows, []

    monkeypatch.setattr(api_catalog, "load_api_cache_records", load)
    settings = _settings(start="2025-10-01", end="2025-10-05", analysis_directory=str(tmp_path / "analysis"),
                         output_directory=str(tmp_path / "experiments"))
    result = run_price_experiment({"price_experiment": settings})
    assert captured["start"] == "2025-09-30T22:00:00+00:00"
    assert captured["end"] == "2025-10-04T22:00:00+00:00"
    assert result["complete_days"] == 3 and result["incomplete_days"] == 1
