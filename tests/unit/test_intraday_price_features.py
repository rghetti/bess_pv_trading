"""Synthetic native-price fixtures for conditional intraday study groundwork."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.intraday_price_features import (
    POLICY, ConditionalSpreadPriceForecast, build_future_intraday_price_features,
    build_intraday_price_features, fit_intraday_ridge, load_intraday_price_records)
from bess_pv_trading.forecasting.prices import RidgePriceModel


def _records(auction, start="2025-01-01", end="2025-01-12", minutes=15):
    """Make explicitly synthetic GME-like rows, preserving native source units."""
    starts = pd.date_range(pd.Timestamp(start, tz="Europe/Rome").tz_convert("UTC"),
        pd.Timestamp(end, tz="Europe/Rome").tz_convert("UTC"), freq=f"{minutes}min", inclusive="left")
    local = starts.tz_convert("Europe/Rome")
    days = np.array([(stamp.date() - local[0].date()).days for stamp in local])
    period = pd.Series(local.date).groupby(pd.Series(local.date)).cumcount().to_numpy() + 1
    result = pd.DataFrame({"provider": "synthetic_gme_fixture", "zone": "NORD", "auction": auction,
        "product_start": starts, "product_end": starts + pd.Timedelta(minutes=minutes),
        "price_eur_mwh": 80 + 2 * days + .5 * local.hour + .1 * local.minute
            + {"DA": 0, "IDA1": 3, "IDA2": -2, "IDA3": 5}[auction],
        "native_interval_minutes": minutes, "source_period": period, "unit": "EUR/MWh",
        "retrieved_at": pd.Timestamp("2026-10-16T00:00Z"), "published_at": pd.NaT,
        "historical_available_at": pd.NaT, "historical_feature_eligible": False,
        "record_status": "observed", "revision": "synthetic_latest_at_retrieval"})
    return result.loc[local.hour >= 12].reset_index(drop=True) if auction == "IDA3" else result


def _sources(start="2025-01-01", end="2025-01-12", da_minutes=60):
    """Provide all auction labels so later-auction exclusion can be tested."""
    return {name: _records(name, start, end, da_minutes if name == "DA" else 15)
            for name in ("DA", "IDA1", "IDA2", "IDA3")}


def _model(auction, features=("previous_auction_price",), coefficients=(0.,), intercept=2.):
    """Create a transparent linear equation for callback integration checks."""
    ridge = RidgePriceModel(tuple(features), np.zeros(len(features)), np.ones(len(features)),
        np.array(coefficients, float), intercept, 0., "2024-12-01T00:00Z", 100,
        f"synthetic_{auction}_spread")
    return ConditionalSpreadPriceForecast(auction, ridge, "2024-11-30")


def test_hourly_da_becomes_only_an_audited_feature_for_native_ida_quarters():
    sources = _sources()
    original = sources["DA"].copy(deep=True)
    rows, names = build_intraday_price_features(sources, "IDA1")
    selected = rows.loc[rows.delivery_day.eq("2025-01-09")].head(4)
    assert len(rows) == len(sources["IDA1"])
    assert rows.native_interval_minutes.eq(15).all()
    assert selected.da_price.nunique() == 1
    assert selected.da_source_native_interval_minutes.eq(60).all()
    assert selected.da_feature_mapping.eq("derived_containing_hourly_DA_price_feature_only").all()
    assert ((selected.da_source_product_end - selected.da_source_product_start).dt.total_seconds() == 3600).all()
    np.testing.assert_allclose(selected.target, selected.actual - selected.previous_auction_price)
    assert "previous_auction_price" in names and "da_price" not in names  # Avoid duplicate ridge columns.
    assert rows.published_at.isna().all() and not rows.historical_feature_eligible.any()
    assert rows.availability_policy.eq(POLICY).all() and not rows.operational_eligible.any()
    pd.testing.assert_frame_equal(sources["DA"], original)


def test_current_and_later_auction_labels_cannot_enter_earlier_auction_features():
    sources = _sources()
    first, names = build_intraday_price_features(sources, "IDA2")
    changed = {name: rows.copy() for name, rows in sources.items()}
    changed["IDA3"]["price_eur_mwh"] = 999999.
    second, other_names = build_intraday_price_features(changed, "IDA2")
    pd.testing.assert_frame_equal(first, second)
    assert names == other_names and "ida3_price" not in first
    assert "da_price" in names and "previous_auction_price" in names
    np.testing.assert_allclose(first.target, sources["IDA2"].price_eur_mwh - sources["IDA1"].price_eur_mwh)
    assert (first.ida1_assumed_or_recorded_available_at <= first.forecast_cutoff).all()


def test_delayed_previous_result_is_missing_even_if_study_default_is_earlier():
    sources = _sources(start="2025-01-09", end="2025-01-10")
    sources["DA"]["published_at"] = pd.Timestamp("2025-01-08T15:01:00+01:00")
    rows, _ = build_intraday_price_features(sources, "IDA1")
    assert rows.previous_auction_price.isna().all() and rows.target.isna().all()
    assert rows.da_feature_mapping.eq("missing_or_unavailable").all()
    assert sources["DA"].published_at.notna().all()
    overridden, _ = build_intraday_price_features(_sources(), "IDA1", {
        "result_overrides": {"DA": {"2025-01-09": None}}})
    assert overridden.loc[overridden.delivery_day.eq("2025-01-09"), "previous_auction_price"].isna().all()


def test_own_same_clock_lags_preserve_dst_fold_and_product_duration():
    sources = _sources("2025-10-18", "2025-10-27", da_minutes=15)
    rows, _ = build_intraday_price_features(sources, "IDA1")
    local = rows.product_start.dt.tz_convert("Europe/Rome")
    repeated = rows.loc[rows.delivery_day.eq("2025-10-26") & local.dt.hour.eq(2)]
    fold = np.array([stamp.fold for stamp in repeated.product_start.dt.tz_convert("Europe/Rome")])
    assert repeated.loc[fold == 0, "lag_1d"].notna().all()
    assert repeated.loc[fold == 1, "lag_1d"].isna().all()
    history = _records("IDA1", "2025-01-08", "2025-01-09", minutes=60)
    products = _records("IDA1", "2025-01-09", "2025-01-10")[["product_start", "product_end"]]
    future, _ = build_future_intraday_price_features(history, products, "IDA1")
    assert future.lag_1d.isna().all()  # Hourly own labels cannot become QH lag labels.


@pytest.mark.parametrize("day,following,first_period", [
    ("2025-03-30", "2025-03-31", 45), ("2025-10-26", "2025-10-27", 53)])
def test_ida3_has_only_48_afternoon_labels_even_on_dst_days(day, following, first_period):
    sources = _sources(day, following, da_minutes=60 if day < "2025-10-01" else 15)
    rows, _ = build_intraday_price_features(sources, "IDA3")
    assert len(rows) == 48 and rows.source_period.min() == first_period
    assert rows.product_start.dt.tz_convert("Europe/Rome").dt.hour.ge(12).all()
    assert rows.previous_auction_price.notna().all()
    sources["IDA3"] = sources["IDA2"].copy().assign(auction="IDA3")
    with pytest.raises(ValueError, match="morning"):
        build_intraday_price_features(sources, "IDA3")


def test_future_builder_has_no_own_target_labels_and_rejects_future_history():
    history = _records("IDA1", "2025-01-01", "2025-01-09")
    products = _records("IDA1", "2025-01-09", "2025-01-10")[["product_start", "product_end"]]
    da = _records("DA", "2025-01-09", "2025-01-10", minutes=60)
    future, names = build_future_intraday_price_features(history, products, "IDA1", {"DA": da})
    assert not {"actual", "target", "price_eur_mwh"} & set(future)
    assert future[names].notna().all().all()
    assert future.static_feature_available_at.le(future.forecast_cutoff).all()
    assert not future.operational_eligible.any()
    with pytest.raises(ValueError, match="earlier delivery"):
        build_future_intraday_price_features(pd.concat([history, _records("IDA1", "2025-01-09", "2025-01-10")]), products, "IDA1")
    with pytest.raises(ValueError, match="target price"):
        build_future_intraday_price_features(history, products.assign(actual=100.), "IDA1")
    with pytest.raises(ValueError, match="later auction"):
        build_future_intraday_price_features(history, products, "IDA1", {"IDA2": _records("IDA2")})


def test_zone_disagreements_are_rejected_in_historical_future_and_callback_inputs():
    sources = _sources()
    sources["DA"]["zone"] = "CNOR"
    with pytest.raises(ValueError, match="different zones"):
        build_intraday_price_features(sources, "IDA1")
    history = _records("IDA1", "2025-01-01", "2025-01-09")
    products = _records("IDA1", "2025-01-09", "2025-01-10")[["product_start", "product_end"]]
    da = _records("DA", "2025-01-09", "2025-01-10", 60).assign(zone="CNOR")
    with pytest.raises(ValueError, match="different zones"):
        build_future_intraday_price_features(history, products, "IDA1", {"DA": da})
    frame, _ = build_future_intraday_price_features(history, products, "IDA1")
    with pytest.raises(ValueError, match="different zones"):
        _model("IDA1").predict_absolute(frame, {"DA": da})


def test_negative_labels_and_missing_source_products_are_never_zero_filled():
    sources = _sources("2025-01-09", "2025-01-10")
    sources["IDA1"].loc[0, "price_eur_mwh"] = -50.
    sources["IDA1"].loc[1, "price_eur_mwh"] = np.nan
    sources["DA"] = sources["DA"].iloc[1:].reset_index(drop=True)
    rows, _ = build_intraday_price_features(sources, "IDA1")
    assert rows.actual.iloc[0] == -50 and np.isnan(rows.actual.iloc[1])
    assert rows.previous_auction_price.iloc[:4].isna().all()
    assert rows.target.iloc[:4].isna().all()


def test_branch_history_replaces_stored_prices_and_retains_linear_spread_dependence():
    frame, _ = build_intraday_price_features(_sources(), "IDA2")
    frame = frame.loc[frame.delivery_day.eq("2025-01-09")].copy()
    model = _model("IDA2", ("previous_auction_price", "da_price"), (.5, .2), 1.)
    da, ida1 = np.full(96, 50.), np.full(96, 80.)
    da.flags.writeable = ida1.flags.writeable = False
    predicted = model.predict_absolute(frame, {"DA": da, "IDA1": ida1})
    np.testing.assert_allclose(predicted, 131.)  # 80 + .5*80 + .2*50 + 1.
    frame["actual"], frame["previous_auction_price"], frame["da_price"] = 999999., 888888., 777777.
    np.testing.assert_allclose(model.predict_absolute(frame, {"DA": da, "IDA1": ida1}), predicted)
    np.testing.assert_allclose(model.predict_absolute(frame, {"DA": da, "IDA1": ida1 + 20}), predicted + 30)
    assert not da.flags.writeable and np.equal(ida1, 80).all()
    with pytest.raises(ValueError, match="later auction"):
        model.predict_absolute(frame, {"DA": da, "IDA1": ida1, "IDA3": ida1})
    with pytest.raises(ValueError, match="target or later"):
        _model("IDA2", ("actual",), (1.,))


@pytest.mark.parametrize("day,following,total,offset", [
    ("2025-01-09", "2025-01-10", 96, 48), ("2025-03-30", "2025-03-31", 92, 44),
    ("2025-10-26", "2025-10-27", 100, 52)])
def test_ida3_wrapper_maps_full_native_ancestor_vectors_to_eligible_afternoon(day, following, total, offset):
    products = _records("IDA3", day, following)[["product_start", "product_end"]]
    frame, _ = build_future_intraday_price_features(pd.DataFrame(), products, "IDA3")
    model = _model("IDA3")
    ancestor = np.arange(total, dtype=float)
    np.testing.assert_allclose(model.predict_absolute(frame, {"IDA2": ancestor}), ancestor[offset:] + 2.)
    # Partial residual grids require an explicit positional mapping.
    frame["scenario_grid_index"] = np.arange(8, 56)
    residual = np.arange(56, dtype=float)
    np.testing.assert_allclose(model.predict_absolute(frame, {"IDA2": residual}), residual[8:] + 2.)


def test_native_dataframe_ancestor_mapping_and_missing_curve_are_explicit():
    products = _records("IDA1", "2025-01-09", "2025-01-10")[["product_start", "product_end"]]
    frame, _ = build_future_intraday_price_features(pd.DataFrame(), products, "IDA1")
    da = _records("DA", "2025-01-09", "2025-01-10", minutes=60)
    model = _model("IDA1")
    values = model.predict_absolute(frame, {"DA": da})
    np.testing.assert_allclose(values, np.repeat(da.price_eur_mwh.to_numpy(), 4) + 2.)
    with pytest.raises(ValueError, match="Missing native DA"):
        model.predict_absolute(frame, {})
    with pytest.raises(ValueError, match="ambiguous"):
        model.predict_absolute(frame, {"DA": np.arange(24.)})


def test_fit_is_strictly_earlier_and_assumed_availability_requires_explicit_opt_in(tmp_path):
    sources = _sources("2025-10-01", "2025-10-17", da_minutes=15)
    sources["IDA1"]["price_eur_mwh"] = 1.5 * sources["DA"].price_eur_mwh + 2.
    rows, _ = build_intraday_price_features(sources, "IDA1")
    arguments = {"fit_cutoff": "2025-10-15T12:55Z", "forecast_delivery_day": "2025-10-16",
        "feature_names": ["previous_auction_price"], "alpha": 0.}
    with pytest.raises(ValueError, match="allow_assumed"):
        fit_intraday_ridge(rows, "IDA1", **arguments)
    model = fit_intraday_ridge(rows, "IDA1", allow_assumed_availability=True, **arguments)
    assert model.training_max_delivery_day < "2025-10-16" and not model.operational_eligible
    assert model.model.coefficients[0] / model.model.scale[0] == pytest.approx(.5)
    changed = rows.copy()
    changed.loc[changed.delivery_day.eq("2025-10-16"), "target"] = 999999.
    other = fit_intraday_ridge(changed, "IDA1", allow_assumed_availability=True, **arguments)
    assert model.to_dict() == other.to_dict()
    model.save(tmp_path / "model.json")
    restored = ConditionalSpreadPriceForecast.load(tmp_path / "model.json")
    np.testing.assert_allclose(model.predict(rows.dropna(subset=["previous_auction_price"])),
                               restored.predict(rows.dropna(subset=["previous_auction_price"])))
    changed["assumed_target_available_at"] = pd.Timestamp("2027-01-01T00:00Z")
    with pytest.raises(ValueError, match="No strictly earlier"):
        fit_intraday_ridge(changed, "IDA1", allow_assumed_availability=True, **arguments)


def test_duckdb_catalog_reader_pins_sources_and_rejects_legacy_directories(monkeypatch):
    from bess_pv_trading.data import api_catalog
    captured = []
    sources = _sources("2025-01-01", "2025-01-02")
    def read(path, **kwargs):
        """Record the catalog contract without opening a real database."""
        captured.append({"path": path, **kwargs})
        if kwargs["provider"] == "gme":
            records = pd.concat([sources[name] for name in ("IDA1", "IDA2", "IDA3")], ignore_index=True)
        else:
            records = _records("DA", "2025-01-01", "2025-01-02")
            hourly = sources["DA"].set_index("product_start").price_eur_mwh
            records["value"] = records.product_start.dt.floor("h").map(hourly)
            records = records.drop(columns=["price_eur_mwh", "native_interval_minutes"])
        return records, [{"storage": {"backend": "duckdb", "snapshot_id": kwargs["provider"]}, "checksums_verified": True}]
    monkeypatch.setattr(api_catalog, "load_api_cache_records", read)
    records, provenance = load_intraday_price_records("fixture/market.duckdb")
    assert all(call["path"] == Path("fixture/market.duckdb") and call["verify_checksums"] for call in captured)
    assert [call["provider"] for call in captured] == ["gme", "entsoe"]
    assert captured[1]["dataset"] == "day_ahead_prices"
    assert set(records) == {"DA", "IDA1", "IDA2", "IDA3"} and len(provenance) == 2
    assert len(records["DA"]) == 24 and records["DA"].native_interval_minutes.eq(60).all()
    assert records["DA"].published_at.isna().all() and not records["DA"].historical_feature_eligible.any()
    with pytest.raises(ValueError, match="DuckDB"):
        load_intraday_price_records("data/api")
