"""Past-only linear feature groups, complete daily history, and DST alignment."""

from datetime import date
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.price_features import build_future_linear_price_features, build_linear_price_features


def _native(start="2025-01-01", end="2025-02-10", minutes=60):
    """Create complete native products with distinguishable day/clock prices."""
    starts = pd.date_range(start, end, inclusive="left", freq=f"{minutes}min", tz="Europe/Rome").tz_convert("UTC")
    local = starts.tz_convert("Europe/Rome")
    ordinal = np.array([stamp.date().toordinal() - date(2025, 1, 1).toordinal() for stamp in local])
    values = ordinal * 10 + local.hour.to_numpy() + local.minute.to_numpy() / 60
    return pd.DataFrame({"product_start": starts, "product_end": starts + pd.Timedelta(minutes=minutes),
                         "price_eur_mwh": values, "native_interval_minutes": minutes,
                         "historical_available_at": pd.NaT, "historical_feature_eligible": False})


def test_expanded_features_keep_strict_clock_lags_and_past_daily_summaries():
    rows = _native()
    frame, groups = build_linear_price_features(rows)
    target = frame.loc[(frame.delivery_day == "2025-02-05") & (frame.local_hour == 12)].iloc[0]
    assert target.lag_2d == 330 + 12
    assert target.lag_3d == 320 + 12
    assert target.lag_14d == 210 + 12
    assert target.daily_lag_1d_mean == pytest.approx(340 + 11.5)
    assert target.daily_lag_1d_min == 340
    assert target.daily_lag_1d_max == 363
    assert target.daily_lag_1d_spread == 23
    assert target.rolling_7d_mean == pytest.approx(np.arange(28, 35).mean() * 10 + 11.5)
    assert target.rolling_7d_mean_std == pytest.approx(np.arange(28, 35).std() * 10)
    assert target.rolling_7d_min == 280
    assert target.rolling_7d_max == 363
    assert target.rolling_28d_mean == pytest.approx(np.arange(7, 35).mean() * 10 + 11.5)
    assert target.profile_lag_1d_last == 363
    assert target.profile_lag_1d_block_00_06_mean == pytest.approx(340 + 2.5)
    assert target.profile_lag_2d_block_18_24_mean == pytest.approx(330 + 20.5)
    assert groups["autoregressive"] == ["lag_1d", "lag_7d"]
    assert groups["autoregressive_extended"] == ["lag_1d", "lag_2d", "lag_3d", "lag_7d", "lag_14d"]
    assert len(groups["hour_categories"]) == 23 and len(groups["weekday_categories"]) == 6
    assert len(groups["rolling_history_7d"]) == 5 and len(groups["rolling_history_28d"]) == 5
    assert len(groups["daily_profile_lags"]) == 20
    assert groups["fundamentals"] == []
    assert frame.historical_available_at.isna().all()
    assert not frame.historical_feature_eligible.any()


def test_same_day_or_future_labels_cannot_change_earlier_features():
    rows = _native()
    original, groups = build_linear_price_features(rows)
    altered = rows.copy()
    local_days = altered.product_start.dt.tz_convert("Europe/Rome").dt.date
    altered.loc[local_days >= date(2025, 2, 5), "price_eur_mwh"] += 10000
    changed, _ = build_linear_price_features(altered)
    names = list(dict.fromkeys(name for values in groups.values() for name in values))
    target = original.delivery_day <= "2025-02-05"
    pd.testing.assert_frame_equal(original.loc[target, names], changed.loc[target, names])


def test_incomplete_or_absent_prior_day_prevents_rolling_window_skipping_gap():
    rows = _native()
    local = rows.product_start.dt.tz_convert("Europe/Rome")
    # Drop one product on Jan 30 and the entirety of Jan 29.
    keep = ~(local.dt.date.eq(date(2025, 1, 29)) | (local.dt.date.eq(date(2025, 1, 30)) & local.dt.hour.eq(12)))
    frame, _ = build_linear_price_features(rows.loc[keep])
    jan31 = frame.loc[frame.delivery_day == "2025-01-31"]
    assert jan31.daily_lag_1d_mean.isna().all()
    assert jan31.rolling_7d_mean.isna().all()
    assert jan31.profile_lag_1d_last.isna().all()
    assert frame.loc[frame.delivery_day == "2025-02-05", "rolling_7d_mean"].isna().all()
    assert frame.loc[frame.delivery_day == "2025-02-07", "rolling_7d_mean"].notna().all()


@pytest.mark.parametrize("start,end,transition,expected_products", [
    ("2025-03-20", "2025-04-03", "2025-03-30", 23),
    ("2025-10-18", "2025-10-29", "2025-10-26", 25),
])
def test_dst_days_have_complete_summaries_without_folding_repeated_price_slots(start, end, transition, expected_products):
    rows = _native(start, end)
    # Match nanosecond/ZoneInfo clocks restored from the workbook staging cache.
    for column in ("product_start", "product_end"):
        rows[column] = rows[column].dt.tz_convert(ZoneInfo("UTC")).dt.as_unit("ns")
    frame, _ = build_linear_price_features(rows, settings={"rolling_windows": [7]})
    change_day = frame.loc[frame.delivery_day == transition]
    assert len(change_day) == expected_products
    following = (pd.Timestamp(transition) + pd.Timedelta(days=1)).date().isoformat()
    assert frame.loc[frame.delivery_day == following, "daily_lag_1d_mean"].eq(change_day.price_eur_mwh.mean()).all()
    assert frame.loc[frame.delivery_day == following, "profile_lag_1d_last"].eq(change_day.price_eur_mwh.iloc[-1]).all()
    if expected_products == 25:
        repeated = change_day.loc[change_day.local_hour == 2].sort_values("product_start")
        assert repeated.local_fold.tolist() == [0, 1]
        assert np.isfinite(repeated.lag_2d.iloc[0]) and np.isnan(repeated.lag_2d.iloc[1])


def test_native_duration_regime_does_not_supply_previous_hourly_daily_history_to_quarters():
    rows = pd.concat([_native("2025-09-01", "2025-10-01", 60), _native("2025-10-01", "2025-10-12", 15)], ignore_index=True)
    frame, _ = build_linear_price_features(rows, settings={"rolling_windows": [7]})
    first_quarters = frame.loc[frame.delivery_day == "2025-10-01"]
    assert first_quarters.lag_1d.isna().all()
    assert first_quarters.daily_lag_1d_mean.isna().all()
    assert first_quarters.daily_lag_7d_mean.isna().all()
    assert first_quarters.rolling_7d_mean.isna().all()
    assert frame.loc[frame.delivery_day == "2025-10-08", "rolling_7d_mean"].notna().all()


def test_forecast_derivatives_preserve_missing_values_and_require_positive_load_for_shares():
    rows = _native("2025-01-01", "2025-01-02")
    rows["load_forecast_mw"] = 1000.
    rows["solar_forecast_mw"] = 100.
    rows["wind_forecast_mw"] = 50.
    rows.loc[0, "load_forecast_mw"] = 0
    rows.loc[1, "solar_forecast_mw"] = np.nan
    # Realized columns are preserved for provenance but excluded from all groups.
    rows["actual_load_mw"] = 9999.
    frame, groups = build_linear_price_features(rows)
    assert frame.residual_load_forecast_mw.iloc[2] == 850
    assert frame.solar_forecast_share.iloc[2] == .1
    assert frame.wind_forecast_share.iloc[2] == .05
    assert np.isnan(frame.solar_forecast_share.iloc[0])
    assert np.isnan(frame.residual_load_forecast_mw.iloc[1])
    assert "actual_load_mw" not in groups["fundamentals"]
    assert set(groups["fundamentals"]) == {"load_forecast_mw", "solar_forecast_mw", "wind_forecast_mw",
                                          "residual_load_forecast_mw", "solar_forecast_share", "wind_forecast_share"}
    assert groups["fundamental_components"] == ["load_forecast_mw", "solar_forecast_mw", "wind_forecast_mw"]
    assert groups["residual_load"] == ["residual_load_forecast_mw"]
    assert groups["renewable_shares"] == ["solar_forecast_share", "wind_forecast_share"]


def test_configured_holidays_are_deterministic_and_do_not_appear_without_calendar():
    rows = _native("2025-01-01", "2025-01-08")
    frame, groups = build_linear_price_features(rows, settings={"holiday_dates": ["2025-01-06"]})
    assert frame.loc[frame.delivery_day == "2025-01-06", "is_configured_holiday"].eq(1).all()
    assert frame.loc[frame.delivery_day == "2025-01-05", "day_before_configured_holiday"].eq(1).all()
    assert frame.loc[frame.delivery_day == "2025-01-07", "day_after_configured_holiday"].eq(1).all()
    assert "is_configured_holiday" in groups["calendar_extended"]
    _, groups_without_calendar = build_linear_price_features(rows)
    assert "is_configured_holiday" not in groups_without_calendar["calendar_extended"]


@pytest.mark.parametrize("settings", [{"lag_days": [1.5]}, {"lag_days": [0]}, {"rolling_windows": [7, 7]},
                                      {"daily_summary_lags": [True]}, {"holiday_dates": ["tomorrow"]},
                                      {"fundamental_columns": ["actual", "actual"]}, {"profile_block_hours": 5}])
def test_invalid_feature_configuration_is_rejected(settings):
    with pytest.raises(ValueError):
        build_linear_price_features(_native("2025-01-01", "2025-01-02"), settings=settings)


def test_unordered_native_rows_are_sorted_but_overlap_naive_timestamps_and_bad_durations_rejected():
    rows = _native("2025-01-01", "2025-01-03")
    frame, _ = build_linear_price_features(rows.sample(frac=1, random_state=9))
    assert frame.product_start.is_monotonic_increasing
    duplicate = pd.concat([rows, rows.iloc[:1]], ignore_index=True)
    with pytest.raises(ValueError, match="overlap"):
        build_linear_price_features(duplicate)
    naive = rows.copy()
    naive["product_start"] = naive.product_start.dt.tz_localize(None)
    with pytest.raises(ValueError, match="timezone-aware"):
        build_linear_price_features(naive)
    bad = rows.copy()
    bad["native_interval_minutes"] = 15
    with pytest.raises(ValueError, match="duration differs"):
        build_linear_price_features(bad)


@pytest.mark.parametrize("start,end,target,minutes", [
    ("2025-01-01", "2025-02-06", "2025-02-05", 60),
    ("2025-03-01", "2025-03-31", "2025-03-30", 60),
    ("2025-10-01", "2025-10-27", "2025-10-26", 15),
])
def test_future_features_need_no_target_and_match_retrospective_formulas_across_dst(start, end, target, minutes):
    native = _native(start, end, minutes)
    for column in ("product_start", "product_end"):
        native[column] = native[column].dt.tz_convert(ZoneInfo("UTC")).dt.as_unit("ns")
    native["load_forecast_mw"] = 1000 + np.arange(len(native)) / 10
    native["solar_forecast_mw"] = 100.
    native["wind_forecast_mw"] = 50.
    days = native.product_start.dt.tz_convert("Europe/Rome").dt.strftime("%Y-%m-%d")
    history = native.loc[days < target]
    unknown = native.loc[days == target].drop(columns=["price_eur_mwh", "native_interval_minutes"])
    retrospective, expected_groups = build_linear_price_features(native)
    expected = retrospective.loc[retrospective.delivery_day == target].reset_index(drop=True)
    actual, groups = build_future_linear_price_features(history, unknown)
    names = list(dict.fromkeys(name for values in groups.values() for name in values))
    pd.testing.assert_frame_equal(actual[names], expected[names])
    assert groups == expected_groups
    assert "price_eur_mwh" not in actual
    assert actual.native_interval_minutes.eq(minutes).all()
    assert actual.historical_available_at.isna().all()
    assert not actual.historical_feature_eligible.any()
    # Extreme target values also leave future features unchanged, confirming
    # that the retrospective builder never reads its target-day sentinel.
    perturbed = native.copy()
    perturbed.loc[days == target, "price_eur_mwh"] += 1e9
    perturbed_features, _ = build_linear_price_features(perturbed)
    perturbed_expected = perturbed_features.loc[perturbed_features.delivery_day == target].reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[names], perturbed_expected[names])


def test_future_quarter_hour_regime_preserves_missing_hourly_lags_and_accepts_unknown_target_column():
    history = _native("2025-09-01", "2025-10-01", 60)
    future = _native("2025-10-01", "2025-10-02", 15)
    future["price_eur_mwh"] = np.nan
    actual, _ = build_future_linear_price_features(history, future)
    assert len(actual) == 96 and "price_eur_mwh" not in actual
    assert actual.lag_1d.isna().all() and actual.lag_7d.isna().all()
    assert actual.daily_lag_1d_mean.isna().all() and actual.rolling_7d_mean.isna().all()
    assert actual.hour_sin.notna().all()


@pytest.mark.parametrize("field", ["price_eur_mwh", "price", "actual", "actual_price_eur_mwh", "realized_price_eur_mwh", "target", "value"])
def test_future_realized_target_fields_are_rejected(field):
    history = _native("2025-01-01", "2025-01-04")
    future = _native("2025-01-04", "2025-01-05").drop(columns="price_eur_mwh")
    future[field] = 99.
    with pytest.raises(ValueError, match="Future realized prices or actual targets"):
        build_future_linear_price_features(history, future)


def test_future_rows_do_not_inherit_empty_actual_target_aliases_from_historical_sources():
    history = _native("2025-01-01", "2025-01-04")
    history["actual"] = history.price_eur_mwh
    history["value"] = history.price_eur_mwh
    future = _native("2025-01-04", "2025-01-05").drop(columns="price_eur_mwh")
    frame, _ = build_future_linear_price_features(history, future)
    assert not {"actual", "value", "price_eur_mwh"}.intersection(frame.columns)


@pytest.mark.parametrize("case", ["missing_product", "duplicate_product", "two_days", "wrong_duration", "naive_clock", "history_overlap"])
def test_future_grid_and_history_boundaries_are_enforced(case):
    history = _native("2025-01-01", "2025-01-04")
    future = _native("2025-01-04", "2025-01-05").drop(columns="price_eur_mwh")
    if case == "missing_product":
        future = future.iloc[1:]
    elif case == "duplicate_product":
        future = pd.concat([future, future.iloc[:1]], ignore_index=True)
    elif case == "two_days":
        future = _native("2025-01-04", "2025-01-06").drop(columns="price_eur_mwh")
    elif case == "wrong_duration":
        future["native_interval_minutes"] = 15
    elif case == "naive_clock":
        future["product_start"] = future.product_start.dt.tz_localize(None)
    elif case == "history_overlap":
        history = _native("2025-01-01", "2025-01-05")
    with pytest.raises(ValueError):
        build_future_linear_price_features(history, future)
