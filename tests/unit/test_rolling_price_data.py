"""Check rolling price input alignment across publication and clock changes."""

import numpy as np
import pandas as pd
import json

from bess_pv_trading.forecasting.studies.rolling_price_data import (
    PriceIndex,
    WeatherIndex,
    delivery_grid,
    pad_curve,
    slot_numbers,
    build_examples,
)
from bess_pv_trading.forecasting.studies.price_feature_data import digest


def test_delivery_slots_preserve_both_autumn_hours_and_spring_gap():
    """Represent actual local delivery products without inventing DST targets."""
    autumn = delivery_grid("2026-10-25", 15, "Europe/Rome")
    spring = delivery_grid("2026-03-29", 15, "Europe/Rome")
    assert len(autumn) == 100
    assert len(spring) == 92
    assert set(slot_numbers(autumn, "Europe/Rome")) == set(range(100))
    assert {8, 9, 10, 11}.isdisjoint(slot_numbers(spring, "Europe/Rome"))
    assert np.isnan(pad_curve(spring, np.ones(92), "Europe/Rome")[[8, 9, 10, 11]]).all()


def test_price_lookup_requires_publication_and_repeats_native_hour():
    """Past hourly prices cover their four quarters only after publication."""
    frame = pd.DataFrame({"product_start": pd.to_datetime(["2025-09-30T12:00Z"]),
        "product_end": pd.to_datetime(["2025-09-30T13:00Z"]),
        "available_at": pd.to_datetime(["2025-09-29T11:00Z"]),
        "price_eur_per_mwh": [115.0]})
    index = PriceIndex(frame)
    starts = pd.date_range("2025-09-30T12:00Z", periods=4, freq="15min")
    assert np.isnan(index.lookup(starts, pd.Timestamp("2025-09-29T10:59Z"))).all()
    np.testing.assert_array_equal(index.lookup(starts, pd.Timestamp("2025-09-29T11:00Z")), np.full(4, 115.0))
    assert index.latest_publication(starts, pd.Timestamp("2025-09-29T10:30Z")) == pd.Timestamp("2025-09-29T11:00Z")


def test_weather_uses_latest_eligible_cycle_and_explicit_revision():
    """A 12 UTC run becomes eligible only six hours after initialization."""
    frame = pd.DataFrame({"location_id": ["milan"] * 4, "variable": ["temperature_2m"] * 4,
        "cycle_time": pd.to_datetime(["2026-07-01T06:00Z"] * 2 + ["2026-07-01T12:00Z"] * 2),
        "valid_time": pd.to_datetime(["2026-07-02T00:00Z", "2026-07-02T01:00Z"] * 2),
        "value": [20.0, 22.0, 23.0, 25.0], "temporal_semantics": ["instantaneous"] * 4})
    options = {"weather_locations": ["milan"], "weather_variables": ["temperature_2m"],
        "weather_cycle_hours_utc": [0, 6, 12], "weather_release_lag_hours": 6}
    weather = WeatherIndex(frame, options)
    starts = pd.date_range("2026-07-02T00:00Z", periods=4, freq="15min")
    current, revision, trace = weather.extract(starts, pd.Timestamp("2026-07-01T18:00Z"),
        pd.Timestamp("2026-07-01T13:00Z"), "Europe/Rome")
    slots = slot_numbers(starts, "Europe/Rome")
    np.testing.assert_allclose(current[slots], [23.0, 23.5, 24.0, 24.5])
    np.testing.assert_allclose(revision[slots], np.full(4, 3.0))
    assert trace["milan/temperature_2m"]["cycle"].startswith("2026-07-01T12:00")


def test_duplicate_weather_runs_allow_identical_values_but_reject_conflicts():
    """Multiple cache identities cannot silently change a forecast cycle."""
    import pytest
    frame = pd.DataFrame({"location_id": ["milan", "milan"], "variable": ["temperature_2m"] * 2,
        "cycle_time": pd.to_datetime(["2026-07-01T12:00Z"] * 2),
        "valid_time": pd.to_datetime(["2026-07-02T00:00Z"] * 2), "value": [20.0, 20.0],
        "temporal_semantics": ["instantaneous"] * 2})
    options = {"weather_locations": ["milan"], "weather_variables": ["temperature_2m"],
        "weather_cycle_hours_utc": [0, 6, 12], "weather_release_lag_hours": 6}
    assert len(WeatherIndex(frame, options).runs[("milan", "temperature_2m")][0][1]) == 1
    frame.loc[1, "value"] = 21.0
    with pytest.raises(ValueError, match="Conflicting weather timestamps"):
        WeatherIndex(frame, options)


def test_weather_falls_back_to_latest_complete_eligible_run():
    """An incomplete latest run cannot displace a full older available cycle."""
    frame = pd.DataFrame({"location_id": ["milan"] * 5, "variable": ["temperature_2m"] * 5,
        "cycle_time": pd.to_datetime(["2026-07-01T06:00Z"] * 2 + ["2026-07-01T12:00Z"]
            + ["2026-07-01T18:00Z"] * 2),
        "valid_time": pd.to_datetime(["2026-07-02T00:00Z", "2026-07-02T01:00Z",
            "2026-07-02T00:00Z", "2026-07-02T00:00Z", "2026-07-02T01:00Z"]),
        "value": [20.0, 22.0, 23.0, 30.0, 32.0], "temporal_semantics": ["instantaneous"] * 5})
    options = {"weather_locations": ["milan"], "weather_variables": ["temperature_2m"],
        "weather_cycle_hours_utc": [0, 6, 12, 18], "weather_release_lag_hours": 6}
    starts = pd.date_range("2026-07-02T00:00Z", periods=4, freq="15min")
    current, revision, trace = WeatherIndex(frame, options).extract(starts,
        pd.Timestamp("2026-07-01T18:00Z"), pd.Timestamp("2026-07-01T13:00Z"), "Europe/Rome")
    slots = slot_numbers(starts, "Europe/Rome")
    np.testing.assert_allclose(current[slots], [20.0, 20.5, 21.0, 21.5])
    np.testing.assert_allclose(revision[slots], np.zeros(4))
    assert trace["milan/temperature_2m"]["cycle"].startswith("2026-07-01T06:00")
    assert trace["milan/temperature_2m"]["latest_eligible_cycle"].startswith("2026-07-01T12:00")
    assert trace["milan/temperature_2m"]["older_complete_cycle_used"]


def test_examples_hide_future_auctions_and_fallback_only_on_cancellation(tmp_path, monkeypatch):
    """The same delivery-day IDA cancellation is visible only at later gates."""
    rows = []
    datasets = {"day_ahead_prices": 100.0, "ida1_prices": 150.0,
                "ida2_prices": 170.0, "ida3_prices": 180.0}
    for day in ("2025-10-01", "2025-10-02"):
        starts = delivery_grid(day, 15, "Europe/Rome")
        for dataset, value in datasets.items():
            if dataset == "ida1_prices" and day == "2025-10-02":
                continue
            for start in starts:
                rows.append({"dataset": dataset, "product_start": start,
                    "product_end": start + pd.Timedelta(minutes=15), "value": value,
                    "published_at": pd.NaT})
    frame = pd.DataFrame(rows)
    path = tmp_path / "nord_auctions.parquet"
    frame.to_parquet(path)
    manifest = {"tables": {"nord_auctions": {"path": path.name, "sha256": digest(path)}},
        "gme_cancellation_notices": [{"dataset": "ida1_prices", "delivery_day": "2025-10-02"}]}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    def empty_external(frames, grid, settings):
        return np.empty((len(grid), 0)), [], {}
    monkeypatch.setattr("bess_pv_trading.forecasting.studies.rolling_price_data._feature_blocks", empty_external)
    settings = {"base_search_config": "configs/research/price_feature_selection_full_year.yaml",
        "timezone": "Europe/Rome", "start": "2025-10-01", "example_start": "2025-10-02",
        "end_exclusive": "2025-10-03", "weather_locations": ["milan"],
        "weather_variables": ["temperature_2m"], "weather_cycle_hours_utc": [0, 6, 12],
        "weather_release_lag_hours": 6}
    examples = build_examples(tmp_path, settings)
    first = next(row for row in examples if row["market"] == "IDA2" and row["gate"] == "DA")
    later = next(row for row in examples if row["market"] == "IDA2" and row["gate"] == "IDA2")
    assert np.isnan(first["feature_blocks"]["core"][1200:1300]).all()
    assert first["feature_blocks"]["core"][1401] == 0
    np.testing.assert_array_equal(later["price_reference"], np.full(96, 100.0))
    np.testing.assert_array_equal(later["feature_blocks"]["core"][1301:1397], np.full(96, 100.0))
    assert later["feature_blocks"]["core"][1401] == 1
    assert not later["metadata"]["weather_usable"]
