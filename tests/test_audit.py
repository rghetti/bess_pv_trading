"""Small synthetic checks for interval alignment and review-only diagnostics."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from bess_pv_trading.data import (
    build_forecast_checks,
    build_historical_context,
    validate_inputs,
)
from bess_pv_trading.diagnostics import detect_events
from bess_pv_trading.report import _selected_plot_events


class ForecastAlignmentTests(unittest.TestCase):
    """Check that a published cycle is used only for forecastable hours."""

    def setUp(self) -> None:
        """Create two cycles whose publication boundary is 10:00 UTC."""
        self.times = pd.date_range("2025-01-01T00:00Z", periods=48, freq="15min")
        cycles = pd.date_range("2025-01-01T00:00Z", periods=2, freq="6h")
        leads = np.arange(13).astype("timedelta64[h]")
        valid_time = cycles.tz_localize(None).to_numpy()[:, None] + leads[None, :]
        radiation = np.zeros((2, 2, 13), dtype=float)
        boundary = (valid_time < self.times[0].tz_localize(None).to_datetime64()) | (
            valid_time > self.times[-1].tz_localize(None).to_datetime64()
        )
        radiation[:, boundary] = np.nan
        radiation[:, :, 0] = np.nan  # Accumulated radiation has no lead-zero hour.
        radiation[:, 0, 10] = [110, 160]
        radiation[:, 1, 5] = [220, 270]
        historical_ghi = np.zeros((2, 48), dtype=float)
        historical_ghi[0, 36:40] = 100
        historical_ghi[0, 40:44] = 200
        historical_ghi[1, 36:40] = 150
        historical_ghi[1, 40:44] = 250
        historical_power = np.array([[15, 30]] * 48, dtype=float)
        historical_power[37, 0] = -5.0  # An implausible raw value must survive alignment.
        self.dataset = xr.Dataset(
            data_vars={
                "power": (("time", "farm_id"), historical_power),
                "ghi": (("farm_id", "time"), historical_ghi),
                "fraction_diffuse": (("farm_id", "time"), np.full((2, 48), 0.5)),
                "shortwave_radiation": (("farm_id", "cycle_time", "lead_time"), radiation),
                "direct_radiation": (("farm_id", "cycle_time", "lead_time"), radiation / 2),
                "precipitation": (("farm_id", "cycle_time", "lead_time"), radiation * 0),
            },
            coords={
                "farm_id": [1, 2],
                "time": self.times.tz_localize(None).to_numpy(),
                "cycle_time": cycles.tz_localize(None).to_numpy(),
                "lead_time": leads,
                "valid_time": (("cycle_time", "lead_time"), valid_time),
            },
            attrs={"timezone": "UTC", "forecast_valid_time": "cycle_time + lead_time"},
        )
        self.metadata = pd.DataFrame({"farm_id": [1, 2], "total_power": [1000.0, 2000.0]})

    def test_latest_available_vintage_and_raw_value(self) -> None:
        """Forecast at 09:00 uses cycle 00; forecast at 10:00 uses cycle 06."""
        context = build_historical_context(
            self.dataset, self.metadata, self.times[0], self.times[-1], 45.4642, 9.19, 4
        )
        farm_one = context.loc[context["farm_id"] == 1].set_index("time_utc")
        self.assertEqual(farm_one.loc[self.times[36], "forecast_shortwave_w_m2"], 110)
        self.assertEqual(farm_one.loc[self.times[40], "forecast_shortwave_w_m2"], 220)
        self.assertEqual(farm_one.loc[self.times[36], "forecast_cycle_time_utc"], pd.Timestamp("2025-01-01T00:00Z"))
        self.assertEqual(farm_one.loc[self.times[40], "forecast_cycle_time_utc"], pd.Timestamp("2025-01-01T06:00Z"))
        self.assertEqual(farm_one.loc[self.times[37], "power_dc_kw"], -5.0)
        self.assertEqual(float(self.dataset["power"].sel(time=self.times[37].tz_localize(None), farm_id=1)), -5.0)

    def test_hourly_ghi_match_uses_four_complete_quarters(self) -> None:
        """An hourly check compares an interval mean, not one endpoint."""
        context = build_historical_context(
            self.dataset, self.metadata, self.times[0], self.times[-1], 45.4642, 9.19, 4
        )
        checks = build_forecast_checks(self.dataset, context, self.times[0], self.times[-1])
        farm_one = checks.loc[checks["farm_id"] == 1]
        self.assertGreater(len(farm_one), 0)
        self.assertIn("mae_w_m2", checks.columns)
        self.assertEqual(float(farm_one.loc[farm_one["lead_hours"] == 5, "mae_w_m2"].iloc[0]), 10.0)
        self.assertEqual(int(farm_one.loc[farm_one["lead_hours"] == 5, "compared_hours"].iloc[0]), 2)
        self.assertEqual(float(farm_one.loc[farm_one["lead_hours"] == 10, "mae_w_m2"].iloc[0]), 10.0)
        validation = validate_inputs(self.dataset, self.metadata, self.times[0], self.times[-1])
        self.assertTrue(all(validation["grid"].values()))
        self.assertEqual(validation["unexpected_forecast_missing_total"], 0)
        self.assertGreater(validation["expected_forecast_boundary_missing_total"], 0)
        self.assertGreater(validation["expected_forecast_lead_zero_missing_total"], 0)

    def test_unexpected_missing_latest_forecast_is_not_hidden(self) -> None:
        """A missing published forecast stays missing instead of using an older value."""
        dataset = self.dataset.copy(deep=True)
        altered = dataset["shortwave_radiation"].values.copy()
        altered[0, 1, 5] = np.nan
        dataset["shortwave_radiation"] = (
            ("farm_id", "cycle_time", "lead_time"), altered
        )
        validation = validate_inputs(dataset, self.metadata, self.times[0], self.times[-1])
        self.assertEqual(validation["unexpected_forecast_missing_total"], 1)
        context = build_historical_context(
            dataset, self.metadata, self.times[0], self.times[-1], 45.4642, 9.19, 4
        )
        selected = context.loc[(context["farm_id"] == 1) & (context["time_utc"] == self.times[40])].iloc[0]
        self.assertTrue(pd.isna(selected["forecast_shortwave_w_m2"]))
        self.assertEqual(selected["forecast_cycle_time_utc"], pd.Timestamp("2025-01-01T06:00Z"))


class EpisodeTests(unittest.TestCase):
    """Check episode boundaries and simultaneous-farm observations."""

    @staticmethod
    def _context(ghi: list[float], power: list[float], *, farms: tuple[int, ...] = (1, 2)) -> pd.DataFrame:
        """Build a small long table with controlled daytime solar elevation."""
        times = pd.date_range("2025-06-01T10:00Z", periods=len(ghi), freq="15min")
        records = []
        for farm in farms:
            for index, time in enumerate(times):
                records.append({
                    "farm_id": farm,
                    "time_utc": time,
                    "power_dc_kw": float(power[index]),
                    "ghi_w_m2": float(ghi[index]),
                    "fraction_diffuse": 0.5,
                    "total_power_dc_kw": 2000.0,
                    "solar_elevation_deg": 25.0,
                    "forecast_shortwave_w_m2": 500.0,
                    "forecast_cycle_time_utc": pd.Timestamp("2025-06-01T06:00Z"),
                    "forecast_lead_hours": 5.0,
                })
        return pd.DataFrame.from_records(records)

    def test_shared_daytime_zero_episodes_remain_separate(self) -> None:
        """A nonzero quarter-hour divides two common GHI-zero runs."""
        context = self._context([500, 0, 0, 500, 0, 0], [800] * 6)
        original = context.copy(deep=True)
        events = detect_events(context, {})
        shared = events.loc[events["category"] == "shared_daylight_zero_ghi"]
        self.assertEqual(len(shared), 2)
        self.assertEqual(set(shared["duration_quarters"]), {2})
        self.assertEqual(events.loc[events["category"] == "daylight_zero_ghi", "farm_id"].nunique(), 2)
        self.assertEqual(events["event_id"].tolist(), detect_events(context, {})["event_id"].tolist())
        pd.testing.assert_frame_equal(context, original)

    def test_power_dip_and_constant_plateau(self) -> None:
        """A one-step power valley and an unchanged positive run are both reviewable."""
        dip_context = self._context([600, 610, 605, 620, 615], [1000, 1000, 20, 1000, 1000], farms=(901,))
        self.assertTrue((detect_events(dip_context, {})["category"] == "power_dip").any())
        plateau_context = self._context([100, 300, 500, 700, 600, 400, 200, 50], [500] * 8, farms=(910,))
        self.assertTrue((detect_events(plateau_context, {})["category"] == "constant_positive_power").any())

    def test_pinned_event_is_plotted_within_limit(self) -> None:
        """A configured event remains selectable even with no score-based slots."""
        context = self._context([600, 610, 605, 620, 615], [1000, 1000, 20, 1000, 1000], farms=(901,))
        events = detect_events(context, {})
        selected = _selected_plot_events(events, {"plots": {
            "max_events_per_farm_category": 0,
            "max_total_events": 1,
            "pinned_events": [{
                "farm_id": 901,
                "category": "power_dip",
                "time_utc": "2025-06-01T10:30:00Z",
            }],
        }})
        self.assertEqual(selected["category"].tolist(), ["power_dip"])


if __name__ == "__main__":
    unittest.main()
