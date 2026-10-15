"""Check forecast interpolation, vintage changes and observed-data denominators."""

import unittest

import numpy as np
import pandas as pd
import xarray as xr

from bess_pv_trading.cloud_check import attach_cloud_context, portfolio_context, hourly_shared_summary, CLOUD_VARIABLES


class CloudAlignmentTests(unittest.TestCase):
    """A newly published forecast must not manufacture a cloud tendency."""

    def test_same_vintage_cloud_tendency_across_cycle_switch(self):
        cycles = pd.to_datetime(["2025-06-01T00:00", "2025-06-01T06:00"])
        leads = np.arange(13).astype("timedelta64[h]")
        values = np.array([[2 * np.arange(13), 80 + np.arange(13)]], dtype=float)
        dataset = xr.Dataset({name: (("farm_id", "cycle_time", "lead_time"), values) for name in CLOUD_VARIABLES},
                             coords={"farm_id": [1], "cycle_time": cycles, "lead_time": leads})
        context = pd.DataFrame({
            "farm_id": [1, 1], "time_utc": pd.to_datetime(["2025-06-01T09:45Z", "2025-06-01T10:00Z"]),
            "forecast_cycle_time_utc": pd.to_datetime(["2025-06-01T00:00Z", "2025-06-01T06:00Z"]),
            "power_dc_kw": [800., 400.], "total_power_dc_kw": [1000., 1000.],
            "ghi_w_m2": [800., 700.], "solar_elevation_deg": [-40., -39.],
            "forecast_shortwave_w_m2": [700., 600.],
        })
        config = {"diagnostics": {"thresholds": {"daylight_solar_elevation_deg": 10, "abrupt_power_fraction": .25}},
                  "cloud_analysis": {"recovery_fraction_of_drop": .8, "activity_ghi_w_m2": 50,
                                     "activity_forecast_shortwave_w_m2": 50, "activity_power_fraction": .01}}
        original = context.copy(deep=True)
        result = attach_cloud_context(dataset, context, config)
        self.assertAlmostEqual(result.loc[0, "cloud_cover_midpoint_pct"], 19.75)
        self.assertAlmostEqual(result.loc[1, "cloud_cover_midpoint_pct"], 84.125)
        self.assertAlmostEqual(result.loc[1, "cloud_tendency_15min_pct"], .25)
        self.assertGreater(result.loc[1, "rolling_cloud_change_pct"], 60)
        self.assertTrue(result.loc[1, "cycle_changed"])
        self.assertTrue(result.loc[1, "power_drop"])
        self.assertFalse(result.loc[1, "proxy_solar_above_threshold"])
        pd.testing.assert_frame_equal(context, original)

    def test_missing_observations_are_excluded_from_shared_controls(self):
        records = []
        times = pd.date_range("2025-06-01T10:00Z", periods=3, freq="15min")
        for farm in (1, 2):
            for index, time in enumerate(times):
                row = {"farm_id": farm, "time_utc": time, "ghi_w_m2": [0., 500., np.nan][index],
                       "activity_evidence": True, "radiation_evidence": index == 1,
                       "power_fraction": .2, "solar_elevation_deg": -40.,
                       "forecast_shortwave_w_m2": 300.}
                row.update({f"{name}_midpoint_pct": 90. for name in CLOUD_VARIABLES})
                records.append(row)
        config = {"diagnostics": {"thresholds": {"positive_power_fraction": .001, "min_shared_farms": 2, "daylight_solar_elevation_deg": 10}}}
        _, portfolio = portfolio_context(pd.DataFrame(records), config)
        self.assertEqual(portfolio["shared_eligible"].tolist(), [True, True, False])
        self.assertEqual(portfolio["shared_zero"].tolist(), [True, False, False])
        self.assertEqual(portfolio["all_zero"].tolist(), [True, False, False])

    def test_hourly_all_farm_controls_require_every_farm_observed(self):
        portfolio = pd.DataFrame({
            "time_utc": pd.date_range("2025-06-01T10:00Z", periods=3, freq="h"),
            "cloud_cover_mean_pct": [95., 50., 20.],
            "shared_eligible": [True, True, True], "all_zero_eligible": [True, False, True],
            "shared_zero": [True, False, False], "all_zero": [True, False, False],
        })
        summary = hourly_shared_summary(portfolio).set_index("scope")
        self.assertEqual(summary.loc["shared_zero", "eligible_hours"], 3)
        self.assertEqual(summary.loc["all_zero", "eligible_hours"], 2)


if __name__ == "__main__":
    unittest.main()
