"""Check local time and signed error summaries for selected price forecasts."""

import pandas as pd
import pytest

from scripts.report_selected_price_errors import profile_rows


def test_error_profiles_use_local_delivery_hour_and_preserve_signed_bias():
    """Monthly WAPE and local-hour bias use the same observed products."""
    frame = pd.DataFrame({"product_start": ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z"],
                          "actual": [10.0, 20.0], "point": [12.0, 17.0], "seasonal_naive": [11.0, 24.0]})
    month = profile_rows(frame, timezone="Europe/Rome", grouping="month")
    hours = profile_rows(frame, timezone="Europe/Rome", grouping="hour")
    assert month[0]["wape_percent"] == pytest.approx(100 * 5 / 30)
    assert month[0]["bias_eur_per_mwh"] == pytest.approx(-0.5)
    assert [row["period"] for row in hours] == [1, 2]
