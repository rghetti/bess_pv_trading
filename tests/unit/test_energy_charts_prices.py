"""Validate Energy Charts normalization without internet or external services."""

import json

import pandas as pd
import pytest

from bess_pv_trading.data.energy_charts import parse_energy_charts_prices


def test_quarter_hour_prices_preserve_fall_dst_and_missing_values():
    """Both repeated local hours remain distinct and null prices remain absent."""
    times = pd.DatetimeIndex(["2026-10-25T00:30Z", "2026-10-25T01:30Z", "2026-10-25T01:45Z"]).as_unit("ns")
    payload = json.dumps({"unit": "EUR / MWh", "unix_seconds": (times.asi8 // 10**9).tolist(),
                          "price": [12.0, -2.0, None], "license_info": "CC BY 4.0"})
    frame, metadata = parse_energy_charts_prices(payload, zone="NORD", start="2026-10-25",
                                                  end_exclusive="2026-10-26")
    assert frame.value.tolist() == [12.0, -2.0]
    assert frame.product_start.nunique() == 2
    assert frame.product_start.dt.tz_convert("Europe/Rome").dt.hour.tolist() == [2, 2]
    assert frame.published_at.isna().all()
    assert metadata["null_prices"] == 1


def test_rejects_duplicate_product_times():
    """Duplicate prices cannot silently create ambiguous market products."""
    payload = json.dumps({"unit": "EUR / MWh", "unix_seconds": [1792890000, 1792890000], "price": [1, 2]})
    with pytest.raises(ValueError, match="unique"):
        parse_energy_charts_prices(payload, zone="NORD", start="2026-10-24", end_exclusive="2026-10-26")
