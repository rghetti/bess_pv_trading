"""Ensure fallback sources never overwrite existing market observations."""

import pandas as pd
import pytest

from bess_pv_trading.data.price_source_fallback import prefer_existing_intervals


def test_fallback_preserves_primary_and_reports_difference():
    """One overlapping source is retained and one new source fills the gap."""
    primary = pd.DataFrame({"product_start": [1], "product_end": [2], "value": [10.0], "provider": ["gme"]})
    fallback = pd.DataFrame({"product_start": [1, 2], "product_end": [2, 3],
                             "value": [10.004, 20.0], "provider": ["energy_charts", "energy_charts"]})
    selected, report = prefer_existing_intervals(primary, fallback, identity_columns=["product_start", "product_end"])
    assert selected.value.tolist() == [10.0, 20.0]
    assert selected.provider.tolist() == ["gme", "energy_charts"]
    assert report["overlap_rows"] == 1
    assert report["fallback_rows_used"] == 1
    assert report["max_overlap_absolute_difference"] == pytest.approx(0.004)


def test_fallback_rejects_unequal_product_durations():
    """An hourly record cannot silently replace a quarter-hour product."""
    primary = pd.DataFrame({"product_start": [1], "product_end": [5], "value": [10.0]})
    fallback = pd.DataFrame({"product_start": [1], "product_end": [2], "value": [11.0]})
    with pytest.raises(ValueError, match="boundaries"):
        prefer_existing_intervals(primary, fallback, identity_columns=["product_start", "product_end"])
