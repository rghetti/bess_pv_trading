"""Check common-product alignment across independently run price searches."""

import pandas as pd
import pytest

from scripts.compare_price_searches import align_predictions


def test_comparison_uses_only_common_products_and_rejects_changed_actuals():
    """Extra days can be excluded, but common observed prices must agree."""
    baseline = pd.DataFrame({"product_start": ["2026-01-01T00:00Z", "2026-01-02T00:00Z"],
                             "actual": [10.0, 20.0], "point": [11.0, 21.0]})
    comparison = pd.DataFrame({"product_start": ["2026-01-02T00:00Z", "2026-01-03T00:00Z"],
                               "actual": [20.0, 30.0], "point": [22.0, 33.0]})
    left, right = align_predictions(baseline, comparison)
    assert len(left) == len(right) == 1
    assert left.actual.iloc[0] == right.actual.iloc[0] == 20.0
    with pytest.raises(ValueError, match="identical actual prices"):
        align_predictions(baseline, comparison.assign(actual=[21.0, 30.0]))
