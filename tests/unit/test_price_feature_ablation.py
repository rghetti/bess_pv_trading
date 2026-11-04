"""Check paired ablation metrics and prevent misleading sample comparisons."""

import pandas as pd
import pytest

from scripts.report_price_feature_ablation import paired_score


def test_paired_errors_and_day_block_intervals():
    """Constant improvement is preserved by every daily bootstrap sample."""
    base = pd.DataFrame({"product_start": ["2025-01-01T00:00:00Z", "2025-01-02T00:00:00Z"],
                         "actual": [10.0, 20.0], "point": [12.0, 22.0]})
    added = base.assign(point=[11.0, 21.0])
    result = paired_score(base, added, timezone="UTC", seed=7, bootstrap_samples=100)
    assert result["mae_difference"] == -1.0
    assert result["difference_day_bootstrap_low"] == -1.0
    assert result["difference_day_bootstrap_high"] == -1.0
    assert result["base_wape_percent"] == pytest.approx(100 * 4 / 30)
    assert result["days"] == 2


def test_paired_errors_reject_misaligned_observations():
    """Different products or observed prices cannot form a paired comparison."""
    base = pd.DataFrame({"product_start": ["2025-01-01T00:00:00Z"], "actual": [10.0], "point": [11.0]})
    with pytest.raises(ValueError, match="identical ordered products"):
        paired_score(base, base.assign(product_start="2025-01-02T00:00:00Z"),
                     timezone="UTC", seed=7, bootstrap_samples=100)
    with pytest.raises(ValueError, match="different actual prices"):
        paired_score(base, base.assign(actual=20.0), timezone="UTC", seed=7, bootstrap_samples=100)
