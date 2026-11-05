"""Check chronological model-ranking information boundaries."""

import numpy as np

from scripts.evaluate_online_price_ranking import online_rank


def test_unpublished_and_future_targets_cannot_affect_prior_rankings():
    """Earlier decisions cannot rank models using unpublished or later labels."""
    day_ns = 24 * 60 * 60 * 1_000_000_000
    starts = np.arange(4, dtype=np.int64) * day_ns
    ends = starts + day_ns // 24
    decisions = starts - day_ns // 2
    available = ends.copy()
    available[0] = 10 * day_ns
    actual = np.array([5., 0., 10., 5.])
    forecasts = np.tile([0., 10.], (4, 1))
    days = np.array(["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04"])
    prediction, selected = online_rank(actual, forecasts, starts, ends, available, decisions, days,
                                        window_days=7, top_k=1)
    assert prediction[0] == prediction[1] == 5
    assert selected["2026-01-03"] == [0]
    changed = actual.copy()
    changed[0] = 1000
    changed[3] = 1000
    again, _ = online_rank(changed, forecasts, starts, ends, available, decisions, days,
                           window_days=7, top_k=1)
    np.testing.assert_array_equal(prediction[:3], again[:3])
