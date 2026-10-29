"""Verify review-only near-flat detection and interval semantics."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.forecasting.plotting.pv_near_flat_review import find_candidates, contiguous_spans


def test_existing_exclusions_do_not_fragment_raw_candidate():
    """Detect on raw values before counting which quarters the prior search used."""
    times = pd.date_range('2026-03-20T10:00Z', periods=8, freq='15min')
    frame = pd.DataFrame({'interval_start_utc': times, 'power_kw': [1012.2, 1011.3] * 4,
                          'ghi_w_m2': np.linspace(0, 400, 8),
                          'search_eligible': [True, True, False, True, True, True, True, True],
                          'episode_excluded': [False, False, True, False, False, False, False, False],
                          'in_training': True})
    section = {'minimum_quarters': 4, 'tolerance_kw': 1.5, 'priority_zero_ghi_w_m2': 0,
               'priority_minimum_ac_fraction': .01, 'priority_minimum_quarters': 8,
               'priority_ghi_range_w_m2': 200, 'timezone': 'Europe/Rome', 'linked_minimum_distinct_days': 3}
    result = find_candidates(frame, SimpleNamespace(farm_id=910, ac_capacity_kw=6638), section)
    assert len(result) == 1
    assert result[0]['quarters'] == 8
    assert result[0]['eligible_training_quarters'] == 7
    assert result[0]['existing_excluded_quarters'] == 1
    assert result[0]['priority'] == 'strong_evidence'


def test_spans_stop_at_missing_quarters():
    """A missing observation must not be visually joined to the following interval."""
    times = pd.date_range('2026-03-20T10:00Z', periods=4, freq='15min')
    spans = contiguous_spans(times, [True, True, False, True])
    assert spans == [(times[0], times[2]), (times[3], times[3] + pd.Timedelta(minutes=15))]
