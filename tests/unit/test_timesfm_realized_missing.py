"""Check the explicit gap threshold and mixed forecast-availability batching."""

import numpy as np
import pandas as pd

from scripts.prepare_timesfm_realized import fill_history
from scripts.run_da_zero_shot_model import request_batches


def test_forecast_precedes_short_interpolation_and_long_gap_zero():
    """Forecast values are retained; eight quarters interpolate and nine become zero."""
    result, forecasts, interpolated, zeros = fill_history([1, np.nan, np.nan, np.nan, 5], [0, 2, np.nan, np.nan, 0])
    assert np.allclose(result, [1, 2, 3, 4, 5])
    assert (forecasts, interpolated, zeros) == (1, 2, 0)
    assert fill_history([1] + [np.nan] * 8 + [10])[2:] == (8, 0)
    result, _, interpolated, zeros = fill_history([1] + [np.nan] * 9 + [10])
    assert interpolated == 0 and zeros == 9 and (result[1:-1] == 0).all()


def test_missing_ida3_overnight_is_not_interpolated_across():
    """IDA3's overnight discontinuity is not mistaken for a quarter-hour step."""
    clock = pd.DatetimeIndex(['2026-01-01T22:45Z', '2026-01-02T12:00Z', '2026-01-02T12:15Z'])
    values, _, _, _ = fill_history([9, np.nan, 7], timestamps=clock)
    assert values.tolist() == [9, 7, 7]


def test_batching_separates_future_and_past_only_covariates():
    """A missing future forecast creates a different, valid worker batch."""
    request = {'context': [1], 'horizon': 1, 'forecast_covariates': {'wind': [1, 2]}, 'past_covariates': {}}
    fallback = {'context': [1], 'horizon': 1, 'forecast_covariates': {}, 'past_covariates': {'wind': [1]}}
    assert len(list(request_batches([('a', 'a', request), ('b', 'b', fallback)], 10))) == 2
