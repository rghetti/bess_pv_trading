"""Check quarter-hour midpoint alignment for individual forecast vintages."""
import numpy as np

from src.forecasting.pv_realized_irradiance import interpolate_instantaneous_midpoints


def test_midpoint_interpolation_keeps_vintages_and_gaps_separate():
    """A linear hourly signal has known values at all quarter-hour midpoints."""
    data = np.array([[0.0, 8.0, 16.0], [100.0, 108.0, np.nan]])
    actual = interpolate_instantaneous_midpoints(data, 2)
    np.testing.assert_allclose(actual[0], [1, 3, 5, 7, 9, 11, 13, 15])
    np.testing.assert_allclose(actual[1, :4], [101, 103, 105, 107])
    assert np.isnan(actual[1, 4:]).all()
