"""Keep full-period fitting and held-out validation semantically distinct."""

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.forecasting.training.pv_realized_search import fit_and_score_masks, ranking_path
from bess_pv_trading.forecasting.data.pv_realized_irradiance import treat_night_targets
from bess_pv_trading.config import load_config
from bess_pv_trading.data import night_window_mask


def test_fit_and_score_scopes_and_ranking_names(tmp_path):
    frame = pd.DataFrame({'split': ['train', 'train', 'validation']})
    fit, score = fit_and_score_masks(frame, 'validation')
    np.testing.assert_array_equal(fit, [True, True, False])
    np.testing.assert_array_equal(score, [False, False, True])
    fit, score = fit_and_score_masks(frame, 'full_period')
    np.testing.assert_array_equal(fit, [True, True, True])
    np.testing.assert_array_equal(score, [True, True, True])
    assert ranking_path(tmp_path, {'selection_scope': 'full_period'}).name == 'fit_ranking.csv'
    assert ranking_path(tmp_path, {'selection_scope': 'validation'}, screen=True).name == 'validation_ranking_screen.csv'
    with pytest.raises(ValueError):
        fit_and_score_masks(frame, 'invalid')


def test_night_target_treatment_preserves_original_and_missing_values():
    starts = pd.to_datetime(['2026-01-01T20:15Z', '2026-01-01T20:30Z',
                             '2026-01-02T03:45Z', '2026-01-02T04:00Z',
                             '2026-01-02T04:15Z'], utc=True)
    source = pd.DataFrame({'interval_start_utc': starts,
                           'measured_ac_mw': [.1, .2, np.nan, .3, .4]})
    policy = {'enabled': True, 'timezone': 'Europe/Rome', 'start': '21:30', 'end': '05:00'}
    result = treat_night_targets(source, policy, 'zero')
    np.testing.assert_allclose(result.measured_ac_raw_mw, source.measured_ac_mw, equal_nan=True)
    np.testing.assert_allclose(result.measured_ac_mw, [.1, 0., np.nan, .3, .4], equal_nan=True)
    assert result.night_target_zeroed.tolist() == [False, True, False, False, False]
    np.testing.assert_allclose(treat_night_targets(source, policy, 'preserve').measured_ac_mw,
                               source.measured_ac_mw, equal_nan=True)


def test_seasonal_night_boundaries_and_spring_autumn_clock_changes():
    config = load_config('configs/standard.yaml')
    policy = config['pv_realized_irradiance_search']['night_policy']
    local = pd.to_datetime(['2026-01-15 06:30', '2026-01-15 06:45',
                            '2026-06-15 21:15', '2026-06-15 21:30',
                            '2026-06-16 04:45', '2026-06-16 05:00']).tz_localize('Europe/Rome')
    assert night_window_mask(local, policy).tolist() == [True, False, False, True, True, False]
    # Both occurrences of the repeated autumn local hour remain nighttime.
    repeated = pd.to_datetime(['2026-10-25T00:30Z', '2026-10-25T01:30Z'], utc=True)
    assert night_window_mask(repeated, policy).tolist() == [True, True]
    year = pd.date_range('2026-05-01', '2026-06-01', freq='15min',
                         tz='Europe/Rome', inclusive='left')
    maximum = {'enabled': True, 'timezone': 'Europe/Rome', 'start': '21:30', 'end': '05:00'}
    np.testing.assert_array_equal(night_window_mask(year, policy), night_window_mask(year, maximum))
