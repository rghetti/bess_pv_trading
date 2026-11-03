"""Verify user-reviewed replacements, restoration, night rules, and local DST dates."""
from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.data.pv_reviewed_targets import (
    apply_reviewed_targets, build_selection, decision_rows,
)


@pytest.fixture
def settings():
    """Use production decisions with individual tests selecting small custom date lists."""
    return load_config('configs/pv_experiments/reviewed_ghi_user_decisions.yaml')['pv_reviewed_ghi']


def sample(local_day, *, farm_id=910, split='train'):
    """Create the real 92/96/100 quarter-hour local day for a controlled test."""
    start = pd.Timestamp(local_day).tz_localize('Europe/Rome')
    end = (pd.Timestamp(local_day) + pd.DateOffset(days=1)).tz_localize('Europe/Rome')
    times = pd.date_range(start, end, freq='15min', inclusive='left').tz_convert('UTC')
    local = times.tz_convert('Europe/Rome')
    night = (local.hour < 6) | (local.hour >= 21)
    return pd.DataFrame({
        'interval_start_utc': times, 'delivery_day': local.strftime('%Y-%m-%d'),
        'split': split, 'night': night, 'evaluation_window': ~night, 'embargoed': False,
        'study_excluded': False, 'episode_excluded': False,
        'low_power_candidate': False, 'night_production_candidate': False,
        'measured_ac_raw_mw': .2, 'measured_ac_mw': .2, 'pvlib_historical_ghi_mw': .8,
        'ghi_raw_w_m2': 400., 'ghi_w_m2': 400., 'temperature_c': 20.,
        'wind_u_10m': 1., 'wind_v_10m': 1.,
    })


def choose(settings, farm_id, **rules):
    """Isolate the exact rule under test without depending on local source files."""
    result = deepcopy(settings)
    result['selections'] = {str(farm_id): rules}
    return result


def test_full_day_includes_both_repeated_hours_and_clips_synthetic_night(settings):
    frame = sample('2025-10-26')
    assert len(frame) == 100
    frame.loc[frame.night, 'pvlib_historical_ghi_mw'] = np.nan
    section = choose(settings, 910, full_days=['2025-10-26'])
    result, _ = apply_reviewed_targets(frame, 910, section)
    assert result.synthetic_target.all()
    assert result.loc[result.night, 'model_target_mw'].eq(0).all()
    assert result.loc[~result.night, 'model_target_mw'].eq(.8).all()
    assert result.training_eligible.all()
    pd.testing.assert_series_equal(result.measured_ac_raw_mw, frame.measured_ac_raw_mw)


def test_date_only_replaces_red_and_preserves_separate_gray_exclusion(settings):
    frame = sample('2025-03-02')
    frame.loc[20:23, 'low_power_candidate'] = True
    frame.loc[50:53, 'study_excluded'] = True
    section = choose(settings, 910, episode_days=['2025-03-02'])
    result, _ = apply_reviewed_targets(frame, 910, section)
    assert result.reviewed_action.iloc[20:24].eq('pvlib').all()
    assert result.loc[50:53, 'retained_prior_exclusion'].all()
    assert result.loc[50:53, 'episode_excluded'].all()
    assert not result.loc[50:53, 'training_eligible'].any()
    assert result.loc[50:53, 'model_target_mw'].eq(.2).all()


def test_explicit_restoration_keeps_red_replacement_and_restores_other_gray(settings):
    frame = sample('2026-04-09', farm_id=345, split='validation')
    frame.loc[25:28, 'study_excluded'] = True
    frame.loc[40:43, 'study_excluded'] = True
    frame.loc[40:43, 'low_power_candidate'] = True
    section = choose(settings, 345, restore_days=['2026-04-09'], episode_days=['2026-04-09'])
    result, _ = apply_reviewed_targets(frame, 345, section)
    assert result.loc[25:28, 'previous_exclusion_restored'].all()
    assert result.loc[25:28, 'model_target_mw'].eq(.2).all()
    assert result.loc[40:43, 'model_target_mw'].eq(.8).all()
    assert not result.retained_prior_exclusion.any()
    assert not result.episode_excluded.any()


def test_complete_restoration_restores_flagged_red_and_prior_exclusions(settings):
    frame = sample('2025-03-23')
    frame.loc[36:39, 'low_power_candidate'] = True
    frame.loc[45:49, 'study_excluded'] = True
    section = choose(settings, 910, restore_days=['2025-03-23'])
    result, _ = apply_reviewed_targets(frame, 910, section)
    assert not result.synthetic_target.any()
    assert result.loc[45:49, 'previous_exclusion_restored'].all()
    assert result.training_eligible.all()


def test_before_noon_uses_local_clock_and_includes_dst_fold(settings):
    frame = sample('2025-10-26', farm_id=345)
    section = choose(settings, 345, replace_before_local={'2025-10-26': '12:00'})
    result, _ = apply_reviewed_targets(frame, 345, section)
    assert result.synthetic_target.sum() == 52
    local = pd.DatetimeIndex(result.interval_start_utc).tz_convert('Europe/Rome')
    assert result.loc[local.hour < 12, 'synthetic_target'].all()
    assert not result.loc[local.hour >= 12, 'synthetic_target'].any()


def test_explicit_zero_from_time_overrides_pvlib_before_and_after_night(settings):
    frame = sample('2026-06-14', farm_id=901, split='test')
    section = choose(settings, 901, full_days=['2026-06-14'],
                     zero_from_local={'2026-06-14': '20:45'})
    result, _ = apply_reviewed_targets(frame, 901, section)
    local = pd.DatetimeIndex(result.interval_start_utc).tz_convert('Europe/Rome')
    since = (local.hour * 60 + local.minute) >= 20 * 60 + 45
    assert result.loc[since, 'reviewed_action'].eq('zero').all()
    assert result.loc[since, 'model_target_mw'].eq(0).all()
    assert not result.loc[result.night, 'test_eligible'].any()
    assert result.loc[~result.night, 'test_eligible'].all()


def test_flagged_night_positive_replaced_and_missing_weather_not_fabricated(settings):
    frame = sample('2025-01-01', farm_id=345)
    frame.loc[0, 'night_production_candidate'] = True
    frame.loc[0, 'ghi_w_m2'] = np.nan
    frame.loc[0, 'pvlib_historical_ghi_mw'] = np.nan
    section = choose(settings, 345, replace_flagged_night_production=True)
    result, _ = apply_reviewed_targets(frame, 345, section)
    assert result.loc[0, 'model_target_mw'] == 0.
    assert not result.loc[0, 'training_eligible']
    assert pd.isna(result.loc[0, 'ghi_w_m2'])


def test_missing_daytime_reference_stays_missing_and_is_not_training_eligible(settings):
    frame = sample('2025-03-02')
    frame.loc[40:43, 'low_power_candidate'] = True
    frame.loc[40, 'pvlib_historical_ghi_mw'] = np.nan
    section = choose(settings, 910, episode_days=['2025-03-02'])
    result, _ = apply_reviewed_targets(frame, 910, section)
    assert pd.isna(result.loc[40, 'model_target_mw'])
    assert result.loc[40, 'model_target_source'] == 'pvlib_unavailable'
    assert not result.loc[40, 'training_eligible']


def test_decisions_csv_runs_reproduce_all_final_raw_exclusions(settings):
    frame = sample('2025-03-02')
    frame.loc[20:23, 'low_power_candidate'] = True
    frame.loc[50:53, 'study_excluded'] = True
    section = choose(settings, 910, episode_days=['2025-03-02'])
    result, _ = apply_reviewed_targets(frame, 910, section)
    decisions = pd.DataFrame(decision_rows(result, 910, 'Europe/Rome'))
    assert set(decisions.replacement) == {'pvlib', 'prior_exclusion'}
    mask = np.zeros(len(result), dtype=bool)
    for row in decisions.itertuples(index=False):
        selected = result.interval_start_utc.between(row.start_utc, row.end_utc)
        assert not (mask & selected).any()
        mask |= selected
    np.testing.assert_array_equal(mask, result.raw_target_excluded)


def test_missing_requested_red_episode_fails_instead_of_guessing(settings):
    frame = sample('2025-03-02')
    section = choose(settings, 910, episode_days=['2025-03-02'])
    with pytest.raises(ValueError, match='no matching quarter-hours'):
        build_selection(frame, 910, section)
