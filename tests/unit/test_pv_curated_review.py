"""Verify episode continuity, split-safe GHI, and the limited synthetic-target scope."""
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.data.pv_curated_review import (
    align_weather, apply_target_policy, classify_candidates, contiguous_positions,
    interpolate_ghi, preserve_review_notes, split_labels, verify_prepared_frame,
)


@pytest.fixture
def configuration():
    """Use the review configuration itself so its production thresholds are exercised."""
    return load_config('configs/pv_experiments/curated_ghi_review.yaml')


def evidence_frame(start='2026-08-02T10:00Z', periods=8, power=.1, reference=.8, ghi=600.):
    """Represent one farm's physical observations independently of forecast auction views."""
    times = pd.date_range(start, periods=periods, freq='15min')
    return pd.DataFrame({
        'interval_start_utc': times, 'delivery_day': times.strftime('%Y-%m-%d'),
        'split': 'test', 'measured_ac_raw_mw': power, 'pvlib_historical_ghi_mw': reference,
        'measured_ac_mw': power,
        'ghi_raw_w_m2': ghi, 'ghi_w_m2': ghi, 'night': False,
        'temperature_c': 20., 'wind_u_10m': 1., 'wind_v_10m': 1.,
        'episode_excluded': False, 'target_available_at_utc': times + pd.Timedelta(minutes=15),
    })


def no_decisions():
    """Provide an empty but valid authoritative decision table."""
    return pd.DataFrame(columns=['farm_id', 'start_utc', 'end_utc', 'reason', 'decision'])


def test_ghi_does_not_cross_split_boundaries_or_extrapolate():
    times = pd.date_range('2026-01-01', periods=6, freq='15min', tz='UTC')
    filled, flags = interpolate_ghi(times, [100, np.nan, np.nan, 400, np.nan, 600],
                                    ['train'] * 3 + ['validation'] * 3)
    np.testing.assert_allclose(filled, [100, np.nan, np.nan, 400, 500, 600], equal_nan=True)
    assert flags.tolist() == [False, False, False, False, True, False]


def test_negative_ghi_is_preserved_and_blocks_interpolation():
    times = pd.date_range('2026-01-01', periods=5, freq='15min', tz='UTC')
    filled, _ = interpolate_ghi(times, [100, np.nan, -1, np.nan, 500], ['train'] * 5)
    assert np.isnan(filled[1]) and np.isnan(filled[3])
    assert np.isnan(filled[2])


def test_runs_break_at_gaps_and_local_day_boundaries():
    times = pd.to_datetime(['2026-08-01T21:30Z', '2026-08-01T21:45Z',
                            '2026-08-01T22:00Z', '2026-08-01T22:30Z'])
    days = times.tz_convert('Europe/Rome').strftime('%Y-%m-%d')
    assert [len(run) for run in contiguous_positions(times, days, [True] * 4)] == [2, 1, 1]
    assert contiguous_positions(times, days, [True] * 4, minimum=3) == []


def test_relative_extreme_requires_long_sustained_evidence(configuration):
    section = configuration['pv_curated_review']
    frame = classify_candidates(evidence_frame(), 1., section)
    assert frame.relative_low_power.all()
    assert frame.extreme_low_power.all()
    short = classify_candidates(evidence_frame(periods=7), 1., section)
    assert short.relative_low_power.all()
    assert not short.extreme_low_power.any()


def test_normal_low_irradiance_and_missing_observations_do_not_trigger(configuration):
    section = configuration['pv_curated_review']
    frame = evidence_frame(ghi=50.)
    assert not classify_candidates(frame, 1., section).low_power_candidate.any()
    frame = evidence_frame()
    frame['ghi_raw_w_m2'] = np.nan
    assert not classify_candidates(frame, 1., section).low_power_candidate.any()


def test_zero_high_ghi_can_trigger_without_a_model_reference(configuration):
    frame = evidence_frame(power=0., reference=np.nan, ghi=150.)
    result = classify_candidates(frame, 1., configuration['pv_curated_review'])
    assert result.zero_power_high_ghi.all()
    assert not result.relative_low_power.any()


def test_only_authorized_farm_and_test_month_receive_synthetic_values(configuration):
    section = configuration['pv_curated_review']
    frame = classify_candidates(evidence_frame(), 1., section)
    treated = apply_target_policy(frame, 910, section, no_decisions())
    assert treated.synthetic_test_target.all()
    assert treated.test_target_mw.eq(.8).all()
    pd.testing.assert_series_equal(treated.measured_ac_raw_mw, frame.measured_ac_raw_mw)
    for farm_id in [1, 345, 901]:
        other = apply_target_policy(frame, farm_id, section, no_decisions())
        assert not other.synthetic_test_target.any()
        assert other.test_target_mw.eq(.1).all()
    for split in ['train', 'validation']:
        changed = frame.assign(split=split)
        assert not apply_target_policy(changed, 910, section, no_decisions()).synthetic_test_target.any()


def test_night_training_keeps_zeros_and_excludes_only_authorized_positive_values(configuration):
    section = configuration['pv_curated_review']
    frame = evidence_frame(start='2025-06-01T00:00Z', periods=4)
    frame['split'] = 'train'
    frame['night'] = True
    frame['ghi_w_m2'] = 0.
    frame['measured_ac_raw_mw'] = [0., .01, 0., .03]
    frame = classify_candidates(frame, 1., section)
    treated = apply_target_policy(frame, 345, section, no_decisions())
    assert treated.training_eligible.tolist() == [True, False, True, False]
    assert not treated.validation_eligible.any()
    assert not treated.evaluation_window.any()
    other = apply_target_policy(frame, 901, section, no_decisions())
    assert other.training_eligible.all()


def test_unreviewed_extreme_candidates_do_not_change_training_eligibility(configuration):
    section = configuration['pv_curated_review']
    frame = evidence_frame(start='2025-06-01T10:00Z').assign(split='train')
    frame = classify_candidates(frame, 1., section)
    assert frame.extreme_low_power.all()
    result = apply_target_policy(frame, 1, section, no_decisions())
    assert result.training_eligible.all()
    assert not result.study_excluded.any()


def test_synthetic_target_cannot_undo_independent_exclusion(configuration):
    section = configuration['pv_curated_review']
    frame = classify_candidates(evidence_frame(), 1., section)
    frame['episode_excluded'] = True
    decisions = pd.DataFrame([{'farm_id': 910, 'start_utc': str(frame.interval_start_utc.iloc[0]),
        'end_utc': str(frame.interval_start_utc.iloc[-1]), 'reason': 'nonzero_plateau', 'decision': 'exclude'}])
    result = apply_target_policy(frame, 910, section, decisions)
    assert result.synthetic_test_target.all()
    assert not result.test_eligible.any()


def test_missing_reference_stays_unresolved_instead_of_becoming_zero(configuration):
    section = configuration['pv_curated_review']
    frame = classify_candidates(evidence_frame(reference=np.nan, power=0., ghi=200.), 1., section)
    result = apply_target_policy(frame, 910, section, no_decisions())
    assert result.test_reconstruction_requested.all()
    assert not result.synthetic_test_target.any()


def test_previous_split_boundaries_and_embargo_are_preserved(configuration):
    section = configuration['pv_curated_review']
    times = pd.to_datetime(['2025-12-31T22:45Z', '2025-12-31T23:00Z',
                           '2026-04-30T21:45Z', '2026-04-30T22:00Z'])
    _, labels = split_labels(times, section)
    assert labels.tolist() == ['train', 'validation', 'validation', 'test']
    frame = evidence_frame(start='2025-12-31T10:00Z').assign(split='train')
    frame = classify_candidates(frame, 1., section)
    result = apply_target_policy(frame, 1, section, no_decisions())
    assert result.embargoed.all()
    assert not result.training_eligible.any()


def test_weather_alignment_respects_end_labeled_means_and_same_vintage(configuration):
    times = pd.date_range('2026-08-02T10:00Z', periods=4, freq='15min')
    leads = np.arange(49)
    variables = {}
    for source in ['shortwave_radiation', 'direct_radiation', 'temperature_2m',
                   'wind_u_component_10m', 'wind_v_component_10m']:
        variables[source] = (('cycle_time', 'lead_time'), leads[None, :].astype(float))
    selected = xr.Dataset(variables, coords={
        'cycle_time': [np.datetime64('2026-08-01T00:00')],
        'lead_time': leads.astype('timedelta64[h]'),
    })
    frame = align_weather(selected, times, configuration['pv_curated_review'], configuration)
    np.testing.assert_allclose(frame.forecast_ghi, 35.)
    np.testing.assert_allclose(frame.temperature_c, [34.125, 34.375, 34.625, 34.875])
    assert frame.forecast_cycle_utc.eq(pd.Timestamp('2026-08-01T00:00Z')).all()


def test_publication_rejects_changed_measurements_and_cross_farm_synthetic_labels(configuration):
    section = configuration['pv_curated_review']
    frame = classify_candidates(evidence_frame(), 1., section)
    treated = apply_target_policy(frame, 910, section, no_decisions())
    assert all(verify_prepared_frame(treated, 910, section).values())
    with pytest.raises(AssertionError):
        verify_prepared_frame(treated, 901, section)
    corrupted = treated.copy()
    corrupted.loc[0, 'measured_ac_mw'] = .9
    with pytest.raises(AssertionError):
        verify_prepared_frame(corrupted, 910, section)


def test_refresh_preserves_review_proposals_without_applying_them(tmp_path):
    episodes = pd.DataFrame([{'episode_id': 'farm_1_episode', 'decision': 'pending_review',
        'review_decision': 'pending_review', 'review_notes': '', 'synthetic_test_quarters': 0}])
    previous = episodes.copy()
    previous['review_decision'] = 'reconstruct'
    previous['review_notes'] = 'Needs approval after examining the plot.'
    previous.to_csv(tmp_path / 'review_episodes.csv', index=False)
    episodes.to_csv(tmp_path / 'pending_low_power_episodes.csv', index=False)
    result = preserve_review_notes(episodes, tmp_path)
    assert result.review_decision.iloc[0] == 'reconstruct'
    assert result.review_notes.iloc[0] == 'Needs approval after examining the plot.'
    assert result.decision.iloc[0] == 'pending_review'
    assert result.synthetic_test_quarters.iloc[0] == 0
