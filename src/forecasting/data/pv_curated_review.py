"""Prepare historical-GHI training data and review-only low-production evidence.

The original metered AC remains immutable. Only the explicitly authorized
farm/month receives a separate synthetic testing target. Other candidates do
not change eligibility, regardless of their severity.
"""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from ...data import config_path, load_metadata, night_window_mask, open_store
from .pv_outages import apply_episode_exclusions
from .pv_satellite_inputs import satellite_inputs
from .pv_study_inputs import farm_settings
from ..models.pv_physical import PhysicalPVModel


def configure_worker_threads(threads):
    """Bound native-library parallelism independently of the process worker count."""
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=int(threads))


def sha256(path):
    """Fingerprint a source or published artifact without relying on timestamps."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def split_labels(times, section):
    """Reuse the previous local-date split boundaries, retaining outside rows for audit."""
    days = pd.DatetimeIndex(times).tz_convert(section['delivery_timezone']).strftime('%Y-%m-%d')
    labels = np.select(
        [days < section['train_end_day_exclusive'], days < section['validation_end_day_exclusive']],
        ['train', 'validation'], default='test')
    outside = (days < section['start_day']) | (days >= section['end_day_exclusive'])
    labels[outside] = 'outside'
    return days, labels


def interpolate_ghi(times, values, splits):
    """Fill interior missing GHI within each split; never bridge negative observations."""
    raw = np.asarray(values, dtype=float)
    filled = raw.copy()
    # Negative observations form barriers so they cannot become interpolation endpoints.
    barriers = np.cumsum(np.isfinite(raw) & (raw < 0))
    keys = pd.DataFrame({'split': splits, 'barrier': barriers})
    for positions in keys.groupby(['split', 'barrier'], sort=False).indices.values():
        series = pd.Series(raw[positions], index=pd.DatetimeIndex(times)[positions])
        valid = series.where(np.isfinite(series) & (series >= 0))
        interpolated = valid.interpolate(method='time', limit_area='inside')
        allowed = ~np.isfinite(series) | (series >= 0)
        filled[positions] = interpolated.where(allowed).to_numpy()
    return filled, ~np.isfinite(raw) & np.isfinite(filled)


def contiguous_positions(times, days, mask, minimum=1):
    """Return native, uninterrupted quarter-hour runs within one local delivery day."""
    index = pd.DatetimeIndex(times)
    selected = np.flatnonzero(np.asarray(mask, dtype=bool))
    if not len(selected):
        return []
    boundaries = np.flatnonzero(
        (np.diff(selected) != 1)
        | (index[selected[1:]] - index[selected[:-1]] != pd.Timedelta(minutes=15))
        | (np.asarray(days)[selected[1:]] != np.asarray(days)[selected[:-1]])) + 1
    return [positions for positions in np.split(selected, boundaries) if len(positions) >= minimum]


def retained_runs(frame, mask, minimum):
    """Keep only quarters belonging to qualifying contiguous evidence runs."""
    result = np.zeros(len(frame), dtype=bool)
    for positions in contiguous_positions(frame.interval_start_utc, frame.delivery_day, mask, minimum):
        result[positions] = True
    return result


def reference_low_mask(frame, rating, settings):
    """Require observed high GHI, a large relative deficit and a material absolute gap."""
    actual = frame.measured_ac_raw_mw.to_numpy()
    reference = frame.pvlib_historical_ghi_mw.to_numpy()
    base = np.isfinite(actual) & (actual >= 0) & np.isfinite(reference)
    base &= frame.ghi_raw_w_m2.to_numpy() >= settings['minimum_ghi_w_m2']
    base &= ~frame.night.to_numpy()
    base &= reference >= rating * settings['minimum_reference_fraction_of_rating']
    base &= actual <= reference * settings['maximum_reference_ratio']
    base &= reference - actual >= rating * settings['minimum_gap_fraction_of_rating']
    return retained_runs(frame, base, int(settings['minimum_quarters']))


def classify_candidates(frame, rating, section):
    """Mark evidence without assigning faults, exclusions, or synthetic production."""
    result = frame.copy()
    actual = result.measured_ac_raw_mw.to_numpy()
    observed = result.ghi_raw_w_m2.to_numpy()
    finite = np.isfinite(actual) & (actual >= 0) & ~result.night.to_numpy()
    absolute = section['absolute_low_power']
    zero = section['zero_power']
    result['relative_low_power'] = reference_low_mask(result, rating, section['low_power'])
    result['absolute_low_power'] = retained_runs(result,
        finite & (observed >= absolute['minimum_ghi_w_m2'])
        & (actual <= rating * absolute['maximum_fraction_of_rating']),
        int(absolute['minimum_quarters']))
    result['zero_power_high_ghi'] = retained_runs(result,
        finite & (actual == 0) & (observed >= zero['minimum_ghi_w_m2']),
        int(zero['minimum_quarters']))
    result['extreme_low_power'] = reference_low_mask(result, rating, section['extreme_low_power'])
    result['low_power_candidate'] = result[
        ['relative_low_power', 'absolute_low_power', 'zero_power_high_ghi']].any(axis=1)
    result['night_production_candidate'] = (
        result.night & (result.measured_ac_raw_mw * 1000 > section['night_positive_threshold_kw']))
    return result


def exclusion_mask(frame, decisions, farm_id, reasons=None):
    """Resolve inclusive UTC quarter starts for one farm and optional reason subset."""
    mask = np.zeros(len(frame), dtype=bool)
    selected = decisions.loc[decisions.farm_id.astype(str).eq(str(farm_id))]
    if reasons is not None:
        selected = selected.loc[selected.reason.isin(reasons)]
    for row in selected.itertuples(index=False):
        if row.decision == 'exclude':
            mask |= frame.interval_start_utc.between(pd.Timestamp(row.start_utc), pd.Timestamp(row.end_utc))
    return mask


def apply_target_policy(frame, farm_id, section, decisions):
    """Keep raw training labels and isolate authorized synthetic targets to testing."""
    result = frame.copy()
    result['new_night_excluded'] = (
        result.night_production_candidate & (farm_id in section['exclude_positive_night_farms']))
    result['study_excluded'] = result.episode_excluded | result.new_night_excluded
    reconstruction = section['test_reconstruction']
    if reconstruction.get('apply_elsewhere', False):
        raise ValueError('Additional synthetic replacements require reviewed episode decisions')
    authorized = ((farm_id == reconstruction['farm_id']) & result['split'].eq('test')
                  & result.delivery_day.str.startswith(reconstruction['month']))
    approved_low = exclusion_mask(result, decisions, farm_id, reconstruction['approved_csv_reasons'])
    selected = approved_low.copy()
    if reconstruction['include_detected_low_power']:
        selected |= result.low_power_candidate.to_numpy()
    result['test_reconstruction_requested'] = authorized & selected
    finite_reference = np.isfinite(result.pvlib_historical_ghi_mw)
    result['synthetic_test_target'] = result.test_reconstruction_requested & finite_reference
    result['test_target_mw'] = result.measured_ac_raw_mw
    result.loc[result.synthetic_test_target, 'test_target_mw'] = result.pvlib_historical_ghi_mw
    result['test_target_source'] = np.where(result.synthetic_test_target,
        'pvlib_historical_ghi_counterfactual', 'metered_ac')
    # Independent exclusions continue to apply even when a low-output reason overlaps them.
    other_reasons = set(decisions.reason) - set(reconstruction['approved_csv_reasons'])
    other_excluded = exclusion_mask(result, decisions, farm_id, other_reasons)
    valid_target = np.isfinite(result.measured_ac_raw_mw) & (result.measured_ac_raw_mw >= 0)
    valid_inputs = (np.isfinite(result.ghi_w_m2) & (result.ghi_w_m2 >= 0)
                    & np.isfinite(result.temperature_c)
                    & np.isfinite(result.wind_u_10m) & np.isfinite(result.wind_v_10m))
    result['eligible'] = valid_target & valid_inputs & ~result.study_excluded
    in_scope = result['split'].isin(['train', 'validation', 'test'])
    result['eligible'] &= in_scope
    # Preserve the previous one-day availability embargo before each split boundary.
    result['embargoed'] = False
    for label, boundary in [('train', 'train_end_day_exclusive'),
                            ('validation', 'validation_end_day_exclusive')]:
        cutoff = (pd.Timestamp(section[boundary]) - pd.DateOffset(days=section['split_embargo_days']))
        cutoff = cutoff.tz_localize(section['delivery_timezone']).tz_convert('UTC')
        result.loc[result['split'].eq(label) & (result.target_available_at_utc > cutoff), 'embargoed'] = True
    result['training_eligible'] = result.eligible & result['split'].eq('train') & ~result.embargoed
    if not section['include_night_training']:
        result['training_eligible'] &= ~result.night
    result['evaluation_window'] = ~result.night
    result['validation_eligible'] = (
        result.eligible & result['split'].eq('validation') & ~result.embargoed & result.evaluation_window)
    test_valid = np.isfinite(result.test_target_mw) & (result.test_target_mw >= 0)
    retained_test = ~result.study_excluded | (result.synthetic_test_target & ~other_excluded)
    result['test_eligible'] = (result['split'].eq('test') & result.evaluation_window
                              & valid_inputs & test_valid & retained_test)
    return result


def align_weather(selected, times, section, config):
    """Vectorize the preceding-day 00 UTC view without mixing forecast vintages."""
    days, _ = split_labels(times, section)
    vintage = pd.DatetimeIndex(pd.to_datetime(days, utc=True)) - pd.Timedelta(days=1)
    cycles = pd.DatetimeIndex(pd.to_datetime(selected.cycle_time.values, utc=True))
    cycle_index = cycles.get_indexer(vintage)
    leads = pd.TimedeltaIndex(selected.lead_time.values) / pd.Timedelta(hours=1)
    midpoint_lead = (times + pd.Timedelta(minutes=7.5) - vintage) / pd.Timedelta(hours=1)
    frame = pd.DataFrame(index=times)
    frame['forecast_cycle_utc'] = vintage
    frame['forecast_available_at_utc'] = vintage + pd.Timedelta(hours=config['source']['availability_lag_hours'])
    frame['lead_hours'] = midpoint_lead
    for canonical in ['forecast_ghi', 'forecast_bhi', 'temperature_c', 'wind_u_10m', 'wind_v_10m']:
        raw = selected[config['forecast_mapping'][canonical]].transpose('cycle_time', 'lead_time').values
        kind = config['forecast_semantics'].get(canonical, config['forecast_semantics']['default'])
        lead = np.asarray(midpoint_lead)
        lower = np.ceil(lead) if kind == 'hourly_mean' else np.floor(lead)
        upper = lower if kind == 'hourly_mean' else lower + 1
        left = pd.Index(leads).get_indexer(lower)
        right = pd.Index(leads).get_indexer(upper)
        valid = (cycle_index >= 0) & (left >= 0) & (right >= 0)
        values = np.full(len(times), np.nan)
        fraction = np.zeros(len(times)) if kind == 'hourly_mean' else lead - lower
        values[valid] = (raw[cycle_index[valid], left[valid]] * (1 - fraction[valid])
                         + raw[cycle_index[valid], right[valid]] * fraction[valid])
        if canonical in config['policy']['nonnegative_forecasts']:
            values = np.maximum(values, 0)
        frame[canonical] = values
    return frame


def prepare_farm(config, farm_id):
    """Read raw observations once and evaluate the frozen physical model without fitting."""
    from threadpoolctl import threadpool_limits
    section = config['pv_curated_review']
    farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    settings = farm_settings(config, farm)
    with open_store(config) as store:
        selected = store.sel(farm_id=farm_id)
        times = pd.DatetimeIndex(pd.to_datetime(selected.time.values, utc=True))
        frame = align_weather(selected, times, section, config).reset_index(names='interval_start_utc')
        frame['measured_ac_raw_mw'] = np.asarray(selected.power.values, dtype=float) * float(config['forecast']['realized_power_scale_to_mw'])
        frame['ghi_raw_w_m2'] = np.asarray(selected.ghi.values, dtype=float)
        weather_digest = hashlib.sha256()
        for name in ['cycle_time', 'lead_time'] + [config['forecast_mapping'][canonical] for canonical in
                ['forecast_ghi', 'forecast_bhi', 'temperature_c', 'wind_u_10m', 'wind_v_10m']]:
            values = np.ascontiguousarray(selected[name].values)
            weather_digest.update(name.encode())
            weather_digest.update(str(values.dtype).encode())
            weather_digest.update(values.tobytes())
    frame['delivery_day'], frame['split'] = split_labels(times, section)
    frame['satellite_ghi_input_w_m2'], frame['ghi_interpolated'] = interpolate_ghi(
        times, frame.ghi_raw_w_m2, frame['split'])
    frame['night'] = night_window_mask(times, section['night_policy'])
    frame['target_available_at_utc'] = times + pd.Timedelta(
        minutes=15 + config['source']['telemetry_latency_minutes'])
    frame['nwp_ghi_w_m2'] = frame.forecast_ghi
    frame['nwp_bhi_w_m2'] = frame.forecast_bhi
    location = settings['location']
    with threadpool_limits(limits=section['threads_per_worker']):
        solar = pvlib.solarposition.get_solarposition(times + pd.Timedelta(minutes=7.5),
            location['latitude'], location['longitude'], altitude=farm.elevation_m)
        frame['solar_zenith_deg'] = solar.apparent_zenith.to_numpy()
        frame['solar_azimuth_deg'] = solar.azimuth.to_numpy()
        physical = satellite_inputs(frame, settings)
        physical['wind_speed_m_s'] = np.hypot(physical.wind_u_10m, physical.wind_v_10m)
        frame['pvlib_historical_ghi_mw'] = PhysicalPVModel(
            physical.set_index('interval_start_utc'), farm, settings, section['night_policy']).evaluate()
    frame['ghi_w_m2'] = physical.forecast_ghi
    frame['bhi_w_m2'] = physical.forecast_bhi
    frame['measured_ac_mw'] = frame.measured_ac_raw_mw
    frame['direct_fraction_erbs_fallback'] = physical.direct_fraction_erbs_fallback
    frame['eligible'] = True
    frame = apply_episode_exclusions(frame, config, farm_id=farm_id)
    decisions = pd.read_csv(config_path(config, config['pv_data_quality']['exclusions_csv']))
    frame = classify_candidates(frame, farm.ac_capacity_mw, section)
    frame = apply_target_policy(frame, farm_id, section, decisions)
    digest = hashlib.sha256()
    # Hash the actual raw values, not only Zarr metadata or modification times.
    for name in ['interval_start_utc', 'measured_ac_raw_mw', 'ghi_raw_w_m2']:
        digest.update(pd.util.hash_pandas_object(frame[name], index=False).to_numpy().tobytes())
    return farm_id, frame, settings, digest.hexdigest(), weather_digest.hexdigest()


def episode_table(frame, farm_id, rating, section):
    """Summarize all candidate episodes with editable decisions and raw-value evidence."""
    rows = []
    for kind in ['low_power', 'night_production']:
        mask = frame.low_power_candidate if kind == 'low_power' else frame.night_production_candidate
        for positions in contiguous_positions(frame.interval_start_utc, frame.delivery_day, mask):
            part = frame.iloc[positions]
            actual = part.measured_ac_raw_mw.to_numpy()
            reference = part.pvlib_historical_ghi_mw.to_numpy()
            finite = np.isfinite(reference) & (reference > 0)
            ratio = float(actual[finite].sum() / reference[finite].sum()) if finite.any() else np.nan
            extreme = bool(part.extreme_low_power.any() or part.zero_power_high_ghi.any())
            if kind == 'night_production':
                extreme = bool((actual >= rating * section['night_extreme_fraction_of_rating']).any())
            applied = bool(part.synthetic_test_target.any() or part.new_night_excluded.any())
            decision = 'pending_review'
            if part.study_excluded.all():
                decision = 'already_excluded'
            if applied:
                decision = 'authorized_scope_applied'
            finite_reference = np.isfinite(reference)
            rows.append({
                'episode_id': f'{farm_id}_{kind}_{part.interval_start_utc.iloc[0].strftime("%Y%m%dT%H%M")}',
                'farm_id': farm_id, 'kind': kind, 'split': part['split'].iloc[0],
                'delivery_day': part.delivery_day.iloc[0],
                'start_utc': part.interval_start_utc.iloc[0], 'end_utc': part.interval_start_utc.iloc[-1],
                'start_local': part.interval_start_utc.iloc[0].tz_convert(section['delivery_timezone']),
                'end_local': part.interval_start_utc.iloc[-1].tz_convert(section['delivery_timezone']),
                'quarters': len(part), 'duration_hours': len(part) / 4,
                'mean_measured_ac_mw': float(np.mean(actual)),
                'max_measured_ac_mw': float(np.max(actual)),
                'mean_pvlib_mw': float(np.nanmean(reference)) if finite_reference.any() else np.nan,
                'power_ratio': ratio, 'mean_ghi_w_m2': float(part.ghi_raw_w_m2.mean()),
                'indicative_gap_mwh': (float(np.nansum(np.maximum(reference - actual, 0)) / 4)
                                      if finite_reference.any() else np.nan),
                'relative_low_quarters': int(part.relative_low_power.sum()),
                'absolute_low_quarters': int(part.absolute_low_power.sum()),
                'zero_high_ghi_quarters': int(part.zero_power_high_ghi.sum()),
                'extreme_quarters': int(part.extreme_low_power.sum()),
                'priority': 'extreme' if extreme else 'suspicious',
                'previously_excluded_quarters': int(part.episode_excluded.sum()),
                'unexcluded_quarters': int((~part.study_excluded).sum()),
                'new_night_excluded_quarters': int(part.new_night_excluded.sum()),
                'synthetic_test_quarters': int(part.synthetic_test_target.sum()),
                'retained_training_quarters': int(part.training_eligible.sum()),
                'decision': decision,
                'review_decision': 'pending_review' if decision == 'pending_review' else '',
                'suggested_action': 'review_for_reconstruction' if extreme and kind == 'low_power' else 'review',
                'review_notes': '',
            })
    return pd.DataFrame(rows)


def preserve_review_notes(episodes, report):
    """Retain user-entered proposals and notes when regenerating the review tables."""
    result = episodes.copy()
    machine_decisions = {'pending_review', 'already_excluded', 'authorized_scope_applied'}
    # Focused review tables may carry edits after the full table was published.
    for name in ['review_episodes.csv', 'pending_review_episodes.csv', 'pending_low_power_episodes.csv']:
        path = Path(report) / name
        if not path.exists():
            continue
        previous = pd.read_csv(path, keep_default_na=False)
        for row in previous.to_dict('records'):
            selected = result.episode_id.eq(row['episode_id'])
            proposal = row.get('review_decision', '')
            if row.get('decision') not in machine_decisions:
                proposal = row.get('decision', proposal)
            if proposal and proposal != 'pending_review':
                result.loc[selected, 'review_decision'] = proposal
            if row.get('review_notes'):
                result.loc[selected, 'review_notes'] = row['review_notes']
    return result


def verify_prepared_frame(frame, farm_id, section):
    """Reject corrupted masks or accidental use of reconstructed values for fitting."""
    times = pd.DatetimeIndex(frame.interval_start_utc)
    if times.has_duplicates or not times.is_monotonic_increasing:
        raise ValueError('The curated corpus must contain ordered unique physical quarters')
    np.testing.assert_allclose(frame.measured_ac_mw, frame.measured_ac_raw_mw, equal_nan=True)
    observed = np.isfinite(frame.ghi_raw_w_m2) & frame.ghi_raw_w_m2.ge(0)
    np.testing.assert_allclose(frame.loc[observed, 'ghi_w_m2'], frame.loc[observed, 'ghi_raw_w_m2'])
    _, expected_splits = split_labels(times, section)
    np.testing.assert_array_equal(frame['split'], expected_splits)
    synthetic = frame.synthetic_test_target
    authorized = section['test_reconstruction']
    if synthetic.any():
        assert farm_id == authorized['farm_id']
        assert frame.loc[synthetic, 'split'].eq('test').all()
        assert frame.loc[synthetic, 'delivery_day'].str.startswith(authorized['month']).all()
    np.testing.assert_allclose(frame.loc[synthetic, 'test_target_mw'],
                               frame.loc[synthetic, 'pvlib_historical_ghi_mw'])
    np.testing.assert_allclose(frame.loc[~synthetic, 'test_target_mw'],
                               frame.loc[~synthetic, 'measured_ac_raw_mw'], equal_nan=True)
    assert not (frame.training_eligible & (frame.study_excluded | frame.embargoed | synthetic)).any()
    assert not ((frame.validation_eligible | frame.test_eligible) & frame.night).any()
    assert not (frame.training_eligible & ~frame['split'].eq('train')).any()
    if farm_id in section['exclude_positive_night_farms']:
        assert not (frame.training_eligible & frame.night_production_candidate).any()
    return {'physical_quarters_unique': True, 'raw_ac_preserved': True,
        'observed_ghi_preserved': True, 'previous_splits_preserved': True,
        'synthetic_targets_only_in_authorized_test_month': True,
        'training_masks_exclude_synthetic_and_excluded_targets': True,
        'evaluation_uses_month_specific_windows': True}


def plot_review_day(job):
    """Plot a complete local day with raw power, synthetic targets and shaded evidence."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    path, destination, farm_id, day, rating, timezone, dpi = job
    frame = pd.read_parquet(path)
    part = frame.loc[frame.delivery_day.eq(day)].copy()
    times = pd.DatetimeIndex(part.interval_start_utc).tz_convert(timezone)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 6), sharex=True, layout='constrained')
    top.plot(times, part.measured_ac_raw_mw, color='black', label='Raw metered AC')
    top.plot(times, part.pvlib_historical_ghi_mw, color='tab:blue', label='Frozen pvlib, historical GHI')
    synthetic = part.synthetic_test_target.to_numpy()
    top.scatter(times[synthetic], part.loc[synthetic, 'test_target_mw'], color='tab:green', s=13,
                label='Synthetic test target (authorized farm/month)')
    top.axhline(rating, color='gray', linestyle=':', label='Installed AC rating')
    for column, color, label in [
            ('episode_excluded', 'gray', 'Existing exclusions'),
            ('low_power_candidate', 'tab:red', 'Low-power review candidate'),
            ('night_production_candidate', 'tab:orange', 'Positive power in monthly night window')]:
        for number, positions in enumerate(contiguous_positions(times, part.delivery_day, part[column])):
            top.axvspan(times[positions[0]], times[positions[-1]] + pd.Timedelta(minutes=15),
                        color=color, alpha=.13, label=label if number == 0 else None)
    bottom.plot(times, part.ghi_w_m2, color='black', label='Historical GHI input')
    bottom.plot(times, part.nwp_ghi_w_m2, color='tab:orange', alpha=.65, label='D−1 00 UTC forecast GHI')
    filled = part.ghi_interpolated.to_numpy()
    bottom.scatter(times[filled], part.loc[filled, 'ghi_w_m2'], color='purple', s=14, label='Interpolated GHI')
    top.set(title=f'Farm {farm_id}, {day}: review evidence; causes unverified', ylabel='AC power (MW)')
    bottom.set(ylabel='GHI (W/m²)', xlabel=f'Local interval start ({timezone})')
    for axis in (top, bottom):
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, loc='upper right')
    fig.savefig(destination, dpi=dpi)
    plt.close(fig)


def run_review(config, *, reuse_prepared=False):
    """Materialize a new, non-overwriting review package and training-ready masked corpus."""
    section = config['pv_curated_review']
    if section['ghi_gap_fill'] != 'linear_inside_each_split':
        raise ValueError('GHI interpolation must remain within each previous split')
    if section['weather_cycle'] != 'preceding_day_00_utc':
        raise ValueError('This preparation requires the preceding-day 00 UTC weather cycle')
    boundaries = [pd.Timestamp(section[key]) for key in ['start_day', 'train_end_day_exclusive',
        'validation_end_day_exclusive', 'end_day_exclusive']]
    if any(left >= right for left, right in zip(boundaries, boundaries[1:])):
        raise ValueError('Training, validation and testing boundaries must be strictly ordered')
    output = config_path(config, section['output_root']) / section['run_id']
    dataset = config_path(config, section['dataset_root']) / section['run_id']
    report = config_path(config, section['report_root'])
    exclusion_path = config_path(config, section['exclusions_csv'])
    saved = None
    if reuse_prepared:
        saved = json.loads((dataset / 'manifest.json').read_text())
        # A redraw cannot silently reinterpret masks or change prepared source inputs.
        mask_settings = ['farm_ids', 'start_day', 'train_end_day_exclusive',
            'validation_end_day_exclusive', 'end_day_exclusive', 'delivery_timezone',
            'split_embargo_days', 'night_policy', 'include_night_training', 'ghi_gap_fill',
            'weather_cycle', 'exclude_positive_night_farms', 'night_positive_threshold_kw',
            'low_power', 'absolute_low_power', 'zero_power', 'extreme_low_power', 'test_reconstruction']
        if any(saved['configuration'][key] != section[key] for key in mask_settings):
            raise ValueError('Prepared masks disagree with configuration; choose a fresh run ID')
        quality_hash = sha256(config_path(config, config['pv_data_quality']['exclusions_csv']))
        if saved['source_quality_csv_sha256'] != quality_hash:
            raise ValueError('Source exclusions changed since dataset preparation')
        if saved['source_settings'] != config['source'] or saved['forecast_semantics'] != config['forecast_semantics']:
            raise ValueError('Forecast source settings changed since dataset preparation')
        metadata_hash = sha256(config_path(config, config['inputs']['metadata_parquet']))
        if saved['metadata_sha256'] != metadata_hash:
            raise ValueError('Farm metadata changed since dataset preparation')
    else:
        for destination in [output, dataset, report, exclusion_path]:
            if destination.exists():
                raise FileExistsError(f'Choose fresh output paths; already exists: {destination}')
    decisions = pd.read_csv(config_path(config, config['pv_data_quality']['exclusions_csv']))
    output.mkdir(parents=True, exist_ok=reuse_prepared)
    dataset.mkdir(parents=True, exist_ok=reuse_prepared)
    report.mkdir(parents=True, exist_ok=reuse_prepared)
    summaries = []
    episodes = []
    added_exclusions = []
    plot_jobs = []
    manifest = {'status': 'prepared_waiting_for_episode_review', 'training_started': False,
        'workers': section['workers'], 'source_zarr': config['inputs']['zarr'],
        'source_quality_csv_sha256': sha256(config_path(config, config['pv_data_quality']['exclusions_csv'])),
        'configuration': section, 'farms': {},
        'source_settings': config['source'], 'forecast_semantics': config['forecast_semantics'],
        'metadata_sha256': sha256(config_path(config, config['inputs']['metadata_parquet'])),
        'review_code_sha256': sha256(Path(__file__)),
        'training_contract': {
            'measured_target_column': 'measured_ac_raw_mw',
            'historical_ghi_column': 'ghi_w_m2',
            'raw_ghi_column': 'ghi_raw_w_m2',
            'weather_feature_columns': ['ghi_w_m2', 'temperature_c', 'wind_u_10m', 'wind_v_10m'],
            'deterministic_features': 'Local midpoint time and annual/month encodings; generated by the future trainer.',
            'training_mask': 'training_eligible',
            'early_stopping_mask': 'validation_eligible',
            'testing_mask': 'test_eligible',
            'testing_target_column': 'test_target_mw',
            'warning': 'Existing training readers must honor these masks; legacy daylight-only selection must not be reused.'},
        'reference': 'Frozen configured 2025 physical parameters; no new fits. Historical GHI; forecast BHI/GHI fraction, Erbs fallback; D−1 00 UTC temperature/wind.',
        'limitations': 'Retrospective curation and reused 2026 report period. Synthetic test values are pvlib estimates, not independent truth. Proxy coordinates and uncertain plant parameters limit low-power classification.'}
    if reuse_prepared:
        prepared = []
        for farm_id in section['farm_ids']:
            path = dataset / f'farm_{farm_id}' / 'corpus.parquet'
            previous = saved['farms'][str(farm_id)]
            if sha256(path) != previous['corpus_sha256']:
                raise ValueError(f'Prepared corpus checksum mismatch for farm {farm_id}')
            prepared.append((farm_id, pd.read_parquet(path), previous['settings'],
                             previous['raw_time_ac_ghi_sha256'], previous['source_weather_sha256']))
    else:
        with ProcessPoolExecutor(max_workers=section['workers'], initializer=configure_worker_threads,
                                 initargs=(section['threads_per_worker'],)) as pool:
            prepared = list(pool.map(prepare_farm, [config] * len(section['farm_ids']), section['farm_ids']))
    for farm_id, frame, settings, raw_digest, weather_digest in prepared:
        checks = verify_prepared_frame(frame, farm_id, section)
        rating = float(config['farms'][str(farm_id)]['ac_capacity_kw']) / 1000
        table = episode_table(frame, farm_id, rating, section)
        folder = dataset / f'farm_{farm_id}'
        folder.mkdir(exist_ok=reuse_prepared)
        path = folder / 'corpus.parquet'
        if not reuse_prepared:
            frame.to_parquet(path, index=False)
            # Create one physically deduplicated training table with measured labels only.
            frame.loc[frame['split'].ne('outside')].to_parquet(folder / 'modeling_views.parquet', index=False)
        for positions in contiguous_positions(frame.interval_start_utc, frame.delivery_day, frame.new_night_excluded):
            part = frame.iloc[positions]
            added_exclusions.append({'farm_id': farm_id, 'start_utc': part.interval_start_utc.iloc[0],
                'end_utc': part.interval_start_utc.iloc[-1], 'reason': 'night_production', 'decision': 'exclude'})
        # Prioritize newly retained candidates and severe episodes, then show all authorized August days.
        ranked = table.sort_values(['priority', 'retained_training_quarters', 'indicative_gap_mwh'],
                                   ascending=[True, False, False])
        chosen = list(ranked.delivery_day.drop_duplicates().head(section['plot_days_per_farm']))
        low_days = table.loc[table.kind.eq('low_power'), 'delivery_day'].unique()
        reconstruction_days = frame.loc[frame.synthetic_test_target, 'delivery_day'].unique()
        chosen = sorted(set(chosen) | set(reconstruction_days) | set(low_days))
        plots = output / f'farm_{farm_id}' / 'plots'
        plots.mkdir(parents=True, exist_ok=reuse_prepared)
        table['plot_path'] = [str(plots / f'{day}.png') if day in chosen else '' for day in table.delivery_day]
        for day in chosen:
            if not (plots / f'{day}.png').exists():
                plot_jobs.append((path, plots / f'{day}.png', farm_id, day, rating,
                                  section['delivery_timezone'], section['plot_dpi']))
        episodes.append(table)
        for label, part in frame.groupby('split', sort=False):
            summaries.append({'farm_id': farm_id, 'split': label, 'quarters': len(part),
                'night_quarters': int(part.night.sum()),
                'night_positive_quarters': int(part.night_production_candidate.sum()),
                'new_night_excluded_quarters': int((part.new_night_excluded & ~part.episode_excluded).sum()),
                'existing_excluded_quarters': int(part.episode_excluded.sum()),
                'low_power_quarters': int(part.low_power_candidate.sum()),
                'extreme_low_quarters': int(part.extreme_low_power.sum()),
                'zero_high_ghi_quarters': int(part.zero_power_high_ghi.sum()),
                'training_quarters': int(part.training_eligible.sum()),
                'night_training_quarters': int((part.training_eligible & part.night).sum()),
                'validation_quarters': int(part.validation_eligible.sum()),
                'testing_quarters': int(part.test_eligible.sum()),
                'synthetic_test_quarters': int(part.synthetic_test_target.sum()),
                'unresolved_test_reconstructions': int((part.test_reconstruction_requested & ~part.synthetic_test_target).sum()),
                'ghi_interpolated_quarters': int(part.ghi_interpolated.sum()),
                'ghi_missing_after_interpolation': int(part.ghi_w_m2.isna().sum())})
        manifest['farms'][str(farm_id)] = {'raw_time_ac_ghi_sha256': raw_digest,
            'source_weather_sha256': weather_digest,
            'corpus_sha256': sha256(path), 'settings': settings,
            'verification': checks,
            'raw_ac_unchanged': True, 'raw_zarr_unchanged': True}
        print(f'Farm {farm_id}: {len(table)} review episodes; {int(frame.synthetic_test_target.sum())} synthetic test quarters', flush=True)
    combined = pd.concat([decisions, pd.DataFrame(added_exclusions)], ignore_index=True).drop_duplicates()
    combined = combined.sort_values(['farm_id', 'start_utc'], key=lambda column: column.astype(str))
    combined.to_csv(exclusion_path, index=False)
    manifest['study_quality_csv'] = str(exclusion_path)
    manifest['study_quality_csv_sha256'] = sha256(exclusion_path)
    all_episodes = pd.concat(episodes, ignore_index=True)
    all_episodes = all_episodes.sort_values(['priority', 'indicative_gap_mwh'], ascending=[True, False])
    all_episodes = preserve_review_notes(all_episodes, report)
    summary = pd.DataFrame(summaries)
    night_rows = []
    for farm_id, frame, *_ in prepared:
        night = frame.loc[frame.night]
        positive = night.loc[night.night_production_candidate, 'measured_ac_raw_mw']
        night_rows.append({'farm_id': farm_id, 'night_quarters': len(night),
            'positive_quarters': len(positive),
            'positive_energy_mwh': float(positive.sum() / 4),
            'median_positive_kw': float(positive.median() * 1000),
            'max_positive_kw': float(positive.max() * 1000),
            'ghi_positive_night_quarters': int(night.ghi_raw_w_m2.gt(0).sum()),
            'ghi_missing_night_quarters': int(night.ghi_raw_w_m2.isna().sum())})
    for destination in [output, report]:
        all_episodes.to_csv(destination / 'review_episodes.csv', index=False)
        pending = all_episodes.loc[all_episodes.decision.eq('pending_review')]
        pending.to_csv(destination / 'pending_review_episodes.csv', index=False)
        pending.loc[pending.kind.eq('low_power')].to_csv(destination / 'pending_low_power_episodes.csv', index=False)
        reconstruction = pd.concat([frame.assign(farm_id=farm_id).loc[frame.synthetic_test_target,
            ['farm_id', 'interval_start_utc', 'measured_ac_raw_mw', 'test_target_mw',
             'ghi_raw_w_m2', 'ghi_interpolated', 'episode_excluded', 'test_eligible']]
            for farm_id, frame, *_ in prepared], ignore_index=True)
        reconstruction.to_csv(destination / 'synthetic_testing_intervals.csv', index=False)
        pd.DataFrame(night_rows).to_csv(destination / 'night_production_summary.csv', index=False)
        summary.to_csv(destination / 'split_summary.csv', index=False)
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    (dataset / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    with ProcessPoolExecutor(max_workers=section['workers'], initializer=configure_worker_threads,
                             initargs=(section['threads_per_worker'],)) as pool:
        list(pool.map(plot_review_day, plot_jobs))
    write_review_index(report, output, all_episodes, summary, section, dataset)
    print(f'Review index: {report / "README.md"}', flush=True)


def write_review_index(report, output, episodes, summary, section, dataset):
    """Publish review links and the exact distinction between training and testing targets."""
    import os
    relative = section['low_power']
    absolute = section['absolute_low_power']
    zero = section['zero_power']
    extreme = section['extreme_low_power']
    lines = ['# Full-history historical-GHI PV curation review', '',
        'Preparation only: no training or new parameter fitting was performed.', '',
        f"Configured compute: {section['workers']} CPU workers, one native-library thread per farm worker.", '',
        'Raw metered AC and raw GHI are preserved. Training uses all eligible measured quarters, including night.',
        'Farm 345 positive production in the configured monthly night windows is excluded; valid nighttime zeros remain.',
        'Validation and test masks use the complement of the month-specific night window, independently of proxy solar elevation.',
        'Splits remain 2025 train, January–April 2026 validation, May–August 2026 test, with the previous one-day embargo.',
        'Interior missing GHI is linearly interpolated within each split and flagged. Unbracketed gaps stay missing; AC is never interpolated.', '',
        '## Candidate definitions', '',
        f"- Relative low output: observed GHI >={relative['minimum_ghi_w_m2']:g} W/m², pvlib reference >={100*relative['minimum_reference_fraction_of_rating']:g}% of AC capacity, measured/reference <={relative['maximum_reference_ratio']:g}, gap >={100*relative['minimum_gap_fraction_of_rating']:g}% of capacity, for >={relative['minimum_quarters']} consecutive quarters.",
        f"- Absolute low output: observed GHI >={absolute['minimum_ghi_w_m2']:g} W/m² and measured AC <={100*absolute['maximum_fraction_of_rating']:g}% of capacity for >={absolute['minimum_quarters']} consecutive quarters.",
        f"- Zero/high-GHI episode: measured AC exactly zero and observed GHI >={zero['minimum_ghi_w_m2']:g} W/m² for >={zero['minimum_quarters']} consecutive quarters.",
        f"- Extreme relative deficit: observed GHI >={extreme['minimum_ghi_w_m2']:g} W/m², reference >={100*extreme['minimum_reference_fraction_of_rating']:g}% of capacity, measured/reference <={extreme['maximum_reference_ratio']:g}, gap >={100*extreme['minimum_gap_fraction_of_rating']:g}% of capacity, for >={extreme['minimum_quarters']} consecutive quarters.",
        f"- Positive nighttime production: raw AC >{section['night_positive_threshold_kw']:g} kW in the configured monthly night window. At least {100*section['night_extreme_fraction_of_rating']:g}% of capacity is highlighted as extreme nighttime evidence.",
        '- Runs stop at missing/failed evidence, a non-15-minute timestamp gap or a local-day boundary. Adjacent qualifying low-output flags are grouped in one review row.', '',
        'Episode end timestamps identify the final included quarter start, so the covered interval ends 15 minutes later.', '',
        'The old relative detector used a cross-fitted statistical median. This review uses the frozen configured pvlib model and performs no reference fitting.',
        'Candidates depend on model/proxy assumptions and are not confirmed faults or operator restrictions. Interpolated GHI does not trigger candidate flags.', '',
        '## Authorized changes and pending decisions', '',
        'Only farm 345 positive-night exclusions and farm 910 August test reconstruction are applied by this workflow.',
        'The farm 910 August mask is the union of existing approved low-output exclusion reasons and newly detected low-output runs.',
        'Its `test_target_mw` is replaced with finite satellite-GHI pvlib predictions. `measured_ac_raw_mw` is unchanged.',
        'Overlapping independent exclusions still prevent scoring. No other farm/month receives synthetic labels.',
        'Already excluded episodes are labeled `already_excluded`; other new candidates retain their current eligibility and are marked `pending_review`.',
        'Synthetic test scores measure agreement with a physical-model reconstruction; they are not independent measured-power validation.', '',
        f'Curated data: `{dataset}`. `training_eligible`, `validation_eligible`, and `test_eligible` are distinct masks.',
        'Input GHI is historical; temperature/wind and the direct/global fraction come from the preceding-day 00 UTC weather view, with Erbs fallback for undefined fractions.',
        'Forecast release time uses the configured six-hour assumption. Historical GHI is retrospective and is not an operational inference input.', '',
        '[Focused review with plot links](PENDING_REVIEW.md) · [Pending low-output decisions](pending_low_power_episodes.csv) · [All pending decisions](pending_review_episodes.csv) · [All episodes](review_episodes.csv)', '',
        '[Split/mask counts](split_summary.csv) · [Exact synthetic testing intervals](synthetic_testing_intervals.csv) · [Night production](night_production_summary.csv) · [Provenance](manifest.json)', '',
        '## Counts by farm', '',
        '| Farm | Low-output episodes | Extreme low/zero episodes | Pending low-output episodes | Night episodes | Training quarters | Night training quarters | Synthetic test quarters |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for farm_id in section['farm_ids']:
        table = episodes.loc[episodes.farm_id.eq(farm_id)]
        counts = summary.loc[summary.farm_id.eq(farm_id)]
        low = table.kind.eq('low_power')
        lines.append(f'| {farm_id} | {int(low.sum())} | {int((low & table.priority.eq("extreme")).sum())} | '
            f'{int((low & table.decision.eq("pending_review")).sum())} | {int(table.kind.eq("night_production").sum())} | '
            f'{int(counts.training_quarters.sum())} | {int(counts.night_training_quarters.sum())} | {int(counts.synthetic_test_quarters.sum())} |')
    lines += ['', '## Month-specific scoring windows', '',
        'Local quarter-hour starts in Europe/Rome. The evaluation window is half-open; all other times are night.', '',
        '| Month | Evaluation window | Night window |', '|---|---|---|']
    for month, window in section['night_policy']['monthly_windows'].items():
        lines.append(f"| {month} | {window['end']}–{window['start']} | {window['start']}–{window['end']} |")
    lines += ['', '## Linked review days', '',
        'Plots show raw AC (black), frozen historical-GHI pvlib (blue), synthetic testing values (green),',
        'existing exclusions (gray), low-output candidates (red), and positive night observations (orange).',
        'The lower panel shows historical and forecast GHI, with interpolated observations marked in purple.',
        'Every low-output day has a linked plot. Additional plots cover prioritized night candidates and reconstructed August days.', '',
        '| Farm | Day | Episodes | Pending | Plot |', '|---|---|---:|---:|---|']
    for (farm_id, day), group in episodes.groupby(['farm_id', 'delivery_day'], sort=True):
        paths = group.loc[group.plot_path.ne(''), 'plot_path']
        link = f'[Open]({os.path.relpath(paths.iloc[0], report)})' if len(paths) else 'CSV evidence'
        lines.append(f'| {farm_id} | {day} | {len(group)} | {int(group.decision.eq("pending_review").sum())} | {link} |')
    lines += ['', '## Reproduce', '',
        'Create a fresh package using the configured unused output paths:', '',
        '```bash',
        'PYTHONPATH=src python scripts/review_curated_pv.py --config configs/pv_experiments/curated_ghi_review.yaml',
        '```', '', 'Refresh this existing fingerprint-checked review without fitting models:', '',
        '```bash',
        'PYTHONPATH=src python scripts/review_curated_pv.py --config configs/pv_experiments/curated_ghi_review.yaml --refresh-review',
        '```', '',
        'The review preserves user-entered `review_decision` and `review_notes` fields across refreshes.',
        'Review proposals are not automatically applied to targets or exclusions.']
    (report / 'README.md').write_text('\n'.join(lines) + '\n')
    pending = episodes.loc[episodes.decision.eq('pending_review') & episodes.kind.eq('low_power')]
    focus = ['# Pending low-output episode review', '',
        'These episodes contain quarters that remain outside the approved exclusion mask.',
        'Previously excluded quarters within each episode remain excluded. No new synthetic values were applied here.',
        'Review the raw AC, historical GHI and frozen pvlib reference before deciding whether to exclude or reconstruct.',
        'A model deficit can also reflect model/geometry errors; it is not proof of an operator restriction.', '',
        'Record `keep`, `exclude`, or `reconstruct` in `review_decision` and explain the choice in `review_notes`.',
        'The machine `decision` column records what is already applied. Review proposals are preserved on refresh and do not change the corpus automatically.', '',
        '[Definitions and full-history summary](README.md) · [Editable decisions CSV](pending_low_power_episodes.csv)', '',
        '| Farm | Local day | Priority | Episode quarters | Still unexcluded | Training quarters | AC / pvlib | Plot |',
        '|---|---|---|---:|---:|---:|---:|---|']
    for row in pending.itertuples(index=False):
        link = f'[Open]({os.path.relpath(row.plot_path, report)})' if row.plot_path else 'CSV evidence'
        focus.append(f'| {row.farm_id} | {row.delivery_day} | {row.priority} | {row.quarters} | '
            f'{row.unexcluded_quarters} | {row.retained_training_quarters} | {row.power_ratio:.3f} | {link} |')
    (report / 'PENDING_REVIEW.md').write_text('\n'.join(focus) + '\n')
