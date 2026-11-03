"""Apply explicit reviewed PV decisions to a separate, traceable target dataset.

The previous curated corpus is immutable. This module preserves earlier
exclusions unless the user requested restoration, and records every selected
quarter as measured, pvlib, explicit zero, or an unresolved missing value.
"""
from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from ...data import config_path
from .pv_curated_review import configure_worker_threads, contiguous_positions, sha256


def _local_minutes(times, timezone):
    """Evaluate a time-of-day rule on actual local interval starts, including DST folds."""
    local = pd.DatetimeIndex(times).tz_convert(timezone)
    return local.hour * 60 + local.minute


def _clock_minutes(value):
    """Validate a local clock boundary and convert it to minutes since midnight."""
    hour, minute = (int(part) for part in value.split(':'))
    if not 0 <= hour < 24 or not 0 <= minute < 60 or minute % 15:
        raise ValueError(f'Expected a quarter-hour local clock boundary: {value}')
    return hour * 60 + minute


def build_selection(frame, farm_id, section):
    """Translate reviewed days and clock ranges to unambiguous physical-quarter masks."""
    rules = section['selections'][str(farm_id)]
    if section['date_only_selection'] != 'low_power_candidate':
        raise ValueError('Date-only instructions must select the reviewed red intervals')
    if not section['keep_unlisted_prior_exclusions']:
        raise ValueError('The user requested preserving unlisted prior exclusions')
    days = frame.delivery_day.to_numpy()
    local_minutes = _local_minutes(frame.interval_start_utc, section['delivery_timezone'])
    action = np.full(len(frame), '', dtype=object)
    source = np.full(len(frame), '', dtype=object)
    restored = np.isin(days, rules.get('restore_days', []))
    rule_summary = []

    def mark(mask, replacement, kind, value):
        """Record one explicit rule; later zero rules override selected pvlib labels."""
        selected = np.asarray(mask, dtype=bool)
        if not selected.any():
            raise ValueError(f'Farm {farm_id}: no matching quarter-hours for {kind} {value}')
        action[selected] = replacement
        source[selected] = kind
        rule_summary.append({'farm_id': farm_id, 'rule': kind, 'value': value,
                             'quarters': int(selected.sum()), 'replacement': replacement})

    # Date-only requests select every red low-output interval on the requested date.
    for day in rules.get('episode_days', []):
        mark((days == day) & frame.low_power_candidate.to_numpy(), 'pvlib', 'episode_day', day)
    for day in rules.get('full_days', []):
        mark(days == day, 'pvlib', 'full_day', day)
    for month in rules.get('full_months', []):
        mark(np.char.startswith(days.astype(str), month), 'pvlib', 'full_month', month)
    for day, boundary in rules.get('replace_before_local', {}).items():
        mark((days == day) & (local_minutes < _clock_minutes(boundary)),
             'pvlib', 'before_local', f'{day} before {boundary}')
    if rules.get('replace_flagged_night_production', False):
        mark(frame.night_production_candidate.to_numpy(), 'pvlib', 'flagged_night_production', 'all history')
    # Explicit zero instructions take precedence when a nighttime rule overlaps.
    for day, boundary in rules.get('zero_from_local', {}).items():
        mark((days == day) & (local_minutes >= _clock_minutes(boundary)),
             'zero', 'zero_from_local', f'{day} from {boundary}')
    for day in rules.get('restore_days', []):
        selected = days == day
        if not selected.any():
            raise ValueError(f'Farm {farm_id}: restoration date {day} is absent from the source')
        rule_summary.append({'farm_id': farm_id, 'rule': 'restore_prior_exclusions', 'value': day,
                             'quarters': int(selected.sum()), 'replacement': 'measured_unless_selected'})
    return action, source, restored, rule_summary


def apply_reviewed_targets(frame, farm_id, section):
    """Construct split-specific training and scoring masks from reviewed target choices."""
    if not section['include_synthetic_labels_in_training']:
        raise ValueError('The user requested synthetic labels in all splits')
    result = frame.copy()
    action, source, restored, rules = build_selection(result, farm_id, section)
    previous_excluded = result.study_excluded.to_numpy(dtype=bool)
    retained_excluded = previous_excluded & ~restored
    selected = action != ''
    # A selected replacement supersedes old exclusions on its own quarters only.
    unresolved_old_exclusion = retained_excluded & ~selected
    pvlib_mask = action == 'pvlib'
    zero_mask = action == 'zero'
    night = result.night.to_numpy(dtype=bool)
    target = result.measured_ac_raw_mw.to_numpy(dtype=float).copy()
    reference = result.pvlib_historical_ghi_mw.to_numpy(dtype=float)
    target[pvlib_mask] = reference[pvlib_mask]
    # The user explicitly requested zero throughout the monthly night windows.
    target[pvlib_mask & night] = 0.0
    target[zero_mask] = 0.0
    source_kind = np.full(len(result), 'metered_ac', dtype=object)
    source_kind[pvlib_mask] = 'pvlib_historical_ghi'
    source_kind[pvlib_mask & night] = 'pvlib_monthly_night_zero'
    source_kind[zero_mask] = 'user_requested_zero'
    source_kind[pvlib_mask & ~night & ~np.isfinite(target)] = 'pvlib_unavailable'
    result['previous_study_excluded'] = previous_excluded
    result['previous_episode_excluded'] = result.episode_excluded
    result['previous_exclusion_restored'] = previous_excluded & restored & ~selected
    result['retained_prior_exclusion'] = unresolved_old_exclusion
    # The active exclusion mask applies to modeled labels; raw exclusions are tracked separately.
    result['episode_excluded'] = unresolved_old_exclusion
    result['study_excluded'] = unresolved_old_exclusion
    result['reviewed_action'] = action
    result['reviewed_rule'] = source
    result['raw_target_excluded'] = previous_excluded & ~restored | selected
    result['model_target_mw'] = target
    result['model_target_source'] = source_kind
    result['test_target_mw'] = target
    result['test_target_source'] = source_kind
    result['synthetic_target'] = selected
    result['synthetic_test_target'] = selected & result['split'].eq('test').to_numpy()
    # Historical GHI remains a required input; target substitution does not fill weather gaps.
    valid_inputs = (np.isfinite(result.ghi_w_m2) & result.ghi_w_m2.ge(0)
                    & np.isfinite(result.temperature_c)
                    & np.isfinite(result.wind_u_10m) & np.isfinite(result.wind_v_10m))
    valid_target = np.isfinite(result.model_target_mw) & result.model_target_mw.ge(0)
    result['eligible'] = (valid_inputs & valid_target & ~result.retained_prior_exclusion
                          & result['split'].isin(['train', 'validation', 'test']))
    result['training_eligible'] = result.eligible & result['split'].eq('train') & ~result.embargoed
    result['validation_eligible'] = (result.eligible & result['split'].eq('validation')
                                     & ~result.embargoed & result.evaluation_window)
    result['test_eligible'] = result.eligible & result['split'].eq('test') & result.evaluation_window
    verify_reviewed_targets(frame, result, farm_id, section)
    return result, rules


def verify_reviewed_targets(original, result, farm_id, section):
    """Check that requested edits cannot silently alter raw measurements or time boundaries."""
    pd.testing.assert_series_equal(original.interval_start_utc, result.interval_start_utc)
    pd.testing.assert_series_equal(original.measured_ac_raw_mw, result.measured_ac_raw_mw)
    pd.testing.assert_series_equal(original.ghi_raw_w_m2, result.ghi_raw_w_m2)
    np.testing.assert_array_equal(result.synthetic_target, result.reviewed_action.ne(''))
    unchanged = ~result.synthetic_target
    np.testing.assert_allclose(result.loc[unchanged, 'model_target_mw'],
                               result.loc[unchanged, 'measured_ac_raw_mw'], equal_nan=True)
    reference = result.reviewed_action.eq('pvlib') & ~result.night
    np.testing.assert_allclose(result.loc[reference, 'model_target_mw'],
                               result.loc[reference, 'pvlib_historical_ghi_mw'], equal_nan=True)
    night_synthetic = result.synthetic_target & result.night
    if not result.loc[night_synthetic, 'model_target_mw'].eq(0).all():
        raise ValueError(f'Farm {farm_id}: synthetic night target must be zero')
    if (result.training_eligible & result.retained_prior_exclusion).any():
        raise ValueError(f'Farm {farm_id}: an unchanged prior exclusion entered training')
    if ((result.validation_eligible | result.test_eligible) & ~result.evaluation_window).any():
        raise ValueError(f'Farm {farm_id}: scoring escaped the monthly window')


def decision_rows(frame, farm_id, timezone):
    """Export nonoverlapping exclusion runs with explicit replacement provenance."""
    category = np.where(frame.reviewed_action.ne(''), frame.reviewed_action,
                        np.where(frame.retained_prior_exclusion, 'prior_exclusion', ''))
    source = np.where(frame.reviewed_action.ne(''), frame.reviewed_rule, 'retained_prior_exclusion')
    combined = np.char.add(np.char.add(category.astype(str), '|'), source.astype(str))
    rows = []
    for value in np.unique(combined[category != '']):
        action, reason = value.split('|', 1)
        mask = combined == value
        for positions in contiguous_positions(frame.interval_start_utc, frame.delivery_day, mask):
            part = frame.iloc[positions]
            first = part.interval_start_utc.iloc[0]
            last = part.interval_start_utc.iloc[-1]
            rows.append({'farm_id': farm_id, 'start_utc': first, 'end_utc': last,
                'decision': 'exclude', 'reason': reason, 'replacement': action,
                'start_local': first.tz_convert(timezone), 'end_local': last.tz_convert(timezone),
                'delivery_day': part.delivery_day.iloc[0], 'quarters': len(part),
                'previous_excluded_quarters': int(part.previous_study_excluded.sum()),
                'unavailable_target_quarters': int(part.model_target_mw.isna().sum())})
    return rows


def prepare_reviewed_farm(job):
    """Apply one farm's user selections and write its independent dataset files."""
    farm_id, section, parent_manifest = job
    source_root = Path(section['input_root'])
    output_root = Path(section['output_root'])
    source_path = source_root / f'farm_{farm_id}' / 'corpus.parquet'
    if sha256(source_path) != parent_manifest['farms'][str(farm_id)]['corpus_sha256']:
        raise ValueError(f'Farm {farm_id}: prepared source checksum does not match the parent manifest')
    original = pd.read_parquet(source_path)
    result, rules = apply_reviewed_targets(original, farm_id, section)
    destination = output_root / f'farm_{farm_id}'
    destination.mkdir()
    corpus_path = destination / 'corpus.parquet'
    result.to_parquet(corpus_path, index=False)
    result.loc[result['split'].ne('outside')].to_parquet(destination / 'modeling_views.parquet', index=False)
    restored_days = []
    for day in section['selections'][str(farm_id)].get('restore_days', []):
        part = result.loc[result.delivery_day.eq(day)]
        restored_days.append({'farm_id': farm_id, 'delivery_day': day,
            'previously_excluded_quarters': int(part.previous_study_excluded.sum()),
            'restored_to_raw_quarters': int(part.previous_exclusion_restored.sum()),
            'new_synthetic_quarters': int(part.synthetic_target.sum())})
    summaries = []
    for split, part in result.groupby('split', sort=False):
        missing_ghi = ~np.isfinite(part.ghi_w_m2)
        summaries.append({'farm_id': farm_id, 'split': split, 'quarters': len(part),
            'pvlib_quarters': int(part.reviewed_action.eq('pvlib').sum()),
            'explicit_zero_quarters': int(part.reviewed_action.eq('zero').sum()),
            'night_window_zero_quarters': int(part.model_target_source.eq('pvlib_monthly_night_zero').sum()),
            'unavailable_pvlib_quarters': int(part.model_target_source.eq('pvlib_unavailable').sum()),
            'synthetic_missing_ghi_quarters': int((part.synthetic_target & missing_ghi).sum()),
            'retained_prior_exclusion_quarters': int(part.retained_prior_exclusion.sum()),
            'restored_prior_exclusion_quarters': int(part.previous_exclusion_restored.sum()),
            'training_quarters': int(part.training_eligible.sum()),
            'synthetic_training_quarters': int((part.training_eligible & part.synthetic_target).sum()),
            'night_training_quarters': int((part.training_eligible & part.night).sum()),
            'validation_quarters': int(part.validation_eligible.sum()),
            'synthetic_validation_quarters': int((part.validation_eligible & part.synthetic_target).sum()),
            'test_quarters': int(part.test_eligible.sum()),
            'synthetic_test_quarters': int((part.test_eligible & part.synthetic_target).sum())})
    print(f'Farm {farm_id}: {int(result.synthetic_target.sum())} reviewed synthetic quarters', flush=True)
    return {'farm_id': farm_id, 'decisions': decision_rows(result, farm_id, section['delivery_timezone']),
        'rules': rules, 'restored_days': restored_days, 'summaries': summaries,
        'fingerprints': {'parent_corpus_sha256': sha256(source_path), 'corpus_sha256': sha256(corpus_path)}}


def run_reviewed_preparation(config):
    """Compile reviewed decisions and write a fresh corpus without starting an experiment."""
    section = config['pv_reviewed_ghi']
    source_root = config_path(config, section['input_root'])
    output_root = config_path(config, section['output_root'])
    report_root = config_path(config, section['report_root'])
    decisions_path = config_path(config, section['output_decisions_csv'])
    for destination in [output_root, report_root, decisions_path]:
        if destination.exists():
            raise FileExistsError(f'Choose fresh output paths; already exists: {destination}')
    parent_manifest = json.loads((source_root / 'manifest.json').read_text())
    output_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    decisions = []
    all_rules = []
    summaries = []
    restored_days = []
    manifest = {'status': 'prepared', 'training_started': False,
        'parent_manifest_sha256': sha256(source_root / 'manifest.json'),
        'configuration': section, 'farms': {},
        'target_contract': {'measured': 'measured_ac_raw_mw', 'training_and_scoring': 'model_target_mw',
            'source': 'model_target_source', 'night_window': 'monthly policy; synthetic PV is zero at night',
            'training_mask': 'training_eligible', 'validation_mask': 'validation_eligible',
            'testing_mask': 'test_eligible',
            'raw_exclusion_mask': 'raw_target_excluded',
            'warning': 'Training readers must use model_target_mw and the supplied masks; reapplying raw-exclusion CSV as a training mask would remove the requested synthetic labels.',
            'limitation': 'Synthetic targets are physical-model estimates, not independent observations.'}}
    worker_section = dict(section)
    worker_section['input_root'] = str(source_root)
    worker_section['output_root'] = str(output_root)
    jobs = [(farm_id, worker_section, parent_manifest) for farm_id in section['farm_ids']]
    with ProcessPoolExecutor(max_workers=section['workers'], initializer=configure_worker_threads,
                             initargs=(section['threads_per_worker'],)) as pool:
        prepared = list(pool.map(prepare_reviewed_farm, jobs))
    for item in prepared:
        decisions.extend(item['decisions'])
        all_rules.extend(item['rules'])
        restored_days.extend(item['restored_days'])
        summaries.extend(item['summaries'])
        manifest['farms'][str(item['farm_id'])] = item['fingerprints']
    decision_table = pd.DataFrame(decisions).sort_values(['farm_id', 'start_utc', 'end_utc'])
    decision_table.to_csv(decisions_path, index=False)
    rules_table = pd.DataFrame(all_rules)
    summary_table = pd.DataFrame(summaries)
    restored_table = pd.DataFrame(restored_days)
    rules_table.to_csv(report_root / 'reviewed_rules.csv', index=False)
    summary_table.to_csv(report_root / 'split_target_summary.csv', index=False)
    restored_table.to_csv(report_root / 'restored_days.csv', index=False)
    manifest['decisions_csv'] = str(decisions_path)
    manifest['decisions_csv_sha256'] = sha256(decisions_path)
    manifest['preparation_code_sha256'] = sha256(Path(__file__))
    for destination in [output_root, report_root]:
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n')
    write_report(report_root, output_root, decisions_path, rules_table, summary_table, restored_table)
    print(f'Reviewed dataset: {output_root}', flush=True)


def write_report(root, dataset, decisions, rules, summary, restored):
    """Explain what each local-date instruction means and expose all affected intervals."""
    import os
    config_link = os.path.relpath('configs/pv_experiments/reviewed_ghi_user_decisions.yaml', root)
    decisions_link = os.path.relpath(decisions, root)
    lines = ['# User-reviewed historical-GHI PV targets', '',
        'Prepared data only; no model training or physical parameter fitting was performed.', '',
        f'[Original reviewed instructions and explicit date lists]({config_link}) · '
        f'[All exclusions and replacement intervals]({decisions_link})', '',
        '[Reviewed rules](reviewed_rules.csv) · [Counts by farm and split](split_target_summary.csv) · '
        '[Explicit restorations](restored_days.csv) · [Dataset provenance](manifest.json)', '',
        f'Curated corpus: `{dataset}`. Raw AC and GHI remain in separate unchanged columns.',
        '`model_target_mw` contains measured or reviewed synthetic values for all splits.',
        'Each source is identified in `model_target_source`; valid synthetic targets are included in training when requested.',
        'Training readers must use `model_target_mw` and the supplied split masks. The CSV marks excluded raw labels; it must not be reapplied as a blanket exclusion of synthetic targets.',
        'A full-day instruction covers every actual quarter-hour of that local day, including DST transitions.',
        'A date-only instruction covers red low-power candidate intervals; separate prior gray exclusions remain excluded.',
        'Prior exclusions remain in force unless the user explicitly requested their restoration or selected them for reconstruction.',
        'Before-time and from-time rules use Europe/Rome interval starts: before is exclusive; from is inclusive.',
        'Every pvlib replacement inside the month-specific night window is zero. Explicit zero rules also apply outside that window.',
        'Targets with unavailable daytime pvlib or missing weather remain missing/ineligible rather than being invented.',
        'Historical GHI and pvlib reconstructions are retrospective. Synthetic labels and scores are not independent measurements.', '',
        '## Per-farm summary', '',
        '| Farm | pvlib quarters | Explicit zero quarters | Prior exclusions retained | Prior quarters restored to raw | Synthetic training quarters | Synthetic validation quarters | Synthetic test quarters |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for farm_id, part in summary.groupby('farm_id'):
        totals = part.sum(numeric_only=True)
        lines.append(f'| {farm_id} | {int(totals.pvlib_quarters)} | {int(totals.explicit_zero_quarters)} | '
            f'{int(totals.retained_prior_exclusion_quarters)} | {int(totals.restored_prior_exclusion_quarters)} | '
            f'{int(totals.synthetic_training_quarters)} | {int(totals.synthetic_validation_quarters)} | '
            f'{int(totals.synthetic_test_quarters)} |')
    lines += ['', '## User-nominated rules', '',
        '| Farm | Rule | Local day, month or time | Quarters matched | Replacement |',
        '|---|---|---|---:|---|']
    for row in rules.itertuples(index=False):
        lines.append(f'| {row.farm_id} | {row.rule} | {row.value} | {row.quarters} | {row.replacement} |')
    lines += ['', '## Explicit restoration', '',
        '| Farm | Local day | Previously excluded | Restored to measured AC | New synthetic quarters |',
        '|---|---|---:|---:|---:|']
    for row in restored.itertuples(index=False):
        lines.append(f'| {row.farm_id} | {row.delivery_day} | {row.previously_excluded_quarters} | '
                     f'{row.restored_to_raw_quarters} | {row.new_synthetic_quarters} |')
    lines += ['', 'The CSV records final, nonoverlapping exclusion spans. End timestamps identify the final included quarter start.',
        'Original exclusion reasons remain in `configs/data_quality/pv_curated_ghi_exclusions_2025_to_2026_08.csv`.', '',
        'To reproduce using a fresh output path:', '', '```bash',
        'PYTHONPATH=src python scripts/prepare_reviewed_pv.py --config configs/pv_experiments/reviewed_ghi_user_decisions.yaml',
        '```']
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
