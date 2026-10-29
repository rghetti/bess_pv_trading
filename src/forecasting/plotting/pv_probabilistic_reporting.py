"""Create paired PV baselines, scores, quantile artifacts and daily plots."""
from __future__ import annotations

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from ...data import config_path, load_metadata, night_window_mask
from ..data.pv_probabilistic_data import (file_hash, study_root, study_days, delivery_split,
                                          write_json, probabilistic_night_policy, validate_prepared_night_policy)
from ..evaluation.pv_probabilistic_metrics import paired_inputs, fit_calibrator, predict_calibrator, quantile_mean, score_frame


def vintage_description(section):
    """Describe issue-time semantics without implying retrospective inputs were available."""
    if section.get('vintage_strategy', 'preceding_day') == 'same_day_retrospective':
        return ('The same-day 00 UTC cycle is used retrospectively after 00 UTC, with the preceding-day cycle filling '
                'local midnight intervals before 00 UTC. Cycle availability is assumed six hours after issue, '
                'so same-day weather was not available at the DA or IDA1 gate for that delivery day. '
                'Stitched member labels are not proven to represent one trajectory across runs.')
    return ('The preceding-day 00 UTC cycle is used for the entire delivery day, with cycle availability '
            'assumed six hours after issue. Deterministic weather is held to the same 00 UTC cycle.')


def load_farm(config, farm_id):
    """Load checksum-matched inputs and configured farm AC rating."""
    root = study_root(config) / f'farm_{farm_id}'
    manifest = json.loads((root / 'input_manifest.json').read_text())
    validate_prepared_night_policy(manifest, config)
    for label in ('ensemble', 'deterministic'):
        if file_hash(root / f'{label}_inputs.parquet') != manifest['files'][label]:
            raise ValueError('Saved weather input checksum mismatch')
    ens = pd.read_parquet(root / 'ensemble_inputs.parquet')
    det = pd.read_parquet(root / 'deterministic_inputs.parquet')
    ids = sorted(map(str, ens.member.unique()), key=int)
    if len(ids) != 50 or det.interval_start_utc.duplicated().any():
        raise ValueError('Expected 50 explicit members and one target per interval')
    rating = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id].ac_capacity_mw
    return root, ens, det, ids, rating


def _phys_predictions(root, label, inputs):
    """Join modeled power by exact interval/member key, rejecting mismatches."""
    pv = pd.read_parquet(root / f'{label}_pvlib.parquet')
    keys = ['interval_start_utc', 'member']
    if pv.duplicated(keys).any():
        raise ValueError('Duplicate PV conversion keys')
    merged = inputs[keys].merge(pv[keys + ['pv_ac_mw']], on=keys, how='left', validate='one_to_one')
    return merged.pv_ac_mw.to_numpy(float)


def comparable(frame, ids):
    """Use identical finite member, deterministic and curated-target rows."""
    return frame.eligible.to_numpy(bool) & np.isfinite(frame.measured_ac_mw.to_numpy(float)) & \
        np.isfinite(frame.deterministic_prediction.to_numpy(float)) & np.isfinite(frame[ids].to_numpy(float)).all(axis=1)


def apply_calibrator(frame, ids, calibrator, levels, night_policy):
    """Calibrate weather-valid rows without consulting targets or exclusion decisions."""
    valid = np.isfinite(frame[ids].to_numpy(float)).all(axis=1)
    q = np.full((len(frame), len(levels)), np.nan)
    raw = q.copy()
    adjustments = {'crossing_rows': 0, 'bounded_values': 0, 'total_rows': 0}
    if valid.any():
        q[valid], adjustments, raw[valid] = predict_calibrator(calibrator, frame.loc[valid, ids].to_numpy(float),
                                                    frame.loc[valid, 'interval_start_utc'], return_raw=True)
    night = night_window_mask(pd.DatetimeIndex(frame.interval_start_utc), night_policy) & valid
    adjustments['nighttime_zero_rows'] = int(np.any(q[night] != 0, axis=1).sum())
    q[night] = 0.
    frame['calibrated_quantile_crossing'] = valid & np.any(np.diff(raw, axis=1) < 0, axis=1)
    frame['calibrated_out_of_bounds'] = valid & np.any((raw < 0) | (raw > calibrator['rating_mw']), axis=1)
    frame['calibrated_nighttime_zero'] = night
    for j, level in enumerate(levels):
        frame[f'calibrated_raw_q{round(100*level):02d}'] = raw[:,j]
        frame[f'calibrated_q{round(100*level):02d}'] = q[:,j]
    frame['calibrated_mean'] = quantile_mean(q, levels)
    return frame, adjustments


def scored_windows(frame, ids, levels, rating):
    """Score only the chronological partitions supplied by the caller."""
    tables=[]
    splits = set(frame.split.unique())
    overall = 'all_january' if splits >= {'train', 'validation'} and frame.delivery_day.str.startswith('2026-01').all() else (
        'all_available_validation' if splits == {'validation'} else 'all_available_test' if splits == {'test'} else 'all_available_study')
    partitions = [(overall, frame), ('train_in_sample', frame[frame.split.eq('train')]),
                  ('validation_model_selection', frame[frame.split.eq('validation')]), ('test_held_out', frame[frame.split.eq('test')])]
    for name, subset in partitions:
        if len(subset): tables.append(score_frame(subset, ids, levels, rating, name))
    return pd.concat(tables, ignore_index=True)


def hourly_frame(frame, ids):
    """Aggregate four paired quarter-hour trajectories before hourly scoring."""
    valid = comparable(frame, ids)
    x = frame.loc[valid].copy()
    x['hour_start_utc'] = pd.DatetimeIndex(x.interval_start_utc).floor('h')
    x['_one'] = 1
    columns = ids + ['measured_ac_mw', 'deterministic_prediction']
    if 'calibrated_mean' in x: columns.append('calibrated_mean')
    grouped = x.groupby(['delivery_day','split','hour_start_utc'], sort=True)
    out = grouped[columns].mean().reset_index()
    out['quarters'] = grouped['_one'].sum().to_numpy()
    out = out.loc[out.quarters.eq(4)].copy()
    out['interval_start_utc'] = out.hour_start_utc
    out['eligible'] = True; out['episode_excluded'] = False
    return out


def _hourly_scores(frame, ids, levels, rating):
    """Score hourly member trajectories; marginal quantiles are not averaged as trajectories."""
    hour = hourly_frame(frame, ids)
    results = scored_windows(hour, ids, levels, rating)
    if 'calibrated_mean' in hour:
        extra = []
        splits = set(hour.split.unique())
        overall = 'all_january' if splits >= {'train','validation'} and hour.delivery_day.str.startswith('2026-01').all() else (
            'all_available_validation' if splits == {'validation'} else 'all_available_test' if splits == {'test'} else 'all_available_study')
        for name, subset in [(overall, hour), ('train_in_sample', hour[hour.split.eq('train')]),
                             ('validation_model_selection', hour[hour.split.eq('validation')]), ('test_held_out', hour[hour.split.eq('test')])]:
            if not len(subset): continue
            valid = np.isfinite(subset.calibrated_mean) & np.isfinite(subset.measured_ac_mw)
            err = subset.loc[valid, 'calibrated_mean'].to_numpy()-subset.loc[valid, 'measured_ac_mw'].to_numpy()
            extra.append({'window': name, 'method': 'calibrated_mean', 'n': len(err),
                          'mae_mw': float(np.mean(abs(err))) if len(err) else None,
                          'rmse_mw': float(np.sqrt(np.mean(err**2))) if len(err) else None,
                          'bias_mw': float(np.mean(err)) if len(err) else None,
                          'nmae_percent_capacity': float(100*np.mean(abs(err))/rating) if len(err) else None,
                          'nrmse_percent_capacity': float(100*np.sqrt(np.mean(err**2))/rating) if len(err) else None,
                          'bias_percent_capacity': float(100*np.mean(err)/rating) if len(err) else None})
        results = pd.concat([results, pd.DataFrame(extra)], ignore_index=True)
    return hour, results


def daily_plot(frame, ids, levels, destination, title):
    """Plot target, same-cycle deterministic and ensemble paths for one local day."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    f = frame.sort_values('interval_start_utc')
    t = pd.DatetimeIndex(f.interval_start_utc).tz_convert('Europe/Rome')
    m = f[ids].to_numpy(float)
    finite_rows = np.isfinite(m).any(axis=1)
    q10 = np.full(len(m), np.nan); median = q10.copy(); q90 = q10.copy(); mean = q10.copy()
    if finite_rows.any():
        q10[finite_rows], median[finite_rows], q90[finite_rows] = np.nanquantile(m[finite_rows], [0.1, 0.5, 0.9], axis=1)
        mean[finite_rows] = np.nanmean(m[finite_rows], axis=1)
    fig, ax = plt.subplots(figsize=(13, 5), layout='constrained')
    ax.fill_between(t, q10, q90, color='steelblue', alpha=.18, label='Raw members p10–p90')
    ax.plot(t, mean, color='#14639a', label='Raw member mean')
    ax.plot(t, median, color='#5f9ec5', linestyle='--', label='Raw member median')
    ax.plot(t, f.deterministic_prediction, color='#e17c05', label='Deterministic weather')
    eligible = f.eligible.to_numpy(bool)
    ax.plot(t, np.where(eligible, f.measured_ac_mw, np.nan), color='black', linewidth=1.7, label='Curated measured AC')
    excluded = f.episode_excluded.to_numpy(bool)
    if excluded.any(): ax.scatter(t[excluded], f.measured_ac_mw.to_numpy()[excluded], marker='x', color='red', label='Reviewed exclusion')
    if {'calibrated_q10', 'calibrated_q90'} <= set(f):
        ax.fill_between(t, f.calibrated_q10, f.calibrated_q90, color='#d86dbe', alpha=.16, label='Calibrated p10–p90')
        ax.plot(t, f.calibrated_q50, color='#a43787', linestyle=':', label='Calibrated median')
    ax.set_title(title); ax.set_ylabel('AC power (MW)'); ax.set_xlabel('Local time (Europe/Rome)')
    ax.grid(alpha=.2); ax.legend(ncol=3, fontsize=8); fig.savefig(destination, dpi=140); plt.close(fig)


def evaluate_pvlib(config, farm_id):
    """Calibrate and report the prior-parameter physical baseline."""
    section = config['pv_probabilistic_study']; levels = list(map(float, section['quantiles']))
    root, ens, det, ids, rating = load_farm(config, farm_id)
    marker = json.loads((root / 'pvlib_manifest.json').read_text())
    for label in ('ensemble', 'deterministic'):
        if file_hash(root / f'{label}_pvlib.parquet') != marker['files'][label]: raise ValueError('pvlib checksum mismatch')
    frame = paired_inputs(ens, det, _phys_predictions(root, 'ensemble', ens), _phys_predictions(root, 'deterministic', det), ids)
    valid = comparable(frame, ids)
    calibration_dates = np.zeros(len(frame), dtype=bool)
    for fold in section['calibration_folds']:
        calibration_dates |= frame.delivery_day.between(fold['predict_start'], fold['predict_end']).to_numpy()
    calibration = valid & calibration_dates & frame.split.eq('train').to_numpy()
    calibrator = fit_calibrator(frame.loc[calibration, ids].to_numpy(float), frame.loc[calibration, 'measured_ac_mw'],
        frame.loc[calibration, 'interval_start_utc'], rating, levels, section['calibration_alpha'])
    frame, adjustments = apply_calibrator(frame, ids, calibrator, levels, probabilistic_night_policy(config))
    # Keep held-out targets out of baseline reports until validation selections are frozen.
    stored = frame.copy()
    if section.get('validation_end_day'):
        stored.loc[stored.split.eq('test'), 'measured_ac_mw'] = np.nan
        stored.loc[stored.split.eq('test'), 'eligible'] = False
    stored.to_parquet(root / 'pvlib_paired_predictions.parquet', index=False)
    joblib.dump(calibrator, root / 'pvlib_quantile_calibrator.joblib')
    visible = frame.loc[~frame.split.eq('test')] if section.get('validation_end_day') else frame
    scored_windows(visible, ids, levels, rating).to_csv(root / 'pvlib_quarter_hour_metrics.csv', index=False)
    pd.concat([score_frame(part, ids, levels, rating, day) for day, part in visible.groupby('delivery_day')],
              ignore_index=True).to_csv(root / 'pvlib_daily_metrics.csv', index=False)
    hourly, metrics = _hourly_scores(visible, ids, levels, rating)
    hourly.to_parquet(root / 'pvlib_hourly_predictions.parquet', index=False)
    metrics.to_csv(root / 'pvlib_hourly_metrics.csv', index=False)
    for day, part in visible.groupby('delivery_day'):
        daily_plot(part, ids, levels, root / 'plots' / day / 'pvlib.png', f'Farm {farm_id} — {day} — farm-parameter pvlib')
    visible_valid = comparable(visible, ids)
    coverage = {'all_intervals': len(visible), 'paired_eligible_intervals': int(visible_valid.sum()),
        'excluded_target_intervals': int(visible.episode_excluded.sum()), 'missing_or_invalid_forecast_or_target': int((~visible_valid & ~visible.episode_excluded).sum()),
        'calibration_rows': int(calibration.sum()), 'calibration_adjustments': adjustments,
        'calibration_training_scores': 'in-sample; pvlib parameters treated as prior metadata',
        'heldout_test_targets_masked_until_selection': bool(section.get('validation_end_day')),
        'hourly_quantiles': 'not produced: marginal quarter-hour quantiles are not joint trajectories'}
    write_json(root / 'pvlib_evaluation_manifest.json', coverage)
    print(f'Farm {farm_id}: pvlib scores and daily plots saved', flush=True)
    return coverage


def write_pvlib_overview(config):
    """Publish the physical baseline even when the later search is still running."""
    root = study_root(config); frames = []
    for farm in config['pv_probabilistic_study']['farm_ids']:
        path = root / f'farm_{farm}' / 'pvlib_quarter_hour_metrics.csv'
        if path.exists():
            values = pd.read_csv(path); values.insert(0, 'farm_id', int(farm)); frames.append(values)
    if not frames: raise ValueError('No physical baseline scores are available')
    table = pd.concat(frames, ignore_index=True)
    table.to_csv(root / 'pvlib_all_farms_quarter_hour_metrics.csv', index=False)
    rows = table.loc[table.window.eq('validation_model_selection') & table.method.isin(
        ['deterministic', 'member_mean', 'member_median', 'calibrated_mean', 'calibrated_median'])]
    section = config['pv_probabilistic_study']
    split_description = (f"Training ends {section['train_end_day']}; validation ends {section['validation_end_day']}; "
                         f"test ends {section['end_day']} and is withheld until validation selection." if section.get('validation_end_day') else
                         f"Training ends {section['train_end_day']}; validation ends {section['end_day']}; there is no independent test.")
    lines = ['# Physical PV ensemble baseline', '',
        'The same farm-specific 2025 parameters are treated as prior metadata for both weather sources. ' + vintage_description(section), '',
        split_description + ' Calibration uses the configured training-only folds. '
        'The first validation day is retrospective at its historical day-ahead gate. January 1 has a missing first local hour.', '',
        '| Farm | Prediction | Quarters | MAE (MW) | MAE (% AC) | RMSE (MW) | RMSE (% AC) | Bias (% AC) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows.itertuples():
        lines.append(f'| {row.farm_id} | {row.method} | {int(row.n)} | {row.mae_mw:.5f} | {row.nmae_percent_capacity:.2f} | {row.rmse_mw:.5f} | {row.nrmse_percent_capacity:.2f} | {row.bias_percent_capacity:.2f} |')
    lines.extend(['', 'The raw ensemble uses equal empirical member weights; this is not a demonstrated calibrated probability distribution. '
        'All percent errors use the farm AC nameplate as denominator. Calibrated curves are marginal quantiles, not calibrated joint trajectories. '
        'The calibrated mean integrates a piecewise linear quantile curve with constant tails.', '',
        'Per-farm folders contain all-study and validation metrics, daily plots, paired predictions, '
        'coverage and quantile-correction counts. Training-period calibration scores are in-sample.'])
    (root / 'README_pvlib.md').write_text('\n'.join(lines) + '\n')


def report_search(config):
    """Gather complete trials, select validation rankings and render winner plots."""
    section = config['pv_probabilistic_study']; levels = list(map(float, section['quantiles']))
    root = study_root(config); summaries=[]; ratings={}
    for farm in section['farm_ids']:
        farm_root, ens, det, ids, rating = load_farm(config, int(farm))
        ratings[int(farm)] = rating
        for result in sorted((farm_root / 'trials').glob('*/result.json')):
            item = json.loads(result.read_text())
            if item.get('status') != 'complete': continue
            for metric in item['metrics']:
                if metric['window'] == 'validation_model_selection':
                    summaries.append({'farm_id': int(farm), 'trial': item['trial']['name'],
                                      'family': item['trial']['family'], **metric, 'trial_path': str(result.parent)})
    if not summaries: raise ValueError('No completed search trials to report')
    table = pd.DataFrame(summaries); table.to_csv(root / 'search_metrics.csv', index=False)
    rankings = []
    for farm, group in table.groupby('farm_id'):
        for method, metric in [('member_mean', 'rmse_mw'), ('member_median', 'mae_mw'),
                               ('calibrated_quantiles', 'crps_mw')]:
            candidates = group.loc[group.method.eq(method) & group[metric].notna()].sort_values(metric)
            for rank, row in enumerate(candidates.itertuples(), 1):
                rankings.append({'farm_id': int(farm), 'objective': metric, 'method': method,
                                 'rank': rank, 'trial': row.trial, 'family': row.family,
                                 'value_mw': getattr(row, metric),
                                 'value_percent_capacity': 100*getattr(row, metric)/ratings[int(farm)],
                                 'trial_path': row.trial_path})
        farm_rank = pd.DataFrame([x for x in rankings if x['farm_id'] == int(farm)])
        farm_rank.to_csv(root / f'farm_{farm}' / 'validation_rankings.csv', index=False)
        for trial_path in farm_rank.loc[farm_rank['rank'].eq(1), 'trial_path'].drop_duplicates():
            trial_path = Path(trial_path); p = pd.read_parquet(trial_path / 'predictions.parquet')
            for day, part in p.groupby('delivery_day'):
                daily_plot(part, ids, levels, root / f'farm_{farm}' / 'plots' / day / f'{trial_path.name}.png',
                    f'Farm {farm} — {day} — {trial_path.name} (validation-selected)')
    pd.DataFrame(rankings).to_csv(root / 'validation_rankings.csv', index=False)
    from ..training.pv_probabilistic_search import trial_plan
    completed = int(table[['farm_id','trial']].drop_duplicates().shape[0])
    expected = len(trial_plan(section))
    status = 'complete' if completed == expected else 'partial'
    days = study_days(section)
    train_days = [x for x in days if delivery_split(x, section) == 'train']
    validation_days = [x for x in days if delivery_split(x, section) == 'validation']
    if not train_days or not validation_days: raise ValueError('Training and validation both need delivery days')
    validation_start = validation_days[0].strftime('%Y-%m-%d')
    write_json(root / 'report_manifest.json', {'status': status, 'completed_trials': completed, 'expected_trials': expected,
        'training_days': len(train_days), 'validation_days': len(validation_days), 'test_days': 0,
        'selection': f'{validation_start} through {section["end_day"]} validation; selected scores are exploratory, no independent test',
        'first_validation_day': 'retrospective because the preceding full-day targets were unavailable at the DA gate',
        'calibration': {'folds': section['calibration_folds'], 'output': 'quarter-hour marginal quantiles'},
        'hourly_quantiles': 'not reported; marginal quantiles do not define joint hourly trajectories'})
    ranking = pd.DataFrame(rankings)
    winners = ranking.loc[ranking['rank'].eq(1)]
    lines = ['# Probabilistic PV ensemble study', '',
        f'Status: **{status}**, {completed}/{expected} model configurations complete.', '',
        f'{len(train_days)} delivery days through {section["train_end_day"]} are training; '
        f'{len(validation_days)} days from {validation_start} through {section["end_day"]} are validation used for model selection. '
        'There is no independent test. The first validation day uses completed prior-day targets retrospectively after its day-ahead gate.', '',
        'Declared forecast delivery windows: ' + '; '.join(
            f"{period['start_day']} through {period['end_day']}" for period in section.get('dataset_periods') or [section]) + '.', '',
        vintage_description(section) + ' January 1 has missing instantaneous weather in its first local hour. '
        'Source NWP identity and member weights are unspecified.', '',
        'Raw member mean, median and quantiles summarize 50 member-wise PV paths; equal empirical weights do not establish calibrated probabilities. '
        'Linear quantile models use the configured chronological out-of-fold predictions from training only. '
        'Calibration metrics on the training period are in-sample, and validation-selected results are exploratory.', '',
        'Calibrated quantiles are sorted and bounded; raw quantiles and adjustment flags are saved. '
        'The calibrated mean integrates a piecewise linear quantile curve with constant tails. '
        'Hourly quantiles are not calculated from marginal quarter-hour quantiles.', '',
        '| Farm | Objective | Selected candidate | Score (MW) | Score (% AC) |', '| --- | --- | --- | ---: | ---: |']
    for row in winners.itertuples():
        lines.append(f'| {row.farm_id} | {row.method} / {row.objective} | {row.trial} | {row.value_mw:.5f} | {row.value_percent_capacity:.2f} |')
    lines.extend(['', 'Each farm folder contains pvlib metrics, paired predictions, calibration diagnostics, '
        'daily plots and per-trial outputs. `validation_rankings.csv` records every candidate. '
        'Quarter-hour CRPS for the raw ensemble is exact; calibrated quantile CRPS is numerical quadrature.'])
    lines.extend(['', '## Physical baseline on the validation period', '',
        '| Farm | Prediction | Quarters | MAE (MW) | MAE (% AC) | RMSE (MW) | RMSE (% AC) |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |'])
    for farm in section['farm_ids']:
        path = root / f'farm_{farm}' / 'pvlib_quarter_hour_metrics.csv'
        if not path.exists(): continue
        baseline = pd.read_csv(path)
        baseline = baseline.loc[baseline.window.eq('validation_model_selection') & baseline.method.isin(
            ['deterministic', 'member_mean', 'member_median', 'calibrated_mean', 'calibrated_median'])]
        for item in baseline.itertuples():
            lines.append(f'| {farm} | {item.method} | {int(item.n)} | {item.mae_mw:.5f} | {item.nmae_percent_capacity:.2f} | {item.rmse_mw:.5f} | {item.nrmse_percent_capacity:.2f} |')
    (root / 'README.md').write_text('\n'.join(lines) + '\n')
    print(f'Search rankings and selected-model plots: {root}', flush=True)
