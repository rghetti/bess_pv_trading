"""Fit and evaluate Mayer-style marginal PV forecasts on hourly ensemble inputs."""
from __future__ import annotations

import json
import os
import hashlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import subprocess
import tempfile
import warnings

import joblib
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.stats import norm
from sklearn.linear_model import QuantileRegressor
from scipy.optimize import OptimizeWarning

from ...data import config_path
from ...markets import AuctionCalendar
from ..data.pv_probabilistic_data import file_hash, write_json
from ..evaluation.pv_probabilistic_metrics import quantile_crps, raw_crps
from .pv_mayer_data import check_coverage, prepare_farm


def censored_normal_quantiles(location, scale, levels):
    """Quantiles of clamp(N(location,scale), 0, 1), including boundary atoms."""
    location = np.asarray(location, float); scale = np.asarray(scale, float)
    if np.any(~np.isfinite(location) | ~np.isfinite(scale) | (scale <= 0)):
        raise ValueError('Censored normal needs finite location and positive scale')
    return np.clip(location[:, None] + scale[:, None] * norm.ppf(np.asarray(levels, float))[None, :], 0, 1)


def censored_normal_mean(location, scale):
    """Analytical mean of a Gaussian censored at zero and one."""
    location = np.asarray(location, float); scale = np.asarray(scale, float)
    a, b = -location / scale, (1 - location) / scale
    return location * (ndtr(b) - ndtr(a)) + scale * (norm.pdf(a) - norm.pdf(b)) + 1 - ndtr(b)


def r_models(train, test, destination, rscript):
    """Fit separate-lead CRPS EMOS and pooled likelihood-boosted EMOS via R."""
    runner = Path(__file__).with_name('pv_mayer_fit.R')
    destination = Path(destination); destination.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='pv_mayer_r_') as temporary:
        fit_path = Path(temporary) / 'train.csv'; test_path = Path(temporary) / 'test.csv'
        train.to_csv(fit_path, index=False); test.to_csv(test_path, index=False)
        result = subprocess.run([str(rscript), str(runner), str(fit_path), str(test_path), str(destination)],
                                capture_output=True, text=True, check=False)
        (destination / 'r_stdout.txt').write_text(result.stdout)
        (destination / 'r_stderr.txt').write_text(result.stderr)
        if result.returncode:
            raise RuntimeError(f'R fitting failed ({result.returncode}): {result.stderr[-2000:]}')
    return pd.read_csv(destination / 'predictions.csv'), pd.read_csv(destination / 'fit_status.csv')


def fit_lqr(train_members, train_control, train_targets, test_members, test_control, levels):
    """Fit unpenalized quantile regressions on proxy control and sorted members."""
    x = np.column_stack([train_control, np.sort(train_members, axis=1)])
    x_test = np.column_stack([test_control, np.sort(test_members, axis=1)])
    models = []
    predictions = []
    for level in levels:
        model = QuantileRegressor(quantile=float(level), alpha=0., solver='highs', solver_options={'threads': 1})
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected.*', category=OptimizeWarning)
            model.fit(x, train_targets)
        models.append(model); predictions.append(model.predict(x_test))
    return models, np.column_stack(predictions)


def scoring_rows(frame, members, levels, r_predictions, lqr_quantiles):
    """Score the same finite, valid-distribution rows across all four methods."""
    n = len(frame); y = frame.y.to_numpy(float); raw = frame[members].to_numpy(float)
    q_raw = np.quantile(raw, levels, axis=1).T
    quantiles = {'raw_ensemble': q_raw, 'linear_quantile': lqr_quantiles}
    means = {'raw_ensemble': raw.mean(axis=1)}
    crps = {'raw_ensemble': raw_crps(raw, y)}
    lqr_valid = np.isfinite(lqr_quantiles).all(axis=1) & (np.diff(lqr_quantiles, axis=1) >= 0).all(axis=1) & \
                (lqr_quantiles >= 0).all(axis=1) & (lqr_quantiles <= 1).all(axis=1)
    means['linear_quantile'] = np.trapezoid(np.column_stack([lqr_quantiles[:, 0], lqr_quantiles, lqr_quantiles[:, -1]]),
                                              x=np.r_[0., levels, 1.], axis=1)
    crps['linear_quantile'] = quantile_crps(lqr_quantiles, levels, y)
    crps['linear_quantile'][~lqr_valid] = np.nan
    for method in ('cn_emos', 'cn_emos_b'):
        table = r_predictions.loc[r_predictions.method.eq(method)].set_index('row_id').reindex(frame.row_id)
        loc = table.location.to_numpy(float); scale = table.scale.to_numpy(float)
        good = np.isfinite(loc) & np.isfinite(scale) & (scale > 0)
        q = np.full((n, len(levels)), np.nan); mean = np.full(n, np.nan)
        if good.any():
            q[good] = censored_normal_quantiles(loc[good], scale[good], levels)
            mean[good] = censored_normal_mean(loc[good], scale[good])
        quantiles[method] = q; means[method] = mean; crps[method] = table.crps.to_numpy(float)
    common = lqr_valid.copy()
    for method in quantiles:
        common &= np.isfinite(quantiles[method]).all(axis=1) & np.isfinite(crps[method]) & np.isfinite(means[method])
    tables = []
    for method, q in quantiles.items():
        lower = np.array([np.interp(.05, levels, row) for row in q])
        upper = np.array([np.interp(.95, levels, row) for row in q])
        median = np.array([np.interp(.5, levels, row) for row in q])
        residual = y[:, None] - q
        pinball = np.maximum(levels * residual, (levels - 1) * residual).mean(axis=1)
        part = frame[['row_id', 'delivery_day', 'hour_start_utc', 'lead_hours', 'measured_ac_mw']].copy()
        part['method'] = method; part['common_score'] = common; part['crps_fraction_capacity'] = crps[method]
        part['pinball_mean_fraction_capacity'] = pinball; part['mean_fraction_capacity'] = means[method]
        part['median_fraction_capacity'] = median; part['lower_90_fraction_capacity'] = lower
        part['upper_90_fraction_capacity'] = upper; part['observation_fraction_capacity'] = y
        part['lqr_crossing'] = (np.diff(lqr_quantiles, axis=1) < 0).any(axis=1)
        part['lqr_out_of_bounds'] = ((lqr_quantiles < 0) | (lqr_quantiles > 1)).any(axis=1)
        for i, level in enumerate(levels): part[f'q_{i+1:02d}'] = q[:, i]
        tables.append(part)
    return pd.concat(tables, ignore_index=True)


def summarize_scores(predictions, levels, rating, destination):
    """Write matched overall/month scores and a quantile reliability plot."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    valid = predictions.loc[predictions.common_score].copy()
    if valid.empty:
        pd.DataFrame(columns=['month', 'method', 'hours', 'crps_mw']).to_csv(destination / 'scores.csv', index=False)
        pd.DataFrame(columns=['method', 'level', 'observed_fraction_le_quantile']).to_csv(
            destination / 'quantile_reliability.csv', index=False)
        return False
    valid['month'] = valid.delivery_day.str[:7]
    rows = []
    for month, subset in [('all', valid), *list(valid.groupby('month', sort=True))]:
        for method, group in subset.groupby('method'):
            error = group.mean_fraction_capacity - group.observation_fraction_capacity
            rows.append({'month': month, 'method': method, 'hours': len(group), 'days': group.delivery_day.nunique(),
                         'crps_mw': rating * group.crps_fraction_capacity.mean(),
                         'mean_pinball_mw': rating * group.pinball_mean_fraction_capacity.mean(),
                         'rmse_mw': rating * np.sqrt(np.mean(error ** 2)), 'mae_mw': rating * np.mean(abs(error)),
                         'bias_mw': rating * np.mean(error),
                         'coverage_90': np.mean((group.observation_fraction_capacity >= group.lower_90_fraction_capacity) &
                                                (group.observation_fraction_capacity <= group.upper_90_fraction_capacity)),
                         'width_90_mw': rating * np.mean(group.upper_90_fraction_capacity - group.lower_90_fraction_capacity)})
    scores = pd.DataFrame(rows)
    baseline = scores.loc[scores.method.eq('raw_ensemble'), ['month', 'crps_mw']].set_index('month').crps_mw
    scores['crps_skill_vs_raw'] = [1 - row.crps_mw / baseline.loc[row.month] if baseline.loc[row.month] > 0 else np.nan
                                   for row in scores.itertuples()]
    scores.to_csv(destination / 'scores.csv', index=False)
    reliability = []
    fig, ax = plt.subplots(figsize=(6, 5), layout='constrained')
    for method, group in valid.groupby('method'):
        observed = group.observation_fraction_capacity.to_numpy(float)
        cover = [float(np.mean(observed <= group[f'q_{i+1:02d}'].to_numpy(float))) for i in range(len(levels))]
        ax.plot(levels, cover, label=method)
        reliability.extend({'method': method, 'level': float(level), 'observed_fraction_le_quantile': value}
                           for level, value in zip(levels, cover))
    ax.plot([0, 1], [0, 1], 'k--', alpha=.6); ax.set(xlabel='Quantile level', ylabel='Observed fraction at or below quantile')
    ax.legend(); ax.grid(alpha=.3); fig.savefig(destination / 'quantile_reliability.png', dpi=150); plt.close(fig)
    pd.DataFrame(reliability).to_csv(destination / 'quantile_reliability.csv', index=False)
    return True


def fit_farm(config, farm, hourly, members, rating, root):
    """Fit only available past labels; compare all methods on common future hours."""
    section = config['pv_mayer_postprocessing']; root.mkdir(parents=True, exist_ok=True)
    frame = hourly.copy().reset_index(drop=True); frame['row_id'] = np.arange(len(frame))
    frame['split'] = np.where(frame.delivery_day <= section['train_end_day'], 'train', 'evaluation')
    first_eval = pd.Timestamp(next(e for e in AuctionCalendar.from_config(config).schedule(
        str((pd.Timestamp(section['train_end_day']) + pd.Timedelta(days=1)).date())) if e.auction == 'DA').submission_at)
    frame['fit_label_available'] = pd.to_datetime(frame.target_available_at_utc, utc=True) <= first_eval
    frame['target_valid'] = frame.valid_quarters.eq(4) & frame.out_of_bounds_quarters.eq(0) & frame.measured_ac_mw.notna()
    frame['eligible'] = frame.daylight & frame.target_valid & frame.forecast_valid
    frame.to_parquet(root / 'hourly_inputs_mw.parquet', index=False)
    frame['y'] = frame.measured_ac_mw / rating; frame['control'] = frame.control_mw / rating
    frame[members] = frame[members] / rating
    frame['ens_mean'] = frame[members].mean(axis=1)
    frame['variance'] = frame[members].var(axis=1, ddof=1); frame['ens_sd'] = np.sqrt(frame.variance)
    frame['log_variance'] = np.nan
    positive = frame.variance > 0
    frame.loc[positive, 'log_variance'] = np.log(frame.loc[positive, 'variance'])
    frame['lead'] = frame.lead_hours
    frame.to_parquet(root / 'hourly_inputs_normalized.parquet', index=False)
    train = frame.loc[frame.split.eq('train') & frame.eligible & frame.fit_label_available].copy()
    test = frame.loc[frame.split.eq('evaluation') & frame.eligible].copy()
    counts = frame.groupby(['split', frame.delivery_day.str[:7]], sort=True).agg(
        total_hours=('row_id', 'size'), incomplete_target_hours=('valid_quarters', lambda x: int((x < 4).sum())),
        out_of_bounds_hours=('out_of_bounds_quarters', lambda x: int((x > 0).sum())),
        invalid_forecast_hours=('forecast_valid', lambda x: int((~x).sum())),
        daylight_hours=('daylight', 'sum'), eligible_hours=('eligible', 'sum'),
        zero_variance_hours=('variance', lambda x: int((x == 0).sum())),
        unavailable_fit_labels=('fit_label_available', lambda x: int((~x).sum()))).reset_index().rename(columns={'delivery_day': 'month'})
    counts.to_csv(root / 'hourly_counts.csv', index=False)
    if train.empty or test.empty: raise ValueError('No eligible training or evaluation hours')
    r_columns = ['row_id', 'lead', 'y', 'control', 'ens_mean', 'ens_sd', 'variance', 'log_variance']
    r_predictions, status = r_models(train[r_columns], test[r_columns], root / 'r_models', config_path(config, section['rscript']))
    levels = np.arange(1, int(section['quantile_count']) + 1) / (int(section['quantile_count']) + 1)
    models, q = fit_lqr(train[members].to_numpy(), train.control.to_numpy(), train.y.to_numpy(),
                        test[members].to_numpy(), test.control.to_numpy(), levels)
    joblib.dump({'models': models, 'levels': levels, 'members': members, 'rating_mw': rating}, root / 'linear_quantile.joblib')
    scored = scoring_rows(test, members, levels, r_predictions, q)
    scored.to_parquet(root / 'evaluation_predictions.parquet', index=False)
    has_comparison = summarize_scores(scored, levels, rating, root)
    return {'farm_id': farm, 'training_hours': len(train), 'evaluation_hours': len(test),
            'matched_scored_hours': int(scored.loc[scored.method.eq('raw_ensemble'), 'common_score'].sum()),
            'zero_variance_training_hours': int(train.variance.eq(0).sum()),
            'zero_variance_evaluation_hours': int(test.variance.eq(0).sum()),
            'lqr_crossing_hours': int((np.diff(q, axis=1) < 0).any(axis=1).sum()),
            'lqr_out_of_bounds_hours': int(((q < 0) | (q > 1)).any(axis=1).sum()),
            'r_failed_fits': int(status.status.eq('failed').sum()), 'comparison_available': has_comparison,
            'fit_cutoff_utc': first_eval.isoformat()}


def _prepare_one_day(config, farm, day, archive):
    """Keep one farm/day/member set isolated and limit nested numerical threads."""
    import torch
    torch.set_num_threads(1)
    return prepare_farm(config, farm, {day: archive})


def prepare_farm_parallel(config, farm, routes, pool):
    """Combine day-level member forecasts in time order without cross-day mixing."""
    ordered = sorted(routes.items())
    futures = [pool.submit(_prepare_one_day, config, farm, day, archive) for day, archive in ordered]
    frames, quarter_frames, names, rating, location = [], [], None, None, None
    source_digest = hashlib.sha256()
    for (day, _), future in zip(ordered, futures):
        hourly, quarters, day_names, day_rating, day_location, day_digest = future.result()
        if names is not None and (names != day_names or rating != day_rating or location != day_location):
            raise ValueError(f'Farm {farm} model or member contract changed across days')
        names, rating, location = day_names, day_rating, day_location
        frames.append(hourly); quarter_frames.append(quarters)
        source_digest.update(str(day.date()).encode()); source_digest.update(day_digest.encode())
    return (pd.concat(frames, ignore_index=True), pd.concat(quarter_frames, ignore_index=True),
            names, rating, location, source_digest.hexdigest())


def run(config):
    """Run a fully covered, explicitly approved retrospective study into a new directory."""
    section = config['pv_mayer_postprocessing']
    routes = check_coverage(config)  # Reject undeclared gaps before creating any output.
    root = config_path(config, section['output_root']) / section['run_id']
    if root.exists(): raise FileExistsError(f'Output already exists; choose a new run ID: {root}')
    root.mkdir(parents=True)
    write_json(root / 'resolved_config.json', config)
    write_json(root / 'manifest.json', {'status': 'running', 'configuration': section})
    summaries = []
    try:
        for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[variable] = '1'
        with ProcessPoolExecutor(max_workers=int(section['workers'])) as pool:
            for farm in section['farm_ids']:
                print(f'Farm {farm}: preparing {len(routes)} days using up to {section["workers"]} single-thread workers', flush=True)
                hourly, quarters, members, rating, location, source_digest = prepare_farm_parallel(config, int(farm), routes, pool)
                farm_root = root / f'farm_{farm}'
                farm_root.mkdir(parents=True, exist_ok=True)
                quarters.to_parquet(farm_root / 'quarter_hour_inputs_mw.parquet', index=False)
                summary = fit_farm(config, int(farm), hourly, members, rating, farm_root)
                summary['daylight_location_proxy'] = location
                bundle = config_path(config, config['pv']['forecast_replacement']['models_by_farm'][str(farm)]['artifact_path'])
                summary['bundle_manifest_sha256'] = file_hash(bundle)
                summary['aligned_weather_and_targets_sha256'] = source_digest
                summaries.append(summary)
                print(f'Farm {farm}: {summary["matched_scored_hours"]} matched scored hours', flush=True)
        status = 'complete_with_model_failures' if any(x['r_failed_fits'] or not x['comparison_available'] for x in summaries) else 'complete'
        write_json(root / 'manifest.json', {'status': status, 'configuration': section, 'farms': summaries,
            'uncovered_delivery_days': [str(d.date()) for d in pd.date_range(section['start_day'], section['end_day']) if d not in routes],
            'exclusions_csv_sha256': file_hash(config_path(config, config['pv_data_quality']['exclusions_csv'])),
            'implementation_sha256': {p.name: file_hash(p) for p in
                [Path(__file__), Path(__file__).with_name('pv_mayer_data.py'), Path(__file__).with_name('pv_mayer_fit.R')]},
            'retrospective': True, 'forecast_availability': 'preceding 00 UTC cycle plus assumed configured lag',
            'control': 'same-cycle deterministic forecast proxy, not an identified ensemble control',
            'zero_targets': 'retained unless in the approved exclusions CSV',
            'limitations': 'saved neural model selection used retrospective outcomes; no copula or MILP scenarios'})
    except Exception as error:
        write_json(root / 'manifest.json', {'status': 'failed', 'configuration': section, 'farms': summaries, 'error': str(error)})
        raise
    return summaries
