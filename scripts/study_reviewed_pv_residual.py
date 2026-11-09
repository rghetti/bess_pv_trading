"""Fit chronological, forecast-only corrections to saved reviewed PV predictions."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

# Limit every numerical library before importing it in the parent or workers.
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[variable] = '1'

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path, night_window_mask
from bess_pv_trading.forecasting.training.pv_reviewed_hpo_common import day_balanced_weights, score_predictions, source_masks


def file_sha256(path):
    """Fingerprint a saved experiment input without changing it."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def load_farm(config, farm):
    """Join frozen neural predictions to their exact forecast and target rows."""
    settings = config['pv_residual_study']
    root = config_path(config, settings['base_run'])
    cache = root / f'farm_{farm}' / 'cache'
    selected = pd.read_csv(config_path(config, settings['selection_file']))
    row = selected.loc[selected.farm_id.eq(farm)]
    if len(row) != 1:
        raise ValueError(f'Farm {farm} does not have exactly one selected neural model')
    bundle = config_path(config, row.iloc[0].model_directory)
    record = json.loads((bundle / 'result.json').read_text())
    frame = pd.read_parquet(cache / 'frame.parquet')
    masks = np.load(cache / 'masks.npz')
    base = np.load(bundle / 'prediction_mw.npy')
    matrix = np.load(cache / 'x_extended.npy', mmap_mode='r')
    preprocessor = json.loads((cache / 'preprocessor_extended.json').read_text())
    columns = preprocessor['columns']
    if any('solar' in name or 'zenith' in name or 'azimuth' in name or 'missing' in name for name in columns):
        raise ValueError('Solar-position and missingness features are not permitted')
    if len(frame) != len(base) or len(frame) != len(matrix) or matrix.shape[1] != len(columns):
        raise ValueError('Cached forecast and prediction rows do not align')
    dates = frame.delivery_day.astype(str).to_numpy()
    eligible_validation = masks['validation'].astype(bool)
    eligible_test = masks['test'].astype(bool)
    periods = {
        'train': eligible_validation & (dates >= settings['train_start']) & (dates < settings['validation_start']),
        'validation': eligible_validation & (dates >= settings['validation_start']) & (dates < settings['test_start']),
        'test': eligible_test & (dates >= settings['test_start']) & (dates < settings['end_exclusive']),
    }
    if any(not mask.any() for mask in periods.values()):
        raise ValueError(f'Farm {farm} has an empty study split')
    if any(not np.isfinite(base[mask]).all() for mask in periods.values()):
        raise ValueError(f'Farm {farm} has a missing saved neural prediction')
    starts = pd.DatetimeIndex(frame.interval_start_utc)
    policy_night = night_window_mask(starts, config['policy']['forecast_pv_night'])
    if any(policy_night[mask].any() for mask in periods.values()):
        raise ValueError('Scoring and correction fitting must use the configured daylight window')
    # Neighboring values must belong to the same forecast issue and to a 2026
    # forecast-input row. This excludes historical GHI from boundary features.
    forecast_context = frame['split'].isin(['validation', 'test']).to_numpy()
    cycles = pd.DatetimeIndex(frame.forecast_cycle_utc).asi8
    times = starts.asi8
    ghi = np.asarray(matrix[:, columns.index('ghi_w_m2')], dtype=np.float64)
    neighbors = []
    hour_ns = pd.Timedelta(hours=1).value
    for offset in (-4, 4):
        other = np.arange(len(frame)) + offset
        inside = (other >= 0) & (other < len(frame))
        safe = np.clip(other, 0, len(frame) - 1)
        valid = inside & forecast_context & forecast_context[safe] & (cycles == cycles[safe])
        valid &= (times[safe] - times == offset // 4 * hour_ns)
        value = ghi.copy()
        value[valid] = ghi[safe[valid]]
        neighbors.append(value - ghi)
    quarter = (starts.minute.to_numpy() / 60.0) - 0.375
    rating = float(record['rating_mw'])
    inputs = np.column_stack([matrix, base / rating, quarter, *neighbors,
                              quarter * ghi, quarter * base / rating])
    names = columns + ['base_prediction_pu', 'quarter_offset', 'ghi_previous_hour_delta',
                       'ghi_next_hour_delta', 'quarter_times_ghi', 'quarter_times_base']
    return {'farm': farm, 'frame': frame, 'periods': periods, 'inputs': inputs, 'names': names,
            'base': base, 'rating': rating, 'night': policy_night, 'bundle': bundle,
            'inputs_sha256': {str(path): file_sha256(path) for path in
                [cache / 'frame.parquet', cache / 'masks.npz', cache / 'x_extended.npy',
                 cache / 'preprocessor_extended.json', bundle / 'result.json', bundle / 'prediction_mw.npy']}}


def candidates(settings):
    """Build deterministic residual estimators selected on the later validation period."""
    result = []
    for alpha in settings['ridge_alphas']:
        result.append((f'ridge_alpha_{alpha:g}', make_pipeline(StandardScaler(), Ridge(alpha=alpha))))
    for leaves in settings['hist_max_leaf_nodes']:
        model = HistGradientBoostingRegressor(
            loss='squared_error', learning_rate=settings['hist_learning_rate'],
            max_iter=settings['hist_iterations'], max_leaf_nodes=leaves,
            min_samples_leaf=settings['hist_min_samples_leaf'],
            l2_regularization=settings['hist_l2_regularization'],
            random_state=settings['seed'], early_stopping=False)
        result.append((f'hist_leaves_{leaves}', model))
    return result


def clipped(base, residual, rating, night):
    """Convert a normalized residual to AC power and enforce the night policy."""
    result = np.clip(base + rating * residual, 0.0, rating)
    result[night] = 0.0
    return result


def plot_day(frame, base, corrected, mask, output, timezone):
    """Plot one available issue cycle against reviewed target quarters."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    part = frame.loc[mask].copy()
    if part.empty:
        return
    cycle = part.forecast_cycle_utc.iloc[0]
    positions = np.flatnonzero(mask & frame.forecast_cycle_utc.eq(cycle).to_numpy())
    day = frame.iloc[positions]
    times = pd.DatetimeIndex(day.interval_start_utc).tz_convert(timezone)
    fig, axis = plt.subplots(figsize=(11, 4), layout='constrained')
    axis.plot(times, day.model_target_mw, label='Reviewed target', color='black')
    axis.plot(times, base[positions], label='Neural baseline', color='tab:blue')
    axis.plot(times, corrected[positions], label='Selected correction', color='tab:orange')
    synthetic = day.synthetic_target.to_numpy(dtype=bool)
    axis.scatter(times[synthetic], day.model_target_mw.to_numpy()[synthetic], s=18, color='red', label='Synthetic target')
    axis.set(title=f"Farm {day.attrs['farm']} · {day.delivery_day.iloc[0]} · cycle {cycle}", ylabel='AC power (MW)', xlabel=f'Local time ({timezone})')
    axis.legend()
    axis.grid(alpha=.2)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def shape_metrics(frame, mask, base, corrected, farm, rating):
    """Measure within-hour shape error on complete measured test hours."""
    measured = mask & ~frame.synthetic_target.to_numpy(dtype=bool)
    part = frame.loc[measured].copy()
    part['baseline'] = base[measured]
    part['corrected'] = corrected[measured]
    part['hour'] = part.interval_start_utc.dt.floor('h')
    groups = part.groupby(['forecast_cycle_utc', 'hour'], sort=False)
    part = part.loc[groups.model_target_mw.transform('size').eq(4)].copy()
    if part.empty:
        return []
    groups = part.groupby(['forecast_cycle_utc', 'hour'], sort=False)
    target_shape = part.model_target_mw - groups.model_target_mw.transform('mean')
    rows = []
    for name in ('baseline', 'corrected'):
        prediction_shape = part[name] - groups[name].transform('mean')
        error = 100 * np.sqrt(np.mean((prediction_shape - target_shape) ** 2)) / rating
        rows.append({'farm_id': farm, 'model': name, 'complete_measured_hour_views': len(part) // 4,
                     'within_hour_shape_rmse_percent': error})
    return rows


def run_farm(config, farm):
    """Fit, select and evaluate one farm without fitting on evaluation labels."""
    settings = config['pv_residual_study']
    item = load_farm(config, farm)
    frame = item['frame']
    frame.attrs['farm'] = farm
    masks = item['periods']
    inputs = item['inputs']
    base = item['base']
    rating = item['rating']
    train = masks['train']
    valid = masks['validation']
    weights = day_balanced_weights(frame.delivery_day.to_numpy()[train]) * train.sum()
    target = (frame.model_target_mw.to_numpy(dtype=float)[train] - base[train]) / rating
    baseline = score_predictions(frame, base, valid, rating)['day_balanced_nrmse_percent']
    candidate_rows = [{'farm_id': farm, 'candidate': 'no_correction', 'validation_nrmse_percent': baseline}]
    best_name, best_score, best_model = 'no_correction', baseline, None
    # Train only January-February labels; later labels only select or score.
    for name, model in candidates(settings):
        fit_args = {'ridge__sample_weight': weights} if name.startswith('ridge_') else {'sample_weight': weights}
        model.fit(inputs[train], target, **fit_args)
        candidate = base.copy()
        candidate[valid] = clipped(base[valid], model.predict(inputs[valid]), rating, item['night'][valid])
        score = score_predictions(frame, candidate, valid, rating)['day_balanced_nrmse_percent']
        candidate_rows.append({'farm_id': farm, 'candidate': name, 'validation_nrmse_percent': score})
        if score < best_score:
            best_name, best_score, best_model = name, score, model
    corrected = base.copy()
    selected_rows = masks['train'] | valid | masks['test']
    if best_model is not None:
        corrected[selected_rows] = clipped(base[selected_rows], best_model.predict(inputs[selected_rows]), rating, item['night'][selected_rows])
    corrected[item['night']] = 0.0
    output = config_path(config, settings['output_root']) / f'farm_{farm}'
    output.mkdir(parents=True, exist_ok=True)
    if best_model is not None:
        joblib.dump(best_model, output / 'residual_model.joblib')
    np.savez_compressed(output / 'predictions.npz', baseline_mw=base, corrected_mw=corrected)
    metrics = []
    monthly = []
    for split, mask in masks.items():
        for source, source_mask in source_masks(frame, mask).items():
            for label, prediction in [('neural_baseline', base), ('selected_correction', corrected)]:
                score = score_predictions(frame, prediction, source_mask, rating)
                if score is not None:
                    metrics.append({'farm_id': farm, 'split': split, 'source': source, 'model': label, **score})
        for month in sorted(frame.loc[mask, 'delivery_day'].str.slice(0, 7).unique()):
            month_mask = mask & frame.delivery_day.str.startswith(month).to_numpy()
            for label, prediction in [('neural_baseline', base), ('selected_correction', corrected)]:
                score = score_predictions(frame, prediction, month_mask, rating)
                monthly.append({'farm_id': farm, 'split': split, 'month': month, 'model': label, **score})
        dates = sorted(frame.loc[mask, 'delivery_day'].unique())
        count = min(int(settings['plots_per_split']), len(dates))
        for index in np.linspace(0, len(dates) - 1, count, dtype=int):
            day = dates[index]
            plot_day(frame, base, corrected, mask & frame.delivery_day.eq(day).to_numpy(),
                     output / 'plots' / split / f'{day}.png', settings['timezone'])
    manifest = {'farm_id': farm, 'selected': best_name, 'validation_nrmse_percent': best_score,
                'base_bundle': str(item['bundle']), 'feature_columns': item['names'],
                'input_sha256': item['inputs_sha256'], 'split_rows': {key: int(value.sum()) for key, value in masks.items()},
                'notes': ['base neural weights were fit in 2025 but selected using January-April 2026',
                          '2026 test period has been inspected previously',
                          'synthetic targets are included and separately reported',
                          'forecast features are standardized using the frozen neural-search preprocessor',
                          'no solar position or missingness flags are inputs']}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Farm {farm}: selected {best_name} (validation nRMSE {best_score:.3f}%)', flush=True)
    shape = shape_metrics(frame, masks['test'], base, corrected, farm, rating)
    return candidate_rows, metrics, monthly, manifest, shape


def main():
    """Check data contracts or run the approved correction study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--run', action='store_true')
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    settings = config['pv_residual_study']
    dates = [settings[key] for key in ('train_start', 'validation_start', 'test_start', 'end_exclusive')]
    if dates != sorted(dates) or len(set(dates)) != 4:
        raise ValueError('Residual splits must be distinct and chronological')
    if arguments.check_only:
        for farm in settings['farm_ids']:
            item = load_farm(config, farm)
            print(json.dumps({'farm': farm, 'features': item['names'],
                              'split_rows': {name: int(mask.sum()) for name, mask in item['periods'].items()}}))
        return
    output = config_path(config, settings['output_root'])
    report = config_path(config, settings['report_root'])
    if output.exists() or report.exists():
        raise FileExistsError('Choose a new run ID to preserve existing residual-study results')
    output.mkdir(parents=True)
    report.mkdir(parents=True)
    (output / 'status.json').write_text(json.dumps({'status': 'running'}) + '\n')
    results = Parallel(n_jobs=int(settings['workers']))(delayed(run_farm)(config, farm) for farm in settings['farm_ids'])
    for name, index in [('candidates.csv', 0), ('metrics.csv', 1), ('monthly.csv', 2)]:
        pd.DataFrame([row for result in results for row in result[index]]).to_csv(report / name, index=False)
    pd.DataFrame([row for result in results for row in result[4]]).to_csv(report / 'within_hour_shape.csv', index=False)
    manifests = [result[3] for result in results]
    (report / 'manifest.json').write_text(json.dumps({'configuration': settings, 'farms': manifests}, indent=2) + '\n')
    (output / 'status.json').write_text(json.dumps({'status': 'complete', 'farms': len(results)}) + '\n')
    print(f'Complete: {report}', flush=True)


if __name__ == '__main__':
    main()
