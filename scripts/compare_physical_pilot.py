"""Score saved physical configurations and neural pilots on matched auction views."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.data import ForecastBundle, config_path, load_metadata
from bess_pv_trading.forecasting.models.pv import forecast_pv
from bess_pv_trading.forecasting.studies.pv_realized_ensemble import selected_settings
from bess_pv_trading.forecasting.training.pv_fitting import balanced_weights


def sha256(path):
    """Record exact inputs and physical model artifacts for reproducibility."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        # Record exact inputs and physical model artifacts for reproducibility for each `chunk` in `iter(lambda: source.read(1024 * 1024), b'')`.
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def predict_physical(frame, farm, settings, night_policy):
    """Recompute geometry and PV power separately for each selected weather cycle."""
    prediction = np.full(len(frame), np.nan)
    # Apply forecast pv for `(cycle, positions)` from `frame.groupby('forecast_cycle_utc', sort=False).indices.items()`.
    for cycle, positions in frame.groupby('forecast_cycle_utc', sort=False).indices.items():
        positions = np.asarray(positions)
        view = frame.iloc[positions].copy()
        available = pd.to_datetime(view.forecast_available_at_utc, utc=True)
        # Enforce the input contract: Weather vintage has inconsistent or late availability.
        if available.nunique() != 1 or (available > pd.to_datetime(view.decision_cutoff_utc, utc=True)).any():
            raise ValueError('Weather vintage has inconsistent or late availability')
        view = view.set_index('interval_start_utc', drop=False)
        bundle = ForecastBundle(farm.farm_id, pd.Timestamp(cycle), available.iloc[0], view, {},
                                provenance={'decision_cutoff_utc': available.iloc[0]})
        result = forecast_pv(bundle, farm, settings, night_policy).values
        prediction[positions] = result.pv_ac_mw.to_numpy()
    return prediction


def score(frame, rating):
    """Calculate equal-day/auction weighted scores on exactly the same rows."""
    weights = balanced_weights(frame)
    rows = []
    # Calculate equal-day/auction weighted scores on exactly the same rows for each `(name, column)` in `[('neural_pilot', 'neural_mw'), ('fitted_pvlib', 'fitted_pvlib_mw')]`.
    for name, column in [('neural_pilot', 'neural_mw'), ('fitted_pvlib', 'fitted_pvlib_mw')]:
        error = frame[column].to_numpy(float) - frame.measured_ac_mw.to_numpy(float)
        mae = float(weights @ np.abs(error))
        rmse = float(np.sqrt(weights @ error**2))
        bias = float(weights @ error)
        rows.append({'model': name, 'rows': len(frame), 'day_auction_views': frame.groupby(['delivery_day', 'auction']).ngroups,
                     'mae_mw': mae, 'rmse_mw': rmse, 'bias_mw': bias,
                     'mae_percent_capacity': 100 * mae / rating,
                     'rmse_percent_capacity': 100 * rmse / rating,
                     'bias_percent_capacity': 100 * bias / rating})
    return rows


def run(config):
    """Score the declared splits without fitting or changing source records."""
    section = config['pv_physical_pilot_comparison']
    root = config_path(config, section['output_root']) / section['run_id']
    # Enforce the input contract: Choose a fresh run ID; output already exists: `root`.
    if root.exists():
        raise FileExistsError(f'Choose a fresh run ID; output already exists: {root}')
    farms = load_metadata(config_path(config, config['inputs']['metadata_parquet']), section['farm_ids'])
    scores, manifests = [], []
    # Apply score for `farm_id` from `section['farm_ids']`.
    for farm_id in section['farm_ids']:
        input_path = config_path(config, section['input_root']) / f'farm_{farm_id}' / 'parquet/diagnostic_instances.parquet'
        model_path = config_path(config, section['physical_models'][str(farm_id)])
        model = json.loads(model_path.read_text())
        # Enforce the input contract: Physical model farm mismatch: `model_path`.
        if model['farm_id'] != farm_id:
            raise ValueError(f'Physical model farm mismatch: {model_path}')
        frame = pd.read_parquet(input_path)
        frame = frame.loc[frame.split.isin(section['splits']) & frame.eligible & frame.daylight
                          & ~frame.satellite_ghi_interpolated].copy().reset_index(drop=True)
        # Enforce the input contract: Empty or duplicate test observations for farm `farm_id`.
        if frame.empty or frame.duplicated(['delivery_day', 'auction', 'interval_start_utc']).any():
            raise ValueError(f'Empty or duplicate test observations for farm {farm_id}')
        settings = selected_settings(config, farms[farm_id], model)
        frame['fitted_pvlib_mw'] = predict_physical(frame, farms[farm_id], settings, config['policy']['night'])
        finite = np.isfinite(frame[['measured_ac_mw', 'neural_mw', 'fitted_pvlib_mw']].to_numpy(float)).all(axis=1)
        matched = frame.loc[finite].copy()
        # Enforce the input contract: No matched observations for farm `farm_id`.
        if matched.empty:
            raise ValueError(f'No matched observations for farm {farm_id}')
        # Apply score for `(split, part)` from `matched.groupby('split', sort=True)`.
        for split, part in matched.groupby('split', sort=True):
            # Apply score for `(label, subset)` from `[('ALL_BALANCED', part), *list(part.groupby('auction', sort=True))]`.
            for label, subset in [('ALL_BALANCED', part), *list(part.groupby('auction', sort=True))]:
                # Score the declared splits without fitting or changing source records for each `row` in `score(subset, farms[farm_id].ac_capacity_mw)`.
                for row in score(subset, farms[farm_id].ac_capacity_mw):
                    scores.append({'farm_id': farm_id, 'split': split, 'auction': label, **row})
        folder = root / f'farm_{farm_id}'
        folder.mkdir(parents=True)
        frame.to_parquet(folder / 'predictions.parquet', index=False)
        monthly = []
        matched['month'] = matched.delivery_day.str[:7]
        # Apply score for `((split, month), subset)` from `matched.groupby(['split', 'month'])`.
        for (split, month), subset in matched.groupby(['split', 'month']):
            monthly.extend({'split': split, 'month': month, **row} for row in score(subset, farms[farm_id].ac_capacity_mw))
        monthly = pd.DataFrame(monthly)
        monthly.to_csv(folder / 'monthly_scores.csv', index=False)
        fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, layout='constrained')
        # Process each `(axis, (metric, label))` in this collection.
        for axis, (metric, label) in zip(axes, [('rmse_percent_capacity', 'RMSE'),
                ('mae_percent_capacity', 'MAE'), ('bias_percent_capacity', 'Bias')]):
            # Score the declared splits without fitting or changing source records for each `(model_name, group)` in `monthly.groupby('model')`.
            for model_name, group in monthly.groupby('model'):
                group = group.sort_values('month')
                axis.plot(group.month, group[metric], marker='o', label=model_name)
            axis.set_ylabel(f'{label} (% AC capacity)')
            axis.grid(alpha=.3)
            # Score the declared splits without fitting or changing source records when `metric == 'bias_percent_capacity'`.
            if metric == 'bias_percent_capacity': axis.axhline(0, color='black', linewidth=.7)
        axes[0].set_title(f'Farm {farm_id}: matched daylight auction views')
        axes[0].legend(); axes[-1].set_xlabel('Delivery month: Jan–Apr validation, May–Aug test')
        fig.savefig(folder / 'monthly_errors.png', dpi=140); plt.close(fig)
        manifests.append({'farm_id': farm_id, 'input': str(input_path), 'input_sha256': sha256(input_path),
                          'physical_model': str(model_path), 'physical_model_sha256': sha256(model_path),
                          'proxy_location': model['candidate']['location'], 'candidate_daylight_rows': len(frame),
                          'matched_daylight_rows': len(matched), 'excluded_nonfinite_rows': int((~finite).sum()),
                          'first_day': frame.delivery_day.min(), 'last_day': frame.delivery_day.max(),
                          'physical_selection_scope': model.get('selection_scope')})
        print(f'Farm {farm_id}: {len(matched)}/{len(frame)} matched daylight rows', flush=True)
    pd.DataFrame(scores).to_csv(root / 'scores.csv', index=False)
    manifest = {'status': 'complete', 'farms': manifests,
                'scope': 'Retrospective; physical parameters were fit and selected using validation and test periods',
                'inputs': 'Each auction view retains its own previously aligned deterministic weather forecast',
                'weights': 'equal mass per day and auction view', 'neural_prediction': 'saved diagnostic neural_mw',
                'daylight': 'original neural-study eligibility and daylight mask; observed-only satellite mask',
                'weather_availability': 'saved cycle availability, assumed six-hour lag; checked against each decision cutoff'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(pd.DataFrame(scores).query("auction == 'ALL_BALANCED'").to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    run(load_config(parser.parse_args().config))


# Process this part of the workflow when `__name__ == '__main__'`.
if __name__ == '__main__':
    main()
