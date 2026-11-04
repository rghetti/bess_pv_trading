"""Target-free inference for validation-selected reviewed-GHI HPO models."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ...data import night_window_mask
from ..data.pv_neural_data import transform_features
from ..data.pv_reviewed_hpo_data import calendar_features
from ..models.pv_reviewed_hpo_models import build_neural_network, build_sequences, predict_neural


def _sha(path):
    """Verify the saved model and preprocessing inputs before making predictions."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_signature(path):
    """Detect replacement of a model file before reusing its in-process loader."""
    stat = Path(path).stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


@lru_cache(maxsize=8)
def _load_reviewed_model(directory, farm_id, signatures):
    """Verify and load one immutable model bundle for reuse in this worker."""
    directory = Path(directory)
    record = json.loads((directory / 'result.json').read_text())
    if record.get('schema') != 'reviewed_ghi_forecast_bundle_v1':
        raise ValueError('Unsupported reviewed HPO model bundle')
    trial = record['trial']
    if int(trial['farm_id']) != int(farm_id):
        raise ValueError('Model bundle belongs to a different farm')
    for name, expected in record['checksums'].items():
        if name != 'prediction_mw.npy' and _sha(directory / name) != expected:
            raise ValueError(f'Model bundle checksum mismatch: {name}')
    preprocessor = json.loads((directory / 'preprocessor.json').read_text())
    if (directory / 'model.joblib').exists():
        model = joblib.load(directory / 'model.joblib')
    else:
        import torch

        # Missing-value indicators double the input columns when enabled.
        count = len(preprocessor['columns']) * (2 if preprocessor.get('include_missing_indicators', True) else 1)
        model = build_neural_network(trial['family'], count, trial['parameters'])
        model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
        model.eval()
    return record, preprocessor, model


def _utc_column(frame, name):
    """Reject naive or missing timestamps instead of assuming an availability time."""
    values = pd.DatetimeIndex(frame[name])
    if values.tz is None or values.isna().any():
        raise ValueError(f'{name} must contain timezone-aware timestamps without gaps')
    return values.tz_convert('UTC')


def inference_features(frame, preprocessor, timezone):
    """Build known forecast-weather inputs without reading observed GHI or AC."""
    required = {'interval_start_utc', 'delivery_day', 'forecast_cycle_utc',
                'forecast_available_at_utc', 'decision_cutoff_utc', 'forecast_ghi', 'forecast_bhi'}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f'Forecast frame is missing required fields: {sorted(missing)}')
    starts = _utc_column(frame, 'interval_start_utc')
    if starts.has_duplicates or not starts.is_monotonic_increasing:
        raise ValueError('Forecast rows must have unique, ordered 15-minute interval starts')
    if np.any(starts.minute % 15) or np.any(starts.second):
        raise ValueError('Forecast rows must start on quarter-hour boundaries')
    days = starts.tz_convert(timezone).strftime('%Y-%m-%d')
    if not np.array_equal(days, frame.delivery_day.to_numpy()):
        raise ValueError('Forecast delivery-day labels disagree with local interval times')
    vintages = _utc_column(frame, 'forecast_cycle_utc')
    available = _utc_column(frame, 'forecast_available_at_utc')
    cutoff = _utc_column(frame, 'decision_cutoff_utc')
    if (available > cutoff).any() or (available < vintages).any():
        raise ValueError('The selected forecast must be available by the decision cutoff')
    ghi = frame.forecast_ghi.to_numpy(dtype=float)
    if not (np.isfinite(ghi) & (ghi >= 0)).all():
        raise ValueError('Forecast GHI must be finite and nonnegative for every prediction interval')
    columns = frame.reset_index(drop=True).copy()
    columns['ghi_w_m2'] = ghi
    columns['nwp_bhi_w_m2'] = frame.forecast_bhi.to_numpy(dtype=float)
    calendar = calendar_features(starts, timezone)
    for name in calendar:
        columns[name] = calendar[name]
    columns['forecast_lead_hours'] = (starts + pd.Timedelta(minutes=7.5) - vintages) / pd.Timedelta(hours=1)
    columns['cycle_hour_sin'] = np.sin(2 * np.pi * vintages.hour / 24)
    columns['cycle_hour_cos'] = np.cos(2 * np.pi * vintages.hour / 24)
    names = preprocessor['columns']
    missing_features = set(names) - set(columns.columns)
    if missing_features:
        raise ValueError(f'Forecast frame is missing model features: {sorted(missing_features)}')
    return columns.loc[:, names].replace([np.inf, -np.inf], np.nan)


def predict_reviewed_hpo(frame, bundle_directory, *, farm_id, night_policy):
    """Predict 15-minute AC MW with a saved winner and causally available forecasts."""
    if not night_policy or not night_policy.get('enabled', True) or 'monthly_windows' not in night_policy:
        raise ValueError('Reviewed PV inference requires an enabled monthly night policy')
    directory = Path(bundle_directory).resolve()
    model_path = directory / ('model.joblib' if (directory / 'model.joblib').exists() else 'model.pt')
    if not model_path.is_file():
        raise ValueError('Model bundle is missing its trained weights')
    metadata_path = directory / 'result.json'
    metadata = json.loads(metadata_path.read_text())
    dependencies = sorted(name for name in metadata.get('checksums', {}) if name != 'prediction_mw.npy')
    signatures = tuple(_file_signature(path) for path in
        [metadata_path, directory / 'preprocessor.json', model_path, *(directory / name for name in dependencies)])
    record, preprocessor, model = _load_reviewed_model(str(directory), int(farm_id), signatures)
    trial = record['trial']
    # Missing nighttime irradiance cannot affect the required zero output;
    # preserve strict forecast-GHI validation for all daylight intervals.
    inputs = frame.copy()
    night = night_window_mask(_utc_column(inputs, 'interval_start_utc'), night_policy)
    if 'forecast_ghi' in inputs:
        missing_night = night & pd.isna(inputs['forecast_ghi']).to_numpy()
        inputs.loc[missing_night, 'forecast_ghi'] = 0.0
    features = inference_features(inputs, preprocessor, record['timezone'])
    matrix = transform_features(features, preprocessor).astype(np.float32)
    rating = float(record['rating_mw'])
    if model_path.suffix == '.joblib':
        result = np.clip(model.predict(matrix), 0, 1) * rating
    else:
        scaffold = inputs.reset_index(drop=True).copy()
        scaffold['retained_prior_exclusion'] = False
        scaffold['model_target_mw'] = np.nan
        scaffold['split'] = 'prediction'
        horizon = record.get('input_contract', {}).get('issue_horizon_hours')
        if horizon is not None:
            # Four-cycle networks learned contiguous issue windows, which can cross local midnight.
            starts = _utc_column(scaffold, 'interval_start_utc')
            cycles = _utc_column(scaffold, 'forecast_cycle_utc')
            elapsed = (starts - cycles) / pd.Timedelta(hours=1)
            if np.any((elapsed < 0) | (elapsed >= float(horizon))):
                raise ValueError('Forecast interval falls outside the trained issue horizon')
            scaffold['delivery_day'] = cycles.astype(str)
        empty = np.zeros(len(scaffold), dtype=bool)
        packed = build_sequences(scaffold, matrix,
            {'train': empty, 'validation': empty, 'test': empty}, rating)
        groups = np.arange(len(packed['split']))
        result = predict_neural(model, packed, groups, int(trial['parameters']['batch_size']), len(frame), rating)
    if not np.isfinite(result).all():
        raise ValueError('A forecast interval has no neural prediction')
    result = np.clip(result, 0, rating)
    result[night] = 0.0
    return result
