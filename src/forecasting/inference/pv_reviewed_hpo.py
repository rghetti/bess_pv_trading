"""Target-free inference for validation-selected reviewed-GHI HPO models."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from ..data.pv_neural_data import transform_features
from ..data.pv_reviewed_hpo_data import calendar_features
from ..models.pv_reviewed_hpo_models import build_neural_network, build_sequences, predict_neural


def _sha(path):
    """Verify the saved model and preprocessing inputs before making predictions."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


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
    expected = pd.DatetimeIndex(pd.to_datetime(frame.delivery_day, utc=True)) - pd.Timedelta(days=1)
    if not vintages.equals(expected):
        raise ValueError('This model requires the preceding delivery day 00 UTC forecast cycle')
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
    names = preprocessor['columns']
    missing_features = set(names) - set(columns.columns)
    if missing_features:
        raise ValueError(f'Forecast frame is missing model features: {sorted(missing_features)}')
    return columns.loc[:, names].replace([np.inf, -np.inf], np.nan)


def predict_reviewed_hpo(frame, bundle_directory, *, farm_id):
    """Predict 15-minute AC MW with a saved winner and causally available forecasts."""
    directory = Path(bundle_directory)
    record = json.loads((directory / 'result.json').read_text())
    if record.get('schema') != 'reviewed_ghi_forecast_bundle_v1':
        raise ValueError('Unsupported reviewed HPO model bundle')
    trial = record['trial']
    if int(trial['farm_id']) != int(farm_id):
        raise ValueError('Model bundle belongs to a different farm')
    for name, expected in record['checksums'].items():
        # Stored retrospective predictions are not an input to future forecast inference.
        if name == 'prediction_mw.npy':
            continue
        if _sha(directory / name) != expected:
            raise ValueError(f'Model bundle checksum mismatch: {name}')
    preprocessor = json.loads((directory / 'preprocessor.json').read_text())
    features = inference_features(frame, preprocessor, record['timezone'])
    matrix = transform_features(features, preprocessor).astype(np.float32)
    rating = float(record['rating_mw'])
    if (directory / 'model.joblib').exists():
        model = joblib.load(directory / 'model.joblib')
        return np.clip(model.predict(matrix), 0, 1) * rating
    if not (directory / 'model.pt').exists():
        raise ValueError('Model bundle is missing its trained weights')
    import torch
    model = build_neural_network(trial['family'], matrix.shape[1], trial['parameters'])
    model.load_state_dict(torch.load(directory / 'model.pt', map_location='cpu', weights_only=True))
    scaffold = frame.reset_index(drop=True).copy()
    scaffold['retained_prior_exclusion'] = False
    scaffold['model_target_mw'] = np.nan
    scaffold['split'] = 'prediction'
    empty = np.zeros(len(scaffold), dtype=bool)
    packed = build_sequences(scaffold, matrix,
        {'train': empty, 'validation': empty, 'test': empty}, rating)
    groups = np.arange(len(packed['split']))
    result = predict_neural(model, packed, groups, int(trial['parameters']['batch_size']), len(frame), rating)
    if not np.isfinite(result).all():
        raise ValueError('A forecast interval has no neural prediction')
    return np.clip(result, 0, rating)
