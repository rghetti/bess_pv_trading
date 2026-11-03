"""Deterministic HPO trial definitions and daylight-only evaluation utilities."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd


def sample_parameters(space, rng):
    """Draw one reviewable configuration from the declared bounded search space."""
    parameters = {}
    for name, definition in space.items():
        kind = definition['kind']
        if kind == 'choice':
            values = definition['values']
            parameters[name] = values[int(rng.integers(len(values)))]
        elif kind == 'uniform':
            parameters[name] = float(rng.uniform(definition['low'], definition['high']))
        elif kind == 'log_uniform':
            low = np.log(float(definition['low']))
            high = np.log(float(definition['high']))
            parameters[name] = float(np.exp(rng.uniform(low, high)))
        else:
            raise ValueError(f'Unsupported hyperparameter distribution {kind}')
    return parameters


def trial_specifications(section):
    """Generate the full farm × feature × model × trial grid with stable seeds."""
    families = list(section['statistical_families']) + list(section['neural_families'])
    if len(set(families)) != len(families):
        raise ValueError('Statistical and neural family names must be unique')
    if any('pvlib' in family.lower() or 'physical' in family.lower() for family in families):
        raise ValueError('This search must not include a physical PV model')
    trials = []
    # Derive seeds by family and farm so paired feature sets use identical draws.
    for farm_id in section['farm_ids']:
        for family in families:
            space = section['statistical_families'].get(family, section['neural_space'])
            for number in range(int(section['trials_per_family_per_feature'])):
                identity = f"{section['seed']}:{farm_id}:{family}:{number}".encode()
                seed = int.from_bytes(hashlib.sha256(identity).digest()[:4], 'big') & 0x7fffffff
                parameters = sample_parameters(space, np.random.default_rng(seed))
                for feature in section['features']:
                    trials.append({'farm_id': farm_id, 'feature': feature, 'family': family,
                        'number': number, 'seed': seed, 'parameters': parameters,
                        'name': f'{family}_{feature}_{number:03d}'})
    farm_order = {farm: number for number, farm in enumerate(section['farm_ids'])}
    family_order = {family: number for number, family in enumerate(families)}
    feature_order = {feature: number for number, feature in enumerate(section['features'])}
    # The first workers see all farms, rather than exhausting one farm at a time.
    trials.sort(key=lambda trial: (trial['number'], family_order[trial['family']],
                                  feature_order[trial['feature']], farm_order[trial['farm_id']]))
    return trials


def day_balanced_weights(days):
    """Give each represented delivery day equal mass, independent of DST and gaps."""
    series = pd.Series(np.asarray(days))
    counts = series.groupby(series).transform('size').to_numpy(dtype=float)
    weights = 1.0 / counts
    return weights / weights.sum()


def score_predictions(frame, prediction_mw, mask, rating_mw):
    """Calculate day-balanced and physical-quarter scores on an explicit mask."""
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        return None
    observed = frame.model_target_mw.to_numpy(dtype=float)[selected]
    prediction = np.asarray(prediction_mw, dtype=float)[selected]
    if not (np.isfinite(observed).all() and np.isfinite(prediction).all()):
        raise ValueError('An eligible evaluation target or prediction is missing')
    residual = (prediction - observed) / float(rating_mw)
    days = frame.delivery_day.to_numpy()[selected]
    weight = day_balanced_weights(days)
    return {'quarters': int(selected.sum()), 'days': int(pd.Series(days).nunique()),
        'day_balanced_nrmse_percent': float(100 * np.sqrt(np.dot(weight, residual ** 2))),
        'day_balanced_nmae_percent': float(100 * np.dot(weight, np.abs(residual))),
        'day_balanced_bias_percent': float(100 * np.dot(weight, residual)),
        'quarter_weighted_nrmse_percent': float(100 * np.sqrt(np.mean(residual ** 2))),
        'quarter_weighted_nmae_percent': float(100 * np.mean(np.abs(residual))),
        'window_energy_bias_mwh': float(np.sum(prediction - observed) / 4)}


def source_masks(frame, base):
    """Expose measured and synthetic target scores without treating them as equivalent."""
    base = np.asarray(base, dtype=bool)
    synthetic = frame.synthetic_target.to_numpy(dtype=bool)
    return {'all': base, 'measured': base & ~synthetic, 'synthetic': base & synthetic}


def identity_digest(value):
    """Identify a frozen search contract independently of dictionary insertion order."""
    encoded = json.dumps(value, sort_keys=True, separators=(',', ':'), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()
