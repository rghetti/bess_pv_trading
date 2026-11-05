"""Diagnose DA errors and test nonlinear radiation response on saved matrices."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.price_search_models import make_price_candidate
from search_energy_price_models import _rolling_blocks


def summarize(frame):
    """Return interval-weighted price errors and daily error concentration."""
    errors = frame['point'] - frame['actual']
    return {
        'rows': len(frame),
        'days': frame['day'].nunique(),
        'mae_eur_per_mwh': float(errors.abs().mean()),
        'wape_percent': float(100 * errors.abs().sum() / frame['actual'].abs().sum()),
        'midday_bias_eur_per_mwh': float(errors[frame['hour'].between(12, 17)].mean()),
    }


def main():
    """Fit matched linear and solar-hinge models without using validation labels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--radiation-indices', nargs='+', type=int, required=True)
    parser.add_argument('--half-life', type=int, default=2880)
    parser.add_argument('--rolling', action='store_true', help='Refit weekly using available previous labels')
    args = parser.parse_args()
    dataset = np.load(args.matrix, allow_pickle=False)
    train = dataset['train_x']
    validation = dataset['validation_x']
    validation_radiation = validation[:, args.radiation_indices].mean(axis=1)
    if args.rolling:
        blocks = list(_rolling_blocks(dataset['starts'], dataset['decisions_ns'], dataset['available_ns'],
            dataset['product_end_ns'], 'Europe/Rome', 7, 1))
    else:
        blocks = [(None, None, np.array([], dtype=int), np.arange(len(validation)))]
    args.output.mkdir(parents=True, exist_ok=True)
    results = {}
    local_times = pd.to_datetime(dataset['starts'], utc=True).tz_convert('Europe/Rome')
    all_knots = []
    for name in ('baseline', 'solar_hinges'):
        prediction = np.full(len(validation), np.nan)
        for _, _, past, test in blocks:
            # Prior validation targets are included only after the shared
            # rolling-split helper verifies their publication and gap rules.
            train_x = np.concatenate((train, validation[past]))
            train_y = np.concatenate((dataset['train_y'], dataset['validation_y'][past]))
            times = pd.DatetimeIndex(pd.to_datetime(np.concatenate((dataset['train_starts'], dataset['starts'][past])), utc=True))
            if times.hasnans or times.has_duplicates or not times.is_monotonic_increasing:
                raise ValueError('Recency weights require unique chronological training rows')
            train_radiation = train_x[:, args.radiation_indices].mean(axis=1)
            daylight = train_radiation[train_radiation > 10]
            if len(daylight) < 100 or not np.isfinite(train_radiation).all():
                raise ValueError('Radiation columns must contain sufficient finite daylight observations')
            # Each refit learns hinge positions from its own training rows.
            knots = np.quantile(daylight, [0.5, 0.75, 0.9])
            test_x = validation[test]
            if name == 'solar_hinges':
                all_knots.append(knots.tolist())
                train_x = np.column_stack((train_x, np.maximum(train_radiation[:, None] - knots, 0) / 100))
                test_x = np.column_stack((test_x, np.maximum(validation_radiation[test, None] - knots, 0) / 100))
            model = make_price_candidate('recency_interaction_elastic_net', {
                'half_life_observations': args.half_life, 'lag_count': 11,
                'harmonics': 1, 'alpha': 0.1, 'l1_ratio': 0.5, 'max_iter': 5000,
            }, seed=17, threads=1)
            model.fit(train_x, train_y)
            prediction[test] = model.predict(test_x)
        if not np.isfinite(prediction).all():
            raise ValueError('Rolling split did not produce every prediction')
        frame = pd.DataFrame({'product_start': dataset['starts'], 'actual': dataset['validation_y'],
                              'point': prediction, 'radiation': validation_radiation,
                              'day': local_times.date, 'hour': local_times.hour})
        frame.to_csv(args.output / f'{name}.csv', index=False)
        results[name] = summarize(frame)
    results['knots_w_m2_by_refit'] = all_knots
    results['rolling'] = args.rolling
    results['matrix_sha256'] = hashlib.sha256(args.matrix.read_bytes()).hexdigest()
    results['radiation_indices'] = args.radiation_indices
    (args.output / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
