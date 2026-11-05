"""Compare auction forecasts with and without a delivery-day Italian holiday flag."""

import argparse
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.price_search_models import make_price_candidate
from bess_pv_trading.forecasting.italian_holidays import italian_holiday_flags
from search_energy_price_models import _rolling_blocks


def main():
    """Fit a matched pair using the same saved price and weather feature rows."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timezone', default='Europe/Rome')
    parser.add_argument('--rolling', action='store_true')
    parser.add_argument('--lag-count', type=int, default=11)
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--l1-ratio', type=float, default=0.5)
    parser.add_argument('--max-iter', type=int, default=5000)
    parser.add_argument('--frequency-days', type=int, default=7)
    parser.add_argument('--gap-days', type=int, default=1)
    parser.add_argument('--family', default='interaction_elastic_net')
    parser.add_argument('--parameters-json', help='Estimator parameters as a JSON object')
    parser.add_argument('--anchor-index', type=int, help='Feature column containing the cleared price')
    parser.add_argument('--anchor-reference', type=Path, help='Matching zero-spread predictions for validation')
    args = parser.parse_args()
    data = np.load(args.matrix, allow_pickle=False)
    if (args.anchor_index is None) != (args.anchor_reference is None):
        raise ValueError('Spread fitting requires both anchor index and zero-spread reference')
    if args.anchor_index is not None:
        reference = pd.read_csv(args.anchor_reference)
        if (len(reference) != len(data['starts'])
                or not np.array_equal(reference['product_start'].to_numpy(), data['starts'])
                or not np.allclose(reference['actual'], data['validation_y'])
                or not np.allclose(reference['point'], data['validation_x'][:, args.anchor_index])):
            raise ValueError('Configured anchor column does not match saved zero-spread predictions')
    parameters = (json.loads(args.parameters_json) if args.parameters_json else {
        'lag_count': args.lag_count, 'alpha': args.alpha, 'l1_ratio': args.l1_ratio,
        'max_iter': args.max_iter})
    train_holidays = italian_holiday_flags(data['train_starts'], args.timezone)
    validation_holidays = italian_holiday_flags(data['starts'], args.timezone)
    if args.rolling:
        blocks = _rolling_blocks(data['starts'], data['decisions_ns'], data['available_ns'],
                                 data['product_end_ns'], args.timezone, args.frequency_days, args.gap_days)
    else:
        blocks = [(None, None, np.array([], dtype=int), np.arange(len(data['validation_y'])))]
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'matrix_sha256': hashlib.sha256(args.matrix.read_bytes()).hexdigest(),
              'timezone': args.timezone, 'rolling': args.rolling,
              'family': args.family, 'model_parameters': parameters,
              'anchor_index': args.anchor_index,
              'rolling_parameters': {'frequency_days': args.frequency_days, 'gap_days': args.gap_days},
              'training_holiday_rows': int(train_holidays.sum()),
              'validation_holiday_dates': sorted(set(pd.DatetimeIndex(pd.to_datetime(
                  data['starts'][validation_holidays.astype(bool)], utc=True)).tz_convert(args.timezone).date.astype(str))),
              'models': {}}
    for include_holidays in (False, True):
        name = 'holiday' if include_holidays else 'baseline'
        prediction = np.full(len(data['validation_y']), np.nan)
        caught_warnings = []
        for _, _, past, test in blocks:
            # Earlier validation labels enter only after the rolling cutoff.
            train_x = np.concatenate((data['train_x'], data['validation_x'][past]))
            train_y = np.concatenate((data['train_y'], data['validation_y'][past]))
            test_x = data['validation_x'][test]
            # Spread targets use the already-cleared auction price as anchor.
            test_anchor = 0 if args.anchor_index is None else test_x[:, args.anchor_index]
            if args.anchor_index is not None:
                train_y = train_y - train_x[:, args.anchor_index]
            if include_holidays:
                train_x = np.column_stack((train_x, np.concatenate((train_holidays, validation_holidays[past]))))
                test_x = np.column_stack((test_x, validation_holidays[test]))
            model = make_price_candidate(args.family, parameters, seed=17, threads=1)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(train_x, train_y)
            caught_warnings.extend(str(warning.message) for warning in caught)
            prediction[test] = model.predict(test_x) + test_anchor
        if not np.isfinite(prediction).all():
            raise ValueError('A prediction is missing or nonfinite')
        actual = data['validation_y']
        errors = np.abs(prediction - actual)
        frame = pd.DataFrame({'product_start': data['starts'], 'actual': actual, 'point': prediction,
                              'holiday': validation_holidays.astype(int)})
        frame.to_csv(args.output / f'{name}.csv', index=False)
        result['models'][name] = {'mae_eur_per_mwh': float(errors.mean()),
                                  'wape_percent': float(100 * errors.sum() / np.abs(actual).sum()),
                                  'warnings': caught_warnings}
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
