"""Prepare IDA1/2/3 own-gate zero-shot bundles; never run model inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.zero_shot_ida import IDAInputs, prepare_ida_bundle
from bess_pv_trading.forecasting.studies.zero_shot_da import write_json


def main():
    """Pin all configured IDA windows, feature variants and cancellation provenance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/zero_shot_ida.yaml')
    parser.add_argument('--output')
    parser.add_argument('--auctions', nargs='+')
    parser.add_argument('--windows', nargs='+', type=int)
    parser.add_argument('--variants', nargs='+', choices=['history', 'nord', 'nord_spread'])
    args = parser.parse_args()
    settings = load_config(args.config)['zero_shot_ida']
    definitions = load_config(settings['auction_config'])['price_feature_search']
    models = load_config(settings['model_config'])['zero_shot_price_study']['models']
    models = {name: models[name] for name in settings['models']}
    inputs = IDAInputs(settings['snapshot'], settings, definitions)
    root = Path(args.output or settings['output'])
    records = []
    for market in args.auctions or settings['auctions']:
        if market not in ('IDA1', 'IDA2', 'IDA3'):
            raise ValueError(f'Unsupported own-gate auction {market}')
        for window in args.windows or settings['windows']:
            for variant in args.variants or settings['variants']:
                path = root / market / f'{window}d' / variant / 'inputs'
                result = prepare_ida_bundle(inputs, path, market, window, variant, models)
                record = {'auction': market, 'context_days': window, 'variant': variant,
                          'eligible_days': result['eligible_days'], 'requested_days': result['requested_days']}
                records.append(record)
                print(json.dumps(record), flush=True)
    write_json(root / 'preparation.json', records)


if __name__ == '__main__':
    main()
