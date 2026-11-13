"""Prepare a separate retrospective gas, France and Swiss DA feature bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.price_feature_data import _external_grid
from bess_pv_trading.forecasting.studies.zero_shot_da import digest, write_json


def main():
    """Align selected lagged source quantities and save complete-case requests."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/zero_shot_da_cross_border.yaml')
    parser.add_argument('--nord-bundle')
    parser.add_argument('--snapshot')
    parser.add_argument('--output')
    args = parser.parse_args()
    options = load_config(args.config)['zero_shot_da_cross_border']
    source_root = Path(args.snapshot or options['source_snapshot'])
    nord = Path(args.nord_bundle or options['nord_bundle'])
    output = Path(args.output or options['output'])
    if output.exists():
        raise FileExistsError(f'Use a new output path: {output}')
    base = json.loads((nord / 'manifest.json').read_text())
    snapshot = json.loads((source_root / 'manifest.json').read_text())
    if digest(source_root / 'manifest.json') != base['source_manifest_sha256']:
        raise ValueError('All input blocks must come from the same pinned snapshot')
    settings = load_config(options['feature_search_config'])['price_feature_search']
    settings['feature_blocks'] = {name: settings['feature_blocks'][name] for name in options['blocks']}
    required = {item['source'] for block in settings['feature_blocks'].values() for item in block['features']}
    frames, hashes = {}, {}
    for name in sorted(required):
        entry = snapshot['tables'][name]
        path = source_root / entry['path']
        if digest(path) != entry['sha256']:
            raise ValueError(f'Changed pinned input: {name}')
        frames[name] = pd.read_parquet(path)
        hashes[name] = entry['sha256']
    timezone = base['settings']['timezone']
    first = pd.Timestamp(snapshot['selection']['start'], tz=timezone).tz_convert('UTC')
    end = pd.Timestamp(snapshot['selection']['end_exclusive'], tz=timezone).tz_convert('UTC')
    grid = pd.date_range(first, end, freq='15min', inclusive='left')
    features, names, _ = _external_grid(frames, grid, settings)
    if any('spread' in name.lower() or 'price_it' in name.lower() for name in names):
        raise ValueError('Target-derived price features are prohibited')
    output.mkdir(parents=True)
    (output / 'requests').mkdir()
    files, coverage = {}, []
    for name in ('labels.parquet', 'source_da.parquet', 'source_manifest.json'):
        shutil.copy2(nord / name, output / name)
        files[name] = digest(output / name)
    for relative, expected in base['files'].items():
        if not relative.startswith('requests/'):
            continue
        original = nord / relative
        if digest(original) != expected:
            raise ValueError(f'Changed source request: {relative}')
        request = json.loads(original.read_text())
        starts = pd.DatetimeIndex(pd.to_datetime(request['product_starts_utc'], utc=True))
        wanted = pd.date_range(pd.Timestamp(request['context_start']), starts[-1] + pd.Timedelta(minutes=15),
                               freq='15min', inclusive='left')
        locations = grid.get_indexer(wanted)
        matrix = np.full((len(wanted), len(names)), np.nan)
        valid = locations >= 0
        matrix[valid] = features[locations[valid]]
        missing = (~np.isfinite(matrix)).sum(axis=0)
        eligible = not missing.any()
        coverage.append({'delivery_day': request['delivery_day'], 'eligible': eligible,
                         **dict(zip(names, missing.astype(int)))})
        if not eligible:
            continue
        request['forecast_covariates'].update({name: matrix[:, index].tolist() for index, name in enumerate(names)})
        request['feature_names'] = [*base['features'], *names]
        destination = output / relative
        write_json(destination, request)
        files[relative] = digest(destination)
    pd.DataFrame(coverage).to_csv(output / 'coverage.csv', index=False)
    files['coverage.csv'] = digest(output / 'coverage.csv')
    result = dict(base)
    result.update(files=files, features=[*base['features'], *names], source_extra_sha256=hashes,
        eligible_days=sum(row['eligible'] for row in coverage), requested_days=len(coverage),
        extra_feature_assumption='Realized prior-day quantities and previous gas prices assumed available; actual publication clocks and revisions are not verified. Prior-day same-clock quantities may occur after the DA gate.',
        extra_missing_policy='Complete case over history and horizon; no imputation.',
        extra_feature_definitions=settings['feature_blocks'])
    write_json(output / 'manifest.json', result)
    print(json.dumps({'eligible_days': result['eligible_days'], 'requested_days': len(coverage), 'features': names}, indent=2))


if __name__ == '__main__':
    main()
