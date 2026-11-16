"""Prepare, run and compare TimesFM realized-input studies for all auctions."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.zero_shot_da import digest, report
from bess_pv_trading.forecasting.studies.zero_shot_price_scoring import score
from scripts.da_zero_shot_probe import verify_bundle
from scripts.ida_gpu_queue import execute_queue
from scripts.prepare_timesfm_realized import prepare


def prepare_one(base, snapshot, directory, variant, settings):
    """Install a validated bundle atomically and reuse only identical pinned inputs."""
    directory = Path(directory)
    output = directory / 'inputs'
    if output.exists():
        manifest = verify_bundle(output)
        if (manifest.get('base_manifest_sha256') != digest(Path(base) / 'manifest.json') or
            manifest.get('actuals_manifest_sha256') != digest(Path(snapshot) / 'manifest.json') or
            manifest.get('realized_builder_sha256') != digest(Path(__file__).with_name('prepare_timesfm_realized.py')) or
            manifest.get('realized_variant') != variant or
            manifest.get('realized_feature_definitions') != settings['variants'][variant] or
            manifest['settings'].get('minimum_context_days', 7) != settings['minimum_context_days']):
            raise ValueError(f'Existing bundle differs from requested study: {output}')
        return str(directory)
    directory.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.realized-preparing-', dir=directory))
    try:
        prepare(base, snapshot, temporary / 'inputs', variant, settings)
        os.rename(temporary / 'inputs', output)
    finally:
        shutil.rmtree(temporary, ignore_errors=True)
    return str(directory)


def render(directory):
    """Produce one independent forecast report in a CPU process."""
    directory = Path(directory)
    report(directory / 'inputs', directory / 'run', directory / 'run/report')
    return str(directory)


def compare(root, market, variants):
    """Score variants on matching products and retain all sample exclusions."""
    frames, counts = [], []
    for variant in variants:
        directory = root / market / variant / 'run/report'
        if not json.loads((directory / 'status.json').read_text())['complete']:
            raise ValueError(f'Incomplete report: {directory}')
        frame = pd.read_parquet(directory / 'matched_forecasts.parquet')
        frame['input_variant'] = variant
        frames.append(frame)
        counts.append({'variant': variant, 'available_rows': len(frame)})
    all_rows = pd.concat(frames, ignore_index=True)
    keys = ['delivery_day', 'product_start', 'auction', 'gate', 'variant']
    if all_rows.duplicated(keys + ['input_variant']).any():
        raise ValueError('Duplicate forecast product')
    sizes = all_rows.groupby(keys).size()
    common = sizes[sizes == len(variants)].reset_index()[keys]
    matched = all_rows.merge(common, on=keys, validate='many_to_one')
    truth = matched.groupby(keys).actual.agg(['min', 'max'])
    if matched.empty or not np.allclose(truth['min'], truth['max']):
        raise ValueError(f'No common products or inconsistent actuals: {market}')
    output = root / market / 'comparison'
    output.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(output / 'matched_forecasts.parquet', index=False)
    rows = []
    for group, data in matched.groupby(['variant', 'period', 'input_variant']):
        rows.append(dict(zip(['model', 'period', 'input_variant'], group)) | score(data))
    scores = pd.DataFrame(rows)
    scores.to_csv(output / 'scores.csv', index=False)
    for item in counts:
        item['matched_rows'] = len(common)
    pd.DataFrame(counts).to_csv(output / 'coverage.csv', index=False)
    text = [f'# {market}: TimesFM realized-covariate comparison', '',
            'All variants use matched products; November is reserved. Prior-day actuals are assumed available retrospectively.', '',
            '| Period | Variant | Products | MAE | WAPE | RMSE | Bias |', '|---|---|---:|---:|---:|---:|---:|']
    for _, row in scores.sort_values(['period', 'mae', 'input_variant']).iterrows():
        text.append(f"| {row['period']} | {row['input_variant']} | {int(row['products'])} | {row['mae']:.3f} | {100*row['wape']:.2f}% | {row['rmse']:.3f} | {row['bias']:.3f} |")
    (output / 'summary.md').write_text('\n'.join(text) + '\n')


def main():
    """Prepare bundles or launch an explicit, resumable historical inference run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/timesfm_realized_study.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2', 'cuda:3'])
    parser.add_argument('--python-root', default='outputs/zero_shot_envs')
    parser.add_argument('--prepare-workers', type=int, default=5)
    parser.add_argument('--report-workers', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if sum((args.prepare_only, args.run, args.report_only)) != 1:
        parser.error('Choose exactly one of --prepare-only, --run or --report-only')
    if min(args.prepare_workers, args.report_workers, args.batch_size, args.threads) < 1 or len(args.devices) != len(set(args.devices)):
        parser.error('Worker counts must be positive and devices distinct')
    settings = load_config(args.config)['realized_timesfm_study']
    root = Path(settings['output'])
    tasks = [(base, settings['actuals_snapshot'], root / market / variant, variant, settings)
             for market, base in settings['base_bundles'].items() for variant in settings['variants']]
    if not args.report_only:
        with ProcessPoolExecutor(max_workers=args.prepare_workers) as executor:
            futures = [executor.submit(prepare_one, *task) for task in tasks]
            for index, future in enumerate(as_completed(futures), 1):
                print(f'Prepared {index}/{len(tasks)}: {future.result()}', flush=True)
    studies = [item[2] for item in tasks]
    if args.prepare_only:
        return
    if args.run:
        execute_queue(studies, settings['models'], args.devices, args.python_root, args.batch_size, args.threads)
    with ProcessPoolExecutor(max_workers=args.report_workers) as executor:
        futures = [executor.submit(render, directory) for directory in studies]
        for index, future in enumerate(as_completed(futures), 1):
            print(f'Reported {index}/{len(tasks)}: {future.result()}', flush=True)
    for market in settings['base_bundles']:
        compare(root, market, list(settings['variants']))


if __name__ == '__main__':
    main()
