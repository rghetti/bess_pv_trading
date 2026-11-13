"""Prepare and launch own-gate IDA studies sequentially, with independent GPU models."""

from __future__ import annotations

import argparse
import fcntl
import shutil
import tempfile
import json
import os
from pathlib import Path
import subprocess
import sys

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.zero_shot_ida import IDAInputs, prepare_ida_bundle
from bess_pv_trading.forecasting.studies.zero_shot_da import digest


def ensure_inputs(config, root, settings, prepare=True):
    """Prepare absent bundles atomically, and verify existing semantic inputs.

    Global windows/variants lists are orchestration choices, so adding a window
    does not invalidate an already pinned 7-day or 28-day experiment.
    """
    root = Path(root)
    definitions = load_config(settings['auction_config'])['price_feature_search']
    all_models = load_config(settings['model_config'])['zero_shot_price_study']['models']
    models = {name: all_models[name] for name in settings['models']}
    source_hash = digest(Path(settings['snapshot']) / 'manifest.json')
    if settings['minimum_context_days'] < 1:
        raise ValueError('minimum_context_days must be positive')
    actions = []
    for market in settings['auctions']:
        if market not in ('IDA1', 'IDA2', 'IDA3'):
            raise ValueError(f'Unsupported auction: {market}')
        for window in settings['windows']:
            if not isinstance(window, int) or window < settings['minimum_context_days']:
                raise ValueError('Every context window must be an integer >= minimum_context_days')
            for variant in settings['variants']:
                if variant not in ('history', 'nord', 'nord_spread'):
                    raise ValueError(f'Unsupported feature variant: {variant}')
                path = root / market / f'{window}d' / variant / 'inputs'
                actions.append({'auction': market, 'window': window, 'variant': variant,
                                'path': str(path), 'action': 'reuse' if path.exists() else 'prepare'})
    lock = None
    if prepare:
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / '.preparation.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    inputs = None
    try:
        for action in actions:
            path = Path(action['path'])
            manifest = path / 'manifest.json'
            if path.exists():
                if not manifest.is_file():
                    raise ValueError(f'Incomplete existing input directory; use a fresh root: {path}')
                saved = json.loads(manifest.read_text())
                expected = {key: settings[key] for key in ('start', 'minimum_context_days', 'timezone',
                            'seed', 'quantiles', 'precision', 'november_holdout_start')}
                expected.update(context_days=action['window'], market=action['auction'], gate=action['auction'])
                auction = definitions['auctions'][action['auction']]
                expected.update(decision_time=auction['decision_time'], decision_day_offset=auction['decision_day_offset'])
                if settings.get('end_exclusive') is not None:
                    expected['end_exclusive'] = settings['end_exclusive']
                mismatch = [key for key, value in expected.items() if saved['settings'].get(key) != value]
                if mismatch or saved.get('models') != models or saved.get('source_manifest_sha256') != source_hash:
                    raise ValueError(f'Existing bundle configuration mismatch ({mismatch}); use a new --root: {path}')
                for relative, checksum in saved['files'].items():
                    if digest(path / relative) != checksum:
                        raise ValueError(f'Changed pinned input: {path / relative}')
                action['action'] = 'reuse'
                continue
            if not prepare:
                continue
            if inputs is None:
                inputs = IDAInputs(settings['snapshot'], settings, definitions)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = Path(tempfile.mkdtemp(prefix='.inputs-preparing-', dir=path.parent))
            try:
                prepared = temporary / 'inputs'
                print(f"Preparing {action['auction']} {action['window']}d {action['variant']}", flush=True)
                prepare_ida_bundle(inputs, prepared, action['auction'], action['window'], action['variant'], models)
                if path.exists():
                    raise FileExistsError(f'Another writer created {path}')
                os.rename(prepared, path)
                action['action'] = 'prepared'
            finally:
                shutil.rmtree(temporary)
    finally:
        if lock is not None:
            lock.close()
    return actions


def main():
    """Plan by default; explicit --run launches the configured historical replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/zero_shot_ida.yaml')
    parser.add_argument('--root')
    parser.add_argument('--python-root', default='outputs/zero_shot_envs')
    parser.add_argument('--devices', nargs='+', default=['auto'])
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare missing input bundles, without inference')
    args = parser.parse_args()
    settings = load_config(args.config)['zero_shot_ida']
    root = Path(args.root or settings['output'])
    if args.prepare_only and args.run:
        parser.error('--prepare-only and --run are mutually exclusive')
    preparation = ensure_inputs(args.config, root, settings, prepare=args.run or args.prepare_only)
    if args.prepare_only:
        print(json.dumps({'mode': 'prepare_only', 'inputs': preparation}, indent=2))
        return
    commands = []
    for market in settings['auctions']:
        for window in settings['windows']:
            for variant in settings['variants']:
                directory = root / market / f'{window}d' / variant
                manifest = directory / 'inputs/manifest.json'
                command = [sys.executable, '-m', 'scripts.da_zero_shot_probe', 'run',
                    '--bundle', str(directory / 'inputs'), '--output', str(directory / 'run'),
                    '--python-root', args.python_root, '--devices', *args.devices,
                    '--batch-size', str(args.batch_size), '--threads', str(args.threads)]
                if args.run:
                    command.append('--run')
                if args.allow_download:
                    command.append('--allow-download')
                if args.preflight_only:
                    command.append('--preflight-only')
                report = [sys.executable, '-m', 'scripts.da_zero_shot_probe', 'report',
                    '--bundle', str(directory / 'inputs'), '--forecast-directory', str(directory / 'run'),
                    '--output', str(directory / 'run/report')]
                commands.append((command, report))
    print(json.dumps({'mode': 'run' if args.run else 'plan_only', 'studies': len(commands),
                      'input_preparation': preparation, 'commands': commands}, indent=2), flush=True)
    if args.run:
        for command, report in commands:
            subprocess.run(command, check=True)
            if not args.preflight_only:
                subprocess.run(report, check=True)
        if not args.preflight_only:
            subprocess.run([sys.executable, '-m', 'scripts.report_ida_zero_shot',
                            '--config', args.config, '--root', str(root)], check=True)


if __name__ == '__main__':
    main()
