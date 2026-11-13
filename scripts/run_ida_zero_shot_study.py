"""Prepare IDA bundles concurrently and stream ready studies to GPU workers."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.zero_shot_ida import IDAInputs, prepare_ida_bundle
from bess_pv_trading.forecasting.studies.zero_shot_da import digest


_PREPARATION_INPUTS = None


def _initialize_preparation_worker(snapshot, settings, definitions):
    """Load the pinned source snapshot once in each CPU worker process."""
    global _PREPARATION_INPUTS
    _PREPARATION_INPUTS = IDAInputs(snapshot, settings, definitions)


def _prepare_input_action(action, models):
    """Build one bundle in an isolated temporary directory and install it atomically."""
    path = Path(action['path'])
    temporary = Path(tempfile.mkdtemp(prefix='.inputs-preparing-', dir=path.parent))
    try:
        prepared = temporary / 'inputs'
        print(f"Preparing {action['auction']} {action['window']}d {action['variant']}", flush=True)
        prepare_ida_bundle(_PREPARATION_INPUTS, prepared, action['auction'], action['window'],
                           action['variant'], models)
        if path.exists():
            raise FileExistsError(f'Another writer created {path}')
        os.rename(prepared, path)
        return action['path']
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def ensure_inputs(config, root, settings, prepare=True, workers=5, on_ready=None):
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
    if workers < 1:
        raise ValueError('Input preparation workers must be positive')
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
        for temporary in root.glob('*/*/*/.inputs-preparing-*'):
            shutil.rmtree(temporary, ignore_errors=True)
    try:
        pending = []
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
                if on_ready is not None:
                    on_ready(path.parent)
                continue
            if not prepare:
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            pending.append(action)
        if pending:
            # Separate processes use independent interpreters, avoiding the GIL
            # bottleneck from the former thread pool during Python-heavy bundling.
            action_by_path = {action['path']: action for action in pending}
            with ProcessPoolExecutor(max_workers=min(workers, len(pending)),
                    initializer=_initialize_preparation_worker,
                    initargs=(settings['snapshot'], settings, definitions)) as executor:
                futures = [executor.submit(_prepare_input_action, action, models) for action in pending]
                for future in as_completed(futures):
                    prepared_path = future.result()
                    action = action_by_path[prepared_path]
                    action['action'] = 'prepared'
                    if on_ready is not None:
                        on_ready(Path(prepared_path).parent)
    finally:
        if lock is not None:
            lock.close()
    return actions


def _report_task(bundle, forecast_directory, output):
    """Render one study report in an isolated CPU process."""
    from bess_pv_trading.forecasting.studies.zero_shot_da import report
    report(bundle, forecast_directory, output)
    return str(output)


def _report_is_complete(bundle, output):
    """Skip a complete saved report when resuming only report generation."""
    try:
        manifest = json.loads((Path(bundle) / 'manifest.json').read_text())
        status = json.loads((Path(output) / 'status.json').read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return False
    expected = manifest['eligible_days'] * len(manifest['models'])
    return (status.get('complete') is True
            and status.get('expected_daily_predictions') == expected
            and status.get('saved_daily_predictions') == expected
            and all((Path(output) / filename).is_file() for filename in
                    ('summary.md', 'scores.csv', 'matched_forecasts.parquet', 'daily_scores.csv')))


def generate_reports(commands, workers, skip_complete=False):
    """Render independent reports concurrently, then let the caller compare them."""
    pending = []
    for _, command in commands:
        bundle = command[command.index('--bundle') + 1]
        output = command[command.index('--output') + 1]
        forecast_directory = command[command.index('--forecast-directory') + 1]
        if skip_complete and _report_is_complete(bundle, output):
            print(f'Reusing completed IDA report: {output}', flush=True)
            continue
        pending.append((bundle, forecast_directory, output))
    if not pending:
        return
    with ProcessPoolExecutor(max_workers=min(workers, len(pending))) as executor:
        futures = [executor.submit(_report_task, *item) for item in pending]
        for future in as_completed(futures):
            print(f'Completed IDA report: {future.result()}', flush=True)


def main():
    """Plan by default; explicit --run launches the configured historical replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/zero_shot_ida.yaml')
    parser.add_argument('--root')
    parser.add_argument('--python-root', default='outputs/zero_shot_envs')
    parser.add_argument('--devices', nargs='+', default=['auto'])
    parser.add_argument('--batch-size', type=int, default=10)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--prepare-workers', type=int, default=10,
                        help='Concurrent CPU processes for independent input bundles')
    parser.add_argument('--report-workers', type=int, default=30,
                        help='Concurrent CPU processes for independent report generation')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report-only', action='store_true',
                        help='Resume report/plot generation from saved forecasts without inference')
    parser.add_argument('--refresh-reports', action='store_true',
                        help='Regenerate complete reports as well as missing reports')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--prepare-only', action='store_true', help='Prepare missing input bundles, without inference')
    args = parser.parse_args()
    settings = load_config(args.config)['zero_shot_ida']
    root = Path(args.root or settings['output'])
    if sum((args.prepare_only, args.run, args.report_only)) > 1:
        parser.error('--prepare-only, --run and --report-only are mutually exclusive')
    if args.report_workers < 1:
        parser.error('--report-workers must be positive')
    devices = None
    if args.run:
        from scripts.da_zero_shot_probe import available_devices
        devices = available_devices()[:4] if args.devices == ['auto'] else args.devices
        if not devices or len(devices) > 4 or len(set(devices)) != len(devices):
            raise ValueError('Provide one to four distinct GPU devices')

        def prepare_while_running(on_ready):
            return ensure_inputs(args.config, root, settings, prepare=True,
                                 workers=args.prepare_workers, on_ready=on_ready)

        from scripts.ida_gpu_queue import execute_queue_while_preparing
        print(json.dumps({'gpu_workers': len(devices), 'devices': devices,
                          'preparation_workers': args.prepare_workers,
                          'batch_size': args.batch_size,
                          'queued_model_studies': len(settings['auctions']) * len(settings['windows']) *
                          len(settings['variants']) * len(settings['models'])}), flush=True)
        preparation = execute_queue_while_preparing(
            settings['models'], devices, args.python_root, args.batch_size, args.threads,
            prepare_while_running, args.allow_download, args.preflight_only)
    else:
        preparation = ensure_inputs(args.config, root, settings,
                                    prepare=args.prepare_only, workers=args.prepare_workers)
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
    if args.report_only:
        print(f'Resuming IDA reports for {len(commands)} studies with {args.report_workers} workers.', flush=True)
        generate_reports(commands, args.report_workers, skip_complete=not args.refresh_reports)
        subprocess.run([sys.executable, '-m', 'scripts.report_ida_zero_shot',
                        '--config', args.config, '--root', str(root)], check=True)
        return
    print(json.dumps({'mode': 'run' if args.run else 'plan_only', 'studies': len(commands),
                      'input_preparation': preparation, 'commands': commands}, indent=2), flush=True)
    if args.report_only or (args.run and not args.preflight_only):
        generate_reports(commands, args.report_workers, skip_complete=args.report_only)
        subprocess.run([sys.executable, '-m', 'scripts.report_ida_zero_shot',
                        '--config', args.config, '--root', str(root)], check=True)


if __name__ == '__main__':
    main()
