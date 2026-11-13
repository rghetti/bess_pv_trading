"""Run the pinned DA context-window and external-feature studies as one GPU queue."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.zero_shot_da import digest
from scripts.da_zero_shot_probe import available_devices, verify_bundle


def _prepare_cross_border_action(action, snapshot, nord_bundle, feature_config):
    """Build one external-feature bundle in a temporary sibling directory."""
    from scripts.prepare_da_cross_border_covariates import prepare_bundle

    output = Path(action['output'])
    temporary = Path(tempfile.mkdtemp(prefix='.bundle-preparing-', dir=output.parent))
    try:
        prepared = temporary / 'inputs'
        if action['blocks']:
            result = prepare_bundle(nord_bundle, snapshot, prepared, feature_config, action['blocks'])
        else:
            shutil.copytree(nord_bundle, prepared)
            result = json.loads((prepared / 'manifest.json').read_text())
            result['extra_feature_definitions'] = {}
            result['nord_bundle_manifest_sha256'] = digest(Path(nord_bundle) / 'manifest.json')
            (prepared / 'manifest.json').write_text(json.dumps(result, indent=2))
        if action.get('holiday'):
            from scripts.prepare_ida_feature_extensions import add_holidays
            add_holidays(prepared, action['variant'])
        if output.exists():
            raise FileExistsError(f'Another writer created {output}')
        os.rename(prepared, output)
        return {'output': str(output), 'eligible_days': result['eligible_days']}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _prepare_core_action(action, snapshot, base_config):
    """Prepare one configurable history or NORD bundle atomically."""
    study = Path(action['study'])
    output = Path(action['path'])
    temporary = Path(tempfile.mkdtemp(prefix='.inputs-preparing-', dir=output.parent))
    prepared = temporary / 'inputs'
    try:
        if action['variant'] == 'history':
            command = [sys.executable, '-m', 'scripts.da_zero_shot_probe', 'prepare',
                       '--config', base_config, '--snapshot', str(snapshot),
                       '--context-days', str(action['window_days']), '--output', str(prepared)]
        elif action['variant'] == 'nord':
            history = study.parent / 'history' / 'inputs'
            command = [sys.executable, '-m', 'scripts.prepare_da_nord_covariates',
                       '--history-bundle', str(history), '--snapshot', str(snapshot),
                       '--output', str(prepared)]
        else:
            raise ValueError(f"Unsupported DA baseline variant: {action['variant']}")
        subprocess.run(command, check=True)
        if output.exists():
            raise FileExistsError(f'Another writer created {output}')
        os.rename(prepared, output)
        return {'study': str(study), 'path': str(output)}
    finally:
        shutil.rmtree(temporary, ignore_errors=True)


def _validate_external_bundle(bundle, nord_bundle, snapshot, blocks, holiday=False):
    """Check content hashes, source lineage and the requested block identity."""
    manifest = verify_bundle(bundle)
    nord_manifest = json.loads((nord_bundle / 'manifest.json').read_text())
    expected_blocks = set(blocks)
    saved_blocks = set(manifest.get('extra_feature_definitions', {}))
    if holiday != ('italian_national_holiday' in manifest.get('features', [])):
        raise ValueError(f'Holiday feature mismatch at {bundle}')
    if saved_blocks != expected_blocks:
        raise ValueError(f'External bundle block mismatch at {bundle}: {sorted(saved_blocks)}')
    nord_manifest_hash = digest(nord_bundle / 'manifest.json')
    if manifest.get('nord_bundle_manifest_sha256') not in (None, nord_manifest_hash):
        raise ValueError(f'External bundle uses a different NORD base: {bundle}')
    # Older combined bundles preserve the history-only parent hash, not the
    # immediate NORD-manifest hash; validate their shared source lineage instead.
    if manifest.get('nord_bundle_manifest_sha256') is None:
        shared_lineage = all(manifest.get(key) == nord_manifest.get(key) for key in (
            'history_bundle_manifest_sha256', 'source_auction_sha256',
            'source_forecasts_sha256', 'source_manifest_sha256'))
        shared_settings = manifest.get('settings') == nord_manifest.get('settings')
        shared_features = manifest.get('features', [])[:len(nord_manifest.get('features', []))] == nord_manifest.get('features', [])
        if not (shared_lineage and shared_settings and shared_features):
            raise ValueError(f'Legacy external bundle lineage does not match NORD base: {bundle}')
    if manifest.get('models') != nord_manifest.get('models'):
        raise ValueError(f'External bundle model specs differ from NORD base: {bundle}')
    if manifest.get('source_manifest_sha256') != digest(snapshot / 'manifest.json'):
        raise ValueError(f'External bundle uses a different pinned snapshot: {bundle}')
    return manifest


def prepare_study_inputs(settings, root, workers, prepare=True, on_ready=None):
    """Validate configured windows and stream parallel core/external preparation."""
    root = Path(root)
    snapshot = Path(settings['snapshot'])
    if workers < 1:
        raise ValueError('Preparation worker count must be positive')
    if settings['cross_border_window'] < 7:
        raise ValueError('cross_border_window must be at least seven days')
    actions = []
    lock = None
    model_definitions = load_config(settings['model_config'])['zero_shot_price_study']['models']
    expected_models = {name: model_definitions[name] for name in settings['models']}
    core_config = settings['base_config']
    base_settings = load_config(core_config)['zero_shot_da_probe']
    if base_settings['models'] != settings['models']:
        raise ValueError('The base DA config and study config must select the same models')
    core_actions = []
    external_actions = []
    if prepare:
        root.mkdir(parents=True, exist_ok=True)
        lock = (root / '.da_preparation.lock').open('w')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for temporary in root.glob('*/*/.inputs-preparing-*'):
            shutil.rmtree(temporary, ignore_errors=True)
        for temporary in root.glob('*/*/.bundle-preparing-*'):
            shutil.rmtree(temporary, ignore_errors=True)
    try:
        # Keep the cross-border NORD parent even if it is not a scored window.
        prepared_windows = sorted(set(settings['windows']) | {settings['cross_border_window']})
        for days in prepared_windows:
            variants = list(settings['baseline_variants'])
            if days == settings['cross_border_window'] and 'nord' not in variants:
                variants.append('nord')
            for variant in variants:
                study = root / f'{days}d' / variant
                bundle = study / 'inputs'
                action = {'path': str(bundle), 'study': str(study), 'variant': variant,
                          'window_days': days, 'action': 'reuse' if bundle.is_dir() else 'prepare'}
                actions.append(action)
                if bundle.is_dir():
                    manifest = verify_bundle(bundle)
                    if (manifest['settings'].get('context_days') != days
                            or manifest['settings'].get('market', 'DA') != 'DA'
                            or manifest.get('models') != expected_models):
                        raise ValueError(f'Unexpected DA baseline settings: {bundle}')
                    if variant == 'nord':
                        history_bundle = root / f'{days}d/history/inputs'
                        if not history_bundle.is_dir():
                            raise ValueError(f'NORD bundle is present without its history parent: {bundle}')
                        if manifest.get('history_bundle_manifest_sha256') != digest(history_bundle / 'manifest.json'):
                            raise ValueError(f'NORD bundle has a different history parent: {bundle}')
                    if on_ready is not None:
                        on_ready(study)
                elif prepare:
                    bundle.parent.mkdir(parents=True, exist_ok=True)
                    core_actions.append(action)

        nord_bundle = (root / f"{settings['cross_border_window']}d" /
                       settings['cross_border_base_variant'] / 'inputs')
        # Existing NORD-at-cross-border-window input is required as the parent
        # for retrospective blocks; configured windows can still add other sizes.
        if nord_bundle.is_dir():
            base_manifest = verify_bundle(nord_bundle)
            if (base_manifest['settings'].get('market', 'DA') != 'DA'
                    or base_manifest.get('models') != expected_models):
                raise ValueError(f'Expected a DA NORD bundle: {nord_bundle}')

        for name, definition in settings['cross_border_variants'].items():
            study = root / f"{settings['cross_border_window']}d" / definition['directory']
            bundle = study / 'inputs'
            action = {'path': str(bundle), 'study': str(study), 'variant': name,
                      'window_days': settings['cross_border_window'], 'blocks': definition['blocks'],
                      'holiday': definition.get('holiday', False),
                      'action': 'reuse' if bundle.exists() else 'prepare'}
            if bundle.exists():
                if not nord_bundle.is_dir():
                    raise FileNotFoundError(f'Missing NORD parent for external bundle: {nord_bundle}')
                manifest = _validate_external_bundle(bundle, nord_bundle, snapshot, definition['blocks'], definition.get('holiday', False))
                action['eligible_days'] = manifest['eligible_days']
                if on_ready is not None:
                    on_ready(study)
            elif prepare:
                bundle.parent.mkdir(parents=True, exist_ok=True)
                action['output'] = str(bundle)
                external_actions.append(action)
            actions.append(action)

        if prepare and (core_actions or external_actions):
            actions_by_study = {action['study']: action for action in actions}
            tasks = {}
            submitted_core = set()
            external_scheduled = False
            with ProcessPoolExecutor(max_workers=workers) as executor:
                def submit_core(action):
                    future = executor.submit(_prepare_core_action, action, str(snapshot), core_config)
                    tasks[future] = ('core', action)
                    submitted_core.add(action['study'])

                def submit_external():
                    nonlocal external_scheduled
                    if external_scheduled:
                        return
                    external_scheduled = True
                    for action in external_actions:
                        future = executor.submit(_prepare_cross_border_action, action, str(snapshot),
                            str(nord_bundle), settings['feature_search_config'])
                        tasks[future] = ('external', action)

                # Existing 60-day NORD permits external work to overlap all other
                # window preparation. Otherwise it is scheduled after its parent.
                if nord_bundle.is_dir():
                    submit_external()
                for action in core_actions:
                    if action['variant'] == 'history':
                        submit_core(action)
                    else:
                        history = root / f"{action['window_days']}d/history/inputs"
                        if history.is_dir():
                            submit_core(action)
                        # Missing history causes the NORD task to be submitted only
                        # after the history bundle has been installed.

                while tasks:
                    done = next(as_completed(tuple(tasks)))
                    kind, action = tasks.pop(done)
                    result = done.result()
                    if kind == 'external':
                        action['action'] = 'prepared'
                        action['eligible_days'] = result['eligible_days']
                        output = Path(result['output'])
                        _validate_external_bundle(output, nord_bundle, snapshot, action['blocks'], action['holiday'])
                        if on_ready is not None:
                            on_ready(Path(action['study']))
                        continue

                    action['action'] = 'prepared'
                    manifest = verify_bundle(Path(result['path']))
                    if on_ready is not None:
                        on_ready(Path(result['study']))
                    if action['variant'] == 'history':
                        nord_action = actions_by_study[str(root / f"{action['window_days']}d/nord")]
                        if nord_action['action'] == 'prepare' and nord_action['study'] not in submitted_core:
                            submit_core(nord_action)
                    if (action['window_days'] == settings['cross_border_window']
                            and action['variant'] == settings['cross_border_base_variant']):
                        nord_bundle = Path(result['path'])
                        submit_external()
    finally:
        if lock is not None:
            lock.close()
    return actions


def report_studies(settings, root, actions, python, workers=1):
    """Write per-study reports and matched baseline/external-feature summaries."""
    commands = []
    for action in actions:
        study = Path(action['study'])
        commands.append([python, '-m', 'scripts.da_zero_shot_probe', 'report',
                         '--bundle', str(study / 'inputs'), '--forecast-directory', str(study / 'run'),
                         '--output', str(study / 'run/report')])
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(subprocess.run, command, check=True) for command in commands]
        for index, future in enumerate(as_completed(futures), 1):
            future.result()
            print(f'Reports completed: {index}/{len(commands)}', flush=True)
    if settings.get('holiday_ablation'):
        from scripts.report_da_holiday_ablation import report_ablation
        report_ablation(settings, root, actions)
        return
    subprocess.run([python, '-m', 'scripts.compare_da_context_windows', '--study-root', str(root),
                    '--windows', *map(str, settings['windows']),
                    '--variants', *settings['baseline_variants']], check=True)
    baseline_report = root / f"{settings['cross_border_window']}d" / settings['cross_border_base_variant'] / 'run/report'
    for action in actions:
        if 'blocks' not in action:
            continue
        study = Path(action['study'])
        output = root / f"{settings['cross_border_window']}d" / 'external_comparisons' / action['variant']
        subprocess.run([python, '-m', 'scripts.compare_da_nord_covariates',
            '--history-report', str(baseline_report), '--nord-report', str(study / 'run/report'),
            '--output', str(output), '--history-label', 'nord_only',
            '--nord-label', f"nord_plus_{action['variant']}",
            '--availability-note', 'Retrospective sensitivity: previous-day realized gas and French/Swiss quantities are assumed available. Historical publication and revision timing is unverified; same-clock quantities may postdate the DA gate.'], check=True)


def main():
    """Plan, prepare, or launch all configured DA studies in one streaming queue."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/zero_shot_da_studies.yaml')
    parser.add_argument('--python-root', default='outputs/zero_shot_envs')
    parser.add_argument('--devices', nargs='+', default=['auto'])
    parser.add_argument('--prepare-workers', type=int)
    parser.add_argument('--report-workers', type=int, default=30)
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--threads', type=int)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if sum((args.prepare_only, args.run, args.report_only)) > 1:
        parser.error('--prepare-only and --run are mutually exclusive')
    settings = load_config(args.config)['zero_shot_da_studies']
    if set(settings['baseline_variants']) != {'history', 'nord'}:
        raise ValueError('Baseline comparison requires history and nord variants')
    if not settings['windows'] or any(not isinstance(days, int) or days < 7 for days in settings['windows']):
        raise ValueError('Configured DA windows must be nonempty whole days of at least seven')
    workers = args.prepare_workers or settings['prepare_workers']
    batch_size = args.batch_size or settings['batch_size']
    threads = args.threads or settings['threads_per_model']
    root = Path(settings['root'])

    if args.run:
        devices = available_devices()[:4] if args.devices == ['auto'] else args.devices
        if not devices or len(devices) > 4 or len(set(devices)) != len(devices):
            raise ValueError('Provide one to four distinct visible GPU devices')

        def prepare_while_running(on_ready):
            return prepare_study_inputs(settings, root, workers, prepare=True, on_ready=on_ready)

        from scripts.ida_gpu_queue import execute_queue_while_preparing
        actions = execute_queue_while_preparing(
            settings['models'], devices, args.python_root, batch_size, threads,
            prepare_while_running, args.allow_download, args.preflight_only)
        if not args.preflight_only:
            report_studies(settings, root, actions, sys.executable, args.report_workers)
        return

    actions = prepare_study_inputs(settings, root, workers, prepare=args.prepare_only)
    if args.report_only:
        report_studies(settings, root, actions, sys.executable, args.report_workers)
        return
    print(json.dumps({'mode': 'prepare_only' if args.prepare_only else 'plan_only',
                      'preparation_workers': workers, 'batch_size': batch_size,
                      'studies': actions}, indent=2), flush=True)


if __name__ == '__main__':
    main()
