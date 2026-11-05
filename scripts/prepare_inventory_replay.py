"""List recorded MPC runs or prepare a relocated, new-ID comparison config.

This tool never runs an optimization. Inputs, model bundles and Gurobi licenses
must be supplied separately and checked using the printed preflight command.
"""
from copy import deepcopy
import argparse
import gzip
import json
from pathlib import Path
import re
import shlex
import sys


def load_inventory(path):
    """Read the exact archived resolved configurations and source evidence."""
    with gzip.open(path, 'rt', encoding='utf-8') as stream:
        return json.load(stream)


def records(data):
    """Expose stable explicit IDs for individual historical runs and modes."""
    result = []
    for item in data['research_jobs']:
        result.append((f"research:{item['batch']}:{item['job_id']}", item))
    for item in data['main_checkout_modes']:
        result.append((f"main:{item['batch']}:{item['mode']}", item))
    return result


def relocate(value, old_root, new_root):
    """Rebase values under the recorded root, leaving unrelated paths explicit."""
    if isinstance(value, dict):
        return {key: relocate(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [relocate(item, old_root, new_root) for item in value]
    if isinstance(value, str) and old_root:
        old = str(Path(old_root))
        if value == old:
            return str(new_root)
        if value.startswith(old + '/'):
            return str(new_root / value[len(old) + 1:])
    return value


def prepare(item, root, run_id, model_map, allow_retrospective):
    """Keep study settings and swap only root, run identity and requested model."""
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', run_id):
        raise ValueError('Run ID must contain only letters, digits, period, underscore or hyphen')
    original = item.get('resolved_config')
    if not isinstance(original, dict):
        raise ValueError('This entry has no archived resolved configuration')
    if item.get('status') != 'complete':
        raise ValueError('This recorded comparison did not complete; inspect its status first')
    if item.get('mode') != 'continuous':
        raise ValueError('This preparer supports continuous comparisons; inspect frozen-ledger evidence separately')
    old_root = original.get('_project_root') or item['repository']
    config = relocate(deepcopy(original), old_root, root)
    config['_project_root'] = str(root)
    config['continuous_optimization']['output_root'] = 'outputs/portable_mpc_replays'
    config['continuous_optimization']['run_id'] = run_id
    config['continuous_optimization']['resume'] = False
    config.setdefault('experiment_outputs', {})['overwrite_existing'] = False
    config['realtime_mpc_comparison']['mode'] = 'continuous'
    config['realtime_mpc_comparison']['check_only'] = False
    config['realtime_mpc_comparison']['frozen_source'] = None
    if model_map is not None:
        farm = str(item.get('farm', '1'))
        model = model_map.get(farm)
        if not isinstance(model, dict) or not all(model.get(key) for key in ('name', 'kind', 'artifact_path', 'available_at_utc')):
            raise ValueError(f'Model map needs name, kind, artifact_path and available_at_utc for farm {farm}')
        # resolve_config derives selected-farm PV from the preserved shared block.
        # Update both locations so the new bundle survives that merge.
        for section in ('pv', '_pv_shared_settings'):
            if section not in config:
                continue
            replacement = config[section].setdefault('forecast_replacement', {})
            replacement['enabled'] = True
            replacement['allow_retrospective_artifacts'] = allow_retrospective
            replacement.setdefault('models_by_farm', {})[farm] = model
    config['include'] = '../standard.yaml'
    return config


def main():
    """Prepare a single reviewed config and print commands for the new host."""
    default_inventory = Path(__file__).resolve().parents[2] / 'inventory/mpc_experiment_inventory_records.json.gz'
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inventory', type=Path, default=default_inventory)
    parser.add_argument('--list', action='store_true')
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument('--id', help='Exact run ID from --list')
    selector.add_argument('--batch', help='Prepare every completed continuous research job in this batch')
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run-id')
    models = parser.add_mutually_exclusive_group()
    models.add_argument('--model-map', type=Path, help='JSON map of farm IDs to new model registry entries')
    models.add_argument('--keep-original-model', action='store_true')
    parser.add_argument('--allow-retrospective-models', action='store_true')
    args = parser.parse_args()
    data = load_inventory(args.inventory)
    entries = records(data)
    if args.list:
        for key, item in entries:
            print('\t'.join([key, item.get('status', ''), str(item.get('farm', '1')),
                item.get('controller', ','.join(item.get('controllers') or [])), item.get('mode', '')]))
        return
    if not (args.id or args.batch) or not args.run_id or not (args.model_map or args.keep_original_model):
        parser.error('Preparation requires --id or --batch, --run-id and a model choice')
    matches = [(key, item) for key, item in entries if key == args.id] if args.id else [
        (key, item) for key, item in entries if key.startswith('research:')
        and item['batch'] == args.batch and item['status'] == 'complete' and item['mode'] == 'continuous']
    if not matches or (args.id and len(matches) != 1):
        parser.error('ID or batch not found; use --list')
    root = args.root.resolve()
    if not (root / 'configs/standard.yaml').is_file():
        parser.error('--root must identify the appropriate checkout with configs/standard.yaml')
    model_map = json.loads(args.model_map.read_text()) if args.model_map else None
    outputs = []
    try:
        for key, item in matches:
            run_id = args.run_id if args.id else f"{args.run_id}__{item['job_id']}"
            config = prepare(item, root, run_id, model_map, args.allow_retrospective_models)
            output = root / 'configs/research' / f'portable_{run_id}.yaml'
            if output.exists():
                raise ValueError(f'Configuration already exists: {output}')
            outputs.append((key, output, config))
    except ValueError as error:
        parser.error(str(error))
    # Preserve numeric YAML types, including scientific-notation tolerances.
    import yaml

    # All entries are checked before any output is created.
    for key, output, config in outputs:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            yaml.safe_dump(config, stream, sort_keys=False)
        command = f'PYTHONPATH=. python -m bess_pv_trading experiment-realtime-mpc --config {shlex.quote(str(output))}'
        print(f'Original record: {key}')
        print(f'Prepared: {output}')
        print('Preflight:', command, '--check-only')
        print('Run only after reviewing inputs and obtaining any required approval:', command)
    print('Relocated data/model paths must be supplied separately; inspect the full config and preflight output.')


if __name__ == '__main__':
    main()
