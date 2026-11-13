"""Prepare, launch or report the approved DA history-only zero-shot comparison."""

from __future__ import annotations

import argparse
import csv
import io
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.zero_shot_da import digest, prepare_bundle, report, write_json


ROOT = Path(__file__).resolve().parents[1]
ENV_NAMES = {'chronos2': 'chronos', 't0_gamma': 't0', 'timesfm3': 'timesfm'}


def verify_bundle(directory):
    """Reject changed labels, requests or source provenance before launch/scoring."""
    directory = Path(directory)
    manifest = json.loads((directory / 'manifest.json').read_text())
    for relative, expected in manifest['files'].items():
        if digest(directory / relative) != expected:
            raise ValueError(f'Bundle checksum mismatch: {relative}')
    return manifest


def available_devices(max_used_mib=1024, max_utilization=20):
    """Select idle visible GPU ordinals without overriding Slurm visibility."""
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.used,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True, check=True)
    inventory = []
    for row in csv.reader(io.StringIO(result.stdout)):
        index, uuid, used, utilization = [item.strip() for item in row]
        inventory.append({'index': index, 'uuid': uuid, 'used': int(used), 'utilization': int(utilization)})
    visible = os.environ.get('CUDA_VISIBLE_DEVICES')
    if visible is None:
        # Match the physical inventory order explicitly so auto detection and CUDA agree.
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(item['uuid'] for item in inventory)
        candidates = inventory
    else:
        candidates = []
        for token in visible.split(','):
            matches = [item for item in inventory if token.strip() == item['index'] or item['uuid'].startswith(token.strip())]
            if len(matches) != 1:
                raise ValueError('Cannot resolve CUDA_VISIBLE_DEVICES; provide explicit --devices for this allocation')
            candidates.append(matches[0])
    devices = [f'cuda:{ordinal}' for ordinal, item in enumerate(candidates)
               if item['used'] <= max_used_mib and item['utilization'] <= max_utilization]
    if not devices:
        raise RuntimeError('No idle visible GPUs found; set explicit --devices or adjust GPU usage thresholds')
    return devices


def forecast_jobs(models, devices):
    """Assign one independent model to each visible GPU, queueing when necessary."""
    jobs = []
    for model in models:
        jobs.append({'model': model, 'shard_index': 0, 'shards': 1,
                     'device': devices[len(jobs) % len(devices)]})
    return jobs


def main():
    """Keep preparation model-free and require an explicit historical launch flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'run', 'report', 'audit'])
    parser.add_argument('--config', default='configs/research/zero_shot_da_history.yaml')
    parser.add_argument('--snapshot')
    parser.add_argument('--bundle')
    parser.add_argument('--output', required=True)
    parser.add_argument('--python-root', default='outputs/zero_shot_envs')
    parser.add_argument('--model-python', action='append', default=[], metavar='MODEL=PYTHON')
    parser.add_argument('--devices', nargs='+', default=['auto'])
    parser.add_argument('--max-gpu-used-mib', type=int, default=1024)
    parser.add_argument('--max-gpu-utilization', type=int, default=20)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--run', action='store_true', help='Actually execute model subprocesses')
    parser.add_argument('--forecast-directory', help='Existing model output root for report')
    args = parser.parse_args()
    if args.action == 'prepare':
        settings = load_config(args.config)['zero_shot_da_probe']
        if settings['minutes'] != 15 or settings['context_days'] < 7 or settings['precision'] != 'float32':
            raise ValueError('This adapter requires quarter-hour targets, >=7 context days and FP32')
        specs = load_config(settings['model_config'])['zero_shot_price_study']['models']
        result = prepare_bundle(args.snapshot or settings['snapshot'], args.output, settings,
                                {name: specs[name] for name in settings['models']})
        print(json.dumps({key: result[key] for key in ('eligible_days', 'requested_days', 'features')}, indent=2))
        return
    if args.bundle is None:
        parser.error('--bundle is required')
    bundle = Path(args.bundle).resolve()
    manifest = verify_bundle(bundle)
    if args.action == 'audit':
        print(json.dumps({key: manifest[key] for key in ('settings', 'eligible_days', 'requested_days', 'features')}, indent=2))
        return
    if args.action == 'report':
        if not args.forecast_directory:
            parser.error('--forecast-directory is required for report')
        report(bundle, args.forecast_directory, args.output)
        return
    models = list(manifest['models'])
    # JSON manifests use sorted keys, so assignments follow the documented name order.
    if args.devices == ['auto']:
        args.devices = available_devices(args.max_gpu_used_mib, args.max_gpu_utilization)[:len(models)]
    if len(args.devices) > len(models):
        parser.error('Use at most one GPU per model (three models)')
    if not args.devices or args.threads < 1 or args.batch_size < 1:
        parser.error('Provide at least one device and positive threads')
    accelerators = [device for device in args.devices if device != 'cpu']
    if len(accelerators) != len(set(accelerators)):
        parser.error('Give each GPU worker a different visible CUDA device')
    explicit = dict(item.split('=', 1) for item in args.model_python)
    commands = []
    jobs = forecast_jobs(models, args.devices)
    for job in jobs:
        model, device = job['model'], job['device']
        shard_name = f"shard-{job['shard_index']}-of-{job['shards']}"
        python = Path(explicit.get(model, Path(args.python_root) / ENV_NAMES[model] / 'bin/python')).absolute()
        if not python.is_file():
            raise FileNotFoundError(f'Missing model environment interpreter: {python}')
        command = [str(python), str(ROOT / 'scripts/run_da_zero_shot_model.py'), '--bundle', str(bundle),
                   '--output', str(Path(args.output).resolve() / model / shard_name), '--model', model,
                   '--shard-index', str(job['shard_index']), '--shards', str(job['shards']),
                   '--device', device, '--threads', str(args.threads), '--batch-size', str(args.batch_size)]
        if args.allow_download:
            command.append('--allow-download')
        if args.preflight_only:
            command.append('--preflight-only')
        commands.append((f'{model}__{shard_name}', command))
    print(json.dumps({'action': 'execute' if args.run else 'plan_only', 'models_in_device_order': models,
        'devices': args.devices, 'jobs': jobs, 'eligible_days': manifest['eligible_days'],
        'daily_forecasts': manifest['eligible_days'] * len(models), 'commands': commands}, indent=2), flush=True)
    if not args.run:
        return
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / '.launcher.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    allocation = destination / 'allocation.json'
    if allocation.exists() and json.loads(allocation.read_text()) != jobs:
        raise ValueError('GPU/shard allocation changed; use the original --devices or a new output directory')
    write_json(allocation, jobs)
    children = []
    def forward_signal(signum, frame):
        # Workers finish their current day and retain all completed checkpoints.
        for child in list(children):
            if child.poll() is None:
                child.send_signal(signum)
    signal.signal(signal.SIGINT, forward_signal)
    signal.signal(signal.SIGTERM, forward_signal)
    def launch(item):
        model, command = item
        with (destination / f'{model}.log').open('a') as stream:
            child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                     cwd=ROOT, env=os.environ.copy())
            children.append(child)
            return model, child.wait()
    def run_lane(lane):
        # Models sharing an accelerator run sequentially; distinct lanes run concurrently.
        return [launch(item) for item in lane]
    lanes = [commands[index::len(args.devices)] for index in range(len(args.devices))]
    with ThreadPoolExecutor(max_workers=len(args.devices)) as pool:
        outcomes = [outcome for lane in pool.map(run_lane, lanes) for outcome in lane]
    print(json.dumps({'exit_codes': dict(outcomes)}, indent=2))
    if any(code for _, code in outcomes):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
