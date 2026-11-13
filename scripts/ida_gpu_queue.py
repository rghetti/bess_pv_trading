"""Schedule model studies across independent GPUs without waiting for study siblings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import signal
import subprocess
import threading

from scripts.da_zero_shot_probe import ROOT, ENV_NAMES


def execute_queue(studies, models, devices, python_root, batch_size, threads, allow_download=False, preflight_only=False):
    """Run one model process per GPU, retaining device affinity for existing fits."""
    jobs = []
    for directory in studies:
        for model in models:
            output = directory / 'run' / model / 'shard-0-of-1'
            identity = output / 'identity.json'
            pinned = json.loads(identity.read_text())['hardware']['device'] if identity.exists() else None
            if pinned is not None and pinned not in devices:
                raise ValueError(f'Existing checkpoint needs {pinned}; include it in --devices: {output}')
            jobs.append((directory, model, output, pinned))
    pending = list(jobs)
    mutex = threading.Lock()
    children = []
    stop = threading.Event()
    def interrupt(signum, frame):
        stop.set()
        with mutex:
            for child in children:
                if child.poll() is None:
                    child.send_signal(signum)
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    def consume(device):
        while not stop.is_set():
            with mutex:
                index = next((i for i, job in enumerate(pending) if job[3] in (None, device)), None)
                if index is None:
                    return
                directory, model, output, pinned = pending.pop(index)
            python = (Path(python_root) / ENV_NAMES[model] / 'bin/python').absolute()
            if not python.is_file():
                raise FileNotFoundError(python)
            output.parent.mkdir(parents=True, exist_ok=True)
            command = [str(python), str(ROOT / 'scripts/run_da_zero_shot_model.py'), '--bundle', str((directory / 'inputs').absolute()),
                       '--output', str(output.absolute()), '--model', model, '--device', device,
                       '--batch-size', str(batch_size), '--threads', str(threads)]
            if allow_download:
                command.append('--allow-download')
            if preflight_only:
                command.append('--preflight-only')
            print(f'{device}: starting {directory} / {model}', flush=True)
            with (directory / 'run' / f'{model}__shard-0-of-1.log').open('a') as log:
                child = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                with mutex:
                    children.append(child)
                code = child.wait()
            if code:
                raise RuntimeError(f'Model worker failed ({code}): {directory} / {model}; see its log')
            print(f'{device}: completed {directory} / {model}', flush=True)
    with ThreadPoolExecutor(max_workers=len(devices)) as executor:
        futures = [executor.submit(consume, device) for device in devices]
        failures = []
        for future in futures:
            try:
                future.result()
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError('\n'.join(failures))
    if stop.is_set():
        raise SystemExit(130)
