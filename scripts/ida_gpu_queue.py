"""Schedule model studies across independent GPUs without waiting for study siblings."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import signal
import subprocess
import threading

from scripts.da_zero_shot_probe import ROOT, ENV_NAMES


class _StudyJobQueue:
    """Accept prepared studies while GPU workers consume model jobs concurrently."""

    def __init__(self, models, devices):
        self.models = models
        self.devices = devices
        self.pending = []
        self.condition = threading.Condition()
        self.closed = False

    def add_study(self, directory):
        """Queue every model for one newly validated or prepared input bundle."""
        jobs = []
        for model in self.models:
            output = directory / 'run' / model / 'shard-0-of-1'
            identity = output / 'identity.json'
            pinned = json.loads(identity.read_text())['hardware']['device'] if identity.exists() else None
            if pinned is not None and pinned not in self.devices:
                raise ValueError(f'Existing checkpoint needs {pinned}; include it in --devices: {output}')
            jobs.append((directory, model, output, pinned))
        with self.condition:
            self.pending.extend(jobs)
            self.condition.notify_all()

    def close(self):
        """Mark that input preparation is complete and wake waiting GPUs."""
        with self.condition:
            self.closed = True
            self.condition.notify_all()

    def take(self, device, stop):
        """Return the next job eligible for a GPU, respecting pinned resumes."""
        with self.condition:
            while not stop.is_set():
                index = next((i for i, job in enumerate(self.pending)
                              if job[3] in (None, device)), None)
                if index is not None:
                    return self.pending.pop(index)
                if self.closed and not self.pending:
                    return None
                self.condition.wait(timeout=0.5)
            return None


def _execute_job_queue(queue, devices, python_root, batch_size, threads,
                       allow_download=False, preflight_only=False, prepare=None):
    """Run the live job queue and optionally produce input studies alongside it."""
    mutex = threading.Lock()
    children = []
    stop = threading.Event()
    def interrupt(signum, frame):
        stop.set()
        queue.close()
        with mutex:
            for child in children:
                if child.poll() is None:
                    child.send_signal(signum)
        raise KeyboardInterrupt
    signal.signal(signal.SIGINT, interrupt)
    signal.signal(signal.SIGTERM, interrupt)
    def consume(device):
        while not stop.is_set():
            job = queue.take(device, stop)
            if job is None:
                return
            directory, model, output, pinned = job
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
        preparation_error = None
        prepared_result = None
        if prepare is not None:
            try:
                prepared_result = prepare(queue.add_study)
            except BaseException as error:
                preparation_error = error
            finally:
                queue.close()
        else:
            queue.close()
        failures = []
        for future in futures:
            try:
                future.result()
            except Exception as error:
                failures.append(str(error))
        if failures:
            raise RuntimeError('\n'.join(failures))
        if preparation_error is not None:
            raise preparation_error
    if stop.is_set():
        raise SystemExit(130)
    return prepared_result


def execute_queue(studies, models, devices, python_root, batch_size, threads, allow_download=False, preflight_only=False):
    """Run a fixed set of studies, retaining device affinity for existing fits."""
    queue = _StudyJobQueue(models, devices)
    for directory in studies:
        queue.add_study(directory)
    return _execute_job_queue(queue, devices, python_root, batch_size, threads,
                              allow_download, preflight_only)


def execute_queue_while_preparing(models, devices, python_root, batch_size, threads, prepare,
                                  allow_download=False, preflight_only=False):
    """Start each study's GPU jobs as soon as its inputs are ready."""
    queue = _StudyJobQueue(models, devices)
    return _execute_job_queue(queue, devices, python_root, batch_size, threads,
                              allow_download, preflight_only, prepare=prepare)
