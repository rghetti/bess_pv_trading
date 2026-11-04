"""Isolated multi-GPU scheduler for the reviewed four-cycle PV search."""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
from copy import deepcopy
import json
import multiprocessing
import os
from pathlib import Path
import time
import warnings

import pandas as pd

from ...data import config_path
from . import pv_reviewed_gpu_hpo as base
from .pv_reviewed_hpo_common import identity_digest, source_masks, score_predictions


def _device_specs(section):
    """Validate the explicit physical device map and per-device worker limits."""
    specs = section.get('runtime', {}).get('devices', [])
    if not specs:
        raise ValueError('Configure at least one GPU in runtime.devices')
    ids = [str(item['visible_id']) for item in specs]
    if len(ids) != len(set(ids)) or any(not item for item in ids):
        raise ValueError('GPU identifiers must be nonempty and unique')
    if any(int(item['workers']) < 1 or float(item['minimum_free_gpu_memory_gib']) < 0 for item in specs):
        raise ValueError('Each GPU needs a positive worker limit and a nonnegative free-memory guard')
    if sum(int(item['workers']) for item in specs) != int(section['workers']):
        raise ValueError('workers must equal the sum of per-GPU worker limits')
    available = len(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else os.cpu_count()
    if available is not None and int(section['workers']) > available:
        raise ValueError(f'Only {available} CPU threads are available for {section["workers"]} GPU workers')
    return specs


def _hardware(section, require_free=False):
    """Check every selected GPU and its runtime memory without changing data."""
    specs = _device_specs(section)
    selected = [str(item['visible_id']) for item in specs]
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '').split(',')
    if visible != selected:
        raise ValueError(f"Set CUDA_VISIBLE_DEVICES={','.join(selected)} to match the configured physical GPU IDs")
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in {':4096:8', ':16:8'}:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != len(specs):
        raise RuntimeError('All configured GPUs must be visible to the installed PyTorch CUDA build')
    details = []
    for index, item in enumerate(specs):
        device = f'cuda:{index}'
        properties = torch.cuda.get_device_properties(index)
        # A tiny kernel catches incompatible architecture binaries before scheduling.
        try:
            torch.ones(1, device=device).add_(1).cpu()
            torch.cuda.synchronize(index)
        except Exception as error:
            raise RuntimeError(f'PyTorch cannot execute on physical GPU {selected[index]}: {error}') from error
        free, total = torch.cuda.mem_get_info(index)
        free_gib = free / 1024 ** 3
        threshold = float(item['minimum_free_gpu_memory_gib'])
        if require_free and free_gib < threshold:
            raise RuntimeError(f'GPU {selected[index]} requires {threshold:g} GiB free; currently {free_gib:.2f} GiB')
        details.append({'visible_id': selected[index], 'name': properties.name,
                        'uuid': str(properties.uuid), 'compute_capability': [properties.major, properties.minor],
                        'total_memory_bytes': total, 'free_memory_gib': free_gib,
                        'workers': int(item['workers']), 'minimum_free_gpu_memory_gib': threshold})
    return {'devices': details, 'torch_cuda_version': torch.version.cuda,
            'compiled_architectures': torch.cuda.get_arch_list(), 'cublas_workspace_config': workspace}


def _single_device_proxy(config):
    """Reuse the reviewed-data preflight without changing its feature preparation."""
    proxy = deepcopy(config)
    section = proxy['pv_reviewed_hpo']
    section['workers'] = 1
    section['runtime'] = {'device': 'cuda:0', 'maximum_gpu_workers': 1,
                          'minimum_free_gpu_memory_gib': 0}
    return proxy


def check_only(config, progress=print):
    """Check all devices and reviewed data without fitting a model."""
    section = config['pv_reviewed_hpo']
    _device_specs(section)
    runtime = _hardware(section)
    plan, prepared, trials, contract = base.check_only(_single_device_proxy(config), progress)
    identity_config = deepcopy(section)
    identity_config.pop('resume', None)
    runtime_identity = deepcopy(runtime)
    for item in runtime_identity['devices']:
        item.pop('free_memory_gib')
    contract['configuration'] = identity_config
    contract['runtime'] = runtime_identity
    contract['code_sha256'][Path(__file__).name] = base._sha(Path(__file__))
    identity = identity_digest(contract)
    plan.update({'identity': identity, 'workers': section['workers'], 'runtime': runtime,
                 'minimum_available_memory_gib': section.get('minimum_available_memory_gib')})
    return plan, prepared, trials, contract


def _init_device_worker(visible_id, expected_uuid):
    """Expose only the assigned physical GPU to this spawned worker."""
    os.environ['CUDA_VISIBLE_DEVICES'] = visible_id
    import torch
    if not torch.cuda.is_available() or str(torch.cuda.get_device_properties(0).uuid) != expected_uuid:
        raise RuntimeError(f'Worker could not access its assigned physical GPU {visible_id}')


def _execute_trial(job):
    """Fit a trial and atomically record the physical GPU with its validation scores."""
    import sklearn
    from threadpoolctl import threadpool_limits

    root = Path(job['root'])
    trial = job['trial']
    path = base._trial_path(root, trial)
    if path.exists():
        record = json.loads(path.read_text())
        if record['identity'] != job['identity'] or record['trial'] != trial or record.get('gpu') != job['gpu']:
            raise ValueError(f"Existing trial provenance mismatch for {trial['name']}")
        return record
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=1):
        model, info, frame, masks, matrix, packed, rating = base._fit_candidate(root, trial, job['config'])
        prediction = base._predict_candidate(model, trial, matrix, packed, frame, ['validation'], rating)
        scores = {name: score_predictions(frame, prediction, mask, rating)
                  for name, mask in source_masks(frame, masks['validation']).items()}
    record = {'status': 'complete', 'identity': job['identity'], 'trial': trial, 'gpu': job['gpu'],
              'validation': scores, 'training': info, 'elapsed_seconds': time.perf_counter() - started,
              'warnings': [str(item.message) for item in caught], 'test_targets_scored': False}
    base._write_json(path, record)
    return record


def _trial_worker(job):
    """Run a trial and free unused CUDA allocations in this device worker."""
    try:
        return _execute_trial(job)
    finally:
        base._clear_cuda_cache()


def _refit_worker(job):
    """Refit on the exact physical GPU that produced the selected validation score."""
    try:
        path = base._model_directory(Path(job['root']), job['trial']) / 'result.json'
        if path.exists():
            previous = json.loads(path.read_text())
            if previous.get('gpu', job['gpu']) != job['gpu']:
                raise ValueError('Refit GPU provenance differs from the selected trial')
        record = base._execute_refit(job)
        record['gpu'] = job['gpu']
        base._write_json(path, record)
        return record
    finally:
        base._clear_cuda_cache()


def _schedule(jobs, worker, devices, progress):
    """Dispatch new jobs dynamically and keep retries and refits on their GPU."""
    import torch

    specs = {item['visible_id']: item for item in devices}
    available = deque()
    pinned = {key: deque() for key in specs}
    for job in jobs:
        target = job.get('gpu', {}).get('visible_id')
        if target is None:
            available.append(job)
        elif target in pinned:
            pinned[target].append((job, False))
        else:
            raise ValueError(f'Job refers to an unconfigured GPU {target}')
    limits = {key: item['workers'] for key, item in specs.items()}
    pending = {}
    finished = 0
    with ExitStack() as stack:
        pools = {key: stack.enter_context(ProcessPoolExecutor(
            max_workers=item['workers'], mp_context=multiprocessing.get_context('spawn'),
            initializer=_init_device_worker, initargs=(key, item['uuid']))) for key, item in specs.items()}
        while pending or available or any(pinned.values()):
            # Refit and retry jobs wait for their own GPU; unassigned trials fill free slots.
            for key, item in specs.items():
                while sum(entry[1] == key for entry in pending.values()) < limits[key]:
                    active_here = [entry for entry in pending.values() if entry[1] == key]
                    if any(entry[2] for entry in active_here):
                        break
                    if pinned[key]:
                        job, serial = pinned[key][0]
                        if serial and active_here:
                            break
                        pinned[key].popleft()
                    elif available:
                        job, serial = available.popleft(), False
                    else:
                        break
                    assigned = {**job, 'gpu': {field: item[field] for field in ('visible_id', 'uuid', 'name')}}
                    pending[pools[key].submit(worker, assigned)] = (assigned, key, serial)
                    if serial:
                        break
            if not pending:
                raise RuntimeError('No schedulable jobs remain for the configured GPU devices')
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                job, key, serial = pending.pop(future)
                try:
                    record = future.result()
                except torch.cuda.OutOfMemoryError as error:
                    if serial:
                        raise RuntimeError(f"GPU {key} could not fit {job['trial']['name']} alone") from error
                    limits[key] = max(1, limits[key] - 1)
                    pinned[key].appendleft((job, True))
                    if progress:
                        progress(f"GPU {key} ran out of memory; retrying {job['trial']['name']} alone and reducing its limit to {limits[key]}")
                    continue
                except Exception as error:
                    for future_pending in pending:
                        future_pending.cancel()
                    raise RuntimeError(f"GPU {key} failed trial {job['trial']['name']} for farm {job['trial']['farm_id']}") from error
                finished += 1
                yield record
                if progress and (finished % 20 == 0 or finished == len(jobs)):
                    progress(f'Completed {finished}/{len(jobs)} scheduled jobs')


def _existing_records(root, trials, identity, gpu_details):
    """Resume complete matching trial records on their recorded physical devices."""
    allowed = {item['visible_id']: {field: item[field] for field in ('visible_id', 'uuid', 'name')}
               for item in gpu_details}
    records, remaining = [], []
    for trial in trials:
        path = base._trial_path(root, trial)
        if not path.exists():
            remaining.append(trial)
            continue
        record = json.loads(path.read_text())
        gpu = record.get('gpu', {})
        if (record.get('identity') != identity or record.get('trial') != trial or
                record.get('status') != 'complete' or allowed.get(gpu.get('visible_id')) != gpu):
            raise ValueError(f"Existing record has incompatible identity or GPU provenance: {path}")
        records.append(record)
    return records, remaining


def run_hpo(config, progress=print):
    """Run a separate three-GPU search after an explicit user invocation."""
    section = config['pv_reviewed_hpo']
    devices = _hardware(section, require_free=True)['devices']
    available_memory = base._available_memory_gib()
    minimum = float(section.get('minimum_available_memory_gib', 0))
    if available_memory is None or available_memory < minimum:
        raise RuntimeError(f'The search requires {minimum:g} GiB available host memory; available: {available_memory}')
    root = config_path(config, section['output_root']) / section['run_id']
    report = config_path(config, section['report_root'])
    root.mkdir(parents=True, exist_ok=True)
    lock = root / 'run.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f'An HPO run may already be active; inspect {lock}') from error
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'started_utc': pd.Timestamp.now(tz='UTC').isoformat()}) + '\n')
    try:
        with (root / 'run.log').open('a', buffering=1) as log_file:
            def log(message):
                """Record progress both in the run directory and the user's terminal."""
                text = f"{pd.Timestamp.now(tz='UTC').isoformat()} {message}"
                print(text, file=log_file, flush=True)
                if progress:
                    progress(text)

            plan, prepared, trials, contract = check_only(config, log)
            identity_path = root / 'identity.json'
            if identity_path.exists():
                old = json.loads(identity_path.read_text())
                if not section['resume'] or old['identity'] != plan['identity']:
                    raise ValueError('Existing run has a different identity or resume is disabled')
            else:
                existing = [path.name for path in root.iterdir() if path.name not in {'run.lock', 'run.log'}]
                if existing:
                    raise ValueError('Existing output has no matching identity; choose a new run ID')
                base._write_json(identity_path, {'identity': plan['identity'], 'contract': contract})
            base._write_json(root / 'preflight.json', plan)
            for farm_id, item in prepared.items():
                log(f'Preparing checked feature cache for farm {farm_id}')
                base._prepare_cache(root, item, config)
            del prepared
            for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                os.environ[name] = '1'
            records, remaining = _existing_records(root, trials, plan['identity'], devices)
            jobs = [{'root': str(root), 'trial': trial, 'identity': plan['identity'],
                     'config': _single_device_proxy(config)} for trial in remaining]
            log(f'Launching {len(jobs)} remaining fits on {len(devices)} GPUs; {len(records)} already complete')
            base._write_json(root / 'status.json', {'status': 'searching', 'identity': plan['identity'],
                'hpo_fits': len(trials), 'completed': len(records), 'training_started': True})
            for record in _schedule(jobs, _trial_worker, devices, log):
                records.append(record)
                if len(records) % 20 == 0 or len(records) == len(trials):
                    base._write_json(root / 'status.json', {'status': 'searching', 'identity': plan['identity'],
                        'hpo_fits': len(trials), 'completed': len(records), 'training_started': True})
            winners, selected = base._family_winners(records)
            top_three = base._top_three(records)
            if len(winners) != plan['family_winner_refits'] or len(selected) != len(section['farm_ids']):
                raise ValueError('Completed search does not cover all family/feature/farm groups')
            base._write_json(root / 'frozen_selection.json', {'identity': plan['identity'], 'selection': 'validation_only',
                'family_winners': [item['trial'] for item in winners.values()],
                'top_three_by_farm': {str(farm): [item['trial'] for item in ranked] for farm, ranked in top_three.items()},
                'selected_by_farm': {str(farm): item['trial'] for farm, item in selected.items()}})
            refit_records = list(winners.values())
            known = {(item['trial']['farm_id'], item['trial']['feature'], item['trial']['family'],
                      item['trial']['number']) for item in refit_records}
            for ranked in top_three.values():
                for item in ranked:
                    trial = item['trial']
                    key = (trial['farm_id'], trial['feature'], trial['family'], trial['number'])
                    if key not in known:
                        refit_records.append(item)
                        known.add(key)
            refit_jobs = [{'root': str(root), 'trial': item['trial'], 'identity': plan['identity'],
                           'config': _single_device_proxy(config), 'gpu': item['gpu'],
                           'expected_validation_score': item['validation']['all']['day_balanced_nrmse_percent']}
                          for item in refit_records]
            log(f'Refitting {len(refit_jobs)} validation-selected configurations on their original GPUs')
            base._write_json(root / 'status.json', {'status': 'refitting', 'identity': plan['identity'],
                'hpo_fits': len(trials), 'completed': len(trials), 'refits_completed': 0, 'training_started': True})
            refits = []
            for record in _schedule(refit_jobs, _refit_worker, devices, log):
                refits.append(record)
                if len(refits) % 20 == 0 or len(refits) == len(refit_jobs):
                    base._write_json(root / 'status.json', {'status': 'refitting', 'identity': plan['identity'],
                        'hpo_fits': len(trials), 'completed': len(trials), 'refits_completed': len(refits),
                        'training_started': True})
            base.publish_results(config, root, report, records, winners, selected, refits, plan['identity'])
            # Make physical-device provenance easily reviewable alongside the metrics.
            pd.DataFrame([{'farm_id': item['trial']['farm_id'], 'trial': item['trial']['name'],
                           'gpu_id': item['gpu']['visible_id'], 'gpu_uuid': item['gpu']['uuid'],
                           'gpu_name': item['gpu']['name']} for item in records]).to_csv(
                               report / 'trial_gpu_provenance.csv', index=False)
            base._write_json(root / 'status.json', {'status': 'complete', 'identity': plan['identity'],
                'hpo_fits': len(trials), 'completed': len(trials), 'refits_completed': len(refits),
                'report': str(report), 'training_started': True})
            log(f'Complete: {report}')
            return {'output': str(root), 'report': str(report), 'identity': plan['identity']}
    finally:
        lock.unlink(missing_ok=True)
