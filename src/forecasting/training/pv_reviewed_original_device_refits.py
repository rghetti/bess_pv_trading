"""Finish frozen reviewed-PV refits on their original physical GPUs."""
from __future__ import annotations

from collections import deque
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from contextlib import ExitStack
import json
import multiprocessing
import os
from pathlib import Path
import time

import pandas as pd

from ...data import config_path
from . import pv_reviewed_gpu_hpo as base
from . import pv_reviewed_gpu0_recovery as recovery
from . import pv_reviewed_multi_gpu_hpo as multi


def _selected_records(root, identity, records):
    """Verify the frozen validation selection and return its distinct trial records."""
    frozen = json.loads((root / 'frozen_selection.json').read_text())
    if frozen.get('identity') != identity or frozen.get('selection') != 'validation_only':
        raise ValueError('Frozen selection identity or selection policy is invalid')
    winners, selected = base._family_winners(records)
    top_three = base._top_three(records)
    expected = {'family_winners': [item['trial'] for item in winners.values()],
                'top_three_by_farm': {str(farm): [item['trial'] for item in ranked]
                                      for farm, ranked in top_three.items()},
                'selected_by_farm': {str(farm): item['trial'] for farm, item in selected.items()}}
    for name, value in expected.items():
        if frozen.get(name) != value:
            raise ValueError(f'Frozen {name} does not match completed validation records')
    unique = {}
    for item in list(winners.values()) + [record for ranked in top_three.values() for record in ranked]:
        trial = item['trial']
        key = (trial['farm_id'], trial['feature'], trial['family'], trial['number'])
        unique.setdefault(key, item)
    return winners, selected, list(unique.values())


def _saved_refit(root, selected, identity, tolerance):
    """Reuse only complete, checksum-valid refits with the original replay guard."""
    trial = selected['trial']
    directory = base._model_directory(root, trial)
    path = directory / 'result.json'
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if (record.get('identity') != identity or record.get('trial') != trial or
            record.get('status') != 'complete' or
            float(record.get('validation_replay_difference_percent', float('inf'))) > tolerance):
        raise ValueError(f'Existing refit is incompatible: {path}')
    original = record.get('original_training_gpu', record.get('gpu'))
    if original != selected['gpu']:
        raise ValueError(f'Existing refit has inconsistent original GPU provenance: {path}')
    for name, checksum in record['checksums'].items():
        if base._sha(directory / name) != checksum:
            raise ValueError(f'Existing refit checksum mismatch: {directory / name}')
    return record


def check_only(config, progress=print):
    """Check devices, data, identity, frozen selections and existing refits."""
    section = config['pv_reviewed_hpo']
    root, saved = recovery._saved_contract(config)
    hardware = multi._hardware(section)
    old_runtime = saved['contract']['runtime']
    current_runtime = {**hardware, 'devices': [{k: v for k, v in item.items() if k != 'free_memory_gib'}
                                               for item in hardware['devices']]}
    if current_runtime != old_runtime:
        raise ValueError('GPU UUIDs, CUDA build, device settings or runtime differ from the saved search')
    plan, prepared, trials, proxy_contract = base.check_only(multi._single_device_proxy(config), progress)
    recovery._verify_contract(config, saved, proxy_contract)
    records, remaining = multi._existing_records(root, trials, saved['identity'], old_runtime['devices'])
    if remaining:
        raise ValueError(f'{len(remaining)} HPO trial records are missing; finish the search before refitting')
    winners, selected, refit_records = _selected_records(root, saved['identity'], records)
    tolerance = float(section['refit_replay_tolerance_nrmse_percent'])
    existing, pending = [], []
    for item in refit_records:
        saved_refit = _saved_refit(root, item, saved['identity'], tolerance)
        if saved_refit is None:
            pending.append(item)
        else:
            existing.append(saved_refit)
    plan.update({'identity': saved['identity'], 'mode': 'original_device_refits', 'runtime': hardware,
                 'completed_trials': len(records), 'existing_refits': len(existing),
                 'pending_refits': len(pending),
                 'pending_by_gpu': {item['visible_id']: sum(record['gpu']['visible_id'] == item['visible_id']
                                                            for record in pending)
                                    for item in hardware['devices']}})
    return plan, prepared, records, winners, selected, existing, pending, saved


def _schedule_refits(jobs, devices, progress, poll_seconds=15):
    """Run pinned refits while waiting for shared GPUs to have sufficient memory."""
    import torch

    specs = {item['visible_id']: item for item in devices}
    queues = {key: deque() for key in specs}
    for job in jobs:
        queues[job['gpu']['visible_id']].append(job)
    thresholds = {key: float(item['minimum_free_gpu_memory_gib']) for key, item in specs.items()}
    last_notice = {key: 0.0 for key in specs}
    pending = {}
    completed_count = 0
    with ExitStack() as stack:
        # GPU 0 uses two slots; shared GPUs 1 and 2 use one each.
        limits = {key: (min(2, item['workers']) if key == '0' else 1) for key, item in specs.items()}
        pools = {key: stack.enter_context(ProcessPoolExecutor(
            max_workers=limits[key], mp_context=multiprocessing.get_context('spawn'),
            initializer=multi._init_device_worker, initargs=(key, item['uuid']))) for key, item in specs.items()}
        while pending or any(queues.values()):
            for index, (key, item) in enumerate(specs.items()):
                running = sum(entry[1] == key for entry in pending.values())
                while queues[key] and running < limits[key]:
                    free, total = torch.cuda.mem_get_info(index)
                    free_gib = free / 1024 ** 3
                    if free_gib < thresholds[key]:
                        now = time.monotonic()
                        if progress and now - last_notice[key] >= 60:
                            progress(f'GPU {key} has {free_gib:.2f} GiB free; waiting for {thresholds[key]:.2f} GiB')
                            last_notice[key] = now
                        break
                    job = queues[key].popleft()
                    pending[pools[key].submit(multi._refit_worker, job)] = (job, key)
                    running += 1
            if not pending:
                time.sleep(poll_seconds)
                continue
            done, _ = wait(pending, timeout=poll_seconds, return_when=FIRST_COMPLETED)
            for future in done:
                job, key = pending.pop(future)
                try:
                    record = future.result()
                except torch.cuda.OutOfMemoryError as error:
                    # Let other refits finish and require more free memory before retrying.
                    limit = specs[key]['total_memory_bytes'] / 1024 ** 3 - 0.25
                    raised = min(limit, thresholds[key] + 1.0)
                    if raised <= thresholds[key]:
                        raise RuntimeError(f"GPU {key} cannot fit {job['trial']['name']} alone") from error
                    thresholds[key] = raised
                    queues[key].appendleft(job)
                    limits[key] = 1
                    if progress:
                        progress(f"GPU {key} ran out of memory; retry {job['trial']['name']} after {raised:.2f} GiB is free")
                    continue
                except Exception as error:
                    for other in pending:
                        other.cancel()
                    raise RuntimeError(f"GPU {key} failed refit {job['trial']['name']} for farm {job['trial']['farm_id']}") from error
                completed_count += 1
                yield record
                if progress:
                    progress(f'Completed {completed_count}/{len(jobs)} remaining refits')


def run_refits(config, progress=print):
    """Complete and report the frozen selection without repeating any search fits."""
    section = config['pv_reviewed_hpo']
    root, saved = recovery._saved_contract(config)
    available = base._available_memory_gib()
    minimum = float(section.get('minimum_available_memory_gib', 0))
    if available is None or available < minimum:
        raise RuntimeError(f'The refits require {minimum:g} GiB available host memory; available: {available}')
    lock = root / 'run.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f'An HPO run may already be active; inspect {lock}') from error
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'mode': 'original_device_refits',
                                 'started_utc': pd.Timestamp.now(tz='UTC').isoformat()}) + '\n')
    try:
        with (root / 'run.log').open('a', buffering=1) as log_file:
            def log(message):
                """Append progress to the original log and the user's terminal."""
                text = f"{pd.Timestamp.now(tz='UTC').isoformat()} {message}"
                print(text, file=log_file, flush=True)
                if progress:
                    progress(text)

            plan, prepared, records, winners, selected, refits, remaining, saved = check_only(config, log)
            base._write_json(root / 'original_device_refits_preflight.json', plan)
            for farm, item in prepared.items():
                log(f'Preparing checked feature cache for farm {farm}')
                base._prepare_cache(root, item, config)
            del prepared
            for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                os.environ[name] = '1'
            jobs = [{'root': str(root), 'trial': item['trial'], 'identity': saved['identity'], 'gpu': item['gpu'],
                     'config': multi._single_device_proxy(config),
                     'expected_validation_score': item['validation']['all']['day_balanced_nrmse_percent']}
                    for item in remaining]
            log(f'Retaining {len(refits)} valid refits and completing {len(jobs)} on their original GPUs')
            base._write_json(root / 'status.json', {'status': 'refitting', 'identity': saved['identity'],
                'hpo_fits': len(records), 'completed': len(records), 'refits_completed': len(refits),
                'training_started': True, 'mode': 'original_device_refits'})
            for record in _schedule_refits(jobs, plan['runtime']['devices'], log):
                refits.append(record)
                base._write_json(root / 'status.json', {'status': 'refitting', 'identity': saved['identity'],
                    'hpo_fits': len(records), 'completed': len(records), 'refits_completed': len(refits),
                    'training_started': True, 'mode': 'original_device_refits'})
            report = config_path(config, section['report_root'])
            base.publish_results(config, root, report, records, winners, selected, refits, saved['identity'])
            pd.DataFrame([{'farm_id': item['trial']['farm_id'], 'trial': item['trial']['name'],
                           'gpu_id': item['gpu']['visible_id'], 'gpu_uuid': item['gpu']['uuid'],
                           'gpu_name': item['gpu']['name']} for item in records]).to_csv(
                               report / 'trial_gpu_provenance.csv', index=False)
            base._write_json(root / 'status.json', {'status': 'complete', 'identity': saved['identity'],
                'hpo_fits': len(records), 'completed': len(records), 'refits_completed': len(refits),
                'report': str(report), 'training_started': True, 'mode': 'original_device_refits'})
            log(f'Complete: {report}')
            return {'output': str(root), 'report': str(report), 'identity': saved['identity']}
    finally:
        lock.unlink(missing_ok=True)
