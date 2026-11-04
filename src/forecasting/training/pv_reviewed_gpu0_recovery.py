"""Resume a reviewed multi-GPU search on its original physical GPU 0.

This recovery entry point preserves the original model code, frozen grid,
validation scores and run identity while changing only the execution device
for unfinished work and refits.
"""
from __future__ import annotations

from copy import deepcopy
import json
import os
from pathlib import Path

import pandas as pd

from ...data import config_path
from . import pv_reviewed_gpu_hpo as base
from . import pv_reviewed_multi_gpu_hpo as multi
from .pv_reviewed_hpo_common import identity_digest


def _saved_contract(config):
    """Read and verify the immutable identity of the interrupted search."""
    section = config['pv_reviewed_hpo']
    root = config_path(config, section['output_root']) / section['run_id']
    path = root / 'identity.json'
    if not path.is_file():
        raise ValueError(f'No interrupted multi-GPU search identity exists at {path}')
    saved = json.loads(path.read_text())
    if identity_digest(saved['contract']) != saved['identity']:
        raise ValueError('Saved search identity does not match its contract')
    return root, saved


def _gpu0_details(saved, require_free=False):
    """Verify that the only visible GPU is the original physical GPU 0."""
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '0':
        raise ValueError('Set CUDA_VISIBLE_DEVICES=0 for GPU 0-only recovery')
    original = next((item for item in saved['contract']['runtime']['devices']
                     if item['visible_id'] == '0'), None)
    if original is None:
        raise ValueError('The saved search did not include physical GPU 0')
    section = {'runtime': {'device': 'cuda:0', 'minimum_free_gpu_memory_gib':
                           original['minimum_free_gpu_memory_gib']}}
    details = base._runtime_details(section, require_free_memory=require_free)
    import torch
    properties = torch.cuda.get_device_properties(0)
    if (str(properties.uuid) != original['uuid'] or properties.name != original['name'] or
            [properties.major, properties.minor] != original['compute_capability'] or
            details['total_memory_bytes'] != original['total_memory_bytes']):
        raise ValueError('Visible GPU 0 differs from the GPU in the saved search identity')
    old_runtime = saved['contract']['runtime']
    for name in ['torch_cuda_version', 'compiled_architectures', 'cublas_workspace_config']:
        if details[name] != old_runtime[name]:
            raise ValueError(f'CUDA runtime setting {name} differs from the saved search')
    # The current physical GPU inherits its original two-worker limit.
    return {**original, 'free_memory_gib': details['free_memory_gib']}


def _verify_contract(config, saved, proxy_contract):
    """Require the original training code, dataset, configuration and versions."""
    reconstructed = deepcopy(proxy_contract)
    section = deepcopy(config['pv_reviewed_hpo'])
    section.pop('resume', None)
    reconstructed['configuration'] = section
    reconstructed['runtime'] = saved['contract']['runtime']
    reconstructed['code_sha256'][Path(multi.__file__).name] = base._sha(Path(multi.__file__))
    if reconstructed != saved['contract'] or identity_digest(reconstructed) != saved['identity']:
        raise ValueError('Training configuration, data, code or versions differ from the interrupted search')


def check_only(config, progress=print):
    """Check the original identity and reviewed datasets without fitting a model."""
    root, saved = _saved_contract(config)
    gpu = _gpu0_details(saved)
    plan, prepared, trials, contract = base.check_only(multi._single_device_proxy(config), progress)
    _verify_contract(config, saved, contract)
    completed, remaining = multi._existing_records(root, trials, saved['identity'],
                                                    saved['contract']['runtime']['devices'])
    plan.update({'identity': saved['identity'], 'recovery': 'gpu0_only', 'gpu0': gpu,
                 'completed_trials': len(completed), 'remaining_trials': len(remaining),
                 'refit_policy': 'GPU 0 with original validation replay tolerance'})
    return plan, prepared, trials, saved


def _refit_worker(job):
    """Refit on GPU 0 and retain both original trial and refit GPU provenance."""
    record = multi._refit_worker(job)
    record['original_training_gpu'] = job['original_training_gpu']
    path = base._model_directory(Path(job['root']), job['trial']) / 'result.json'
    base._write_json(path, record)
    return record


def run_hpo(config, progress=print):
    """Resume unfinished trials on GPU 0 under the original search identity."""
    root, saved = _saved_contract(config)
    gpu = _gpu0_details(saved, require_free=True)
    available = base._available_memory_gib()
    minimum = float(config['pv_reviewed_hpo'].get('minimum_available_memory_gib', 0))
    if available is None or available < minimum:
        raise RuntimeError(f'The search requires {minimum:g} GiB available host memory; available: {available}')
    lock = root / 'run.lock'
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError as error:
        raise RuntimeError(f'An HPO run may already be active; inspect {lock}') from error
    with os.fdopen(descriptor, 'w') as stream:
        stream.write(json.dumps({'pid': os.getpid(), 'mode': 'gpu0_recovery',
                                 'started_utc': pd.Timestamp.now(tz='UTC').isoformat()}) + '\n')
    try:
        with (root / 'run.log').open('a', buffering=1) as log_file:
            def log(message):
                """Show recovery progress and append it to the original run log."""
                text = f"{pd.Timestamp.now(tz='UTC').isoformat()} {message}"
                print(text, file=log_file, flush=True)
                if progress:
                    progress(text)

            plan, prepared, trials, saved = check_only(config, log)
            base._write_json(root / 'gpu0_recovery_preflight.json', plan)
            for farm, item in prepared.items():
                log(f'Preparing checked feature cache for farm {farm}')
                base._prepare_cache(root, item, config)
            del prepared
            for name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS']:
                os.environ[name] = '1'
            records, remaining = multi._existing_records(root, trials, saved['identity'],
                                                          saved['contract']['runtime']['devices'])
            jobs = [{'root': str(root), 'trial': trial, 'identity': saved['identity'],
                     'config': multi._single_device_proxy(config)} for trial in remaining]
            log(f'GPU 0 recovery: {len(records)} completed trials retained, {len(jobs)} remaining, {gpu["workers"]} workers')
            base._write_json(root / 'status.json', {'status': 'searching', 'identity': saved['identity'],
                'hpo_fits': len(trials), 'completed': len(records), 'training_started': True, 'mode': 'gpu0_recovery'})
            for record in multi._schedule(jobs, multi._trial_worker, [gpu], log):
                records.append(record)
                if len(records) % 20 == 0 or len(records) == len(trials):
                    base._write_json(root / 'status.json', {'status': 'searching', 'identity': saved['identity'],
                        'hpo_fits': len(trials), 'completed': len(records), 'training_started': True,
                        'mode': 'gpu0_recovery'})
            winners, selected = base._family_winners(records)
            top_three = base._top_three(records)
            if len(winners) != plan['family_winner_refits'] or len(selected) != len(config['pv_reviewed_hpo']['farm_ids']):
                raise ValueError('Completed search does not cover all family/feature/farm groups')
            base._write_json(root / 'frozen_selection.json', {'identity': saved['identity'], 'selection': 'validation_only',
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
            refit_gpu = {field: gpu[field] for field in ('visible_id', 'uuid', 'name')}
            refit_jobs = [{'root': str(root), 'trial': item['trial'], 'identity': saved['identity'],
                           'config': multi._single_device_proxy(config), 'gpu': refit_gpu,
                           'original_training_gpu': item['gpu'],
                           'expected_validation_score': item['validation']['all']['day_balanced_nrmse_percent']}
                          for item in refit_records]
            log(f'Refitting {len(refit_jobs)} validation-selected trials on GPU 0 with strict validation replay')
            base._write_json(root / 'status.json', {'status': 'refitting', 'identity': saved['identity'],
                'hpo_fits': len(trials), 'completed': len(trials), 'refits_completed': 0,
                'training_started': True, 'mode': 'gpu0_recovery'})
            refits = []
            for record in multi._schedule(refit_jobs, _refit_worker, [gpu], log):
                refits.append(record)
                if len(refits) % 20 == 0 or len(refits) == len(refit_jobs):
                    base._write_json(root / 'status.json', {'status': 'refitting', 'identity': saved['identity'],
                        'hpo_fits': len(trials), 'completed': len(trials), 'refits_completed': len(refits),
                        'training_started': True, 'mode': 'gpu0_recovery'})
            report = config_path(config, config['pv_reviewed_hpo']['report_root'])
            base.publish_results(config, root, report, records, winners, selected, refits, saved['identity'])
            pd.DataFrame([{'farm_id': item['trial']['farm_id'], 'trial': item['trial']['name'],
                           'gpu_id': item['gpu']['visible_id'], 'gpu_uuid': item['gpu']['uuid'],
                           'gpu_name': item['gpu']['name']} for item in records]).to_csv(
                               report / 'trial_gpu_provenance.csv', index=False)
            base._write_json(root / 'status.json', {'status': 'complete', 'identity': saved['identity'],
                'hpo_fits': len(trials), 'completed': len(trials), 'refits_completed': len(refits),
                'report': str(report), 'training_started': True, 'mode': 'gpu0_recovery'})
            log(f'Complete: {report}')
            return {'output': str(root), 'report': str(report), 'identity': saved['identity']}
    finally:
        lock.unlink(missing_ok=True)
