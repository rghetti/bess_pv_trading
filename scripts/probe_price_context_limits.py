"""Bounded synthetic GPU capacity checks for frozen price foundation models.

Only artificial target/covariate sequences are forecast. Each candidate runs in
its model-specific environment and process so timeouts and OOMs are isolated.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
ENVS = {'chronos2': 'chronos', 't0_gamma': 't0', 'timesfm3': 'timesfm'}
CANDIDATES = {'chronos2': [60, 85, 86], 't0_gamma': [60, 90, 120, 180, 365],
              'timesfm3': [60, 120, 159, 160, 161]}
CAPS = {'chronos2': 8192, 't0_gamma': None, 'timesfm3': 15360}


def write_json(path, data):
    """Persist one complete probe record; outputs contain no historical inputs."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')


def probe(args):
    """Load one pinned checkpoint and forecast one synthetic context in FP32."""
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    result = {'model': args.model, 'days': args.days, 'steps': args.days * 96 + args.extra_steps,
              'extra_steps': args.extra_steps, 'synthetic_covariates': args.covariates,
              'batch_size': 1, 'horizon': 100, 'precision': 'float32', 'physical_gpu': args.gpu,
              'python': platform.python_version(), 'status': 'failed', 'configured_step_cap': CAPS[args.model]}
    started = time.monotonic()
    try:
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download
        sys.path.insert(0, str(ROOT))
        from scripts.run_da_zero_shot_model import load_model

        spec = json.loads(args.spec)
        result['repository'] = spec['repository']
        result['revision'] = spec['revision']
        package = spec['package'].split('==')[0].split('[')[0]
        result['model_package'] = importlib.metadata.version(package)
        result['torch'] = torch.__version__
        result['cuda'] = torch.version.cuda
        torch.set_num_threads(1)
        torch.manual_seed(1729)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_device(0)
        result['gpu_name'] = torch.cuda.get_device_name(0)
        result['gpu_capability'] = list(torch.cuda.get_device_capability(0))
        result['total_vram_bytes'] = torch.cuda.get_device_properties(0).total_memory
        checkpoint = snapshot_download(spec['repository'], revision=spec['revision'], local_files_only=True)
        predictor = load_model(args.model, checkpoint, 'cuda:0', torch, batch_size=1)
        result['load_seconds'] = time.monotonic() - started
        result['adapter_sha256'] = hashlib.sha256((ROOT / 'scripts/run_da_zero_shot_model.py').read_bytes()).hexdigest()
        attended_shapes = []
        if args.model == 't0_gamma':
            # Capture the actual transformer input to detect hidden context truncation.
            for cell in predictor.__closure__ or ():
                candidate_model = cell.cell_contents
                if hasattr(candidate_model, 'transformer'):
                    candidate_model.transformer.register_forward_pre_hook(
                        lambda module, inputs: attended_shapes.append([list(value.shape) for value in inputs if hasattr(value, 'shape')]))
                    break
        torch.cuda.reset_peak_memory_stats()
        steps = result['steps']
        context = (100 + 30 * np.sin(np.arange(steps, dtype=np.float32) * (2 * np.pi / 96)))[None, :]
        covariates = None
        if args.covariates:
            covariates = {f'synthetic_{i}': (i + np.sin(np.arange(steps + 100, dtype=np.float32) / (30 + i)))[None, :]
                          for i in range(args.covariates)}
        infer_start = time.monotonic()
        with torch.inference_mode():
            output = np.asarray(predictor(context, 100, [i / 10 for i in range(1, 10)], covariates))
        torch.cuda.synchronize()
        result['inference_seconds'] = time.monotonic() - infer_start
        if output.shape != (1, 100, 9) or not np.isfinite(output).all():
            raise ValueError(f'Invalid synthetic output: {output.shape}')
        result['output_shape'] = list(output.shape)
        if attended_shapes:
            result['transformer_input_shapes'] = attended_shapes
        result['output_sha256'] = hashlib.sha256(output.tobytes()).hexdigest()
        result['status'] = 'passed'
    except Exception as error:
        result['error'] = f'{type(error).__name__}: {error}'
        if 'would be truncated' in str(error):
            result['status'] = 'rejected_context_cap'
        elif 'out of memory' in str(error).lower():
            result['status'] = 'out_of_memory'
    finally:
        if 'torch' in locals() and torch.cuda.is_available():
            result['peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
            result['peak_reserved_bytes'] = torch.cuda.max_memory_reserved()
        result['elapsed_seconds'] = time.monotonic() - started
        write_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


def run_model(model, gpu, spec, args):
    """Run sequential bounded candidates on one GPU without sharing its memory."""
    executable = Path(args.python_root).absolute() / ENVS[model] / 'bin/python'
    destination = Path(args.output) / model
    records = []
    start = time.monotonic()
    def candidate(days, extra_steps=0, covariates=0):
        name = f'{days}d_extra{extra_steps}_cov{covariates}'
        output = destination / f'{name}.json'
        command = [str(executable), str(Path(__file__).absolute()), '--worker', '--model', model,
                   '--gpu', str(gpu), '--days', str(days), '--extra-steps', str(extra_steps),
                   '--covariates', str(covariates), '--spec', json.dumps(spec), '--output', str(output)]
        destination.mkdir(parents=True, exist_ok=True)
        output.unlink(missing_ok=True)
        with (destination / f'{name}.log').open('w') as log:
            try:
                completed = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, timeout=args.timeout)
                if output.exists():
                    record = json.loads(output.read_text())
                else:
                    record = {'model': model, 'days': days, 'steps': days*96+extra_steps,
                              'status': 'worker_failed', 'exit_code': completed.returncode,
                              'synthetic_covariates': covariates, 'extra_steps': extra_steps}
            except subprocess.TimeoutExpired:
                record = {'model': model, 'days': days, 'steps': days*96+extra_steps,
                          'status': 'timeout', 'timeout_seconds': args.timeout,
                          'synthetic_covariates': covariates, 'extra_steps': extra_steps}
            write_json(output, record)
        records.append(record)
        return record
    for days in CANDIDATES[model]:
        if time.monotonic() - start > args.model_budget_seconds:
            break
        record = candidate(days)
        # A memory failure cannot justify further larger allocations on a shared GPU.
        if record['status'] in ('out_of_memory', 'timeout', 'worker_failed', 'failed'):
            break
    passed = [item for item in records if item['status'] == 'passed']
    if passed and time.monotonic() - start < args.model_budget_seconds:
        longest = max(passed, key=lambda item: item['steps'])
        candidate(longest['days'], covariates=4)
    if model == 'timesfm3' and time.monotonic() - start < args.model_budget_seconds:
        candidate(160, extra_steps=4)
    return records


def main():
    """Launch independent model probes and summarize measured limits honestly."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', action='store_true')
    parser.add_argument('--model', choices=list(ENVS))
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--days', type=int, default=60)
    parser.add_argument('--extra-steps', type=int, default=0)
    parser.add_argument('--covariates', type=int, default=0)
    parser.add_argument('--spec')
    parser.add_argument('--output', default='outputs/price_context_capacity')
    parser.add_argument('--python-root', default='outputs/zero_shot_envs')
    parser.add_argument('--gpus', nargs=3, type=int, default=[0, 1, 2])
    parser.add_argument('--timeout', type=int, default=60)
    parser.add_argument('--model-budget-seconds', type=int, default=120)
    args = parser.parse_args()
    if args.worker:
        probe(args)
        return
    from bess_pv_trading.config import load_config
    specs = load_config(ROOT / 'configs/research/zero_shot_price_study.yaml')['zero_shot_price_study']['models']
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(run_model, model, gpu, specs[model], args) for model, gpu in zip(ENVS, args.gpus)]
        results = [record for future in futures for record in future.result()]
    destination = Path(args.output)
    write_json(destination / 'results.json', results)
    lines = ['# Synthetic context-capacity probe', '',
             'Batch size 1, FP32, one GPU per model, 100-step forecast and nine quantiles. '
             'Only artificial sequences were used. Largest tested success is not a proven maximum or an accuracy result. '
             'Longer contexts and more covariates may require smaller batches in historical studies.', '',
             '| Model | Days | Steps | Covariates | Status | Inference seconds | Peak reserved GiB |',
             '|---|---:|---:|---:|---|---:|---:|']
    for item in results:
        seconds = item.get('inference_seconds')
        memory = item.get('peak_reserved_bytes')
        seconds_text = f'{seconds:.3f}' if seconds is not None else 'n/a'
        memory_text = f'{memory / 1024**3:.3f}' if memory is not None else 'n/a'
        lines.append(f"| {item['model']} | {item['days']} | {item['steps']} | {item['synthetic_covariates']} | {item['status']} | {seconds_text} | {memory_text} |")
    lines += ['', 'Chronos-2: configured cap 8,192 time steps. TimesFM 3: configured cap 15,360 time steps. '
              'At quarter-hour resolution, 85 calendar days fit Chronos-2 even with one autumn repeated hour. '
              '160 ordinary days exactly fill TimesFM; 160 days spanning an autumn transition can exceed its cap. '
              'No fixed t0-gamma context cap was found in its pinned checkpoint configuration; tested successes do not establish an unlimited context.', '']
    (destination / 'summary.md').write_text('\n'.join(lines))
    print(f'Wrote {destination / "summary.md"}', flush=True)


if __name__ == '__main__':
    main()
