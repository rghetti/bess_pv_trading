"""Independent model-environment worker for published DA price history only.

This file deliberately imports no project data readers: request files contain
historical inputs only, while evaluation labels stay in the orchestration layer.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import signal
import tempfile
import time


STOP = False


def stop_after_day(signum, frame):
    """Finish the current atomic daily forecast when interrupted."""
    global STOP
    STOP = True


def digest(path):
    """Hash a request, source or weight file in bounded chunks."""
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def object_digest(value):
    """Give JSON metadata a stable, path-independent fingerprint."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_json(path, value):
    """Install only fully written checkpoints; interrupted temporary files are ignored."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w', dir=destination.parent, delete=False) as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
        temporary = stream.name
    os.replace(temporary, destination)


def verify_prediction(saved, identity, request_sha):
    """Reject stale or damaged daily forecasts before marking them resumed."""
    content = {key: value for key, value in saved.items() if key != 'payload_sha256'}
    if saved.get('payload_sha256') != object_digest(content):
        raise ValueError('Forecast payload checksum mismatch')
    if saved.get('run_identity') != identity or saved.get('request_sha256') != request_sha:
        raise ValueError('Checkpoint belongs to different inputs, software, device or model settings')


def request_batches(requests, batch_size):
    """Batch adjacent days only when context length and target product count match."""
    pending = []
    signature = None
    for item in requests:
        request = item[2]
        key = (len(request['context']), request['horizon'])
        if pending and (key != signature or len(pending) == batch_size):
            yield pending
            pending = []
        signature = key
        pending.append(item)
    if pending:
        yield pending


def load_model(name, checkpoint, device, torch, batch_size=1):
    """Build a frozen FP32 forecaster; each batch row is an independent day."""
    import numpy as np

    if name == 'chronos2':
        from chronos import Chronos2Pipeline

        model = Chronos2Pipeline.from_pretrained(checkpoint, device_map=device, torch_dtype=torch.float32)
        def predict(contexts, horizon, levels, covariates=None):
            contexts = np.atleast_2d(contexts)
            if contexts.shape[1] > model.model.chronos_config.context_length:
                raise ValueError('Chronos context would be truncated')
            inputs = []
            for index, row in enumerate(contexts):
                item = {'target': row}
                if covariates is not None:
                    item['past_covariates'] = {key: value[index, :len(row)] for key, value in covariates.items()}
                    item['future_covariates'] = {key: value[index, len(row):] for key, value in covariates.items()}
                inputs.append(item)
            values, _ = model.predict_quantiles(inputs, prediction_length=horizon,
                quantile_levels=levels, context_length=contexts.shape[1], batch_size=batch_size, cross_learning=False)
            return np.stack([item.detach().cpu().numpy().reshape(horizon, len(levels)) for item in values])
    elif name == 't0_gamma':
        from t0 import T0Forecaster

        model = T0Forecaster.from_pretrained(checkpoint).to(device=device, dtype=torch.float32).eval()
        def predict(contexts, horizon, levels, covariates=None):
            contexts = np.atleast_2d(contexts)
            future = None if covariates is None else torch.as_tensor(np.stack(list(covariates.values()), axis=1), device=device)
            forecast = model.predict(torch.as_tensor(contexts, device=device), horizon=horizon,
                future_covariates=future, context_length=contexts.shape[1], quantile_levels=levels, symmetric_averaging=False,
                clip_nonnegative=False, snap_integers=False)
            return forecast.quantiles.detach().cpu().numpy().reshape(len(contexts), horizon, len(levels))
    elif name == 'timesfm3':
        from timesfm3 import ModelConfig, TimesFM3Evaluator

        model = TimesFM3Evaluator(ModelConfig(checkpoint_path=checkpoint, per_core_batch_size=batch_size, device=device))
        def predict(contexts, horizon, levels, covariates=None):
            contexts = np.atleast_2d(contexts)
            if contexts.shape[1] > model.global_context:
                raise ValueError('TimesFM context would be truncated')
            if not np.allclose(levels, np.arange(1, 10) / 10):
                raise ValueError('This TimesFM adapter uses the nine native deciles')
            future = None if covariates is None else [np.stack([value[index] for value in covariates.values()]) for index in range(len(contexts))]
            forecasts = list(model.predict_batch(list(contexts), horizon=horizon, return_quantiles=True,
                past_future_covariates=future,
                use_symmetric_averaging=False, make_positive=False, sort_quantiles=False))
            return np.stack([np.asarray(item.quantiles).reshape(horizon, len(levels)) for item in forecasts])
    else:
        raise ValueError(f'Unapproved model: {name}')
    return predict


def main():
    """Probe compatibility, then forecast each eligible day with durable restarts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--shards', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--allow-download', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        parser.error('Invalid day shard')
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.threads < 1:
        parser.error('--threads must be positive')
    for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[variable] = str(args.threads)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    os.environ['OMP_DYNAMIC'] = 'FALSE'
    if not args.allow_download:
        os.environ['HF_HUB_OFFLINE'] = '1'
    bundle, destination = Path(args.bundle), Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    lock = (destination / '.worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    signal.signal(signal.SIGINT, stop_after_day)
    signal.signal(signal.SIGTERM, stop_after_day)
    started = time.monotonic()
    state = {'model': args.model, 'status': 'starting'}
    try:
        manifest = json.loads((bundle / 'manifest.json').read_text())
        spec = manifest['models'][args.model]
        settings = manifest['settings']
        package, expected = spec['package'].split('==')
        package = package.split('[')[0]
        if importlib.metadata.version(package) != expected:
            raise ValueError(f'Expected {package}=={expected}')
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download

        seed = settings['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.set_num_threads(args.threads)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        device = torch.device(args.device)
        hardware = {'device': str(device), 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')}
        if device.type == 'cuda':
            torch.cuda.set_device(device)
            # A real kernel catches unsupported SM architectures before weight loading.
            (torch.ones(1, device=device) + 1).item()
            hardware.update(name=torch.cuda.get_device_name(device), capability=list(torch.cuda.get_device_capability(device)))
        checkpoint = Path(snapshot_download(spec['repository'], revision=spec['revision'], local_files_only=not args.allow_download))
        weight_hashes = {str(path.relative_to(checkpoint)): digest(path) for path in sorted(checkpoint.rglob('*'))
                         if path.is_file() and path.suffix in ('.safetensors', '.json')}
        if not any(name.endswith('.safetensors') for name in weight_hashes):
            raise ValueError('Pinned checkpoint has no safetensors weights')
        software = sorted((item.metadata['Name'], item.version) for item in importlib.metadata.distributions())
        identity_data = {'bundle_sha256': digest(bundle / 'manifest.json'), 'model': args.model, 'spec': spec,
            'worker_sha256': digest(__file__), 'python': platform.python_version(), 'packages': software,
            'hardware': hardware, 'precision': 'float32', 'threads': args.threads, 'seed': seed,
            'shard_index': args.shard_index, 'shards': args.shards, 'batch_size': args.batch_size,
            'weight_sha256': weight_hashes, 'postprocessing': 'raw quantiles; no clipping, sorting or averaging'}
        identity = object_digest(identity_data)
        previous = destination / 'identity.json'
        if previous.exists() and object_digest(json.loads(previous.read_text())) != identity:
            raise ValueError('Run identity changed; use a new output directory')
        write_json(previous, identity_data)
        requests = sorted((name, checksum) for name, checksum in manifest['files'].items() if name.startswith('requests/'))
        requests = requests[args.shard_index::args.shards]
        if not requests:
            raise ValueError('No eligible forecast requests')
        parsed = []
        for name, checksum in requests:
            if digest(bundle / name) != checksum:
                raise ValueError(f'Request checksum mismatch: {name}')
            request = json.loads((bundle / name).read_text())
            if request['feature_names'] != manifest['features'] or not np.isfinite(request['context']).all():
                raise ValueError('Unexpected feature contract or missing context')
            forecasts = request.get('forecast_covariates', {})
            if list(forecasts) != sorted(manifest['features'][1:]) or any(len(value) != len(request['context']) + request['horizon'] or not np.isfinite(value).all() for value in forecasts.values()):
                raise ValueError('Invalid forecast covariate values or dimensions')
            parsed.append((name, checksum, request))
        predict = load_model(args.model, str(checkpoint), str(device), torch, args.batch_size)
        # Check every requested horizon on the longest synthetic context before any replay.
        maximum = max(len(item[2]['context']) for item in parsed)
        synthetic = np.sin(np.arange(maximum, dtype=np.float32) / 30) * 20 - 5
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        probe_start = time.monotonic()
        with torch.inference_mode():
            for horizon in sorted({item[2]['horizon'] for item in parsed}):
                synthetic_cov = {key: np.tile(np.sin(np.arange(maximum + horizon, dtype=np.float32) / 50) + index, (args.batch_size, 1)) for index, key in enumerate(sorted(manifest['features'][1:]))}
                values = np.asarray(predict(np.tile(synthetic, (args.batch_size, 1)), horizon, settings['quantiles'], synthetic_cov or None))
                if values.shape != (args.batch_size, horizon, 9) or not np.isfinite(values).all():
                    raise ValueError('Synthetic preflight failed shape/finiteness check')
        preflight = {'status': 'passed', 'context_steps': maximum, 'horizons': sorted({item[2]['horizon'] for item in parsed}),
                     'elapsed_seconds': time.monotonic() - probe_start, 'run_identity': identity, 'batch_size': args.batch_size}
        if device.type == 'cuda':
            torch.cuda.synchronize(device)
            preflight.update(peak_reserved_bytes=torch.cuda.max_memory_reserved(device), peak_allocated_bytes=torch.cuda.max_memory_allocated(device))
        write_json(destination / 'preflight.json', preflight)
        if args.preflight_only:
            state.update(status='preflight_complete')
            return
        completed = 0
        for batch in request_batches(parsed, args.batch_size):
            if STOP:
                break
            existing = []
            for name, checksum, request in batch:
                path = destination / 'predictions' / Path(name).name
                if path.exists():
                    verify_prediction(json.loads(path.read_text()), identity, checksum)
                existing.append(path.exists())
            if all(existing):
                completed += len(batch)
                continue
            # Keep original batch membership when partially checkpointed: restart
            # repeats that forward pass and only writes missing daily files.
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            begin = time.monotonic()
            contexts = np.asarray([item[2]['context'] for item in batch], dtype=np.float32)
            horizon = batch[0][2]['horizon']
            with torch.inference_mode():
                covariates = {key: np.asarray([item[2]['forecast_covariates'][key] for item in batch], dtype=np.float32) for key in sorted(manifest['features'][1:])}
                values = np.asarray(predict(contexts, horizon, settings['quantiles'], covariates or None))
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            elapsed = time.monotonic() - begin
            if values.shape != (len(batch), horizon, 9) or not np.isfinite(values).all():
                raise ValueError('Invalid model output')
            for position, (name, checksum, request) in enumerate(batch):
                if existing[position]:
                    continue
                path = destination / 'predictions' / Path(name).name
                quantiles = values[position]
                saved = {'delivery_day': request['delivery_day'], 'product_starts_utc': request['product_starts_utc'],
                    'quantile_levels': settings['quantiles'], 'quantiles': quantiles.tolist(),
                    'elapsed_seconds': elapsed / len(batch), 'batch_elapsed_seconds': elapsed,
                    'batch_size': len(batch), 'runtime_convention': 'batch runtime divided by days in batch',
                    'run_identity': identity, 'request_sha256': checksum,
                    'bundle_sha256': identity_data['bundle_sha256'],
                    'quantile_crossing_products': int((np.diff(quantiles, axis=1) < 0).any(axis=1).sum())}
                saved['payload_sha256'] = object_digest(saved)
                write_json(path, saved)
            completed += len(batch)
            state.update(status='running', completed_days=completed, expected_days=len(parsed))
            write_json(destination / 'status.json', state)
            print(f"{args.model}: through {batch[-1][2]['delivery_day']} {completed}/{len(parsed)}", flush=True)
        state.update(status='interrupted' if STOP else 'complete', completed_days=completed, expected_days=len(parsed))
    except Exception as error:
        state.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        state['elapsed_seconds'] = time.monotonic() - started
        write_json(destination / 'status.json', state)
    if STOP:
        raise SystemExit(130)


if __name__ == '__main__':
    main()
