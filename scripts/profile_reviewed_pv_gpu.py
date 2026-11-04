"""Measure synthetic reviewed-PV training batches on one visible CUDA GPU.

This explicit benchmark does not read historical labels or run the HPO search.
It compares the existing per-batch transfer pattern with device-resident inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from bess_pv_trading.forecasting.models.pv_reviewed_gpu_hpo_models import build_neural_network


def measure_case(torch, inputs, targets, family, batch_size, resident, warmup, steps, seed):
    """Time training steps and validation batches with identical synthetic inputs."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device('cuda:0')
    parameters = {'width': 256, 'depth': 2, 'dropout': 0.1, 'kernel_size': 3, 'heads': 8}
    model = build_neural_network(family, inputs.shape[-1], parameters).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    valid = np.ones(inputs.shape[:2], dtype=bool)
    # The resident case transfers the complete synthetic data once per candidate.
    if resident:
        input_store = torch.as_tensor(inputs, device=device)
        target_store = torch.as_tensor(targets, device=device)
        valid_store = torch.as_tensor(valid, device=device)
    else:
        input_store, target_store, valid_store = inputs, targets, valid

    def batch(indices):
        """Select one batch and put it on the same device as the model."""
        if resident:
            selected = torch.as_tensor(indices, device=device)
            return input_store[selected], target_store[selected], valid_store[selected]
        return (torch.as_tensor(input_store[indices], device=device),
                torch.as_tensor(target_store[indices], device=device),
                torch.as_tensor(valid_store[indices], device=device))

    # Synchronize around the measured section so queued work is counted.
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(device)
    started = None
    model.train()
    for number in range(warmup + steps):
        if number == warmup:
            torch.cuda.synchronize()
            started = time.perf_counter()
        indices = rng.integers(0, len(inputs), size=batch_size)
        x_batch, y_batch, valid_batch = batch(indices)
        prediction = model(x_batch, valid_batch)
        objective = (prediction - y_batch).square().mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    torch.cuda.synchronize()
    training_seconds = time.perf_counter() - started
    # Match the current per-batch validation copies or combine them into one copy.
    model.eval()
    outputs = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            indices = np.arange(start, min(start + batch_size, len(inputs)))
            x_batch, _, valid_batch = batch(indices)
            output = model(x_batch, valid_batch)
            outputs.append(output if resident else output.cpu())
        if resident:
            torch.cat(outputs).cpu()
    torch.cuda.synchronize()
    validation_seconds = time.perf_counter() - started
    validation_checksum = float(torch.cat(outputs).sum().item())
    result = {'family': family, 'batch_size': batch_size, 'data_placement': 'gpu' if resident else 'cpu',
              'measured_steps': steps, 'training_seconds': training_seconds,
              'training_sequences_per_second': steps * batch_size / training_seconds,
              'validation_sequences': len(inputs), 'validation_seconds': validation_seconds,
              'final_training_loss': float(objective.item()), 'validation_checksum': validation_checksum,
              'peak_gpu_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024 ** 3,
              'peak_gpu_reserved_gib': torch.cuda.max_memory_reserved(device) / 1024 ** 3}
    return result


def main():
    """Run only when explicitly invoked, then print timings as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', choices=['mlp', 'tcn', 'bigru', 'bilstm', 'transformer'],
                        default=['mlp', 'bilstm', 'transformer'])
    parser.add_argument('--batch-sizes', nargs='+', type=int, default=[8, 32])
    parser.add_argument('--sequences', type=int, default=482)
    parser.add_argument('--sequence-length', type=int, default=192)
    parser.add_argument('--features', type=int, default=24)
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--minimum-free-gib', type=float, default=4.0)
    parser.add_argument('--seed', type=int, default=418721)
    args = parser.parse_args()
    if min(args.batch_sizes + [args.sequences, args.sequence_length, args.features, args.steps]) < 1 or args.warmup < 0:
        parser.error('Batch sizes, sequence counts, lengths, features, and steps must be positive')
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in {':4096:8', ':16:8'}:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8 for deterministic CUDA')
    import torch
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select an accessible GPU using CUDA_VISIBLE_DEVICES')
    free, total = torch.cuda.mem_get_info(0)
    if free / 1024 ** 3 < args.minimum_free_gib:
        raise RuntimeError(f'GPU has {free / 1024 ** 3:.2f} GiB free; require {args.minimum_free_gib:g} GiB')
    rng = np.random.default_rng(args.seed)
    inputs = rng.normal(size=(args.sequences, args.sequence_length, args.features)).astype(np.float32)
    targets = rng.random(size=inputs.shape[:2], dtype=np.float32)
    results = []
    # Run the same model and batch cases with both data-transfer patterns.
    for family in args.families:
        for batch_size in args.batch_sizes:
            for resident in (False, True):
                try:
                    result = measure_case(torch, inputs, targets, family, batch_size, resident,
                                          args.warmup, args.steps, args.seed)
                    result['status'] = 'complete'
                except torch.cuda.OutOfMemoryError:
                    result = {'family': family, 'batch_size': batch_size,
                              'data_placement': 'gpu' if resident else 'cpu', 'status': 'out_of_memory'}
                # Release unreferenced allocations after the case has unwound.
                torch.cuda.empty_cache()
                results.append(result)
    print(json.dumps({'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                      'torch': torch.__version__, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'free_gpu_gib_at_start': free / 1024 ** 3, 'total_gpu_gib': total / 1024 ** 3,
                      'settings': vars(args), 'results': results}, indent=2))


if __name__ == '__main__':
    main()
