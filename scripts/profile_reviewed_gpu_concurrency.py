"""Stress synthetic reviewed-PV fits at increasing GPU concurrency.

This explicit profiling script never reads curated data or runs the HPO search.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import queue
import sys
import time
import traceback


def _fit_synthetic(job, barrier, results):
    """Fit a worst-width synthetic model and return timing and memory metrics."""
    try:
        import torch
        from bess_pv_trading.forecasting.models.pv_reviewed_gpu_hpo_models import build_neural_network

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(job['seed'])
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        torch.cuda.reset_peak_memory_stats(device)
        parameters = {'width': 512, 'depth': 4, 'dropout': 0.1, 'kernel_size': 9, 'heads': 8}
        model = build_neural_network(job['family'], 24, parameters).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        inputs = torch.randn((48, 192, 24), device=device)
        targets = torch.rand((48, 192), device=device)
        valid = torch.ones((48, 192), dtype=torch.bool, device=device)

        def step():
            """Perform the same kind of bounded-output update as the search."""
            prediction = model(inputs, valid)
            objective = (prediction - targets).square().mean()
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        # Warm up before timing and synchronize workers at the measured section.
        step()
        torch.cuda.synchronize()
        barrier.wait(timeout=90)
        start = time.perf_counter()
        for _ in range(job['steps']):
            step()
        torch.cuda.synchronize()
        results.put({'status': 'complete', 'seconds': time.perf_counter() - start,
                     'peak_allocated_gib': torch.cuda.max_memory_allocated(device) / 1024 ** 3,
                     'peak_reserved_gib': torch.cuda.max_memory_reserved(device) / 1024 ** 3})
    except Exception as error:
        # Release any workers waiting at the timed-section barrier after a failure.
        barrier.abort()
        results.put({'status': 'out_of_memory' if 'out of memory' in str(error).lower() else 'error',
                     'error': str(error), 'exception': type(error).__name__, 'traceback': traceback.format_exc()})


def main():
    """Run independent processes at each requested concurrency and report JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--families', nargs='+', choices=['mlp', 'tcn', 'bigru', 'bilstm', 'transformer'],
                        default=['mlp', 'tcn', 'bigru', 'bilstm', 'transformer'])
    parser.add_argument('--workers', nargs='+', type=int, default=[1, 2, 4])
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--timeout-seconds', type=int, default=120)
    parser.add_argument('--minimum-free-gib', type=float, default=8.0)
    args = parser.parse_args()
    if min(args.workers + [args.steps, args.timeout_seconds]) < 1 or args.minimum_free_gib < 0:
        parser.error('Worker counts, steps and timeout must be positive, and free memory must be nonnegative')
    workspace = os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if workspace not in {':4096:8', ':16:8'}:
        raise ValueError('CUBLAS_WORKSPACE_CONFIG must be :4096:8 or :16:8')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; select the GPU with CUDA_VISIBLE_DEVICES')
    context = multiprocessing.get_context('spawn')
    records = []
    # Each family is tested from low to high concurrency; do not escalate after failure.
    for family in args.families:
        for count in sorted(set(args.workers)):
            free, _ = torch.cuda.mem_get_info(0)
            if free / 1024 ** 3 < args.minimum_free_gib:
                records.append({'family': family, 'workers': count, 'status': 'insufficient_free_memory',
                                'free_gib': free / 1024 ** 3})
                break
            print(f'Profiling {family} with {count} GPU worker(s)', file=sys.stderr, flush=True)
            barrier = context.Barrier(count)
            results = context.Queue()
            processes = [context.Process(target=_fit_synthetic,
                         args=({'family': family, 'steps': args.steps, 'seed': 400 + index}, barrier, results))
                         for index in range(count)]
            for process in processes:
                process.start()
            deadline = time.monotonic() + args.timeout_seconds
            for process in processes:
                process.join(max(0, deadline - time.monotonic()))
            # Terminate only profiling children if they exceed the configured deadline.
            timed_out = any(process.is_alive() for process in processes)
            if timed_out:
                barrier.abort()
                for process in processes:
                    if process.is_alive():
                        process.terminate()
                for process in processes:
                    process.join()
            outcomes = []
            for _ in processes:
                try:
                    outcomes.append(results.get(timeout=2))
                except queue.Empty:
                    break
            results.close()
            success = not timed_out and len(outcomes) == count and all(item['status'] == 'complete' for item in outcomes)
            record = {'family': family, 'workers': count, 'status': 'complete' if success else 'failed',
                      'timed_out': timed_out, 'outcomes': outcomes}
            if success:
                record['aggregate_sequences_per_second'] = count * 48 * args.steps / max(item['seconds'] for item in outcomes)
            records.append(record)
            if not success:
                break
    print(json.dumps({'device': torch.cuda.get_device_name(0), 'cuda': torch.version.cuda,
                      'torch': torch.__version__, 'visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                      'settings': vars(args), 'results': records}, indent=2))


if __name__ == '__main__':
    main()
