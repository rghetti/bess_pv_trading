# Four-cycle reviewed PV search on one GPU

## GPU 0-only recovery of the interrupted three-GPU search

The multi-GPU search stopped after an out-of-memory failure while another
process used most of GPU 2. The GPU 0 recovery entry point reads the original
run identity and requires the same dataset, trial grid, model/training code,
dependencies and GPU 0 UUID. It retains every complete matching trial record,
including trials originally trained on GPUs 1 and 2, and schedules only
remaining trials on GPU 0 with the original two-worker limit. It never changes
the original identity or prior trial records.

```bash
cd /home/rghetti/bess_pv_trading
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/resume_reviewed_pv_gpu0_hpo.py --config configs/pv_experiments/reviewed_four_cycle_neural_hpo_multi_gpu.yaml --check-only
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/resume_reviewed_pv_gpu0_hpo.py --config configs/pv_experiments/reviewed_four_cycle_neural_hpo_multi_gpu.yaml --run
```

The `--check-only` command checks data, saved trial identities and GPU 0; the
`--run` command resumes the historical search. The user starts it in tmux.
Selected trials originally trained on GPUs 1 or 2 are refit on GPU 0, with
the original validation replay tolerance. Differences across GPU architectures
can exceed that tolerance: in that case reporting stops rather than silently
altering the validation-selected ranking. Refit records retain the original
training GPU as well as the GPU 0 refit provenance.

## Separate three-GPU search

`configs/pv_experiments/reviewed_four_cycle_neural_hpo_multi_gpu.yaml` uses
two fit workers on each of physical GPUs 0, 1 and 2, for six workers total.
Select those physical GPUs in the configured order with `CUDA_VISIBLE_DEVICES`.
Each spawned worker sees only its assigned physical GPU. Every trial records
the device name and UUID, and each validation-selected refit runs on the same
physical GPU as its trial. `trial_gpu_provenance.csv` accompanies the results.

The initial guards require at least 4 GiB free per GPU and 16 GiB available
host memory. Other processes can use the shared GPUs after the checks. If a
trial encounters CUDA out of memory, it is retried alone on its assigned GPU
and that GPU's concurrency is reduced. If it still cannot fit alone, the run
stops with completed trial records available for resume. Resume requires the
same run identity, device UUIDs, code, data, configuration and dependencies.

```bash
cd /home/rghetti/bess_pv_trading
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_multi_gpu_hpo.py --config configs/pv_experiments/reviewed_four_cycle_neural_hpo_multi_gpu.yaml --check-only
CUDA_VISIBLE_DEVICES=0,1,2 PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_multi_gpu_hpo.py --config configs/pv_experiments/reviewed_four_cycle_neural_hpo_multi_gpu.yaml --run
```

This is a new 2,560-trial search and does not import the earlier GPU 0 trial
records. It preserves the 16/32/48 batch-size grid, curated data, training and
validation GHI rules, 48-hour horizons, splits and selection. Before starting,
check that any older search you no longer want to run has stopped. The shared
GPUs 1 and 2 had other processes when inspected, so capacity may change.

## Parallel search with batches 16, 32 and 48

`configs/pv_experiments/reviewed_four_cycle_neural_hpo_gpu_parallel.yaml`
defines a separate search with four GPU workers and the requested batch-size
choices. The user selects a physical GPU with `CUDA_VISIBLE_DEVICES`. It has
separate run and report roots from both earlier CPU and single-worker GPU
searches. The start guard requires 8 GiB free GPU memory and 16 GiB available
host memory. Other users can consume memory after the check, so this is not a
reservation or a guarantee.

If a trial runs out of GPU memory, the scheduler waits for other running jobs,
retries the failed trial alone, and reduces the subsequent concurrency limit.
If the same trial cannot run alone, the search stops and completed records are
available for resume under the same identity. Each worker releases its unused
CUDA allocations after each trial and refit. The concurrency limit is set by
`workers` and `runtime.maximum_gpu_workers` in the study YAML.

```bash
cd /home/rghetti/bess_pv_trading
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_gpu_hpo.py --config configs/pv_experiments/reviewed_four_cycle_neural_hpo_gpu_parallel.yaml --check-only
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_gpu_hpo.py --config configs/pv_experiments/reviewed_four_cycle_neural_hpo_gpu_parallel.yaml --run
```

The `--run` command starts 2,560 candidate fits. The user chooses when to run
it in tmux; preflight never trains. Selected models are refit on the same GPU,
with the same validation replay tolerance as the single-worker search. A new
batch-size grid produces different trial specifications and does not resume
the stopped CPU search or reuse the older GPU records.

On 2026-11-04, synthetic stress checks on GPU 0 ran 1, 2 and 4 independent
workers for each of the five model families. Each used width 512, depth 4,
batch 48, 192 steps, 24 features, one warmup and two measured optimizer
updates, with data resident on the GPU. All cases completed. Summed per-worker
peak reserved memory at four workers was 1.30 GiB for MLP, 4.36 GiB for TCN,
7.22 GiB for BiGRU, 7.24 GiB for BiLSTM and 5.38 GiB for Transformer. These
are sums of individual peaks, not direct measurements of simultaneous device
usage, and exclude some context overhead.

The measured aggregate throughput at four workers was 5–10% lower than at one
for these brief GPU-saturated cases. Four workers honor the requested
parallelism, but may not minimize wall time. The synthetic check excludes data
preparation, validation, longer training and other users' activity; it does not
establish the fastest setting for the full search. The detailed check output
is `/tmp/reviewed_pv_gpu_concurrency.json` on the current machine.

## Earlier single-worker GPU search

This search uses the data, models, losses, splits, selection and metrics described
in [the four-cycle CPU plan](reviewed_pv_four_cycle_hpo_plan.md). It has a
separate configuration, runner, model implementation, run identity and output
directories. It does not alter or resume the existing CPU search.

The GPU runner fits one trial at a time on the selected visible CUDA device,
with one CPU thread per worker. The GPU must have at least 4 GiB free and the
host at least 8 GiB available when the run starts. These configurable guards
are not a guarantee that every model will fit or that other users will leave
memory free. A CUDA out-of-memory error stops the run; completed trials remain
resumable. Check GPU usage before starting. The runner checks a CUDA kernel,
records device architecture and CUDA build in the run identity, and uses
deterministic PyTorch algorithms with a configured cuBLAS workspace. Resume
requires the same code, device contract, data and software versions.

The example selects physical GPU 0. `CUDA_VISIBLE_DEVICES` makes it device 0
inside the process. Change the number to select a different card, and use a
PyTorch build that supports that card. The GPU environment must contain the
project dependencies as well as a compatible CUDA PyTorch build. On this
machine, synthetic fits for all five model families succeeded on both GPU 0
and GPU 2 with the existing `bess` environment's PyTorch CUDA 12.9 build.
Check a different device with preflight before using it.

The following preflight checks data, GPU compatibility and trial configuration;
it does not train. It reports currently available GPU memory without reserving it.

```bash
cd /home/rghetti/bess_pv_trading
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_gpu_hpo.py --config configs/pv_experiments/reviewed_four_cycle_neural_hpo_gpu.yaml --check-only
```

After preflight, the user can start the search in tmux:

```bash
cd /home/rghetti/bess_pv_trading
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/train_reviewed_pv_gpu_hpo.py --config configs/pv_experiments/reviewed_four_cycle_neural_hpo_gpu.yaml --run
```

The 2,560 candidate fits are followed by 40 family/feature winner refits and
up to eight additional refits for the top three per farm. Results go to
`outputs/pv_reviewed_hpo/four_cycle_neural_night3000_gpu_v1/` and
`docs/results/pv_reviewed_hpo/four_cycle_neural_night3000_gpu_v1/`.
Selected bundles use CPU-portable saved weights and retain the existing
nighttime clipping policy in the inference adapter.
