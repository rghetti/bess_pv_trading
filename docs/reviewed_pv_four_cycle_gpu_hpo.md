# Four-cycle reviewed PV search on one GPU

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
