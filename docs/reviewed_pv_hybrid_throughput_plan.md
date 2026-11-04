# Reviewed PV CPU and GPU throughput plan

The active `four_cycle_neural_night3000_v2` CPU search and the earlier
`four_cycle_neural_night3000_gpu_v1` search do not represent the same trial
grid. CPU batch sizes are 8, 16 and 32; the GPU search uses 2, 4 and 8. The
five trial IDs completed in both searches had different batch sizes. Their
scores and trial identities therefore cannot be combined as one search.
The active CPU search also has `resume: false`. Do not modify its inputs,
outputs or running processes while preparing an alternative.

## Proposed GPU measurement

`scripts/profile_reviewed_pv_gpu.py` generates synthetic arrays and compares
the current per-batch CPU-to-GPU copies with keeping the inputs and targets on
the GPU. It uses no historical GHI, AC targets, curated labels or held-out
data. It measures training-step throughput, validation forward-pass time,
and peak allocated and reserved GPU memory. Output includes final synthetic
loss and a validation checksum as consistency indicators, not accuracy scores.

Default cases are MLP, bidirectional LSTM and Transformer; width 256, depth 2,
192 quarter-hours, 24 features, synthetic MSE, AdamW, batch sizes 8 and 32,
five warmup steps and 20 measured training steps. Each case runs both transfer
patterns. Validation uses 482 synthetic sequences. The inputs use random
sampling with replacement, full sequence masks and unweighted MSE. In the
reviewed farm 1 training cache, 1,463 sequences contain 195,264 supervised
quarter-hours, a median of 142 per sequence. A batch is a group of sequences,
not individual quarter-hours. Timings are indicative and
are not a model experiment or a substitute for end-to-end HPO timing. GPU
allocation failures are recorded per case. Shared GPU load can affect results;
the initial 4 GiB free-memory check does not reserve memory.

After explicit approval, the benchmark can be run on GPU 0 with:

```bash
cd /home/rghetti/bess_pv_trading
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=src /home/rghetti/miniforge3/envs/bess/bin/python scripts/profile_reviewed_pv_gpu.py > /tmp/reviewed_pv_gpu_throughput.json
```

No code here starts the historical HPO. Benchmark results should be used to
choose whether device-resident arrays and batch size are worth incorporating,
and to estimate safe per-device concurrency before any mixed search begins.

## Requirements for a mixed search

- Generate one frozen trial grid and assign each trial exactly once to a CPU
  or GPU worker; avoid duplicated work and give each device a concurrency cap.
- Record the actual device, software version and configuration with every
  trial. Refit validation-selected configurations on their original device;
  CPU and GPU results can differ numerically even with the same seed.
- Use the existing reviewed labels, splits, 48-hour context, historical GHI
  for training and forecast GHI for validation/test, night clipping, losses,
  validation selection and top-three reports. GPU throughput measurements do
  not establish model quality.
- Use a new run identity and separate output/report roots. Import prior CPU
  records only after explicitly checking the full trial specification, data,
  feature transforms, code and device provenance. The earlier GPU records have
  a different batch-size grid and cannot be silently substituted.
- Make CPU count, GPU IDs and per-GPU worker counts configurable; check free
  host/GPU memory and leave shared-machine capacity for other work. Measure
  throughput on the actual machine before choosing concurrency.
- Keep selection based solely on daylight validation scores after the assumed
  six-hour forecast release. The first six hours remain training-only
  retrospective views; 2026 labels were previously reviewed and include
  synthetic targets. Final test scoring happens only after selection is frozen.
