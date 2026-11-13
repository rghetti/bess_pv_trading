# DA zero-shot context and regional-feature studies

The DA runner queues all configured history-only and NORD-forecast studies across up to four GPUs. Set `windows` in `configs/research/zero_shot_da_studies.yaml` to any nonempty list of whole-day contexts of at least seven days. Missing history and NORD input bundles are built automatically with that context length. The runner also keeps `cross_border_window` prepared as the NORD parent for regional features, even when it is not in the scored window list. Missing regional bundles are built concurrently in separate CPU processes; each completed bundle enters the GPU queue immediately. No experiment is started by plan or prepare-only mode.

The regional blocks use previous-day assumed-available quantities: PSV gas prices lagged one and seven days; French nuclear and hydro generation, balance and commercial Italy flows lagged 24 hours; Swiss generation, consumption, physical Italy/France flows lagged 24 hours, and reservoir levels lagged one week. The France/Swiss variants contain no Italian-Swiss price spread or other target-price field. These are retrospective inputs: archived values and release times are not fully verified, and a previous-day same-clock quantity can postdate the DA gate. Results must be described with this availability limitation.

Pinned-input eligibility in the current snapshot is 398 days for 60-day NORD, 397 for gas, 258 for France, 356 for Switzerland, 234 for France plus Switzerland, and 234 for the combined gas/France/Switzerland block. These are input coverage counts, not forecast scores. Every report includes the full available sample, while each paired comparison scores only products present in both that block and the 60-day NORD baseline, verifies realized targets agree, and records paired coverage. November remains a separate report period and is not used for selection.

Run in tmux after the IDA queue releases the GPUs. The launcher prepares missing bundles, starts GPU inference as each is ready, writes each study's Markdown/CSV report, and creates matched comparisons:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_da_zero_shot_studies --run --prepare-workers 10 --batch-size 10 --threads 1
```

To prepare the regional bundles without launching inference:

```bash
/home/rghetti/miniforge3/envs/bess/bin/python -m scripts.run_da_zero_shot_studies --prepare-only --prepare-workers 10
```

TimesFM 3 weights are non-commercial. The forecast quantiles are marginal. No neural model is fine-tuned in this workflow.
