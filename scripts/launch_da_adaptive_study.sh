#!/bin/bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
for days in 7 28 60; do
  for variant in history nord; do
    PROBE_BUNDLE="outputs/da_zero_shot_adaptive/${days}d/$variant/inputs" \
    PROBE_OUTPUT="outputs/da_zero_shot_adaptive/${days}d/$variant/run" \
    bash scripts/launch_da_zero_shot_probe.sh "$@"
  done
done

"${DATA_PYTHON:-/home/rghetti/miniforge3/envs/bess/bin/python}" -m scripts.compare_da_context_windows --study-root outputs/da_zero_shot_adaptive
