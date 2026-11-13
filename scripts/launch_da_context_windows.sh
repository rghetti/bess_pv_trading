#!/bin/bash
# Run the remaining histories, then the covariate variants, and compare all six.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
DATA_PYTHON="${DATA_PYTHON:-/home/rghetti/miniforge3/envs/bess/bin/python}"
for days in 7 60; do
  PROBE_BUNDLE="outputs/da_zero_shot_inputs_${days}d" PROBE_OUTPUT="outputs/da_zero_shot_run_${days}d" \
    bash scripts/launch_da_zero_shot_probe.sh "$@"
done
for days in 7 28 60; do
  suffix="_${days}d"
  if [[ "$days" == 28 ]]; then suffix=''; fi
  PROBE_BUNDLE="outputs/da_zero_shot_nord_inputs${suffix}" PROBE_OUTPUT="outputs/da_zero_shot_nord_run${suffix}" \
    bash scripts/launch_da_zero_shot_probe.sh "$@"
done
"$DATA_PYTHON" -m scripts.compare_da_context_windows
