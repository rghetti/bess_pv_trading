#!/bin/bash
# Run from any directory. Optional first argument --plan prints commands without inference.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
DATA_PYTHON="${DATA_PYTHON:-/home/rghetti/miniforge3/envs/bess/bin/python}"
MODEL_ENV_ROOT="${MODEL_ENV_ROOT:-outputs/zero_shot_envs}"
PROBE_BUNDLE="${PROBE_BUNDLE:-outputs/da_zero_shot_inputs}"
PROBE_OUTPUT="${PROBE_OUTPUT:-outputs/da_zero_shot_run}"
run_flag=(--run)
if [[ "${1:-}" == '--plan' ]]; then
  shift
  run_flag=()
fi
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
"$DATA_PYTHON" -m scripts.da_zero_shot_probe run \
  --bundle "$PROBE_BUNDLE" --output "$PROBE_OUTPUT" \
  --python-root "$MODEL_ENV_ROOT" --devices auto --threads 1 "${run_flag[@]}" "$@"
if [[ ${#run_flag[@]} -gt 0 && " $* " != *' --preflight-only '* ]]; then
  "$DATA_PYTHON" -m scripts.da_zero_shot_probe report \
    --bundle "$PROBE_BUNDLE" --forecast-directory "$PROBE_OUTPUT" \
    --output "$PROBE_OUTPUT/report"
fi
