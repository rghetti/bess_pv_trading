#!/bin/bash
# Run unfinished histories first, then covariate variants, and compare all six.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
DATA_PYTHON="${DATA_PYTHON:-/home/rghetti/miniforge3/envs/bess/bin/python}"
study_root="outputs/da_zero_shot_context_windows"
run_variant() {
  local days="$1" variant="$2"
  shift 2
  local bundle="$study_root/${days}d/$variant/inputs"
  local run="$study_root/${days}d/$variant/run"
  # Only reuse a completed report when all workers identify the same input manifest.
  if "$DATA_PYTHON" -c 'import hashlib,json,pathlib,sys; b=pathlib.Path(sys.argv[1]); r=pathlib.Path(sys.argv[2]); h=hashlib.sha256((b/"manifest.json").read_bytes()).hexdigest(); s=json.loads((r/"report/status.json").read_text()); assert s["complete"]; assert all(json.loads((r/m/"shard-0-of-1/identity.json").read_text())["bundle_sha256"]==h for m in ("chronos2","t0_gamma","timesfm3"))' "$bundle" "$run" 2>/dev/null; then
    echo "Reusing completed $days-day $variant report"
    return
  fi
  PROBE_BUNDLE="$bundle" PROBE_OUTPUT="$run" bash scripts/launch_da_zero_shot_probe.sh "$@"
}
for days in 7 60; do
  run_variant "$days" history "${@:1}"
done
for days in 7 28 60; do
  run_variant "$days" nord "${@:1}"
done
"$DATA_PYTHON" -m scripts.compare_da_context_windows
