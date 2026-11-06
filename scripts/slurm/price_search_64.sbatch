#!/usr/bin/env bash
#SBATCH --job-name=bess-price-search
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --output=price-search-%j.out
#SBATCH --error=price-search-%j.err

set -euo pipefail

# Submit from the repository root, or point PRICE_SEARCH_ROOT at that checkout.
study_root="${PRICE_SEARCH_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}"
cd "$study_root"
study_python="${PRICE_SEARCH_PYTHON:-python}"
study_database="${PRICE_DB:-$study_root/data/api/market.duckdb}"
study_output="${PRICE_SEARCH_OUTPUT:-$study_root/outputs/price_feature_search_full}"
study_config="${PRICE_SEARCH_CONFIG:-$study_root/configs/research/price_feature_selection.yaml}"
study_workers="${SLURM_CPUS_PER_TASK:-64}"

# Each process fits one CPU model at a time; BLAS/OpenMP cannot multiply the allocation.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export BLIS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 OMP_DYNAMIC=FALSE
export CUDA_VISIBLE_DEVICES=""
export PYTHONUNBUFFERED=1

test -f "$study_database"
test -f "$study_config"
mkdir -p "$study_output"
export MPLCONFIGDIR="${PRICE_SEARCH_MPL_CACHE:-$study_output/matplotlib_cache}"
mkdir -p "$MPLCONFIGDIR"
arguments=(--config "$study_config" --database "$study_database" --output "$study_output" --workers "$study_workers" --run)
if [[ -n "${PRICE_SEARCH_INPUT_MANIFEST:-}" ]]; then
    arguments+=(--input-manifest "$PRICE_SEARCH_INPUT_MANIFEST")
fi
if [[ "${PRICE_SEARCH_RETRY_FAILED:-0}" == "1" ]]; then
    arguments+=(--retry-failed)
fi

# Existing compatible checkpoints are resumed automatically after a time-limit interruption.
srun --ntasks=1 --cpus-per-task="$study_workers" --cpu-bind=cores \
    "$study_python" -m scripts.search_price_models "${arguments[@]}"
