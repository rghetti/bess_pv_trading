"""Launch the reproducible CPU feature search from a single market DuckDB catalog."""

from __future__ import annotations

import os

# Set caps before importing numerical libraries, including in multiprocessing children.
for variable in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
                 "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import argparse
import json
from pathlib import Path

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.price_feature_search import run_search


def main():
    """A plan is the default; explicit prepare-only and run flags control costly work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/price_feature_selection.yaml")
    parser.add_argument("--database")
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--input-manifest", help="Optional immutable source-selection manifest for a fresh preparation")
    parser.add_argument("--retry-failed", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = load_config(args.config)["price_feature_search"]
    for name, value in {"database_path": args.database, "output_directory": args.output, "workers": args.workers}.items():
        if value is not None:
            settings[name] = value
    if settings.get("requires_merged_sources") and (args.run or args.prepare_only):
        parser.error("Use scripts.search_price_full_year; the standard runner would ignore source fallbacks")
    if not args.run and not args.prepare_only:
        subsets = sum(2 ** len(blocks) for blocks in settings["cohorts"].values())
        print(json.dumps({"action": "plan_only", "groups": list(settings["product_groups"]),
            "families": list(settings["parameter_spaces"]), "trials_per_family": settings["trials_per_family"],
            "maximum_arms_before_coverage_checks": len(settings["product_groups"]) * subsets,
            "maximum_inner_fits_before_coverage_checks": len(settings["product_groups"]) * subsets * len(settings["parameter_spaces"]) * settings["trials_per_family"],
            "workers": settings["workers"], "threads_per_worker": 1, "farm_files_required": False}, indent=2))
        return
    input_settings = load_config(settings["input_config"])["price_search_inputs"]
    result = run_search(settings, input_settings, Path(settings["output_directory"]), prepare_only=args.prepare_only,
                        pinned_input_manifest=args.input_manifest, retry_failed=args.retry_failed)
    if args.prepare_only:
        print(json.dumps({"status": result["status"], "groups": len(result["groups"]), "arms": len(result["arms"]),
                          "inner_fits": result["inner_fits"], "output": settings["output_directory"],
                          "cohorts": {group: {name: {"eligible": cohort["eligible"], "counts": cohort["counts"]}
                                       for name, cohort in details["cohorts"].items()} for group, details in result["groups"].items()}}, indent=2))
    else:
        print(json.dumps(result, indent=2))
        if result["status"] != "complete":
            raise SystemExit(2)


if __name__ == "__main__":
    main()
