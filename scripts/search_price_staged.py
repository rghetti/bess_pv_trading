"""Prepare and run a checkpointed, staged search of auction price models."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Set limits before importing numerical libraries in the coordinator and workers.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.price_staged_search import prepare_staged, run_staged


def main():
    """Require an explicit preparation or search action before accessing data."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/price_staged_search.yaml")
    parser.add_argument("--database", required=True)
    parser.add_argument("--output")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--retry-failed", action="store_true")
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--prepare-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args()
    options = load_config(args.config)["price_staged_search"]
    base = load_config(options["base_config"])["price_feature_search"]
    base["database_path"] = args.database
    output = Path(args.output or options["output_directory"]).resolve()
    workers = args.workers if args.workers is not None else options["workers"]
    if workers < 1:
        raise ValueError("workers must be positive")
    if not (args.prepare_only or args.run):
        print(json.dumps({"action": "plan_only", "output": str(output), "workers": workers,
            "families": list(base["parameter_spaces"]), "products": list(base["product_groups"]),
            "blocks": options["initial_blocks"], "configurations_per_family": options["configurations_per_family"],
            "retrospective_features": options["retrospective_features"]}, indent=2))
        return
    if args.prepare_only:
        prepared = prepare_staged(base, options, output)
        print(json.dumps({"status": "prepared", "output": str(output),
                          "groups": list(prepared["groups"])}, indent=2))
    else:
        # The fitting command never accesses the database or changes a frozen snapshot.
        if not (output / "prepared.json").is_file():
            raise ValueError("Run --prepare-only after the downloader completes before --run")
        result = run_staged(base, options, output, workers, retry_failed=args.retry_failed)
        print(json.dumps({"status": result["status"], "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
