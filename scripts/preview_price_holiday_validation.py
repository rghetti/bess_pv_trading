"""Compare saved price candidates after adding shared holiday-time indicators.

The default action verifies the inputs and prints a plan. With --run it refits
the frozen candidates into a separate directory; the original search and its
baseline preview remain untouched. No test-period labels are loaded.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Cap native threads before numerical and model packages are imported.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.price_feature_data import digest
from bess_pv_trading.forecasting.studies.price_feature_models import point_metrics
from bess_pv_trading.forecasting.studies.price_feature_search import _fit, _score_key
from scripts.preview_price_validation import plot_winner


def validate_candidate(job, source, group, holiday_file, output):
    """Fit one unchanged candidate with extra calendar columns on the original split."""
    source = Path(source)
    files = group["files"]
    arrays = {name: np.load(source / item["path"], mmap_mode="r", allow_pickle=False)
              for name, item in files.items() if name in {"x", "y", "anchor", "reference", "starts"}}
    indices = group["cohorts"][job["arm"]["cohort"]]["indices"]
    train = np.load(source / indices["train"]["path"], allow_pickle=False)
    validation = np.load(source / indices["validation"]["path"], allow_pickle=False)
    holiday = np.load(holiday_file, mmap_mode="r", allow_pickle=False)
    if len(holiday) != len(arrays["x"]):
        raise ValueError(f"Holiday row count differs for {job['group']}")
    columns = job["arm"]["columns"]
    # Append the same calendar features for every family using identical row indices.
    x_train = np.column_stack((arrays["x"][np.ix_(train, columns)], holiday[train]))
    x_validation = np.column_stack((arrays["x"][np.ix_(validation, columns)], holiday[validation]))
    _, prediction, warnings = _fit(job["candidate"]["family"], job["candidate"], x_train,
        arrays["y"][train] - arrays["anchor"][train], x_validation)
    prediction = prediction + arrays["anchor"][validation]
    frame = pd.DataFrame({"product_start": pd.to_datetime(arrays["starts"][validation], utc=True),
        "actual": arrays["y"][validation], "prediction": prediction,
        "reference": arrays["reference"][validation]})
    path = Path(output) / "predictions" / f"{job['group']}__{job['candidate']['family']}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    return {**job, "metrics": point_metrics(frame.actual, frame.prediction, frame.reference),
        "training_rows": len(train), "prediction_path": str(path), "warnings": warnings}


def main():
    """Verify the frozen baseline and holiday inputs before any optional refits."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", default="outputs/price_feature_search_full")
    parser.add_argument("--baseline-preview", default="outputs/price_validation_preview_nov10_v2")
    parser.add_argument("--holiday-features", default="outputs/price_holiday_feature_expansion_general_v2")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    source = Path(args.search).resolve()
    prepared_path = source / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    baseline = json.loads((Path(args.baseline_preview) / "plan.json").read_text())
    holiday_root = Path(args.holiday_features).resolve()
    holiday = json.loads((holiday_root / "manifest.json").read_text())
    if baseline["scientific_config_sha256"] != prepared["scientific_config_sha256"]:
        raise ValueError("Baseline uses a different scientific configuration")
    if holiday["source_prepared_sha256"] != digest(prepared_path):
        raise ValueError("Holiday features use different prepared inputs")
    jobs = baseline["jobs"]
    # Verify each sidecar's source timestamps and content before refitting.
    for group_name in sorted({job["group"] for job in jobs}):
        entry = holiday["groups"][group_name]
        if entry["source_starts_sha256"] != prepared["groups"][group_name]["files"]["starts"]["sha256"]:
            raise ValueError(f"Holiday timestamp identity differs: {group_name}")
        if digest(holiday_root / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Holiday sidecar checksum mismatch: {group_name}")
    print(json.dumps({"refits": len(jobs), "groups": sorted({j['group'] for j in jobs}),
        "holiday_features": len(next(iter(holiday["groups"].values()))["features"]),
        "workers": args.workers, "run": args.run, "selection": "baseline candidates held fixed"}, indent=2), flush=True)
    if not args.run:
        return
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps({"baseline": args.baseline_preview,
        "holiday_manifest_sha256": digest(holiday_root / "manifest.json"), "jobs": jobs}, indent=2) + "\n")
    remaining = Counter(job["group"] for job in jobs)
    results, failures, winners, completed_winners = [], [], {}, {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(validate_candidate, job, str(source), prepared["groups"][job["group"]],
                   str(holiday_root / holiday["groups"][job["group"]]["path"]), str(output)): job for job in jobs}
        for future in as_completed(futures):
            job = futures[future]
            group = job["group"]
            try:
                result = future.result()
                results.append(result)
                if group not in winners or _score_key(result) < _score_key(winners[group]):
                    winners[group] = result
            except Exception as error:
                failures.append({"group": group, "family": job["candidate"]["family"],
                                 "error": f"{type(error).__name__}: {error}"})
            remaining[group] -= 1
            (output / "validation_results.json").write_text(json.dumps(results, indent=2) + "\n")
            (output / "validation_failures.json").write_text(json.dumps(failures, indent=2) + "\n")
            if remaining[group] == 0 and group in winners:
                plot_winner(winners[group], output, prepared["scientific_settings"]["timezone"])
                completed_winners[group] = winners[group]
                (output / "validation_winners.json").write_text(json.dumps(completed_winners, indent=2) + "\n")
            print(f"Validated {len(results)}/{len(jobs)} candidates; failures={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
