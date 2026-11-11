"""Prepare a user-run comparison of a saved DA winner with and without holiday interactions."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# A diagnostic fit is independent of the running search and uses one CPU thread.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.price_feature_data import digest
from bess_pv_trading.forecasting.studies.price_feature_models import point_metrics
from bess_pv_trading.forecasting.studies.price_feature_search import _fit, _score_key


def best_saved_candidate(root, prepared, group_name, cohort_name):
    """Select the best completed inner score on one matched cohort, without writing the search."""
    best = None
    seen = 0
    for task in prepared["tasks"]:
        if task["arm"]["group"] != group_name or task["arm"]["cohort"] != cohort_name:
            continue
        path = root / "trials" / f"{task['id']}.json"
        if not path.exists():
            continue
        checkpoint = json.loads(path.read_text())
        if checkpoint["signature"] != task["signature"]:
            raise ValueError(f"Checkpoint signature mismatch: {path}")
        for candidate in checkpoint["results"]:
            if candidate["status"] != "complete" or candidate["metrics"].get("wape_percent") is None:
                continue
            seen += 1
            result = {"arm": task["arm"], "candidate": candidate, "metrics": candidate["metrics"]}
            if best is None or _score_key(result) < _score_key(best):
                best = result
    if best is None:
        raise ValueError("No completed candidates for the selected group and cohort")
    return best, seen


def load_array(root, descriptor):
    """Verify the snapshot checksum before reading a prepared array."""
    path = root / descriptor["path"]
    if digest(path) != descriptor["sha256"]:
        raise ValueError(f"Prepared array checksum mismatch: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def feature_variants(group, selected):
    """Remove only the 27 expanded indicators, retaining the original holiday flag."""
    all_columns = selected["arm"]["columns"]
    names = group["feature_names"]
    expanded = {name for name in names if name.startswith("holiday_hour_")}
    expanded.update({"day_before_holiday", "day_after_holiday", "weekday_bridge_between_holiday_and_weekend"})
    if len(expanded) != 27 or not expanded.issubset(names):
        raise ValueError("Expected all 27 expanded holiday indicators in the snapshot")
    return {"expanded_holidays": all_columns,
            "original_holiday_flag_only": [index for index in all_columns if names[index] not in expanded]}


def main():
    """Print the selected candidate; fit only after the user explicitly adds --run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", default="outputs/price_feature_search_full_year_2026_cancellations_v2")
    parser.add_argument("--cohort", default="full_history")
    parser.add_argument("--output", default="outputs/price_holiday_winner_diagnostic")
    parser.add_argument("--day", default="2026-05-01")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    root = Path(args.search).resolve()
    prepared_path = root / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    group_name = "da_quarter_hourly"
    group = prepared["groups"][group_name]
    selected, seen = best_saved_candidate(root, prepared, group_name, args.cohort)
    variants = feature_variants(group, selected)
    plan = {"status": "plan_only", "group": group_name, "cohort": args.cohort,
        "saved_candidates_compared": seen, "candidate": selected, "fits_if_run": 2,
        "selected_day": args.day, "source_prepared_sha256": digest(prepared_path),
        "interpretation": "inner validation was used for model selection; selected training-day scores are in sample"}
    print(json.dumps(plan, indent=2), flush=True)
    if not args.run:
        return
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    files = group["files"]
    arrays = {name: load_array(root, files[name]) for name in ("x", "y", "anchor", "reference", "starts", "days")}
    cohort = group["cohorts"][args.cohort]
    training = load_array(root, cohort["indices"]["inner_train"])
    validation = load_array(root, cohort["indices"]["inner_validation"])
    selected_day = training[arrays["days"][training] == args.day]
    if not len(validation):
        raise ValueError("No inner-validation rows in selected cohort")
    evaluation = np.concatenate([validation, selected_day])
    parts = np.array(["inner_validation"] * len(validation) + ["training_day_in_sample"] * len(selected_day))
    outputs = []
    summary = {}
    for label, columns in variants.items():
        print(f"Fitting {label}", flush=True)
        _, prediction, warnings = _fit(selected["candidate"]["family"], selected["candidate"],
            np.asarray(arrays["x"][np.ix_(training, columns)]), arrays["y"][training] - arrays["anchor"][training],
            np.asarray(arrays["x"][np.ix_(evaluation, columns)]))
        prediction += arrays["anchor"][evaluation]
        frame = pd.DataFrame({"variant": label, "part": parts, "local_day": arrays["days"][evaluation],
            "product_start": pd.to_datetime(arrays["starts"][evaluation], utc=True),
            "actual": arrays["y"][evaluation], "prediction": prediction,
            "reference": arrays["reference"][evaluation]})
        frame["absolute_error"] = (frame.prediction - frame.actual).abs()
        outputs.append(frame)
        scores = point_metrics(frame.actual.iloc[:len(validation)], frame.prediction.iloc[:len(validation)],
                               frame.reference.iloc[:len(validation)])
        summary[label] = {"inner_validation_scores": scores, "warnings": warnings,
            "selected_training_day_mae": None if not len(selected_day) else float(frame.absolute_error.iloc[len(validation):].mean())}
    predictions = pd.concat(outputs, ignore_index=True)
    predictions.to_parquet(output / "predictions.parquet", index=False)
    daily = predictions.groupby(["variant", "part", "local_day"], as_index=False).agg(
        mae_eur_per_mwh=("absolute_error", "mean"), products=("absolute_error", "size"))
    daily.to_csv(output / "daily_mae.csv", index=False)
    final = {**plan, "status": "complete", "training_rows": len(training), "validation_rows": len(validation),
        "selected_training_day_rows": len(selected_day), "results": summary,
        "warning": "This diagnostic reuses the tail for model selection and May 1 is in-sample; it is not an unbiased test."}
    (output / "summary.json").write_text(json.dumps(final, indent=2, allow_nan=False) + "\n")
    print(json.dumps(final["results"], indent=2), flush=True)
    print(f"Saved daily errors and predictions in {output}", flush=True)


if __name__ == "__main__":
    main()
