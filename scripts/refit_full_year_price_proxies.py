"""Refit selected full-year search candidates to make retrospective 2026 prices."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

# Match search worker limits and leave shared GPUs untouched.
for variable in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[variable] = "1"

import joblib
import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.price_feature_data import digest
from bess_pv_trading.forecasting.studies.price_feature_models import point_metrics
from bess_pv_trading.forecasting.studies.price_feature_search import _fit, _implementation, _score_key, _software, run_lock
from scripts.run_price_in_sample_baseline import plot_group


def select_candidates(root, prepared, cohort):
    """Compare completed candidates on the same cohort's chronological inner tail."""
    chosen = {}
    failures = []
    tasks = [task for task in prepared["tasks"] if task["arm"]["cohort"] == cohort]
    if not tasks:
        raise ValueError(f"No eligible search candidates for cohort {cohort}")
    for task in tasks:
        path = root / "trials" / f"{task['id']}.json"
        if not path.is_file():
            raise ValueError(f"Search task is not complete: {task['id']}")
        checkpoint = json.loads(path.read_text())
        if checkpoint["signature"] != task["signature"]:
            raise ValueError(f"Checkpoint signature mismatch: {path}")
        rows = checkpoint["results"]
        if len(rows) != len(task["candidates"]) or {row["trial"] for row in rows} != {
                candidate["trial"] for candidate in task["candidates"]}:
            raise ValueError(f"Search task has unfinished candidates: {task['id']}")
        for row in rows:
            if row["status"] != "complete":
                failures.append({"task": task["id"], "trial": row["trial"], "error": row.get("error")})
                continue
            item = {"arm": task["arm"], "candidate": row, "metrics": row["metrics"]}
            group = task["arm"]["group"]
            if group not in chosen or _score_key(item) < _score_key(chosen[group]):
                chosen[group] = item
    missing = set(prepared["groups"]) - set(chosen)
    if missing:
        raise ValueError(f"No successful candidates for requested cohort in: {sorted(missing)}")
    return chosen, failures


def load_array(root, descriptor):
    """Check a prepared array's original checksum before reading it."""
    path = root / descriptor["path"]
    if digest(path) != descriptor["sha256"]:
        raise ValueError(f"Prepared input array changed: {path}")
    return np.load(path, mmap_mode="r", allow_pickle=False)


def coverage_table(days, predicted_days, market, settings, timezone):
    """Expose every missing observed or predicted product, including future days."""
    first = pd.Timestamp(settings["prediction_start"], tz=timezone)
    last = pd.Timestamp(settings["prediction_end_exclusive"], tz=timezone)
    grid = pd.date_range(first, last, freq="15min", inclusive="left")
    if market == "IDA3":
        grid = grid[grid.hour >= 12]
    expected = pd.Series(grid.strftime("%Y-%m-%d")).value_counts().sort_index()
    observed = pd.Series(days).value_counts()
    predicted = pd.Series(predicted_days).value_counts()
    return pd.DataFrame({"local_day": expected.index, "expected_products": expected.to_numpy(),
        "observed_products": observed.reindex(expected.index, fill_value=0).to_numpy(),
        "predicted_products": predicted.reindex(expected.index, fill_value=0).to_numpy()})


def refit_group(root, destination, group_name, group, selected, settings):
    """Fit the winner on its full matched annual cohort and export available rows."""
    files = group["files"]
    arrays = {name: load_array(root, files[name]) for name in
              ("x", "y", "anchor", "reference", "starts", "decisions", "days")}
    indices = load_array(root, group["cohorts"][settings["cohort"]]["indices"]["train"])
    columns = selected["arm"]["columns"]
    matrix = np.asarray(arrays["x"][:, columns])
    days = arrays["days"]
    eligible = ((days >= settings["prediction_start"]) & (days < settings["prediction_end_exclusive"])
                & np.isfinite(matrix).all(axis=1) & np.isfinite(arrays["y"])
                & np.isfinite(arrays["anchor"]) & np.isfinite(arrays["reference"]))
    positions = np.flatnonzero(eligible)
    if not len(indices) or not len(positions):
        raise ValueError(f"No usable annual training or 2026 prediction rows for {group_name}")
    model, spread, warnings = _fit(selected["candidate"]["family"], selected["candidate"],
        matrix[indices], arrays["y"][indices] - arrays["anchor"][indices], matrix[positions])
    predicted = spread + arrays["anchor"][positions]
    frame = pd.DataFrame({"auction": group["market"],
        "product_start": pd.to_datetime(arrays["starts"][positions], utc=True),
        "decision_cutoff": pd.to_datetime(arrays["decisions"][positions], utc=True),
        "actual_eur_per_mwh": arrays["y"][positions], "forecast_eur_per_mwh": predicted,
        "reference_eur_per_mwh": arrays["reference"][positions], "local_day": days[positions],
        "used_for_training": np.isin(positions, indices),
        "availability_class": selected["arm"]["availability_class"],
        "provenance": "retrospective_selected_on_2025_2026_training_data"})
    model_path = destination / f"{group_name}.joblib"
    prediction_path = destination / f"{group_name}.parquet"
    joblib.dump(model, model_path, compress=3)
    frame.to_parquet(prediction_path, index=False)
    coverage = coverage_table(days, frame.local_day, group["market"], settings, settings["timezone"])
    coverage_path = destination / f"{group_name}_daily_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    return {"group": group_name, "auction": group["market"], "selected": selected,
        "feature_names": [group["feature_names"][column] for column in columns],
        "train_rows": len(indices), "train_first_day": str(sorted(days[indices])[0]),
        "train_last_day": str(sorted(days[indices])[-1]), "prediction_rows": len(frame),
        "prediction_first_day": frame.local_day.min(), "prediction_last_day": frame.local_day.max(),
        "prediction_rows_used_for_training": int(frame.used_for_training.sum()),
        "scores_descriptive": point_metrics(frame.actual_eur_per_mwh, frame.forecast_eur_per_mwh,
                                             frame.reference_eur_per_mwh),
        "warnings": warnings, "model_path": str(model_path), "model_sha256": digest(model_path),
        "prediction_path": str(prediction_path), "prediction_sha256": digest(prediction_path),
        "coverage_path": str(coverage_path), "coverage_sha256": digest(coverage_path),
        "complete_prediction_days": int((coverage.predicted_products == coverage.expected_products).sum())}


def main():
    """Run only on explicit request, after all selected-cohort search tasks finish."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/price_full_year_proxy_refit.yaml")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = load_config(args.config)["price_full_year_proxy_refit"]
    root = Path(settings["source_search"]).resolve()
    destination = Path(settings["output_directory"]).resolve()
    if not args.run:
        print(json.dumps({"action": "plan_only", "settings": settings, "fits": 4,
            "selection": "best inner WAPE then MAE within a matched cohort", "in_sample": True}, indent=2))
        return
    with run_lock(root):
        prepared_path = root / "prepared.json"
        prepared = json.loads(prepared_path.read_text())
        if prepared["implementation_sha256"] != _implementation() or prepared["software"] != _software(prepared["scientific_settings"]):
            raise ValueError("Search code or software differs from the frozen search; use its original environment")
        settings = {**settings, "timezone": prepared["scientific_settings"]["timezone"]}
        selected, failures = select_candidates(root, prepared, settings["cohort"])
        destination.mkdir(parents=True, exist_ok=False)
        results = []
        for group_name in sorted(selected):
            print(f"Refitting {group_name}: {selected[group_name]['candidate']['family']}", flush=True)
            result = refit_group(root, destination, group_name, prepared["groups"][group_name],
                                 selected[group_name], settings)
            plot_group(result["prediction_path"], destination / "plots" / group_name,
                       prepared["scientific_settings"]["timezone"])
            results.append(result)
        manifest = {"status": "complete", "settings": settings, "source_prepared_sha256": digest(prepared_path),
            "scientific_config_sha256": prepared["scientific_config_sha256"],
            "selection": "pooled inner WAPE then MAE on matched cohort rows", "failed_candidates": failures,
            "interpretation": "Retrospective; training includes 2026 and IDAs use earlier realized auction prices.",
            "results": results}
        (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps([{key: row[key] for key in ("auction", "train_rows", "prediction_rows",
        "prediction_first_day", "prediction_last_day", "scores_descriptive")} for row in results], indent=2))


if __name__ == "__main__":
    main()
