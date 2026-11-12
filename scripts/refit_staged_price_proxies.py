"""Refit selected staged-search price models and export retrospective price proxies.

The source search supplies immutable matrices, the selected model specifications,
and the exact annual training rows. Run this script explicitly after the search.
"""

from __future__ import annotations

import argparse
from datetime import date, timedelta
import hashlib
import json
import os
from pathlib import Path

# Prevent the numerical libraries imported below from using shared GPU/CPU pools.
for variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "BLIS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import joblib
import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.price_feature_data import digest
from bess_pv_trading.forecasting.studies.price_feature_models import point_metrics
from bess_pv_trading.forecasting.studies.price_feature_search import (
    _fit,
    _implementation,
    _software,
    run_lock,
    write_json,
)
from scripts.refit_full_year_price_proxies import coverage_table, load_array


def validate_winners(root, prepared, winners):
    """Reject changed inputs, incomplete selection, and target-derived columns."""
    prepared_path = root / "prepared.json"
    if winners.get("status") != "complete":
        raise ValueError("The staged search has not completed winner selection")
    if winners.get("prepared_sha256") != digest(prepared_path):
        raise ValueError("The staged winners do not match the frozen prepared inputs")
    if winners.get("scientific_config_sha256") != prepared["scientific_config_sha256"]:
        raise ValueError("The staged winners have a different scientific configuration")
    if prepared["implementation_sha256"] != _implementation():
        raise ValueError("The model implementation differs from the frozen search")
    if prepared["software"] != _software(prepared["scientific_settings"]):
        raise ValueError("The model libraries differ from the frozen search")

    # Source-builder adapters determine historical fallbacks and holidays.
    for key in ("full_year_adapter_sha256", "staged_implementation_sha256"):
        for path, expected in prepared["scientific_settings"].get(key, {}).items():
            if not Path(path).is_file() or digest(Path(path)) != expected:
                raise ValueError(f"The frozen source adapter changed: {path}")

    groups = winners.get("groups", {})
    if set(groups) != set(prepared["groups"]):
        raise ValueError("The staged winners do not cover every prepared auction")
    for name, group in prepared["groups"].items():
        selected = groups[name]
        if not selected.get("main"):
            raise ValueError(f"No main winner is available for {name}")
        for variant in ("main", "retrospective"):
            winner = selected.get(variant)
            if winner is None:
                continue
            arm = winner["arm"]
            if arm["group"] != name or arm["cohort"] not in group["cohorts"]:
                raise ValueError(f"The {variant} winner has an invalid group/cohort for {name}")
            columns = arm["columns"]
            if not columns or len(set(columns)) != len(columns):
                raise ValueError(f"The {variant} winner has invalid feature columns for {name}")
            if min(columns) < 0 or max(columns) >= len(group["feature_names"]):
                raise ValueError(f"The {variant} winner contains an out-of-range feature")
            features = [group["feature_names"][column] for column in columns]
            if any("spread_same_day" in feature for feature in features):
                raise ValueError(f"The {variant} winner uses a forbidden same-day price spread")
            if arm.get("target_price_embedded_in_features"):
                raise ValueError(f"The {variant} winner contains its target price")
            # Main models must not silently acquire retrospective columns.
            if variant == "main":
                for block, block_columns in group["block_columns"].items():
                    availability = prepared["scientific_settings"]["feature_blocks"][block]["availability"]
                    if availability.startswith("retrospective") and set(columns) & set(block_columns):
                        raise ValueError(f"The main winner contains retrospective block {block}")


def plot_predictions(frame, destination, timezone):
    """Plot monthly descriptive errors and the best and worst observed days."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination.mkdir(parents=True, exist_ok=True)
    observed = frame.loc[frame.actual_eur_per_mwh.notna()].copy()
    if observed.empty:
        return []
    observed["absolute_error"] = (
        observed.forecast_eur_per_mwh - observed.actual_eur_per_mwh
    ).abs()
    observed["month"] = observed.local_day.str.slice(0, 7)
    figure, axis = plt.subplots(figsize=(9, 4))
    observed.groupby("month").absolute_error.mean().plot.bar(ax=axis)
    axis.set(xlabel="Delivery month", ylabel="MAE (EUR/MWh)",
             title="Retrospective proxy: descriptive monthly MAE")
    figure.tight_layout()
    monthly = destination / "monthly_mae.png"
    figure.savefig(monthly, dpi=130)
    plt.close(figure)

    # Daily examples display observed prices and the saved proxy on the local clock.
    daily = observed.groupby("local_day").absolute_error.mean().sort_values()
    outputs = [str(monthly)]
    for label, day in (("best", daily.index[0]), ("worst", daily.index[-1])):
        selected = observed.loc[observed.local_day.eq(day)]
        figure, axis = plt.subplots(figsize=(10, 4))
        local_time = selected.product_start.dt.tz_convert(timezone)
        axis.plot(local_time, selected.actual_eur_per_mwh, label="Actual")
        axis.plot(local_time, selected.forecast_eur_per_mwh, label="Proxy")
        axis.set(title=f"{day}: {label} observed day", ylabel="Price (EUR/MWh)")
        axis.legend()
        figure.tight_layout()
        path = destination / f"{label}_{day}.png"
        figure.savefig(path, dpi=130)
        plt.close(figure)
        outputs.append(str(path))
    return outputs


def refit_winner(root, destination, group_name, group, winner, variant, settings, split, auction,
                 eligibility_columns=None):
    """Fit all annual rows eligible for the selected features and export observed targets."""
    arrays = {
        name: load_array(root, group["files"][name])
        for name in ("x", "y", "anchor", "reference", "starts", "decisions", "days", "publication")
    }
    arm = winner["arm"]
    selection_indices = np.asarray(load_array(root, group["cohorts"][arm["cohort"]]["indices"]["train"]))
    columns = arm["columns"]
    matrix = np.asarray(arrays["x"][:, columns])
    # A matched baseline uses its own features but the retrospective model's row eligibility.
    row_columns = columns if eligibility_columns is None else eligibility_columns
    eligibility_matrix = np.asarray(arrays["x"][:, row_columns])
    days = arrays["days"]
    if not len(selection_indices) or np.any(selection_indices < 0) or np.any(selection_indices >= len(days)):
        raise ValueError(f"The winning selection cohort has no valid annual rows for {group_name}/{variant}")

    # Refitting can recover rows omitted solely because an unselected block was missing.
    # Require annual labels to have been published by the first outer decision cutoff.
    decision_date = date.fromisoformat(split["validation_start"]) + timedelta(
        days=int(auction["decision_day_offset"])
    )
    cutoff = pd.Timestamp(f"{decision_date.isoformat()} {auction['decision_time']}",
                          tz=settings["timezone"]).tz_convert("UTC").value
    complete = (np.isfinite(matrix).all(axis=1) & np.isfinite(eligibility_matrix).all(axis=1)
                & np.isfinite(arrays["y"])
                & np.isfinite(arrays["anchor"]) & np.isfinite(arrays["reference"]))
    published = (arrays["publication"] != pd.NaT.value) & (arrays["publication"] <= cutoff)
    annual = (days >= split["train_start"]) & (days < split["train_end"])
    indices = np.flatnonzero(complete & published & annual)
    if not len(indices):
        raise ValueError(f"No eligible annual training rows for {group_name}/{variant}")

    # Prepared matrices contain observed native target rows; they are not a future calendar.
    eligible = ((days >= settings["prediction_start"]) & (days < settings["prediction_end_exclusive"])
                & complete)
    positions = np.flatnonzero(eligible)
    if not len(positions):
        raise ValueError(f"No observed target rows can be predicted for {group_name}/{variant}")
    candidate = winner["candidate"]
    model, predicted_spread, warnings = _fit(
        candidate["family"], candidate, matrix[indices],
        arrays["y"][indices] - arrays["anchor"][indices], matrix[positions],
    )
    frame = pd.DataFrame({
        "auction": group["market"],
        "product_start": pd.to_datetime(arrays["starts"][positions], utc=True),
        "decision_cutoff": pd.to_datetime(arrays["decisions"][positions], utc=True),
        "actual_eur_per_mwh": arrays["y"][positions],
        "forecast_eur_per_mwh": predicted_spread + arrays["anchor"][positions],
        "reference_eur_per_mwh": arrays["reference"][positions],
        "local_day": days[positions],
        "used_for_training": np.isin(positions, indices),
        "availability_class": arm["availability_class"],
        "provenance": "retrospective_annual_refit_on_2025_2026_labels",
    })
    name = f"{group_name}__{variant}"
    model_path = destination / f"{name}.joblib"
    prediction_path = destination / f"{name}.parquet"
    joblib.dump(model, model_path, compress=3)
    frame.to_parquet(prediction_path, index=False)
    coverage = coverage_table(days, frame.local_day, group["market"], settings, settings["timezone"],
                              settings.get("cancelled_days", {}).get(group["market"], ()))
    coverage_path = destination / f"{name}_daily_coverage.csv"
    coverage.to_csv(coverage_path, index=False)
    plots = plot_predictions(frame, destination / "plots" / name, settings["timezone"])
    return {
        "group": group_name,
        "auction": group["market"],
        "variant": variant,
        "selected": winner,
        "feature_names": [group["feature_names"][column] for column in columns],
        "train_rows": len(indices),
        "selection_cohort_train_rows": len(selection_indices),
        "train_index_sha256": hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest(),
        "prediction_index_sha256": hashlib.sha256(np.asarray(positions, dtype="<i8").tobytes()).hexdigest(),
        "row_eligibility_feature_names": [group["feature_names"][column] for column in row_columns],
        "training_row_policy": "All annual observed target rows with complete eligibility inputs, published by the first outer validation decision.",
        "first_validation_decision_utc": pd.Timestamp(cutoff, tz="UTC").isoformat(),
        "train_first_day": str(min(days[indices])),
        "train_last_day": str(max(days[indices])),
        "prediction_rows": len(frame),
        "prediction_first_day": frame.local_day.min(),
        "prediction_last_day": frame.local_day.max(),
        "prediction_rows_used_for_training": int(frame.used_for_training.sum()),
        "scores_descriptive": point_metrics(frame.actual_eur_per_mwh, frame.forecast_eur_per_mwh,
                                              frame.reference_eur_per_mwh),
        "warnings": warnings,
        "model_path": str(model_path),
        "model_sha256": digest(model_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": digest(prediction_path),
        "coverage_path": str(coverage_path),
        "coverage_sha256": digest(coverage_path),
        "plots": plots,
        "complete_prediction_days": int(coverage.status.eq("complete").sum()),
        "documented_cancelled_days": int(coverage.documented_cancelled.sum()),
    }


def main():
    """Refuse to run a model experiment until the user supplies --run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/price_staged_proxy_refit.yaml")
    parser.add_argument("--search", help="Completed staged search output directory")
    parser.add_argument("--output", help="New output directory for proxy artifacts")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    settings = load_config(args.config)["price_staged_proxy_refit"]
    if args.search:
        settings["source_search"] = args.search
    if args.output:
        settings["output_directory"] = args.output
    root = Path(settings["source_search"]).resolve()
    destination = Path(settings["output_directory"]).resolve()
    if not args.run:
        print(json.dumps({"action": "plan_only", "settings": settings,
                          "fits": "one main plus, where available, one retrospective and one matched-main refit per auction (up to 12 total)",
                          "forecast_coverage": "observed native target rows only"}, indent=2))
        return
    if root == destination or destination in root.parents:
        raise ValueError("The proxy output must be a new directory distinct from the search")
    with run_lock(root):
        prepared_path = root / "prepared.json"
        winners_path = root / "winners.json"
        prepared = json.loads(prepared_path.read_text())
        winners = json.loads(winners_path.read_text())
        validate_winners(root, prepared, winners)
        settings = {**settings, "timezone": prepared["scientific_settings"]["timezone"]}
        input_manifest_path = root / "inputs" / "manifest.json"
        input_manifest = json.loads(input_manifest_path.read_text())
        notices = input_manifest["gme_cancellation_notices"]
        auctions = prepared["scientific_settings"]["auctions"]
        settings["cancelled_days"] = {
            market: sorted({row["delivery_day"] for row in notices if row["dataset"] == auction["dataset"]})
            for market, auction in auctions.items()
        }
        # A new destination prevents accidental overwrites of existing proxy results.
        destination.mkdir(parents=True, exist_ok=False)
        results = []
        unavailable = []
        for group_name, selected in sorted(winners["groups"].items()):
            main_winner = selected["main"]
            retro_winner = selected.get("retrospective")
            variants = [("main", main_winner, None)]
            if retro_winner is None:
                unavailable.append({"group": group_name, "variant": "retrospective"})
            else:
                # The added matched baseline isolates the retrospective block's effect
                # on identical annual training and prediction rows.
                if main_winner["candidate"] != retro_winner["candidate"]:
                    raise ValueError(f"Retrospective candidate differs from main winner for {group_name}")
                if not set(main_winner["arm"]["columns"]) <= set(retro_winner["arm"]["columns"]):
                    raise ValueError(f"Retrospective features do not include the main winner for {group_name}")
                matched = {**main_winner, "arm": {**main_winner["arm"], "cohort": retro_winner["arm"]["cohort"]}}
                variants.extend([("main_matched_retrospective", matched, retro_winner["arm"]["columns"]),
                                 ("retrospective", retro_winner, None)])
            for variant, winner, eligibility_columns in variants:
                print(f"Refitting {group_name}/{variant}: {winner['candidate']['family']}", flush=True)
                results.append(refit_winner(
                    root, destination, group_name, prepared["groups"][group_name], winner, variant,
                    settings, prepared["scientific_settings"]["product_groups"][group_name],
                    auctions[prepared["groups"][group_name]["market"]], eligibility_columns,
                ))
        paired = {}
        for group_name in sorted(winners["groups"]):
            rows = {item["variant"]: item for item in results if item["group"] == group_name}
            if "retrospective" not in rows:
                continue
            baseline = rows["main_matched_retrospective"]
            retrospective = rows["retrospective"]
            if (baseline["train_index_sha256"] != retrospective["train_index_sha256"]
                    or baseline["prediction_index_sha256"] != retrospective["prediction_index_sha256"]):
                raise ValueError(f"Retrospective export lost matched row identity for {group_name}")
            paired[group_name] = {"train_rows": baseline["train_rows"],
                "prediction_rows": baseline["prediction_rows"],
                "prediction_rows_used_for_training": baseline["prediction_rows_used_for_training"],
                "matched_main_scores_descriptive": baseline["scores_descriptive"],
                "retrospective_scores_descriptive": retrospective["scores_descriptive"],
                "interpretation": "Matched annual refits and prediction rows; scores include training rows and are descriptive."}
        manifest = {
            "status": "complete",
            "settings": settings,
            "source_prepared_sha256": digest(prepared_path),
            "source_winners_sha256": digest(winners_path),
            "source_inputs_manifest_sha256": digest(input_manifest_path),
            "scientific_config_sha256": prepared["scientific_config_sha256"],
            "forecast_coverage": "Predictions exist only for observed native target rows in the pinned snapshot.",
            "interpretation": "Retrospective annual refit; some predicted rows were used in training. "
                              "IDA anchors use earlier cleared prices; the export is not a DA-gate forecast "
                              "of all later auctions. Descriptive scores are not independent validation.",
            "unavailable_variants": unavailable,
            "matched_retrospective_comparisons": paired,
            "results": results,
        }
        write_json(destination / "manifest.json", manifest)
    print(json.dumps({"status": "complete", "output": str(destination),
                      "variants": len(results), "unavailable": unavailable}, indent=2))


if __name__ == "__main__":
    main()
