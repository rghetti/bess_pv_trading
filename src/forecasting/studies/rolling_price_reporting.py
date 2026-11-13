"""Evaluation of rolling auction-price forecasts on explicitly matched products."""

from pathlib import Path
import json

import numpy as np
import pandas as pd


def _metrics(frame):
    """Calculate point and spread-direction scores on finite observed products."""
    actual = frame["actual"].to_numpy(dtype=float)
    prediction = frame["prediction"].to_numpy(dtype=float)
    reference = frame["reference"].to_numpy(dtype=float)
    valid = np.isfinite(actual) & np.isfinite(prediction) & np.isfinite(reference)
    actual, prediction, reference = actual[valid], prediction[valid], reference[valid]
    count = len(actual)
    if not count:
        return {"observations": 0, "mae": None, "wape_percent": None, "rmse": None,
                "bias": None, "reference_mae": None, "relative_mae": None,
                "spread_direction_accuracy": None}
    error = prediction - actual
    baseline = float(np.mean(np.abs(reference - actual)))
    denominator = float(np.abs(actual).sum())
    return {"observations": count, "mae": float(np.abs(error).mean()),
            "wape_percent": float(100 * np.abs(error).sum() / denominator) if denominator else None,
            "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean()),
            "reference_mae": baseline,
            "relative_mae": float(np.abs(error).mean() / baseline) if baseline else None,
            "spread_direction_accuracy": float(np.mean(np.sign(prediction - reference) == np.sign(actual - reference)))}


def rolling_price_report(forecasts, output, *, timezone="Europe/Rome", evaluation_end_exclusive="2026-11-01",
                         pair_baseline=None, expected_variants=None, minimum_common_rows=96, plots=False):
    """Write grouped metrics and paired variants without scoring the reserved period.

    ``forecasts`` contains one row per variant, gate, auction and UTC product.
    Variant comparisons use their common finite product observations, and also
    record rows lost from each side. No estimator fitting occurs in this module.
    """
    required = {"delivery_day", "product_start", "auction", "gate", "variant", "actual", "prediction", "reference"}
    missing = required - set(forecasts.columns)
    if missing:
        raise ValueError(f"Missing forecast columns: {sorted(missing)}")
    frame = forecasts.copy()
    frame["product_start"] = pd.to_datetime(frame["product_start"], utc=True)
    frame["delivery_day"] = frame["delivery_day"].astype(str)
    keys = ["auction", "gate", "variant", "product_start"]
    if frame.duplicated(keys).any():
        raise ValueError("Duplicate forecast product for the same variant and gate")
    reserved = int((frame.delivery_day >= evaluation_end_exclusive).sum())
    frame = frame.loc[frame.delivery_day < evaluation_end_exclusive].copy()
    frame["month"] = frame.delivery_day.str[:7]
    frame["local_hour"] = frame.product_start.dt.tz_convert(timezone).dt.hour
    if "is_holiday" not in frame:
        frame["is_holiday"] = pd.NA
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    identity = [name for name in ("family", "window_days", "ida_target") if name in frame.columns]
    grouping = ["auction", "gate", *identity, "variant"]
    for label, columns in (("overall", grouping), ("monthly", grouping + ["month"]),
                           ("hourly", grouping + ["local_hour"]),
                           ("holiday", grouping + ["is_holiday"])):
        records = []
        for key, rows in frame.groupby(columns, dropna=False, sort=True):
            records.append({**dict(zip(columns, key)), "input_rows": len(rows), **_metrics(rows)})
        pd.DataFrame(records).to_csv(destination / f"{label}.csv", index=False)
    paired = []
    # Compare every pair on the same products, never by independent aggregate MAEs.
    pair_columns = ["auction", "gate", *identity]
    for key, group in frame.groupby(pair_columns, sort=True):
        pair_identity = dict(zip(pair_columns, key))
        variants = sorted(expected_variants if expected_variants is not None else group.variant.dropna().unique())
        for number, left in enumerate(variants):
            for right in variants[number + 1:]:
                if pair_baseline is not None and pair_baseline not in (left, right):
                    continue
                columns = ["product_start", "actual", "prediction", "reference"]
                if "feature_inputs_complete" in group.columns:
                    columns.append("feature_inputs_complete")
                a = group.loc[group.variant.eq(left), columns]
                b = group.loc[group.variant.eq(right), columns]
                joined = a.merge(b, on="product_start", suffixes=("_a", "_b"), validate="one_to_one")
                valid = np.isfinite(joined[["actual_a", "prediction_a", "reference_a", "actual_b", "prediction_b", "reference_b"]]).all(axis=1)
                if "feature_inputs_complete" in group.columns:
                    valid &= joined.feature_inputs_complete_a.fillna(False) & joined.feature_inputs_complete_b.fillna(False)
                joined = joined.loc[valid]
                if not np.allclose(joined.actual_a, joined.actual_b, rtol=0, atol=1e-8):
                    raise ValueError(f"Actual prices differ between {left} and {right}")
                first = pd.DataFrame({"actual": joined.actual_a, "prediction": joined.prediction_a, "reference": joined.reference_a})
                second = pd.DataFrame({"actual": joined.actual_b, "prediction": joined.prediction_b, "reference": joined.reference_b})
                paired.append({**pair_identity, "variant_a": left, "variant_b": right,
                    "rows_a": len(a), "rows_b": len(b), "matched_rows": len(joined),
                    "excluded_a": len(a) - len(joined), "excluded_b": len(b) - len(joined),
                    "status": "available" if len(joined) >= minimum_common_rows else "insufficient_common_data",
                    "mae_a": _metrics(first)["mae"], "mae_b": _metrics(second)["mae"],
                    "reference_mae_a": _metrics(first)["reference_mae"],
                    "reference_mae_b": _metrics(second)["reference_mae"],
                    "references_match": bool(np.allclose(joined.reference_a, joined.reference_b, rtol=0, atol=1e-8))})
    pd.DataFrame(paired).to_csv(destination / "paired_variants.csv", index=False)
    report = {"evaluation_end_exclusive": evaluation_end_exclusive, "reserved_rows_not_scored": reserved,
              "scored_input_rows": len(frame), "matched_comparisons": len(paired),
              "interpretation": "historical forecast evaluation; selection must use earlier data only"}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if plots and len(frame):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        monthly = pd.read_csv(destination / "monthly.csv")
        figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
        for axis, auction in zip(axes.flat, ("DA", "IDA1", "IDA2", "IDA3")):
            # Show each auction at its own gate; all other gate tables remain available.
            subset = monthly[(monthly.auction == auction) & (monthly.gate == auction)]
            for variant, rows in subset.groupby("variant", sort=True):
                rows = rows.sort_values("month")
                axis.plot(rows.month, rows.mae, marker=".", label=variant)
            axis.set_title(auction)
            axis.set_ylabel("MAE (EUR/MWh)")
            axis.tick_params(axis="x", labelrotation=60)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            figure.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
        figure.suptitle("Rolling auction price forecasts: own-gate monthly MAE", y=1.02)
        figure.tight_layout()
        figure.savefig(destination / "monthly_mae.png", dpi=140, bbox_inches="tight")
        plt.close(figure)
    return report


def compare_october_pointwise(forecasts, pointwise_dir, output):
    """Compare own-gate October rows with annual models without training labels."""
    root = Path(pointwise_dir)
    groups = ["auction", "gate", "variant"]
    groups += [name for name in ("family", "window_days", "ida_target") if name in forecasts]
    frame = forecasts.loc[forecasts.delivery_day.astype(str).str.startswith("2026-10")].copy()
    frame = frame.loc[frame.auction.eq(frame.gate)]
    records = []
    for key, rows in frame.groupby(groups, sort=True):
        identity = dict(zip(groups, key))
        filename = root / f"{identity['auction'].lower()}_quarter_hourly__main.parquet"
        if not filename.is_file():
            continue
        baseline = pd.read_parquet(filename)
        baseline = baseline.loc[~baseline.used_for_training & baseline.local_day.str.startswith("2026-10")]
        baseline = baseline[["product_start", "actual_eur_per_mwh", "forecast_eur_per_mwh"]]
        candidate = rows.copy()
        candidate["product_start"] = pd.to_datetime(candidate.product_start, utc=True)
        merged = candidate.merge(baseline, on="product_start", how="inner", validate="one_to_one")
        valid = np.isfinite(merged[["actual", "prediction", "actual_eur_per_mwh", "forecast_eur_per_mwh"]]).all(axis=1)
        merged = merged.loc[valid]
        if not np.allclose(merged.actual, merged.actual_eur_per_mwh, rtol=0, atol=1e-8):
            raise ValueError("Pointwise and rolling actual prices differ on matched products")
        records.append({**identity, "rolling_rows": len(candidate), "pointwise_rows": len(baseline),
            "matched_rows": len(merged), "rolling_mae": float(abs(merged.prediction - merged.actual).mean()) if len(merged) else None,
            "pointwise_mae": float(abs(merged.forecast_eur_per_mwh - merged.actual).mean()) if len(merged) else None,
            "interpretation": "October only; annual pointwise refit excluded training-period rows"})
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)
    return len(records)
