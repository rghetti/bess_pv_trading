"""Pilot full-curve intraday spread prediction on a frozen price snapshot."""

import os
for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[name] = "1"

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def metric(actual, predicted, reference):
    """Score identical product observations against their latest cleared price."""
    error = predicted - actual
    baseline = np.abs(reference - actual).mean()
    return {"observations": int(actual.size), "mae": float(np.abs(error).mean()),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "wape_percent": float(100 * np.abs(error).sum() / np.abs(actual).sum()),
        "bias": float(error.mean()), "reference_mae": float(baseline),
        "relative_mae": float(np.abs(error).mean() / baseline)}


def main():
    """Fit small daily multi-output ridge models and score a held-out month."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", default="outputs/price_staged_search_full_year")
    parser.add_argument("--output", default="outputs/price_intraday_curve_ridge_pilot")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if not args.run:
        parser.error("Supply --run to fit the authorized CPU pilot")
    root = Path(args.search)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    prepared_path = root / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    report = {"prepared_sha256": hashlib.sha256(prepared_path.read_bytes()).hexdigest(),
        "training": "2025-10-01 through 2026-09-30; labels published by first October gate",
        "validation": "saved chronological inner-validation boundary; October is development",
        "model": "standardized multi-output ridge predicting price minus anchor; DA anchor is zero",
        "input_missingness": "training-median imputation and missing indicators; no target imputation",
        "excluded": "nonstandard product count days or missing labels/anchors; November untouched",
        "threads": 1, "auctions": {}}
    for market, nproducts in (("da", 96), ("ida1", 96), ("ida2", 96), ("ida3", 48)):
        group = prepared["groups"][f"{market}_quarter_hourly"]
        def array(name):
            desc = group["files"][name]
            path = root / desc["path"]
            if hashlib.sha256(path.read_bytes()).hexdigest() != desc["sha256"]:
                raise ValueError(f"Snapshot checksum mismatch: {path}")
            return np.load(path, mmap_mode="r")
        x, y, anchor, reference, days, starts, publication, decisions = (array(n) for n in
            ("x", "y", "anchor", "reference", "days", "starts", "publication", "decisions"))
        names = group["feature_names"]
        daily_names = [n for n in names if n.startswith("own_curve_") or n in
            {"weekday_sin", "weekday_cos", "weekend", "annual_sin", "annual_cos",
             "italian_public_holiday", "day_before_holiday", "day_after_holiday",
             "weekday_bridge_between_holiday_and_weekend", "earlier_IDA1_cancelled",
             "earlier_IDA2_cancelled"}]
        curve_names = [n for n in ("cleared_DA", "cleared_IDA1", "cleared_IDA2",
            "own_price_lag_24h", "own_price_lag_48h", "own_price_lag_168h") if n in names]
        extra_names = [n for n in names if n.startswith(("nord_", "psv_", "fr_")) and "same_day" not in n]
        daily_columns = [names.index(n) for n in daily_names]
        curve_columns = [names.index(n) for n in curve_names]
        extra_columns = [names.index(n) for n in extra_names]
        split = group["split"]
        cohort = group["cohorts"]["stage_core"]["indices"]["inner_validation"]
        inner_idx = np.load(root / cohort["path"])
        inner_start = min(days[inner_idx].tolist())
        october = (days >= "2026-10-01") & (days < "2026-11-01")
        first_decision = int(np.min(decisions[october]))
        samples, excluded = [], []
        for day in np.unique(days[(days >= "2025-10-01") & (days < "2026-11-01")]):
            ix = np.flatnonzero(days == day)
            ix = ix[np.argsort(starts[ix])]
            if len(ix) != nproducts:
                excluded.append({"day": str(day), "reason": "nonstandard_or_missing_product_count", "rows": len(ix)})
                continue
            if not np.isfinite(y[ix]).all() or not np.isfinite(anchor[ix]).all() or not np.isfinite(reference[ix]).all():
                excluded.append({"day": str(day), "reason": "missing_target_anchor_or_reference", "rows": len(ix)})
                continue
            if day < "2026-10-01" and ((publication[ix] == pd.NaT.value).any() or (publication[ix] > first_decision).any()):
                excluded.append({"day": str(day), "reason": "label_not_published_by_october_gate", "rows": len(ix)})
                continue
            # The same cleared curve and historic data are available at the auction gate.
            features = np.r_[np.asarray(x[ix[0], daily_columns]), np.asarray(x[np.ix_(ix, curve_columns)]).ravel()]
            extra = np.asarray(x[np.ix_(ix, extra_columns)]).ravel()
            samples.append({"day": str(day), "features": features, "extra": extra, "actual": np.asarray(y[ix]), "reference": np.asarray(reference[ix]),
                "anchor": np.asarray(anchor[ix]), "starts": np.asarray(starts[ix])})
        matrix = np.stack([s["features"] for s in samples])
        target = np.stack([s["actual"] - s["anchor"] for s in samples])
        labels = np.stack([s["actual"] for s in samples])
        anchors = np.stack([s["anchor"] for s in samples])
        refs = np.stack([s["reference"] for s in samples])
        dates = np.array([s["day"] for s in samples])
        inner_train = dates < inner_start
        inner_validation = (dates >= inner_start) & (dates < "2026-10-01")
        annual = dates < "2026-10-01"
        test = dates >= "2026-10-01"
        def model(alpha):
            return make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                StandardScaler(), Ridge(alpha=alpha))
        extra_matrix = np.stack([s["extra"] for s in samples])
        variants = {}
        for variant, features in (("core", matrix), ("core_plus_available", np.column_stack([matrix, extra_matrix]))):
            trials = []
            for alpha in (100.0, 1000.0, 10000.0):
                fitted = model(alpha).fit(features[inner_train], target[inner_train])
                score = metric(labels[inner_validation], fitted.predict(features[inner_validation]) + anchors[inner_validation], refs[inner_validation])
                trials.append({"alpha": alpha, **score})
            chosen = min(trials, key=lambda row: (row["mae"], row["alpha"]))
            fitted = model(chosen["alpha"]).fit(features[annual], target[annual])
            joblib.dump(fitted, output / f"{market}_{variant}.joblib", compress=3)
            prediction = fitted.predict(features[test]) + anchors[test]
            forecast_rows = []
            for sample, forecast in zip(np.asarray(samples, dtype=object)[test], prediction):
                for start, actual, baseline, value in zip(sample["starts"], sample["actual"], sample["reference"], forecast):
                    forecast_rows.append({"local_day": sample["day"], "product_start": pd.Timestamp(int(start), tz="UTC"),
                        "actual_eur_per_mwh": actual, "reference_eur_per_mwh": baseline, "forecast_eur_per_mwh": value})
            pd.DataFrame(forecast_rows).to_parquet(output / f"{market}_{variant}_october.parquet", index=False)
            variants[variant] = {"input_width": int(features.shape[1]), "trials": trials,
                "chosen_alpha": chosen["alpha"], "october": metric(labels[test], prediction, refs[test]),
                "annual_missing_input_values": int((~np.isfinite(features[annual])).sum()),
                "october_missing_input_values": int((~np.isfinite(features[test])).sum())}
            print(market, variant, json.dumps(variants[variant]["october"]), flush=True)
        expected_october = pd.date_range("2026-10-01", "2026-10-31").strftime("%Y-%m-%d")
        absent_october = sorted(set(expected_october) - set(days[october].tolist()))
        report["auctions"][market] = {"input_daily_features": daily_names, "input_curve_features": curve_names,
            "input_width": int(matrix.shape[1]), "inner_start": inner_start,
            "inner_train_days": int(inner_train.sum()), "inner_validation_days": int(inner_validation.sum()),
            "annual_days": int(annual.sum()), "october_days": int(test.sum()), "extra_features": extra_names,
            "variants": variants, "excluded_days": excluded, "days_without_any_october_targets": absent_october}
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
