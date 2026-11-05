"""Evaluate chronological ranking of saved DA forecast candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config


def checksum(path):
    """Fingerprint a saved input used by the ranking experiment."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def online_rank(actual, forecasts, starts_ns, ends_ns, available_ns, decisions_ns, days, *, window_days, top_k):
    """Rank by eligible earlier-product MAE and average the top candidates."""
    actual = np.asarray(actual, dtype=float)
    forecasts = np.asarray(forecasts, dtype=float)
    days = np.asarray(days, dtype=str)
    starts_ns, ends_ns, available_ns, decisions_ns = [np.asarray(x, dtype=np.int64) for x in
        (starts_ns, ends_ns, available_ns, decisions_ns)]
    if (forecasts.ndim != 2 or forecasts.shape[0] != len(actual) or not len(actual)
            or any(len(x) != len(actual) for x in (starts_ns, ends_ns, available_ns, decisions_ns, days))
            or not np.isfinite(actual).all() or not np.isfinite(forecasts).all()
            or np.any(np.diff(starts_ns) <= 0)):
        raise ValueError("Ranking inputs must be aligned, finite and chronological")
    if (isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 1
            or isinstance(top_k, bool) or not isinstance(top_k, int) or not 1 <= top_k <= forecasts.shape[1]):
        raise ValueError("Ranking window and top-k must be valid positive integers")
    prediction = np.empty(len(actual))
    selections = {}
    window_ns = pd.Timedelta(days=window_days).value
    for day in np.unique(days):
        target = days == day
        cutoff = decisions_ns[target].min()
        # A previous forecast error is eligible only after publication and delivery end.
        eligible = ((days < day) & (available_ns <= cutoff) & (ends_ns <= cutoff)
                    & (starts_ns >= cutoff - window_ns))
        if eligible.any():
            errors = np.abs(forecasts[eligible] - actual[eligible, None]).mean(axis=0)
            selected = np.argsort(errors, kind="stable")[:top_k]
        else:
            # With no prior evidence, use a symmetric equal-weight fallback.
            selected = np.arange(forecasts.shape[1])
        prediction[target] = forecasts[target][:, selected].mean(axis=1)
        selections[day] = selected.tolist()
    return prediction, selections


def run(settings, root):
    """Tune the ranking rule on earlier dates and evaluate it on later dates."""
    if not (settings["tune_start"] < settings["tune_end_exclusive"] <= settings["test_start"]
            < settings["test_end_exclusive"]):
        raise ValueError("Tuning and test dates must be chronological and nonoverlapping")
    source = root / settings["source_search"]
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("failures") or not manifest.get("selection"):
        raise ValueError("Ranking requires a completed early-selected source replay")
    group = settings["group"]
    split = manifest["config"]["product_groups"][group]
    if settings["tune_start"] < split["validation_start"] or settings["test_end_exclusive"] > split["validation_end"]:
        raise ValueError("Ranking dates extend beyond source predictions")
    candidates = sorted((x for x in manifest["results"] if x["group"] == group and not x.get("convergence_warnings")),
                        key=lambda x: (x["feature_set"], x["candidate"]))
    if not candidates:
        raise ValueError("No converged candidate forecasts are available")
    # Read clocks from one source matrix and verify all prediction streams match it.
    array_path = source / f"{group}__{candidates[0]['feature_set']}.npz"
    with np.load(array_path, allow_pickle=False) as data:
        starts = pd.DatetimeIndex(pd.to_datetime(data["starts"], utc=True))
        actual = data["validation_y"].copy()
        ends_ns, available_ns, decisions_ns = [data[key].copy() for key in ("product_end_ns", "available_ns", "decisions_ns")]
    names, columns, source_hashes = [], [], {}
    for candidate in candidates:
        name = candidate["feature_set"] + "/" + candidate["candidate"]
        path = source / f"{group}__{candidate['feature_set']}__{candidate['candidate']}.csv"
        frame = pd.read_csv(path, parse_dates=["product_start"])
        if (not starts.equals(pd.DatetimeIndex(frame.product_start))
                or not np.allclose(actual, frame.actual, rtol=0, atol=1e-10)):
            raise ValueError(f"Forecast {name} does not match source products and targets")
        names.append(name)
        columns.append(frame.point.to_numpy(float))
        source_hashes[name] = checksum(path)
    reference_name = settings["reference_candidate"]
    if reference_name not in names:
        raise ValueError("Declared reference candidate is absent")
    forecasts = np.column_stack(columns)
    days = starts.tz_convert(settings["timezone"]).strftime("%Y-%m-%d").to_numpy()
    tune = (days >= settings["tune_start"]) & (days < settings["tune_end_exclusive"])
    test = (days >= settings["test_start"]) & (days < settings["test_end_exclusive"])
    if not tune.any() or not test.any():
        raise ValueError("Both tuning and evaluation need observed products")
    candidates_and_predictions = []
    for window in settings["window_days"]:
        for count in settings["top_k"]:
            prediction, selections = online_rank(actual, forecasts, starts.as_unit("ns").asi8,
                ends_ns, available_ns, decisions_ns, days, window_days=window, top_k=count)
            candidates_and_predictions.append({"window_days": window, "top_k": count,
                "tune_mae": float(np.mean(np.abs(actual[tune] - prediction[tune]))),
                "prediction": prediction, "selections": selections})
    if not candidates_and_predictions:
        raise ValueError("Ranking grid cannot be empty")
    selected = min(candidates_and_predictions, key=lambda item: (item["tune_mae"], item["window_days"], item["top_k"]))
    reference = forecasts[:, names.index(reference_name)]
    denominator = float(np.abs(actual[test]).sum())
    if not denominator:
        raise ValueError("WAPE is undefined when actual prices sum to zero in absolute value")

    def score(prediction):
        """Summarize one forecast on exactly the same later products."""
        errors = np.abs(actual[test] - prediction[test])
        return {"mae_eur_per_mwh": float(errors.mean()), "wape_percent": float(100 * errors.sum() / denominator)}

    result = {"status": "complete", "settings": settings, "source_manifest_sha256": checksum(manifest_path),
        "source_array_sha256": checksum(array_path), "source_prediction_sha256": source_hashes,
        "candidate_names": names, "tuning_products": int(tune.sum()), "test_products": int(test.sum()),
        "selected": {"window_days": selected["window_days"], "top_k": selected["top_k"], "tune_mae": selected["tune_mae"]},
        "trials": [{key: item[key] for key in ("window_days", "top_k", "tune_mae")} for item in candidates_and_predictions],
        "reference": score(reference), "equal_weight": score(forecasts.mean(axis=1)), "ranked": score(selected["prediction"]),
        "selected_by_day": {day: [names[index] for index in indices] for day, indices in selected["selections"].items()}}
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"product_start": starts[test], "actual": actual[test], "reference": reference[test],
                  "equal_weight": forecasts.mean(axis=1)[test], "ranked": selected["prediction"][test]}).to_csv(
        output / "test_predictions.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return output


def main():
    """Read the configured ranking experiment and explicit output root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    print(run(load_config(args.config)["online_price_ranking"], args.project_root.resolve()))


if __name__ == "__main__":
    main()
