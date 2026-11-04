"""Evaluate forecast combinations using strictly ordered selection periods."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.price_search import point_metrics


def _hash(path):
    """Fingerprint the source manifest without relying on retrieval time."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _period_mask(days, start, end):
    """Use a half-open local-date interval for one chronological stage."""
    return (days >= start) & (days < end)


def evaluate_stacking(actual, reference, predictions, identifiers, days, settings):
    """Rank, fit and select convex combinations on successive days before testing."""
    from scipy.optimize import minimize

    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    days = np.asarray(days, dtype=str)
    if (predictions.ndim != 2 or predictions.shape != (len(actual), len(identifiers))
            or reference.shape != actual.shape or len(days) != len(actual) or not len(identifiers)
            or len(set(identifiers)) != len(identifiers) or not np.isfinite(predictions).all()
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
        raise ValueError("Stacking inputs must be aligned, unique and finite")
    rank = _period_mask(days, settings["rank_start"], settings["rank_end_exclusive"])
    tune = _period_mask(days, settings["tune_start"], settings["tune_end_exclusive"])
    test = _period_mask(days, settings["test_start"], settings["test_end_exclusive"])
    if not (settings["rank_start"] < settings["rank_end_exclusive"] <= settings["tune_start"]
            < settings["tune_end_exclusive"] <= settings["test_start"] < settings["test_end_exclusive"]):
        raise ValueError("Stacking periods must be chronological and nonoverlapping")
    tuning_days = np.unique(days[tune])
    fraction = settings["stacking"]["weight_fit_fraction"]
    if not np.isfinite(fraction) or not 0 < fraction < 1 or len(tuning_days) < 4 or not rank.any() or not test.any():
        raise ValueError("Stacking requires nonempty rank/test periods and at least four tuning days")
    split = max(1, min(len(tuning_days) - 1, int(len(tuning_days) * fraction)))
    fit = tune & np.isin(days, tuning_days[:split])
    validate = tune & np.isin(days, tuning_days[split:])
    errors = np.abs(predictions[rank] - actual[rank, None]).mean(axis=0)
    order = sorted(range(len(identifiers)), key=lambda index: (errors[index], identifiers[index]))
    sizes = settings["stacking"]["ensemble_sizes"]
    penalties = settings["stacking"]["ridge_penalties"]
    if (not sizes or len(set(sizes)) != len(sizes) or any(isinstance(k, bool) or not isinstance(k, int)
            or k < 1 or k > len(order) for k in sizes) or not penalties
            or any(not np.isfinite(value) or value < 0 for value in penalties)):
        raise ValueError("Invalid stacking sizes or regularization penalties")

    def fit_weights(indices, mask, penalty):
        """Fit nonnegative weights summing to one with shrinkage to equal weights."""
        x, y = predictions[mask][:, indices], actual[mask]
        uniform = np.full(len(indices), 1 / len(indices))
        # Scaling keeps regularization stable across differently priced periods.
        scale = max(float(np.mean(np.abs(y))), 1.0)
        objective = lambda weights: float(np.mean(((x @ weights - y) / scale) ** 2)
                                          + penalty * np.sum((weights - uniform) ** 2))
        result = minimize(objective, uniform, method="SLSQP", bounds=[(0, 1)] * len(indices),
                          constraints=[{"type": "eq", "fun": lambda weights: weights.sum() - 1}],
                          options={"maxiter": 500, "ftol": 1e-10})
        if not result.success or not np.isfinite(result.x).all() or abs(result.x.sum() - 1) > 1e-6:
            raise ValueError(f"Stacking optimization failed: {result.message}")
        return result.x

    trials = []
    for size in sizes:
        for penalty in penalties:
            indices = order[:size]
            weights = fit_weights(indices, fit, penalty)
            metrics = point_metrics(actual[validate], predictions[validate][:, indices] @ weights, reference[validate])
            trials.append({"size": size, "penalty": penalty, **metrics})
    selected = min(trials, key=lambda row: (row["mae_eur_per_mwh"], row["size"], row["penalty"]))
    indices = order[:selected["size"]]
    weights = fit_weights(indices, tune, selected["penalty"])
    forecast = predictions[test][:, indices] @ weights
    return {"selected_candidates": [identifiers[index] for index in indices], "weights": weights.tolist(),
            "selected_penalty": selected["penalty"], "trials": trials,
            "weight_fit_days": tuning_days[:split].tolist(), "weight_validation_days": tuning_days[split:].tolist(),
            "test": point_metrics(actual[test], forecast, reference[test])}, test, forecast


def evaluate_group(actual, reference, predictions, identifiers, days, settings):
    """Rank individual models, choose ensemble size, then score later observations."""
    actual = np.asarray(actual, dtype=float)
    reference = np.asarray(reference, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    days = np.asarray(days, dtype=str)
    if (predictions.ndim != 2 or predictions.shape != (len(actual), len(identifiers))
            or reference.shape != actual.shape or len(days) != len(actual) or not len(identifiers)):
        raise ValueError("Ensemble inputs must have aligned nonempty predictions and observations")
    if (len(set(identifiers)) != len(identifiers) or not np.isfinite(predictions).all()
            or not np.isfinite(actual).all() or not np.isfinite(reference).all()):
        raise ValueError("Ensemble candidates must be unique and finite")
    rank_start, rank_end = settings["rank_start"], settings["rank_end_exclusive"]
    tune_start, tune_end = settings["tune_start"], settings["tune_end_exclusive"]
    test_start, test_end = settings["test_start"], settings["test_end_exclusive"]
    if not rank_start < rank_end <= tune_start < tune_end <= test_start < test_end:
        raise ValueError("Ranking, ensemble-size tuning and evaluation periods must be ordered and nonoverlapping")
    rank = _period_mask(days, rank_start, rank_end)
    tune = _period_mask(days, tune_start, tune_end)
    test = _period_mask(days, test_start, test_end)
    if not rank.any() or not tune.any() or not test.any():
        raise ValueError("Each ensemble evaluation period requires observed intervals")
    # Deterministic ties ensure the selected list can be independently reproduced.
    errors = np.abs(predictions[rank] - actual[rank, None]).mean(axis=0)
    order = sorted(range(len(identifiers)), key=lambda index: (errors[index], identifiers[index]))
    sizes = settings["ensemble_sizes"]
    if (not sizes or len(set(sizes)) != len(sizes) or any(isinstance(k, bool) or not isinstance(k, int)
            or k < 1 or k > len(order) for k in sizes)):
        raise ValueError("Ensemble sizes must be distinct integers within the candidate count")
    tuning = []
    for size in sizes:
        forecast = predictions[tune][:, order[:size]].mean(axis=1)
        tuning.append({"size": size, **point_metrics(actual[tune], forecast, reference[tune])})
    chosen = min(tuning, key=lambda row: (row["mae_eur_per_mwh"], row["size"]))["size"]
    combined = predictions[test][:, order[:chosen]].mean(axis=1)
    single = predictions[test, order[0]]
    return {"ranked_candidates": [{"id": identifiers[index], "ranking_mae": float(errors[index])} for index in order],
            "tuning": tuning, "selected_size": chosen,
            "selected_candidates": [identifiers[index] for index in order[:chosen]],
            "test_combination": point_metrics(actual[test], combined, reference[test]),
            "test_best_single": point_metrics(actual[test], single, reference[test]),
            "period_intervals": {"rank": int(rank.sum()), "tune": int(tune.sum()), "test": int(test.sum())}}, test, combined


def run(config):
    """Read one completed search and retain selection evidence and later predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = config["price_ensemble_evaluation"]
    root = Path(config.get("_project_root", ROOT))
    source = root / settings["source_search"]
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("failures"):
        raise ValueError("Ensemble evaluation requires a completed search without failed candidates")
    if any(settings["test_end_exclusive"] > split["validation_end"]
           or settings["rank_start"] < split["validation_start"]
           for split in manifest["config"]["product_groups"].values()):
        raise ValueError("Ensemble periods extend beyond the source search evaluation")
    if {row["group"] for row in manifest["results"]} != set(manifest["config"]["product_groups"]):
        raise ValueError("Source search does not contain all declared market groups")
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    results = {"status": "running", "settings": settings, "source_manifest_sha256": _hash(manifest_path), "groups": {}}
    for group in sorted({row["group"] for row in manifest["results"]}):
        records = sorted((row for row in manifest["results"] if row["group"] == group),
                         key=lambda row: (row["feature_set"], row["candidate"]))
        split = manifest["config"]["product_groups"][group]
        group_features = split.get("feature_sets", manifest["config"]["feature_sets"])
        expected = {(feature, candidate["id"]) for feature in group_features
                    for candidate in manifest["config"]["candidates"]}
        observed = {(row["feature_set"], row["candidate"]) for row in records}
        if len(records) != len(expected) or observed != expected:
            raise ValueError(f"Incomplete or duplicated candidate results for {group}")
        identifiers, forecasts, actual, reference, starts = [], [], None, None, None
        for row in records:
            identifier = row["feature_set"] + "/" + row["candidate"]
            path = source / f"{group}__{row['feature_set']}__{row['candidate']}.csv"
            frame = pd.read_csv(path, parse_dates=["product_start"])
            if frame.product_start.duplicated().any() or not frame.product_start.is_monotonic_increasing:
                raise ValueError(f"Invalid product grid in {path}")
            if starts is None:
                starts = pd.DatetimeIndex(frame.product_start)
                actual = frame.actual.to_numpy(float)
                reference = frame.seasonal_naive.to_numpy(float)
            elif (not starts.equals(pd.DatetimeIndex(frame.product_start))
                    or not np.array_equal(actual, frame.actual.to_numpy(float))
                    or not np.array_equal(reference, frame.seasonal_naive.to_numpy(float))):
                raise ValueError(f"Candidate {identifier} does not share the same targets and reference")
            identifiers.append(identifier)
            forecasts.append(frame.point.to_numpy(float))
        days = starts.tz_convert(settings["timezone"]).strftime("%Y-%m-%d").to_numpy()
        result, test, forecast = evaluate_group(actual, reference, np.column_stack(forecasts), identifiers, days, settings)
        if settings.get("stacking"):
            # Fit and validate weights on earlier tuning days before scoring the later slice.
            stack, stack_test, stack_forecast = evaluate_stacking(
                actual, reference, np.column_stack(forecasts), identifiers, days, settings)
            if not np.array_equal(test, stack_test):
                raise ValueError("Stacking and equal-weight evaluation periods differ")
            result["stacking"] = stack
            pd.DataFrame({"product_start": starts[test], "actual": actual[test], "stacking": stack_forecast}).to_csv(
                output / f"{group}_stacking_predictions.csv", index=False)
        pd.DataFrame({"product_start": starts[test], "actual": actual[test], "seasonal_naive": reference[test],
                      "ensemble": forecast}).to_csv(output / f"{group}_test_predictions.csv", index=False)
        daily = pd.DataFrame({"day": days[test], "ensemble": np.abs(forecast - actual[test]),
                              "seasonal_naive": np.abs(reference[test] - actual[test])}).groupby("day").mean()
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(pd.to_datetime(daily.index), daily.ensemble, label="Selected combination")
        axis.plot(pd.to_datetime(daily.index), daily.seasonal_naive, label="Seasonal reference", alpha=0.8)
        axis.set(ylabel="Daily MAE (EUR/MWh)", title=group)
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(output / f"{group}_daily_mae.png", dpi=130)
        plt.close(figure)
        results["groups"][group] = result
    results["status"] = "complete"
    (output / "manifest.json").write_text(json.dumps(results, indent=2, allow_nan=False) + "\n")
    lines = ["# Chronological forecast combinations", "",
             "Candidates are ranked on the first period; ensemble size is selected on the second, "
             "then evaluated on the later period. The source validation period was previously inspected, "
             "so this later slice is not a globally untouched test.", "",
             "| Market | Models | Later MAE EUR/MWh | Later WAPE % | Relative MAE |",
             "|---|---:|---:|---:|---:|"]
    for group, result in results["groups"].items():
        scores = result["test_combination"]
        wape = "n/a" if scores["wape_percent"] is None else f"{scores['wape_percent']:.2f}"
        relative = "n/a" if scores["relative_mae_to_naive"] is None else f"{scores['relative_mae_to_naive']:.3f}"
        lines.append(f"| {group} | {result['selected_size']} | {scores['mae_eur_per_mwh']:.3f} | {wape} | {relative} |")
    if settings.get("stacking"):
        lines.extend(["", "## Convex weighted combinations", "",
                      "Weights minimize scaled squared error with nonnegative weights summing to one. "
                      "The tuning period is split chronologically to select size and regularization by MAE; "
                      "weights are then refit on the full tuning period before later evaluation.", "",
                      "| Market | Models | Later MAE EUR/MWh | Later WAPE % | Relative MAE |",
                      "|---|---:|---:|---:|---:|"])
        for group, result in results["groups"].items():
            scores = result["stacking"]["test"]
            wape = "n/a" if scores["wape_percent"] is None else f"{scores['wape_percent']:.2f}"
            relative = "n/a" if scores["relative_mae_to_naive"] is None else f"{scores['relative_mae_to_naive']:.3f}"
            lines.append(f"| {group} | {len(result['stacking']['weights'])} | {scores['mae_eur_per_mwh']:.3f} | "
                         f"{wape} | {relative} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    return results


def main():
    """Plan by default; an explicit flag executes only the configured evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/price_ensemble_evaluation.yaml")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    print(json.dumps(run(config) if args.run else config["price_ensemble_evaluation"], indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
