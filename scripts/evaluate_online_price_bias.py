"""Evaluate chronological residual-bias corrections on saved price forecasts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.price_search import point_metrics
from scripts.report_price_search_monthly import _sha256


def online_bias(actual, prediction, starts_ns, ends_ns, available_ns, decisions_ns, local_days, local_hours,
                *, window_days, block_hours, shrinkage):
    """Estimate blockwise prior residuals using only labels eligible at each decision."""
    actual, prediction = np.asarray(actual, dtype=float), np.asarray(prediction, dtype=float)
    starts_ns, ends_ns = np.asarray(starts_ns, dtype=np.int64), np.asarray(ends_ns, dtype=np.int64)
    available_ns, decisions_ns = np.asarray(available_ns, dtype=np.int64), np.asarray(decisions_ns, dtype=np.int64)
    local_days, local_hours = np.asarray(local_days, dtype=str), np.asarray(local_hours, dtype=int)
    arrays = (prediction, starts_ns, ends_ns, available_ns, decisions_ns, local_days, local_hours)
    if not len(actual) or any(len(value) != len(actual) for value in arrays):
        raise ValueError("Online bias inputs must be nonempty and aligned")
    if not np.isfinite(actual).all() or not np.isfinite(prediction).all() or (np.diff(starts_ns) <= 0).any():
        raise ValueError("Online bias requires finite prices and ordered unique products")
    if (isinstance(window_days, bool) or not isinstance(window_days, int) or window_days < 1
            or isinstance(block_hours, bool) or not isinstance(block_hours, int) or block_hours < 1
            or 24 % block_hours or isinstance(shrinkage, bool) or not np.isfinite(shrinkage) or shrinkage < 0
            or np.any((local_hours < 0) | (local_hours >= 24))):
        raise ValueError("Invalid online bias window, time block, shrinkage or local hour")
    residual = actual - prediction
    correction = np.zeros(len(actual), dtype=float)
    blocks = local_hours // block_hours
    window_ns = pd.Timedelta(days=window_days).value
    for day in np.unique(local_days):
        target = local_days == day
        # Use the earliest decision on that delivery day as a conservative
        # shared cutoff; never use unpublished or unfinished earlier products.
        cutoff = decisions_ns[target].min()
        eligible = ((local_days < day) & (available_ns <= cutoff) & (ends_ns <= cutoff)
                    & (starts_ns >= cutoff - window_ns))
        for block in np.unique(blocks[target]):
            history = eligible & (blocks == block)
            count = int(history.sum())
            if count:
                correction[target & (blocks == block)] = residual[history].sum() / (count + shrinkage)
    return correction


def evaluate_bias(actual, prediction, reference, starts_ns, ends_ns, available_ns, decisions_ns, days, hours, settings):
    """Select correction parameters on early days and score a later period."""
    days = np.asarray(days, dtype=str)
    if not (settings["tune_start"] < settings["tune_end_exclusive"] <= settings["test_start"]
            < settings["test_end_exclusive"]):
        raise ValueError("Bias tuning and evaluation periods must be chronological")
    tune = (days >= settings["tune_start"]) & (days < settings["tune_end_exclusive"])
    test = (days >= settings["test_start"]) & (days < settings["test_end_exclusive"])
    if not tune.any() or not test.any():
        raise ValueError("Bias tuning and evaluation both need observed products")
    actual, prediction, reference = map(lambda value: np.asarray(value, dtype=float), (actual, prediction, reference))
    candidates = [{"window_days": None, "block_hours": None, "shrinkage": None,
                   "tune": point_metrics(actual[tune], prediction[tune], reference[tune])}]
    corrections = [np.zeros(len(actual))]
    for window in settings["window_days"]:
        for block in settings["block_hours"]:
            for shrinkage in settings["shrinkage"]:
                correction = online_bias(actual, prediction, starts_ns, ends_ns, available_ns, decisions_ns, days, hours,
                                         window_days=window, block_hours=block, shrinkage=shrinkage)
                corrections.append(correction)
                candidates.append({"window_days": window, "block_hours": block, "shrinkage": shrinkage,
                                   "tune": point_metrics(actual[tune], prediction[tune] + correction[tune], reference[tune])})
    # Preserve the uncorrected forecast on exact ties.
    chosen = min(range(len(candidates)), key=lambda index: (candidates[index]["tune"]["mae_eur_per_mwh"], index))
    corrected = prediction[test] + corrections[chosen][test]
    return {"selected": candidates[chosen], "candidates": candidates,
            "test_corrected": point_metrics(actual[test], corrected, reference[test]),
            "test_uncorrected": point_metrics(actual[test], prediction[test], reference[test])}, test, corrected


def run(config):
    """Follow the earlier-selected candidate and evaluate its online correction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = config["online_price_bias"]
    root = Path(config.get("_project_root", ROOT))
    source = root / settings["selected_evaluation"]
    evaluation_path = source / "manifest.json"
    evaluation = json.loads(evaluation_path.read_text())
    selection = evaluation.get("selection")
    if evaluation.get("status") != "complete" or evaluation.get("failures") or not selection:
        raise ValueError("Online bias requires a complete selected evaluation")
    source_path = Path(selection["source_manifest"])
    if _sha256(source_path) != selection["source_manifest_sha256"]:
        raise ValueError("Original validation selection manifest changed")
    validation = json.loads(source_path.read_text())
    if validation.get("status") != "complete" or validation.get("failures"):
        raise ValueError("Original validation selection must be complete")
    results, outputs = {}, {}
    for group, split in evaluation["config"]["product_groups"].items():
        if (settings["tune_start"] < split["validation_start"]
                or settings["test_end_exclusive"] > split["validation_end"]):
            raise ValueError("Bias evaluation extends beyond saved prediction period")
        candidates = [row for row in validation["results"] if row["group"] == group
                      and not row.get("convergence_warnings")]
        if not candidates:
            raise ValueError(f"No rankable validation candidate for {group}")
        best = min(candidates, key=lambda row: (row["mae_eur_per_mwh"], row["feature_set"], row["candidate"]))
        matches = [row for row in evaluation["results"] if row["group"] == group
                   and row["feature_set"] == best["feature_set"] and row["candidate"] == best["candidate"]]
        if len(matches) != 1:
            raise ValueError("Selected candidate is absent or duplicated in the later evaluation")
        prediction_path = source / f"{group}__{best['feature_set']}__{best['candidate']}.csv"
        frame = pd.read_csv(prediction_path)
        dataset_path = source / f"{group}__{best['feature_set']}.npz"
        with np.load(dataset_path, allow_pickle=False) as data:
            starts = pd.DatetimeIndex(pd.to_datetime(data["starts"], utc=True))
            observed_starts = pd.DatetimeIndex(pd.to_datetime(frame.product_start, utc=True))
            if not starts.equals(observed_starts) or not np.allclose(frame.actual, data["validation_y"], rtol=0, atol=1e-10):
                raise ValueError("Saved predictions do not match the evaluation dataset")
            local = starts.tz_convert(settings["timezone"])
            result, test, corrected = evaluate_bias(frame.actual, frame.point, frame.seasonal_naive, starts.as_unit("ns").asi8,
                data["product_end_ns"], data["available_ns"], data["decisions_ns"],
                local.strftime("%Y-%m-%d"), local.hour, settings)
        results[group] = {"feature_set": best["feature_set"], "candidate": best["candidate"], **result}
        outputs[group] = pd.DataFrame({"product_start": starts[test], "actual": frame.actual.to_numpy()[test],
                                       "uncorrected": frame.point.to_numpy()[test], "corrected": corrected})
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    lines = ["# Online price-bias correction", "",
             "The underlying model was selected earlier. Correction settings are selected on the first "
             "period and scored on the later one. Only prior published and completed products are used; "
             "publication clocks are assumptions. Both periods were inspected in related studies.", "",
             "| Market | Correction window | Block hours | Shrinkage | Later MAE before | Later MAE after | Later WAPE before | Later WAPE after |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for group, result in results.items():
        outputs[group].to_csv(output / f"{group}_predictions.csv", index=False)
        # Plot daily errors on identical products to expose when adaptation helps.
        sample = outputs[group].copy()
        sample["day"] = pd.DatetimeIndex(sample.product_start).tz_convert(settings["timezone"]).strftime("%Y-%m-%d")
        sample["before_error"] = (sample.uncorrected - sample.actual).abs()
        sample["after_error"] = (sample.corrected - sample.actual).abs()
        daily = sample.groupby("day")[["before_error", "after_error"]].mean()
        figure, axis = plt.subplots(figsize=(10, 4))
        axis.plot(pd.to_datetime(daily.index), daily.before_error, label="Original")
        axis.plot(pd.to_datetime(daily.index), daily.after_error, label="Bias corrected")
        axis.set(ylabel="Daily MAE (EUR/MWh)", title=group)
        axis.legend()
        figure.autofmt_xdate()
        figure.tight_layout()
        figure.savefig(output / f"{group}_daily_mae.png", dpi=130)
        plt.close(figure)
        choice, before, after = result["selected"], result["test_uncorrected"], result["test_corrected"]
        before_wape = "n/a" if before["wape_percent"] is None else f"{before['wape_percent']:.2f}%"
        after_wape = "n/a" if after["wape_percent"] is None else f"{after['wape_percent']:.2f}%"
        lines.append(f"| {group} | {choice['window_days']} | {choice['block_hours']} | {choice['shrinkage']} | "
                     f"{before['mae_eur_per_mwh']:.3f} | {after['mae_eur_per_mwh']:.3f} | "
                     f"{before_wape} | {after_wape} |")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "manifest.json").write_text(json.dumps({"status": "complete", "settings": settings,
        "source_manifest_sha256": _sha256(evaluation_path), "results": results}, indent=2, allow_nan=False) + "\n")
    return output


def main():
    """Evaluate configured correction choices on existing forecast outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
