"""Preview saved price-search candidates on their development-validation periods.

This independent diagnostic never writes to the running search or reads its test
indices. The default action only prints the candidate plan; --run performs the
explicitly requested refits and writes a separate, timestamped output directory.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

# Cap native libraries before importing the model and plotting dependencies.
for variable in (
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "BLIS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
):
    os.environ[variable] = "1"
os.environ["OMP_DYNAMIC"] = "FALSE"

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.price_feature_search import _fit, _score_key
from bess_pv_trading.forecasting.studies.price_feature_models import point_metrics


def saved_candidates(root: Path, cohort: str, availability: str) -> tuple[dict, list[dict]]:
    """Freeze the best completed inner-tail candidate per group and family."""
    prepared = json.loads((root / "prepared.json").read_text())
    best = {}
    # Search checkpoints are atomically replaced, so each read is a consistent snapshot.
    for task in prepared["tasks"]:
        arm = task["arm"]
        if arm["cohort"] != cohort or arm["availability_class"] != availability:
            continue
        path = root / "trials" / f"{task['id']}.json"
        if not path.exists():
            continue
        checkpoint = json.loads(path.read_text())
        if checkpoint["signature"] != task["signature"]:
            raise ValueError(f"Incompatible checkpoint: {path}")
        for candidate in checkpoint["results"]:
            if candidate["status"] != "complete":
                continue
            key = (arm["group"], candidate["family"])
            if key not in best or _score_key(candidate) < _score_key(best[key]["candidate"]):
                best[key] = {"group": arm["group"], "arm": arm, "candidate": candidate}
    return prepared, [best[key] for key in sorted(best)]


def validate_candidate(job: dict, root: str, group: dict, output: str) -> dict:
    """Refit one frozen candidate on training rows and predict validation rows."""
    source = Path(root)
    group_files = group["files"]
    cohort_files = group["cohorts"][job["arm"]["cohort"]]["indices"]
    arrays = {name: np.load(source / item["path"], mmap_mode="r", allow_pickle=False)
              for name, item in group_files.items() if name in {"x", "y", "anchor", "reference", "starts"}}
    train = np.load(source / cohort_files["train"]["path"], allow_pickle=False)
    validation = np.load(source / cohort_files["validation"]["path"], allow_pickle=False)
    columns = job["arm"]["columns"]
    # Use the identical model, hyperparameters and target transformation as the main search.
    _, prediction, warnings = _fit(job["candidate"]["family"], job["candidate"],
        np.asarray(arrays["x"][np.ix_(train, columns)]), arrays["y"][train] - arrays["anchor"][train],
        np.asarray(arrays["x"][np.ix_(validation, columns)]))
    prediction = prediction + arrays["anchor"][validation]
    frame = pd.DataFrame({"product_start": pd.to_datetime(arrays["starts"][validation], utc=True),
        "actual": arrays["y"][validation], "prediction": prediction,
        "reference": arrays["reference"][validation]})
    prediction_path = Path(output) / "predictions" / f"{job['group']}__{job['candidate']['family']}.parquet"
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(prediction_path, index=False)
    return {**job, "metrics": point_metrics(frame.actual, frame.prediction, frame.reference),
        "training_rows": len(train), "prediction_path": str(prediction_path), "warnings": warnings}


def plot_winner(result: dict, output: Path, local_timezone: str) -> None:
    """Plot best and worst daily errors and monthly MAE by local delivery time."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    frame = pd.read_parquet(result["prediction_path"])
    local = pd.DatetimeIndex(frame.product_start).tz_convert(local_timezone)
    frame["local_time"] = local
    frame["day"] = local.strftime("%Y-%m-%d")
    frame["month"] = local.strftime("%Y-%m")
    frame["delivery_minute"] = local.hour * 60 + local.minute
    frame["absolute_error"] = (frame.prediction - frame.actual).abs()
    daily = frame.groupby("day").agg(mae=("absolute_error", "mean"), products=("actual", "size"))
    daily = daily.sort_values(["mae"], kind="stable")
    if len(daily) < 6:
        raise ValueError(f"Insufficient validation days for {result['group']}")
    chosen = list(daily.index[:3]) + list(daily.index[-3:][::-1])
    folder = output / "plots" / result["group"]
    folder.mkdir(parents=True, exist_ok=True)
    # Each selected day displays observed and predicted prices and the number of scored products.
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), constrained_layout=True)
    for axis, day, rank in zip(axes.flat, chosen, ["best 1", "best 2", "best 3", "worst 1", "worst 2", "worst 3"]):
        day_frame = frame.loc[frame.day == day].sort_values("local_time")
        axis.plot(day_frame.local_time, day_frame.actual, label="Realized", linewidth=1.4)
        axis.plot(day_frame.local_time, day_frame.prediction, label="Forecast", linewidth=1.2)
        axis.set_title(f"{rank}: {day}; MAE {daily.loc[day, 'mae']:.2f} EUR/MWh; n={len(day_frame)}")
        axis.tick_params(axis="x", labelrotation=20)
        axis.set_ylabel("EUR/MWh")
        axis.grid(alpha=0.2)
    axes.flat[0].legend()
    fig.suptitle(f"{result['group']}: validation-selected {result['candidate']['family']}")
    fig.savefig(folder / "best_and_worst_days.png", dpi=160)
    plt.close(fig)
    # Average absolute errors at each local delivery time separately by month.
    grouped = frame.groupby(["month", "delivery_minute"]).absolute_error.agg(["mean", "size"]).reset_index()
    fig, axis = plt.subplots(figsize=(12, 5), constrained_layout=True)
    for month, values in grouped.groupby("month"):
        axis.plot(values.delivery_minute / 60, values["mean"], label=month)
    axis.set(xlabel=f"Local delivery time ({local_timezone})", ylabel="Mean absolute error (EUR/MWh)",
             title=f"{result['group']}: validation monthly error by delivery time")
    axis.grid(alpha=0.2)
    axis.legend(title="Month")
    fig.savefig(folder / "monthly_mae_by_delivery_time.png", dpi=160)
    plt.close(fig)
    daily.to_csv(folder / "daily_mae.csv")
    grouped.to_csv(folder / "monthly_mae_by_delivery_time.csv", index=False)


def main() -> None:
    """Plan by default; run only an explicitly requested isolated validation preview."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--search", default="outputs/price_feature_search_full")
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--cohort", default="full_history")
    parser.add_argument("--availability", default="assumed_previous_day")
    parser.add_argument("--timezone", default="Europe/Rome")
    parser.add_argument("--run", action="store_true")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    root = Path(args.search).resolve()
    prepared, jobs = saved_candidates(root, args.cohort, args.availability)
    groups = sorted({job["group"] for job in jobs})
    print(json.dumps({"candidate_refits": len(jobs), "groups": groups,
        "missing_groups": sorted(set(prepared["groups"]) - set(groups)), "workers": args.workers,
        "selection": "best saved inner-tail candidate per family, then best family on validation",
        "cohort": args.cohort, "availability": args.availability, "run": args.run}, indent=2), flush=True)
    if not args.run:
        return
    output = Path(args.output).resolve()
    # A new directory prevents overwriting another preview or the live search.
    output.mkdir(parents=True, exist_ok=False)
    (output / "plan.json").write_text(json.dumps({"created_at": datetime.now(timezone.utc).isoformat(),
        "scientific_config_sha256": prepared["scientific_config_sha256"], "jobs": jobs}, indent=2) + "\n")
    results = []
    failures = []
    remaining = Counter(job["group"] for job in jobs)
    winners = {}
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        futures = {pool.submit(validate_candidate, job, str(root), prepared["groups"][job["group"]], str(output)): job
                   for job in jobs}
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
            # Publish each group as soon as its selected families have finished.
            if remaining[group] == 0 and group in winners:
                plot_winner(winners[group], output, args.timezone)
                (output / "validation_winners.json").write_text(json.dumps(winners, indent=2) + "\n")
            print(f"Validated {len(results)}/{len(jobs)} candidates; failures={len(failures)}", flush=True)
    (output / "validation_results.json").write_text(json.dumps(results, indent=2) + "\n")
    (output / "validation_winners.json").write_text(json.dumps(winners, indent=2) + "\n")


if __name__ == "__main__":
    main()
