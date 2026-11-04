"""Search native-resolution price models on earlier chronological validation data."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.energy_prices import PriceForecastSettings, load_price_history, prepare_exogenous_features
from bess_pv_trading.forecasting.price_search import complete_shared_days, point_metrics, prepare_search_matrix
from bess_pv_trading.forecasting.price_search_models import make_price_candidate
from bess_pv_trading.parallel import configure_worker_threads


def _sha256(path):
    """Verify the immutable feature table inherited from the prior catalog read."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fit_candidate(job):
    """Fit one estimator and retain its interval-level validation predictions."""
    configure_worker_threads(job["threads"], configure_torch=False)
    data = np.load(job["dataset"], allow_pickle=False)
    spec = job["candidate"]
    estimator = make_price_candidate(spec["family"], spec["parameters"], seed=job["seed"], threads=job["threads"])
    estimator.fit(data["train_x"], data["train_y"])
    prediction = estimator.predict(data["validation_x"])
    metrics = point_metrics(data["validation_y"], prediction, data["reference"])
    output = Path(job["output"])
    pd.DataFrame({"product_start": data["starts"], "actual": data["validation_y"],
                  "point": prediction, "seasonal_naive": data["reference"]}).to_csv(output, index=False)
    return {"group": job["group"], "feature_set": job["feature_set"], "candidate": spec["id"],
            "family": spec["family"], "training_rows": len(data["train_y"]), **metrics}


def run_search(config, *, check_only=False):
    """Prepare source-native matrices, then fit candidate workers on shared days."""
    study = config["price_model_search"]
    root = Path(config.get("_project_root", ROOT))
    source = root / study["source_study"]
    source_manifest = json.loads((source / "manifest.json").read_text())
    feature_path = source / "exogenous.parquet"
    if _sha256(feature_path) != source_manifest["exogenous_sha256"]:
        raise ValueError("Pinned feature table checksum mismatch")
    configure_worker_threads(study["threads_per_worker"], configure_torch=False)
    exogenous = prepare_exogenous_features(pd.read_parquet(feature_path))
    feature_sets = source_manifest["plan"]["feature_sets"]
    destination = root / study["output_root"] / study["run_id"]
    if destination.exists():
        raise FileExistsError(f"Search output already exists: {destination}")
    if not check_only:
        destination.mkdir(parents=True)
    jobs, coverage, lineage = [], [], {}
    for group, split in study["product_groups"].items():
        resolution = split["native_minutes"]
        history, evidence = load_price_history(study["database"], provider="gme", dataset="day_ahead_prices",
            zone="NORD", interval_minutes=resolution, native_resolution_minutes=resolution,
            publication_local_time="12:57", publication_day_offset=-1, timezone=study["timezone"],
            start=split["history_start"], end=split["history_end"])
        lineage[group] = evidence
        datasets = {}
        for feature_set in study["feature_sets"]:
            names = tuple(item["name"] for item in feature_sets[feature_set])
            settings = PriceForecastSettings(lag_hours=tuple(study["lag_hours"]), interval_minutes=resolution,
                timezone=study["timezone"], exogenous_features=names, decision_local_time="11:45",
                decision_day_offset=-1, daily_lag_fallback="target_clock")
            datasets[feature_set] = prepare_search_matrix(history, settings, exogenous,
                cutoff=split["cutoff"], validation_start=split["validation_start"], validation_end=split["validation_end"])
        days = complete_shared_days(list(datasets.values()), start=split["validation_start"],
                                    end=split["validation_end"], settings=settings)
        if not days:
            raise ValueError(f"No shared complete validation days for {group}")
        for feature_set, data in datasets.items():
            selected = data["validation"] & np.isin(data["days"], days)
            training_rows = int(data["training"].sum())
            if training_rows < 100:
                raise ValueError(f"Insufficient native training rows for {group}/{feature_set}")
            coverage.append({"group": group, "feature_set": feature_set, "native_minutes": resolution,
                             "training_rows": training_rows, "shared_days": len(days),
                             "validation_rows": int(selected.sum()), "fallback_count": data["fallback_count"]})
            if check_only:
                continue
            dataset = destination / f"{group}__{feature_set}.npz"
            np.savez(dataset, train_x=data["matrix"][data["training"]], train_y=data["target"][data["training"]],
                     validation_x=data["matrix"][selected], validation_y=data["target"][selected],
                     reference=data["matrix"][selected, 0],
                     starts=np.asarray(data["starts"][selected].astype(str), dtype=str))
            for candidate in study["candidates"]:
                jobs.append({"dataset": str(dataset), "output": str(destination / f"{group}__{feature_set}__{candidate['id']}.csv"),
                    "candidate": candidate, "group": group, "feature_set": feature_set,
                    "seed": study["seed"], "threads": study["threads_per_worker"]})
    if check_only:
        return {"status": "preflight_complete", "coverage": coverage, "feature_checksum": _sha256(feature_path),
                "candidate_count": len(study["candidates"])}
    manifest = {"status": "running", "config": study, "coverage": coverage, "source_lineage": lineage,
                "feature_sha256": _sha256(feature_path), "results": [], "failures": []}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    with ProcessPoolExecutor(max_workers=study["workers"], mp_context=multiprocessing.get_context("spawn")) as executor:
        pending = {executor.submit(_fit_candidate, job): job for job in jobs}
        for future in as_completed(pending):
            job = pending[future]
            try:
                manifest["results"].append(future.result())
            except Exception as error:
                manifest["failures"].append({"group": job["group"], "feature_set": job["feature_set"],
                    "candidate": job["candidate"]["id"], "error": f"{type(error).__name__}: {error}"})
            (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    manifest["status"] = "complete" if not manifest["failures"] else "partial"
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str) + "\n")
    pd.DataFrame(manifest["results"]).to_csv(destination / "validation_scores.csv", index=False)
    return manifest


def main():
    """A no-data plan is the default; run or check-only require an explicit flag."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/research/price_model_search.yaml")
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--check-only", action="store_true")
    actions.add_argument("--run", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    result = run_search(config, check_only=args.check_only) if args.run or args.check_only else config["price_model_search"]
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
