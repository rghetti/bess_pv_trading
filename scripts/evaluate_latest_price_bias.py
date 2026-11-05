"""Evaluate chronological correction of saved latest-cleared IDA prices."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from scripts.evaluate_online_price_bias import evaluate_bias


def sha256(path):
    """Identify the exact input bytes used in a retrospective evaluation."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(config, root):
    """Tune prior-residual correction and score later latest-cleared baselines."""
    settings = config["online_price_bias"]
    source = root / settings["selected_evaluation"]
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("status") != "complete" or manifest.get("failures"):
        raise ValueError("Source replay must complete without failures")
    results, predictions = {}, {}
    for group, split in manifest["config"]["product_groups"].items():
        # The declared zero-spread model must be a unique completed source fit.
        matches = [item for item in manifest["results"] if item["group"] == group
                   and item["feature_set"] == "spread_price" and item["candidate"] == "zero_spread"]
        if len(matches) != 1 or matches[0].get("convergence_warnings"):
            raise ValueError(f"Expected one valid latest-cleared reference for {group}")
        if (settings["tune_start"] < split["validation_start"]
                or settings["test_end_exclusive"] > split["validation_end"]):
            raise ValueError("Correction periods extend beyond source predictions")
        csv_path = source / f"{group}__spread_price__zero_spread.csv"
        array_path = source / f"{group}__spread_price.npz"
        frame = pd.read_csv(csv_path)
        with np.load(array_path, allow_pickle=False) as data:
            starts = pd.DatetimeIndex(pd.to_datetime(data["starts"], utc=True))
            if (not starts.equals(pd.DatetimeIndex(pd.to_datetime(frame.product_start, utc=True)))
                    or not np.allclose(frame.actual, data["validation_y"], rtol=0, atol=1e-10)):
                raise ValueError("Reference predictions do not match the saved dataset")
            local = starts.tz_convert(settings["timezone"])
            result, test, corrected = evaluate_bias(frame.actual, frame.point, frame.seasonal_naive,
                starts.as_unit("ns").asi8, data["product_end_ns"], data["available_ns"],
                data["decisions_ns"], local.strftime("%Y-%m-%d"), local.hour, settings)
        results[group] = {"reference_csv_sha256": sha256(csv_path), "dataset_sha256": sha256(array_path), **result}
        predictions[group] = pd.DataFrame({"product_start": starts[test], "actual": frame.actual.to_numpy()[test],
            "latest_cleared": frame.point.to_numpy()[test], "corrected": corrected})
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    for group, frame in predictions.items():
        frame.to_csv(output / f"{group}_predictions.csv", index=False)
    evidence = {"status": "complete", "settings": settings, "source_manifest_sha256": sha256(manifest_path),
                "results": results}
    (output / "manifest.json").write_text(json.dumps(evidence, indent=2, allow_nan=False) + "\n")
    return output


def main():
    """Resolve the study configuration and project root supplied by the caller."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    print(run(load_config(args.config), args.project_root.resolve()))


if __name__ == "__main__":
    main()
