"""Compare two saved price forecasts and an earlier-tuned convex blend."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config


def checksum(path):
    """Fingerprint a source file used by the comparison."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(settings, root):
    """Select a blend weight on earlier local dates and score later shared rows."""
    if not (settings["tune_start"] < settings["tune_end_exclusive"] <= settings["test_start"]
            < settings["test_end_exclusive"]):
        raise ValueError("Tuning and test dates must be chronological and nonoverlapping")
    weights = np.asarray(settings["first_model_weights"], dtype=float)
    if not len(weights) or not np.isfinite(weights).all() or np.any((weights < 0) | (weights > 1)):
        raise ValueError("Blend weights must be finite and between zero and one")
    frames, sources = [], []
    for model in settings["models"]:
        # Follow only complete source runs and record the exact inputs.
        folder = root / model["run"]
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete" or manifest.get("failures"):
            raise ValueError("Each source replay must be complete without failures")
        path = folder / model["prediction_file"]
        frame = pd.read_csv(path, parse_dates=["product_start"])
        if frame.product_start.duplicated().any() or not frame.product_start.is_monotonic_increasing:
            raise ValueError("Source products must be unique and chronological")
        frames.append(frame[["product_start", "actual", "point"]])
        sources.append({"run": model["run"], "prediction_file": model["prediction_file"],
                        "manifest_sha256": checksum(manifest_path), "prediction_sha256": checksum(path)})
    if len(frames) != 2:
        raise ValueError("Exactly two source forecasts are required")
    joined = frames[0].merge(frames[1], on="product_start", suffixes=("_first", "_second"), validate="one_to_one")
    if not len(joined) or not np.allclose(joined.actual_first, joined.actual_second, rtol=0, atol=1e-10):
        raise ValueError("Shared product targets must agree")
    values = joined[["actual_first", "point_first", "point_second"]].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Shared targets and predictions must be finite")
    days = pd.DatetimeIndex(joined.product_start).tz_convert(settings["timezone"]).strftime("%Y-%m-%d")
    tune = (days >= settings["tune_start"]) & (days < settings["tune_end_exclusive"])
    test = (days >= settings["test_start"]) & (days < settings["test_end_exclusive"])
    if not tune.any() or not test.any():
        raise ValueError("Both periods need shared products")
    actual, first, second = values.T
    losses = [float(np.mean(np.abs(actual[tune] - (weight * first[tune] + (1 - weight) * second[tune]))))
              for weight in weights]
    weight = float(weights[int(np.argmin(losses))])
    blended = weight * first + (1 - weight) * second

    def scores(prediction):
        """Use a shared target denominator for the later comparison."""
        error = np.abs(actual[test] - prediction[test])
        denominator = float(np.abs(actual[test]).sum())
        if not denominator:
            raise ValueError("WAPE is undefined for zero absolute target sum")
        return {"mae_eur_per_mwh": float(error.mean()), "wape_percent": float(100 * error.sum() / denominator)}

    result = {"status": "complete", "settings": settings, "sources": sources,
              "shared_products": len(joined), "tuning_products": int(tune.sum()),
              "test_products": int(test.sum()), "first_model_weight": weight,
              "tuning_mae": losses, "first": scores(first), "second": scores(second), "blend": scores(blended)}
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame({"product_start": joined.product_start[test], "actual": actual[test], "first": first[test],
                  "second": second[test], "blend": blended[test]}).to_csv(output / "test_predictions.csv", index=False)
    (output / "manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return output


def main():
    """Resolve explicit study and output roots."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    print(run(load_config(args.config)["two_price_forecasts"], args.project_root.resolve()))


if __name__ == "__main__":
    main()
