"""Plot matched forecast and reference errors from complete price replays."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config


def sha256(path):
    """Retain the exact source bytes behind each plotted comparison."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(settings, root):
    """Join each specified model/reference pair and plot its common products."""
    rows, sources = [], []
    for study in settings["studies"]:
        folder = root / study["run"]
        manifest_path = folder / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("status") != "complete" or manifest.get("failures"):
            raise ValueError("Every plotted replay must complete without failures")
        for group in study["groups"]:
            # The two saved series must describe the same ordered targets.
            reference_path = folder / group["reference_file"]
            model_path = folder / group["model_file"]
            reference = pd.read_csv(reference_path, parse_dates=["product_start"])
            model = pd.read_csv(model_path, parse_dates=["product_start"])
            if (reference.product_start.duplicated().any() or not reference.product_start.is_monotonic_increasing
                    or not reference.product_start.equals(model.product_start)
                    or not np.allclose(reference.actual, model.actual, rtol=0, atol=1e-10)):
                raise ValueError("Model and baseline must share ordered unique products and targets")
            actual = reference.actual.to_numpy(float)
            if not np.isfinite(actual).all() or not np.isfinite(reference.point).all() or not np.isfinite(model.point).all():
                raise ValueError("Plotted observations and forecasts must be finite")
            denominator = float(np.abs(actual).sum())
            if not denominator:
                raise ValueError("WAPE is undefined when absolute target sum is zero")
            days = pd.DatetimeIndex(reference.product_start).tz_convert(settings["timezone"]).strftime("%Y-%m-%d")
            for label, values in (("Latest-cleared price", reference.point), ("Selected weather model", model.point)):
                errors = np.abs(actual - values.to_numpy(float))
                rows.append({"period": study["label"], "market": group["market"], "model": label,
                    "days": len(np.unique(days)), "products": len(actual), "mae_eur_per_mwh": float(errors.mean()),
                    "wape_percent": float(100 * errors.sum() / denominator)})
            sources.append({"period": study["label"], "market": group["market"],
                "manifest_sha256": sha256(manifest_path), "reference_sha256": sha256(reference_path),
                "model_sha256": sha256(model_path)})
    frame = pd.DataFrame(rows)
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    frame.to_csv(output / "matched_scores.csv", index=False)
    (output / "manifest.json").write_text(json.dumps({"status": "complete", "settings": settings,
        "sources": sources}, indent=2, allow_nan=False) + "\n")
    periods = list(dict.fromkeys(frame.period))
    markets = list(settings["markets"])
    figure, axes = plt.subplots(1, len(periods), figsize=(6 * len(periods), 4), squeeze=False,
                                sharey=True, constrained_layout=True)
    for axis, period in zip(axes[0], periods):
        for index, label in enumerate(("Latest-cleared price", "Selected weather model")):
            values = [frame.loc[(frame.period == period) & (frame.market == market) & (frame.model == label),
                                "wape_percent"].item() for market in markets]
            axis.bar(np.arange(len(markets)) + (index - 0.5) * 0.36, values, width=0.36, label=label)
        axis.set_xticks(range(len(markets)), markets)
        axis.set(title=period, ylabel="WAPE (%)")
        axis.grid(axis="y", alpha=0.2)
    axes[0, 0].legend()
    figure.suptitle("IDA errors on matched products; periods have different coverage")
    figure.savefig(output / "matched_wape.png", dpi=160)
    plt.close(figure)
    return output


def main():
    """Resolve the explicit study configuration and output root."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    args = parser.parse_args()
    print(run(load_config(args.config)["matched_price_baselines"], args.project_root.resolve()))


if __name__ == "__main__":
    main()
