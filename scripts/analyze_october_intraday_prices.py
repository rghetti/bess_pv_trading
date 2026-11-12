"""Describe saved out-of-sample October intraday price predictions."""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def summarize(frame, group):
    """Compare each model and its price reference on identical observations."""
    def metrics(rows):
        error = rows.forecast_eur_per_mwh - rows.actual_eur_per_mwh
        reference_error = rows.reference_eur_per_mwh - rows.actual_eur_per_mwh
        denominator = rows.actual_eur_per_mwh.abs().sum()
        return pd.Series({"rows": len(rows), "days": rows.local_day.nunique(),
            "mae": error.abs().mean(), "reference_mae": reference_error.abs().mean(),
            "bias": error.mean(), "rmse": np.sqrt((error ** 2).mean()),
            "wape_pct": 100 * error.abs().sum() / denominator if denominator else np.nan,
            "mean_actual": rows.actual_eur_per_mwh.mean()})
    records = []
    for key, rows in frame.groupby(group, observed=True, sort=True):
        records.append({group: key, **metrics(rows).to_dict()})
    return pd.DataFrame(records)


def main():
    """Write grouped metrics and figures using already exported predictions."""
    source = Path("outputs/price_staged_proxies_2026")
    output = Path("outputs/price_october_intraday_analysis")
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((source / "manifest.json").read_text())
    if manifest["status"] != "complete":
        raise ValueError("The annual proxy refit has not completed")
    for auction in ("ida2", "ida3"):
        frame = pd.read_parquet(source / f"{auction}_quarter_hourly__main.parquet")
        frame = frame.loc[frame.local_day.str.startswith("2026-10")].copy()
        if frame.empty or frame.used_for_training.any():
            raise ValueError(f"{auction} has no holdout rows or includes training rows")
        local = frame.product_start.dt.tz_convert("Europe/Rome")
        frame["hour"] = local.dt.hour
        frame["weekday"] = local.dt.day_name()
        # Price movement and actual price bins use the realized outcome and are diagnostic.
        frame["reference_move"] = frame.actual_eur_per_mwh - frame.reference_eur_per_mwh
        frame["movement_bin"] = pd.cut(frame.reference_move, [-np.inf, -50, -20, -5, 5, 20, 50, np.inf],
            labels=["below -50", "-50 to -20", "-20 to -5", "-5 to 5", "5 to 20", "20 to 50", "above 50"])
        frame["price_bin"] = pd.qcut(frame.actual_eur_per_mwh, 4, labels=["lowest", "lower middle", "upper middle", "highest"], duplicates="drop")
        tables = {name: summarize(frame, field) for name, field in
                  (("daily", "local_day"), ("hourly", "hour"), ("weekday", "weekday"),
                   ("movement", "movement_bin"), ("actual_price_quartile", "price_bin"))}
        for name, table in tables.items():
            table.to_csv(output / f"{auction}_{name}.csv", index=False)
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        hourly = tables["hourly"]
        axes[0].plot(hourly.hour, hourly.mae, label="Model")
        axes[0].plot(hourly.hour, hourly.reference_mae, label="Previous auction")
        axes[0].set_ylabel("MAE (EUR/MWh)")
        axes[0].legend()
        axes[1].bar(hourly.hour, hourly.bias)
        axes[1].axhline(0, color="black", linewidth=0.7)
        axes[1].set(xlabel="Delivery hour, Europe/Rome", ylabel="Mean error (EUR/MWh)")
        fig.suptitle(f"{auction.upper()}: October 2026 held-out errors")
        fig.tight_layout()
        fig.savefig(output / f"{auction}_hourly.png", dpi=140)
        plt.close(fig)
        print(auction, "rows", len(frame), "days", frame.local_day.nunique())


if __name__ == "__main__":
    main()
