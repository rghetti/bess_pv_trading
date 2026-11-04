"""Profile frozen-selection price forecast errors by month and local hour."""

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


def profile_rows(frame, *, timezone, grouping):
    """Compute pooled errors and signed bias for each month or local hour."""
    starts = pd.DatetimeIndex(pd.to_datetime(frame.product_start, utc=True))
    if starts.has_duplicates or not starts.is_monotonic_increasing:
        raise ValueError("Error profiles require unique increasing delivery products")
    values = frame.copy()
    local = starts.tz_convert(timezone)
    if grouping == "month":
        values["period"] = local.strftime("%Y-%m")
    elif grouping == "hour":
        values["period"] = local.hour
    else:
        raise ValueError("Unknown price-error profile grouping")
    rows = []
    for period, sample in values.groupby("period", sort=True):
        scores = point_metrics(sample.actual, sample.point, sample.seasonal_naive)
        rows.append({"period": period, **scores,
                     "bias_eur_per_mwh": float((sample.point - sample.actual).mean()),
                     "mean_absolute_actual_eur_per_mwh": float(sample.actual.abs().mean())})
    return rows


def run(config):
    """Follow the first validation-ranked candidate and profile later predictions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    settings = config["selected_price_error_report"]
    root = Path(config.get("_project_root", ROOT))
    source = root / settings["selected_evaluation"]
    evaluation_path = source / "manifest.json"
    evaluation = json.loads(evaluation_path.read_text())
    selection = evaluation.get("selection")
    if evaluation.get("status") != "complete" or evaluation.get("failures") or not selection:
        raise ValueError("Error profiles require a complete selected-model evaluation")
    selection_path = Path(selection["source_manifest"])
    if _sha256(selection_path) != selection["source_manifest_sha256"]:
        raise ValueError("Selection manifest checksum differs from evaluation provenance")
    validation = json.loads(selection_path.read_text())
    if validation.get("status") != "complete" or validation.get("failures"):
        raise ValueError("Selection search must be complete and failure-free")
    rows, selected = [], {}
    for group in evaluation["config"]["product_groups"]:
        candidates = [row for row in validation["results"] if row["group"] == group
                      and not row.get("convergence_warnings")]
        if not candidates:
            raise ValueError(f"No rankable selection candidates for {group}")
        best = min(candidates, key=lambda row: (row["mae_eur_per_mwh"], row["feature_set"], row["candidate"]))
        matches = [row for row in evaluation["results"] if row["group"] == group
                   and row["feature_set"] == best["feature_set"] and row["candidate"] == best["candidate"]]
        if len(matches) != 1:
            raise ValueError(f"Selected evaluation is missing or duplicates {group}'s first-ranked model")
        path = source / f"{group}__{best['feature_set']}__{best['candidate']}.csv"
        frame = pd.read_csv(path)
        actual_scores = point_metrics(frame.actual, frame.point, frame.seasonal_naive)
        if not np.isclose(actual_scores["mae_eur_per_mwh"], matches[0]["mae_eur_per_mwh"], rtol=1e-9, atol=1e-9):
            raise ValueError("Selected prediction file does not match its saved score")
        selected[group] = {"feature_set": best["feature_set"], "candidate": best["candidate"],
                           "evaluation_score": actual_scores}
        for grouping in ("month", "hour"):
            rows.extend({"group": group, "grouping": grouping, **row}
                        for row in profile_rows(frame, timezone=settings["timezone"], grouping=grouping))
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / "error_profiles.csv", index=False)
    lines = ["# Selected price-model error profiles", "",
             "The first candidate ranked on the earlier validation period is followed into the later "
             "evaluation. Monthly and hour-of-day summaries are descriptive; historical publication "
             "times are assumed and these periods may have been inspected in related studies.", ""]
    for group, record in selected.items():
        scores = record["evaluation_score"]
        overall_wape = "n/a" if scores["wape_percent"] is None else f"{scores['wape_percent']:.2f}%"
        lines.extend([f"## {group}", "", f"Selected: `{record['feature_set']}/{record['candidate']}`. "
                      f"Later MAE {scores['mae_eur_per_mwh']:.3f} €/MWh; WAPE {overall_wape}.", "",
                      "| Local month | Products | MAE €/MWh | WAPE % | Bias €/MWh |",
                      "|---|---:|---:|---:|---:|"])
        months = table.loc[(table.group == group) & (table.grouping == "month")]
        for row in months.itertuples(index=False):
            wape = "n/a" if pd.isna(row.wape_percent) else f"{row.wape_percent:.2f}"
            lines.append(f"| {row.period} | {row.observations} | {row.mae_eur_per_mwh:.3f} | {wape} | {row.bias_eur_per_mwh:.3f} |")
        hours = table.loc[(table.group == group) & (table.grouping == "hour")]
        figure, axis = plt.subplots(figsize=(9, 4))
        axis.plot(hours.period.astype(int), hours.mae_eur_per_mwh, marker="o", label="MAE")
        axis.plot(hours.period.astype(int), hours.bias_eur_per_mwh, marker=".", label="Forecast bias")
        axis.axhline(0, color="black", linewidth=0.7)
        axis.set(xlabel=f"Delivery hour ({settings['timezone']})", ylabel="EUR/MWh", title=group)
        axis.legend()
        figure.tight_layout()
        figure.savefig(output / f"{group}_hourly_error.png", dpi=130)
        plt.close(figure)
        lines.append("")
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "manifest.json").write_text(json.dumps({"status": "complete", "settings": settings,
        "evaluation_manifest_sha256": _sha256(evaluation_path), "selection_manifest_sha256": _sha256(selection_path),
        "selected": selected}, indent=2, allow_nan=False) + "\n")
    return output


def main():
    """Write a configured profile of an existing selected evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
