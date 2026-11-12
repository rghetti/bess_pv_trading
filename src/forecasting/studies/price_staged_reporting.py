"""Write readable per-auction tables and figures for staged price searches."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import re


def write_stage_reports(output, stage, rows, groups):
    """Export each completed stage with its exact features and model settings."""
    parent_features = {}
    combinations_path = Path(output) / "stages" / "combinations.json"
    if stage == "ablation" and combinations_path.exists():
        for prior in json.loads(combinations_path.read_text())["results"]:
            metrics = prior.get("metrics", {})
            if prior.get("status") != "complete" or metrics.get("wape_percent") is None:
                continue
            key = (prior["arm"]["group"], prior["candidate"]["family"])
            rank = (metrics["wape_percent"], metrics["mae_eur_per_mwh"],
                    prior["candidate"]["family"], prior["candidate"]["trial"],
                    tuple(prior["arm"]["columns"]))
            if key not in parent_features or rank < parent_features[key][0]:
                parent_features[key] = (rank, prior["arm"]["columns"])
    grouped = {}
    for row in rows:
        grouped.setdefault(row["arm"]["group"], []).append(row)
    for group_name, records in sorted(grouped.items()):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", group_name):
            raise ValueError(f"Unsafe product group name: {group_name}")
        directory = Path(output) / "reports" / group_name
        directory.mkdir(parents=True, exist_ok=True)
        group = groups[group_name]
        fields = ["status", "family", "trial", "seed", "blocks", "features", "removed_features",
                  "parameters", "availability_class", "train_rows", "validation_rows", "mae_eur_per_mwh",
                  "wape_percent", "rmse_eur_per_mwh", "relative_mae_to_reference", "training_objective", "error"]
        path = directory / f"{stage}.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for row in records:
                arm = row["arm"]
                candidate = row["candidate"]
                metrics = row.get("metrics", {})
                full_columns = {column for block in arm["blocks"]
                                for column in group["block_columns"][block]}
                parent = parent_features.get((group_name, candidate["family"]))
                if parent is not None:
                    full_columns = set(parent[1])
                removed = sorted(full_columns - set(arm["columns"]))
                writer.writerow({
                    "status": row["status"], "family": candidate["family"],
                    "trial": candidate["trial"], "seed": candidate["seed"],
                    "blocks": json.dumps(arm["blocks"]),
                    "features": json.dumps(row.get("feature_names", [])),
                    "removed_features": json.dumps([group["feature_names"][column] for column in removed]),
                    "parameters": json.dumps(candidate["parameters"], sort_keys=True),
                    "availability_class": arm["availability_class"],
                    "train_rows": row.get("train_rows"), "validation_rows": row.get("validation_rows"),
                    **{name: metrics.get(name) for name in ("mae_eur_per_mwh", "wape_percent",
                        "rmse_eur_per_mwh", "relative_mae_to_reference")},
                    "training_objective": row.get("training_objective"), "error": row.get("error"),
                })
        if stage in {"screen", "combinations", "refinement", "retrospective"}:
            _plot_stage(directory / f"{stage}.png", group_name, stage, records)


def _plot_stage(path, group_name, stage, rows):
    """Plot the best inner MAE for each comparable feature group or family."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    best = {}
    for row in rows:
        if row.get("status") != "complete" or row.get("metrics", {}).get("mae_eur_per_mwh") is None:
            continue
        if stage == "refinement":
            label = row["candidate"]["family"]
        elif stage == "retrospective":
            label = "Same-day quantities" if row["arm"]["availability_class"] == "retrospective_realized" else "Matched baseline"
        else:
            label = "+".join(row["arm"]["blocks"]) or "Core"
        value = row["metrics"]["mae_eur_per_mwh"]
        best[label] = min(value, best.get(label, float("inf")))
    if not best:
        return
    selected = sorted(best.items(), key=lambda item: (item[1], item[0]))[:10]
    labels, values = zip(*reversed(selected))
    figure, axis = plt.subplots(figsize=(10, max(3.0, len(labels) * 0.42 + 1.5)))
    axis.barh(labels, values)
    axis.set_xlabel("Inner-validation MAE (EUR/MWh)")
    axis.set_title(f"{group_name}: {stage} (best candidate per group)")
    figure.tight_layout()
    figure.savefig(path, dpi=130)
    plt.close(figure)
