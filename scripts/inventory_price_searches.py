"""Inventory saved native-resolution price searches without ranking across studies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd

from bess_pv_trading.config import load_config


def inventory_manifest(manifest, *, run_name, source_hash):
    """Describe each group, distinguishing absent warning records from no warnings."""
    study = manifest.get("config", {})
    if not isinstance(study, dict) or not isinstance(study.get("product_groups"), dict):
        return []
    if not isinstance(manifest.get("results"), list) or not isinstance(manifest.get("coverage"), list):
        return []
    rows = []
    for group, split in study["product_groups"].items():
        records = [row for row in manifest["results"] if row.get("group") == group]
        coverage = [row for row in manifest["coverage"] if row.get("group") == group]
        days = sorted({row["shared_days"] for row in coverage})
        if len(days) > 1:
            raise ValueError(f"Inconsistent shared-day coverage in {run_name}/{group}")
        warned = sum(bool(row.get("convergence_warnings")) for row in records if "convergence_warnings" in row)
        unknown = sum("convergence_warnings" not in row for row in records)
        rows.append({"run": run_name, "group": group, "target": split.get("target", "DA"),
            "native_minutes": split.get("native_minutes"), "validation_start": split.get("validation_start"),
            "validation_end_exclusive": split.get("validation_end"), "training_cutoff": split.get("cutoff"),
            "status": manifest.get("status"), "selected_evaluation": bool(manifest.get("selection")),
            "rolling_refit": bool(study.get("rolling_refit")), "shared_days": days[0] if days else None,
            "completed_fits": len(records), "warning_fits": warned, "warning_status_unknown_fits": unknown,
            "failed_fits": sum(row.get("group") == group for row in manifest.get("failures", [])),
            "source_manifest_sha256": source_hash})
    return rows


def run(config):
    """Read each available manifest once and write a dated, non-ranking inventory."""
    settings = config["price_search_inventory"]
    root = Path(config.get("_project_root", ROOT))
    source = root / settings["search_root"]
    rows, unreadable = [], []
    for path in sorted(source.glob("*/manifest.json")):
        try:
            contents = path.read_bytes()
            manifest = json.loads(contents)
        except (OSError, ValueError):
            # A concurrently written manifest may be temporarily incomplete.
            unreadable.append(path.parent.name)
            continue
        rows.extend(inventory_manifest(manifest, run_name=path.parent.name,
                                       source_hash=hashlib.sha256(contents).hexdigest()))
    if not rows:
        raise ValueError("No price-search manifests found")
    output = root / settings["output_root"] / settings["run_id"]
    output.mkdir(parents=True, exist_ok=False)
    table = pd.DataFrame(rows)
    table.to_csv(output / "inventory.csv", index=False)
    lines = ["# Price-search inventory", "",
             "This is a manifest snapshot, not a leaderboard. Periods, feature coverage and model-selection "
             "rules differ across rows. A missing convergence-warning field in an older fit means unknown, "
             "not verified convergence. Running manifests can change after this inventory was generated.", "",
             "| Run | Group | Native min | Evaluation dates | Status | Fits | Warning status unknown |",
             "|---|---|---:|---|---|---:|---:|"]
    for row in table.itertuples(index=False):
        lines.append(f"| {row.run} | {row.group} | {row.native_minutes} | {row.validation_start} to "
                     f"{row.validation_end_exclusive} exclusive | {row.status} | {row.completed_fits} | "
                     f"{row.warning_status_unknown_fits} |")
    if unreadable:
        lines.extend(["", "Manifests unreadable at snapshot time: " + ", ".join(unreadable) + "."])
    (output / "summary.md").write_text("\n".join(lines) + "\n")
    (output / "manifest.json").write_text(json.dumps({"status": "complete", "settings": settings,
        "groups": len(rows), "unreadable_manifests": unreadable}, indent=2) + "\n")
    return output


def main():
    """Write the configured inventory using existing saved artifacts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(run(load_config(args.config)))


if __name__ == "__main__":
    main()
