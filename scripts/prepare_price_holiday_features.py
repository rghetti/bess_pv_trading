"""Prepare independent holiday-feature sidecars for a frozen price search.

The script reads calendar timestamps and split indices only. It never reads
price targets, fits models, opens DuckDB or changes the running search.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.price_feature_data import digest
from bess_pv_trading.forecasting.studies.price_holiday_features import build_holiday_features


def main():
    """Write one row-aligned feature matrix per market group and support counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/price_holiday_feature_expansion.yaml")
    args = parser.parse_args()
    settings = load_config(args.config)["price_holiday_expansion"]
    source = Path(settings["source_search"]).resolve()
    prepared_path = source / "prepared.json"
    prepared = json.loads(prepared_path.read_text())
    output = Path(settings["output_directory"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"source_prepared_sha256": digest(prepared_path),
                "source_scientific_config_sha256": prepared["scientific_config_sha256"],
                "settings": settings, "groups": {}}
    # Reuse each original group's exact target timestamps and cohort split indices.
    for group_name, group in prepared["groups"].items():
        starts_record = group["files"]["starts"]
        starts_path = source / starts_record["path"]
        if digest(starts_path) != starts_record["sha256"]:
            raise ValueError(f"Timestamp checksum mismatch: {group_name}")
        starts = pd.to_datetime(np.load(starts_path, allow_pickle=False), utc=True)
        matrix, names = build_holiday_features(starts, prepared["scientific_settings"], settings)
        path = output / f"{group_name}.npy"
        np.save(path, matrix, allow_pickle=False)
        cohort = group["cohorts"][settings["cohort"]]
        support = {}
        # Count distinct training and validation dates with each feature present.
        for split in ("inner_train", "inner_validation", "train", "validation"):
            descriptor = cohort["indices"][split]
            index_path = source / descriptor["path"]
            if digest(index_path) != descriptor["sha256"]:
                raise ValueError(f"Split checksum mismatch: {group_name}/{split}")
            indices = np.load(index_path, allow_pickle=False)
            local_days = starts.tz_convert(prepared["scientific_settings"]["timezone"]).strftime("%Y-%m-%d")
            support[split] = {name: int(len(np.unique(local_days[indices[matrix[indices, column] != 0]])))
                              for column, name in enumerate(names)}
        manifest["groups"][group_name] = {"path": path.name, "sha256": digest(path),
            "rows": int(len(matrix)), "features": names, "day_support": support,
            "source_starts_sha256": starts_record["sha256"]}
        print(f"Prepared {group_name}: {len(matrix)} rows, {len(names)} calendar features", flush=True)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote holiday-feature manifest to {output / 'manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
