"""Audit a pinned rolling-price snapshot without loading or running a model."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.rolling_price_data import build_examples


def main():
    """Summarize target, block and eligible weather coverage by month and gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, help="Directory containing the merged manifest")
    parser.add_argument("--config", default="configs/research/zero_shot_price_study.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    study = load_config(args.config)["zero_shot_price_study"]
    options = load_config(study["inputs_config"])["rolling_price_inputs"]
    snapshot = Path(args.snapshot)
    examples = build_examples(snapshot, options)
    records = []
    for example in examples:
        day = example["delivery_day"]
        if not study["start"] <= day < study["end_exclusive"]:
            continue
        metadata = example["metadata"]
        actual = np.asarray(example["target_values"], dtype=float)
        record = {"delivery_day": day, "month": day[:7], "auction": example["market"],
                  "gate": example["gate"], "products": len(actual),
                  "observed_products": int(np.isfinite(actual).sum()),
                  "cancelled": metadata["cancelled"],
                  "weather_current_complete": metadata["weather_current_coverage"]["complete"],
                  "weather_with_revisions_complete": metadata["weather_usable"]}
        for block, coverage in metadata["feature_block_coverage"].items():
            record[f"{block}_complete"] = coverage["complete"]
            record[f"{block}_available"] = coverage["available"]
            record[f"{block}_required"] = coverage["required"]
        records.append(record)
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    detail = pd.DataFrame.from_records(records)
    detail.to_csv(destination / "daily_coverage.csv", index=False)
    if not detail.empty:
        sums = [column for column in detail.columns if column not in
                ("delivery_day", "month", "auction", "gate")]
        detail.groupby(["month", "auction", "gate"], dropna=False)[sums].sum().to_csv(
            destination / "monthly_coverage.csv")
    manifest = snapshot / "manifest.json"
    summary = {"snapshot": str(snapshot.resolve()), "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
               "start": study["start"], "end_exclusive": study["end_exclusive"],
               "reserved_start": study["reserved_start"], "records": len(records),
               "weather_rule": "latest complete eligible cycle using configured release lag; fallback is recorded per example"}
    (destination / "audit.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
