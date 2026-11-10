"""Prepare a provenance-preserving fall-2026 price input snapshot; never fit models."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil

import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.data.price_search_inputs import _digest, build_price_search_inputs
from bess_pv_trading.data.price_source_fallback import prefer_existing_intervals


def prepare_merged_inputs(config, raw, merged, *, source_builder=build_price_search_inputs):
    """Export source snapshots, then create independent audited fallback tables."""
    raw = Path(raw)
    merged = Path(merged)
    if raw.exists() or merged.exists():
        raise FileExistsError("Use fresh source and merged directories to preserve input snapshots")
    config = {**config, "price_search_inputs": {**config["price_search_inputs"], "output_directory": str(raw)}}
    original = source_builder(config)
    merged.mkdir(parents=True)
    # Keep all raw tables and original snapshot manifests; derived tables go separately.
    result = {**original, "tables": {}}
    tables = {name: pd.read_parquet(raw / item["path"]) for name, item in original["tables"].items()}
    # Save source-backed cancellation notices; absence of prices alone is not proof.
    settings = config["price_search_inputs"]
    if settings.get("include_gme_cancellation_notices", False):
        from bess_pv_trading.forecasting.studies.price_cancellation_features import catalog_cancellations
        result["gme_cancellation_notices"] = catalog_cancellations(settings["database_path"],
            settings["start"], settings["end_exclusive"], settings["timezone"])
    da = tables["nord_auctions"]
    replacement, da_audit = prefer_existing_intervals(da, tables["energy_charts_nord_da"],
        identity_columns=["dataset", "zone", "product_start", "product_end"])
    france, france_audit = prefer_existing_intervals(tables["france_balance"], tables["france_balance_realtime"],
        identity_columns=["zone", "product_start", "product_end"])
    derived = {"nord_auctions": replacement, "france_balance": france}
    fallback_names = {"nord_auctions": "energy_charts_nord_da", "france_balance": "france_balance_realtime"}
    for name, item in original["tables"].items():
        destination = merged / item["path"]
        if name in derived:
            derived[name].to_parquet(destination, index=False)
            row_hashes = pd.util.hash_pandas_object(derived[name], index=False, categorize=False).to_numpy(dtype="uint64")
            record = {**item, "rows": len(derived[name]), "sha256": _digest(destination),
                      "primary_selected_row_sha256": item["selected_row_sha256"],
                      "selected_row_sha256": hashlib.sha256(row_hashes.astype("<u8").tobytes()).hexdigest(),
                      "derived_fallback": True,
                      "fallback_snapshot_sources": original["tables"][fallback_names[name]]["snapshot_sources"]}
        else:
            shutil.copyfile(raw / item["path"], destination)
            record = dict(item)
        result["tables"][name] = record
    result["raw_manifest_sha256"] = _digest(raw / "manifest.json")
    result["fallback_policy"] = {"da": {"priority": ["gme", "energy_charts"], **da_audit},
        "france": {"priority": ["consolidated", "realtime"], **france_audit},
        "record_provenance": "provider, dataset, snapshot ID and row ID preserved in every row"}
    (merged / "manifest.json").write_text(json.dumps(result, indent=2, default=str) + "\n")
    print(json.dumps(result["fallback_policy"], indent=2))
    print(f"Prepared inputs: {merged}")
    return result


def main():
    """Expose the same snapshot preparation for inspection without fitting models."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/research/price_search_inputs_full_year.yaml")
    parser.add_argument("--database", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    config["price_search_inputs"]["database_path"] = args.database
    root = Path(args.output).resolve()
    prepare_merged_inputs(config, root / "raw", root / "merged")


if __name__ == "__main__":
    main()
