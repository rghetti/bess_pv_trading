#!/usr/bin/env python
"""Inspect inputs or run an explicitly approved neural PV calibration study."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    """Keep input inspection read-only and training an explicit separate action."""
    from bess_pv_trading.config import load_config
    from bess_pv_trading.data import config_path
    from bess_pv_trading.forecasting.training.pv_mayer_neural import prepare_examples
    from bess_pv_trading.forecasting.training.pv_mayer_neural import prepare_search_examples
    from bess_pv_trading.forecasting.training.pv_mayer_neural import run
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config["pv_mayer_neural"]

    # Inspect eligible days and features without creating files or training weights.
    if args.check_only:
        source = config_path(config, settings["input_root"])
        summary = []
        # Keep input inspection read-only and training an explicit separate action for each `farm_id` in `settings['farm_ids']`.
        for farm_id in settings["farm_ids"]:
            path = source / f"farm_{farm_id}" / "hourly_inputs_normalized.parquet"
            frame = pd.read_parquet(path)
            training, stopping, evaluation, members = prepare_examples(
                frame,
                int(settings["member_count"]),
                int(settings["early_stopping_every_n_days"]),
            )
            # Show the separate three/one/one partitions used during the search.
            search_training, search_stopping, search_selection = prepare_search_examples(
                frame,
                int(settings["early_stopping_every_n_days"]),
            )
            summary.append({
                "farm_id": farm_id,
                "training_hours": len(training),
                "early_stopping_hours": len(stopping),
                "evaluation_hours": len(evaluation),
                "members": len(members),
                "search_training_hours": len(search_training),
                "search_early_stopping_hours": len(search_stopping),
                "search_selection_hours": len(search_selection),
            })
        print(json.dumps({"settings": settings, "farms": summary}, indent=2))
        return

    # Running this command is separate from approving any training experiment.
    run(config)


# Run the command only when this file is executed directly.
if __name__ == "__main__":
    main()
