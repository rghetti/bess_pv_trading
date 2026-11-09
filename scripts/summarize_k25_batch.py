"""Regenerate the combined summary for an already completed K25 batch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.reporting.k25_batch_summary import write_k25_batch_summary


def main(argv=None):
    """Require a completed batch before reading its existing results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args(argv)
    manifest = json.loads((args.batch / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or manifest.get("reports", {}).get("status") != "complete":
        raise ValueError("A completed batch with completed research reports is required")
    path = write_k25_batch_summary(args.batch, manifest["plan"]["jobs"], manifest["results"],
                                   plots=not args.no_plots)
    print(f"Summary written to {path}")


if __name__ == "__main__":
    main()
