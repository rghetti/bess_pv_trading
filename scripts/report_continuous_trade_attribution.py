"""Generate immutable-ledger integrated versus sequential explanations."""
from __future__ import annotations

import argparse
from pathlib import Path

from bess_pv_trading.reporting.continuous_trade_attribution import generate_continuous_trade_attribution


def main():
    """Regenerate attribution from an existing continuous participation run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True,
        help="Existing farm participation output directory containing strategies and ledgers")
    args = parser.parse_args()
    output = generate_continuous_trade_attribution(args.run_dir)
    print(f"Trade attribution saved to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
