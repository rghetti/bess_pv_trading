"""Preflight or run the May 2026 three-controller comparison for all four farms."""

from __future__ import annotations

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.config import load_config
from bess_pv_trading.simulation.continuous_inputs import prepare_continuous_inputs
from bess_pv_trading.simulation.realtime_mpc_study import run_realtime_mpc_experiment
from run_realtime_mpc_comparison import comparison_plan


FARMS = (1, 345, 901, 910)
MODES = ("frozen_commitments", "continuous")
CONTROLLERS = ("track_committed_grid", "deterministic_mpc", "deterministic_mpc_price_oracle")
SUMMARY_NAME = "MAY_2026_ALL_FARMS_SUMMARY.md"


def _markdown_table(rows, columns):
    """Render a compact Markdown table and escape characters from metric strings."""
    render = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(render(row.get(column, "")) for column in columns) + " |" for row in rows)
    return "\n".join(lines)


def _summary_text(records, status, output_root):
    """Write a single cross-farm report with both study modes kept distinct."""
    lines = [
        "# May 2026 all-farm real-time MPC comparison",
        "",
        f"Status: **{status}**  ",
        f"Updated: {datetime.now(timezone.utc).isoformat()}  ",
        f"Output root: `{output_root}`",
        "",
        "## Study setup",
        "",
        "- Delivery period: 1–31 May 2026, Europe/Rome; four auctions where available.",
        "- Controllers: committed-grid tracking, deterministic MPC, deterministic MPC with the explicit price oracle.",
        "- Frozen-ledger mode compares physical control under one shared set of auction submissions.",
        "- Continuous mode allows each controller's auction reoptimization to change later commitments.",
        "- Current-interval realized PV is disclosed to physical control. The configured farm TCN supplies future PV.",
        "- Model artifacts postdate May 2026; the TCN comparison is retrospective. No training is performed.",
        "- Price-blind imbalance penalty: €500/MWh; oracle additional-imbalance budget: zero; two solver threads.",
        "- Detailed daily plots are capped at three dates per run.",
        "",
        "## Results",
        "",
        "Profit includes realized auction revenue, signed imbalance settlement, executed degradation and imports, and fees as recorded by the run. Planning penalties are excluded from realized profit.",
        "",
    ]
    if not records:
        lines.extend(["No comparison mode has completed yet.", ""])
        return "\n".join(lines)
    metric_columns = (
        "realized_operating_profit_eur", "absolute_imbalance_mwh", "absolute_settlement_exposure_eur",
        "imbalance_cashflow_eur", "auction_revenue_eur", "trading_fees_eur", "degradation_cost_eur",
        "import_cost_eur", "final_energy_mwh", "incremental_profit_eur",
        "absolute_imbalance_reduction_mwh", "settlement_exposure_difference_eur",
    )
    for mode in MODES:
        selected = [record for record in records if record["mode"] == mode]
        lines.extend([f"### {mode}", ""])
        if selected:
            rows = []
            for record in selected:
                for controller in record.get("comparison", []):
                    rows.append({"farm": record["farm"], **controller})
            present = [column for column in metric_columns if any(column in row for row in rows)]
            lines.append(_markdown_table(rows, ["farm", "controller", *present]) if rows
                         else "Preflight passed; no optimization or comparison metrics were generated.")
        else:
            lines.append("Not completed.")
        lines.append("")
    lines.extend(["## Run records", "", _markdown_table(records, ["farm", "mode", "status", "output_dir"]), ""])
    return "\n".join(lines)


def main(argv=None):
    """Run the four studies sequentially and update the shared summary as work completes."""
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true", help="Preflight all four farms without optimization")
    action.add_argument("--run", action="store_true", help="Run both comparison modes for all four farms")
    parser.add_argument("--output-dir", type=Path,
        default=ROOT / "outputs/realtime_mpc/may_2026_all_farms",
        help="One parent folder for all farm runs and the combined Markdown summary")
    args = parser.parse_args(argv)
    output_root = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    output_root.mkdir(parents=True, exist_ok=True)
    summary_path = output_root / SUMMARY_NAME
    records = []
    status = "preflighting" if args.check_only else "running"
    progress = lambda message: print(message, file=sys.stderr, flush=True)
    summary_path.write_text(_summary_text(records, status, output_root))

    try:
        for farm in FARMS:
            config_path = ROOT / f"configs/full_experiments/realtime_farm{farm}_may_2026_all_auctions.yaml"
            config = load_config(config_path)
            config["continuous_optimization"]["output_root"] = str(output_root)
            config["realtime_mpc_comparison"]["controllers"] = list(CONTROLLERS)
            progress(f"Farm {farm}: preparing shared May 2026 inputs")
            prepared = prepare_continuous_inputs(config, progress=progress)
            plan = comparison_plan(config, MODES)
            progress(f"Farm {farm}: {plan['physical_intervals_per_trajectory']} intervals, "
                     f"{plan['mpc_decisions']} MPC decisions across both modes")
            for mode in MODES:
                selected = deepcopy(config)
                selected["realtime_mpc_comparison"].update(mode=mode, check_only=args.check_only)
                result = run_realtime_mpc_experiment(selected, progress=progress, prepared_inputs=prepared)
                record = {"farm": farm, "mode": mode, **result}
                records.append(record)
                summary_path.write_text(_summary_text(records, status, output_root))
            progress(f"Farm {farm}: both modes complete")
    except BaseException as error:
        status = "interrupted" if isinstance(error, KeyboardInterrupt) else "failed"
        summary_path.write_text(_summary_text(records, status, output_root) + f"\nError: `{error}`\n")
        raise

    status = "preflight passed" if args.check_only else "complete"
    summary_path.write_text(_summary_text(records, status, output_root))
    print(json.dumps({"status": status, "output_root": str(output_root),
                      "summary": str(summary_path), "farm_modes": len(records)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
