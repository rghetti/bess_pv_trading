"""Compare a completed Naumann run with a historical throughput trajectory.

The common fixed-rate aging calculation is a model-based counterfactual, not
measured cell health; the historical trajectory retains its fixed capacity.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

# Make both the checkout and the neighboring rainflow report importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bess_pv_trading.domain import NaumannSettings
from bess_pv_trading.optimization.naumann import calendar_increment
from report_naumann_resolution import rainflow_benchmark


def load_run(path, strategy=None):
    """Load a completed continuous run and its unmodified study configuration."""
    root = Path(path)
    config = json.loads((root / "json/study_config.json").read_text())
    if strategy:
        dispatch_root = root / "strategies" / strategy
        manifest = json.loads((root / "json/manifest.json").read_text())
    else:
        dispatch_root = root
        manifest = json.loads((root / "json/manifest.json").read_text())
    if manifest.get("status") != "complete":
        raise ValueError(f"A complete run is required: {root}")
    return config, pd.read_csv(dispatch_root / "csv/realized_dispatch.csv")


def common_proxy(frame, capacity_mwh, settings, replacement_eur):
    """Value the same realized path with analytical stress at nominal SOC."""
    start = frame.energy_start_mwh.to_numpy(dtype=float)
    end = frame.energy_end_mwh.to_numpy(dtype=float)
    if not np.allclose(start[1:], end[:-1], atol=1e-7, rtol=0):
        raise ValueError("The battery energy trajectory must be continuous")
    calendar = sum(calendar_increment(a / capacity_mwh, b / capacity_mwh, seconds, settings)
                   for a, b, seconds in zip(start, end, frame.duration_hours * 3600))
    if settings.fixed_cycle_rate_c is None:
        raise ValueError("Common proxy requires a configured fixed cycle C-rate")
    cycle = rainflow_benchmark(np.r_[start[0], end], capacity_mwh, settings, settings.fixed_cycle_rate_c)
    calendar_loss = float(np.sqrt(calendar))
    cycle_loss = cycle["rainflow_analytic_cycle_loss_fraction"]
    loss = calendar_loss + cycle_loss
    cash = (frame.auction_revenue_eur + frame.imbalance_cashflow_eur
            - frame.import_fee_eur - frame.transaction_fee_eur).sum()
    wear = replacement_eur / (1 - settings.end_of_life_soh) * loss
    return {"market_cash_eur": float(cash), "auction_revenue_eur": float(frame.auction_revenue_eur.sum()),
            "imbalance_cashflow_eur": float(frame.imbalance_cashflow_eur.sum()),
            "original_reported_wear_eur": float(frame.degradation_eur.sum()),
            "common_calendar_loss_fraction": calendar_loss, "common_cycle_loss_fraction": cycle_loss,
            "common_total_loss_fraction": loss, "common_wear_proxy_eur": wear,
            "cash_minus_common_wear_proxy_eur": float(cash - wear),
            "equivalent_full_cycles": cycle["equivalent_full_cycles"]}


def compare(baseline, aging_run):
    """Require matched delivered intervals and compare both paths on one basis."""
    config_path = Path(aging_run) / "json/study_config.json"
    configured = json.loads(config_path.read_text())
    modes = configured["continuous_optimization"]["planning_modes"]
    if len(modes) != 1:
        raise ValueError("Aging run must have exactly one planning mode")
    aging_config, aging = load_run(aging_run, modes[0])
    baseline_config, historical = load_run(baseline)
    if baseline_config["degradation"]["model"] != "throughput":
        raise ValueError("Baseline must use linear throughput degradation")
    if aging_config["degradation"]["model"] != "naumann_lfp" or aging.empty:
        raise ValueError("A nonempty Naumann trajectory is required")
    if (abs(float(aging.naumann_calendar_exposure_start.iloc[0])) > 1e-15 or
            abs(float(aging.naumann_cycle_exposure_start.iloc[0])) > 1e-15):
        raise ValueError("Common proxy requires a fresh battery without prior exposure")
    capacity = float(aging_config["battery"]["capacity_mwh"])
    if float(baseline_config["battery"]["capacity_mwh"]) != capacity:
        raise ValueError("Battery capacities differ")
    first, last = aging.delivery_day.min(), aging.delivery_day.max()
    if first != baseline_config["continuous_optimization"]["start_date"]:
        raise ValueError("Historical baseline must start fresh on the first comparison day")
    historical = historical[historical.delivery_day.between(first, last)].reset_index(drop=True)
    aging = aging.reset_index(drop=True)
    for column in ("start", "end"):
        if not pd.to_datetime(aging[column], utc=True).equals(pd.to_datetime(historical[column], utc=True)):
            raise ValueError(f"The runs do not share identical {column} timestamps")
    settings = NaumannSettings(**aging_config["degradation"]["naumann"])
    economics = aging_config["degradation"]["replacement_economics"]
    replacement = capacity * 1000 * economics["pack_price_usd_per_kwh"] / economics["usd_per_eur"]
    replacement = (replacement + capacity * 1000 * economics["additional_replacement_eur_per_kwh"])
    replacement *= 1 - economics["salvage_fraction"]
    return {"period": [first, last], "intervals": len(aging), "common_replacement_eur": replacement,
            "maximum_actual_pv_difference_mw": float(np.max(np.abs(aging.pv_ac_mw - historical.pv_ac_mw))),
            "baseline": common_proxy(historical, capacity, settings, replacement),
            "naumann": common_proxy(aging, capacity, settings, replacement),
            "reference_rate_c": settings.fixed_cycle_rate_c,
            "baseline_planning_mode": baseline_config["continuous_optimization"]["planning_modes"],
            "naumann_planning_mode": modes,
            "limitations": "Historical fixed-capacity baseline and different replacement-cost assumptions; "
                           "common nominal-SOC fixed-rate Naumann proxy, not measured health."}


def main():
    """Print or persist the matched economic and common-aging comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", help="Historical throughput controller output directory")
    parser.add_argument("aging_run", help="Completed Naumann experiment output directory")
    parser.add_argument("--output", type=Path, help="Optional JSON result path")
    args = parser.parse_args()
    result = json.dumps(compare(args.baseline, args.aging_run), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
    print(result)


if __name__ == "__main__":
    main()
