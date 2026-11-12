"""Compare completed Naumann studies using independent fixed-rate rainflow.

This is post-processing only: it does not start or resume any optimization.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow direct execution from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bess_pv_trading.domain import NaumannSettings
from bess_pv_trading.optimization.degradation import rainflow_discharge_depths
from bess_pv_trading.optimization.naumann import cycle_rate_factor, depth_exposure, rate_bin


def rainflow_benchmark(energy_mwh, capacity_mwh, settings, reference_rate):
    """Evaluate complete and residual half-cycles with one common fixed rate."""
    energy = np.asarray(energy_mwh, dtype=float)
    if capacity_mwh <= 0 or np.any(energy < -1e-8) or np.any(energy > capacity_mwh + 1e-8):
        raise ValueError("Energy trajectory is outside its nominal capacity")
    if not np.isfinite(reference_rate) or not 0 <= reference_rate <= settings.nominal_c_rate_max:
        raise ValueError("Reference rate must be within the nominal rate range")
    # Both directions contribute half dose, including unresolved half-cycles.
    depths = np.r_[rainflow_discharge_depths(energy, capacity_mwh),
                   rainflow_discharge_depths(capacity_mwh - energy, capacity_mwh)]
    knots = np.asarray(settings.depth_knots)
    multiplier = float(cycle_rate_factor(reference_rate, settings) ** 2)
    analytic = float(.5 * np.sum(depth_exposure(depths, settings)) * multiplier)
    piecewise = float(.5 * np.sum(np.interp(depths, knots, depth_exposure(knots, settings))) * multiplier)
    return {
        "reference_rate_c": reference_rate,
        "counted_half_cycles": int(len(depths)),
        "equivalent_full_cycles": float(.5 * np.sum(depths)),
        "rainflow_analytic_cycle_exposure": analytic,
        "rainflow_pwl_cycle_exposure": piecewise,
        "rainflow_analytic_cycle_loss_fraction": float(np.sqrt(analytic)),
        "rainflow_pwl_cycle_loss_fraction": float(np.sqrt(piecewise)),
    }


def unbinned_local_rate_exposure(frame, settings):
    """Revalue recorded interval doses at actual rates, holding the path fixed."""
    increments = (frame.naumann_cycle_exposure_end.to_numpy(dtype=float)
                  - frame.naumann_cycle_exposure_start.to_numpy(dtype=float))
    rates = (frame.charge_c_rate_nominal.to_numpy(dtype=float)
             + frame.discharge_c_rate_nominal.to_numpy(dtype=float))
    if not np.isfinite(increments).all() or not np.isfinite(rates).all() or np.min(increments) < -1e-12:
        raise ValueError("Invalid interval cycling exposure or rate")
    # Divide out the recorded upper-edge factor and apply the analytical rate.
    factors = np.asarray([rate_bin(rate, settings)[1] for rate in rates])
    exact = cycle_rate_factor(rates, settings) ** 2
    return float(np.sum(np.maximum(increments, 0) * exact / factors))


def summarize_run(directory, reference_rate):
    """Report solver evidence for every case and aging only for complete runs."""
    directory = Path(directory)
    if not (directory / "json/manifest.json").exists():
        return {"run": str(directory), "run_status": "not_started_or_missing"}
    manifest = json.loads((directory / "json/manifest.json").read_text())
    output = directory / "strategies/integrated"
    attempts_path = output / "json/optimization_attempts.json"
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    times = [item["runtime_seconds"] for item in attempts if item.get("runtime_seconds") is not None]
    gaps = [item["mip_gap"] for item in attempts if item.get("mip_gap") is not None]
    result = {
        "run": str(directory),
        "run_status": manifest.get("status", "unknown"),
        "run_error": manifest.get("error"),
        "auction_attempts": len(attempts),
        "optimal_solves": sum(item.get("status") == "optimal" for item in attempts),
        "time_limit_feasible": sum(item.get("status") == "feasible_time_limit" for item in attempts),
        "validation_failures": sum(item.get("validation", {}).get("valid") is False for item in attempts),
        "solver_seconds_total": sum(times),
        "solver_seconds_mean": float(np.mean(times)) if times else None,
        "maximum_reported_gap": max(gaps) if gaps else None,
    }
    # Never interpret a partial trajectory as a completed study's health result.
    if manifest.get("status") != "complete":
        return result
    config = json.loads((directory / "json/study_config.json").read_text())
    settings = NaumannSettings(**config["degradation"]["naumann"])
    capacity = float(config["battery"]["capacity_mwh"])
    frame = pd.read_csv(output / "csv/realized_dispatch.csv")
    if frame.empty:
        raise ValueError(f"Empty dispatch: {directory}")
    # Rainflow initialization cannot infer cycle memory from a used battery.
    if (abs(float(frame.naumann_calendar_exposure_start.iloc[0])) > 1e-15 or
            abs(float(frame.naumann_cycle_exposure_start.iloc[0])) > 1e-15):
        raise ValueError(f"Benchmark requires a fresh battery with no prior aging: {directory}")
    start = pd.to_datetime(frame["start"], utc=True)
    end = pd.to_datetime(frame["end"], utc=True)
    if not (end > start).all() or not (start.iloc[1:].reset_index(drop=True) == end.iloc[:-1].reset_index(drop=True)).all():
        raise ValueError(f"Non-contiguous dispatch timestamps: {directory}")
    if not np.allclose(frame.energy_start_mwh.iloc[1:], frame.energy_end_mwh.iloc[:-1], atol=1e-7, rtol=0):
        raise ValueError(f"Non-contiguous energy states: {directory}")
    energy = np.r_[frame.energy_start_mwh.iloc[0], frame.energy_end_mwh.to_numpy()]
    result.update(rainflow_benchmark(energy, capacity, settings, reference_rate))
    result.update({
        "rate_bin_width_c": settings.c_rate_bin_width,
        "intervals": len(frame),
        "initial_soh": float(frame.battery_soh_start.iloc[0]),
        "final_soh": float(frame.battery_soh_end.iloc[-1]),
        "recorded_cycle_loss_fraction": float(frame.naumann_cycle_loss_fraction_end.iloc[-1] - frame.naumann_cycle_loss_fraction_start.iloc[0]),
        "recorded_cycle_exposure": float(frame.naumann_cycle_exposure_end.iloc[-1] - frame.naumann_cycle_exposure_start.iloc[0]),
        "local_unbinned_rate_cycle_exposure": unbinned_local_rate_exposure(frame, settings),
        "recorded_calendar_loss_fraction": float(frame.naumann_calendar_loss_fraction_end.iloc[-1] - frame.naumann_calendar_loss_fraction_start.iloc[0]),
        "throughput_mwh": float(frame.throughput_mwh.sum()),
        "market_cashflow_eur": float(frame.realized_market_cashflow_eur.sum()),
        "recorded_wear_eur": float(frame.noncash_degradation_cost_eur.sum()),
        "absolute_imbalance_mwh": float(frame.imbalance_mwh.abs().sum()),
    })
    return result


def main():
    """Print one row per requested run, including failures and missing cases."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", help="Continuous experiment output directories")
    parser.add_argument("--reference-rate", type=float, default=0.5, help="Fixed reference C-rate")
    arguments = parser.parse_args()
    rows = [summarize_run(path, arguments.reference_rate) for path in arguments.runs]
    # Tiny squared-loss exposures must not be rounded to identical values.
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.9g}"))


if __name__ == "__main__":
    main()
