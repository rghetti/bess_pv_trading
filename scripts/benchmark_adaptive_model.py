"""Rebuild and benchmark a recorded economic MPC horizon with one Gurobi thread."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.domain import DeliveryGrid, PhysicalState, PlantParameters, battery_parameters_from_config
from bess_pv_trading.optimization.adaptive_mpc import build_research_model
from bess_pv_trading.optimization.degradation import degradation_settings_from_config
from bess_pv_trading.optimization.mpc import ForecastPrices, MpcInput, MpcSettings


def recorded_horizon(batch, job_id, controller, decision_index=0):
    """Recover one previously saved forecast horizon and submitted positions."""
    config = json.loads((batch / f"config__{job_id}.json").read_text())
    farm = int(config["farm_id"])
    directory = batch / "runs" / f"farm_{farm}" / job_id / "continuous/controllers" / controller
    solves = json.loads((directory / "json/mpc_solves.json").read_text())
    if isinstance(decision_index, bool) or not 0 <= decision_index < len(solves):
        raise ValueError("decision_index must identify a recorded MPC solve")
    first = solves[decision_index]
    cutoff = pd.Timestamp(first["cutoff_utc"])
    starts = pd.date_range(cutoff, periods=first["horizon_intervals"], freq="15min")
    day = cutoff.tz_convert(config["market"]["timezone"]).date().isoformat()
    files = list((directory / "diagnostics/mpc_horizon" / day).glob("scenarios-*.parquet"))
    if len(files) != 1:
        raise ValueError("Expected one immutable scenario partition for the selected day")
    frame = pd.read_parquet(files[0], filters=[("solve_id", "=", first["solve_id"])])
    ids = first["scenario_ids"]
    intervals = sorted(frame.interval_id.unique(), key=pd.Timestamp)
    if not pd.DatetimeIndex(intervals).equals(starts):
        raise ValueError("Saved scenario products do not match the first horizon")
    def matrix(column):
        return frame.pivot(index="scenario_id", columns="interval_id", values=column).reindex(
            index=ids, columns=intervals).to_numpy(float)
    pv = matrix("pv_ac_mw")
    positive = matrix("forecast_positive_price_eur_per_mwh")
    negative = matrix("forecast_negative_price_eur_per_mwh")
    if not np.isfinite(positive).all() or not np.isfinite(negative).all():
        raise ValueError("Benchmark requires finite saved forecast-price paths")
    ledger = pd.DataFrame(json.loads((directory / "json/accepted_trades.json").read_text()))
    ledger = ledger[pd.to_datetime(ledger.submission_time, utc=True) <= cutoff]
    committed = ledger.groupby("interval_id").quantity_mwh.sum().reindex(intervals, fill_value=0).to_numpy()
    actual = pd.read_csv(directory / "csv/controller_intervals.csv").set_index("interval_id")
    energy = float(actual.loc[intervals[0], "energy_start_mwh"])
    grid = DeliveryGrid(tuple(intervals), tuple(starts.to_pydatetime()),
        tuple((starts + pd.Timedelta(minutes=15)).to_pydatetime()))
    battery = battery_parameters_from_config(config["battery"])
    plant = PlantParameters(farm, config["plant"]["poi_import_mw"], config["plant"]["poi_export_mw"], battery)
    settings = MpcSettings.from_config(config)
    forecast = ForecastPrices(positive, cutoff, negative if not np.array_equal(positive, negative) else None)
    return MpcInput(cutoff, grid, plant, PhysicalState(cutoff.to_pydatetime(), energy), committed,
        pv, tuple(ids), np.asarray(first["scenario_weights"]), degradation_settings_from_config(config),
        policy=controller, settings=settings, price_forecast=forecast,
        import_fee_eur_per_mwh=config["trading"]["grid_import_fee_eur_per_mwh"])


def main():
    """Measure model construction and solver output; never execute a replay."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("job_id")
    parser.add_argument("controller")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--decision-index", type=int, default=0)
    parser.add_argument("--time-limit", type=float, default=10.0)
    args = parser.parse_args()
    if args.repeat < 1 or args.time_limit <= 0:
        parser.error("repeat and time-limit must be positive")
    inputs = recorded_horizon(args.batch, args.job_id, args.controller, args.decision_index)
    results = []
    for _ in range(args.repeat):
        started = perf_counter()
        model, variables, expressions = build_research_model(inputs)
        try:
            build = perf_counter() - started
            model.setParam("Threads", 1)
            model.setParam("MIPGap", 1e-4)
            model.setParam("TimeLimit", args.time_limit)
            model.setParam("FeasibilityTol", 1e-8)
            model.setParam("IntFeasTol", 1e-8)
            settings = inputs.settings
            objective = (settings.economic_imbalance_penalty_eur_per_mwh * expressions["absolute_imbalance_mwh"]
                + expressions["degradation_eur"] + expressions["import_cost_eur"]
                + expressions["terminal_cost_eur"] - expressions["settlement_eur"]
                + settings.monetary_cvar_weight * expressions["monetary_cvar_eur"])
            model.setObjective(objective)
            model.optimize()
            results.append({"build_seconds": build, "solve_seconds": model.Runtime,
                "variables": model.NumVars, "binary_variables": model.NumBinVars,
                "constraints": model.NumConstrs, "status_code": model.Status,
                "objective_eur": model.ObjVal if model.SolCount else None,
                "bound_eur": model.ObjBound if model.SolCount else None,
                "first_action": [variables["charge"][0, 0].X, variables["discharge"][0, 0].X,
                                 variables["utilization"][0, 0].X] if model.SolCount else None})
        finally:
            model.dispose()
    payload = {"batch": str(args.batch), "job_id": args.job_id, "controller": args.controller,
        "decision_index": args.decision_index,
        "cutoff": str(inputs.cutoff), "scenarios": len(inputs.scenario_ids),
        "intervals": len(inputs.grid.interval_ids), "results": results,
        "scope": "recorded retrospective forecast horizon; one solver thread, no replay"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps(payload, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
