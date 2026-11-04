"""Benchmark equivalent first-stage MPC formulations on a recorded replay solve."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.domain import (DegradationSettings, DeliveryGrid, PhysicalState, PlantParameters, SolverSettings,
    battery_parameters_from_config)
from bess_pv_trading.optimization.degradation import degradation_settings_from_config
from bess_pv_trading.optimization.mpc import MpcInput, MpcSettings, OraclePrices, build_mpc_model, solve_mpc
from bess_pv_trading.optimization.physical import add_battery_controls, energy_transition


def recorded_first_stage(batch, job_id):
    """Reconstruct the first solve's forecasts, state and already submitted ledger."""
    config = json.loads((batch / f"config__{job_id}.json").read_text())
    if config["state"].get("terminal_soc_percent") is not None or config["state"].get("terminal_soc_min_percent") is not None:
        raise ValueError("This benchmark requires the recorded replay to have no hard terminal SOC bound")
    if config["degradation"].get("model", "throughput") != "throughput":
        raise ValueError("This benchmark reconstructs throughput-wear first-stage constraints only")
    farm = str(config["farm_id"])
    directory = batch / "runs" / f"farm_{farm}" / job_id / "continuous/controllers/ensemble_mpc_price_oracle"
    solves = json.loads((directory / "json/mpc_solves.json").read_text())
    first = solves[0]
    cutoff = pd.Timestamp(first["cutoff_utc"])
    count, length = first["scenario_count"], first["horizon_intervals"]
    # The original writer emits all interval/member pairs for each solve in order.
    rows = pd.read_csv(directory / "csv/mpc_scenarios.csv", nrows=count * length)
    if len(rows) != count * length or set(rows.solve_id) != {first["solve_id"]}:
        raise ValueError("Recorded scenario rows do not contain exactly the first horizon")
    ids = first["scenario_ids"]
    intervals = sorted(rows.interval_id.unique(), key=pd.Timestamp)
    starts = pd.DatetimeIndex(intervals)
    ends = starts + pd.Timedelta(minutes=15)
    grid = DeliveryGrid(tuple(intervals), tuple(starts.to_pydatetime()), tuple(ends.to_pydatetime()))
    pv = rows.pivot(index="scenario_id", columns="interval_id", values="pv_ac_mw").reindex(
        index=ids, columns=intervals).to_numpy(float)
    ledger = pd.DataFrame(json.loads((directory / "json/accepted_trades.json").read_text()))
    eligible = ledger[pd.to_datetime(ledger.submission_time, utc=True) <= cutoff]
    commitments = eligible.groupby("interval_id").quantity_mwh.sum().reindex(intervals, fill_value=0).to_numpy()
    actual = pd.read_csv(directory / "csv/controller_intervals.csv")
    energy = float(actual.loc[actual.interval_id == intervals[0], "energy_start_mwh"].iloc[0])
    battery = battery_parameters_from_config(config["battery"])
    plant_config = config["plant"]
    plant = PlantParameters(int(farm), plant_config["poi_import_mw"], plant_config["poi_export_mw"], battery)
    # First-stage imbalance is price blind. For a full fallback benchmark, use
    # the disclosed historical settlement curve already recorded by the replay.
    prices = actual.set_index("interval_id").reindex(intervals).imbalance_price_eur_mwh.to_numpy(float)
    inputs = MpcInput(cutoff, grid, plant, PhysicalState(cutoff.to_pydatetime(), energy), commitments,
        pv, tuple(ids), np.array(first["scenario_weights"]), DegradationSettings(),
        policy="ensemble_mpc_price_oracle", settings=MpcSettings(terminal_inventory_mode="none"),
        oracle=OraclePrices(prices))
    return inputs, first, config


def build_epigraph_stage1(inputs):
    """Use a single absolute-deviation epigraph per distinct PV/interval support."""
    import gurobipy as gp

    battery = inputs.plant.battery
    count, length = inputs.pv_ac_mw.shape
    model = gp.Model("stage1_epigraph")
    model.setParam("OutputFlag", 0)
    charge, discharge, modes, utilization = {}, {}, {}, {}
    for t in range(length):
        charge[t], discharge[t], modes[t] = add_battery_controls(model, battery, gp, f"mpc,t{t}")
        utilization[t] = model.addVar(ub=1, name=f"pv_utilization[t{t}]")
        if np.all(inputs.pv_ac_mw[:, t] == 0):
            model.addConstr(utilization[t] == 1)
    energy = {t: model.addVar(lb=battery.energy_min_mwh, ub=battery.maximum_energy,
                              name=f"energy[t{t}]") for t in range(length + 1)}
    model.addConstr(energy[0] == inputs.state.energy_mwh)
    for t, duration in enumerate(inputs.grid.delta_hours):
        model.addConstr(energy[t + 1] == energy_transition(energy[t], charge[t], discharge[t], battery, duration))
    deviations = {}
    objective = gp.LinExpr()
    for scenario in range(count):
        for t, duration in enumerate(inputs.grid.delta_hours):
            key = (t, float(inputs.pv_ac_mw[scenario, t]))
            if key not in deviations:
                grid = key[1] * utilization[t] + discharge[t] - charge[t]
                model.addConstr(grid >= -inputs.plant.poi_import_mw)
                model.addConstr(grid <= inputs.plant.poi_export_mw)
                deviation = grid * duration - inputs.committed_mwh[t]
                absolute = model.addVar(name=f"absolute[s{scenario},t{t}]")
                model.addConstr(absolute >= deviation)
                model.addConstr(absolute >= -deviation)
                deviations[key] = absolute
            objective += float(inputs.weights[scenario]) * deviations[key]
    model.update()
    return model, {}, {"absolute_imbalance_mwh": objective}


def benchmark(inputs, variant, time_limit, mip_gap):
    """Solve the same MILP under a named solver setting or valid SOC tightening."""
    if variant == "qualified_fallback":
        # Each stage has its own budget. The first incumbent must meet the
        # independently reconstructed 1% threshold before economic refinement.
        settings = replace(inputs.settings, solver_time_limit_seconds=time_limit,
            oracle_fallback_max_relative_gap=0.01, oracle_stage2_time_limit_seconds=time_limit)
        solved = solve_mpc(replace(inputs, settings=settings), SolverSettings(threads=1, mip_gap=mip_gap,
            feasibility_tolerance=1e-8, integrality_tolerance=1e-8))
        return {"variant": variant, "status": solved.status, "diagnostics": solved.diagnostics}
    started = perf_counter()
    model, variables, expressions = (build_epigraph_stage1(inputs) if variant == "epigraph"
        else build_mpc_model(inputs))
    try:
        model.setParam("Threads", 1)
        model.setParam("MIPGap", mip_gap)
        model.setParam("TimeLimit", time_limit)
        model.setParam("FeasibilityTol", 1e-8)
        model.setParam("IntFeasTol", 1e-8)
        if variant.startswith("focus_"):
            model.setParam("MIPFocus", int(variant[-1]))
        if variant == "tight_soc":
            battery = inputs.plant.battery
            # These are valid for either binary mode, including standby loss at
            # the lower energy bound; they strengthen the fractional relaxation.
            for t, duration in enumerate(inputs.grid.delta_hours):
                energy = variables["energy"][0, t]
                charge = variables["charge"][t, "mpc"]
                discharge = variables["discharge"][t, "mpc"]
                mode = variables["mode"][t]
                retention = (1 - battery.self_discharge_per_hour) ** duration
                model.addConstr(retention * energy + battery.eta_charge * duration * charge
                    <= battery.maximum_energy)
                model.addConstr(retention * energy - duration * discharge / battery.eta_discharge
                    >= battery.energy_min_mwh - (1 - retention) * battery.energy_min_mwh * mode)
        model.setObjective(expressions["absolute_imbalance_mwh"])
        model.update()
        if variant == "relaxation":
            # This is a diagnostic lower bound only, not a deployable policy.
            relaxed = model.relax()
            model.dispose()
            model = relaxed
            model.setParam("Threads", 1)
            model.setParam("TimeLimit", time_limit)
            model.setParam("OutputFlag", 0)
        model.optimize()
        fractional_modes = None
        simultaneous_intervals = None
        if model.SolCount:
            modes = [model.getVarByName(f"charge_mode[mpc,t{t}]").X
                for t in range(len(inputs.grid.interval_ids))]
            charge = [model.getVarByName(f"charge[mpc,t{t}]").X
                for t in range(len(inputs.grid.interval_ids))]
            discharge = [model.getVarByName(f"discharge[mpc,t{t}]").X
                for t in range(len(inputs.grid.interval_ids))]
            fractional_modes = sum(1e-7 < value < 1 - 1e-7 for value in modes)
            simultaneous_intervals = int(np.count_nonzero(np.minimum(charge, discharge) > 1e-7))
        bound = model.ObjBound if model.IsMIP else (model.ObjVal if model.Status == 2 else None)
        result = {"variant": variant, "variables": model.NumVars, "binary_variables": model.NumBinVars,
            "constraints": model.NumConstrs, "status_code": model.Status, "runtime_seconds": model.Runtime,
            "wall_seconds": perf_counter() - started, "nodes": model.NodeCount if model.IsMIP else None,
            "objective_mwh": model.ObjVal if model.SolCount else None,
            "bound_mwh": bound if model.SolCount else None,
            "relative_gap": model.MIPGap if model.IsMIP and model.SolCount and np.isfinite(model.MIPGap) else None,
            "fractional_mode_intervals": fractional_modes,
            "simultaneous_charge_discharge_intervals": simultaneous_intervals}
        return result
    finally:
        model.dispose()


def main():
    """Read one immutable completed batch and compare equivalent one-thread solves."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("job_id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=10)
    parser.add_argument("--mip-gap", type=float, default=1e-4)
    parser.add_argument("--variant", action="append", choices=("baseline", "tight_soc", "focus_1", "focus_2", "focus_3", "epigraph", "relaxation", "qualified_fallback"))
    args = parser.parse_args()
    inputs, original, config = recorded_first_stage(args.batch, args.job_id)
    variants = args.variant or ("baseline", "tight_soc", "focus_1", "focus_2", "focus_3", "epigraph")
    values = []
    for variant in variants:
        selected = inputs
        if variant == "qualified_fallback":
            selected = replace(inputs, settings=MpcSettings.from_config(config),
                degradation=degradation_settings_from_config(config))
        values.append(benchmark(selected, variant, args.time_limit, args.mip_gap))
    result = {"batch": str(args.batch), "job_id": args.job_id, "cutoff_utc": str(inputs.cutoff),
        "members": len(inputs.scenario_ids), "intervals": len(inputs.grid.interval_ids),
        "original_stage": original["stages"][0], "variants": values,
        "scope": "one recorded retrospective first-stage problem, one solver thread per variant"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
