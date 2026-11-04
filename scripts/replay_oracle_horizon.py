"""Re-solve one archived legacy oracle horizon to investigate numerical behavior."""

from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bess_pv_trading.domain import DeliveryGrid, PhysicalState, PlantParameters, SolverSettings, battery_parameters_from_config
from bess_pv_trading.optimization.degradation import degradation_settings_from_config
from bess_pv_trading.optimization.mpc import MpcInput, MpcSettings, OraclePrices, solve_mpc
from bess_pv_trading.reporting.mpc_history import iter_mpc_history


def recorded_input(batch, job_id, interval_id):
    """Reconstruct a throughput-only oracle input from completed study artifacts."""
    batch = Path(batch).resolve()
    config_path = batch / f"config__{job_id}.json"
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    if config["degradation"]["model"] != "throughput":
        raise ValueError("Replay currently requires throughput degradation; Xu segment state is not reconstructed")
    state_settings = config["state"]
    if state_settings.get("terminal_soc_percent") is not None or state_settings.get("terminal_soc_min_percent") is not None:
        raise ValueError("Replay currently requires disabled hard terminal targets and floors")
    manifest = json.loads((batch / "manifest.json").read_text())
    matches = [item for item in manifest["results"] if item["job_id"] == job_id and item["status"] == "complete"]
    if len(matches) != 1:
        raise ValueError("The manifest must contain exactly one completed matching job")
    item = matches[0]
    controller = item["controller"]
    if controller not in {"deterministic_mpc_price_oracle", "ensemble_mpc_price_oracle"}:
        raise ValueError("Replay supports only legacy physical settlement-price oracle controllers")
    directory = batch / "runs" / f"farm_{item['farm_id']}" / job_id / "continuous/controllers" / controller
    solves = json.loads((directory / "json/mpc_solves.json").read_text())
    selected = [row for row in solves if row["interval_id"] == interval_id]
    if len(selected) != 1:
        raise ValueError("The selected interval must identify exactly one recorded solve")
    record = selected[0]
    cutoff = pd.Timestamp(record["cutoff_utc"])
    # Stream checksummed partitions and retain only the requested solve.
    fragments = []
    for part in iter_mpc_history(directory, kind="scenarios", batch_size=65536):
        selected_rows = part[part.solve_id == record["solve_id"]]
        if not selected_rows.empty:
            fragments.append(selected_rows)
    if not fragments:
        raise ValueError("No archived scenario history exists for this solve")
    frame = pd.concat(fragments, ignore_index=True)
    scenario_ids = record["scenario_ids"]
    intervals = sorted(frame.interval_id.unique(), key=pd.Timestamp)
    starts = pd.DatetimeIndex(intervals)
    if len(intervals) != record["horizon_intervals"] or starts[0] != cutoff:
        raise ValueError("Archived products do not match the recorded horizon")
    pv = frame.pivot(index="scenario_id", columns="interval_id", values="pv_ac_mw").reindex(
        index=scenario_ids, columns=intervals).to_numpy(dtype=float)
    grid = DeliveryGrid(tuple(intervals), tuple(starts.to_pydatetime()),
                        tuple((starts + pd.Timedelta(minutes=15)).to_pydatetime()))
    ledger = pd.DataFrame(json.loads((directory / "json/accepted_trades.json").read_text()))
    if ledger.empty:
        committed = np.zeros(len(intervals))
    else:
        eligible = ledger[pd.to_datetime(ledger.submission_time, utc=True) <= cutoff]
        committed = eligible.groupby("interval_id").quantity_mwh.sum().reindex(intervals, fill_value=0).to_numpy()
    actual = pd.read_csv(directory / "csv/controller_intervals.csv").set_index("interval_id").reindex(intervals)
    if actual.energy_start_mwh.isna().any() or actual.imbalance_price_eur_mwh.isna().any():
        raise ValueError("Archived state or realized settlement prices are incomplete")
    battery = battery_parameters_from_config(config["battery"])
    plant = PlantParameters(int(item["farm_id"]), config["plant"]["poi_import_mw"],
                            config["plant"]["poi_export_mw"], battery)
    negative = actual.get("imbalance_negative_price_eur_mwh")
    oracle = OraclePrices(actual.imbalance_price_eur_mwh.to_numpy(),
                          None if negative is None else negative.to_numpy(), {"diagnostic": True})
    inputs = MpcInput(cutoff, grid, plant, PhysicalState(cutoff.to_pydatetime(), float(actual.energy_start_mwh.iloc[0])),
                      committed, pv, tuple(scenario_ids), np.asarray(record["scenario_weights"]),
                      degradation_settings_from_config(config), policy=controller,
                      settings=MpcSettings.from_config(config), oracle=oracle,
                      import_fee_eur_per_mwh=config["trading"]["grid_import_fee_eur_per_mwh"])
    return inputs, config, hashlib.sha256(config_bytes).hexdigest()


def main(argv=None):
    """Solve the archived horizon once; do not alter or resume its source study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("job_id")
    parser.add_argument("interval_id")
    parser.add_argument("--feasibility-tolerance", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.resolve().is_relative_to(args.batch.resolve()):
        parser.error("Diagnostic output must be outside the source batch")
    inputs, config, config_hash = recorded_input(args.batch, args.job_id, args.interval_id)
    solver = deepcopy(config["solver"])
    if args.feasibility_tolerance is not None:
        solver["feasibility_tolerance"] = args.feasibility_tolerance
    result = solve_mpc(inputs, SolverSettings(**solver))
    payload = {"batch": str(args.batch.resolve()), "job_id": args.job_id, "interval_id": args.interval_id,
               "config_sha256": config_hash, "solver_settings": solver,
               "scope": "retrospective archived oracle horizon; no physical action or auction submission",
               "status": result.status, "diagnostics": result.diagnostics}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result.status, "output": str(args.output)}))
    return 0 if result.plan is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
