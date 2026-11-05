"""Independently reconstruct grid-neutral curtailment recovery in saved runs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from bess_pv_trading.domain import battery_parameters_from_config

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('batch', type=Path)
parser.add_argument('--output', type=Path, required=True)
arguments = parser.parse_args()
batch = arguments.batch.resolve()
manifest = json.loads((batch / 'manifest.json').read_text())
assert manifest['status'] == 'complete'
entries, violations = [], []
for job in manifest['results']:
    if job['case_id'] != 'mpc_recovery':
        continue
    config = json.loads((batch / f"config__{job['job_id']}.json").read_text())
    battery = battery_parameters_from_config(config['battery'])
    out = Path(job['result']['output_dir']) / 'controllers/deterministic_mpc/csv/controller_intervals.csv'
    frame = pd.read_csv(out)
    frame = frame.loc[~frame.mpc_fallback.astype(bool)].copy()
    dt = frame.duration_hours.to_numpy()
    pv = frame.pv_ac_mw.to_numpy()
    retained = frame.energy_start_mwh.to_numpy() * (1 - battery.self_discharge_per_hour) ** dt
    charge_limit = np.minimum(battery.max_power_mw, np.maximum(0, battery.maximum_energy - retained) / (dt * battery.eta_charge))
    discharge_limit = np.minimum(battery.max_power_mw, np.maximum(0, retained - battery.energy_min_mwh) * battery.eta_discharge / dt)
    charge = np.minimum(frame.requested_charge_mw.to_numpy(), charge_limit)
    discharge = np.minimum(frame.requested_discharge_mw.to_numpy(), discharge_limit)
    curtailed = np.minimum(pv, frame.requested_curtailment_mw.to_numpy())
    grid = pv - curtailed + discharge - charge
    reduce_import = np.minimum(charge, np.maximum(0, -config['plant']['poi_import_mw'] - grid))
    charge -= reduce_import
    grid += reduce_import
    reduce_export = np.minimum(discharge, np.maximum(0, grid - config['plant']['poi_export_mw']))
    discharge -= reduce_export
    grid -= reduce_export
    curtailed += np.maximum(0, grid - config['plant']['poi_export_mw'])
    protected_grid = np.minimum(grid, config['plant']['poi_export_mw'])
    available = np.minimum(curtailed, np.maximum(0, discharge - charge + charge_limit))
    power = discharge - charge - available
    expected_charge = np.maximum(0, -power)
    expected_discharge = np.maximum(0, power)
    expected_curtailment = curtailed - available
    residuals = {
        'grid_neutrality_mw': float(np.max(np.abs(frame.grid_mw - protected_grid))),
        'maximal_recovery_mw': float(np.max(np.abs(frame.curtailment_recovered_mw - available))),
        'charge_mw': float(np.max(np.abs(frame.charge_mw - expected_charge))),
        'discharge_mw': float(np.max(np.abs(frame.discharge_mw - expected_discharge))),
        'curtailment_mw': float(np.max(np.abs(frame.curtailment_mw - expected_curtailment))),
        'soc_upper_mwh': float(max(0, (frame.energy_end_mwh - battery.maximum_energy).max())),
        'soc_lower_mwh': float(max(0, (battery.energy_min_mwh - frame.energy_end_mwh).max())),
    }
    failed = {key: value for key, value in residuals.items() if not np.isfinite(value) or value > 1e-6}
    if failed:
        violations.append({'farm': job['farm_id'], 'residuals': failed})
    entries.append({'farm': str(job['farm_id']), 'intervals_checked': len(frame),
        'recoveries': int(np.count_nonzero(available > 1e-8)),
        'recovered_mwh': float(np.sum(available * dt)), 'maximum_residuals': residuals})
report = {'recovery_audit': entries, 'violations': violations}
destination = arguments.output
destination.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
if violations:
    raise SystemExit(1)
