"""Compare fitted and pvlib-default Faiman coefficients on common observations."""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path, load_metadata, night_window_mask
from bess_pv_trading.forecasting.plotting.pv_review_predictions import historical_predictions
from bess_pv_trading.forecasting.studies.pv_realized_ensemble import selected_settings


def digest(path):
    """Identify the exact aligned input and model used in the comparison."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def compare(config, farm_id, model_path, output):
    """Evaluate both temperature settings on identical eligible four-view means."""
    section = config['pv_realized_irradiance_search']
    root = config_path(config, section['output_root']) / section['run_id']
    input_path = root / f'farm_{farm_id}' / 'aligned_inputs.parquet'
    model_path = Path(model_path)
    result = json.loads(model_path.read_text())
    if int(result['farm_id']) != farm_id:
        raise ValueError('Model farm does not match requested farm')
    farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    frame = pd.read_parquet(input_path)
    frame = frame.loc[frame.eligible].copy()
    observed = frame.groupby('interval_start_utc').agg(
        target_mw=('measured_ac_mw', 'first'), ghi_w_m2=('ghi_w_m2', 'first'))
    fitted = selected_settings(config, farm, result)
    defaults = deepcopy(fitted)
    signature = inspect.signature(pvlib.temperature.faiman)
    defaults['temperature'].update(u0=float(signature.parameters['u0'].default),
                                   u1=float(signature.parameters['u1'].default))
    night_policy = section.get('night_policy', config['policy']['night'])
    for name, settings in [('fitted', fitted), ('pvlib_default_faiman', defaults)]:
        forecast = historical_predictions(frame, farm, settings, night_policy)
        observed[name] = forecast.historical_ghi_prediction_mw
    common = observed.dropna(subset=['target_mw', 'fitted', 'pvlib_default_faiman']).copy()
    night = night_window_mask(common.index, night_policy)
    scopes = {'all_hours': np.ones(len(common), dtype=bool), 'non_night': ~night,
              'ghi_above_20': common.ghi_w_m2.to_numpy(float) > 20}
    rows = []
    for scope, mask in scopes.items():
        for name in ['fitted', 'pvlib_default_faiman']:
            part = common.loc[mask]
            error = (part[name] - part.target_mw).to_numpy(float)
            if not len(error):
                raise ValueError(f'No common eligible observations for {scope}')
            mae, rmse, bias = float(np.mean(abs(error))), float(np.sqrt(np.mean(error**2))), float(np.mean(error))
            rows.append({'farm_id': farm_id, 'scope': scope, 'model': name, 'quarters': len(error),
                         'mae_mw': mae, 'rmse_mw': rmse, 'bias_mw': bias,
                         'mae_percent_capacity': 100*mae/farm.ac_capacity_mw,
                         'rmse_percent_capacity': 100*rmse/farm.ac_capacity_mw,
                         'bias_percent_capacity': 100*bias/farm.ac_capacity_mw})
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    common.reset_index().to_parquet(output / 'paired_predictions.parquet', index=False)
    pd.DataFrame(rows).to_csv(output / 'metrics.csv', index=False)
    manifest = {'farm_id': farm_id, 'input_path': str(input_path), 'input_sha256': digest(input_path),
        'selected_model': str(model_path), 'selected_model_sha256': digest(model_path),
        'fitted_u0': result['parameters']['u0'], 'fitted_u1': result['parameters']['u1'],
        'pvlib_default_u0': defaults['temperature']['u0'], 'pvlib_default_u1': defaults['temperature']['u1'],
        'gamma_fixed_per_degree_c': result['parameters']['gamma'],
        'method': 'same physical model, weather, treated targets, geometry and scale; only Faiman u0/u1 differ',
        'aggregation': 'mean of four eligible weather views per physical quarter-hour',
        'limitation': 'full-period in-sample retrospective comparison; parameters not refitted for defaults'}
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(pd.DataFrame(rows).to_string(index=False), flush=True)


def main():
    """Compare one farm using explicitly selected input and saved model paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--farm', required=True, type=int)
    parser.add_argument('--selected', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    compare(load_config(args.config), args.farm, args.selected, args.output)


if __name__ == '__main__':
    main()
