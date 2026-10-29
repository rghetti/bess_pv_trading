"""Compare physical predictions with measured AC power by season and split."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import load_config
from ...data import config_path, load_metadata
from ..plotting.pv_review_predictions import historical_predictions
from ..studies.pv_realized_ensemble import selected_settings


def evaluate(config, farm_id, baseline_path=None):
    """Score complete four-view means on the same eligible physical quarters."""
    section = config['pv_realized_irradiance_search']
    root = config_path(config, section['output_root']) / section['run_id']
    directory = root / f'farm_{farm_id}'
    farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    frame = pd.read_parquet(directory / 'aligned_inputs.parquet')
    frame = frame.loc[frame.eligible].copy()
    observations = frame.groupby('interval_start_utc').agg(
        measured_ac_mw=('measured_ac_mw', 'first'), ghi_w_m2=('ghi_w_m2', 'first'),
        split=('split', 'first'))
    paths = {'selected': directory / 'selected_parameters.json'}
    if baseline_path:
        paths['previous_baseline'] = Path(baseline_path)
    predictions = {}
    for label, path in paths.items():
        result = json.loads(path.read_text())
        if int(result['farm_id']) != farm_id:
            raise ValueError('Model and requested farm differ')
        settings = selected_settings(config, farm, result)
        converted = historical_predictions(frame, farm, settings,
            section.get('night_policy', config['policy']['night']))
        predictions[label] = converted.historical_ghi_prediction_mw
    comparable = observations.join(pd.DataFrame(predictions)).dropna(subset=list(predictions))
    local = comparable.index.tz_convert(section['delivery_timezone'])
    comparable['season'] = np.select(
        [local.month.isin([12, 1, 2]), local.month.isin([3, 4, 5]), local.month.isin([6, 7, 8])],
        ['winter', 'spring', 'summer'], default='autumn')
    # Match the prior seasonal analysis, keeping low-irradiance and night out.
    comparable = comparable.loc[comparable.ghi_w_m2 > 20].copy()
    rows = []
    for split in ['train', 'validation', 'all']:
        subset = comparable if split == 'all' else comparable.loc[comparable['split'].eq(split)]
        reported_split = ({'train': 'pre_august_fit', 'validation': 'august_fit', 'all': 'all_fit'}[split]
                          if section.get('selection_scope', 'validation') == 'full_period' else split)
        for season, part in subset.groupby('season', sort=True):
            for label in paths:
                error = (part[label] - part.measured_ac_mw).to_numpy(float)
                rows.append({'model': label, 'split': reported_split, 'season': season,
                    'quarters': len(error), 'over_count': int((error > 1e-9).sum()),
                    'under_count': int((error < -1e-9).sum()),
                    'equal_count': int((np.abs(error) <= 1e-9).sum()),
                    'over_percent': 100 * float((error > 1e-9).mean()),
                    'mae_percent_capacity': 100 * float(np.mean(abs(error))) / farm.ac_capacity_mw,
                    'rmse_percent_capacity': 100 * float(np.sqrt(np.mean(error**2))) / farm.ac_capacity_mw,
                    'bias_percent_capacity': 100 * float(np.mean(error)) / farm.ac_capacity_mw,
                    'bias_kw': 1000 * float(np.mean(error))})
    output = directory / 'seasonal_comparison.csv'
    pd.DataFrame(rows).to_csv(output, index=False)
    return output


def main():
    """Write seasonal diagnostics for the selected chain and optional baseline."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--baseline-selected')
    parser.add_argument('--baseline-root', help='Prior run containing farm_<id>/selected_parameters.json')
    args = parser.parse_args()
    config = load_config(args.config)
    if args.baseline_selected and (args.baseline_root or len(config['pv_realized_irradiance_search']['farm_ids']) != 1):
        raise ValueError('A single baseline file requires exactly one farm and no baseline root')
    for farm_id in config['pv_realized_irradiance_search']['farm_ids']:
        baseline = (Path(args.baseline_root) / f'farm_{farm_id}' / 'selected_parameters.json'
                    if args.baseline_root else args.baseline_selected)
        print(evaluate(config, int(farm_id), baseline), flush=True)


if __name__ == '__main__':
    main()
