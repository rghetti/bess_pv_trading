"""Score every old/new physical-setting combination at fixed temperature."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import itertools
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path, load_metadata, night_window_mask
from bess_pv_trading.forecasting.models import pv_physical
from bess_pv_trading.forecasting.plotting import pv_review_predictions
from bess_pv_trading.forecasting.data.pv_study_inputs import farm_settings
from bess_pv_trading.forecasting.plotting.pv_review_predictions import historical_predictions


def sha(path):
    """Hash cached inputs and candidate artifacts for resumable comparison."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def set_path(settings, path, value):
    """Set one declared nested physical setting without changing other fields."""
    part = settings
    for key in path[:-1]:
        part = part[key]
    part[path[-1]] = value


def factorial_plan(config, farm, result):
    """Enumerate every distinct old/new parameter choice and identify exact duplicates."""
    if int(result['farm_id']) != int(farm.farm_id):
        raise ValueError('Candidate belongs to another farm')
    base = farm_settings(config, farm)
    choice, params = result['candidate'], result['parameters']
    latitude, longitude = config['pv_realized_irradiance_search']['locations'][choice['location']]
    possibilities = [
        ('latitude', ('location', 'latitude'), base['location']['latitude'], latitude),
        ('longitude', ('location', 'longitude'), base['location']['longitude'], longitude),
        ('gcr', ('tracker', 'gcr'), base['tracker']['gcr'], choice['gcr']),
        ('max_angle', ('tracker', 'max_angle'), base['tracker']['max_angle'], choice['max_angle']),
        ('backtrack', ('tracker', 'backtrack'), base['tracker']['backtrack'], choice['backtrack']),
        ('transposition', ('transposition_model',), base['transposition_model'], choice['transposition']),
        ('scale', ('output', 'calibrated_coefficient'), base['output']['calibrated_coefficient'], params['coefficient']),
        ('albedo', ('albedo',), base['albedo'], choice['albedo']),
        ('rear_enabled', ('bifacial', 'enabled'), base['bifacial']['enabled'], choice['bifacial_enabled']),
        ('bifaciality', ('bifacial', 'bifaciality'), base['bifacial']['bifaciality'], choice['bifaciality']),
        ('height_to_pitch', ('bifacial', 'height_to_pitch_ratio'), base['bifacial']['height_to_pitch_ratio'], choice['height_to_pitch']),
    ]
    # Latitude and longitude form one real location; never combine coordinates from different cities.
    old_location = (possibilities[0][2], possibilities[1][2])
    new_location = (latitude, longitude)
    axes = [('location', None, old_location, new_location)] if old_location != new_location else []
    axes += [item for item in possibilities[2:] if item[2] != item[3]]
    all_cases, unique, keys = [], [], {}
    for index, bits in enumerate(itertools.product([0, 1], repeat=len(axes))):
        settings = deepcopy(base)
        labels = {}
        for (name, path, old, new), bit in zip(axes, bits):
            value = new if bit else old
            labels[name] = 'new' if bit else 'old'
            if name == 'location':
                settings['location'].update(latitude=value[0], longitude=value[1])
            else:
                set_path(settings, path, value)
        # Both old and new output settings are calibrated scales, not efficiencies.
        settings['output']['method'] = 'calibrated_scale'
        canonical = deepcopy(settings)
        if not canonical['bifacial']['enabled']:
            # These two fields are unused by the front-only forward model.
            canonical['bifacial']['bifaciality'] = None
            canonical['bifacial']['height_to_pitch_ratio'] = None
        key = json.dumps(canonical, sort_keys=True, default=str)
        if key not in keys:
            keys[key] = len(unique)
            unique.append({'unique_id': keys[key], 'settings': settings})
        all_cases.append({'case_id': index, 'choices': labels, 'unique_id': keys[key]})
    return {'farm_id': farm.farm_id, 'axes': [{'name': a[0], 'old': a[2], 'new': a[3]} for a in axes],
            'cases': all_cases, 'unique_models': unique,
            'fixed_temperature': {'u0': base['temperature']['u0'], 'u1': base['temperature']['u1'],
                                  'gamma': base['dc']['gamma_pdc_per_c']}}


def run_one(config, farm_id, source, output, job):
    """Evaluate and atomically checkpoint one unique physical forward model."""
    directory = output / f'farm_{farm_id}' / 'checkpoints'
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{job['unique_id']:04d}.parquet"
    checksum = path.with_suffix('.sha256')
    if path.exists() and checksum.exists():
        if sha(path) != checksum.read_text().strip():
            raise ValueError(f'Forecast checkpoint checksum mismatch: {path}')
        return job['unique_id']
    farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    frame = pd.read_parquet(source)
    frame = frame.loc[frame.eligible].copy()
    section = config['pv_realized_irradiance_search']
    policy = section.get('night_policy', config['policy']['night'])
    result = historical_predictions(frame, farm, job['settings'], policy).reset_index()
    temporary = path.with_suffix('.tmp.parquet')
    result.to_parquet(temporary, index=False)
    temporary.replace(path)
    checksum.write_text(sha(path) + '\n')
    return job['unique_id']


def report(config, farm_id, source, output, plan):
    """Score every enumerated case on the common physical quarter-hour population."""
    directory = output / f'farm_{farm_id}'
    section = config['pv_realized_irradiance_search']
    farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    frame = pd.read_parquet(source)
    frame = frame.loc[frame.eligible]
    base = frame.groupby('interval_start_utc').agg(target_mw=('measured_ac_mw', 'first'),
                                                  ghi_w_m2=('ghi_w_m2', 'first'))
    series = []
    for model in plan['unique_models']:
        i = model['unique_id']
        part = pd.read_parquet(directory / 'checkpoints' / f'{i:04d}.parquet').set_index('interval_start_utc')
        series.append(part.historical_ghi_prediction_mw.rename(str(i)))
    data = base.join(pd.concat(series, axis=1)).dropna()
    policy = section.get('night_policy', config['policy']['night'])
    local = data.index.tz_convert(section['delivery_timezone'])
    data['season'] = np.select([local.month.isin([12, 1, 2]), local.month.isin([3, 4, 5]),
                                local.month.isin([6, 7, 8])], ['winter', 'spring', 'summer'], default='autumn')
    subsets = [('all_hours', data), ('non_night', data.loc[~night_window_mask(data.index, policy)])]
    subsets += [(f'ghi_above_20_{season}', part) for season, part in
                data.loc[data.ghi_w_m2.gt(20)].groupby('season', sort=True)]
    rows = []
    for scope, part in subsets:
        if part.empty:
            continue
        for case in plan['cases']:
            error = part[str(case['unique_id'])].to_numpy(float) - part.target_mw.to_numpy(float)
            rows.append({'farm_id': farm_id, 'scope': scope, 'case_id': case['case_id'],
                'unique_id': case['unique_id'], **case['choices'], 'quarters': len(error),
                'mae_mw': float(np.mean(abs(error))), 'rmse_mw': float(np.sqrt(np.mean(error**2))),
                'bias_mw': float(np.mean(error)),
                'mae_percent_capacity': 100 * float(np.mean(abs(error))) / farm.ac_capacity_mw,
                'rmse_percent_capacity': 100 * float(np.sqrt(np.mean(error**2))) / farm.ac_capacity_mw,
                'bias_percent_capacity': 100 * float(np.mean(error)) / farm.ac_capacity_mw})
    scores = pd.DataFrame(rows)
    scores.to_csv(directory / 'metrics.csv', index=False)
    ranked = scores.loc[scores.scope.eq('non_night')].sort_values('rmse_percent_capacity')
    ranked.to_csv(directory / 'ranking.csv', index=False)
    ax = ranked.head(15).set_index('case_id').rmse_percent_capacity.plot.bar(
        figsize=(10, 5), ylabel='RMSE (% installed AC capacity)',
        title=f'Farm {farm_id}: best paired in-sample combinations')
    ax.figure.tight_layout()
    ax.figure.savefig(directory / 'best_combination_rmse.png', dpi=150)
    plt.close(ax.figure)
    return scores


def main():
    """Print the full plan or execute its resumable, bounded-CPU evaluation."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--candidate-root', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    section = config['pv_realized_irradiance_search']
    output = Path(args.output)
    source_root = config_path(config, section['output_root']) / section['run_id']
    candidates = Path(args.candidate_root)
    plans, source_hashes = {}, {}
    for farm_id in section['farm_ids']:
        farm_id = int(farm_id)
        farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
        candidate = candidates / f'farm_{farm_id}' / 'selected_parameters.json'
        source = source_root / f'farm_{farm_id}' / 'aligned_inputs.parquet'
        plans[str(farm_id)] = factorial_plan(config, farm, json.loads(candidate.read_text()))
        source_hashes[str(candidate)] = sha(candidate)
        source_hashes[str(source)] = sha(source)
    metadata_path = config_path(config, config['inputs']['metadata_parquet'])
    source_hashes[str(metadata_path)] = sha(metadata_path)
    summary = {farm: {'axes': plan['axes'], 'combinations': len(plan['cases']),
        'unique_forward_models': len(plan['unique_models']), 'fixed_temperature': plan['fixed_temperature']}
        for farm, plan in plans.items()}
    if args.plan_only:
        print(json.dumps(summary, indent=2, default=str))
        return
    workers = int(section['workers'])
    if not 1 <= workers <= 42:
        raise ValueError('Worker count must be between 1 and 42')
    manifest = {'configuration': section, 'night_policy': section.get('night_policy', config['policy']['night']),
        'quality_csv_sha256': sha(config_path(config, config['pv_data_quality']['exclusions_csv'])),
        'sources': source_hashes, 'code_sha256': sha(__file__),
        'dependencies_sha256': {'physical_model': sha(pv_physical.__file__),
                                'prediction_alignment': sha(pv_review_predictions.__file__)},
        'plans_sha256': hashlib.sha256(json.dumps(plans, sort_keys=True, default=str).encode()).hexdigest(),
        'summary': summary,
        'interpretation': 'retrospective full-period in-sample; no physical fitting or independent test',
        'weather': 'historical satellite-derived GHI/diffuse and four forecast temperature/wind views',
        'location': 'solar position only; source weather remains unchanged',
        'deduplication': 'bifaciality and rear height do not affect the front-only model'}
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'study_manifest.json'
    serialized = json.dumps(manifest, sort_keys=True, indent=2, default=str) + '\n'
    if manifest_path.exists() and manifest_path.read_text() != serialized:
        raise ValueError('Study inputs, settings or code changed; use a fresh output directory')
    manifest_path.write_text(serialized)
    jobs = [(int(f), source_root / f'farm_{f}' / 'aligned_inputs.parquet', job)
            for f, plan in plans.items() for job in plan['unique_models']]
    print(f'{len(jobs)} unique forward models, up to {workers} workers', flush=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(run_one, config, farm, source, output, job): (farm, job['unique_id'])
                   for farm, source, job in jobs}
        for index, future in enumerate(as_completed(futures), 1):
            farm, unique_id = futures[future]
            future.result()
            print(f'Completed {index}/{len(jobs)}: farm {farm}, unique model {unique_id}', flush=True)
    results = [report(config, int(farm), source_root / f'farm_{farm}' / 'aligned_inputs.parquet', output, plan)
               for farm, plan in plans.items()]
    pd.concat(results, ignore_index=True).to_csv(output / 'combined_metrics.csv', index=False)


if __name__ == '__main__':
    main()
