"""Resumable retrospective physical-model search on observed irradiance."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib
from scipy.optimize import minimize

from ...config import load_config
from ...data import config_path, load_metadata
from ..models.pv_physical import PhysicalPVModel, parameter_defaults
from ..data.pv_study_inputs import farm_settings


def structural_plan(section, farm_id, tracking, settings):
    """Draw reproducible, distinct structural settings across the declared search space."""
    locations = list(section['locations'])
    tracker = list(itertools.product(section['tracker_gcr'], section['tracker_max_angle'],
                                     section['tracker_backtracking'])) if tracking else [(0.35, 60.0, True)]
    rear = [(False, 0.2, 0.8, 0.4)]
    if tracking == 3:
        rear += [(True, *item) for item in itertools.product(section['bifacial_albedo'],
                    section['bifaciality'], section['bifacial_height_to_pitch'])]
    candidates = []
    for location, model, geometry, bifacial in itertools.product(
            locations, section['transposition_models'], tracker, rear):
        enabled, albedo, bifaciality, height = bifacial
        gcr, angle, backtrack = geometry
        if enabled and (model == 'perez-driesse' or height <= gcr * np.sin(np.radians(angle)) / 2):
            continue
        candidates.append({'location': location, 'transposition': model, 'gcr': gcr,
                           'max_angle': angle, 'backtrack': backtrack,
                           'bifacial_enabled': enabled, 'albedo': albedo,
                           'bifaciality': bifaciality, 'height_to_pitch': height})
    rng = np.random.default_rng(int(section['seed']) + farm_id)
    selection = rng.choice(len(candidates), size=min(len(candidates),
                           int(section['structural_trials_per_farm'])), replace=False)
    reference = {'location': 'milan', 'transposition': settings['transposition_model'],
                 'gcr': settings['tracker']['gcr'], 'max_angle': settings['tracker']['max_angle'],
                 'backtrack': settings['tracker']['backtrack'],
                 'bifacial_enabled': settings['bifacial']['enabled'], 'albedo': settings['albedo'],
                 'bifaciality': settings['bifacial']['bifaciality'],
                 'height_to_pitch': settings['bifacial']['height_to_pitch_ratio'], 'loss': 'fixed'}
    return [reference] + [dict(candidates[int(i)], loss=loss) for i in selection for loss in section['losses']]


def run_trial(config, farm_id, trial_id, candidate, root):
    """Fit one physical chain and write a resumable result with validation metrics."""
    section = config['pv_realized_irradiance_search']
    destination = root / f'farm_{farm_id}' / 'trials' / f'{trial_id:04d}.json'
    if destination.exists():
        return json.loads(destination.read_text())
    frame = pd.read_parquet(root / f'farm_{farm_id}' / 'aligned_inputs.parquet')
    frame = frame.loc[frame.eligible].copy().reset_index(drop=True)
    metadata = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    settings = farm_settings(config, metadata)
    settings['transposition_model'] = candidate['transposition']
    settings['tracker'].update(gcr=candidate['gcr'], max_angle=candidate['max_angle'],
                               backtrack=candidate['backtrack'])
    settings['albedo'] = candidate['albedo']
    settings['bifacial'].update(enabled=candidate['bifacial_enabled'],
        bifaciality=candidate['bifaciality'], height_to_pitch_ratio=candidate['height_to_pitch'])
    settings['location'].update(latitude=section['locations'][candidate['location']][0],
                                longitude=section['locations'][candidate['location']][1])
    times = pd.DatetimeIndex(frame.interval_start_utc) + pd.Timedelta(minutes=7.5)
    solar = pvlib.solarposition.get_solarposition(times, settings['location']['latitude'],
        settings['location']['longitude'], altitude=metadata.elevation_m)
    zenith = solar.apparent_zenith.to_numpy()
    ghi = frame.ghi_w_m2.to_numpy(float)
    fraction = frame.diffuse_fraction.fillna(0).to_numpy(float)
    dhi = ghi * fraction
    dni = np.asarray(pvlib.irradiance.dni(pd.Series(ghi), pd.Series(dhi), pd.Series(zenith),
        zenith_threshold_for_zero_dni=float(settings['low_sun_zenith_deg'])), dtype=float)
    dni = np.where(zenith >= float(settings['low_sun_zenith_deg']), 0, dni)
    dni = np.minimum(dni, float(settings['dni_limit_w_m2']))
    physical = pd.DataFrame({'solar_zenith_deg': zenith, 'solar_azimuth_deg': solar.azimuth.to_numpy(),
        'ghi_w_m2': ghi, 'dhi_w_m2': dhi, 'dni_w_m2': dni,
        'temperature_c': frame.temperature_c.to_numpy(), 'wind_speed_m_s': frame.wind_speed_m_s.to_numpy(),
        'pv_input_valid': np.isfinite(dni)}, index=pd.DatetimeIndex(frame.interval_start_utc))
    engine = PhysicalPVModel(physical, metadata, settings, config['policy']['night'])
    defaults = parameter_defaults(settings)
    defaults.update(gcr=float(candidate['gcr']), max_angle=float(candidate['max_angle']))
    names = ['coefficient', 'u0', 'u1', 'gamma']
    bounds = np.array([section['bounds'][name] for name in names], dtype=float)
    lower, upper = bounds[:, 0], bounds[:, 1]
    width = upper - lower
    target = frame.measured_ac_mw.to_numpy(float) / metadata.ac_capacity_mw
    train = frame['split'].eq('train').to_numpy()
    valid = frame['split'].eq('validation').to_numpy()
    weights = frame.weight.to_numpy(float)
    if not train.any() or not valid.any():
        raise ValueError('Missing eligible training or validation observations')

    def prediction(vector):
        params = dict(defaults, **dict(zip(names[:3], vector[:3])))
        engine.settings['dc']['gamma_pdc_per_c'] = float(vector[3])
        return engine.evaluate(params) / metadata.ac_capacity_mw

    def objective(unit_vector):
        result = prediction(lower + unit_vector * width)
        residual = result[train] - target[train]
        if not np.isfinite(residual).all():
            return 1e20
        if candidate['loss'] == 'mse':
            loss = residual ** 2
        else:
            delta = float(section['huber_delta'])
            absolute = np.abs(residual)
            loss = np.where(absolute <= delta, residual ** 2, 2 * delta * absolute - delta ** 2)
        return float(np.average(loss, weights=weights[train]))

    initial = np.array([defaults['coefficient'], defaults['u0'], defaults['u1'],
                        settings['dc']['gamma_pdc_per_c']])
    rng = np.random.default_rng(int(section['seed']) + farm_id * 10000 + trial_id)
    starts = [np.clip((initial-lower)/width, 0, 1)]
    starts += [rng.uniform(0, 1, len(names)) for _ in range(int(section['multistarts'])-1)]
    solutions = []
    for start in starts:
        if candidate['loss'] == 'fixed':
            solutions.append((objective(start), start, True, 'unchanged configured physical baseline'))
            break
        fit = minimize(objective, start, method='L-BFGS-B', bounds=[(0, 1)] * len(names),
                       options={'maxfun': int(section['optimization_max_evaluations'])})
        solutions.append((float(fit.fun), fit.x, bool(fit.success), str(fit.message)))
    best = min(solutions, key=lambda x: x[0])
    vector = lower + best[1] * width
    predicted = prediction(vector)
    residual = predicted[valid] - target[valid]
    if not np.isfinite(residual).all():
        raise ValueError('Nonfinite validation prediction')
    daytime = pd.DatetimeIndex(frame.interval_start_utc).tz_convert(section['delivery_timezone'])
    hours = daytime.hour + daytime.minute / 60
    masks = {'all_hours': valid, 'daytime': valid & (hours >= 6) & (hours < 21.5)}
    metrics = {}
    for name, mask in masks.items():
        error = predicted[mask] - target[mask]
        w = weights[mask]
        metrics[name] = {'rows': int(mask.sum()), 'mae_percent_capacity': float(100*np.average(abs(error), weights=w)),
                         'rmse_percent_capacity': float(100*np.sqrt(np.average(error**2, weights=w))),
                         'bias_percent_capacity': float(100*np.average(error, weights=w))}
    result = {'farm_id': farm_id, 'trial_id': trial_id, 'candidate': candidate,
              'parameters': dict(zip(names, map(float, vector))), 'training_loss': best[0],
              'converged': best[2], 'optimizer_message': best[3], 'metrics': metrics,
              'bound_hits': [name for name, x in zip(names, best[1]) if x < 0.0001 or x > 0.9999]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(destination)
    return result


def main():
    """Save the exact trial list, then resume or execute the approved search."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config)
    section = config['pv_realized_irradiance_search']
    root = config_path(config, section['output_root']) / section['run_id']
    root.mkdir(parents=True, exist_ok=True)
    farms = load_metadata(config_path(config, config['inputs']['metadata_parquet']), section['farm_ids'])
    jobs = [(farm, i, candidate) for farm in section['farm_ids']
            for i, candidate in enumerate(structural_plan(section, farm, farms[farm].tracking_type_id,
                                                          farm_settings(config, farms[farm])))]
    fingerprints = {}
    sources = [root / f'farm_{farm}' / 'aligned_inputs.parquet' for farm in section['farm_ids']]
    sources += [Path(__file__), Path(__file__).resolve().parents[1] / 'data/pv_realized_irradiance.py',
                Path(__file__).resolve().parents[1] / 'models/pv_physical.py',
                config_path(config, config['inputs']['metadata_parquet'])]
    for path in sources:
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        fingerprints[str(path)] = digest.hexdigest()
    plan = {'section': section, 'input_and_code_sha256': fingerprints,
            'jobs': [{'farm': farm, 'trial': i, **candidate} for farm, i, candidate in jobs]}
    text = json.dumps(plan, sort_keys=True, indent=2, allow_nan=False) + '\n'
    plan_path = root / 'trial_plan.json'
    if plan_path.exists() and plan_path.read_text() != text:
        raise ValueError('Search plan changed; use a new run ID')
    plan_path.write_text(text)
    print(f'{len(jobs)} trials; up to {section["workers"]} workers; plan: {plan_path}', flush=True)
    if args.plan_only:
        return
    results = []
    with ProcessPoolExecutor(max_workers=int(section['workers'])) as pool:
        futures = {pool.submit(run_trial, config, farm, i, candidate, root): (farm, i)
                   for farm, i, candidate in jobs}
        for future in as_completed(futures):
            farm, i = futures[future]
            try:
                result = future.result()
                results.append(result)
                print(f'Completed {len(results)}/{len(jobs)}: farm {farm} trial {i}', flush=True)
            except Exception as error:
                print(f'FAILED farm {farm} trial {i}: {error}', flush=True)
                for pending in futures:
                    pending.cancel()
                raise
    rows = [dict(farm_id=x['farm_id'], trial_id=x['trial_id'], loss=x['candidate']['loss'],
                 location=x['candidate']['location'], transposition=x['candidate']['transposition'],
                 bifacial=x['candidate']['bifacial_enabled'], **x['metrics']['daytime']) for x in results]
    pd.DataFrame(rows).sort_values(['farm_id', 'rmse_percent_capacity']).to_csv(root / 'validation_ranking.csv', index=False)


if __name__ == '__main__':
    main()
