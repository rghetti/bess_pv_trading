"""Evaluate selected retrospective physical chains on saved weather ensembles."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

from ...config import load_config
from ...data import ForecastBundle, config_path, load_metadata
from ..models.pv import forecast_pv
from ..evaluation.pv_probabilistic_metrics import fit_calibrator, score_frame
from ..plotting.pv_probabilistic_reporting import apply_calibrator
from ..data.pv_study_inputs import farm_settings


def selected_settings(config, farm, result):
    """Translate a saved candidate into the existing operational PV engine."""
    section = config['pv_realized_irradiance_search']
    settings = farm_settings(config, farm)
    candidate, params = result['candidate'], result['parameters']
    settings['location'].update(latitude=section['locations'][candidate['location']][0],
                                longitude=section['locations'][candidate['location']][1])
    settings['transposition_model'] = candidate['transposition']
    settings['tracker'].update(gcr=candidate['gcr'], max_angle=candidate['max_angle'],
                               backtrack=candidate['backtrack'])
    settings['albedo'] = candidate['albedo']
    settings['bifacial'].update(enabled=candidate['bifacial_enabled'],
        bifaciality=candidate['bifaciality'], height_to_pitch_ratio=candidate['height_to_pitch'])
    settings['temperature'].update(u0=params['u0'], u1=params['u1'])
    settings['dc']['gamma_pdc_per_c'] = params['gamma']
    settings['output'].update(method='calibrated_scale', calibrated_coefficient=params['coefficient'])
    return settings


def convert(inputs, config, farm, settings):
    """Apply the same physical chain to every saved weather member or deterministic view."""
    outputs = []
    for (_, cycle_value), part in inputs.groupby(['delivery_day', 'forecast_cycle_utc'], sort=True):
        part = part.copy().set_index('interval_start_utc', drop=False)
        members = tuple(map(str, part.member.drop_duplicates()))
        available = pd.Timestamp(part.forecast_available_at_utc.iloc[0])
        bundle = ForecastBundle(farm.farm_id, pd.Timestamp(cycle_value), available, part,
            {name: '' for name in config['forecast_mapping']}, members, 'saved_weather',
            {'decision_cutoff_utc': available.isoformat()})
        result = forecast_pv(bundle, farm, settings, config['policy']['night']).values
        outputs.append(result[['interval_start_utc', 'member', 'pv_ac_mw']].reset_index(drop=True))
    return pd.concat(outputs, ignore_index=True)


def run_farm(config, ensemble_config, root, farm_id):
    """Fit January calibration on converted ensemble summaries and score August forecasts."""
    rank = pd.read_csv(root / 'validation_ranking.csv')
    chosen = rank.loc[rank.farm_id.eq(farm_id)].sort_values('rmse_percent_capacity').iloc[0]
    trial_path = root / f'farm_{farm_id}' / 'trials' / f'{int(chosen.trial_id):04d}.json'
    result = json.loads(trial_path.read_text())
    farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    source = (config_path(ensemble_config, ensemble_config['pv_probabilistic_study']['output_root']) /
              ensemble_config['pv_probabilistic_study']['run_id'] / f'farm_{farm_id}')
    ens = pd.read_parquet(source / 'ensemble_inputs.parquet')
    det = pd.read_parquet(source / 'deterministic_inputs.parquet')
    ids = sorted(map(str, ens.member.unique()), key=int)
    if len(ids) != 50:
        raise ValueError('Expected all 50 ensemble members')
    settings = selected_settings(config, farm, result)
    print(f'Farm {farm_id}: convert 50 forecast members with trial {int(chosen.trial_id)}', flush=True)
    converted = convert(ens, config, farm, settings)
    wide = converted.pivot(index='interval_start_utc', columns='member', values='pv_ac_mw').reindex(columns=ids)
    deterministic = convert(det, config, farm, settings)
    if deterministic.interval_start_utc.duplicated().any():
        raise ValueError('Duplicate deterministic forecast timestamps')
    base = det[['interval_start_utc', 'delivery_day']].drop_duplicates('interval_start_utc').set_index('interval_start_utc')
    base = base.join(deterministic.set_index('interval_start_utc').pv_ac_mw.rename('deterministic_prediction')).join(wide)
    source_power = (config_path(config, config['pv_realized_irradiance_search']['curated_power_root']) /
                    f'farm_{farm_id}' / 'parquet' / 'corpus.parquet')
    power = pd.read_parquet(source_power, columns=['interval_start_utc', 'measured_ac_mw', 'eligible'])
    power = power.loc[power.eligible]
    if power.groupby('interval_start_utc').measured_ac_mw.nunique(dropna=False).gt(1).any():
        raise ValueError('Curated power differs between views')
    base = base.join(power.groupby('interval_start_utc').measured_ac_mw.first())
    base['eligible'] = (np.isfinite(base.measured_ac_mw) & np.isfinite(base.deterministic_prediction)
                        & np.isfinite(base[ids]).all(axis=1))
    base = base.reset_index()
    levels = list(map(float, ensemble_config['pv_probabilistic_study']['quantiles']))
    jan = base.eligible & base.delivery_day.str.startswith('2026-01')
    calibrator = fit_calibrator(base.loc[jan, ids].to_numpy(float), base.loc[jan, 'measured_ac_mw'],
        base.loc[jan, 'interval_start_utc'], farm.ac_capacity_mw, levels,
        ensemble_config['pv_probabilistic_study']['calibration_alpha'])
    base, adjustments = apply_calibrator(base, ids, calibrator, levels, config['policy']['night'])
    joblib.dump(calibrator, root / f'farm_{farm_id}' / 'forecast_ensemble_quantile_calibrator.joblib')
    base.to_parquet(root / f'farm_{farm_id}' / 'forecast_ensemble_predictions.parquet', index=False)
    validation = base.delivery_day.between('2026-08-20', '2026-08-31')
    local = pd.DatetimeIndex(base.interval_start_utc).tz_convert('Europe/Rome')
    hours = local.hour + local.minute / 60
    tables = []
    for name, mask in [('all_hours', validation), ('daytime', validation & (hours >= 6) & (hours < 21.5))]:
        tables.append(score_frame(base.loc[mask], ids, levels, farm.ac_capacity_mw, name))
    summary = pd.concat(tables, ignore_index=True)
    summary.insert(0, 'farm_id', farm_id)
    summary.to_csv(root / f'farm_{farm_id}' / 'forecast_ensemble_metrics.csv', index=False)
    manifest = {'farm_id': farm_id, 'selected_trial': str(trial_path), 'january_calibration_intervals': int(jan.sum()),
                'calibration_adjustments': adjustments,
                'caveat': 'physical fit uses 2025-July 2026 and August model selection; January calibration is retrospective/in-sample for the physical fit',
                'validation_dates': '2026-08-20 through 2026-08-31',
                'ensemble_cycle': 'preceding-day 00 UTC', 'tail_assumption': 'constant beyond p05 and p95'}
    (root / f'farm_{farm_id}' / 'forecast_ensemble_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    return summary


def main():
    """Run the forecast ensemble evaluation after physical search completes."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ensemble-config', default='configs/pv_experiments/probabilistic_january_august_2026.yaml')
    args = parser.parse_args()
    config = load_config(args.config)
    ensemble_config = load_config(args.ensemble_config)
    section = config['pv_realized_irradiance_search']
    root = config_path(config, section['output_root']) / section['run_id']
    scores = [run_farm(config, ensemble_config, root, int(farm)) for farm in section['farm_ids']]
    pd.concat(scores, ignore_index=True).to_csv(root / 'forecast_ensemble_metrics.csv', index=False)


if __name__ == '__main__':
    main()
