"""Compare saved physical and learned forecast distributions on matched intervals."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import load_config
from ..data import config_path, load_metadata
from .pv_probabilistic_metrics import fit_calibrator, score_frame
from .pv_probabilistic_reporting import apply_calibrator


def main():
    """Score previous saved models and the new chain on common curated observations."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--ensemble-config', default='configs/pv_experiments/probabilistic_january_august_2026.yaml')
    args = parser.parse_args()
    config, previous = load_config(args.config), load_config(args.ensemble_config)
    section = config['pv_realized_irradiance_search']
    root = config_path(config, section['output_root']) / section['run_id']
    oldroot = config_path(previous, previous['pv_probabilistic_study']['output_root']) / previous['pv_probabilistic_study']['run_id']
    rankings = pd.read_csv('docs/results/probabilistic_pv_january_august_2026/preceding_day/validation_rankings.csv')
    ids = list(map(str, range(1, 51)))
    levels = list(map(float, previous['pv_probabilistic_study']['quantiles']))
    qcols = [f'calibrated_q{round(100*x):02d}' for x in levels]
    collected = []
    for farm_id in section['farm_ids']:
        farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
        selection = rankings.loc[(rankings.farm_id == farm_id) & rankings.objective.eq('crps_mw') &
                                  rankings.method.eq('calibrated_quantiles') & rankings['rank'].eq(1)].iloc[0]
        paths = {'new_physical': root / f'farm_{farm_id}' / 'forecast_ensemble_predictions.parquet',
                 'previous_physical': oldroot / f'farm_{farm_id}' / 'pvlib_paired_predictions.parquet',
                 'previous_learned': Path(selection.trial_path) / 'predictions.parquet'}
        frames = {name: pd.read_parquet(path).set_index('interval_start_utc') for name, path in paths.items()}
        new = frames['new_physical']
        old = frames['previous_physical'].copy()
        calibration_times = old.index.intersection(new.index)
        january = old.loc[calibration_times]
        new_january = new.loc[calibration_times]
        training = (january.delivery_day.str.startswith('2026-01') & january.eligible & new_january.eligible &
                    np.isfinite(january.measured_ac_mw) & np.isfinite(january[ids]).all(axis=1))
        eligible_january = january.loc[training]
        if not np.allclose(eligible_january.measured_ac_mw, new_january.loc[training].measured_ac_mw):
            raise ValueError('January calibration targets differ')
        calibrator = fit_calibrator(eligible_january[ids].to_numpy(float), eligible_january.measured_ac_mw,
            eligible_january.index, farm.ac_capacity_mw, levels,
            previous['pv_probabilistic_study']['calibration_alpha'])
        recalibrated, _ = apply_calibrator(old.reset_index(), ids, calibrator, levels, config['policy']['night'])
        frames['previous_physical_january_calibration'] = recalibrated.set_index('interval_start_utc')
        common = None
        for name, frame in frames.items():
            if frame.index.has_duplicates:
                raise ValueError(f'Duplicate forecast interval in {name}')
            mask = (frame.eligible & np.isfinite(frame.measured_ac_mw) &
                    np.isfinite(frame.deterministic_prediction) & np.isfinite(frame[ids+qcols]).all(axis=1) &
                    frame.delivery_day.between('2026-08-20', '2026-08-31'))
            common = frame.index[mask] if common is None else common.intersection(frame.index[mask])
        targets = frames['new_physical'].loc[common, 'measured_ac_mw']
        for name, frame in frames.items():
            if not np.allclose(frame.loc[common, 'measured_ac_mw'], targets, equal_nan=True):
                raise ValueError('Matched forecasts do not share identical observed power')
            part = frame.loc[common].copy().reset_index()
            local = pd.DatetimeIndex(part.interval_start_utc).tz_convert('Europe/Rome')
            hour = local.hour + local.minute / 60
            for label, mask in [('all_hours', np.ones(len(part), dtype=bool)),
                                ('daytime', (hour >= 6) & (hour < 21.5))]:
                scored = score_frame(part.loc[mask], ids, levels, farm.ac_capacity_mw, label)
                scored.insert(0, 'forecast_model', name)
                scored.insert(0, 'farm_id', farm_id)
                collected.append(scored)
        print(f'Farm {farm_id}: {len(common)} matched validation quarters', flush=True)
    pd.concat(collected, ignore_index=True).to_csv(root / 'paired_forecast_metrics.csv', index=False)
    (root / 'paired_forecast_interpretation.txt').write_text(
        'Exploratory matched comparison, August 20–31 2026. New physical parameters were selected using August daytime results; '
        'its quantile calibrator uses January training examples from a physical fit that included January. '
        'The previous_physical_january_calibration model uses the same January calibration dates as the new physical model. '
        'Previous physical and learned calibrators use the original January and August out-of-fold calibration period. '
        'Previous learned models and new physical selection were both validation-selected; first-day retrospective caveats remain. '
        'Different calibration histories mean this is not a controlled head-to-head comparison of the underlying conversions.\n')


if __name__ == '__main__':
    main()
