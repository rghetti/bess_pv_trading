"""Plot curated fitting-day profiles for the selected physical chains."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pvlib

from ...config import load_config
from ...data import config_path, load_metadata, night_window_mask
from ..models.pv_physical import PhysicalPVModel, parameter_defaults
from ..studies.pv_realized_ensemble import selected_settings
from ..training.pv_realized_search import ranking_path


def plot_farm(config, root, farm_id):
    """Evaluate one selected chain and rank fitting days by daytime mean-prediction RMSE."""
    section = config['pv_realized_irradiance_search']
    rankings = pd.read_csv(ranking_path(root, section))
    chosen = rankings.loc[rankings.farm_id.eq(farm_id)].sort_values('rmse_percent_capacity').iloc[0]
    trial_path = root / f'farm_{farm_id}' / 'trials' / f'{int(chosen.trial_id):04d}.json'
    result = json.loads(trial_path.read_text())
    farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    settings = selected_settings(config, farm, result)
    frame = pd.read_parquet(root / f'farm_{farm_id}' / 'aligned_inputs.parquet')
    use = frame.eligible
    if section.get('selection_scope', 'validation') != 'full_period':
        use &= frame['split'].eq('train')
    frame = frame.loc[use].copy().reset_index(drop=True)
    times = pd.DatetimeIndex(frame.interval_start_utc) + pd.Timedelta(minutes=7.5)
    solar = pvlib.solarposition.get_solarposition(times, settings['location']['latitude'],
        settings['location']['longitude'], altitude=farm.elevation_m)
    zenith = solar.apparent_zenith.to_numpy()
    ghi = frame.ghi_w_m2.to_numpy(float)
    dhi = ghi * frame.diffuse_fraction.fillna(0).to_numpy(float)
    dni = np.asarray(pvlib.irradiance.dni(pd.Series(ghi), pd.Series(dhi), pd.Series(zenith),
        zenith_threshold_for_zero_dni=float(settings['low_sun_zenith_deg'])), dtype=float)
    dni = np.where(zenith >= float(settings['low_sun_zenith_deg']), 0, dni)
    dni = np.minimum(dni, float(settings['dni_limit_w_m2']))
    physical = pd.DataFrame({'solar_zenith_deg': zenith, 'solar_azimuth_deg': solar.azimuth.to_numpy(),
        'ghi_w_m2': ghi, 'dhi_w_m2': dhi, 'dni_w_m2': dni,
        'temperature_c': frame.temperature_c.to_numpy(), 'wind_speed_m_s': frame.wind_speed_m_s.to_numpy(),
        'pv_input_valid': np.isfinite(dni)}, index=pd.DatetimeIndex(frame.interval_start_utc))
    model = PhysicalPVModel(physical, farm, settings,
                            section.get('night_policy', config['policy']['night']))
    params = parameter_defaults(settings)
    frame['predicted_mw'] = model.evaluate(params)
    if not np.isfinite(frame.predicted_mw).all():
        raise ValueError('Selected chain produced nonfinite predictions on eligible rows')
    by_interval = frame.groupby('interval_start_utc').agg(
        prediction_mean=('predicted_mw', 'mean'), prediction_min=('predicted_mw', 'min'),
        prediction_max=('predicted_mw', 'max'), observed=('measured_ac_mw', 'first'),
        vintage_count=('forecast_cycle_utc', 'nunique'), ghi_interpolated=('ghi_interpolated', 'first'))
    if 'measured_ac_raw_mw' in frame:
        by_interval['observed_raw'] = frame.groupby('interval_start_utc').measured_ac_raw_mw.first()
    local = by_interval.index.tz_convert(section['delivery_timezone'])
    by_interval['day'] = local.strftime('%Y-%m-%d')
    by_interval['hour'] = local.hour + local.minute / 60
    by_interval['square_error'] = (by_interval.prediction_mean-by_interval.observed)**2
    if 'night_policy' in section:
        daylight = by_interval.loc[~night_window_mask(by_interval.index, section['night_policy'])]
    else:
        daylight = by_interval.loc[(by_interval.hour >= 6) & (by_interval.hour < 21.5)]
    daily = daylight.groupby('day').agg(daytime_intervals=('square_error', 'size'),
        mse=('square_error', 'mean'), interpolated_ghi_intervals=('ghi_interpolated', 'sum'))
    daily['rmse_percent_capacity'] = 100 * np.sqrt(daily.mse) / farm.ac_capacity_mw
    daily = daily.sort_values(['rmse_percent_capacity', 'day'])
    daily.to_csv(root / f'farm_{farm_id}' / 'training_day_ranking.csv')
    complete = daily.loc[daily.daytime_intervals.eq(62)] if 'night_policy' not in section else daily.iloc[0:0]
    selected = ([('best', day) for day in daily.head(3).index] +
                [('worst', day) for day in daily.tail(3).index] +
                [('best_complete', day) for day in complete.head(3).index])
    folder = root / f'farm_{farm_id}' / 'training_day_plots' / f'trial_{int(chosen.trial_id)}'
    folder.mkdir(parents=True, exist_ok=True)
    for kind, day in selected:
        midnight = pd.Timestamp(day, tz=section['delivery_timezone'])
        grid = pd.date_range(midnight, midnight + pd.DateOffset(days=1),
                             freq='15min', inclusive='left').tz_convert('UTC')
        # Missing and excluded quarters remain gaps rather than connected lines.
        part = by_interval.reindex(grid)
        local_index = part.index.tz_convert(section['delivery_timezone'])
        fig, ax = plt.subplots(figsize=(12, 5))
        if ('observed_raw' in part and
                (part.observed_raw.notna() & part.observed.notna() & part.observed_raw.ne(part.observed)).any()):
            ax.plot(local_index, part.observed_raw, color='tab:orange', linestyle='--',
                    label='Original measured AC power')
        ax.plot(local_index, part.observed, color='black', label='Treated AC fitting target')
        ax.plot(local_index, part.prediction_mean, color='tab:blue', label='Mean of four weather views')
        ax.fill_between(local_index, part.prediction_min, part.prediction_max, color='tab:blue', alpha=.2,
                        label='Range of four weather-view predictions')
        ax.set(title=f'Farm {farm_id}: {day} ({kind} training day); daytime RMSE '
               f'{daily.loc[day, "rmse_percent_capacity"]:.2f}% of AC capacity; '
               f'{int(daily.loc[day, "daytime_intervals"])} eligible daytime quarters',
               ylabel='AC power (MW)', xlabel='Local time (Europe/Rome)')
        ax.legend(); ax.grid(alpha=.2); fig.autofmt_xdate(); fig.tight_layout()
        fig.savefig(folder / f'{kind}_{day}.png', dpi=150)
        plt.close(fig)
    (root / f'farm_{farm_id}' / 'selected_parameters.json').write_text(json.dumps(result, indent=2)+'\n')
    print(f'Farm {farm_id}: selected trial {int(chosen.trial_id)} and saved {len(selected)} day plots', flush=True)


def main():
    """Write selected parameters and six per-farm profile plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    section = config['pv_realized_irradiance_search']
    root = config_path(config, section['output_root']) / section['run_id']
    for farm in section['farm_ids']:
        plot_farm(config, root, int(farm))


if __name__ == '__main__':
    main()
