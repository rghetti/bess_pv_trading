"""Overlay saved physical PV model predictions on existing power-review plots."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pvlib

from ...config import load_config
from ...data import config_path, load_metadata, open_store
from ..models.pv import forecast_pv, local_delivery_starts
from ..plotting.pv_near_flat_review import plot_day
from ..models.pv_physical import PhysicalPVModel, parameter_defaults
from ..data.pv_probabilistic_data import align_cycle
from ..studies.pv_realized_ensemble import selected_settings


def _digest(path):
    """Fingerprint saved model and inputs without modifying them."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        # Fingerprint saved model and inputs without modifying them for each `block` in `iter(lambda: stream.read(1024 * 1024), b'')`.
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def historical_predictions(frame, farm, settings, night_policy):
    """Apply the fitted physical chain to historical GHI and four weather views."""
    starts = pd.DatetimeIndex(frame.interval_start_utc).as_unit('ns')
    times = starts + pd.Timedelta(minutes=7.5)
    solar = pvlib.solarposition.get_solarposition(times, settings['location']['latitude'],
        settings['location']['longitude'], altitude=farm.elevation_m)
    zenith = solar.apparent_zenith.to_numpy()
    ghi = frame.ghi_w_m2.to_numpy(float)
    fraction = frame.diffuse_fraction.to_numpy(float)
    valid = np.isfinite(ghi) & (ghi >= 0) & ((ghi == 0) | (np.isfinite(fraction) & (fraction >= 0) & (fraction <= 1)))
    dhi = ghi * np.where(ghi == 0, 0, fraction)
    dni = np.asarray(pvlib.irradiance.dni(pd.Series(ghi), pd.Series(dhi), pd.Series(zenith),
        zenith_threshold_for_zero_dni=float(settings['low_sun_zenith_deg'])), dtype=float)
    dni = np.where(zenith >= float(settings['low_sun_zenith_deg']), 0, dni)
    # Enforce the input contract: Historical overlay requires the configured cap-and-flag DNI policy.
    if settings['dni_limit_action'] != 'cap_and_flag':
        raise ValueError('Historical overlay requires the configured cap-and-flag DNI policy')
    dni = np.minimum(dni, float(settings['dni_limit_w_m2']))
    physical = pd.DataFrame({'solar_zenith_deg': zenith, 'solar_azimuth_deg': solar.azimuth.to_numpy(),
        'ghi_w_m2': ghi, 'dhi_w_m2': dhi, 'dni_w_m2': dni,
        'temperature_c': frame.temperature_c.to_numpy(float), 'wind_speed_m_s': frame.wind_speed_m_s.to_numpy(float),
        'pv_input_valid': valid & np.isfinite(dni)}, index=starts)
    predictions = PhysicalPVModel(physical, farm, settings, night_policy).evaluate(parameter_defaults(settings))
    values = pd.DataFrame({'interval_start_utc': starts, 'prediction': predictions})
    result = values.groupby('interval_start_utc').prediction.agg(['mean', 'count'])
    # Do not silently replace a missing weather view by a different weighting.
    result['historical_ghi_prediction_mw'] = result['mean'].where(result['count'].eq(4))
    return result[['historical_ghi_prediction_mw', 'count']].rename(columns={'count': 'historical_weather_views'})


def forecast_predictions(store, farm, settings, config, days):
    """Predict each local delivery day using its preceding-day 00 UTC weather cycle."""
    rows = []
    # Apply forecast pv for `day` from `days`.
    for day in days:
        starts = local_delivery_starts(day).as_unit('ns')
        cycle = (pd.Timestamp(day) - pd.Timedelta(days=1)).tz_localize('UTC')
        bundle = align_cycle(store, farm.farm_id, cycle, starts, config, ensemble=False)
        converted = forecast_pv(bundle, farm, settings, config['policy']['night']).values
        rows.append(pd.DataFrame({'interval_start_utc': starts, 'forecast_weather_prediction_mw':
                                  converted.pv_ac_mw.to_numpy(float), 'forecast_cycle_utc': cycle,
                                  'forecast_available_utc': bundle.available_at}))
    return pd.concat(rows, ignore_index=True).set_index('interval_start_utc')


def run_farm(config, store, farm_id):
    """Preserve original plots, add both model curves, and save the values for review."""
    section, review = config['pv_review_predictions'], config['pv_near_flat_review']
    review_root = config_path(config, section['review_root']) / f'farm_{farm_id}'
    search_root = config_path(config, section['physical_search_root']) / f'farm_{farm_id}'
    model_path, inputs_path = search_root / 'selected_parameters.json', search_root / 'aligned_inputs.parquet'
    result = json.loads(model_path.read_text())
    farm = load_metadata(config_path(config, config['inputs']['metadata_parquet']), [farm_id])[farm_id]
    settings = selected_settings(config, farm, result)
    frame = pd.read_csv(review_root / 'review_intervals.csv', parse_dates=['interval_start_utc'],
                        keep_default_na=False, na_values={'power_kw': [''], 'ghi_w_m2': ['']})
    days = sorted(frame.local_day.unique())
    inputs = pd.read_parquet(inputs_path)
    local_day = inputs.interval_start_utc.dt.tz_convert(review['timezone']).dt.strftime('%Y-%m-%d')
    inputs = inputs.loc[local_day.isin(days)].copy()
    observed = historical_predictions(inputs, farm, settings, config['policy']['night'])
    forecast = forecast_predictions(store, farm, settings, config, days)
    joined = observed.join(forecast, how='outer').sort_index()
    output = review_root / 'model_overlay_predictions.parquet'
    joined.reset_index().to_parquet(output, index=False)
    original_folder = review_root / 'plots_without_model'
    original_folder.mkdir(exist_ok=True)
    # Preserve original plots, add both model curves, and save the values for review for each `day` in `days`.
    for day in days:
        destination = review_root / 'plots' / f'{day}.png'
        backup = original_folder / f'{day}.png'
        # Preserve original plots, add both model curves, and save the values for review when `not backup.exists()`.
        if not backup.exists():
            shutil.copy2(destination, backup)
        plot_day(frame.loc[frame.local_day.eq(day)], farm, review, destination, joined)
    manifest = {'farm_id': farm_id, 'selected_trial': result['trial_id'], 'model': 'selected pvlib physical chain',
        'model_sha256': _digest(model_path), 'aligned_input_sha256': _digest(inputs_path),
        'days': len(days), 'historical_ghi': 'retrospective historical satellite GHI and confirmed diffuse fraction; mean of four temperature/wind forecast cycles',
        'forecast_weather': 'preceding delivery-day 00 UTC deterministic weather cycle',
        'assumed_release_lag_hours': float(section['assumed_forecast_availability_lag_hours']),
        'model_parameter_timing': '2025 fitted coefficients used retrospectively as prior metadata; not a historical real-time backtest',
        'targets': 'raw measured AC; shaded approved exclusions and review markers remain visible',
        'original_plots': str(original_folder), 'prediction_file_sha256': _digest(output)}
    (review_root / 'model_overlay_manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Farm {farm_id}: overlaid {len(days)} days; saved original figures and predictions', flush=True)
    return joined


def main():
    """Draw approved model-inference overlays without changing the exclusion decisions."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    expected_lag = float(config['pv_review_predictions']['assumed_forecast_availability_lag_hours'])
    # Enforce the input contract: Forecast release-lag assumption differs from the declared overlay policy.
    if float(config['source']['availability_lag_hours']) != expected_lag:
        raise ValueError('Forecast release-lag assumption differs from the declared overlay policy')
    with open_store(config) as store:
        # Draw approved model-inference overlays without changing the exclusion decisions for each `farm` in `config['pv_review_predictions']['farm_ids']`.
        for farm in config['pv_review_predictions']['farm_ids']:
            run_farm(config, store, int(farm))


# Process this part of the workflow when `__name__ == '__main__'`.
if __name__ == '__main__':
    main()
