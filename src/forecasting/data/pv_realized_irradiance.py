"""Prepare retrospective observed-irradiance inputs for the physical PV search."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import load_config
from ...data import config_path, night_window_mask, open_store
from ..data.pv_outages import apply_episode_exclusions


def interpolate_instantaneous_midpoints(hourly_values, horizon):
    """Interpolate within each vintage at quarter-hour interval midpoints."""
    values = np.asarray(hourly_values, dtype=float)
    if values.ndim != 2 or values.shape[1] != horizon + 1:
        raise ValueError('One hourly value is required for every horizon endpoint')
    steps = np.arange(horizon * 4)
    hour = steps // 4
    fraction = ((steps % 4) + 0.5) / 4.0
    return values[:, hour] * (1 - fraction) + values[:, hour + 1] * fraction


def treat_night_targets(frame, night_policy, treatment):
    """Preserve measured AC and record zero treatment without creating missing targets."""
    if treatment not in {'zero', 'preserve'}:
        raise ValueError(f'Unsupported night target treatment: {treatment}')
    result = frame.copy()
    result['measured_ac_raw_mw'] = result.measured_ac_mw
    night = night_window_mask(pd.DatetimeIndex(result.interval_start_utc), night_policy)
    finite = np.isfinite(result.measured_ac_mw)
    result['night_target_zeroed'] = (night & finite & result.measured_ac_mw.ne(0)
                                      if treatment == 'zero' else False)
    if treatment == 'zero':
        result.loc[night & finite, 'measured_ac_mw'] = 0.0
    return result


def prepare_farm(config: dict, farm_id: int, output: Path) -> dict:
    """Align every cycle's first 24 hours to historical quarter-hour observations."""
    section = config['pv_realized_irradiance_search']
    timezone = section['delivery_timezone']
    start = pd.Timestamp(section['training_start'], tz=timezone).tz_convert('UTC')
    end = pd.Timestamp(section['end_exclusive'], tz=timezone).tz_convert('UTC')
    validation = pd.Timestamp(section['validation_start'], tz=timezone).tz_convert('UTC')
    horizon = int(section['cycle_horizon_hours'])
    steps = horizon * 4
    power_path = (config_path(config, section['curated_power_root']) /
                  f'farm_{farm_id}' / 'parquet' / 'corpus.parquet')
    power = pd.read_parquet(power_path, columns=['interval_start_utc', 'measured_ac_mw', 'eligible'])
    power = power.loc[power.eligible]
    grouped = power.groupby('interval_start_utc').measured_ac_mw
    if grouped.nunique(dropna=False).gt(1).any():
        raise ValueError('Curated auction views disagree on measured power')
    power = grouped.first()
    with open_store(config) as store:
        selected = store.sel(farm_id=farm_id)
        times = pd.DatetimeIndex(pd.to_datetime(selected.time.values, utc=True))
        historic = pd.DataFrame({
            'ghi': selected.ghi.values,
            'diffuse_fraction': selected.fraction_diffuse.values,
        }, index=times)
        if section['ghi_gap_fill'] != 'linear_inside':
            raise ValueError('Unsupported GHI gap fill')
        historic['ghi_raw'] = historic.ghi
        historic['ghi'] = historic.ghi.interpolate(method='time', limit_area='inside')
        historic['ghi_interpolated'] = historic.ghi_raw.isna() & historic.ghi.notna()
        historic['measured_ac_mw'] = power.reindex(times)
        cycles = pd.DatetimeIndex(pd.to_datetime(selected.cycle_time.values, utc=True))
        leads = pd.TimedeltaIndex(selected.lead_time.values) / pd.Timedelta(hours=1)
        required = np.arange(horizon + 1)
        indices = pd.Index(leads).get_indexer(required)
        if (indices < 0).any():
            raise ValueError('Forecast archive lacks required hourly endpoints')
        wanted = (cycles < end) & (cycles + pd.Timedelta(hours=horizon) > start)
        cycle_indices = np.flatnonzero(wanted)
        cycles = cycles[wanted]
        cycle_starts = np.repeat(cycles.to_numpy(), steps)
        offset = np.tile(np.arange(steps), len(cycles))
        interval = pd.DatetimeIndex(cycle_starts) + pd.to_timedelta(offset * 15, unit='min')
        inside = (interval >= start) & (interval < end)
        base = historic.reindex(interval)
        frame = pd.DataFrame({
            'interval_start_utc': interval, 'forecast_cycle_utc': cycle_starts,
            'lead_minutes': offset * 15, 'ghi_w_m2': base.ghi.to_numpy(),
            'ghi_raw_w_m2': base.ghi_raw.to_numpy(),
            'ghi_interpolated': base.ghi_interpolated.fillna(False).to_numpy(dtype=bool),
            'diffuse_fraction': base.diffuse_fraction.to_numpy(),
            'measured_ac_mw': base.measured_ac_mw.to_numpy(),
        })
        for canonical, source in [('temperature_c', 'temperature_2m'),
                                  ('wind_u_10m', 'wind_u_component_10m'),
                                  ('wind_v_10m', 'wind_v_component_10m')]:
            raw = selected[source].isel(cycle_time=cycle_indices, lead_time=indices).values
            aligned = interpolate_instantaneous_midpoints(raw, horizon)
            frame[canonical] = aligned.reshape(-1)
        frame = frame.loc[inside].reset_index(drop=True)
    frame['wind_speed_m_s'] = np.hypot(frame.wind_u_10m, frame.wind_v_10m)
    frame['weight'] = float(section['cycle_weight'])
    frame['split'] = np.where(frame.interval_start_utc < validation, 'train', 'validation')
    frame = treat_night_targets(frame, section.get('night_policy', config['policy']['night']),
                                section.get('night_target_treatment', 'preserve'))
    valid_ghi = np.isfinite(frame.ghi_w_m2) & (frame.ghi_w_m2 >= 0)
    valid_fraction = np.isfinite(frame.diffuse_fraction) & frame.diffuse_fraction.between(0, 1)
    frame['eligible'] = (valid_ghi & ((frame.ghi_w_m2 == 0) | valid_fraction)
                         & np.isfinite(frame.measured_ac_mw) & np.isfinite(frame.temperature_c)
                         & np.isfinite(frame.wind_speed_m_s))
    frame = apply_episode_exclusions(frame, config, farm_id=farm_id)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)
    counts = frame.groupby(['split', 'interval_start_utc']).eligible.sum()
    stats = {split: {'rows': int(len(part)), 'eligible_rows': int(part.eligible.sum()),
                     'excluded_rows': int(part.episode_excluded.sum()),
                     'night_target_zeroed_rows': int(part.night_target_zeroed.sum()),
                     'ghi_interpolated_rows': int(part.ghi_interpolated.sum()),
                     'vintages_per_interval': {str(k): int(v) for k, v in
                         counts.loc[split].value_counts().sort_index().items()}}
             for split, part in frame.groupby('split')}
    if section.get('selection_scope', 'validation') == 'full_period':
        stats = {('pre_august_fit' if split == 'train' else 'august_fit'): value
                 for split, value in stats.items()}
    return {'farm_id': farm_id, 'statistics': stats, 'weight_per_view': float(section['cycle_weight']),
            'selection_scope': section.get('selection_scope', 'validation'),
            'night_target_treatment': section.get('night_target_treatment', 'preserve'),
            'night_policy': section.get('night_policy', config['policy']['night']),
            'raw_target_column': 'measured_ac_raw_mw',
            'cycle_horizon_hours': horizon, 'split_timezone': timezone,
            'diffuse_definition': 'confirmed DHI/GHI', 'ghi_source': 'historical satellite-derived',
            'ghi_gap_fill': 'linear inside historical timeline only; flagged',
            'curated_power_path': str(power_path),
            'temperature_and_wind': 'forecast views, interpolated within each vintage at quarter-hour midpoint'}


def main() -> None:
    """Write one reusable aligned corpus and coverage summary for each farm."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    section = config['pv_realized_irradiance_search']
    root = config_path(config, section['output_root']) / section['run_id']
    root.mkdir(parents=True, exist_ok=True)
    summary = []
    for farm in section['farm_ids']:
        print(f'Preparing farm {farm}', flush=True)
        item = prepare_farm(config, int(farm), root / f'farm_{farm}' / 'aligned_inputs.parquet')
        summary.append(item)
        print(json.dumps(item, default=str), flush=True)
    (root / 'input_coverage.json').write_text(json.dumps(summary, indent=2) + '\n')


if __name__ == '__main__':
    main()
