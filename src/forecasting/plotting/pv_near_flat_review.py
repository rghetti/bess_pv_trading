"""Build a review-only queue and daily plots for near-flat PV observations."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from ...config import load_config
from ...data import config_path, load_metadata, open_store
from ...data.policy import plateau_runs
from ..data.pv_outages import apply_episode_exclusions


def find_candidates(frame, farm, section):
    """Find anchor-bounded near-flat native runs before applying exclusions."""
    policy = {'minimum_run': int(section['minimum_quarters']), 'atol_kw': float(section['tolerance_kw']),
              'rtol': 0.0, 'native_minutes': 15}
    runs = plateau_runs(frame.power_kw.to_numpy(float), frame.interval_start_utc, policy)
    episodes = []
    for positions in runs:
        part = frame.iloc[positions]
        retained = part.search_eligible & part.in_training
        if not retained.any():
            continue
        ghi = part.ghi_w_m2.dropna()
        change = float(ghi.max() - ghi.min()) if len(ghi) else None
        zero_ghi = bool(((part.ghi_w_m2 <= section['priority_zero_ghi_w_m2']) &
                         (part.power_kw >= farm.ac_capacity_kw * section['priority_minimum_ac_fraction'])).any())
        high_priority = (len(part) >= int(section['priority_minimum_quarters']) and
                         ((change is not None and change >= section['priority_ghi_range_w_m2']) or zero_ghi))
        episodes.append({'episode_id': f"farm_{farm.farm_id}_{part.interval_start_utc.iloc[0].strftime('%Y%m%dT%H%MZ')}",
            'positions': positions, 'start_utc': part.interval_start_utc.iloc[0],
            'end_utc': part.interval_start_utc.iloc[-1], 'quarters': len(part),
            'level_kw': float(part.power_kw.median()), 'min_power_kw': float(part.power_kw.min()),
            'max_power_kw': float(part.power_kw.max()), 'ghi_min_w_m2': float(ghi.min()) if len(ghi) else None,
            'ghi_max_w_m2': float(ghi.max()) if len(ghi) else None, 'ghi_range_w_m2': change,
            'has_zero_ghi_and_power': zero_ghi, 'strong_evidence': high_priority,
            'existing_excluded_quarters': int(part.episode_excluded.sum()),
            'eligible_training_quarters': int(retained.sum()),
            'first_local_day': part.interval_start_utc.iloc[0].tz_convert(section['timezone']).date().isoformat()})
    # A short run is linked only to a strong nearby-level seed observed on at least three distinct days.
    for item in episodes:
        matching = [x for x in episodes if abs(x['level_kw'] - item['level_kw']) <= section['tolerance_kw']]
        days = len({x['first_local_day'] for x in matching})
        item['same_level_days'] = days
        item['priority'] = ('strong_evidence' if item['strong_evidence'] else
                            'linked_recurring' if days >= section['linked_minimum_distinct_days'] and
                            any(x['strong_evidence'] for x in matching) else 'review_other')
    return episodes


def contiguous_spans(times, active):
    """Group adjacent active quarter-hours while preserving gaps and DST correctly."""
    positions = np.flatnonzero(np.asarray(active, dtype=bool))
    if not len(positions):
        return []
    boundaries = np.flatnonzero((np.diff(positions) != 1) |
        ((times[positions[1:]] - times[positions[:-1]]) != pd.Timedelta(minutes=15))) + 1
    return [(times[group[0]], times[group[-1]] + pd.Timedelta(minutes=15))
            for group in np.split(positions, boundaries)]


def plot_day(day_frame, farm, section, destination, predictions=None):
    """Plot raw observations and make candidate versus previously excluded quarters visible."""
    times = pd.DatetimeIndex(day_frame.interval_start_utc).tz_convert(section['timezone'])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(13, 7.4), sharex=True,
                                      gridspec_kw={'height_ratios': [2, 1]}, layout='constrained')
    for axis in (top, bottom):
        for start, end in contiguous_spans(times, day_frame.episode_excluded):
            axis.axvspan(start, end, color='0.65', alpha=.25, zorder=0)
    top.plot(times, day_frame.power_kw / 1000, color='black', lw=1.5, label='Raw measured AC power')
    if predictions is not None:
        aligned = predictions.reindex(pd.DatetimeIndex(day_frame.interval_start_utc).as_unit('ns'))
        top.plot(times, aligned.historical_ghi_prediction_mw, color='#cf4f17', linewidth=1.5,
                 linestyle='--', label='pvlib: historical GHI, four-cycle mean')
        top.plot(times, aligned.forecast_weather_prediction_mw, color='#4c54b2', linewidth=1.5,
                 label='pvlib: D−1 00 UTC forecast weather')
    colors = {'strong_evidence': '#c02f36', 'linked_recurring': '#9b56a7', 'review_other': '#de8b26'}
    for priority, color in colors.items():
        active = day_frame.priority.eq(priority) & day_frame.search_eligible & day_frame.in_training
        if active.any():
            top.scatter(times[active], day_frame.loc[active, 'power_kw'] / 1000,
                        color=color, s=24, zorder=4, label=priority.replace('_', ' ') + ': eligible')
    top.axhline(farm.ac_capacity_mw, color='0.5', linestyle='--', lw=.8, label='Installed AC capacity')
    if predictions is not None:
        percent = top.secondary_yaxis('right', functions=(lambda x: 100*x/farm.ac_capacity_mw,
                                                        lambda x: x*farm.ac_capacity_mw/100))
        percent.set_ylabel('% of installed AC capacity')
    bottom.plot(times, day_frame.ghi_w_m2, color='#2362a6', lw=1.3, label='Satellite GHI')
    total = int((day_frame.priority.ne('') & day_frame.search_eligible & day_frame.in_training).sum())
    day = times[0].strftime('%Y-%m-%d')
    top.set(title=f'Farm {farm.farm_id} · {day} · {total} candidate quarters used in training'
                  + (f' · AC rating {farm.ac_capacity_mw:.3f} MW' if predictions is not None else ''),
            ylabel='AC power (MW)')
    bottom.set(ylabel='Satellite GHI (W/m²)', xlabel=f"Local time ({section['timezone']})")
    handles, labels = top.get_legend_handles_labels()
    handles.append(Patch(facecolor='0.65', alpha=.25)); labels.append('Already excluded in approved CSV')
    top.legend(handles, labels, fontsize=8, loc='upper right'); bottom.legend(fontsize=8, loc='upper right')
    top.grid(alpha=.2); bottom.grid(alpha=.2); fig.autofmt_xdate(); fig.savefig(destination, dpi=section['plot_dpi'])
    plt.close(fig)


def review_farm(config, section, root, farm, store):
    """Assemble one farm's raw-day evidence and an unapproved decisions template."""
    selected = store.sel(farm_id=farm.farm_id)
    times = pd.DatetimeIndex(pd.to_datetime(selected.time.values, utc=True))
    frame = pd.DataFrame({'interval_start_utc': times, 'power_kw': selected.power.values,
                          'ghi_w_m2': selected.ghi.values, 'eligible': True})
    frame = apply_episode_exclusions(frame, config, farm_id=farm.farm_id)
    path = config_path(config, section['input_run']) / f'farm_{farm.farm_id}' / 'aligned_inputs.parquet'
    views = pd.read_parquet(path, columns=['interval_start_utc', 'eligible', 'split'])
    grouped = views.groupby('interval_start_utc', sort=True)
    if grouped.eligible.nunique(dropna=False).gt(1).any() or grouped['split'].nunique().gt(1).any():
        raise ValueError('The four forecast views disagree on target eligibility or split')
    statuses = grouped.agg(search_eligible=('eligible', 'first'), split=('split', 'first'))
    frame = frame.join(statuses, on='interval_start_utc')
    frame['search_eligible'] = frame.search_eligible.fillna(False).astype(bool)
    frame['in_training'] = frame['split'].eq('train')
    episodes = find_candidates(frame, farm, section)
    frame['episode_id'], frame['priority'] = '', ''
    for episode in episodes:
        frame.loc[episode['positions'], 'episode_id'] = episode['episode_id']
        frame.loc[episode['positions'], 'priority'] = episode['priority']
    times_local = pd.DatetimeIndex(frame.interval_start_utc).tz_convert(section['timezone'])
    frame['local_day'] = times_local.strftime('%Y-%m-%d')
    selected_days = sorted(frame.loc[frame.episode_id.ne('') & frame.search_eligible & frame.in_training, 'local_day'].unique())
    folder = root / f'farm_{farm.farm_id}'
    (folder / 'plots').mkdir(parents=True, exist_ok=True)
    candidates = frame.loc[frame.local_day.isin(selected_days)].copy()
    candidates.to_csv(folder / 'review_intervals.csv', index=False)
    rows = []
    for episode in episodes:
        row = {key: value for key, value in episode.items() if key != 'positions'}
        row['start_local'] = row['start_utc'].tz_convert(section['timezone'])
        row['end_local'] = row['end_utc'].tz_convert(section['timezone'])
        row['decision'] = 'pending_review'
        row['review_notes'] = ''
        rows.append(row)
    columns = ['episode_id', 'start_utc', 'end_utc', 'start_local', 'end_local', 'quarters',
               'level_kw', 'min_power_kw', 'max_power_kw', 'ghi_min_w_m2', 'ghi_max_w_m2',
               'ghi_range_w_m2', 'has_zero_ghi_and_power', 'strong_evidence', 'same_level_days',
               'priority', 'existing_excluded_quarters', 'eligible_training_quarters',
               'first_local_day', 'decision', 'review_notes']
    pd.DataFrame(rows, columns=columns).to_csv(folder / 'review_episodes.csv', index=False)
    for index, day in enumerate(selected_days, start=1):
        part = candidates.loc[candidates.local_day.eq(day)]
        plot_day(part, farm, section, folder / 'plots' / f'{day}.png')
        if index % 10 == 0 or index == len(selected_days):
            print(f'Farm {farm.farm_id}: plots {index}/{len(selected_days)}', flush=True)
    active = frame.search_eligible & frame.in_training & frame.episode_id.ne('')
    return {'farm_id': farm.farm_id, 'episodes': len(episodes), 'days': len(selected_days),
            'eligible_candidate_quarters': int(active.sum()),
            'strong_eligible_quarters': int((active & frame.priority.eq('strong_evidence')).sum()),
            'linked_eligible_quarters': int((active & frame.priority.eq('linked_recurring')).sum()),
            'other_eligible_quarters': int((active & frame.priority.eq('review_other')).sum()),
            'input_path': str(path)}


def write_indexes(root, farm_ids):
    """Link every review day and expose the per-day candidate counts."""
    root_lines = ['# Near-flat review: farm indexes', '',
                  'Open a farm index for linked day plots and counts. Classification is review-only.', '']
    for farm in farm_ids:
        folder = root / f'farm_{farm}'
        frame = pd.read_csv(folder / 'review_intervals.csv', keep_default_na=False)
        eligible = frame.loc[frame.search_eligible & frame.in_training & frame.priority.ne('')]
        counts = eligible.groupby(['local_day', 'priority']).size().unstack(fill_value=0)
        lines = [f'# Farm {farm} review days', '']
        if (folder / 'model_overlay_manifest.json').exists():
            lines += ['The selected pvlib chain is shown on every day. Orange dashed: historical GHI and diffuse',
                      'fraction with the mean of four temperature/wind forecast views. Blue: preceding-day',
                      '00 UTC forecast weather. Historical-GHI predictions and 2025 coefficients are retrospective;',
                      'this is not an unbiased real-time backtest. The power axis shows nameplate capacity',
                      'and its percentage scale.',
                      '[Overlay provenance](model_overlay_manifest.json).', '']
        lines += ['[Episode decisions template](review_episodes.csv) · [Quarter-hour evidence](review_intervals.csv)', '',
                 '| Local day and plot | Strong evidence | Linked recurring | Other candidates |',
                 '|---|---:|---:|---:|']
        for day, row in counts.iterrows():
            lines.append(f"| [{day}](plots/{day}.png) | {row.get('strong_evidence', 0)} | "
                         f"{row.get('linked_recurring', 0)} | {row.get('review_other', 0)} |")
        if counts.empty:
            lines.append('| No eligible candidate days | 0 | 0 | 0 |')
        (folder / 'INDEX.md').write_text('\n'.join(lines)+'\n')
        root_lines.append(f'- [Farm {farm}](farm_{farm}/INDEX.md): {len(counts)} review days')
    root_lines.extend(['', '[Study definitions and full summary](README.md)'])
    (root / 'INDEX.md').write_text('\n'.join(root_lines)+'\n')


def write_provenance(config, section, root):
    """Record fingerprints for the raw time series and the search eligibility inputs."""
    def file_digest(path):
        digest = hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                digest.update(block)
        return digest.hexdigest()
    sources = {}
    with open_store(config) as store:
        for farm in section['farm_ids']:
            selected = store.sel(farm_id=int(farm))
            digest = hashlib.sha256()
            for name in ('time', 'power', 'ghi'):
                array = np.ascontiguousarray(selected[name].values)
                digest.update(name.encode()); digest.update(str(array.dtype).encode()); digest.update(array.tobytes())
            aligned = config_path(config, section['input_run']) / f'farm_{farm}' / 'aligned_inputs.parquet'
            sources[str(farm)] = {'raw_time_power_ghi_sha256': digest.hexdigest(),
                                  'aligned_search_inputs_sha256': file_digest(aligned)}
    zarr = config_path(config, config['inputs']['zarr'])
    provenance = {'raw_zarr_path': str(zarr), 'zarr_metadata_sha256': file_digest(zarr / '.zmetadata'),
                  'approved_exclusions_sha256': file_digest(config_path(config, config['pv_data_quality']['exclusions_csv'])),
                  'farm_sources': sources,
                  'capture': 'fingerprints captured after generating the review package'}
    (root / 'provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


def main():
    """Create a new, non-overwriting review package without changing decisions or source data."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--redraw-existing', action='store_true',
                        help='Redraw plots from an already completed review package')
    parser.add_argument('--index-existing', action='store_true',
                        help='Generate day indexes for an already completed review package')
    parser.add_argument('--farm', type=int, help='Limit a redraw to one configured farm')
    args = parser.parse_args()
    config = load_config(args.config)
    section = config['pv_near_flat_review']
    root = config_path(config, section['output_root']) / section['run_id']
    farms = load_metadata(config_path(config, config['inputs']['metadata_parquet']), section['farm_ids'])
    if args.index_existing:
        if not (root / 'manifest.json').exists():
            raise ValueError('Only a complete review package can be indexed')
        write_indexes(root, section['farm_ids'])
        write_provenance(config, section, root)
        return
    if args.redraw_existing:
        if not (root / 'manifest.json').exists():
            raise ValueError('Only a complete review package can be redrawn')
        if args.farm is not None and args.farm not in section['farm_ids']:
            raise ValueError('Selected farm is not in the review configuration')
        for farm_id in ([args.farm] if args.farm is not None else section['farm_ids']):
            folder = root / f'farm_{farm_id}'
            frame = pd.read_csv(folder / 'review_intervals.csv', parse_dates=['interval_start_utc'],
                                keep_default_na=False, na_values={'power_kw': [''], 'ghi_w_m2': ['']})
            overlay_path = folder / 'model_overlay_predictions.parquet'
            predictions = pd.read_parquet(overlay_path).set_index('interval_start_utc') if overlay_path.exists() else None
            for day, part in frame.groupby('local_day', sort=True):
                plot_day(part, farms[int(farm_id)], section, folder / 'plots' / f'{day}.png', predictions)
            print(f'Farm {farm_id}: redrew {frame.local_day.nunique()} plots', flush=True)
        return
    root.mkdir(parents=True, exist_ok=False)
    with open_store(config) as store:
        summary = [review_farm(config, section, root, farms[int(farm)], store) for farm in section['farm_ids']]
    exclusion_file = config_path(config, config['pv_data_quality']['exclusions_csv'])
    manifest = {'status': 'review_only', 'configuration': section, 'farms': summary,
                'approved_exclusion_csv_sha256': hashlib.sha256(exclusion_file.read_bytes()).hexdigest(),
                'timezone_and_interval': 'Europe/Rome plots; UTC interval-start CSV; inclusive end is a quarter start',
                'decisions': 'all pending; original CSV and source data unchanged'}
    (root / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    lines = ['# Near-flat PV power review', '',
        'This is a review queue, not a revised exclusion policy. Every episode decision is pending.',
        'The black curve is raw AC power; the blue curve is historical satellite-derived GHI.',
        'Gray shading shows intervals already listed in the approved exclusion CSV. Colored dots show candidate',
        'quarters used by the physical search: red = long run with irradiance evidence, purple = shorter linked',
        'recurring-level run, orange = other near-flat candidate. Colors are review priorities, not fault diagnoses.',
        'Missing values remain gaps; no solar-position proxy is used. All plot times are Europe/Rome.', '',
        'Runs use four or more contiguous nonzero observations within 1.5 kW of the anchor. Strong priority requires',
        'eight quarters and either at least 200 W/m² GHI range or a zero-GHI quarter with AC at least 1% of nameplate.',
        'Linked runs are within 1.5 kW of a strong candidate with matching runs on three or more distinct days.',
        'These thresholds are sensitivity settings, not an automatic exclusion decision.', '',
        '| Farm | Episodes | Plot days | Eligible candidate quarters | Strong | Linked | Other |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for item in summary:
        lines.append(f"| {item['farm_id']} | {item['episodes']} | {item['days']} | {item['eligible_candidate_quarters']} | {item['strong_eligible_quarters']} | {item['linked_eligible_quarters']} | {item['other_eligible_quarters']} |")
    lines.extend(['', 'For each farm, open `plots/` for daily PNGs, `review_episodes.csv` for a decisions template,',
                  'and `review_intervals.csv` for raw quarter-hour evidence and existing search eligibility.',
                  'The end timestamp of an episode is the final included quarter-hour start.'])
    (root / 'README.md').write_text('\n'.join(lines)+'\n')
    write_indexes(root, section['farm_ids'])
    write_provenance(config, section, root)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == '__main__':
    main()
