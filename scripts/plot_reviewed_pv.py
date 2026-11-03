"""Plot reviewed PV labels and their original observations for selected local days."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.data import config_path
from bess_pv_trading.forecasting.data.pv_curated_review import configure_worker_threads, contiguous_positions


def plot_profile(job):
    """Display original AC, reviewed targets, old exclusions, restorations, and GHI."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    source, output, farm_id, day, rating, timezone = job
    frame = pd.read_parquet(source)
    part = frame.loc[frame.delivery_day.eq(day)].copy()
    if part.empty:
        raise ValueError(f'Farm {farm_id} has no observations for {day}')
    times = pd.DatetimeIndex(part.interval_start_utc).tz_convert(timezone)
    fig, (power, irradiance) = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True, layout='constrained')
    power.plot(times, part.measured_ac_raw_mw, color='black', label='Raw measured AC')
    power.plot(times, part.pvlib_historical_ghi_mw, color='tab:blue', linestyle='--',
               alpha=.8, label='Frozen pvlib, historical GHI')
    selected = part.synthetic_target.to_numpy(dtype=bool)
    power.plot(times, np.where(selected, part.model_target_mw, np.nan), color='tab:green',
               linewidth=2, label='Reviewed synthetic / zero target')
    power.scatter(times[selected], part.loc[selected, 'model_target_mw'], color='tab:green', s=9)
    power.axhline(rating, color='gray', linestyle=':', label='Installed AC rating')
    # Shade each decision independently so old and restored spans remain distinguishable.
    for name, color, label in [
            ('retained_prior_exclusion', 'gray', 'Prior exclusion retained'),
            ('previous_exclusion_restored', 'tab:orange', 'Prior exclusion restored'),
            ('synthetic_target', 'tab:green', 'Reviewed target replacement')]:
        for index, positions in enumerate(contiguous_positions(times, part.delivery_day, part[name])):
            power.axvspan(times[positions[0]], times[positions[-1]] + pd.Timedelta(minutes=15),
                          color=color, alpha=.13, label=label if index == 0 else None)
    irradiance.plot(times, part.ghi_w_m2, color='black', label='Historical GHI')
    interpolated = part.ghi_interpolated.to_numpy(dtype=bool)
    irradiance.scatter(times[interpolated], part.loc[interpolated, 'ghi_w_m2'],
                       color='purple', s=15, label='Interpolated GHI')
    power.set(title=f'Farm {farm_id} · {day} · reviewed quarter-hour targets', ylabel='AC power (MW)')
    irradiance.set(ylabel='GHI (W/m²)', xlabel=f'Local interval start ({timezone})')
    for axis in (power, irradiance):
        axis.grid(alpha=.2)
        axis.legend(fontsize=7, loc='upper right')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=120)
    plt.close(fig)


def main():
    """Render named days from the fingerprinted dataset and add a report index."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--profile', action='append', required=True, help='FARM_ID:YYYY-MM-DD')
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config['pv_reviewed_ghi']
    dataset = config_path(config, settings['output_root'])
    report = config_path(config, settings['report_root'])
    output_root = config_path(config, args.output_root)
    if output_root.exists():
        raise FileExistsError(f'Choose a fresh plot output path: {output_root}')
    jobs = []
    rows = []
    for item in args.profile:
        farm, day = item.split(':', 1)
        farm_id = int(farm)
        if farm_id not in settings['farm_ids']:
            raise ValueError(f'Farm {farm_id} is not in the reviewed dataset')
        pd.Timestamp(day)
        path = output_root / f'farm_{farm_id}' / 'plots' / f'{day}.png'
        source = dataset / f'farm_{farm_id}' / 'corpus.parquet'
        rating = float(config['farms'][str(farm_id)]['ac_capacity_kw']) / 1000
        jobs.append((source, path, farm_id, day, rating, settings['delivery_timezone']))
        rows.append({'farm_id': farm_id, 'delivery_day': day, 'path': str(path)})
    if len(set(args.profile)) != len(args.profile):
        raise ValueError('Plot profiles must be unique')
    output_root.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=settings['workers'], initializer=configure_worker_threads,
                             initargs=(settings['threads_per_worker'],)) as pool:
        list(pool.map(plot_profile, jobs))
    pd.DataFrame(rows).to_csv(report / 'reviewed_plot_profiles.csv', index=False)
    (output_root / 'manifest.json').write_text(json.dumps({'dataset': str(dataset),
        'profiles': rows, 'workers': settings['workers'], 'training_started': False}, indent=2) + '\n')
    readme = report / 'README.md'
    existing = readme.read_text()
    marker = '\n## Reviewed target plots\n'
    if marker in existing:
        raise ValueError('The report already has a reviewed target plot section')
    lines = [marker, '', 'Black is raw metered AC; green is the reviewed synthetic target.',
             'Gray shading indicates retained prior exclusions; orange indicates restored quarters.',
             'The blue dashed line is the frozen historical-GHI pvlib reference.', '',
             '| Farm | Local delivery day | Plot |', '|---|---|---|']
    for row in rows:
        link = os.path.relpath(row['path'], report)
        lines.append(f"| {row['farm_id']} | {row['delivery_day']} | [Open]({link}) |")
    readme.write_text(existing + '\n'.join(lines) + '\n')
    print(f'Wrote {len(rows)} reviewed target plots to {output_root}', flush=True)


if __name__ == '__main__':
    main()
