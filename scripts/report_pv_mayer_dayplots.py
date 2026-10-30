#!/usr/bin/env python
"""Plot one representative validation day of saved, matched PV distributions."""
import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    """Plot the median raw-ensemble error day for every farm on common hours."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    config = json.loads((args.run_dir / 'resolved_config.json').read_text())
    settings = config['pv_mayer_postprocessing']
    levels = np.arange(1, int(settings['quantile_count']) + 1) / (int(settings['quantile_count']) + 1)
    methods = [('raw_ensemble', 'Raw ensemble'), ('cn_emos', 'Censored Gaussian EMOS'),
               ('cn_emos_b', 'Boosted censored EMOS'), ('linear_quantile', 'Rearranged linear quantiles')]
    for farm in settings['farm_ids']:
        root = args.run_dir / 'rearranged_quantiles' / f'farm_{farm}'
        data = pd.read_parquet(root / 'evaluation_predictions.parquet')
        data = data.loc[data.common_score].copy()
        raw = data.loc[data.method.eq('raw_ensemble')].copy()
        raw['squared_error'] = (raw.mean_fraction_capacity - raw.observation_fraction_capacity) ** 2
        daily = raw.groupby('delivery_day').squared_error.mean().sort_values()
        if daily.empty: continue
        day = daily.index[len(daily)//2]
        fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True, layout='constrained')
        rating = float(config['farms'][str(farm)]['ac_capacity_kw']) / 1000
        for ax, (method, label) in zip(axes.flat, methods):
            part = data.loc[data.delivery_day.eq(day) & data.method.eq(method)].copy().set_index('hour_start_utc').sort_index()
            if part.empty: continue
            grid = pd.date_range(part.index.min(), part.index.max(), freq='h')
            part = part.reindex(grid)
            q = part[[f'q_{i+1:02d}' for i in range(len(levels))]].to_numpy(float)
            lower = np.array([np.interp(.05, levels, row) for row in q]) * rating
            upper = np.array([np.interp(.95, levels, row) for row in q]) * rating
            time = grid.tz_convert(settings['timezone'])
            ax.fill_between(time, lower, upper, alpha=.25, label='90% interval')
            ax.plot(time, part.median_fraction_capacity * rating, label='Forecast median')
            ax.plot(time, part.observation_fraction_capacity * rating, color='black', marker='.', label='Measured hourly mean')
            ax.set_title(label); ax.grid(alpha=.25)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(f'Farm {farm}, {day}: median raw-ensemble-error day on matched hours')
        fig.supylabel('AC power (MW)'); fig.supxlabel(settings['timezone'])
        output = root / 'plots'; output.mkdir(exist_ok=True)
        target = output / f'day_{day}.png'
        fig.savefig(target, dpi=150); plt.close(fig)
        print(f'Farm {farm}: {target}', flush=True)


if __name__ == '__main__':
    main()
