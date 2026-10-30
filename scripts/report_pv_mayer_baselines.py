#!/usr/bin/env python
"""Score the raw ensemble and fitted Gaussian models on their full matched subset."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    """Preserve existing outputs and report hours independently of linear quantile bounds."""
    from bess_pv_trading.forecasting.data.pv_probabilistic_data import file_hash, write_json
    from bess_pv_trading.forecasting.studies.pv_mayer import summarize_scores
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', required=True, type=Path)
    args = parser.parse_args()
    root = args.run_dir
    config = json.loads((root / 'resolved_config.json').read_text())
    section = config['pv_mayer_postprocessing']
    levels = np.arange(1, int(section['quantile_count']) + 1) / (int(section['quantile_count']) + 1)
    dest = root / 'baselines_all_eligible'
    if dest.exists(): raise FileExistsError(dest)
    dest.mkdir()
    summaries = []
    for farm in section['farm_ids']:
        path = root / f'farm_{farm}' / 'evaluation_predictions.parquet'
        data = pd.read_parquet(path)
        data = data.loc[data.method.isin(['raw_ensemble', 'cn_emos', 'cn_emos_b'])].copy()
        needed = ['crps_fraction_capacity', 'mean_fraction_capacity'] + [f'q_{i+1:02d}' for i in range(len(levels))]
        data['valid_baseline'] = np.isfinite(data[needed].to_numpy(float)).all(axis=1)
        common_ids = data.groupby('row_id').valid_baseline.agg(['sum', 'size'])
        common_ids = common_ids.index[common_ids['sum'].eq(3) & common_ids['size'].eq(3)]
        data['common_score'] = data.row_id.isin(common_ids)
        output = dest / f'farm_{farm}'; output.mkdir()
        data.to_parquet(output / 'evaluation_predictions.parquet', index=False)
        available = summarize_scores(data, levels, float(config['farms'][str(farm)]['ac_capacity_kw'])/1000, output)
        summaries.append({'farm_id': farm, 'matched_hours': len(common_ids), 'comparison_available': available,
                          'source_predictions_sha256': file_hash(path)})
    write_json(dest / 'manifest.json', {'status': 'complete', 'farms': summaries,
        'scope': 'raw ensemble and both Gaussian methods on all common valid hours; independent of LQR bounds',
        'source_manifest_sha256': file_hash(root / 'manifest.json'), 'model_refits': False})
    for row in summaries: print(row, flush=True)


if __name__ == '__main__':
    main()
