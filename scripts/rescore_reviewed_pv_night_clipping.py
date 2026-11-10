"""Rescore saved reviewed-PV predictions with monthly nighttime clipping."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.training.pv_reviewed_hpo import _score_splits


def digest(path):
    """Check the original saved predictions before deriving new artifacts."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    """Apply the saved monthly night mask and score the retained model outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', required=True, type=Path)
    parser.add_argument('--original-report', required=True, type=Path)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    root = args.run_root
    status = json.loads((root / 'status.json').read_text())
    if status['status'] != 'complete':
        raise ValueError('The source search must have completed')
    if args.report.exists() or (root / 'night_clipped').exists():
        raise FileExistsError('A night-clipped output already exists; choose a fresh location')
    args.report.mkdir(parents=True)
    metrics = []
    diagnostics = []
    models = sorted(root.glob('farm_*/models/**/result.json'))
    if len(models) != status['refits_completed']:
        raise ValueError('Retained model count disagrees with completed refits')
    for path in models:
        # Each retained model has its own forecast array and shared farm masks.
        record = json.loads(path.read_text())
        trial = record['trial']
        cache = root / f"farm_{trial['farm_id']}" / 'cache'
        frame = pd.read_parquet(cache / 'frame.parquet')
        masks = dict(np.load(cache / 'masks.npz', allow_pickle=False))
        source = path.parent / 'prediction_mw.npy'
        if digest(source) != record['checksums']['prediction_mw.npy']:
            raise ValueError(f'Prediction checksum mismatch: {source}')
        prediction = np.load(source, allow_pickle=False)
        night = frame.night.to_numpy(dtype=bool)
        clipped = prediction.copy()
        clipped[night & np.isfinite(clipped)] = 0.0
        scores = _score_splits(frame, clipped, masks, float(record['rating_mw']))
        original = {(row['split'], row['target_source']): row for row in record['metrics']}
        for score in scores:
            before = original[(score['split'], score['target_source'])]
            if abs(score['day_balanced_nrmse_percent'] - before['day_balanced_nrmse_percent']) > 1e-4:
                raise ValueError('Daylight score changed after nighttime clipping')
            metrics.append({'farm_id': trial['farm_id'], 'family': trial['family'],
                            'feature': trial['feature'], 'name': trial['name'], **score})
        for split in ('validation', 'test'):
            keep = night & frame['split'].eq(split).to_numpy() & np.isfinite(prediction)
            values = prediction[keep]
            diagnostics.append({'farm_id': trial['farm_id'], 'family': trial['family'],
                'feature': trial['feature'], 'name': trial['name'], 'split': split,
                'night_quarters_with_prediction': int(keep.sum()),
                'original_mean_night_mw': float(values.mean()),
                'original_max_night_mw': float(values.max()),
                'original_night_prediction_mwh': float(values.sum() / 4),
                'clipped_max_night_mw': float(clipped[keep].max())})
        relative = path.parent.relative_to(root)
        output = root / 'night_clipped' / relative / 'prediction_mw.npy'
        output.parent.mkdir(parents=True, exist_ok=True)
        np.save(output, clipped, allow_pickle=False)
    table = pd.DataFrame(metrics)
    table.to_csv(args.report / 'model_metrics_night_clipped.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(args.report / 'night_prediction_diagnostics.csv', index=False)
    top = pd.read_csv(args.original_report / 'top_three_by_farm.csv')
    top_metrics = top.merge(table, on=['farm_id', 'family', 'feature', 'name'], validate='one_to_many')
    top_metrics.to_csv(args.report / 'top_three_metrics_night_clipped.csv', index=False)
    trials = pd.read_csv(args.original_report / 'all_trials_validation.csv')
    trials['night_clipping_effect_on_daylight_score'] = 'none; selection mask excludes night'
    trials.to_csv(args.report / 'all_trials_validation_night_clipped.csv', index=False)
    summary = {'source_identity': status['identity'], 'saved_models_rescored': len(models),
        'validation_trials_with_unchanged_daylight_score': len(trials),
        'evaluation': 'original eligible month-specific daylight mask',
        'night_policy': 'zero finite predictions inside stored monthly night mask',
        'limitation': 'Only selected model predictions were retained; other trials have validation scores only.'}
    (args.report / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.report / 'README.md').write_text(
        '# Nighttime-clipped reviewed PV results\n\n'
        f"Rescored {len(models)} saved selected models after zeroing finite predictions in the stored monthly night window. "
        'The existing daylight-only metrics and all 8192 validation rankings are unchanged. '
        'The other trial predictions were not retained. Original artifacts were not modified.\n\n'
        '[Model metrics](model_metrics_night_clipped.csv) · '
        '[Top-three metrics](top_three_metrics_night_clipped.csv) · '
        '[All validation trials](all_trials_validation_night_clipped.csv) · '
        '[Nighttime prediction diagnostics](night_prediction_diagnostics.csv)\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
