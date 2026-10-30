#!/usr/bin/env python
"""Run the approved auction-aware online CRPS PV comparison."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))


def main():
    from bess_pv_trading.config import load_config
    from bess_pv_trading.data import config_path
    from bess_pv_trading.forecasting.online.pv_online_crps_prepare import prepare_farm
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=('prepare', 'evaluate', 'report', 'all'), default='all')
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--workers', type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    settings = config['pv_online_crps_study']
    source = load_config(config_path(config, settings['source_config']))
    root = config_path(config, settings['output_root']) / settings['run_id']
    workers = min(args.workers or settings['workers'], settings['workers'], 42)
    # Enforce the input contract: workers must be positive.
    if workers < 1: raise ValueError('workers must be positive')
    # Perform main when `args.plan_only`.
    if args.plan_only:
        print(json.dumps({'settings': settings, 'source': settings['source_config'],
                          'workers_max': workers, 'output': str(root),
                          'limitation': 'Validation-selected predictors; first-day DA/IDA1/IDA2 retrospective'}, indent=2))
        return
    root.mkdir(parents=True, exist_ok=True)
    from bess_pv_trading.forecasting.data.pv_probabilistic_data import file_hash, write_json
    sources = Path(__file__).resolve().parents[1] / 'src' / 'forecasting'
    code = {name: file_hash(sources / group / name) for name, group in (
        ('pv_online_crps.py', 'online'), ('pv_online_crps_prepare.py', 'online'),
        ('pv_online_crps_evaluate.py', 'online'), ('pv_online_crps_report.py', 'plotting'))}
    run_manifest = {'settings': settings, 'source_settings': source['pv_probabilistic_study'],
                    'market': source['market'], 'source': source['source'], 'policy': source['policy'],
                    'code_sha256': code, 'validation_use': 'model selection; not an independent test'}
    manifest_path = root / 'study_manifest.json'
    # Perform main when `manifest_path.exists()`.
    if manifest_path.exists():
        # Enforce the input contract: Study configuration or code changed; use a new run ID.
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(run_manifest, default=str)):
            raise ValueError('Study configuration or code changed; use a new run ID')
    else:
        write_json(manifest_path, run_manifest)
    # Perform main when `args.stage in ('prepare', 'all')`.
    if args.stage in ('prepare', 'all'):
        with ProcessPoolExecutor(max_workers=min(workers, len(settings['farms'])), mp_context=multiprocessing.get_context('fork')) as executor:
            futures = {executor.submit(prepare_farm, config, source, int(farm), root): farm for farm in settings['farms']}
            # Perform main for each `future` in `as_completed(futures)`.
            for future in as_completed(futures):
                print(f'Prepared farm {futures[future]}: {future.result()["rows"]} intervals', flush=True)
    # Perform main when `args.stage in ('evaluate', 'all')`.
    if args.stage in ('evaluate', 'all'):
        from bess_pv_trading.forecasting.online.pv_online_crps_evaluate import evaluate_farm_auction
        jobs = [(int(farm), auction) for farm in settings['farms'] for auction in settings['auctions']]
        with ProcessPoolExecutor(max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context('fork')) as executor:
            futures = {executor.submit(evaluate_farm_auction, config, source, farm, auction, root): (farm, auction)
                       for farm, auction in jobs}
            # Perform main for each `future` in `as_completed(futures)`.
            for future in as_completed(futures):
                print(f'Evaluated {futures[future]}: {future.result()}', flush=True)
    # Perform main when `args.stage in ('report', 'all')`.
    if args.stage in ('report', 'all'):
        from bess_pv_trading.forecasting.plotting.pv_online_crps_report import report
        print(report(config, root), flush=True)


# Process this part of the workflow when `__name__ == '__main__'`.
if __name__ == '__main__':
    main()
