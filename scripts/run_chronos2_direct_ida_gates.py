"""Prepare and optionally run direct Chronos-2 IDA forecasts at each gate."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from bess_pv_trading.config import load_config
from bess_pv_trading.forecasting.studies.zero_shot_da import digest
from bess_pv_trading.forecasting.studies.zero_shot_ida import IDAInputs, MARKETS, prepare_ida_bundle
from scripts.da_zero_shot_probe import verify_bundle
from scripts.ida_gpu_queue import execute_queue
from scripts.run_timesfm_realized_study import prepare_one, render


def build_one(gate, market, settings):
    """Pin a direct price-history base and add gate-aligned realized covariates."""
    if gate not in MARKETS or market not in MARKETS[1:] or MARKETS.index(gate) > MARKETS.index(market):
        raise ValueError(f'Invalid decision-gate/target pair: {gate}/{market}')
    directory = Path(settings['output']) / gate / market
    base = directory / 'base'
    source_manifest = Path(settings['forecast_snapshot']) / 'manifest.json'
    model_specs = load_config(settings['model_config'])['zero_shot_price_study']['models']
    models = {name: model_specs[name] for name in settings['models']}
    auctions = load_config(settings['auction_config'])['price_feature_search']
    contract = {'settings': settings, 'auctions': auctions, 'models': models}
    contract_sha = hashlib.sha256(json.dumps(contract, sort_keys=True, default=str).encode()).hexdigest()
    if base.exists():
        manifest = verify_bundle(base)
        observed = manifest['settings']
        if (manifest['source_manifest_sha256'] != digest(source_manifest) or
            observed['start'] != settings['start'] or observed['end_exclusive'] != settings['end_exclusive'] or
            observed['context_days'] != settings['context_days'] or
            observed['minimum_context_days'] != settings['minimum_context_days'] or
            observed['gate'] != gate or observed['market'] != market or manifest['models'] != models or
            manifest.get('direct_contract_sha256') != contract_sha or
            manifest.get('direct_gate_builder_sha256') != digest(Path(__file__)) or
            manifest.get('ida_builder_sha256') != digest(Path(__file__).parents[1] / 'src/forecasting/studies/zero_shot_ida.py')):
            raise ValueError(f'Existing base differs from requested study: {base}')
    else:
        directory.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix='.direct-base-', dir=directory))
        try:
            source = IDAInputs(settings['forecast_snapshot'], settings, auctions)
            generated = temporary / 'base'
            manifest = prepare_ida_bundle(source, generated, market, settings['context_days'],
                                          'history', models, decision_gate=gate)
            manifest['direct_gate_builder_sha256'] = digest(Path(__file__))
            manifest['ida_builder_sha256'] = digest(Path(__file__).parents[1] / 'src/forecasting/studies/zero_shot_ida.py')
            manifest['direct_contract_sha256'] = contract_sha
            from bess_pv_trading.forecasting.studies.zero_shot_da import write_json
            write_json(generated / 'manifest.json', manifest)
            os.rename(generated, base)
        finally:
            shutil.rmtree(temporary, ignore_errors=True)
    return prepare_one(base, settings['actuals_snapshot'], directory, 'direct', settings)


def main():
    """Prepare immutable bundles or run an explicitly requested inference study."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/research/chronos2_direct_ida_gates.yaml')
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    parser.add_argument('--prepare-workers', type=int, default=3)
    parser.add_argument('--report-workers', type=int, default=3)
    parser.add_argument('--devices', nargs='+', default=['cuda:0', 'cuda:1', 'cuda:2'])
    parser.add_argument('--python-root', default='outputs/zero_shot_envs')
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if sum((args.prepare_only, args.run, args.report_only)) != 1:
        parser.error('Choose exactly one of --prepare-only, --run or --report-only')
    if min(args.prepare_workers, args.report_workers, args.batch_size, args.threads) < 1 or len(args.devices) != len(set(args.devices)):
        parser.error('Worker counts must be positive and devices distinct')
    settings = load_config(args.config)['direct_ida_gate_study']
    tasks = [(gate, market) for gate, markets in settings['gates'].items() for market in markets]
    if len(tasks) != len(set(tasks)) or not tasks:
        raise ValueError('Configure distinct decision-gate/target pairs')
    directories = [Path(settings['output']) / gate / market for gate, market in tasks]
    if not args.report_only:
        if args.prepare_workers == 1:
            for index, (gate, market) in enumerate(tasks, 1):
                print(f'Prepared {index}/{len(tasks)}: {build_one(gate, market, settings)}', flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.prepare_workers) as executor:
                futures = [executor.submit(build_one, gate, market, settings) for gate, market in tasks]
                for index, future in enumerate(as_completed(futures), 1):
                    print(f'Prepared {index}/{len(tasks)}: {future.result()}', flush=True)
    if args.prepare_only:
        return
    if args.run:
        execute_queue(directories, settings['models'], args.devices, args.python_root,
                      args.batch_size, args.threads)
    if args.report_workers == 1:
        for index, directory in enumerate(directories, 1):
            print(f'Reported {index}/{len(tasks)}: {render(directory)}', flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.report_workers) as executor:
            futures = [executor.submit(render, directory) for directory in directories]
            for index, future in enumerate(as_completed(futures), 1):
                print(f'Reported {index}/{len(tasks)}: {future.result()}', flush=True)


if __name__ == '__main__':
    main()
