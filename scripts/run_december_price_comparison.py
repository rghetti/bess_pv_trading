"""Preflight, run, and report the December auction-price participation study."""
import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from bess_pv_trading.config import load_config
from bess_pv_trading.participation import participation_profiles, profile_id

FARMS = (1, 345, 901, 910)
MODES = ('forecast', 'oracle')
OUTPUT = ROOT / 'outputs/continuous_optimization/december_2025_price_study'
THREAD_VARIABLES = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'NUMEXPR_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')
COMPONENTS = ['auction_revenue_eur', 'imbalance_cashflow_eur', 'degradation_eur',
              'import_fee_eur', 'transaction_fee_eur', 'imbalance_penalty_eur',
              'absolute_imbalance_mwh', 'signed_imbalance_mwh', 'charge_mwh',
              'discharge_mwh', 'curtailment_mwh']


def matched_settings(config):
    """Remove only the three declared arm differences before comparing settings."""
    common = deepcopy(config)
    common['continuous_optimization'].pop('run_id')
    common['energy_price_forecast'].pop('auction_enabled')
    common['gme_oracle_experiment'].pop('price_information')
    return common


def common_provenance(provenance):
    """Compare physical inputs, actual prices and weather independently of models."""
    return {key: value for key, value in provenance.items()
            if key not in ('energy_price_forecasts', 'energy_price_prediction_checks')}


def expected_profiles(config):
    """Name the eight DA-inclusive profiles and the sequential reference."""
    return [profile_id(profile) for profile in participation_profiles(
        config['continuous_participation_study'])] + ['sequential_all_auctions']


def config_path(farm, mode):
    """Resolve one of the declared per-farm comparison configurations."""
    return ROOT / 'configs/research' / f'december_2025_price_{mode}_farm_{farm}.yaml'


def preflight(farm):
    """Prepare both information modes without fitting or optimizing."""
    from bess_pv_trading.simulation.continuous_inputs import prepare_continuous_inputs
    from bess_pv_trading.simulation.continuous import _configuration_fingerprint
    from bess_pv_trading.simulation.synthetic import write_json

    destination = OUTPUT / 'preflight'
    destination.mkdir(parents=True, exist_ok=True)
    reference = None
    evidence = {}
    # Prepare arms serially, retaining evidence rather than two large input bundles.
    for mode in MODES:
        config = load_config(config_path(farm, mode))
        prepared = prepare_continuous_inputs(config, progress=lambda msg: print(f'farm {farm} {mode}: {msg}', flush=True))
        expected_days = [f'2025-12-{day:02}' for day in range(1, 32)]
        if list(prepared['days']) != expected_days:
            raise ValueError('Preflight must cover exactly December 2025')
        skipped = {day: item['skipped'] for day, item in prepared['days_data'].items() if item['skipped']}
        if skipped != {'2025-12-10': [{'auction': 'IDA1', 'dataset': 'ida1_prices',
                                      'reason': 'documented_cancelled_auction'}]}:
            raise ValueError(f'Unexpected auction exclusions: {skipped}')
        signature = (matched_settings(config), common_provenance(prepared['provenance']))
        if reference is not None and signature != reference:
            raise ValueError(f'Farm {farm}: forecast/oracle physical inputs differ')
        reference = signature
        models = {}
        if mode == 'forecast':
            for name, model in prepared['energy_price_provider'].models.items():
                models[name] = {'settings': asdict(model.settings), 'provenance': model.provenance,
                                'fit_cutoff': model.fit_cutoff.isoformat()}
        evidence[mode] = {'status': 'passed', 'days': expected_days,
            'profiles': expected_profiles(config), 'config': config,
            'config_fingerprint': _configuration_fingerprint(config),
            'input_fingerprint': prepared['input_fingerprint'],
            'provenance': prepared['provenance'], 'models': models, 'skipped': skipped,
            'available_auction_events': sum(len(item['schedule']) for item in prepared['days_data'].values())}
        write_json(destination / f'farm_{farm}_{mode}.json', evidence[mode])
        print(f"farm {farm} {mode}: preflight passed ({len(prepared['days'])} days, {prepared['input_fingerprint']})", flush=True)
        del prepared
    write_json(destination / f'farm_{farm}_matched.json', {'status': 'passed', 'physical_inputs_matched': True,
        'input_fingerprints': {mode: item['input_fingerprint'] for mode, item in evidence.items()}})
    return {mode: item['input_fingerprint'] for mode, item in evidence.items()}


def run_farm(farm):
    """Run the two modes serially so each worker owns one farm at a time."""
    env = dict(os.environ)
    env['BESS_PARTICIPATION_WORKERS'] = '1'
    env.update({name: '1' for name in THREAD_VARIABLES})
    env['PYTHONPATH'] = str(ROOT / 'src')
    env['MPLBACKEND'] = 'Agg'
    logs = OUTPUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    for mode in MODES:
        print(f'Starting farm {farm}, {mode}', flush=True)
        with (logs / f'farm_{farm}_{mode}.log').open('w') as log:
            subprocess.run([sys.executable, '-u', '-m', 'bess_pv_trading', 'experiment-continuous',
                            '--config', str(config_path(farm, mode))], cwd=ROOT, env=env,
                            stdout=log, stderr=subprocess.STDOUT, check=True)
        print(f'Completed farm {farm}, {mode}', flush=True)


def require_preflight():
    """Refuse optimization unless every arm has matching preparation evidence."""
    from bess_pv_trading.simulation.continuous import _configuration_fingerprint

    # Check all farms before launching any child, avoiding a partially started study.
    for farm in FARMS:
        matched = json.loads((OUTPUT / 'preflight' / f'farm_{farm}_matched.json').read_text())
        if matched.get('status') != 'passed' or not matched.get('physical_inputs_matched'):
            raise ValueError(f'Farm {farm} has no successful paired preflight')
        for mode in MODES:
            evidence = json.loads((OUTPUT / 'preflight' / f'farm_{farm}_{mode}.json').read_text())
            config = load_config(config_path(farm, mode))
            if (evidence['status'] != 'passed'
                    or evidence['config_fingerprint'] != _configuration_fingerprint(config)
                    or evidence['input_fingerprint'] != matched['input_fingerprints'][mode]):
                raise ValueError(f'Farm {farm} {mode}: stale or incomplete preflight')
            destination = OUTPUT / f'farm_{farm}' / config['continuous_optimization']['run_id']
            if destination.exists():
                raise FileExistsError(f'Study output already exists; inspect before resuming: {destination}')


def validate_tables(monthly, daily, profiles):
    """Reject wrong dates/profiles, discontinuous state and unreconciled cash."""
    import numpy as np

    days = {f'2025-12-{day:02}' for day in range(1, 32)}
    if (monthly.profile.duplicated().any() or set(monthly.profile) != set(profiles)
            or set(daily.profile) != set(profiles)
            or daily.duplicated(['profile', 'delivery_day']).any()):
        raise ValueError('Incomplete or duplicate participation profiles')
    fields = COMPONENTS + ['net_profit_eur', 'initial_energy_mwh', 'final_energy_mwh']
    if not np.isfinite(daily[fields].to_numpy(float)).all():
        raise ValueError('Nonfinite daily accounting')
    cash = (daily.auction_revenue_eur + daily.imbalance_cashflow_eur
            - daily.degradation_eur - daily.import_fee_eur - daily.transaction_fee_eur
            - daily.imbalance_penalty_eur)
    if not np.allclose(cash, daily.net_profit_eur, rtol=1e-9, atol=1e-5):
        raise ValueError('Daily cash components do not reconcile')
    # Each profile is one uninterrupted month, with no artificial midnight reset.
    for profile, group in daily.groupby('profile'):
        if set(group.delivery_day) != days:
            raise ValueError(f'Incomplete December dates for {profile}')
        group = group.sort_values('delivery_day')
        row = monthly.set_index('profile').loc[profile]
        if not np.allclose(group.final_energy_mwh.iloc[:-1], group.initial_energy_mwh.iloc[1:], atol=1e-6, rtol=0):
            raise ValueError(f'Discontinuous SOC for {profile}')
        if not np.isclose(group.net_profit_eur.sum(), row.total_net_profit_eur, atol=1e-5, rtol=1e-9):
            raise ValueError(f'Monthly profit does not reconcile for {profile}')
        if not np.isclose(group.final_energy_mwh.iloc[-1], row.final_energy_mwh, atol=1e-6, rtol=0):
            raise ValueError(f'Monthly final energy does not reconcile for {profile}')


def solver_audit(directory, profiles):
    """Expose nonoptimal solves and fallbacks instead of hiding them in totals."""
    rows = []
    for profile in profiles:
        path = directory / 'strategies' / profile / 'json/optimization_attempts.json'
        attempts = json.loads(path.read_text())
        # Resumed work can contain discarded attempts; retain their count separately.
        active = [row for row in attempts if not row.get('rolled_back_for_resume', False)]
        gaps = [row['mip_gap'] for row in active if row.get('mip_gap') is not None]
        rows.append({'profile': profile, 'attempt_count': len(attempts),
                     'active_attempt_count': len(active),
                     'statuses': dict(Counter(row['status'] for row in active)),
                     'fallback_count': sum(bool(row.get('fallback_applied')) for row in active),
                     'maximum_mip_gap': max(gaps, default=None)})
    return rows


def report():
    """Join only completed monthly trajectories and write a paired summary."""
    import numpy as np
    import pandas as pd

    records = []
    daily_parts = []
    sources = []
    for farm in FARMS:
        common = None
        for mode in MODES:
            config = load_config(config_path(farm, mode))
            directory = OUTPUT / f'farm_{farm}' / f'december_2025_price_{mode}_v1'
            manifest = json.loads((directory / 'json/manifest.json').read_text())
            preflight_evidence = json.loads((OUTPUT / 'preflight' / f'farm_{farm}_{mode}.json').read_text())
            provenance = json.loads((directory / 'json/input_provenance.json').read_text())
            saved_config = json.loads((directory / 'json/study_config.json').read_text())
            profiles = expected_profiles(config)
            if (manifest['status'] != 'complete' or set(manifest['completed']) != set(profiles)
                    or manifest['delivery_days'] != preflight_evidence['days']):
                raise ValueError(f'Incomplete source run: {directory}')
            for key in ('input_fingerprint', 'config_fingerprint'):
                if manifest[key] != preflight_evidence[key]:
                    raise ValueError(f'Run differs from preflight {key}: {directory}')
            signature = (matched_settings(saved_config), common_provenance(provenance))
            if common is not None and signature != common:
                raise ValueError(f'Unmatched saved physical inputs: farm {farm}')
            common = signature
            target_sources = Counter()
            for day in provenance['days'].values():
                target_sources.update(day['pv_provenance']['curated_realized_pv']['source_counts'])
            sources.append({'farm': farm, 'mode': mode, 'directory': str(directory), **manifest,
                            'solver_audit': solver_audit(directory, profiles),
                            'pv_target_source_counts': dict(target_sources)})
            monthly = pd.read_csv(directory / 'csv/monthly_participation.csv')
            daily = pd.read_csv(directory / 'csv/daily_participation.csv')
            validate_tables(monthly, daily, profiles)
            sums = daily.groupby('profile', as_index=False)[COMPONENTS].sum()
            monthly = monthly.merge(sums, on='profile', validate='one_to_one')
            monthly['farm'] = farm
            monthly['mode'] = mode
            records.append(monthly)
            daily['farm'] = farm
            daily['mode'] = mode
            daily_parts.append(daily)
    table = pd.concat(records, ignore_index=True)
    forecast = table[table['mode'].eq('forecast')].set_index(['farm', 'profile'])
    oracle = table[table['mode'].eq('oracle')].set_index(['farm', 'profile'])
    if not forecast.index.equals(oracle.index):
        oracle = oracle.reindex(forecast.index)
    if oracle.total_net_profit_eur.isna().any():
        raise ValueError('Forecast and oracle profiles do not match')
    paired = forecast.join(oracle, lsuffix='_forecast', rsuffix='_oracle')
    paired['profit_forecast_minus_oracle_eur'] = paired.total_net_profit_eur_forecast - paired.total_net_profit_eur_oracle
    for field in COMPONENTS:
        paired[f'{field}_difference'] = paired[f'{field}_forecast'] - paired[f'{field}_oracle']
    paired = paired.sort_index()
    if not np.isfinite(paired.select_dtypes(include='number').to_numpy()).all():
        raise ValueError('Nonfinite monthly report values')
    OUTPUT.mkdir(parents=True, exist_ok=True)
    paired.reset_index().to_csv(OUTPUT / 'paired_monthly.csv', index=False)
    pd.concat(daily_parts, ignore_index=True).to_csv(OUTPUT / 'daily_all_farms.csv', index=False)
    lines = ['# December 2025: retrospective forecast versus perfect auction prices', '',
        'Completed 1–31 December 2025 (Europe/Rome), four farms, nine profiles per arm: '
        '72 independent continuous trajectories. Differences are forecast minus oracle.', '',
        'The forecast model is a frozen ridge model per auction with fit cutoff 30 September 2025, with causally updated price lags. '
        'The perfect-price arm knows actual auction curves in planning; both arms use actual settlements and independent continuous trajectories.', '',
        '| Farm | Profile | Forecast profit EUR | Oracle profit EUR | Difference EUR | Forecast final SOC % | Oracle final SOC % |',
        '| --- | --- | ---: | ---: | ---: | ---: | ---: |']
    for (farm, profile), row in paired.iterrows():
        vals = [row.total_net_profit_eur_forecast, row.total_net_profit_eur_oracle,
                row.profit_forecast_minus_oracle_eur, row.final_soc_percent_forecast, row.final_soc_percent_oracle]
        if not np.isfinite(vals).all():
            raise ValueError('Nonfinite report values')
        lines.append(f'| {farm} | {profile} | ' + ' | '.join(f'{x:.2f}' for x in vals) + ' |')
    # Keep all components visible in Markdown, as well as machine-readable CSV.
    lines.extend(['', 'Net profit = auction revenue + signed imbalance settlement − degradation '
        '− import fees − transaction fees − charged imbalance penalty. Objective values and '
        'illustrative penalties are not realized cash. The configured charged imbalance penalty is zero.', ''])
    fields = COMPONENTS[:7]
    for farm in FARMS:
        lines.extend([f'## Farm {farm}: profit components', '',
            'F = forecast, O = oracle, Δ = F − O. Monetary columns are EUR; absolute imbalance is MWh.', '',
            '| Profile | Arm | Auction revenue | Settlement | Degradation | Import fee | Trading fee | Imbalance penalty | Absolute imbalance |',
            '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |'])
        for (_, profile), row in paired.loc[[farm]].iterrows():
            for label, suffix in (('F', 'forecast'), ('O', 'oracle'), ('Δ', 'difference')):
                lines.append(f'| {profile} | {label} | ' + ' | '.join(
                    f'{row[f"{field}_{suffix}"]:.3f}' for field in fields) + ' |')
        lines.extend(['', '| Profile | Forecast fallbacks | Oracle fallbacks | Forecast final energy MWh | Oracle final energy MWh |',
                      '| --- | ---: | ---: | ---: | ---: |'])
        for (_, profile), row in paired.loc[[farm]].iterrows():
            lines.append(f'| {profile} | {row.fallback_count_forecast:.0f} | {row.fallback_count_oracle:.0f} | '
                         f'{row.final_energy_mwh_forecast:.6f} | {row.final_energy_mwh_oracle:.6f} |')
    lines.extend(['', '## Solver diagnostics', '',
        '| Farm | Arm | Active attempts | Status counts | Fallbacks | Maximum reported relative gap |',
        '| --- | --- | ---: | --- | ---: | ---: |'])
    for source in sources:
        audit = source['solver_audit']
        statuses = Counter()
        for item in audit:
            statuses.update(item['statuses'])
        gaps = [item['maximum_mip_gap'] for item in audit if item['maximum_mip_gap'] is not None]
        gap_text = f'{max(gaps):.6g}' if gaps else 'unavailable'
        lines.append(f"| {source['farm']} | {source['mode']} | {sum(x['active_attempt_count'] for x in audit)} | "
            + ', '.join(f'{key}: {value}' for key, value in sorted(statuses.items()))
            + f" | {sum(x['fallback_count'] for x in audit)} | {gap_text} |")
    lines.extend(['', '## Assumptions and validation', ''])
    lines.extend(['Execution-target source counts below are quarter-hours per farm, shared by '
                  'all profiles and both arms; night-policy counts are listed separately from metered values.', '',
                  '| Farm | Target source | Quarter-hours |', '| --- | --- | ---: |'])
    for source in sources:
        if source['mode'] == 'forecast':
            for label, count in sorted(source['pv_target_source_counts'].items()):
                lines.append(f"| {source['farm']} | {label} | {count} |")
    lines.extend(['', 'The model produces point forecasts, not stochastic price scenarios. '
        'PV uses unfitted pvlib conversion and the reviewed counterfactual execution targets; '
        'forecasts follow the configured night-zero and six-hour weather availability rules. '
        'The comparison is retrospective; price publication times and some source availability are assumed. '
        'Final SOC is shown because remaining inventory has no cash value in profit.', ''])
    lines.extend(['IDA1 for delivery 10 December was documented as cancelled and skipped in both arms. '
        'No missing auction prices were replaced by zero. Cancellation knowledge is treated as known '
        'throughout that delivery day\'s planning, including earlier gates; this is a retrospective scheduling assumption.', '',
        'All eight profiles contain DA; the sequential reference uses all available auctions but plans '
        'only the current auction. The oracle is perfect in auction prices only, with the same updating '
        'PV forecasts, execution targets and daily planning horizon. It is not a global perfect-control '
        'upper bound. Each profile carries SOC, accepted commitments, degradation and fee accounting '
        'across midnight. The final-day optimization targets 50% SOC; execution can deviate.', '',
        'Model features are price lags at 24/48/168/336 hours and seven calendar terms, with standardized '
        'ridge alpha 20. Missing nominal lags use older published same-clock observations within the '
        'configured age limit. No models were trained or selected on December profits. Artifact creation '
        'and source retrieval occurred retrospectively; fit cutoffs describe data eligibility, not proof '
        'of historical model deployment. Some pre-October hourly training labels were repeated within '
        'quarters. No residual scenarios or probability calibration enter optimization.', '',
        'Battery sizes, symmetric grid limits, proxy locations, generic throughput aging and operator '
        'fee assumptions are illustrative. Curated PV replacements can use retrospective historical '
        'GHI and are counterfactual rather than metered production. Availability uses assumed source '
        'publication clocks; the comparison does not reconstruct historical revisions.', '',
        'Validation: complete unique profile/date sets; matching saved physical settings, weather, '
        'realized PV, clearing and settlement inputs; run/preflight fingerprints; daily-to-monthly '
        'cash reconciliation; continuous daily battery boundaries. Runtime reporting also reconciles '
        'executed trade revenue and fees and checks interval-level battery continuity. See each '
        'trajectory\'s JSON solver diagnostics for status, gaps and fallback details.', '',
        'Compute: at most two farms concurrently, one solver and one native numerical thread per '
        'worker, 120 seconds per auction solve, relative MIP gap 1e-4, no absolute gap, no GPUs. '
        'Per-date dispatch plots are disabled (zero dates); monthly comparison and daily-profit '
        'distribution plots are retained.', '',
        'Source manifests and validation evidence: [report_sources.json](report_sources.json). '
        'Tables: [paired_monthly.csv](paired_monthly.csv), [daily_all_farms.csv](daily_all_farms.csv).', ''])
    (OUTPUT / 'report_sources.json').write_text(json.dumps(sources, indent=2) + '\n')
    (OUTPUT / 'summary.md').write_text('\n'.join(lines))
    print(f'Report: {OUTPUT / "summary.md"}', flush=True)


def main():
    """Use two farm workers, then produce a report after every run completes."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--check-only', action='store_true')
    group.add_argument('--run', action='store_true')
    group.add_argument('--report-only', action='store_true')
    args = parser.parse_args()
    # Bound BLAS/OpenMP as well as Gurobi, before numerical imports in preflight.
    for name in THREAD_VARIABLES:
        os.environ[name] = '1'
    if args.report_only:
        report()
        return
    if args.run:
        require_preflight()
    action = preflight if args.check_only else run_farm
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(action, FARMS))
    if args.run:
        report()


if __name__ == '__main__':
    main()
