"""Compare all PV members and their pointwise minimum in continuous replay."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np
import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.domain import SolverSettings
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.simulation.continuous import (CONTINUOUS_POLICY_VERSION, ContinuousReplay,
                                                  ContinuousAuction, _atomic_json, _configuration_fingerprint)
from bess_pv_trading.simulation.continuous_inputs import prepare_continuous_inputs
if __package__:
    from .compare_pv_lower_envelope import variants, cross_scenario_shortfall
else:
    from compare_pv_lower_envelope import variants, cross_scenario_shortfall


def envelope_input(inputs):
    """Use the same single-price, fixed-commitment inputs with minimum PV."""
    for prices in inputs.scenarios.auction_prices_eur_per_mwh.values():
        if not np.array_equal(prices, np.broadcast_to(prices[0], prices.shape)):
            raise ValueError('The comparison requires identical auction prices across members')
    if any(len(set(owners)) != 1 for owners in inputs.tree.bid_nodes.values()):
        raise ValueError('The comparison requires shared auction orders')
    if len(set(inputs.tree.control_nodes)) != 1:
        raise ValueError('The comparison requires shared battery commands')
    return variants(inputs)['pointwise_minimum']


class EnvelopeReplay(ContinuousReplay):
    """Execute the usual replay, replacing each planning scenario set only."""

    def _planning_inputs(self, instance, cutoff):
        inputs, diagnostic = super()._planning_inputs(instance, cutoff)
        reduced = envelope_input(inputs)
        diagnostic['pv_formulation'] = 'pointwise_minimum'
        diagnostic['original_pv_scenario_count'] = len(inputs.scenarios.scenario_ids)
        return reduced, diagnostic


class FullReplayWithShadow(ContinuousReplay):
    """Solve the envelope on exactly each full replay's input state and commitments."""

    def _build_model(self, inputs):
        # This experiment deliberately measures the unreduced reference model.
        return build_model(inputs, reduce_exact_pv_scenarios=False)

    def _planning_inputs(self, instance, cutoff):
        inputs, diagnostic = super()._planning_inputs(instance, cutoff)
        reduced = envelope_input(inputs)
        started = perf_counter()
        built = build_model(reduced)
        try:
            solution = solve_model(built, SolverSettings(**self.config['solver']))
        finally:
            built.model.dispose()
        validation = (asdict(validate_solution(reduced, solution, tolerance=1e-6,
                          integrality_tolerance=float(self.config['solver']['integrality_tolerance'])))
                      if solution.has_incumbent else None)
        result = {'delivery_day': instance.delivery_day, 'auction': instance.event.auction,
                  'cutoff_utc': cutoff.isoformat(), 'status': solution.status,
                  'has_incumbent': solution.has_incumbent, 'objective_eur': solution.objective_eur,
                  'diagnostics': solution.diagnostics, 'validation': validation,
                  'cross_scenario_pv_shortfall_mw': cross_scenario_shortfall(
                      reduced, solution, inputs.scenarios.pv_ac_mw),
                  'elapsed_seconds': perf_counter() - started,
                  'projected_boundary_energy_mwh': inputs.state.energy_mwh,
                  'scenario_count_full': len(inputs.scenarios.scenario_ids)}
        _atomic_json(self.destination / 'json' / 'envelope_shadow' /
                     f'{instance.delivery_day}_{instance.event.auction}.json', result)
        return inputs, diagnostic


def compare(root, days, tolerance):
    """Write matched-state solver comparisons and independent replay outcomes."""
    full = root / 'all_51'
    envelope = root / 'pointwise_minimum'
    attempts = json.loads((full / 'json/optimization_attempts.json').read_text())
    active = [row for row in attempts if not row.get('rolled_back_for_resume')]
    rows = []
    for attempt in active:
        path = full / 'json/envelope_shadow' / f"{attempt['delivery_day']}_{attempt['current_auction']}.json"
        shadow = json.loads(path.read_text())
        fobj, eobj = attempt.get('objective_eur'), shadow.get('objective_eur')
        fbound = attempt.get('objective_bound_eur')
        ebound = shadow['diagnostics'].get('objective_bound_eur')
        comparable = (fobj is not None and eobj is not None and not attempt.get('fallback_applied')
                      and bool(attempt.get('validation', {}).get('valid'))
                      and bool((shadow.get('validation') or {}).get('valid')))
        overlap = (None if not comparable or fbound is None or ebound is None else
                   max(fobj, eobj) <= min(fbound, ebound) + tolerance)
        rows.append({'delivery_day': attempt['delivery_day'], 'auction': attempt['current_auction'],
                     'cutoff_utc': attempt['cutoff'], 'full_status': attempt['status'],
                     'envelope_status': shadow['status'], 'full_fallback': attempt.get('fallback_applied', False),
                     'full_objective_eur': fobj, 'envelope_objective_eur': eobj,
                     'objective_difference_eur': eobj-fobj if comparable else None,
                     'full_bound_eur': fbound, 'envelope_bound_eur': ebound,
                     'full_gap': attempt.get('mip_gap'), 'envelope_gap': shadow['diagnostics'].get('mip_gap'),
                     'optimality_intervals_overlap': overlap,
                     'full_solution_valid': attempt.get('validation', {}).get('valid', False),
                     'envelope_solution_valid': (shadow.get('validation') or {}).get('valid', False),
                     'envelope_max_pv_shortfall_mw': shadow['cross_scenario_pv_shortfall_mw'],
                     'projected_boundary_energy_mwh': shadow['projected_boundary_energy_mwh'],
                     'full_solve_seconds': attempt.get('solve_seconds'),
                     'envelope_solve_seconds': shadow['diagnostics'].get('solve_seconds')})
    theory = pd.DataFrame(rows)
    theory.to_csv(root / 'matched_state_solves.csv', index=False)
    frames, execution_frames = [], []
    for label, path in [('all_51', full), ('pointwise_minimum', envelope)]:
        frame = pd.read_csv(path / 'csv/daily_kpis.csv')
        if list(frame.delivery_day) != list(days):
            raise ValueError(f'Incomplete daily results for {label}')
        execution = pd.read_csv(path / 'csv/realized_dispatch.csv')
        recovered = (execution.curtailment_recovered_mw * execution.duration_hours).groupby(
            execution.delivery_day).sum()
        frame['recovered_curtailment_mwh'] = frame.delivery_day.map(recovered)
        frame.insert(0, 'formulation', label)
        frames.append(frame)
        execution_frames.append(execution)
    daily = pd.concat(frames, ignore_index=True)
    daily.to_csv(root / 'daily_comparison.csv', index=False)
    metrics = ['net_profit_eur', 'auction_revenue_eur', 'imbalance_cashflow_eur', 'degradation_eur',
               'import_fee_eur', 'transaction_fee_eur', 'curtailment_mwh', 'charge_mwh', 'discharge_mwh',
               'signed_imbalance_mwh', 'absolute_imbalance_mwh', 'recovered_curtailment_mwh', 'fallback_count']
    monthly = daily.groupby('formulation', sort=False)[metrics].sum().reset_index()
    monthly['final_soc_percent'] = [frame.final_soc_percent.iloc[-1] for frame in frames]
    monthly.to_csv(root / 'period_comparison.csv', index=False)
    fields = ['start', 'delivery_day', 'net_position_mwh', 'grid_mw', 'charge_mw', 'discharge_mw',
              'curtailment_mw', 'energy_end_mwh', 'imbalance_mwh', 'net_profit_eur']
    intervals = execution_frames[0][fields].merge(execution_frames[1][fields], on=['start', 'delivery_day'],
                                                   suffixes=('_full', '_envelope'), validate='one_to_one')
    if len(intervals) != len(execution_frames[0]) or len(intervals) != len(execution_frames[1]):
        raise ValueError('Replay interval grids do not match')
    for field in fields[2:]:
        intervals[f'{field}_difference'] = intervals[f'{field}_envelope'] - intervals[f'{field}_full']
    intervals.to_csv(root / 'interval_comparison.csv', index=False)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, layout='constrained')
    for label, group in daily.groupby('formulation', sort=False):
        dates = pd.to_datetime(group.delivery_day)
        axes[0].plot(dates, group.net_profit_eur.cumsum(), marker='o', label=label)
        axes[1].plot(dates, group.absolute_imbalance_mwh, marker='o', label=label)
        axes[2].plot(dates, group.curtailment_mwh, marker='o', label=label)
    axes[0].set_ylabel('Cumulative profit (EUR)')
    axes[1].set_ylabel('Absolute imbalance (MWh)')
    axes[2].set_ylabel('Curtailment (MWh)')
    axes[2].set_xlabel('Local delivery day')
    axes[0].legend()
    fig.savefig(root / 'replay_comparison.png', dpi=150)
    plt.close(fig)
    return len(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    config = load_config(args.config, overrides={'farm_id': 1})
    study = config['envelope_replay_experiment']
    # Reject checkpoints if the study code or its scenario reduction changes.
    study['implementation_sha256'] = hashlib.sha256(
        Path(__file__).read_bytes() + Path(__file__).with_name('compare_pv_lower_envelope.py').read_bytes()
    ).hexdigest()
    if study['farm_id'] != 1 or config['trading']['optimize_imbalances']:
        raise ValueError('This comparison requires farm 1 and exact delivery')
    prepared = prepare_continuous_inputs(config, progress=print)
    if len(prepared['days']) != 10:
        raise ValueError('The comparison requires exactly ten continuous delivery days')
    if prepared['imbalance_prices_eur_per_mwh'] is None:
        raise ValueError('Actual imbalance settlement prices are required')
    gates = sum(len([e for e in prepared['days_data'][d]['schedule'] if e.available]) for d in prepared['days'])
    print(f"Preflight: {len(prepared['days'])} days, {gates} available auction gates per replay; "
          f"input fingerprint {prepared['input_fingerprint']}", flush=True)
    if args.check_only:
        with tempfile.TemporaryDirectory(prefix='envelope_replay_preflight_') as directory:
            replay = ContinuousReplay(config, prepared, 'integrated', Path(directory) / 'replay')
            day = prepared['days'][0]
            event = prepared['days_data'][day]['schedule'][0]
            full, _ = replay._planning_inputs(ContinuousAuction(day, event), event.submission_at)
            reduced = envelope_input(full)
            print(f'First gate {day}/{event.auction}: {len(full.scenarios.scenario_ids)} full members, '
                  f'{len(reduced.scenarios.scenario_ids)} minimum envelope', flush=True)
        return
    root = Path(study['output_root'])
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'run.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError('This envelope experiment is already running') from None
        identity = {'config_fingerprint': _configuration_fingerprint(config),
                    'input_fingerprint': prepared['input_fingerprint'],
                    'policy_version': CONTINUOUS_POLICY_VERSION}
        identity_path = root / 'study_identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
            raise ValueError('Cannot resume envelope comparison with changed inputs, configuration or policy')
        if not identity_path.exists() and any((root / label).exists() for label in ('all_51', 'pointwise_minimum')):
            raise ValueError('Cannot reuse existing replay outputs without a study identity')
        _atomic_json(identity_path, identity)
        _atomic_json(root / 'input_provenance.json', prepared['provenance'])
        _atomic_json(root / 'resolved_config.json', config)
        for label, cls in [('all_51', FullReplayWithShadow), ('pointwise_minimum', EnvelopeReplay)]:
            print(f'Running {label}', flush=True)
            cls(config, prepared, 'integrated', root / label, resume=True, progress=print).run()
        count = compare(root, prepared['days'], float(study['objective_tolerance_eur']))
    print(f'Comparison saved to {root}; matched auction states: {count}', flush=True)


if __name__ == '__main__':
    main()
