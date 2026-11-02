"""Compare full PV scenarios, their lower envelope and two whole trajectories."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
from time import perf_counter

import numpy as np

from bess_pv_trading.config import load_config
from bess_pv_trading.domain import SolverSettings
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.scenarios import shared_dispatch_pv_tree
from bess_pv_trading.simulation.continuous import ContinuousAuction, ContinuousReplay
from bess_pv_trading.simulation.continuous_inputs import prepare_continuous_inputs
from bess_pv_trading.simulation.synthetic import json_default


def variants(inputs):
    """Keep all non-PV inputs fixed while constructing each scenario set."""
    original = inputs.scenarios
    values = np.asarray(original.pv_ac_mw, dtype=float)
    if values.shape[0] != 51 or inputs.optimize_imbalances:
        raise ValueError("The pilot requires 51 PV scenarios and exact delivery")
    # The two-path comparison uses total available PV energy, not pointwise extrema.
    energy = values @ inputs.grid.delta_hours
    low, high = int(np.argmin(energy)), int(np.argmax(energy))
    chosen = sorted(set((low, high)))
    base = replace(original, scenario_ids=('base',), probabilities=np.ones(1),
        pv_ac_mw=values[:1], auction_prices_eur_per_mwh={k: v[:1] for k, v in original.auction_prices_eur_per_mwh.items()},
        imbalance_prices_eur_per_mwh=original.imbalance_prices_eur_per_mwh[:1],
        imbalance_negative_prices_eur_per_mwh=(None if original.imbalance_negative_prices_eur_per_mwh is None
            else original.imbalance_negative_prices_eur_per_mwh[:1]))
    cases = {'pointwise_minimum': (values.min(axis=0, keepdims=True), ('pointwise_minimum',)),
             'lowest_and_highest_total_energy': (values[chosen], tuple(original.scenario_ids[i] for i in chosen))}
    output = {'all_51': inputs}
    for name, (pv, ids) in cases.items():
        bundle, tree = shared_dispatch_pv_tree(base, inputs.tree, inputs.grid, pv, ids)
        output[name] = replace(inputs, scenarios=bundle, tree=tree)
    return output


def cross_scenario_shortfall(inputs, result, all_pv):
    """Check whether shared PV injection can be supplied by every original member."""
    if not result.has_incumbent:
        return None
    injection = np.asarray(result.arrays['pv_injection_mw'][0], dtype=float)
    return float(max(0.0, np.max(injection[None, :] - all_pv)))


def prepare_day(config, day, destination):
    """Build one hypothetical DA input without executing a replay or solving a model."""
    config = load_config(config, overrides={'farm_id': 1,
        'continuous_optimization': {'start_date': day, 'end_date': day},
        'state': {'terminal_soc_percent': None}})
    prepared = prepare_continuous_inputs(config)
    replay = ContinuousReplay(config, prepared, 'integrated', destination)
    events = [event for event in prepared['days_data'][day]['schedule'] if event.auction == 'DA']
    if len(events) != 1:
        raise ValueError(f'{day} does not have one available day-ahead auction')
    event = events[0]
    inputs, diagnostic = replay._planning_inputs(ContinuousAuction(day, event), event.submission_at)
    return config, prepared, inputs, diagnostic


def run_day(config, day, work):
    """Solve and independently validate every formulation on identical DA inputs."""
    day_config, prepared, inputs, diagnostics = prepare_day(config, day, work / 'replay_input')
    forms = variants(inputs)
    all_pv = inputs.scenarios.pv_ac_mw
    results, rows = {}, []
    for name, case in forms.items():
        started = perf_counter()
        built = build_model(case, reduce_exact_pv_scenarios=False)
        try:
            solution = solve_model(built, SolverSettings(**day_config['solver']))
        finally:
            built.model.dispose()
        validation = validate_solution(case, solution, tolerance=1e-6, integrality_tolerance=1e-6)
        row = {'formulation': name, 'scenarios': len(case.scenarios.scenario_ids),
            'scenario_ids': list(case.scenarios.scenario_ids), 'status': solution.status,
            'has_incumbent': solution.has_incumbent, 'objective_eur': solution.objective_eur,
            'diagnostics': solution.diagnostics, 'validation': asdict(validation),
            'cross_scenario_pv_shortfall_mw': cross_scenario_shortfall(case, solution, all_pv),
            'elapsed_seconds': perf_counter() - started}
        if solution.has_incumbent:
            row['shared_schedule'] = {key: solution.arrays[key][0].tolist() for key in
                ('charge_mw', 'discharge_mw', 'grid_mw', 'pv_injection_mw', 'energy_mwh')}
            row['trades'] = [{'auction': a, 'node': n, 'interval': t, 'quantity_mwh': q}
                             for (a, n, t), q in solution.trades.items()]
        rows.append(row)
        results[name] = solution
    complete = all(row['status'] == 'optimal' and row['has_incumbent'] and row['validation']['valid'] for row in rows)
    tolerance = float(day_config['envelope_experiment']['objective_tolerance_eur'])
    comparison = {}
    reference = results['all_51']
    for name in ('pointwise_minimum', 'lowest_and_highest_total_energy'):
        candidate = results[name]
        difference = (candidate.objective_eur - reference.objective_eur
                      if candidate.has_incumbent and reference.has_incumbent else None)
        row = next(item for item in rows if item['formulation'] == name)
        full_bound = reference.diagnostics.get('objective_bound_eur')
        candidate_bound = candidate.diagnostics.get('objective_bound_eur')
        overlaps = (None if difference is None or full_bound is None or candidate_bound is None else
            max(candidate.objective_eur, reference.objective_eur) <= min(full_bound, candidate_bound) + tolerance)
        comparison[name] = {'objective_difference_eur': difference,
            'objective_within_tolerance': None if difference is None else abs(difference) <= tolerance,
            'optimality_intervals_overlap': overlaps,
            'feasible_for_all_members': None if row['cross_scenario_pv_shortfall_mw'] is None
                else row['cross_scenario_pv_shortfall_mw'] <= 1e-6}
    return {'day': day, 'status': 'complete' if complete else 'incomplete',
        'input_fingerprint': prepared['input_fingerprint'], 'planning': diagnostics,
        'input_provenance': prepared['provenance'], 'formulations': rows,
        'objective_tolerance_eur': tolerance, 'comparison': comparison}


def summarize(root):
    """Write a compact result table without changing the saved solve evidence."""
    rows = []
    for path in sorted(Path(root).glob('*/result.json')):
        result = json.loads(path.read_text())
        for form in result.get('formulations', []):
            diagnostics = form['diagnostics']
            rows.append({'delivery_day': result['day'], 'formulation': form['formulation'],
                'status': form['status'], 'objective_eur': form['objective_eur'],
                'bound_eur': diagnostics.get('objective_bound_eur'), 'mip_gap': diagnostics.get('mip_gap'),
                'variables': diagnostics.get('variables'), 'constraints': diagnostics.get('constraints'),
                'build_seconds': diagnostics.get('build_seconds'), 'solve_seconds': diagnostics.get('solve_seconds'),
                'elapsed_seconds': form['elapsed_seconds'],
                'cross_scenario_pv_shortfall_mw': form['cross_scenario_pv_shortfall_mw']})
    if rows:
        with (Path(root) / 'summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--check-only', action='store_true', help='Preflight inputs without solving or writing outputs')
    args = parser.parse_args()
    config = load_config(args.config)
    study = config['envelope_experiment']
    if study['farm_id'] != 1 or study['auction'] != 'DA':
        raise ValueError('This pilot is specified for farm 1 and the DA cutoff')
    days = study['delivery_days']
    if args.check_only:
        with tempfile.TemporaryDirectory(prefix='pv_envelope_preflight_') as directory:
            for day in days:
                _, prepared, inputs, _ = prepare_day(args.config, day, Path(directory) / day)
                forms = variants(inputs)
                print(f"{day}: input {prepared['input_fingerprint']}, " +
                      ', '.join(f'{name}={len(case.scenarios.scenario_ids)}' for name, case in forms.items()), flush=True)
        return
    root = Path(study['output_root'])
    root.mkdir(parents=True, exist_ok=False)
    failed = False
    for day in days:
        destination = root / day
        destination.mkdir()
        try:
            result = run_day(args.config, day, destination)
        except Exception as error:
            result = {'day': day, 'status': 'failed', 'error': f'{type(error).__name__}: {error}'}
        (destination / 'result.json').write_text(json.dumps(result, default=json_default, indent=2, allow_nan=False) + '\n')
        print(f"{day}: {result['status']}", flush=True)
        failed |= result['status'] != 'complete'
    summarize(root)
    if failed:
        raise SystemExit('At least one pilot day was incomplete or failed; inspect the saved diagnostics')


if __name__ == '__main__':
    main()
