"""Matched PV envelope assumptions and replay comparison accounting."""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from bess_pv_trading.domain import SolverSettings
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.scenarios import shared_dispatch_pv_tree
from battery_case_config import configured_battery
from test_optimization_contracts import example_inputs

SCRIPTS = Path(__file__).resolve().parents[2] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('pv_envelope_replay', SCRIPTS / 'compare_pv_envelope_replay.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def inputs():
    original = example_inputs(prices=[[40., -10., 80.]], pv=np.array([[1., 1., 1.]]), hours=1.)
    pv = np.full((51, 3), 0.8)
    pv[0] = [0.3, 0.7, 0.6]
    pv[1] = [0.6, 0.2, 0.5]
    pv[2] = [0.4, 0.4, 0.1]
    bundle, tree = shared_dispatch_pv_tree(original.scenarios, original.tree, original.grid, pv,
                                          tuple(f'member_{i}' for i in range(51)))
    return replace(original, scenarios=bundle, tree=tree, optimize_imbalances=False)


def test_envelope_uses_pointwise_min_and_preserves_commitments():
    original = replace(inputs(), pending_position_mwh=(0.1, 0.0, -0.1))
    reduced = module.envelope_input(original)
    np.testing.assert_allclose(reduced.scenarios.pv_ac_mw, [[0.3, 0.2, 0.1]])
    assert reduced.pending_position_mwh == original.pending_position_mwh
    assert reduced.state == original.state
    np.testing.assert_allclose(reduced.scenarios.auction_prices_eur_per_mwh['DA'], [[40., -10., 80.]])


def test_rejects_scenario_dependent_prices_and_imbalance_model():
    original = inputs()
    prices = original.scenarios.auction_prices_eur_per_mwh['DA'].copy()
    prices[1, 0] += 1
    scenarios = replace(original.scenarios, auction_prices_eur_per_mwh={'DA': prices})
    with pytest.raises(ValueError, match='identical auction prices'):
        module.envelope_input(replace(original, scenarios=scenarios))
    with pytest.raises(ValueError, match='exact delivery'):
        module.envelope_input(replace(original, optimize_imbalances=True))


def test_full_and_envelope_have_same_optimum_on_synthetic_pv():
    gp = pytest.importorskip('gurobipy')
    try:
        with gp.Model() as model:
            model.Params.OutputFlag = 0
    except gp.GurobiError as error:
        pytest.skip(f'Gurobi license unavailable: {error}')
    full = inputs()
    full = replace(full, plant=replace(full.plant, battery=configured_battery('unit')),
                   state=replace(full.state, energy_mwh=0.5),
                   auction_eligibility={'DA': (False, True, True)},
                   pending_position_mwh=(0.2, 0.0, 0.0), terminal_target_mwh=None,
                   terminal_minimum_mwh=0.2)
    reduced = module.envelope_input(full)
    results = []
    for case in (full, reduced):
        model = build_model(case, reduce_exact_pv_scenarios=False)
        try:
            result = solve_model(model, SolverSettings(threads=1))
        finally:
            model.model.dispose()
        assert result.status == 'optimal'
        assert validate_solution(case, result).valid
        results.append(result)
    assert results[0].objective_eur == pytest.approx(results[1].objective_eur, abs=1e-6)
    assert module.cross_scenario_shortfall(reduced, results[1], full.scenarios.pv_ac_mw) < 1e-6


def test_report_matches_state_diagnostics_and_sums_execution(tmp_path):
    import json
    import pandas as pd

    root = tmp_path
    metrics = {'net_profit_eur': 100., 'auction_revenue_eur': 110., 'imbalance_cashflow_eur': -4.,
               'degradation_eur': 2., 'import_fee_eur': 1., 'transaction_fee_eur': 3.,
               'curtailment_mwh': 0.3, 'charge_mwh': 0.2, 'discharge_mwh': 0.1,
               'signed_imbalance_mwh': -0.05, 'absolute_imbalance_mwh': 0.05,
               'fallback_count': 0, 'final_soc_percent': 50.}
    for label in ('all_51', 'pointwise_minimum'):
        path = root / label
        (path / 'csv').mkdir(parents=True)
        (path / 'json').mkdir()
        pd.DataFrame([{'delivery_day': '2026-08-02', **metrics}]).to_csv(path / 'csv/daily_kpis.csv', index=False)
        pd.DataFrame([{'start': '2026-08-01T22:00:00Z', 'delivery_day': '2026-08-02',
                       'duration_hours': 0.25, 'curtailment_recovered_mw': 0.4,
                       'net_position_mwh': 1., 'grid_mw': 3.8, 'charge_mw': 0.2, 'discharge_mw': 0.,
                       'curtailment_mw': 0.3, 'energy_end_mwh': 1., 'imbalance_mwh': -0.05,
                       'net_profit_eur': 100.}]).to_csv(path / 'csv/realized_dispatch.csv', index=False)
    attempt = {'delivery_day': '2026-08-02', 'current_auction': 'DA', 'cutoff': '2026-08-01T10:00:00Z',
               'objective_eur': 20., 'objective_bound_eur': 20.001, 'status': 'optimal',
               'validation': {'valid': True}, 'mip_gap': 0.00005}
    full = root / 'all_51/json'
    (full / 'optimization_attempts.json').write_text(json.dumps([attempt]))
    (full / 'envelope_shadow').mkdir()
    shadow = {'status': 'optimal', 'objective_eur': 20.0001,
              'diagnostics': {'objective_bound_eur': 20.001}, 'validation': {'valid': True},
              'cross_scenario_pv_shortfall_mw': 0., 'projected_boundary_energy_mwh': 1.}
    (full / 'envelope_shadow/2026-08-02_DA.json').write_text(json.dumps(shadow))
    assert module.compare(root, ['2026-08-02'], 0.01) == 1
    rows = pd.read_csv(root / 'matched_state_solves.csv')
    assert rows.objective_difference_eur.iloc[0] == pytest.approx(0.0001)
    assert rows.optimality_intervals_overlap.iloc[0]
    totals = pd.read_csv(root / 'period_comparison.csv')
    np.testing.assert_allclose(totals.recovered_curtailment_mwh, [0.1, 0.1])
    intervals = pd.read_csv(root / 'interval_comparison.csv')
    assert intervals.imbalance_mwh_difference.iloc[0] == pytest.approx(0.)
