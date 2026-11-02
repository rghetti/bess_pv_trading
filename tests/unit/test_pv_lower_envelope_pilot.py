"""Check the distinct lower-envelope and whole-trajectory experiment inputs."""
from dataclasses import replace
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from bess_pv_trading.scenarios import shared_dispatch_pv_tree
from test_optimization_contracts import example_inputs


def test_variants_preserve_prices_and_detect_omitted_low_interval():
    script = Path(__file__).resolve().parents[2] / 'scripts/compare_pv_lower_envelope.py'
    spec = importlib.util.spec_from_file_location('pv_lower_envelope_pilot', script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    base = example_inputs(prices=[[40., 50.]], pv=np.array([[1., 1.]]), hours=1.)
    pv = np.full((51, 2), 0.6)
    pv[0] = [0.1, 0.4]  # Lowest total energy.
    pv[1] = [0.4, 0.2]  # Constrains the second interval.
    pv[2] = [0.9, 0.9]  # Highest total energy.
    bundle, tree = shared_dispatch_pv_tree(base.scenarios, base.tree, base.grid, pv,
                                            tuple(f'member_{i}' for i in range(51)))
    inputs = replace(base, scenarios=bundle, tree=tree, optimize_imbalances=False)
    forms = module.variants(inputs)
    np.testing.assert_allclose(forms['pointwise_minimum'].scenarios.pv_ac_mw, [[0.1, 0.2]])
    np.testing.assert_allclose(forms['lowest_and_highest_total_energy'].scenarios.pv_ac_mw,
                               [[0.1, 0.4], [0.9, 0.9]])
    np.testing.assert_allclose(forms['pointwise_minimum'].scenarios.auction_prices_eur_per_mwh['DA'], [[40, 50]])
    candidate = SimpleNamespace(has_incumbent=True, arrays={'pv_injection_mw': np.array([[0.1, 0.4]])})
    assert module.cross_scenario_shortfall(inputs, candidate, pv) == 0.2
