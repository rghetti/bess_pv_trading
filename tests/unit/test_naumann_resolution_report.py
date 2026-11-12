"""Check the independent cycle-counting report on known synthetic trajectories."""
import importlib.util
from pathlib import Path

import pytest

from bess_pv_trading.domain import NaumannSettings
from bess_pv_trading.optimization.naumann import cycle_rate_factor, depth_exposure


# Load the standalone report without invoking its command-line entry point.
path = Path(__file__).resolve().parents[2] / "scripts/report_naumann_resolution.py"
spec = importlib.util.spec_from_file_location("resolution_report", path)
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


def test_nested_cycles_and_unfinished_half_cycle():
    """Nested 20% and 60% cycles plus one 10% half-cycle are preserved."""
    settings = NaumannSettings(depth_knots=(0, .05, .2, .5, 1))
    result = report.rainflow_benchmark([.2, .8, .4, .6, .4, .2, .3], 1., settings, .5)
    expected = (depth_exposure(.2, settings) + depth_exposure(.6, settings)
                + .5 * depth_exposure(.1, settings)) * cycle_rate_factor(.5, settings) ** 2
    assert result["equivalent_full_cycles"] == pytest.approx(.85)
    assert result["rainflow_analytic_cycle_exposure"] == pytest.approx(expected)


def test_out_of_capacity_path_is_rejected():
    """An invalid trajectory must not be silently clipped before counting."""
    with pytest.raises(ValueError, match="outside"):
        report.rainflow_benchmark([.5, 1.1], 1., NaumannSettings(), .5)
