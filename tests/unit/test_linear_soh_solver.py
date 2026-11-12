"""Exercise the linear health MILP against its independent replay."""
from dataclasses import replace
import numpy as np
import pytest

from bess_pv_trading.domain import DegradationSettings, LinearSOHSettings, SolverSettings
from bess_pv_trading.optimization import build_model, solve_model, validate_solution
from bess_pv_trading.optimization.linear_soh import linear_forecast
from test_optimization_contracts import example_inputs
from test_xu_degradation import battery


@pytest.mark.solver
def test_linear_health_forecast_and_objective_across_midnight():
    """The MILP stays linear and commits forecast capacity at midnight."""
    pytest.importorskip("gurobipy")
    b = replace(battery(), energy_min_mwh=0.)
    inputs = example_inputs(prices=((0., 0.),), pv=np.zeros((1, 2)), battery=b,
                            hours=1., start="2026-01-01T22:00:00+00:00")
    settings = DegradationSettings(mode="linear_soh", replacement_cost_eur=1000.,
        linear_soh=LinearSOHSettings(.01, 6000, 1., .17, .8, 31557600., "Europe/Rome", "test"))
    inputs = replace(inputs, state=replace(inputs.state, energy_mwh=.5), degradation=settings,
                     terminal_target_mwh=.5)
    built = build_model(inputs)
    result = solve_model(built, SolverSettings())
    assert result.has_incumbent
    assert validate_solution(inputs, result).valid
    expected, cost = linear_forecast(inputs, result.arrays["charge_mw"], result.arrays["discharge_mw"])
    np.testing.assert_allclose(result.arrays["linear_capacity_mwh"], expected, atol=1e-7)
    np.testing.assert_allclose(result.components["battery_degradation_cost"], cost, atol=1e-6)
    assert expected[0, 1] < expected[0, 0]
    assert expected[0, 2] == expected[0, 1]
    assert built.model.NumQConstrs == 0


@pytest.mark.solver
def test_synthetic_auction_replay_carries_linear_health(tmp_path):
    """Exercise actual auction submission, execution and persisted SOH together."""
    from pathlib import Path
    import pandas as pd
    from bess_pv_trading.config import load_config
    from bess_pv_trading.domain import DeliveryGrid
    from bess_pv_trading.simulation.synthetic import synthetic_inputs, _replay_strategy

    config = load_config(Path(__file__).resolve().parents[2] / "configs/full_experiments/synthetic_xu.yaml")
    config["degradation"] = {"model": "linear_soh", "replacement_cost_eur": 1000.,
        "cost_eur_per_mwh_throughput": 0., "assumptions": "unit test",
        "linear_soh": {"calendar_loss_per_year": .01, "cycle_life": 6000,
            "reference_depth_of_discharge": 1., "cycle_capacity_loss_at_reference_life": .17,
            "end_of_life_soh": .8, "calendar_year_seconds": 31557600.,
            "timezone": "Europe/Rome", "calibration_id": "unit"}}
    config["outputs"]["plots"] = False
    fixture = synthetic_inputs(config)
    selected = slice(40, 50)
    full = fixture["grid"]
    fixture["grid"] = DeliveryGrid(full.interval_ids[selected], full.starts[selected], full.ends[selected])
    for key in ("pv", "base"):
        fixture[key] = fixture[key][selected]
    fixture["actual_prices"] = {key: value[selected] for key, value in fixture["actual_prices"].items()}
    fixture["schedule"] = tuple(replace(event, eligible_intervals=tuple(t - selected.start
        for t in event.eligible_intervals if selected.start <= t < selected.stop)) for event in fixture["schedule"])
    summary = _replay_strategy(config, fixture, "deterministic_integrated", tmp_path / "replay")
    dispatch = pd.read_csv(tmp_path / "replay" / "dispatch.csv")
    assert summary["final_state"]["battery_health"]["calendar_loss"] > 0
    assert summary["final_state"]["cumulative_degradation_eur"] == pytest.approx(dispatch.degradation_eur.sum())
