"""Independent physical and numerical checks of the opt-in LFP exposure model."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pytest

from bess_pv_trading.domain import BatteryParameters, DegradationSettings, NaumannSettings, PhysicalState
from bess_pv_trading.optimization.naumann import (
    advance_naumann_state, calendar_coefficient, calendar_graph, calendar_increment,
    calibration_identity, depth_exposure, evaluate_naumann_dispatch, initialize_health,
    naumann_segment_parameters, rate_bin, rate_bin_parameters, sqrt_graph,
)


def reference_case(timestamp=None, **settings_overrides):
    """Use a lossless one-MWh pack so rates, energy and hand calculations agree."""
    battery = BatteryParameters(1., 1., energy_max_mwh=1.,
        efficiency_convention="combined", combined_charge_efficiency=1.,
        combined_discharge_efficiency=1.)
    settings = DegradationSettings(mode="naumann_lfp", replacement_cost_eur=1000.,
        naumann=NaumannSettings(**settings_overrides))
    state = PhysicalState(timestamp or datetime(2026, 11, 11, 12, tzinfo=timezone.utc), .5)
    return battery, settings, state


def test_constant_soc_time_law_and_midnight_only_capacity_update():
    zone = ZoneInfo("Europe/Rome")
    start = datetime(2026, 11, 11, tzinfo=zone).astimezone(timezone.utc)
    battery, settings, state = reference_case(start)
    midday, first_cost = advance_naumann_state(state=state, battery=battery, settings=settings,
        end=start + timedelta(hours=12), energy_mwh=.5, charge_mw=0., discharge_mw=0.)
    assert midday.battery_health.committed_soh == 1.
    assert midday.battery_health.calendar_exposure > 0
    end, last_cost = advance_naumann_state(state=midday, battery=battery, settings=settings,
        end=start + timedelta(hours=24), energy_mwh=.5, charge_mw=0., discharge_mw=0.)
    expected_loss = float(calendar_coefficient(.5, settings)) * np.sqrt(86400)
    assert end.battery_health.committed_soh == pytest.approx(1 - expected_loss)
    assert end.battery_health.active_day == "2026-11-12"
    assert first_cost + last_cost == pytest.approx(5000 * expected_loss)


@pytest.mark.parametrize("date, hours", [((2026, 3, 29), 23), ((2026, 10, 25), 25)])
def test_dst_uses_real_elapsed_seconds(date, hours):
    zone = ZoneInfo("Europe/Rome")
    local = datetime(*date, tzinfo=zone)
    start = local.astimezone(timezone.utc)
    end = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), zone)
    battery, settings, state = reference_case(start)
    result, _ = advance_naumann_state(state=state, battery=battery, settings=settings,
        end=end.astimezone(timezone.utc), energy_mwh=.5, charge_mw=0., discharge_mw=0.)
    assert (end.astimezone(timezone.utc) - start).total_seconds() == hours * 3600
    expected = calendar_coefficient(.5, settings) ** 2 * hours * 3600
    assert result.battery_health.calendar_exposure == pytest.approx(expected)


def test_calendar_integrates_whole_transition_and_is_split_invariant():
    _, settings, _ = reference_case()
    # An independent eight-point rule integrates the degree-six stress exactly.
    nodes, weights = np.polynomial.legendre.leggauss(8)
    soc = .6 + (nodes + 1) / 2 * .3
    expected = 3600 * np.dot(weights / 2,
        (1.2571e-5 * (2.8575 * (soc - .5) ** 3 + .60225)) ** 2)
    full = calendar_increment(.6, .9, 3600, settings)
    split = (calendar_increment(.6, .711, 1332, settings)
             + calendar_increment(.711, .9, 2268, settings))
    assert full == pytest.approx(expected, rel=1e-13)
    assert split == pytest.approx(full, rel=1e-13)
    assert full != pytest.approx(float(calendar_coefficient(.75, settings) ** 2 * 3600), rel=1e-4)
    assert calendar_increment(.9, .6, 3600, settings) == pytest.approx(full)


def test_graph_certificates_cover_concave_soc_stress_and_fresh_sqrt():
    _, settings, _ = reference_case()
    x, upper, maximum = calendar_graph(settings)
    samples = np.linspace(0, 1, 10001)
    difference = np.interp(samples, x, upper) - calendar_coefficient(samples, settings) ** 2
    assert difference.min() >= -1e-23
    assert difference.max() <= maximum * (1 + 1e-10)
    exposure, loss, error = sqrt_graph(settings)
    # Include the analytical worst-error point in every interval, not only a uniform grid.
    worst = ((loss[1:] + loss[:-1]) / 2) ** 2
    errors = np.sqrt(worst) - np.interp(worst, exposure, loss)
    assert errors.max() <= error
    assert error <= settings.naumann.sqrt_loss_error_budget * 1.000001
    assert exposure[0] == loss[0] == 0


def test_naumann_depth_and_rate_units_and_nonconvex_marginals():
    battery, settings, _ = reference_case()
    capacities, marginal = naumann_segment_parameters(battery, settings)
    assert capacities.sum() == pytest.approx(1)
    assert np.all(marginal >= 0)
    assert np.any(np.diff(marginal) < 0)
    depth = .4
    rate = .5
    expected = depth * (((.063 * rate + .0971)
        * (4.0253 * (depth - .6) ** 3 + 1.0923)) / 100) ** 2
    _, rate_multiplier = rate_bin(rate, settings)
    assert depth_exposure(depth, settings) * rate_multiplier == pytest.approx(expected)
    edges, multipliers = rate_bin_parameters(settings)
    assert edges[0] == 0 and edges[-1] == 1
    assert len(edges) == len(multipliers) + 1
    assert rate_bin(.5, settings)[1] == pytest.approx((.063 * .5 + .0971) ** 2)
    assert rate_bin(.5001, settings)[1] == pytest.approx((.063 * .525 + .0971) ** 2)


def test_two_bank_history_and_accrual_survive_arbitrary_replay_splits():
    battery, settings, state = reference_case()
    charge = np.array([.5, 0., .4, 0.])
    discharge = np.array([0., .3, 0., .6])
    duration = np.full(4, .5)
    whole = evaluate_naumann_dispatch(battery=battery, settings=settings, state=state,
        charge_mw=charge, discharge_mw=discharge, delta_hours=duration)
    first = evaluate_naumann_dispatch(battery=battery, settings=settings, state=state,
        charge_mw=charge[:2], discharge_mw=discharge[:2], delta_hours=duration[:2])
    second = evaluate_naumann_dispatch(battery=battery, settings=settings, state=first["final_state"],
        charge_mw=charge[2:], discharge_mw=discharge[2:], delta_hours=duration[2:])
    assert second["final_state"] == whole["final_state"]
    assert first["degradation_eur"] + second["degradation_eur"] == pytest.approx(whole["degradation_eur"])
    assert whole["cycle_exposure"][-1] > 0
    assert np.all(whole["committed_soh"] == 1)


def test_no_silent_history_reset_or_calibration_change():
    battery, settings, state = reference_case()
    with pytest.raises(ValueError, match="explicit health"):
        initialize_health(replace(state, cumulative_throughput_mwh=1), battery, settings)
    initialized = replace(state, battery_health=initialize_health(state, battery, settings))
    changed = replace(settings, naumann=replace(settings.naumann, c_rate_bin_width=.05))
    assert calibration_identity(settings) != calibration_identity(changed)
    with pytest.raises(ValueError, match="calibration"):
        initialize_health(initialized, battery, changed)
    with pytest.raises(ValueError, match="self-discharge"):
        initialize_health(state, replace(battery, self_discharge_per_hour=.001), settings)


def test_full_cycle_constant_condition_matches_sqrt_fec_law():
    battery, settings, state = reference_case(depth_knots=(0., .25, .5, .75, 1.))
    state = replace(state, energy_mwh=0.)
    # Four depth-0.5 cycles at 0.5C have FEC=2; each hour is one half cycle.
    charge = np.tile([.5, 0.], 4)
    discharge = np.tile([0., .5], 4)
    trajectory = evaluate_naumann_dispatch(battery=battery, settings=settings, state=state,
        charge_mw=charge, discharge_mw=discharge, delta_hours=np.ones(8))
    coefficient = .01 * (.063 * .5 + .0971) * (4.0253 * (.5 - .6) ** 3 + 1.0923)
    assert np.sqrt(trajectory["cycle_exposure"][-1]) == pytest.approx(coefficient * np.sqrt(2))
