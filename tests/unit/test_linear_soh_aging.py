"""Check physical units, chronological commits and calibration history."""
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from bess_pv_trading.domain import BatteryParameters, DegradationSettings, LinearSOHSettings, PhysicalState
from bess_pv_trading.optimization.linear_soh import advance_linear_health, cycle_loss_per_internal_mwh
from bess_pv_trading.simulation.battery_health import restore_battery_health


def example(timestamp):
    """Use a lossless one-MWh pack and independently chosen aging coefficients."""
    battery = BatteryParameters(1., 1., energy_max_mwh=1., efficiency_convention="combined",
                                combined_charge_efficiency=1., combined_discharge_efficiency=1.)
    settings = DegradationSettings(mode="linear_soh", replacement_cost_eur=1000.,
        linear_soh=LinearSOHSettings(calendar_loss_per_year=.01, cycle_life=6000,
            reference_depth_of_discharge=1., cycle_capacity_loss_at_reference_life=.17,
            end_of_life_soh=.8, calendar_year_seconds=31557600., timezone="Europe/Rome",
            calibration_id="test"))
    return battery, settings, PhysicalState(timestamp, .5)


def test_cycle_loss_is_linear_and_charged_once_across_bookkeeping_splits():
    """A full EFC consumes two internal MWh and loses 0.17 / 6000 SOH."""
    start = datetime(2026, 1, 1, 0, tzinfo=timezone.utc)
    battery, settings, state = example(start)
    assert cycle_loss_per_internal_mwh(battery, settings) * 2 == pytest.approx(.17 / 6000)
    first, first_cost = advance_linear_health(state=state, battery=battery, settings=settings,
        end=start + timedelta(hours=1), energy_mwh=1., charge_mw=.5, discharge_mw=0.)
    second, second_cost = advance_linear_health(state=first, battery=battery, settings=settings,
        end=start + timedelta(hours=3), energy_mwh=0., charge_mw=0., discharge_mw=.5)
    third, third_cost = advance_linear_health(state=second, battery=battery, settings=settings,
        end=start + timedelta(hours=4), energy_mwh=.5, charge_mw=.5, discharge_mw=0.)
    assert third.battery_health.cycle_loss == pytest.approx(.17 / 6000)
    assert third.battery_health.committed_soh == 1.
    assert first_cost + second_cost + third_cost == pytest.approx(
        5000 * (third.battery_health.calendar_loss + third.battery_health.cycle_loss))


@pytest.mark.parametrize("date,hours", [((2026, 3, 29), 23), ((2026, 10, 25), 25)])
def test_daily_health_commit_uses_elapsed_dst_seconds(date, hours):
    """Calendar loss follows elapsed time on both nonstandard Italian days."""
    zone = ZoneInfo("Europe/Rome")
    local = datetime(*date, tzinfo=zone)
    start = local.astimezone(timezone.utc)
    end = datetime.combine(local.date() + timedelta(days=1), datetime.min.time(), zone).astimezone(timezone.utc)
    battery, settings, state = example(start)
    result, _ = advance_linear_health(state=state, battery=battery, settings=settings,
        end=end, energy_mwh=.5, charge_mw=0., discharge_mw=0.)
    loss = .01 * hours * 3600 / 31557600.
    assert result.battery_health.calendar_loss == pytest.approx(loss)
    assert result.battery_health.committed_soh == pytest.approx(1 - loss)
    assert restore_battery_health(asdict(result.battery_health)) == result.battery_health


def test_history_requires_matching_calibration():
    """An existing battery cannot silently switch lifetime coefficients."""
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    battery, settings, state = example(start)
    saved, _ = advance_linear_health(state=state, battery=battery, settings=settings,
        end=start + timedelta(hours=1), energy_mwh=.5, charge_mw=0., discharge_mw=0.)
    changed = replace(settings, linear_soh=replace(settings.linear_soh, cycle_life=5000))
    with pytest.raises(ValueError, match="calibration identity"):
        advance_linear_health(state=saved, battery=battery, settings=changed,
            end=saved.timestamp + timedelta(hours=1), energy_mwh=.5, charge_mw=0., discharge_mw=0.)
