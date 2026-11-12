"""Check daily linear-health reporting across real local calendar boundaries."""
import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.reporting.continuous_outputs import _save_battery_health


@pytest.mark.parametrize("day,count", [("2026-03-29", 92), ("2026-10-25", 100)])
def test_daily_health_components_sum_executed_losses_on_dst_days(tmp_path, day, count):
    """Report separate executed cycle/calendar increments and never count wear as cash."""
    timezone = "Europe/Rome"
    first = pd.Timestamp(day, tz=timezone)
    last = (pd.Timestamp(day) + pd.DateOffset(days=1)).tz_localize(timezone)
    boundaries = pd.date_range(first, last, freq="15min").tz_convert("UTC")
    starts, ends = boundaries[:-1], boundaries[1:]
    active = ends.tz_convert(timezone).strftime("%Y-%m-%d")
    soh_end = np.where(active == day, 1., .99)
    frame = pd.DataFrame({"start": starts, "end": ends, "delivery_day": day,
        "battery_soh_start": np.r_[1., soh_end[:-1]], "battery_soh_end": soh_end,
        "available_capacity_mwh_start": np.r_[1., soh_end[:-1]],
        "available_capacity_mwh_end": soh_end, "linear_active_day_end": active,
        "calendar_capacity_loss_fraction": 2e-7, "cycle_capacity_loss_fraction": 3e-7,
        "calendar_degradation_eur": .01, "cycle_degradation_eur": .02,
        "degradation_eur": .03, "realized_market_cashflow_eur": 1.})
    (tmp_path / "csv").mkdir()
    (tmp_path / "json").mkdir()
    summary = _save_battery_health({"market": {"timezone": timezone},
        "degradation": {"model": "linear_soh", "linear_soh": {"timezone": timezone}},
        "experiment_outputs": {"plots": False}}, tmp_path, frame)
    row = pd.read_csv(tmp_path / "csv/battery_health_daily.csv").iloc[0]
    assert len(frame) == count
    assert bool(row.completed_calendar_day)
    assert pd.Timestamp(row.committed_at_utc) == last.tz_convert("UTC")
    assert row.calendar_capacity_loss_fraction == pytest.approx(2e-7 * count)
    assert row.cycle_capacity_loss_fraction == pytest.approx(3e-7 * count)
    assert row.calendar_degradation_eur == pytest.approx(.01 * count)
    assert row.cycle_degradation_eur == pytest.approx(.02 * count)
    assert summary["noncash_degradation_cost_eur"] == pytest.approx(.03 * count)
    assert summary["realized_market_cashflow_eur"] == pytest.approx(count)
