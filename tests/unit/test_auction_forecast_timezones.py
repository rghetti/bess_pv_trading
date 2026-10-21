"""Italian civil auction gates must select only published UTC weather cycles."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from bess_pv_trading.config import load_config
from bess_pv_trading.data.adapters import select_forecast_vintage
from bess_pv_trading.markets import AuctionCalendar

ROOT = Path(__file__).resolve().parents[2]
CASES = load_config(ROOT / "tests/fixtures/configs/timezone_cases.yaml")


def clock_case(case, lag):
    """Use shared runtime clocks with independent fixture expectations."""
    config = load_config(ROOT / "configs/full_experiments/gme_day.yaml", overrides={
        "gme_oracle_experiment": {"delivery_day": case["delivery_day"]},
        "source": {"availability_lag_hours": lag}})
    calendar = AuctionCalendar.from_config(config)
    day = pd.Timestamp(case["delivery_day"], tz="UTC")
    cycles = pd.date_range(day - pd.Timedelta(days=2), day + pd.Timedelta(days=1),
                          freq=pd.Timedelta(hours=config["source"]["cycle_cadence_hours"]))
    dataset = xr.Dataset({"weather": (("farm_id", "cycle_time", "lead_time"),
        np.zeros((1, len(cycles), 1)))}, coords={"farm_id": [config["farm_id"]],
        "cycle_time": cycles.tz_localize(None), "lead_time": [0]})
    return config, calendar, dataset


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("lag", CASES["lag_hours"])
def test_local_gates_choose_expected_utc_cycles_across_seasons_and_dst(case, lag):
    config, calendar, dataset = clock_case(case, lag)
    grid = calendar.delivery_grid(case["delivery_day"])
    assert len(grid) == case["delivery_intervals"]
    for event in calendar.schedule(case["delivery_day"]):
        local = event.gate_at.tz_convert(calendar.timezone)
        assert local.strftime("%H:%M") == config["market"]["gates"][event.auction]
        assert event.gate_at.strftime("%H:%M") == case["utc_gate_clocks"][event.auction]
        local_day = pd.Timestamp(case["delivery_day"]) + pd.Timedelta(
            days=config["market"]["gate_day_offsets"][event.auction])
        assert local.date() == local_day.date()
        assert event.submission_at == event.gate_at - pd.Timedelta(
            minutes=config["market"]["decision_buffer_minutes"])
        # The same instant expressed in either timezone must select the same vintage.
        for cutoff in (event.submission_at, event.submission_at.tz_convert(calendar.timezone)):
            weather = select_forecast_vintage(dataset, config["farm_id"], cutoff,
                {"forecast_ghi": "weather"}, latency_hours=lag)
            expected = pd.Timestamp(f"{local_day.date()} {case['cycle_clocks_by_lag'][lag][event.auction]}", tz="UTC")
            assert weather.forecast_vintage == expected
            assert weather.available_at == expected + pd.Timedelta(hours=lag)
            assert weather.available_at <= event.submission_at
            assert weather.provenance["decision_cutoff_utc"] == event.submission_at.isoformat()
        # IDA3 starts trading at local noon even on a 23/25-hour delivery day.
        first_ida3 = config["market"]["delivery_start_times"][event.auction]
        if first_ida3 is not None:
            assert grid.starts[event.eligible_intervals[0]].tz_convert(calendar.timezone).strftime("%H:%M") == first_ida3


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda case: case["name"])
@pytest.mark.parametrize("lag", CASES["lag_hours"])
def test_publication_boundary_uses_elapsed_utc_hours_and_inclusive_cutoff(case, lag):
    config, calendar, dataset = clock_case(case, lag)
    # This UTC cycle crosses the civil clock change on each DST test day.
    cycle = pd.Timestamp(case["delivery_day"], tz="UTC")
    publication = cycle + pd.Timedelta(hours=lag)
    before = select_forecast_vintage(dataset, config["farm_id"], publication - pd.Timedelta(nanoseconds=1),
        {"forecast_ghi": "weather"}, latency_hours=lag)
    exactly = select_forecast_vintage(dataset, config["farm_id"], publication.tz_convert(calendar.timezone),
        {"forecast_ghi": "weather"}, latency_hours=lag)
    assert before.forecast_vintage < cycle
    assert exactly.forecast_vintage == cycle
    assert exactly.available_at == publication
    assert exactly.available_at - exactly.forecast_vintage == pd.Timedelta(hours=lag)


@pytest.mark.parametrize("case", CASES["cases"], ids=lambda case: case["name"])
def test_delivery_plot_axis_keeps_italian_timezone_and_midnight(case):
    """Check the actual plot helper, including timezone units used for tick labels."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from bess_pv_trading.reporting.plots import plot_interval_series

    config = load_config(ROOT / "configs/full_experiments/gme_day.yaml", overrides={
        "gme_oracle_experiment": {"delivery_day": case["delivery_day"]}})
    calendar = AuctionCalendar.from_config(config)
    grid = calendar.delivery_grid(case["delivery_day"])
    fig, axis = plt.subplots()
    try:
        line, = plot_interval_series(axis, grid.starts.tz_convert(calendar.timezone),
            grid.ends.tz_convert(calendar.timezone), np.zeros(len(grid)))
        assert str(axis.xaxis.get_units()) == calendar.timezone
        last = pd.Timestamp(line.get_xdata()[-1])
        assert last == grid.ends[-1]
        assert last.strftime("%H:%M") == "00:00"
        assert last.date() == (pd.Timestamp(case["delivery_day"]) + pd.Timedelta(days=1)).date()
    finally:
        plt.close(fig)
