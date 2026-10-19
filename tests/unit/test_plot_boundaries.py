"""Delivery plots retain interval ends and the actual final battery state."""
import numpy as np
import pandas as pd
import pytest

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
from matplotlib.figure import Figure

from battery_case_config import battery_config, case_config, configured_battery
from bess_pv_trading.domain import DeliveryGrid
from bess_pv_trading.reporting.plots import plot_interval_series, plot_dispatch
from bess_pv_trading.simulation.gme_oracle import _plot_experiment, _plot_settlement


def plot_fixture():
    """Independent hourly/quarter-hour example ending at delivery's local midnight."""
    case = case_config("plot_boundaries")
    starts, ends = pd.to_datetime(case["starts_utc"]), pd.to_datetime(case["ends_utc"])
    grid = DeliveryGrid(tuple(starts.astype(str)), tuple(starts.to_pydatetime()), tuple(ends.to_pydatetime()))
    power = np.asarray(case["power_mw"])
    dispatch = pd.DataFrame({"start": starts, "end": ends, "pv_ac_mw": power,
        "grid_mw": power, "grid_target_mw": power,
        "charge_mw": np.zeros(len(starts)), "discharge_mw": np.zeros(len(starts)),
        "requested_charge_mw": np.zeros(len(starts)), "requested_discharge_mw": np.zeros(len(starts)),
        "energy_start_mwh": case["energy_boundaries_mwh"][:-1],
        "energy_end_mwh": case["energy_boundaries_mwh"][1:],
        "net_position_mwh": power * grid.delta_hours,
        "imbalance_mwh": np.zeros(len(starts)),
        "imbalance_price_eur_mwh": case["prices_eur_per_mwh"],
        "realized_imbalance_cashflow_eur": case["settlement_cashflow_eur"]})
    prices = {auction: np.asarray(case["prices_eur_per_mwh"])
              for auction in case_config("poi_trading")["reversal_prices_eur_per_mwh"]}
    prices["IDA3"] = prices["DA"].copy()
    plans = pd.concat([pd.DataFrame({"auction": auction,
        "interval_start_utc": starts, "interval_end_utc": ends,
        "pv_ac_forecast_mw": power, "net_battery_dispatch_mw": np.zeros(len(starts)),
        "soc_start_mwh": case["energy_boundaries_mwh"][:-1],
        "soc_end_mwh": case["energy_boundaries_mwh"][1:]}) for auction in prices], ignore_index=True)
    return case, grid, dispatch, prices, plans


def test_mixed_interval_steps_preserve_energy_and_include_end_without_bridging_gaps():
    case, grid, dispatch, _, _ = plot_fixture()
    fig, axis = plt.subplots()
    try:
        line, = plot_interval_series(axis, dispatch.start, dispatch.end, dispatch.pv_ac_mw)
        times = date2num(line.get_xdata())
        assert times[-1] == pytest.approx(date2num(grid.ends[-1]))
        plotted_mwh = np.sum(np.diff(times) * (line.get_ydata()[:-1] + line.get_ydata()[1:]) / 2) * 24
        assert plotted_mwh == pytest.approx(np.dot(case["power_mw"], grid.delta_hours))
        # An absent middle interval must remain a visible break.
        kept = [0, len(dispatch)-1]
        gap_line, = plot_interval_series(axis, dispatch.start.iloc[kept],
            dispatch.end.iloc[kept], dispatch.pv_ac_mw.iloc[kept])
        assert np.isnan(gap_line.get_ydata()).any()
    finally:
        plt.close(fig)


@pytest.mark.parametrize("legacy_plan", [False, True])
def test_every_delivery_panel_reaches_midnight_and_preserves_final_soc(tmp_path, monkeypatch, legacy_plan):
    case, grid, dispatch, prices, plans = plot_fixture()
    if legacy_plan:
        plans = plans.drop(columns="interval_end_utc")
    figures = {}
    def capture_figure(figure, path, **kwargs):
        figures[path.name] = figure
    monkeypatch.setattr(Figure, "savefig", capture_figure)
    battery = configured_battery()
    _plot_experiment(tmp_path, grid, prices, plans, dispatch, case["timezone"],
        {auction: np.ones(len(dispatch), dtype=bool) for auction in prices},
        battery_capacity_mwh=battery.capacity_mwh,
        soc_bounds_percent=(battery_config("unit")["soc_min_percent"], battery_config("unit")["soc_max_percent"]))
    summary = {"strategy": "plot_fixture", "realized_operating_profit_eur": 0,
        "imbalance_settlement_status": "priced", "realized_profit_after_imbalance_eur": 0,
        "realized_imbalance_cashflow_eur": sum(case["settlement_cashflow_eur"]),
        "components": {"auction_revenue_eur": 0, "degradation_eur": 0, "import_fee_eur": 0}}
    _plot_settlement(tmp_path, dispatch, summary, case["timezone"])
    plot_dispatch(dispatch, prices, summary, tmp_path / "dispatch.png", timezone=case["timezone"])
    final = pd.Timestamp(grid.ends[-1]).tz_convert(case["timezone"])
    assert final == final.normalize()  # This endpoint is next-day local midnight.
    for name, figure in figures.items():
        if name == "realized_profit_breakdown.png":
            continue  # A categorical cash-flow chart has no delivery-time axis.
        for axis in figure.axes:
            assert axis.get_xlim()[1] == pytest.approx(date2num(final))
            for line in axis.lines:
                if line.get_label().startswith("_") and len(line.get_xdata()) == 2:
                    continue  # Horizontal reference line uses axis coordinates.
                assert pd.Timestamp(line.get_xdata()[-1]) == final
    soc_lines = figures["soc_evolution.png"].axes[0].lines
    assert len(soc_lines) == len(prices) + 1
    for line in soc_lines:
        assert line.get_ydata()[-1] == pytest.approx(case["energy_boundaries_mwh"][-1] * 100 / battery.capacity_mwh)
