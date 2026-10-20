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
from bess_pv_trading.reporting.plots import plot_interval_series, plot_dispatch, pv_power_allocation
from bess_pv_trading.simulation.gme_oracle import _plot_experiment, _plot_settlement


def plot_fixture():
    """Independent hourly/quarter-hour example ending at delivery's local midnight."""
    case = case_config("plot_boundaries")
    starts, ends = pd.to_datetime(case["starts_utc"]), pd.to_datetime(case["ends_utc"])
    grid = DeliveryGrid(tuple(starts.astype(str)), tuple(starts.to_pydatetime()), tuple(ends.to_pydatetime()))
    power = np.asarray(case["power_mw"])
    dispatch = pd.DataFrame({"start": starts, "end": ends, "pv_ac_mw": power,
        "grid_mw": power, "grid_target_mw": power,
        "curtailment_mw": np.zeros(len(starts)),
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


def test_pv_allocation_excludes_grid_charging_and_battery_discharge():
    case = case_config("pv_power_allocation")
    dispatch = pd.DataFrame(case["input"])
    allocation = pv_power_allocation(dispatch)
    for column, expected in case["expected"].items():
        np.testing.assert_allclose(allocation[column], expected)
    np.testing.assert_allclose(allocation[list(case["expected"])].sum(axis=1),
                               dispatch.pv_ac_mw)


@pytest.mark.parametrize("import_fee", [0.0, 3.0])
def test_profit_breakdown_shows_trading_fees_and_gross_volume(tmp_path, monkeypatch, import_fee):
    """Trading fees occupy the old import slot; nonzero grid charges stay separate."""
    case, grid, dispatch, prices, plans = plot_fixture()
    figures = {}
    monkeypatch.setattr(Figure, "savefig", lambda figure, path, **kwargs: figures.update({path.name: figure}))
    imbalance = float(dispatch.realized_imbalance_cashflow_eur.sum())
    summary = {"imbalance_settlement_status": "priced", "realized_imbalance_cashflow_eur": imbalance,
        "realized_profit_after_imbalance_eur": 100 + imbalance - 10 - 2.5 - import_fee,
        "components": {"auction_revenue_eur": 100, "degradation_eur": 10,
                       "transaction_fee_eur": 2.5, "import_fee_eur": import_fee},
        "transaction_fees": {"gross_traded_mwh": 62.5}}
    _plot_settlement(tmp_path, dispatch, summary, case["timezone"])
    axis = figures["realized_profit_breakdown.png"].axes[0]
    labels = [item.get_text() for item in axis.get_xticklabels()]
    bars = axis.containers[0]
    assert labels[3] == "Trading fees\n62.50 MWh gross"
    assert bars[3].get_height() == -2.5
    if import_fee:
        assert labels[4] == "Grid import fees"
        assert bars[4].get_height() == -import_fee
    assert sum(bar.get_height() for bar in bars[:-1]) == pytest.approx(bars[-1].get_height())


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
    assert "expected_dispatch_by_auction.png" not in figures
    allocation_figure = figures["pv_power_allocation.png"]
    assert len(allocation_figure.axes) == 1
    stacks = allocation_figure.axes[0].containers
    assert len(stacks) == len(case_config("pv_power_allocation")["expected"])
    np.testing.assert_allclose(np.sum([[patch.get_height() for patch in stack]
        for stack in stacks], axis=0), dispatch.pv_ac_mw)
    allocation = pd.read_csv(tmp_path / "pv_power_allocation.csv")
    np.testing.assert_allclose(allocation[["pv_to_grid_mw", "pv_to_battery_mw",
        "pv_curtailed_mw"]].sum(axis=1), allocation.pv_ac_mw)
    for auction in prices:
        name = f"reoptimization_{auction}.png"
        assert name in figures
        figure = figures[name]
        assert len(figure.axes) == 2
        assert figure.axes[0].get_shared_x_axes().joined(figure.axes[0], figure.axes[1])
        labels = [[line.get_label() for line in axis.lines] for axis in figure.axes]
        assert any(f"forecast used at {auction}" in label for label in labels[0])
        assert "Realized PV" in labels[0]
        assert any(f"scheduled at {auction}" in label for label in labels[1])
        assert "Realized battery dispatch" in labels[1]
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
