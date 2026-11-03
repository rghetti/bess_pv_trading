"""Realtime detailed plots have a three-day cap and remove obsolete day figures."""

import json

import pandas as pd

from bess_pv_trading.config import load_config
from bess_pv_trading.reporting.continuous_outputs import _plot_strategy, realtime_daily_plot_limit
from bess_pv_trading.reporting.realtime_mpc import _comparison_plots


def five_days():
    """Small artificial plotting data needs no market replay or optimization."""
    starts = pd.date_range("2026-01-01T10:00:00Z", periods=5, freq="D")
    return pd.DataFrame({"start": starts, "end": starts + pd.Timedelta(minutes=15),
        "delivery_day": starts.strftime("%Y-%m-%d"), "energy_start_mwh": 0.375,
        "energy_end_mwh": 0.375, "pv_ac_mw": 0.1, "pv_ac_forecast_mw": 0.1,
        "charge_mw": 0.0, "discharge_mw": 0.0, "requested_charge_mw": 0.0,
        "requested_discharge_mw": 0.0, "grid_mw": 0.1, "net_position_mwh": 0.025,
        "duration_hours": 0.25, "cumulative_net_profit_eur": 0.0, "net_profit_eur": 0.0,
        "imbalance_mwh": 0.0, "imbalance_price_eur_mwh": 10.0,
        "executed_battery_mw": 0.0, "planned_battery_mw": 0.0})


def test_per_controller_day_figures_are_capped_and_stale_figures_removed(tmp_path):
    config = load_config("configs/full_experiments/realtime_continuous_smoke.yaml")
    config["realtime_mpc_comparison"]["representative_days"] = 20
    assert realtime_daily_plot_limit(config) == 3
    frame = five_days()
    daily = tmp_path / "daily"
    daily.mkdir()
    (daily / "2026-01-05.png").write_bytes(b"stale generated figure")
    _plot_strategy(config, tmp_path, "deterministic_mpc", frame, pd.DataFrame())
    assert sorted(path.stem for path in daily.glob("*.png")) == ["2026-01-01", "2026-01-02", "2026-01-03"]
    legacy = load_config("configs/standard.yaml")
    assert realtime_daily_plot_limit(legacy) is None


def test_comparison_day_figures_use_the_same_cap(tmp_path):
    config = load_config("configs/full_experiments/realtime_continuous_smoke.yaml")
    config["realtime_mpc_comparison"]["representative_days"] = 20
    frame = five_days()
    daily = pd.DataFrame({"delivery_day": frame.delivery_day, "controller": "track_committed_grid",
        "imbalance_cashflow_eur": 0.0, "profit_minus_tracking_eur": 0.0,
        "absolute_imbalance_mwh": 0.0, "realized_operating_profit_eur": 0.0})
    source = tmp_path / "controller"
    (source / "csv").mkdir(parents=True)
    (source / "json").mkdir()
    pd.DataFrame(columns=["boundary_kind", "prediction_error_mwh", "cutoff_utc"]).to_csv(source / "csv/state_projections.csv", index=False)
    (source / "json/mpc_solves.json").write_text(json.dumps([]))
    output = tmp_path / "plots"
    output.mkdir()
    (output / "soc_and_planned_executed_power_2026-01-05.png").write_bytes(b"stale generated figure")
    _comparison_plots(config, tmp_path, daily, {"track_committed_grid": frame},
        {"track_committed_grid": {"output_dir": str(source)}}, [])
    assert len(list(output.glob("soc_and_planned_executed_power_*.png"))) == 3
