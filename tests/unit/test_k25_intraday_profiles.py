"""Check quarter-hour units, signed commitments and Italian clock changes."""

import pandas as pd
import pytest

from bess_pv_trading.reporting.k25_intraday_profiles import (_plot_months, aggregate_intraday, aggregate_monthly_prices,
    prepare_interval_profile, prepare_price_samples)


def test_repeated_dst_hour_uses_both_utc_intervals_and_no_netting():
    """Repeated autumn times remain two samples, including opposite positions."""
    dispatch = pd.DataFrame([
        {"start": "2026-10-25T00:00:00Z", "delivery_day": "2026-10-25", "duration_hours": 0.25,
         "curtailment_mw": 1, "imbalance_mwh": 0.2, "net_position_mwh": 0.5, "grid_mw": 2.8,
         "charge_mw": 0, "discharge_mw": 1, "energy_start_mwh": 0.4},
        {"start": "2026-10-25T01:00:00Z", "delivery_day": "2026-10-25", "duration_hours": 0.25,
         "curtailment_mw": 2, "imbalance_mwh": -0.4, "net_position_mwh": -0.5, "grid_mw": -3.6,
         "charge_mw": 1, "discharge_mw": 0, "energy_start_mwh": 0.6},
    ])
    data = prepare_interval_profile(dispatch, "Europe/Rome", 1.0)
    result = aggregate_intraday(data, ["month", "slot"]).iloc[0]
    assert result["slot"] == 8
    assert result["samples"] == 2
    assert result["avg_absolute_imbalance_mwh"] == pytest.approx(0.3)
    assert result["total_absolute_imbalance_mwh"] == pytest.approx(0.6)
    assert result["avg_curtailment_mwh"] == pytest.approx(0.375)
    assert result["avg_net_position_mwh"] == pytest.approx(0)
    assert result["avg_absolute_net_position_mwh"] == pytest.approx(0.5)
    assert result["avg_soc_start_percent"] == pytest.approx(50)
    assert result["avg_charge_energy_mwh"] == pytest.approx(0.125)
    assert result["avg_discharge_energy_mwh"] == pytest.approx(0.125)


def test_invalid_duration_and_local_delivery_day_are_rejected():
    """Data integrity failures cannot silently shift the local-time profile."""
    row = {"start": "2026-03-29T01:00:00Z", "delivery_day": "2026-03-29", "duration_hours": 1,
           "curtailment_mw": 1, "imbalance_mwh": 0, "net_position_mwh": 0, "grid_mw": 0, "charge_mw": 0,
           "discharge_mw": 0, "energy_start_mwh": 0.5}
    with pytest.raises(ValueError, match="15-minute"):
        prepare_interval_profile(pd.DataFrame([row]), "Europe/Rome", 1.0)
    row.update(duration_hours=0.25, delivery_day="2026-03-28")
    with pytest.raises(ValueError, match="delivery days"):
        prepare_interval_profile(pd.DataFrame([row]), "Europe/Rome", 1.0)


def test_price_profiles_deduplicate_controllers_and_preserve_distinct_prices():
    """Shared actual prices are counted once; different case prices are not mixed."""
    samples = pd.DataFrame([
        {"farm": "1", "label": "tracker", "interval_id": "a", "month": "2026-01", "slot": 0,
         "curve": "DA", "price_eur_per_mwh": 50},
        {"farm": "1", "label": "mpc", "interval_id": "a", "month": "2026-01", "slot": 0,
         "curve": "DA", "price_eur_per_mwh": 50},
    ])
    grouped = aggregate_monthly_prices(samples)
    assert grouped.iloc[0].samples == 1
    assert grouped.iloc[0].avg_price_eur_mwh == 50
    samples.loc[1, "price_eur_per_mwh"] = 60
    grouped = aggregate_monthly_prices(samples)
    assert set(grouped.curve) == {"DA (tracker)", "DA (mpc)"}
    assert set(grouped.avg_price_eur_mwh) == {50, 60}


def test_price_samples_retain_different_shortfall_and_surplus_prices():
    """Two settlement directions remain visible when their actual prices differ."""
    dispatch = pd.DataFrame([{"interval_id": "a", "imbalance_price_eur_mwh": 40,
                              "imbalance_negative_price_eur_mwh": 60}])
    ledger = pd.DataFrame([{"interval_id": "a", "auction": "DA", "price_eur_per_mwh": 50}])
    intervals = pd.DataFrame([{"month": "2026-01", "slot": 0}])
    samples = prepare_price_samples(dispatch, ledger, intervals, "1", "tracker")
    assert dict(zip(samples.curve, samples.price_eur_per_mwh)) == {
        "DA": 50, "Imbalance surplus": 40, "Imbalance shortfall": 60}


def test_monthly_plot_uses_only_selected_controllers(tmp_path, monkeypatch):
    """Monthly plots retain every panel and omit any unselected K=25 series."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.axes import Axes

    labels = []
    original_plot = Axes.plot

    def capture(axis, *args, **kwargs):
        labels.append(kwargs.get("label"))
        return original_plot(axis, *args, **kwargs)

    monkeypatch.setattr(Axes, "plot", capture)
    profiles = pd.DataFrame([
        {"farm": "901", "month": "2026-01", "slot": 0, "label": label,
         "avg_absolute_imbalance_mwh": 0.1, "avg_curtailment_mwh": 0.2,
         "avg_net_position_mwh": 0.3, "avg_soc_start_percent": 50}
        for label in ("Grid tracker", "MPC equal weights")])
    prices = pd.DataFrame([{"farm": "901", "month": "2026-01", "slot": 0,
                            "curve": "DA", "avg_price_eur_mwh": 100}])
    paths = _plot_months(tmp_path, profiles, prices, "Europe/Rome")
    assert len(paths) == 1 and paths[0].is_file()
    assert set(labels) == {"Grid tracker", "MPC equal weights", "DA"}
    assert labels.count("Grid tracker") == labels.count("MPC equal weights") == 4
