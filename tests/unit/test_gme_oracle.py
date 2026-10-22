"""One-day study SOC configuration preserves explicit unconstrained endpoints."""
import pytest
from zoneinfo import ZoneInfo

from bess_pv_trading.domain import BatteryParameters
from bess_pv_trading.simulation.gme_oracle import _soc_energy_targets
from battery_case_config import case_config, configured_battery


def test_soc_percent_targets_allow_null_and_respect_stored_energy_box():
    battery = configured_battery("two_mwh_soc_box")
    cases = case_config("soc_targets")
    for case in cases["valid"]:
        assert _soc_energy_targets(case, battery) == (case["initial_energy_mwh"], case["terminal_energy_mwh"])
    for case in cases["invalid"]:
        with pytest.raises(ValueError):
            _soc_energy_targets(case, battery)


def test_completed_dispatch_is_repriced_with_actual_imbalance_curve(tmp_path, monkeypatch):
    import pandas as pd
    from test_realized_settlement import settlement_case
    from bess_pv_trading.simulation import gme_oracle
    from bess_pv_trading.simulation.settlement import expected_realized_kpis
    case, frame = settlement_case()
    frame["imbalance_cashflow_eur"] = 0.0
    summary = {"initial_expected_profit_eur": case["initial_expected_profit_eur"],
        "initial_expected_auction_revenue_eur": case["initial_expected_auction_revenue_eur"],
        "components": {"auction_revenue_eur": case["auction_revenue_eur"],
        "imbalance_cashflow_eur": 0.0, "degradation_eur": float(frame.degradation_eur.sum()),
        "import_fee_eur": float(frame.import_fee_eur.sum())}}
    summary.update(expected_realized_kpis(
        initial_expected_profit_eur=case["initial_expected_profit_eur"],
        initial_expected_auction_revenue_eur=case["initial_expected_auction_revenue_eur"],
        realized_profit_after_imbalance_eur=None, auction_revenue_eur=case["auction_revenue_eur"]))
    provenance = {"supplied_to_optimizer": False}
    monkeypatch.setattr(gme_oracle, "load_realized_imbalance_prices", lambda config, grid:
        (pd.DataFrame({"settlement_price_eur_mwh": case["single_prices"]}), provenance))
    dispatch, result, _ = gme_oracle._apply_actual_settlement({}, None, frame, summary, tmp_path)
    assert result["realized_profit_after_imbalance_eur"] == pytest.approx(case["expected"]["single_profit_eur"])
    assert result["realized_operating_profit_eur"] == result["realized_profit_after_imbalance_eur"]
    assert result["components"]["imbalance_cashflow_eur"] == pytest.approx(case["expected"]["single_cashflow_eur"])
    assert dispatch.imbalance_cashflow_eur.sum() == pytest.approx(result["components"]["imbalance_cashflow_eur"])
    assert result["initial_expected_profit_eur"] == case["initial_expected_profit_eur"]
    assert result["realized_minus_expected_profit_eur"] == pytest.approx(case["expected"]["single_profit_mismatch_eur"])
    assert result["realized_minus_expected_profit_percent"] == pytest.approx(case["expected"]["single_profit_mismatch_percent"])
    assert result["realized_minus_expected_auction_revenue_eur"] == pytest.approx(case["expected"]["auction_revenue_mismatch_eur"])
    assert result["realized_minus_expected_auction_revenue_percent"] == pytest.approx(case["expected"]["auction_revenue_mismatch_percent"])
    assert (tmp_path / "imbalance_prices.csv").is_file()


@pytest.mark.parametrize("ida_status", ["quarter_hour", "canceled", "missing", "hourly_config"])
def test_native_auction_loader_separates_hourly_da_from_quarter_hour_ida(monkeypatch, ida_status):
    import numpy as np
    import pandas as pd
    from pathlib import Path
    from bess_pv_trading.config import load_config
    from bess_pv_trading.markets import AuctionCalendar
    from bess_pv_trading.simulation import gme_oracle
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/full_experiments/gme_oracle_2025-01-01.yaml")
    study = config["gme_oracle_experiment"]
    study["auction_datasets"] = {name: spec for name, spec in study["auction_datasets"].items()
                                 if name in ("DA", "IDA1")}
    if ida_status == "hourly_config":
        study["auction_datasets"]["IDA1"]["native_interval_minutes"] = 60
    calendar = AuctionCalendar.from_config(config)
    grid = calendar.delivery_grid(study["delivery_day"]).to_domain()
    case = load_config(root / "tests/fixtures/configs/gme_workbook_cases.yaml")
    price_pattern = np.asarray(case["prices_eur_mwh"])
    def read_prices(database, *, dataset, **kwargs):
        if dataset == "ida1_prices" and ida_status in ("canceled", "missing"):
            return pd.DataFrame(), [{"cancelled_delivery_days": [study["delivery_day"]]
                                     if ida_status == "canceled" else []}]
        minutes = 60 if dataset == "day_ahead_prices" else 15
        times = pd.date_range(grid.starts[0], grid.ends[-1], freq=f"{minutes}min", inclusive="left").tz_convert(ZoneInfo("UTC")).as_unit("ns")
        return pd.DataFrame({"product_start": times, "product_end": times+pd.Timedelta(minutes=minutes),
            "native_interval_minutes": minutes, "price_eur_mwh": np.resize(price_pattern, len(times))}), []
    monkeypatch.setattr(gme_oracle, "load_api_cache_records", read_prices)
    if ida_status in ("missing", "hourly_config"):
        with pytest.raises(ValueError, match="no native prices|requires native 15-minute"):
            gme_oracle._load_auction_prices(config, calendar, grid, calendar.schedule(study["delivery_day"]))
        return
    actual, _, skipped = gme_oracle._load_auction_prices(config, calendar, grid,
        calendar.schedule(study["delivery_day"]))
    np.testing.assert_allclose(actual["DA"][:len(price_pattern)], price_pattern[0])
    if ida_status == "canceled":
        assert set(actual) == {"DA"}
        assert skipped[0]["reason"] == "documented_cancelled_auction"
    else:
        np.testing.assert_allclose(actual["IDA1"][:len(price_pattern)], price_pattern)


def test_native_auction_loader_uses_quarter_hour_da_after_transition(monkeypatch):
    import numpy as np
    import pandas as pd
    from datetime import date
    from pathlib import Path
    from bess_pv_trading.config import load_config
    from bess_pv_trading.data.gme import native_price_resolution_minutes
    from bess_pv_trading.markets import AuctionCalendar
    from bess_pv_trading.simulation import gme_oracle
    root = Path(__file__).resolve().parents[2]
    config = load_config(root / "configs/full_experiments/gme_oracle_2025-01-01.yaml")
    study = config["gme_oracle_experiment"]
    day = date(2026, 1, 1)
    study["delivery_day"] = day.isoformat()
    config["market"].update(valid_from="2025-10-01", valid_to=day.isoformat())
    study["auction_datasets"] = {name: spec for name, spec in study["auction_datasets"].items()
                                 if name in ("DA", "IDA1")}
    calendar = AuctionCalendar.from_config(config)
    grid = calendar.delivery_grid(day).to_domain()
    case = load_config(root / "tests/fixtures/configs/gme_workbook_cases.yaml")
    price_pattern = np.asarray(case["prices_eur_mwh"])

    def read_prices(database, *, dataset, **kwargs):
        minutes = native_price_resolution_minutes(dataset, day)
        times = pd.date_range(grid.starts[0], grid.ends[-1], freq=f"{minutes}min", inclusive="left")
        return pd.DataFrame({"product_start": times, "product_end": times+pd.Timedelta(minutes=minutes),
            "native_interval_minutes": minutes, "price_eur_mwh": np.resize(price_pattern, len(times))}), []

    monkeypatch.setattr(gme_oracle, "load_api_cache_records", read_prices)
    actual, provenance, _ = gme_oracle._load_auction_prices(
        config, calendar, grid, calendar.schedule(day))
    expected_resolution = native_price_resolution_minutes("day_ahead_prices", day)
    assert provenance["DA"]["native_interval_minutes"] == expected_resolution
    assert provenance["DA"]["eligible_products"] == len(grid.starts) // (expected_resolution // 15)
    assert len(actual["DA"]) == len(grid.starts)
