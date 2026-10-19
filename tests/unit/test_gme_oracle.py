"""One-day study SOC configuration preserves explicit unconstrained endpoints."""
import pytest

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
