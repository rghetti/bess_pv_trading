"""Independent ex-post pricing, including unknown and negative settlement prices."""
import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.simulation.settlement import expected_realized_kpis, settle_realized_dispatch
from battery_case_config import case_config


def settlement_case():
    case = case_config("realized_settlement")
    starts = pd.date_range(case["start"], periods=len(case["grid_mw"]),
                           freq=pd.Timedelta(minutes=case["interval_minutes"]))
    frame = pd.DataFrame({key: case[key] for key in
        ("grid_mw", "net_position_mwh", "degradation_eur", "import_fee_eur")})
    frame["start"] = starts
    frame["end"] = starts + pd.Timedelta(minutes=case["interval_minutes"])
    frame["interval_id"] = starts.astype(str)
    return case, frame


@pytest.mark.parametrize("name", case_config("expected_realized_comparison"))
def test_comparison_handles_zero_losses_and_unknown_settlement(name):
    case = case_config("expected_realized_comparison")[name]
    result = expected_realized_kpis(**case["inputs"])
    for key, expected in case["expected"].items():
        if expected is None:
            assert result[key] is None
        else:
            assert result[key] == pytest.approx(expected)


def test_unknown_prices_preserve_quantities_and_leave_corrected_profit_null():
    case, frame = settlement_case()
    result, kpis = settle_realized_dispatch(frame, auction_revenue_eur=case["auction_revenue_eur"])
    np.testing.assert_allclose(result.imbalance_mwh, case["expected"]["imbalance_mwh"])
    for key in ("profit_before_imbalance_settlement_eur", "net_imbalance_mwh",
                "absolute_imbalance_mwh", "surplus_imbalance_mwh", "shortfall_imbalance_mwh"):
        assert kpis[key] == pytest.approx(case["expected"][key])
    assert kpis["realized_profit_after_imbalance_eur"] is None
    assert kpis["realized_imbalance_cashflow_eur"] is None
    assert result.realized_imbalance_cashflow_eur.isna().all()
    assert "imbalance_mwh" not in frame


@pytest.mark.parametrize("dual_prices", [False, True])
def test_realized_cashflow_uses_signed_interval_errors_and_prices(dual_prices):
    case, frame = settlement_case()
    prefix = "dual" if dual_prices else "single"
    result, kpis = settle_realized_dispatch(frame, auction_revenue_eur=case["auction_revenue_eur"],
        imbalance_prices=case["positive_prices"] if dual_prices else case["single_prices"],
        imbalance_negative_prices=case["negative_prices"] if dual_prices else None)
    assert kpis["imbalance_settlement_status"] == "priced"
    assert result.realized_imbalance_cashflow_eur.sum() == pytest.approx(case["expected"][f"{prefix}_cashflow_eur"])
    assert kpis["realized_profit_after_imbalance_eur"] == pytest.approx(case["expected"][f"{prefix}_profit_eur"])


def test_realized_settlement_rejects_missing_or_nonfinite_price_intervals():
    case, frame = settlement_case()
    for prices in (case["single_prices"][:-1], [np.nan] * len(frame)):
        with pytest.raises(ValueError, match="aligned"):
            settle_realized_dispatch(frame, auction_revenue_eur=case["auction_revenue_eur"], imbalance_prices=prices)
    frame.loc[frame.index[-1], "interval_id"] = frame.interval_id.iloc[0]
    with pytest.raises(ValueError, match="unique"):
        settle_realized_dispatch(frame, auction_revenue_eur=case["auction_revenue_eur"])
