"""Check the shared physical and financial basis for baseline comparisons."""
import importlib.util
import sys
from pathlib import Path

import pandas as pd
import pytest

from bess_pv_trading.domain import NaumannSettings
from bess_pv_trading.optimization.naumann import calendar_increment


scripts = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(scripts))
spec = importlib.util.spec_from_file_location("compare_aging_baseline", scripts / "compare_aging_baseline.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_stationary_path_uses_calendar_and_cash_only_once():
    """A stationary battery has calendar loss but no cycling loss."""
    settings = NaumannSettings(fixed_cycle_rate_c=.5)
    frame = pd.DataFrame({"energy_start_mwh": [.5], "energy_end_mwh": [.5],
                          "duration_hours": [1.], "auction_revenue_eur": [100.],
                          "imbalance_cashflow_eur": [-10.], "import_fee_eur": [1.],
                          "transaction_fee_eur": [2.], "degradation_eur": [50.]})
    result = module.common_proxy(frame, 1., settings, 1000.)
    loss = calendar_increment(.5, .5, 3600., settings) ** .5
    assert result["common_calendar_loss_fraction"] == pytest.approx(loss)
    assert result["common_cycle_loss_fraction"] == 0
    assert result["market_cash_eur"] == 87.
    assert result["cash_minus_common_wear_proxy_eur"] == pytest.approx(87. - 5000. * loss)
