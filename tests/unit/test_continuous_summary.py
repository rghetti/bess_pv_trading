"""Period tables preserve signed deviations, cash accounting and solve failures."""

import numpy as np
import pandas as pd
import pytest

from bess_pv_trading.reporting.continuous_summary import aggregate_continuous_daily, write_continuous_markdown


def daily_rows():
    """Two days with offsetting imbalances and one retained-plan fallback."""
    base = {'strategy': 'integrated', 'auction_revenue_eur': 100., 'imbalance_cashflow_eur': -5.,
            'degradation_eur': 4., 'import_fee_eur': 1., 'transaction_fee_eur': 2.,
            'net_profit_eur': 88., 'pv_realized_energy_mwh': 3., 'charge_mwh': 0.4,
            'discharge_mwh': 0.3, 'curtailment_mwh': 0.2}
    return pd.DataFrame([{**base, 'delivery_day': '2026-01-01', 'signed_imbalance_mwh': -0.25,
                          'absolute_imbalance_mwh': 0.25, 'fallback_count': 1},
                         {**base, 'delivery_day': '2026-01-02', 'signed_imbalance_mwh': 0.25,
                          'absolute_imbalance_mwh': 0.25, 'fallback_count': 0}])


def test_monthly_markdown_reports_actual_totals_and_solver_status(tmp_path):
    daily = daily_rows()
    totals = aggregate_continuous_daily(daily, strategy_column='strategy').iloc[0]
    assert totals.net_profit_eur == pytest.approx(176.)
    assert totals.signed_imbalance_mwh == pytest.approx(0.)
    assert totals.absolute_imbalance_mwh == pytest.approx(0.5)
    assert totals.charge_mwh == pytest.approx(0.8)
    assert totals.curtailment_mwh == pytest.approx(0.4)
    path = tmp_path / 'summary.md'
    statuses = {'integrated': [{'status': 'optimal'}, {'status': 'infeasible'},
                               {'status': 'feasible_time_limit'},
                               {'status': 'infeasible', 'rolled_back_for_resume': True}]}
    write_continuous_markdown(path, daily, farm_id=345, solves_by_strategy=statuses)
    text = path.read_text()
    assert '2026-01-01 to 2026-01-02 (2 days)' in text
    assert '| 345 | integrated | 176.00 | 200.00 | -10.00 | 8.00 | 6.00 | 0.000 | 0.500 | 6.000 | 0.800 | 0.600 | 0.400 | 1 | 2 |' in text
    assert 'information benchmark' in text


def test_summary_rejects_missing_values_and_duplicate_days(tmp_path):
    daily = daily_rows()
    daily.loc[0, 'curtailment_mwh'] = np.nan
    with pytest.raises(ValueError, match='finite daily accounting'):
        write_continuous_markdown(tmp_path / 'missing.md', daily, farm_id=1)
    daily = daily_rows()
    daily.loc[1, 'delivery_day'] = daily.loc[0, 'delivery_day']
    with pytest.raises(ValueError, match='duplicate'):
        aggregate_continuous_daily(daily, strategy_column='strategy')
    assert not (tmp_path / 'missing.md').exists()
