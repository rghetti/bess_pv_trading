"""Readable, reconciled period summaries for continuous PV and battery replays."""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


SUMMED_METRICS = (
    'net_profit_eur', 'auction_revenue_eur', 'imbalance_cashflow_eur',
    'degradation_eur', 'import_fee_eur', 'transaction_fee_eur',
    'signed_imbalance_mwh', 'absolute_imbalance_mwh', 'pv_realized_energy_mwh',
    'charge_mwh', 'discharge_mwh', 'curtailment_mwh', 'fallback_count',
)


def _display_number(value: float, decimals: int) -> str:
    """Suppress negative zero introduced by numerical solver roundoff."""
    if abs(value) < 0.5 * 10 ** -decimals:
        value = 0.0
    return f'{value:,.{decimals}f}'


def aggregate_continuous_daily(daily: pd.DataFrame, *, strategy_column: str) -> pd.DataFrame:
    """Sum actual daily quantities, rejecting missing values rather than hiding them."""
    required = {strategy_column, 'delivery_day', *SUMMED_METRICS}
    if daily.empty or not required.issubset(daily.columns):
        raise ValueError(f'Continuous summary is missing columns: {sorted(required - set(daily.columns))}')
    if daily[strategy_column].isna().any() or daily.delivery_day.isna().any():
        raise ValueError('Continuous summary needs a strategy and delivery day for each row')
    if daily.duplicated([strategy_column, 'delivery_day']).any():
        raise ValueError('Continuous summary cannot contain duplicate strategy and delivery day rows')
    values = daily[list(SUMMED_METRICS)].apply(pd.to_numeric, errors='raise')
    if not np.isfinite(values.to_numpy(dtype=float)).all():
        raise ValueError('Continuous summary requires finite daily accounting values')
    if (values.fallback_count < 0).any() or not np.equal(values.fallback_count, np.floor(values.fallback_count)).all():
        raise ValueError('Fallback counts must be nonnegative integers')
    frame = daily.copy()
    frame[list(SUMMED_METRICS)] = values
    return frame.groupby(strategy_column, sort=False)[list(SUMMED_METRICS)].sum().reset_index()


def write_continuous_markdown(path: Path, daily: pd.DataFrame, *, farm_id: int,
                              strategy_column: str = 'strategy',
                              labels: Mapping[str, str] | None = None,
                              solves_by_strategy: Mapping[str, Sequence[dict] | None] | None = None) -> None:
    """Write period totals with explicit units and visible fallback/solver status."""
    totals = aggregate_continuous_daily(daily, strategy_column=strategy_column)
    days = sorted(str(day) for day in daily.delivery_day.unique())
    if not days:
        raise ValueError('Continuous summary needs delivery days')
    labels = labels or {}
    lines = [f'# Farm {farm_id}: continuous replay summary', '',
             f'Local delivery dates: {days[0]} to {days[-1]} ({len(days)} days).', '',
             '| Farm | Strategy | Profit (€) | Auction revenue (€) | Imbalance settlement (€) | Battery wear (€) | Fees (€) | Signed imbalance (MWh) | Absolute imbalance (MWh) | PV production (MWh) | Battery charge (MWh) | Battery discharge (MWh) | Curtailment (MWh) | Fallbacks | Non-optimal solves |',
             '| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in totals.to_dict('records'):
        strategy = str(row[strategy_column])
        label = str(labels.get(strategy, strategy)).replace('|', r'\|').replace('\n', ' ')
        solves = None if solves_by_strategy is None else solves_by_strategy.get(strategy)
        if solves is None or any(not solve.get('status') for solve in solves if not solve.get('rolled_back_for_resume')):
            nonoptimal = 'unknown'
        else:
            nonoptimal = str(sum(solve['status'] != 'optimal' for solve in solves
                                 if not solve.get('rolled_back_for_resume')))
        fees = row['import_fee_eur'] + row['transaction_fee_eur']
        money = [row['net_profit_eur'], row['auction_revenue_eur'], row['imbalance_cashflow_eur'],
                 row['degradation_eur'], fees]
        energy = [row['signed_imbalance_mwh'], row['absolute_imbalance_mwh'],
                  row['pv_realized_energy_mwh'], row['charge_mwh'], row['discharge_mwh'], row['curtailment_mwh']]
        cells = [str(farm_id), label, *(_display_number(value, 2) for value in money),
                 *(_display_number(value, 3) for value in energy), str(int(row['fallback_count'])), nonoptimal]
        lines.append('| ' + ' | '.join(cells) + ' |')
    lines += ['', 'Profit is realized auction revenue plus actual imbalance settlement, less executed battery wear, fees and any recorded realized imbalance penalty. '
              'Fees combine import and market transaction fees. Positive signed imbalance means surplus net export; '
              'absolute imbalance sums interval magnitudes.',
              '', 'Fallbacks count decisions that retained an earlier plan; non-optimal solves include time limits and '
              'infeasibility, excluding attempts rolled back on resume. A perfect-PV reference, when present, is an '
              'information benchmark and is not guaranteed to bound realized profit. Optimal solver status is subject '
              'to the configured MIP gap and numerical tolerances.', '']
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')
