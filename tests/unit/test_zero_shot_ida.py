"""Verify native IDA sequences, historical spread anchors and context trimming."""

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.rolling_price_data import PriceIndex
from bess_pv_trading.forecasting.studies.zero_shot_ida import auction_grid, latest_anchor, complete_suffix


DEFINITIONS = {
    'DA': {'dataset': 'day_ahead_prices', 'decision_time': '11:45', 'decision_day_offset': -1,
           'publication_time': '12:57', 'publication_day_offset': -1, 'delivery_start_hour': 0},
    'IDA1': {'dataset': 'ida1_prices', 'decision_time': '14:45', 'decision_day_offset': -1,
             'publication_time': '15:30', 'publication_day_offset': -1, 'delivery_start_hour': 0},
    'IDA2': {'dataset': 'ida2_prices', 'decision_time': '21:45', 'decision_day_offset': -1,
             'publication_time': '22:30', 'publication_day_offset': -1, 'delivery_start_hour': 0},
    'IDA3': {'dataset': 'ida3_prices', 'decision_time': '09:45', 'decision_day_offset': 0,
             'publication_time': '10:30', 'publication_day_offset': 0, 'delivery_start_hour': 12},
}


def price_index(grid, value, publication):
    """Create a native complete earlier-auction source with an explicit publication."""
    return PriceIndex(pd.DataFrame({'product_start': grid, 'product_end': grid + pd.Timedelta(minutes=15),
                                   'available_at': pd.Timestamp(publication), 'price_eur_per_mwh': value}))


def test_ida3_has_only_native_half_day_products_and_full_day_idas_keep_dst():
    """No overnight IDA3 zeros or duplicated fake products enter the model sequence."""
    for day in ('2026-03-29', '2026-10-25', '2026-07-01'):
        half = auction_grid(day, 'IDA3', DEFINITIONS, 'Europe/Rome')
        assert len(half) == 48
        assert half.tz_convert('Europe/Rome').hour.min() == 12
    assert len(auction_grid('2026-03-29', 'IDA1', DEFINITIONS, 'Europe/Rome')) == 92
    assert len(auction_grid('2026-10-25', 'IDA2', DEFINITIONS, 'Europe/Rome')) == 100


def test_cancelled_latest_anchor_falls_back_same_delivery_to_next_earlier_auction():
    """IDA3 uses IDA1 after a documented IDA2 cancellation, rather than yesterday's IDA2."""
    day = '2026-07-02'
    grid = auction_grid(day, 'IDA3', DEFINITIONS, 'Europe/Rome')
    prices = {market: price_index(grid, value, '2026-07-01T10:00Z')
              for market, value in [('DA', 100.), ('IDA1', 110.), ('IDA2', 120.)]}
    values, trace = latest_anchor(day, 'IDA3', grid, prices, DEFINITIONS,
                                  {('ida2_prices', day)}, 'Europe/Rome')
    assert np.all(values == 110.)
    assert trace['auction'] == 'IDA1'
    assert trace['delivery_day'] == day
    assert trace['fallbacks'] == [{'auction': 'IDA2', 'reason': 'cancelled'}]


def test_unpublished_anchor_is_never_visible_at_own_gate():
    """Recorded publication later than IDA3 gate forces an earlier available source."""
    day = '2026-07-02'
    grid = auction_grid(day, 'IDA3', DEFINITIONS, 'Europe/Rome')
    prices = {market: price_index(grid, value, publication) for market, value, publication in
              [('DA', 100., '2026-07-01T11:00Z'), ('IDA1', 110., '2026-07-01T14:00Z'),
               ('IDA2', 120., '2026-07-02T09:00Z')]}
    values, trace = latest_anchor(day, 'IDA3', grid, prices, DEFINITIONS, set(), 'Europe/Rome')
    assert trace['auction'] == 'IDA1'
    assert np.all(values == 110.)


def test_complete_suffix_retains_calendar_days_and_minimum_history():
    """Missing auction days break the history; they cannot be removed from its middle."""
    chunks = [{'day': day, 'complete': day != 3} for day in range(12)]
    retained, reason = complete_suffix(chunks, 7)
    assert [item['day'] for item in retained] == list(range(4, 12))
    assert reason == 'trimmed_after_missing_history'
    retained, reason = complete_suffix(chunks, 9)
    assert retained == []


def test_past_spread_has_only_historical_products_and_cancellation_aware_values():
    """A cancelled historical IDA2 changes its same-day IDA3 spread anchor to IDA1."""
    from bess_pv_trading.forecasting.studies.zero_shot_ida import IDAInputs
    inputs = object.__new__(IDAInputs)
    inputs.timezone = 'Europe/Rome'
    inputs.settings = {'minimum_context_days': 7}
    inputs.definitions = DEFINITIONS
    inputs.notices = {('ida2_prices', '2026-07-01')}
    inputs.cache = {}
    full_grid = pd.date_range('2026-06-20', '2026-07-03', tz='Europe/Rome', freq='15min', inclusive='left').tz_convert('UTC')
    inputs.forecast_grid = full_grid
    inputs.forecast_values = {name: np.ones(len(full_grid)) for name in
                             ('load_forecast', 'generation_forecast', 'solar_forecast', 'wind_onshore_forecast')}
    prices = {}
    for market, value in [('DA', 100.), ('IDA1', 110.), ('IDA2', 120.), ('IDA3', 130.)]:
        rows = []
        for day in pd.date_range('2026-06-20', '2026-07-03', inclusive='left'):
            grid = auction_grid(day, market, DEFINITIONS, 'Europe/Rome')
            publication_day = day + pd.Timedelta(days=DEFINITIONS[market]['publication_day_offset'])
            publication = pd.Timestamp(f"{publication_day.date()} {DEFINITIONS[market]['publication_time']}", tz='Europe/Rome').tz_convert('UTC')
            rows.append(pd.DataFrame({'product_start': grid, 'product_end': grid + pd.Timedelta(minutes=15),
                                      'available_at': publication, 'price_eur_per_mwh': value}))
        prices[market] = PriceIndex(pd.concat(rows, ignore_index=True))
    inputs.prices = prices
    request, coverage = inputs.request('2026-07-02', 'IDA3', 7, 'nord_spread')
    assert coverage['eligible']
    assert len(request['context']) == 7 * 48
    spread = request['past_covariates']['own_minus_latest_cleared']
    assert len(spread) == len(request['context'])
    assert spread[-48:] == [20.] * 48
    assert spread[:-48] == [10.] * (6 * 48)
    assert request['historical_anchor_trace'][-1]['auction'] == 'IDA1'
    assert all(len(values) == len(spread) + 48 for values in request['forecast_covariates'].values())
    assert pd.Timestamp(request['max_context_publication_at']) < pd.Timestamp(request['decision_at'])


def test_ida_comparison_writes_markdown_and_spread_feature_differences(tmp_path, monkeypatch):
    """All IDA settings compare identical products and automatic reports include deltas."""
    import json
    import sys
    from scripts import report_ida_zero_shot
    settings = {'output': str(tmp_path), 'auctions': ['IDA3'], 'windows': [7, 28, 60],
                'variants': ['history', 'nord', 'nord_spread']}
    monkeypatch.setattr(report_ida_zero_shot, 'load_config', lambda path: {'zero_shot_ida': settings})
    monkeypatch.setattr(sys, 'argv', ['report'])
    for window in settings['windows']:
        for index, variant in enumerate(settings['variants']):
            directory = tmp_path / 'IDA3' / f'{window}d' / variant / 'run/report'
            directory.mkdir(parents=True)
            (directory / 'status.json').write_text(json.dumps({'complete': True}))
            pd.DataFrame([{'delivery_day': '2026-07-02', 'product_start': '2026-07-02T10:00Z',
                           'auction': 'IDA3', 'gate': 'IDA3', 'variant': 'model',
                           'period': 'retrospective_development', 'actual': 10.,
                           'prediction': 13. - index}]).to_parquet(directory / 'matched_forecasts.parquet')
    report_ida_zero_shot.main()
    comparison = tmp_path / 'IDA3/comparison'
    scores = pd.read_csv(comparison / 'scores.csv')
    assert scores.products.eq(1).all() and len(scores) == 9
    delta = pd.read_csv(comparison / 'paired_deltas.csv')
    assert len(delta) == 6 and delta.mae_change.eq(-1.).all()
    text = (comparison / 'summary.md').read_text()
    assert 'nord_spread_minus_nord' in text
    assert 'November' in text
