"""Test historical missing-data trimming without inventing future covariates."""

import numpy as np
import pandas as pd

from bess_pv_trading.forecasting.studies.adaptive_price_context import adapt_request


def example(first='2025-08-02', end='2025-10-01'):
    """Make full local-day contexts and one future-day covariate profile."""
    start = pd.Timestamp(first, tz='Europe/Rome')
    stop = pd.Timestamp(end, tz='Europe/Rome')
    times = pd.date_range(start, stop, freq='15min', inclusive='left').tz_convert('UTC')
    return {'context_start': times[0].isoformat(), 'context_end_exclusive': stop.tz_convert('UTC').isoformat(),
            'context': [1.] * len(times), 'context_native_minutes': [15] * len(times), 'horizon': 96,
            'forecast_covariates': {'load': [2.] * (len(times) + 96)},
            'past_covariates': {'spread': [3.] * len(times)}}


def test_two_august_gaps_shorten_sixty_days_and_preserve_horizon():
    """Missing August 10–11 leaves the complete August12–September30 suffix."""
    request = example()
    times = pd.date_range(pd.Timestamp(request['context_start']), pd.Timestamp(request['context_end_exclusive']), freq='15min', inclusive='left')
    local = times.tz_convert('Europe/Rome')
    missing = np.flatnonzero((local.strftime('%Y-%m-%d') >= '2025-08-10') & (local.strftime('%Y-%m-%d') <= '2025-08-11'))
    for index in missing:
        request['forecast_covariates']['load'][index] = float('nan')
    adjusted, info = adapt_request(request)
    assert info['requested_context_days'] == 60
    assert info['effective_context_days'] == 50
    assert pd.Timestamp(adjusted['context_start']).tz_convert('Europe/Rome').strftime('%Y-%m-%d %H:%M') == '2025-08-12 00:00'
    assert adjusted['forecast_covariates']['load'][-96:] == [2.] * 96
    assert len(adjusted['past_covariates']['spread']) == len(adjusted['context'])
    assert len(adjusted['context_native_minutes']) == len(adjusted['context'])


def test_missing_horizon_never_fixed_by_shortening_history():
    request = example()
    request['forecast_covariates']['load'][-1] = float('nan')
    adjusted, info = adapt_request(request)
    assert adjusted is None
    assert info['exclusion_reason'] == 'missing_delivery_covariates'


def test_recent_gap_excludes_day_below_minimum():
    request = example()
    request['context'][-200] = float('nan')
    adjusted, info = adapt_request(request, minimum_days=7)
    assert adjusted is None
    assert info['exclusion_reason'] == 'insufficient_recent_complete_days'


def test_calendar_days_are_not_step_count_across_dst():
    request = example('2025-10-01', '2025-11-01')
    adjusted, info = adapt_request(request)
    assert info['effective_context_days'] == 31
    assert len(adjusted['context']) == 31 * 96 + 4


def test_missing_price_and_past_only_feature_share_the_same_trim():
    request = example()
    request['context'][100] = float('nan')
    request['past_covariates']['spread'][500] = float('nan')
    adjusted, info = adapt_request(request)
    assert adjusted is not None and info['trimmed_products'] == 576
    assert all(np.isfinite(adjusted['context']))
    assert all(np.isfinite(adjusted['past_covariates']['spread']))
    assert len(adjusted['forecast_covariates']['load']) == len(adjusted['context']) + 96


def test_ida3_half_day_sequence_uses_explicit_timestamps():
    """A missing afternoon does not create fake overnight prices during trimming."""
    days = pd.date_range('2025-10-01', '2025-10-20', inclusive='left')
    stamps = []
    for day in days:
        first = (day + pd.Timedelta(hours=12)).tz_localize('Europe/Rome')
        end = (day + pd.Timedelta(days=1)).tz_localize('Europe/Rome')
        stamps.extend(pd.date_range(first, end, freq='15min', inclusive='left').tz_convert('UTC'))
    request = {'context_start': stamps[0].isoformat(), 'context_end_exclusive': '2025-10-20T10:00:00Z',
               'context_product_starts_utc': [stamp.isoformat() for stamp in stamps],
               'context': [1.] * len(stamps), 'horizon': 48, 'past_covariates': {'spread': [2.] * len(stamps)}}
    request['past_covariates']['spread'][48 * 5] = float('nan')
    adjusted, info = adapt_request(request)
    assert info['effective_context_days'] == 13
    assert len(adjusted['context']) == 13 * 48
    assert len(adjusted['context_product_starts_utc']) == 13 * 48
