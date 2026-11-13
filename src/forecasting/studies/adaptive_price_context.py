"""Keep a complete suffix of historical inputs without inventing observations."""

from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd


def adapt_request(request, minimum_days=7, timezone='Europe/Rome', policy='adaptive'):
    """Trim through the last bad historical local day; never shorten the horizon.

    DA histories use the uniform quarter-hour grid. Explicit history timestamps
    also support auction histories containing only the products actually traded.
    All target, past-only and known-future covariates share the exact same slice.
    """
    if policy not in ('adaptive', 'exclude_day') or minimum_days < 1:
        raise ValueError('Invalid missing-context policy or minimum days')
    context = np.asarray(request['context'], dtype=float)
    horizon = int(request['horizon'])
    timestamps = request.get('context_starts_utc', request.get('context_product_starts_utc'))
    if timestamps is not None:
        starts = pd.DatetimeIndex(pd.to_datetime(timestamps, utc=True))
    else:
        starts = pd.date_range(pd.Timestamp(request['context_start']).tz_convert('UTC'),
            pd.Timestamp(request['context_end_exclusive']).tz_convert('UTC'), freq='15min', inclusive='left')
    if len(starts) != len(context) or not len(starts) or not starts.is_monotonic_increasing or not starts.is_unique:
        raise ValueError('Context timestamps must match a nonempty ordered history')
    bad = ~np.isfinite(context)
    horizon_missing = 0
    for name, values in request.get('forecast_covariates', {}).items():
        values = np.asarray(values, dtype=float)
        if len(values) != len(context) + horizon:
            raise ValueError(f'Wrong covariate width: {name}')
        bad |= ~np.isfinite(values[:len(context)])
        horizon_missing += int((~np.isfinite(values[len(context):])).sum())
    for name, values in request.get('past_covariates', {}).items():
        values = np.asarray(values, dtype=float)
        if len(values) != len(context):
            raise ValueError(f'Wrong past-only covariate width: {name}')
        bad |= ~np.isfinite(values)
    local = starts.tz_convert(timezone)
    end_day = pd.Timestamp(request['context_end_exclusive']).tz_convert(timezone).date()
    requested_days = (end_day - local[0].date()).days
    info = {'requested_context_days': request.get('requested_context_days', requested_days),
            'input_context_days': requested_days, 'historical_missing_products': int(bad.sum()),
            'horizon_missing_values': horizon_missing, 'trimmed_products': 0,
            'effective_context_days': requested_days, 'context_policy': policy, 'exclusion_reason': None}
    if horizon_missing:
        info['exclusion_reason'] = 'missing_delivery_covariates'
        return None, info
    offset = 0
    if bad.any():
        if policy == 'exclude_day':
            info['exclusion_reason'] = 'missing_historical_inputs'
            return None, info
        last_bad_day = local[np.flatnonzero(bad)[-1]].date()
        candidates = np.flatnonzero(np.asarray(local.date) > last_bad_day)
        offset = int(candidates[0]) if len(candidates) else len(context)
    effective = (end_day - local[offset].date()).days if offset < len(context) else 0
    info.update(trimmed_products=offset, effective_context_days=effective)
    if effective < minimum_days:
        info['exclusion_reason'] = 'insufficient_recent_complete_days'
        return None, info
    result = deepcopy(request)
    result['context'] = result['context'][offset:]
    result['context_start'] = starts[offset].isoformat()
    for field in ('context_native_minutes', 'context_starts_utc', 'context_product_starts_utc'):
        if field in result:
            result[field] = result[field][offset:]
    for values in ('forecast_covariates', 'past_covariates'):
        if values in result:
            result[values] = {name: value[offset:] for name, value in result[values].items()}
    result['requested_context_days'] = info['requested_context_days']
    result['effective_context_days'] = effective
    result['context_adjustment'] = info
    return result, info
