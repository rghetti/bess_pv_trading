"""Prepare own-gate IDA price histories and cancellation-aware past spread curves.

IDA3 history concatenates only its native 12:00–24:00 products. It does not invent
prices for the first half of each delivery day. All historical feature anchors
are selected at that delivery day's historical own gate.
"""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from threading import RLock

import numpy as np
import pandas as pd

from .price_feature_data import _price_sources, interval_values
from .rolling_price_data import PriceIndex, delivery_grid, pad_curve, slot_numbers
from .zero_shot_da import digest, write_json


MARKETS = ('DA', 'IDA1', 'IDA2', 'IDA3')
NORD_FEATURES = ('load_forecast', 'generation_forecast', 'solar_forecast', 'wind_onshore_forecast')


def clock(day, definition, prefix, timezone):
    """Resolve a documented local decision/publication clock with calendar offsets."""
    date = pd.Timestamp(day) + pd.Timedelta(days=definition[f'{prefix}_day_offset'])
    return pd.Timestamp(f"{date.date()} {definition[f'{prefix}_time']}", tz=timezone).tz_convert('UTC')


def auction_grid(day, market, definitions, timezone):
    """Enumerate only native auction delivery products, retaining both DST folds."""
    return delivery_grid(day, 15, timezone, definitions[market]['delivery_start_hour'])


def latest_anchor(day, market, starts, prices, definitions, notices, timezone):
    """Choose one complete earlier same-delivery curve published by the own gate."""
    cutoff = clock(day, definitions[market], 'decision', timezone)
    attempts = []
    for earlier in reversed(MARKETS[:MARKETS.index(market)]):
        definition = definitions[earlier]
        dataset = definition['dataset']
        if (dataset, str(pd.Timestamp(day).date())) in notices:
            attempts.append({'auction': earlier, 'reason': 'cancelled'})
            continue
        scheduled = clock(day, definition, 'publication', timezone)
        values = prices[earlier].lookup(starts, cutoff)
        if scheduled > cutoff or not np.isfinite(values).all():
            attempts.append({'auction': earlier, 'reason': 'unpublished_or_incomplete'})
            continue
        return values, {'auction': earlier, 'published_at': prices[earlier].latest_publication(starts, scheduled).isoformat(), 'decision_at': cutoff.isoformat(),
                        'delivery_day': str(pd.Timestamp(day).date()), 'fallbacks': attempts}
    return np.full(len(starts), np.nan), {'auction': None, 'decision_at': cutoff.isoformat(),
                                        'delivery_day': str(pd.Timestamp(day).date()), 'fallbacks': attempts}


def local_reference(day, market, starts, lag, cutoff, prices, definitions, notices, timezone):
    """Match weekly/prior-day native products and retain a missing cancelled reference."""
    prior_day = pd.Timestamp(day) - pd.Timedelta(days=lag)
    if (definitions[market]['dataset'], str(prior_day.date())) in notices:
        return np.full(len(starts), np.nan)
    prior_grid = auction_grid(prior_day, market, definitions, timezone)
    curve = pad_curve(prior_grid, prices[market].lookup(prior_grid, cutoff), timezone)
    for position in range(4):
        if not np.isfinite(curve[96 + position]):
            curve[96 + position] = curve[8 + position]
    return curve[slot_numbers(starts, timezone)]


def complete_suffix(chunks, minimum_days):
    """Keep full calendar days after the most recent missing historical chunk."""
    last_bad = max((index for index, chunk in enumerate(chunks) if not chunk['complete']), default=-1)
    retained = chunks[last_bad + 1:]
    if len(retained) < minimum_days:
        return [], 'fewer_than_minimum_complete_recent_days'
    return retained, 'trimmed_after_missing_history' if last_bad >= 0 else 'full_requested_window'


class IDAInputs:
    """Cache immutable source indices and day-level covariate profiles for a study."""

    def __init__(self, snapshot, settings, auction_settings):
        self.root = Path(snapshot)
        self.settings = settings
        self.definitions = auction_settings['auctions']
        self.timezone = settings['timezone']
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.frames = {}
        for name in ('nord_auctions', 'nord_forecasts'):
            entry = self.manifest['tables'][name]
            path = self.root / entry['path']
            if digest(path) != entry['sha256']:
                raise ValueError(f'Pinned source changed: {name}')
            self.frames[name] = pd.read_parquet(path)
        self.sources = _price_sources(self.frames['nord_auctions'], auction_settings)
        self.prices = {market: PriceIndex(frame) for market, frame in self.sources.items()}
        self.notices = {(row['dataset'], row['delivery_day']) for row in self.manifest.get('gme_cancellation_notices', [])}
        self.forecasts = {}
        for name in NORD_FEATURES:
            frame = self.frames['nord_forecasts']
            part = frame.loc[frame.dataset.eq(name) & frame.zone.eq('NORD')].sort_values('product_start')
            if part.empty or not part.unit.eq('MW').all():
                raise ValueError(f'Missing NORD forecast or wrong unit: {name}')
            self.forecasts[name] = part
        # Align each national forecast source only once; repeated daily lookups
        # then index a shared array rather than reparsing whole source frames.
        selection = self.manifest['selection']
        first = pd.Timestamp(selection['start'], tz=self.timezone).tz_convert('UTC')
        end = pd.Timestamp(selection['end_exclusive'], tz=self.timezone).tz_convert('UTC')
        self.forecast_grid = pd.date_range(first, end, freq='15min', inclusive='left')
        self.forecast_values = {name: interval_values(frame, self.forecast_grid,
            pd.to_numeric(frame.value, errors='coerce').to_numpy(float)) for name, frame in self.forecasts.items()}
        self.cache = {}
        self._cache_lock = RLock()

    def day(self, day, market):
        """Return the target day's native grid, target prices, anchors and forecasts."""
        key = (str(pd.Timestamp(day).date()), market)
        with self._cache_lock:
            if key not in self.cache:
                starts = auction_grid(day, market, self.definitions, self.timezone)
                anchor, trace = latest_anchor(day, market, starts, self.prices, self.definitions,
                                              self.notices, self.timezone)
                target = self.prices[market].lookup(starts)
                cancelled = (self.definitions[market]['dataset'], key[0]) in self.notices
                if cancelled:
                    target[:] = np.nan
                positions = self.forecast_grid.get_indexer(starts)
                present = positions >= 0
                covariates = {}
                for name, values in self.forecast_values.items():
                    profile = np.full(len(starts), np.nan)
                    profile[present] = values[positions[present]]
                    covariates[name] = profile
                self.cache[key] = {'day': key[0], 'starts': starts, 'target': target, 'anchor': anchor,
                                   'anchor_trace': trace, 'covariates': covariates, 'cancelled': cancelled}
            return self.cache[key]

    def request(self, day, market, window, variant):
        """Trim only historical full days; require complete delivery-day covariates."""
        current = self.day(day, market)
        cutoff = clock(day, self.definitions[market], 'decision', self.timezone)
        include_nord = variant in ('nord', 'nord_spread')
        include_spread = variant == 'nord_spread'
        chunks = []
        for prior in pd.date_range(pd.Timestamp(day) - pd.Timedelta(days=window), pd.Timestamp(day), inclusive='left'):
            item = self.day(prior, market)
            # Historical own-auction targets may be used only after publication
            # at the current replay decision; their anchors retain historical gates.
            values = self.prices[market].lookup(item['starts'], cutoff)
            if item['cancelled']:
                values[:] = np.nan
            complete = np.isfinite(values).all()
            if include_nord:
                complete &= all(np.isfinite(array).all() for array in item['covariates'].values())
            if include_spread:
                complete &= np.isfinite(item['anchor']).all()
            chunks.append({**item, 'values': values, 'complete': bool(complete)})
        retained, reason = complete_suffix(chunks, self.settings['minimum_context_days'])
        future_complete = (not include_nord or all(np.isfinite(array).all() for array in current['covariates'].values()))
        target_complete = np.isfinite(current['target']).any() and not current['cancelled']
        eligible = bool(retained and future_complete and target_complete)
        coverage = {'delivery_day': str(pd.Timestamp(day).date()), 'requested_context_days': window,
                    'effective_context_days': len(retained), 'trimmed_days': window - len(retained),
                    'reason': reason if retained else 'fewer_than_minimum_complete_recent_days',
                    'future_covariates_complete': future_complete, 'target_cancelled': current['cancelled'],
                    'target_observed_products': int(np.isfinite(current['target']).sum()), 'eligible': eligible}
        if not eligible:
            return None, coverage
        history_grid = pd.DatetimeIndex(np.concatenate([item['starts'].to_numpy() for item in retained]))
        context = np.concatenate([item['values'] for item in retained])
        future = {}
        if include_nord:
            future = {name: np.concatenate([*[item['covariates'][name] for item in retained], current['covariates'][name]]).tolist()
                      for name in NORD_FEATURES}
        query = history_grid.as_unit('ns').asi8
        price_index = self.prices[market]
        positions = np.searchsorted(price_index.starts, query, side='right') - 1
        maximum_publication = pd.Timestamp(int(price_index.published[positions].max()), tz='UTC')
        if maximum_publication > cutoff:
            raise AssertionError('Historical own-auction publication exceeds current own gate')
        request = {'max_context_publication_at': maximum_publication.isoformat(), 'delivery_day': current['day'], 'decision_at': cutoff.isoformat(), 'market': market, 'gate': market,
                   'context': context.tolist(), 'context_start': history_grid[0].isoformat(),
                   'context_end_exclusive': current['starts'][0].isoformat(),
                   'context_product_starts_utc': [stamp.isoformat() for stamp in history_grid],
                   'context_delivery_days': [item['day'] for item in retained],
                   'product_starts_utc': [stamp.isoformat() for stamp in current['starts']],
                   'horizon': len(current['starts']), 'feature_names': [f'past_NORD_{market}_price', *future],
                   'forecast_covariates': future, 'requested_context_days': window,
                   'effective_context_days': len(retained), 'context_policy_reason': reason,
                   'latest_cleared_anchor': current['anchor_trace']}
        if include_spread:
            request['past_covariates'] = {'own_minus_latest_cleared': np.concatenate([
                item['values'] - item['anchor'] for item in retained]).tolist()}
            request['historical_anchor_trace'] = [item['anchor_trace'] for item in retained]
        return request, coverage

    def labels(self, day, market):
        """Keep actual labels and baseline prices separate from input request files."""
        item = self.day(day, market)
        cutoff = clock(day, self.definitions[market], 'decision', self.timezone)
        return pd.DataFrame({'delivery_day': item['day'], 'product_start': item['starts'],
            'actual': item['target'], 'latest_reference': item['anchor'],
            'weekly_reference': local_reference(day, market, item['starts'], 7, cutoff, self.prices,
                                                self.definitions, self.notices, self.timezone),
            'target_cancelled': item['cancelled']})


def prepare_ida_bundle(inputs, output, market, window, variant, models):
    """Write a standalone own-gate input bundle compatible with the shared worker."""
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'Choose a fresh immutable bundle directory: {output}')
    definition = inputs.definitions[market]
    complete_days = []
    native = inputs.sources[market]
    source_days = pd.to_datetime(native.product_start, utc=True).dt.tz_convert(inputs.timezone).dt.strftime('%Y-%m-%d')
    for text in sorted(source_days.unique()):
        wanted = auction_grid(text, market, inputs.definitions, inputs.timezone)
        if np.isfinite(inputs.prices[market].lookup(wanted)).all():
            complete_days.append(text)
    if not complete_days:
        raise ValueError(f'No complete native target day for {market}')
    end = inputs.settings.get('end_exclusive') or str((pd.Timestamp(complete_days[-1]) + pd.Timedelta(days=1)).date())
    days = pd.date_range(inputs.settings['start'], end, inclusive='left')
    output.mkdir(parents=True)
    files, records, labels = {}, [], []
    features = [f'past_NORD_{market}_price', *(NORD_FEATURES if variant != 'history' else ())]
    for day in days:
        request, coverage = inputs.request(day, market, window, variant)
        records.append(coverage)
        labels.append(inputs.labels(day, market))
        if request is not None:
            path = output / 'requests' / f'{day.date()}.json'
            write_json(path, request)
            files[str(path.relative_to(output))] = digest(path)
    pd.concat(labels, ignore_index=True).to_parquet(output / 'labels.parquet', index=False)
    pd.DataFrame(records).to_csv(output / 'coverage.csv', index=False)
    shutil.copy2(inputs.root / 'manifest.json', output / 'source_manifest.json')
    # Source indices need not be copied into every bundle: the immutable parent
    # source manifest records the full native auction and forecast checksums.
    for name in ('labels.parquet', 'coverage.csv', 'source_manifest.json'):
        files[name] = digest(output / name)
    settings = {**inputs.settings, 'context_days': window, 'end_exclusive': end,
                'market': market, 'gate': market, 'auction': market,
                'decision_time': definition['decision_time'], 'decision_day_offset': definition['decision_day_offset']}
    result = {'schema': 1, 'settings': settings, 'models': models, 'files': files,
        'features': features, 'past_features': ['own_minus_latest_cleared'] if variant == 'nord_spread' else [],
        'requested_days': len(records), 'eligible_days': sum(item['eligible'] for item in records),
        'source_manifest_sha256': digest(inputs.root / 'manifest.json'),
        'context_policy': 'Longest recent complete whole-calendar-day suffix, minimum configured days; no zero fill.',
        'sequence_semantics': 'Native delivery products concatenated; IDA3 only 12:00–24:00, 48 products/day.',
        'scope': 'Own-gate feature-isolation benchmark only. DA-gate forecasts of future IDAs are not generated by this bundle.',
        'current_anchor_usage': 'Current same-delivery earlier clearing is a baseline only, not a model input; nord_spread adds historical spread curves only.',
        'spread_semantics': 'Past own price minus newest complete earlier same-delivery non-cancelled clearing published at that historical own gate. Future spread is never supplied.',
        'forecast_availability_assumption': 'Archived NORD forecasts assumed available at each gate; issue vintages unverified.',
        'pretrained_overlap': 'Possible; pretrained checkpoint releases postdate part of retrospective evaluation.',
        'november_policy': 'November 2026 reported separately; no model selection on November.'}
    write_json(output / 'manifest.json', result)
    return result
