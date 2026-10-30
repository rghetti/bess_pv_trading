"""Gate-causal quantile fitting and delayed online replay for one farm and auction."""
from __future__ import annotations

from collections import defaultdict
import heapq
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from ...data import night_window_mask
from ..online.pv_online_crps import MLPoly
from ..data.pv_probabilistic_data import file_hash, write_json, probabilistic_night_policy


def _features(frame, rating):
    local = pd.DatetimeIndex(frame.interval_start_utc).tz_convert('Europe/Rome') + pd.Timedelta(minutes=7.5)
    hour = local.hour.to_numpy() + local.minute.to_numpy()/60
    return np.column_stack([frame.deterministic_prediction.to_numpy(float)/rating,
                            np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)])


def deterministic_quantiles(history, current, levels, rating, alpha, night_policy):
    """Fit quantile regressions to past deterministic errors and predict this issue."""
    scaler = StandardScaler().fit(_features(history, rating))
    x = scaler.transform(_features(history, rating))
    residual = (history.measured_ac_mw.to_numpy(float)-history.deterministic_prediction.to_numpy(float))/rating
    valid = np.isfinite(_features(current, rating)).all(axis=1)
    raw = np.full((len(current), len(levels)), np.nan)
    # When `valid.any()`, set `xp` using `scaler.transform(_features(current.loc[valid], rating))`.
    if valid.any():
        xp = scaler.transform(_features(current.loc[valid], rating))
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
            # Apply fit for `(j, q)` from `enumerate(levels)`.
            for j, q in enumerate(levels):
                model = QuantileRegressor(quantile=q, alpha=alpha, solver='highs', solver_options={'threads': 1})
                model.fit(x, residual)
                raw[valid, j] = current.loc[valid, 'deterministic_prediction'].to_numpy(float) + rating*model.predict(xp)
    crossed = np.zeros(len(current), dtype=bool)
    bounded = np.zeros(len(current), dtype=bool)
    crossed[valid] = np.any(np.diff(raw[valid], axis=1) < 0, axis=1)
    bounded[valid] = np.any((raw[valid] < 0) | (raw[valid] > rating), axis=1)
    corrected = np.sort(np.clip(raw, 0, rating), axis=1)
    night = night_window_mask(pd.DatetimeIndex(current.interval_start_utc), night_policy) & valid
    corrected[night] = 0.
    return corrected, raw, crossed, bounded, night


def _learner(learners, key, n):
    # When `key not in learners`, set `learners[key]` using `MLPoly(n)`.
    if key not in learners: learners[key] = MLPoly(n)
    return learners[key]


def evaluate_farm_auction(config, source_config, farm, auction, output_root):
    """Replay chronological issue events with separate warm and cold learners."""
    settings = config['pv_online_crps_study']
    destination = Path(output_root) / f'farm_{farm}'
    manifest = json.loads((destination / 'prepared_manifest.json').read_text())
    path = destination / 'prepared.parquet'
    # Enforce the input contract: Prepared artifact checksum mismatch.
    if file_hash(path) != manifest['prepared_sha256']: raise ValueError('Prepared artifact checksum mismatch')
    table = pd.read_parquet(path)
    table = table.loc[table.auction.eq(auction)].copy().sort_values(['decision_cutoff_utc','interval_start_utc'])
    ids = manifest['member_ids']; rating = manifest['rating_mw']; levels = np.asarray(settings['quantiles'], float)
    source_levels = np.asarray(source_config['pv_probabilistic_study']['quantiles'], float)
    quantile_columns = [f'calibrated_q{int(round(100*q)):02d}' for q in source_levels]
    # Replay chronological issue events with separate warm and cold learners for each `col` in `('interval_start_utc', 'decision_cutoff_utc', 'target_available_utc')`.
    for col in ('interval_start_utc','decision_cutoff_utc','target_available_utc'):
        table[col] = pd.to_datetime(table[col], utc=True)
    table['history_ok'] = table.eligible & ~table.retrospective_model & np.isfinite(table.measured_ac_mw) & np.isfinite(table.deterministic_prediction)
    learners = {}; pending = []; sequence = 0; issue_rows = []; update_rows = []; issue_log = []
    initializations = ('warm', 'cold')
    with threadpool_limits(limits=1):
        # Replay chronological issue events with separate warm and cold learners for each `(cutoff, day)` in `table.groupby('decision_cutoff_utc', sort=True)`.
        for cutoff, day in table.groupby('decision_cutoff_utc', sort=True):
            cutoff = pd.Timestamp(cutoff)
            # Continue until `pending and pending[0][0] <= cutoff.value` is false.
            while pending and pending[0][0] <= cutoff.value:
                _, _, key, issued_x, issued_w, observed, meta = heapq.heappop(pending)
                learner = learners[key]
                excess = learner.update(issued_x/rating, issued_w, observed/rating)
                update_rows.append({**meta, 'update_processed_at_utc': cutoff, 'observed_mw': observed,
                    'learner': '/'.join(map(str,key)), 'issued_weights': issued_w.tolist(),
                    'weights_after_update': learner.weights.tolist(), 'excess_gradient': excess.tolist()})
            # The target for every quantile training row must be available at this exact decision.
            history = table.loc[table.history_ok & (table.target_available_utc <= cutoff) &
                                (table.decision_cutoff_utc < cutoff)]
            history_days = history.delivery_day.nunique()
            current = day.reset_index(drop=True)
            q = np.full((len(current), len(levels)), np.nan)
            raw = q.copy(); crossed = bounded = night = np.zeros(len(current), dtype=bool)
            # Update `(q, raw, crossed, bounded, night)` when this condition is met.
            if history_days >= int(settings['minimum_quantile_days']):
                q, raw, crossed, bounded, night = deterministic_quantiles(history, current, levels, rating,
                    float(settings['quantile_alpha']), probabilistic_night_policy(source_config))
            local = pd.DatetimeIndex(current.interval_start_utc).tz_convert('Europe/Rome')
            blocks = local.hour.to_numpy() // int(settings['block_hours'])
            issue_log.append({'farm_id': farm, 'auction': auction, 'decision_cutoff_utc': cutoff,
                'delivery_day': current.delivery_day.iloc[0], 'history_days': int(history_days),
                'history_rows': len(history), 'quantile_crossing_rows': int(crossed.sum()),
                'quantile_out_of_bounds_rows': int(bounded.sum()), 'night_rows': int(night.sum()),
                'missing_quantile_rows': int((~np.isfinite(q).all(axis=1)).sum())})
            # Replay chronological issue events with separate warm and cold learners for each `(i, row)` in `current.iterrows()`.
            for i, row in current.iterrows():
                members = row[ids].to_numpy(dtype=float)
                member_ok = np.isfinite(members).all()
                det_ok = np.isfinite(row.deterministic_prediction)
                q_ok = np.isfinite(q[i]).all()
                # Skip the current item when `not (member_ok and det_ok)`.
                if not (member_ok and det_ok): continue
                members.sort(kind='stable')
                block = int(blocks[i]); issued_is_training = row['split'] == 'train'
                common = {'farm_id': farm, 'auction': auction, 'delivery_day': row.delivery_day,
                    'interval_start_utc': row.interval_start_utc, 'target_available_utc': row.target_available_utc,
                    'decision_cutoff_utc': cutoff, 'deterministic_cycle_utc': row.deterministic_cycle_utc,
                    'deterministic_available_utc': row.deterministic_available_utc,
                    'model_fit_available_utc': row.model_fit_available_utc, 'retrospective_model': bool(row.retrospective_model),
                    'eligible': bool(row.eligible), 'episode_excluded': bool(row.episode_excluded),
                    'observed_mw': float(row.measured_ac_mw), 'block': block,
                    'det_quantile_raw': raw[i].tolist(), 'det_quantile_corrected': q[i].tolist(),
                    'det_quantile_crossed': bool(crossed[i]), 'det_quantile_bounded': bool(bounded[i])}
                # All slots in this issue use the same frozen learner state; feedback is processed only before issues.
                for kind, values in [('ensemble', members), ('combined', np.r_[members, row.deterministic_prediction, q[i]] if q_ok else None)]:
                    # Skip the current item when `values is None`.
                    if values is None: continue
                    # Replay chronological issue events with separate warm and cold learners for each `init` in `initializations`.
                    for init in initializations:
                        # Skip the current item when `init == 'cold' and issued_is_training`.
                        if init == 'cold' and issued_is_training: continue
                        key = (init, kind, block)
                        learner = _learner(learners, key, len(values))
                        weights = learner.weights.copy()
                        # Replay chronological issue events with separate warm and cold learners when `not issued_is_training`.
                        if not issued_is_training:
                            issue_rows.append({**common, 'method': f'online_{kind}_{init}',
                                               'values': values.tolist(), 'weights': weights.tolist(), 'levels': None})
                        # Retrospective source-model predictions cannot train online weights.
                        if row.eligible and np.isfinite(row.measured_ac_mw) and not row.retrospective_model:
                            sequence += 1
                            meta = {'forecast_interval_utc': row.interval_start_utc, 'forecast_decision_utc': cutoff,
                                    'target_available_utc': row.target_available_utc, 'delivery_day': row.delivery_day}
                            heapq.heappush(pending, (row.target_available_utc.value, sequence, key, values.copy(),
                                                     weights.copy(), float(row.measured_ac_mw), meta))
                # Replay chronological issue events with separate warm and cold learners when `not issued_is_training`.
                if not issued_is_training:
                    issue_rows.append({**common, 'method': 'equal_ensemble', 'values': members.tolist(),
                                       'weights': (np.ones(50)/50).tolist(), 'levels': None})
                    # Replay chronological issue events with separate warm and cold learners when `q_ok`.
                    if q_ok:
                        issue_rows.append({**common, 'method': 'deterministic_quantiles', 'values': q[i].tolist(),
                                           'weights': None, 'levels': levels.tolist()})
                    # When `all((c in current.columns for c in quantile_columns))`, set `fixed` using `row[quantile_columns].to_numpy(dtype=float)`.
                    if all(c in current.columns for c in quantile_columns):
                        fixed = row[quantile_columns].to_numpy(dtype=float)
                        # Replay chronological issue events with separate warm and cold learners when `np.isfinite(fixed).all()`.
                        if np.isfinite(fixed).all():
                            issue_rows.append({**common, 'method': 'fixed_ensemble_quantiles', 'values': fixed.tolist(),
                                               'weights': None, 'levels': source_levels.tolist()})
            print(f'Farm {farm} {auction}: issued {current.delivery_day.iloc[0]}, history {history_days} days', flush=True)
    # Enforce the input contract: No validation predictions for farm `farm` `auction`.
    if not issue_rows: raise ValueError(f'No validation predictions for farm {farm} {auction}')
    prefix = destination / auction
    prefix.mkdir(parents=True, exist_ok=True)
    paths = {}
    # Write the corresponding artifacts for `(name, rows)` from `[('forecasts', issue_rows), ('updates', update_rows), ('issues', issue_log)]`.
    for name, rows in [('forecasts', issue_rows), ('updates', update_rows), ('issues', issue_log)]:
        out = prefix / f'{name}.parquet'
        pd.DataFrame(rows).to_parquet(out, index=False)
        paths[name] = file_hash(out)
    write_json(prefix / 'manifest.json', {'farm_id': farm, 'auction': auction, 'prepared_sha256': manifest['prepared_sha256'],
        'files': paths, 'pending_after_final_issue': len(pending), 'validation_selection_exploratory': True,
        'initial_source_weights': {'ensemble_only': {'ensemble': 1.0},
            'combined': {'ensemble': 50/70, 'deterministic': 1/70, 'quantiles': 19/70}},
        'delay_guarantee': 'Original ML-Poly regret bound not asserted for delayed block feedback.'})
    return {'forecasts': len(issue_rows), 'updates': len(update_rows), 'pending': len(pending)}
