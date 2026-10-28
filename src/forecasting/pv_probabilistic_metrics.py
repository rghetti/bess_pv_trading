"""Quantile calibration and paired deterministic/ensemble scoring for PV."""
from __future__ import annotations
import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import QuantileRegressor
from sklearn.preprocessing import StandardScaler


def paired_inputs(ensemble, deterministic, members_prediction, deterministic_prediction, member_ids):
    """Assemble a single target per instant and require comparable forecast inputs."""
    ens = ensemble[['interval_start_utc', 'member']].copy()
    ens['prediction'] = np.asarray(members_prediction, float)
    wide = ens.pivot(index='interval_start_utc', columns='member', values='prediction').reindex(columns=member_ids)
    metadata = ['interval_start_utc', 'delivery_day', 'split', 'measured_ac_mw', 'eligible', 'episode_excluded']
    metadata += [name for name in ('forecast_cycle_utc', 'forecast_available_at_utc') if name in deterministic]
    base = deterministic[metadata].copy()
    base['deterministic_prediction'] = np.asarray(deterministic_prediction, float)
    base = base.set_index('interval_start_utc').sort_index()
    if base.index.has_duplicates or wide.index.has_duplicates:
        raise ValueError('Duplicate forecast or target keys')
    base = base.join(wide, how='left').reset_index()
    return base


def summary_features(member_values, times, rating):
    """Describe the converted member distribution and known local time."""
    values = np.asarray(member_values, dtype=float) / rating
    qs = np.quantile(values, [0.1, 0.5, 0.9], axis=1).T
    local = pd.DatetimeIndex(times).tz_convert('Europe/Rome') + pd.Timedelta(minutes=7.5)
    hour = local.hour.to_numpy() + local.minute.to_numpy() / 60
    return np.column_stack([values.mean(axis=1), qs, np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)])


def fit_calibrator(member_values, targets, times, rating, quantiles, alpha):
    """Fit independent bounded-scale quantile regressions only on calibration rows."""
    features = summary_features(member_values, times, rating)
    target = np.asarray(targets, dtype=float) / rating
    valid = np.isfinite(features).all(axis=1) & np.isfinite(target)
    if valid.sum() < 30:
        raise ValueError('Insufficient finite out-of-fold calibration intervals')
    scaler = StandardScaler().fit(features[valid])
    x = scaler.transform(features[valid])
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Unrecognized options detected.*')
        models = [QuantileRegressor(quantile=float(q), alpha=float(alpha), solver='highs',
                    solver_options={'threads': 1}).fit(x, target[valid]) for q in quantiles]
    return {'scaler': scaler, 'models': models, 'quantiles': list(map(float, quantiles)),
            'rating_mw': float(rating), 'calibration_intervals': int(valid.sum()),
            'features': ['pv_mean','pv_p10','pv_p50','pv_p90','hour_sin','hour_cos']}


def predict_calibrator(calibrator, member_values, times, *, return_raw=False):
    """Rearrange crossing quantiles and bound AC power; record all adjustments."""
    features = summary_features(member_values, times, calibrator['rating_mw'])
    x = calibrator['scaler'].transform(features)
    raw = np.column_stack([m.predict(x) for m in calibrator['models']]) * calibrator['rating_mw']
    crossing = np.any(np.diff(raw, axis=1) < 0, axis=1)
    bounded = np.clip(raw, 0, calibrator['rating_mw'])
    corrected = np.sort(bounded, axis=1)
    adjustments = {'crossing_rows': int(crossing.sum()), 'bounded_values': int(np.sum(raw != bounded)),
                   'total_rows': len(raw)}
    return (corrected, adjustments, raw) if return_raw else (corrected, adjustments)


def quantile_mean(quantiles, levels):
    """Integrate the linearly interpolated quantile curve with constant tails."""
    grid = np.r_[0., np.asarray(levels, float), 1.]
    values = np.column_stack([quantiles[:, 0], quantiles, quantiles[:, -1]])
    return np.trapezoid(values, x=grid, axis=1)


def raw_crps(members, target):
    """Compute empirical-ensemble CRPS from sorted member values."""
    members = np.asarray(members, float)
    n = members.shape[1]
    ordered = np.sort(members, axis=1)
    coefficients = 2*np.arange(1, n+1)-n-1
    return np.mean(np.abs(members - np.asarray(target)[:, None]), axis=1) - (ordered @ coefficients)/(n*n)


def quantile_crps(quantiles, levels, target):
    """Approximate CRPS by integrating pinball loss over the quantile curve."""
    grid = np.linspace(0, 1, 201)
    anchors = np.r_[0., np.asarray(levels, float), 1.]
    rows = np.column_stack([quantiles[:, 0], quantiles, quantiles[:, -1]])
    curve = np.stack([np.interp(grid, anchors, row) for row in rows])
    error = np.asarray(target)[:, None] - curve
    loss = np.maximum(grid * error, (grid-1)*error)
    return 2*np.trapezoid(loss, x=grid, axis=1)


def score_frame(frame, member_ids, levels, rating, label):
    """Return paired point and probabilistic scores for one analysis window."""
    m = frame[member_ids].to_numpy(float)
    y = frame.measured_ac_mw.to_numpy(float)
    valid = frame.eligible.to_numpy(bool) & np.isfinite(y) & np.isfinite(m).all(axis=1) & np.isfinite(frame.deterministic_prediction)
    z = frame.loc[valid].copy(); m = m[valid]; y = y[valid]
    rows = []
    if not len(z):
        return pd.DataFrame([{'window': label, 'method': 'no_comparable_observations', 'n': 0}])
    points = {'deterministic': z.deterministic_prediction.to_numpy(float),
              'member_mean': m.mean(axis=1), 'member_median': np.median(m, axis=1)}
    qcolumns = [f'calibrated_q{int(round(100*q)):02d}' for q in levels]
    has_cal = all(col in z for col in qcolumns)
    if has_cal:
        q = z[qcolumns].to_numpy(float)
        good = np.isfinite(q).all(axis=1)
        points['calibrated_mean'] = np.where(good, quantile_mean(q, levels), np.nan)
        points['calibrated_median'] = q[:, list(levels).index(0.5)]
    for name, pred in points.items():
        mask = np.isfinite(pred)
        err = pred[mask]-y[mask]
        rows.append({'window': label, 'method': name, 'n': int(mask.sum()),
            'mae_mw': float(np.mean(abs(err))) if len(err) else None,
            'rmse_mw': float(np.sqrt(np.mean(err**2))) if len(err) else None,
            'bias_mw': float(np.mean(err)) if len(err) else None,
            'nmae_percent_capacity': float(100*np.mean(abs(err))/rating) if len(err) else None,
            'nrmse_percent_capacity': float(100*np.sqrt(np.mean(err**2))/rating) if len(err) else None,
            'bias_percent_capacity': float(100*np.mean(err)/rating) if len(err) else None})
    empirical_crps = float(np.mean(raw_crps(m, y)))
    rows.append({'window': label, 'method': 'raw_ensemble_distribution', 'n': len(z),
                 'crps_mw': empirical_crps, 'crps_percent_capacity': 100*empirical_crps/rating})
    for width in (0.5, 0.8, 0.9):
        lo, hi = np.quantile(m, [(1-width)/2, 1-(1-width)/2], axis=1)
        rows[-1][f'coverage_{int(width*100)}'] = float(np.mean((y >= lo) & (y <= hi)))
        rows[-1][f'width_{int(width*100)}_mw'] = float(np.mean(hi-lo))
        rows[-1][f'width_{int(width*100)}_percent_capacity'] = float(100*np.mean(hi-lo)/rating)
    if has_cal and good.any():
        cq, cy = q[good], y[good]
        result = {'window': label, 'method': 'calibrated_quantiles', 'n': int(good.sum()),
                  'crps_mw': float(np.mean(quantile_crps(cq, levels, cy))), 'crps_method': 'quantile_curve_quadrature_constant_tails'}
        result['crps_percent_capacity'] = 100*result['crps_mw']/rating
        for j, tau in enumerate(levels):
            residual = cy-cq[:, j]
            result[f'pinball_q{int(round(100*tau)):02d}_mw'] = float(np.mean(np.maximum(tau*residual, (tau-1)*residual)))
            result[f'pinball_q{int(round(100*tau)):02d}_percent_capacity'] = 100*result[f'pinball_q{int(round(100*tau)):02d}_mw']/rating
        for width in (0.5, 0.8, 0.9):
            lo = list(levels).index(round((1-width)/2, 2)); hi = list(levels).index(round(1-(1-width)/2, 2))
            result[f'coverage_{int(width*100)}'] = float(np.mean((cy >= cq[:,lo]) & (cy <= cq[:,hi])))
            result[f'width_{int(width*100)}_mw'] = float(np.mean(cq[:,hi]-cq[:,lo]))
            result[f'width_{int(width*100)}_percent_capacity'] = float(100*np.mean(cq[:,hi]-cq[:,lo])/rating)
        rows.append(result)
    return pd.DataFrame(rows)
