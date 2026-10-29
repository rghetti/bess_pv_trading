"""Weighted empirical CRPS and delayed ML-Poly updates for PV forecasts."""
from __future__ import annotations

import numpy as np


def weighted_crps(values, weights, target):
    """Score the issued discrete distribution, including all ordered pairs."""
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    if x.ndim != 1 or w.shape != x.shape or not np.isfinite(x).all() or not np.isfinite(target):
        raise ValueError('CRPS inputs must be finite aligned vectors and a finite target')
    if not np.isfinite(w).all() or (w < 0).any() or not np.isclose(w.sum(), 1):
        raise ValueError('CRPS weights must be a probability vector')
    return float(w @ np.abs(x-target) - .5 * w @ np.abs(x[:, None]-x[None, :]) @ w)


def weighted_quantile(values, weights, levels):
    """Invert the weighted empirical CDF without interpolating fictitious values."""
    order = np.argsort(values, kind='stable')
    x, w = np.asarray(values)[order], np.asarray(weights)[order]
    cumulative = np.cumsum(w)
    return x[np.minimum(np.searchsorted(cumulative, levels, side='left'), len(x)-1)]


class MLPoly:
    """Maintain cumulative linearized regrets and adaptive ML-Poly weights."""
    def __init__(self, size):
        if size < 1: raise ValueError('At least one expert is required')
        self.regret = np.zeros(size)
        self.squared_regret = np.zeros(size)
        self.weights = np.full(size, 1.0 / size)

    def update(self, issued_values, issued_weights, target):
        """Use the issued snapshot when feedback arrives, including delayed feedback."""
        x = np.asarray(issued_values, dtype=float)
        w = np.asarray(issued_weights, dtype=float)
        if len(x) != len(self.weights): raise ValueError('Expert set changed')
        weighted_crps(x, w, target)  # Validate before changing learner state.
        gradient = np.abs(x-target) - np.abs(x[:, None]-x[None, :]) @ w
        excess = w @ gradient - gradient
        self.regret += excess
        self.squared_regret += excess ** 2
        learning_rate = 1.0 / (1.0 + self.squared_regret)
        positive = learning_rate * np.maximum(self.regret, 0.0)
        self.weights = positive / positive.sum() if positive.sum() > 0 else np.full(len(x), 1.0/len(x))
        return excess
