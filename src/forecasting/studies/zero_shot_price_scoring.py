"""Score zero-shot price forecasts on explicitly matched delivery products."""

from __future__ import annotations

import numpy as np
import pandas as pd


KEYS = ["delivery_day", "product_start", "auction", "gate"]
QUANTILES = tuple(range(10, 100, 10))


def matched_predictions(frame, variants):
    """Retain products observed for every variant, rejecting mismatched truths."""
    selected = frame.loc[frame.variant.isin(variants)].copy()
    if selected.duplicated(KEYS + ["variant"]).any():
        raise ValueError("Duplicate prediction for a product and variant")
    selected = selected.loc[np.isfinite(selected.actual) & np.isfinite(selected.prediction)]
    truth = selected.groupby(KEYS, dropna=False).actual.agg(["min", "max"])
    if not np.allclose(truth["min"], truth["max"], equal_nan=True):
        raise ValueError("Variants have inconsistent actual prices")
    counts = selected.groupby(KEYS, dropna=False).variant.nunique()
    common = counts[counts == len(set(variants))].reset_index()[KEYS]
    return selected.merge(common, on=KEYS, how="inner", validate="many_to_one")


def score(frame):
    """Compute point, reference-relative and marginal quantile scores."""
    actual = frame.actual.to_numpy(float)
    prediction = frame.prediction.to_numpy(float)
    valid = np.isfinite(actual) & np.isfinite(prediction)
    actual, prediction = actual[valid], prediction[valid]
    if not len(actual):
        return {"products": 0}
    error = prediction - actual
    total = np.abs(actual).sum()
    result = {"products": len(actual), "mae": float(np.abs(error).mean()),
              "wape": float(np.abs(error).sum() / total) if total else None,
              "rmse": float(np.sqrt(np.mean(error ** 2))), "bias": float(error.mean())}
    for column, name in (("weekly_reference", "weekly"), ("latest_reference", "latest")):
        if column not in frame:
            continue
        reference = frame.loc[valid, column].to_numpy(float)
        present = np.isfinite(reference)
        denominator = np.abs(actual[present] - reference[present]).sum()
        result[f"{name}_reference_products"] = int(present.sum())
        result[f"{name}_relative_mae"] = (float(np.abs(error[present]).sum() / denominator)
                                             if denominator else None)
        if name == "latest":
            true_spread = actual[present] - reference[present]
            forecast_spread = prediction[present] - reference[present]
            result["spread_direction_accuracy"] = (float(np.mean(np.sign(true_spread) == np.sign(forecast_spread)))
                                                   if present.any() else None)
    for level in QUANTILES:
        column = f"q{level:02d}"
        if column not in frame:
            continue
        forecast = frame.loc[valid, column].to_numpy(float)
        present = np.isfinite(forecast)
        if present.any():
            residual = actual[present] - forecast[present]
            quantile = level / 100
            result[f"{column}_pinball"] = float(np.maximum(quantile * residual, (quantile - 1) * residual).mean())
            result[f"{column}_coverage"] = float(np.mean(actual[present] <= forecast[present]))
            result[f"{column}_products"] = int(present.sum())
    return result
