"""Ex-post imbalance accounting from executed powers and committed positions."""
from __future__ import annotations

import numpy as np
import pandas as pd


def expected_realized_kpis(*, initial_expected_profit_eur: float,
                           initial_expected_auction_revenue_eur: float,
                           realized_profit_after_imbalance_eur: float | None,
                           auction_revenue_eur: float) -> dict:
    """Compare the initial auction solve with final cash accounting.

    Differences are realized minus expected. Percentages use the absolute
    expectation so improvements keep a positive sign even for expected losses.
    A zero expectation has no defined percentage; unpriced profit stays unknown.
    Gross auction revenue excludes imbalance settlement and operating costs.
    """
    comparisons = {
        "profit": (initial_expected_profit_eur, realized_profit_after_imbalance_eur),
        "auction_revenue": (initial_expected_auction_revenue_eur, auction_revenue_eur),
    }
    kpis = {"initial_expected_profit_eur": float(initial_expected_profit_eur),
        "initial_expected_auction_revenue_eur": float(initial_expected_auction_revenue_eur),
        "expectation_baseline": "initial_auction_solve"}
    for name, (expected, realized) in comparisons.items():
        difference = None if realized is None else float(realized - expected)
        kpis[f"realized_minus_expected_{name}_eur"] = difference
        kpis[f"realized_minus_expected_{name}_percent"] = (
            None if difference is None or expected == 0 else 100 * difference / abs(expected))
    return kpis


def load_realized_imbalance_prices(config, grid) -> tuple[pd.DataFrame | None, dict]:
    """Align an exact actual-price curve from pinned DuckDB snapshots, ex post only."""
    settings = config.get("settlement", {})
    if not settings.get("enabled", False):
        return None, {"status": "disabled"}
    if settings["missing_price_policy"] != "error":
        raise ValueError("Actual imbalance prices require missing_price_policy=error; no filling is allowed")
    from ..data.api_catalog import load_api_cache_records
    from ..data import config_path
    frame, sources = load_api_cache_records(config_path(config, settings["database_path"]),
        provider=settings["provider"], dataset=settings["dataset"], zone=settings["zone"],
        start=grid.starts[0], end=grid.ends[-1], verify_checksums=True)
    expected = pd.DatetimeIndex(grid.starts).tz_convert("UTC")
    if frame.empty or frame.product_start.duplicated().any():
        raise ValueError("Actual imbalance settlement requires a nonempty unique price curve")
    frame = frame.set_index("product_start").sort_index()
    if not pd.DatetimeIndex(frame.index).equals(expected):
        raise ValueError("Actual imbalance price starts do not match the executed quarter-hour grid")
    if not pd.DatetimeIndex(frame.product_end).equals(pd.DatetimeIndex(grid.ends).tz_convert("UTC")):
        raise ValueError("Actual imbalance price ends do not match the executed quarter-hour grid")
    minutes = settings["native_interval_minutes"]
    durations = (pd.DatetimeIndex(frame.product_end) - expected).total_seconds() / 60
    if minutes != 15 or np.any(durations != minutes) or not frame.unit.eq("EUR/MWh").all():
        raise ValueError("Actual imbalance settlement requires native 15-minute EUR/MWh prices")
    if not frame.zone.eq(settings["zone"]).all():
        raise ValueError("Actual imbalance prices contain an unexpected macrozone")
    values = pd.to_numeric(frame[settings["price_column"]], errors="raise").to_numpy(float)
    if not np.all(np.isfinite(values)):
        raise ValueError("Actual imbalance settlement prices must all be finite")
    frame = frame.reset_index()
    frame["settlement_price_eur_mwh"] = values
    return frame, {"status": "loaded_for_ex_post_settlement_only", "provider": settings["provider"],
        "dataset": settings["dataset"], "zone": settings["zone"], "records": len(frame),
        "native_interval_minutes": minutes, "price_column": settings["price_column"],
        "snapshots": sources, "supplied_to_optimizer": False}


def settle_realized_dispatch(dispatch: pd.DataFrame, *, auction_revenue_eur: float,
                             imbalance_prices=None, imbalance_negative_prices=None
                             ) -> tuple[pd.DataFrame, dict]:
    """Apply aligned settlement prices only after dispatch; missing prices stay unknown.

    Positive imbalance means excess net export. Prices are EUR/MWh, including
    negative prices. With a single price curve both directions use that curve.
    Price ingestion/alignment belongs to the dataset adapter, not the optimizer.
    """
    frame = dispatch.copy()
    starts = pd.to_datetime(frame.start, utc=True)
    ends = pd.to_datetime(frame.end, utc=True)
    hours = (ends - starts).dt.total_seconds().to_numpy() / 3600
    if frame.interval_id.duplicated().any() or np.any(hours <= 0):
        raise ValueError("Settlement requires unique intervals with positive duration")
    imbalance = frame.grid_mw.to_numpy(dtype=float) * hours - frame.net_position_mwh.to_numpy(dtype=float)
    if not np.all(np.isfinite(imbalance)):
        raise ValueError("Settlement requires finite executed powers and positions")
    frame["imbalance_mwh"] = imbalance
    # Financial wear is allocated only from execution; model deterrence penalties
    # are not actual settlement charges and do not enter this cash correction.
    before = float(auction_revenue_eur - frame.degradation_eur.sum() - frame.import_fee_eur.sum())
    kpis = {"profit_before_imbalance_settlement_eur": before,
        "net_imbalance_mwh": float(imbalance.sum()),
        "absolute_imbalance_mwh": float(np.abs(imbalance).sum()),
        "surplus_imbalance_mwh": float(np.maximum(imbalance, 0).sum()),
        "shortfall_imbalance_mwh": float(np.maximum(-imbalance, 0).sum()),
        "imbalance_settlement_status": "unavailable",
        "realized_imbalance_cashflow_eur": None,
        "realized_profit_after_imbalance_eur": None}
    if imbalance_prices is None:
        if imbalance_negative_prices is not None:
            raise ValueError("Negative-direction prices require a positive-direction price curve")
        frame["realized_imbalance_cashflow_eur"] = np.nan
        return frame, kpis
    positive = np.asarray(imbalance_prices, dtype=float)
    negative = positive if imbalance_negative_prices is None else np.asarray(imbalance_negative_prices, dtype=float)
    if any(values.shape != imbalance.shape or not np.all(np.isfinite(values)) for values in (positive, negative)):
        raise ValueError("Settlement prices must be finite and aligned to every executed interval")
    cashflow = np.maximum(imbalance, 0) * positive + np.minimum(imbalance, 0) * negative
    frame["imbalance_price_eur_mwh"] = positive
    frame["imbalance_negative_price_eur_mwh"] = negative
    frame["realized_imbalance_cashflow_eur"] = cashflow
    kpis.update(imbalance_settlement_status="priced",
        realized_imbalance_cashflow_eur=float(cashflow.sum()),
        realized_imbalance_receipts_eur=float(np.maximum(cashflow, 0).sum()),
        realized_imbalance_payments_eur=float(np.maximum(-cashflow, 0).sum()),
        realized_profit_after_imbalance_eur=before + float(cashflow.sum()))
    return frame, kpis
