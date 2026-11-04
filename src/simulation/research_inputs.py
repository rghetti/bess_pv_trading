"""Information-set ablations shared by physical control and auction planning."""

from __future__ import annotations

import numpy as np
import pandas as pd


def validate_information_settings(config):
    """Keep forecast refreshes independent of the configured auction access."""
    settings = config.get("research_information", {})
    if not isinstance(settings, dict) or set(settings) - {"pv_updates", "price_updates"}:
        raise ValueError("Unknown research_information settings")
    for name in ("pv_updates", "price_updates"):
        if settings.get(name, "latest") not in {"latest", "frozen_initial"}:
            raise ValueError(f"research_information.{name} must be latest or frozen_initial")


def initial_information_cutoff(daily):
    """Use the first scheduled gate even when a study removes that auction."""
    settings = daily.get("research_initial_cutoff")
    if settings is not None:
        return pd.Timestamp(settings)
    return min(pd.Timestamp(event.submission_at) for event in daily["schedule"])


def price_information_cutoff(config, daily, cutoff):
    """Freeze predictions only; actual publication events still book cash once."""
    if config.get("research_information", {}).get("price_updates", "latest") == "frozen_initial":
        initial = initial_information_cutoff(daily)
        if initial > pd.Timestamp(cutoff):
            raise ValueError("Initial price information is after the current decision")
        return initial
    return pd.Timestamp(cutoff)


def frozen_initial_pv(replay, cutoff, starts):
    """Join already pinned initial forecasts for all requested delivery dates.

    A projection can cross midnight. Each interval keeps its own delivery book's
    initial vintage, and all such vintages must already be available at cutoff.
    No measurement or retrospectively selected replacement is evaluated here.
    """
    starts = pd.DatetimeIndex(starts)
    positions = replay.starts.get_indexer(starts)
    if np.any(positions < 0):
        raise ValueError("Frozen PV request exceeds the prepared delivery grid")
    values = np.empty(len(starts))
    vintages, availability = [], []
    by_day, model_evidence = {}, {}
    for output_index, global_index in enumerate(positions):
        product = replay.grid.interval_ids[global_index]
        day = replay.day_for_product[product]
        by_day.setdefault(day, []).append((output_index, global_index - replay.day_offsets[day]))
    # Vectorize each date's slice while retaining the actual vintage evidence.
    for day, indices in by_day.items():
        daily = replay.daily[day]
        metadata = daily["pv_metadata"]
        vintage = pd.Timestamp(metadata["pv_forecast_vintage_utc"])
        available = pd.Timestamp(metadata["pv_forecast_available_at_utc"])
        if vintage.tzinfo is None or available.tzinfo is None or max(vintage, available) > cutoff:
            raise ValueError("Frozen initial PV is unavailable at the decision cutoff")
        output, local = np.asarray(indices).T
        values[output] = daily["initial_forecast"][local]
        vintages.append(vintage)
        availability.append(available)
        model_evidence[day] = daily.get("pv_provenance", {})
    return values, {"pv_forecast_policy": "frozen_initial",
        "pv_forecast_vintage_utc": max(vintages).isoformat(),
        "pv_forecast_available_at_utc": max(availability).isoformat(),
        "frozen_forecast_delivery_days": sorted(by_day),
        "frozen_initial_pv_provenance": model_evidence}


def auction_price_predictions(replay, day, cutoff):
    """Replace only unpublished curves with causal point forecasts.

    Forecast profits remain planning quantities; the event runner keeps its
    pinned realized curves for publication and settlement accounting.
    """
    daily = replay.daily[day]
    provider = replay.prepared.get("energy_price_provider")
    if provider is None or not replay.config.get("energy_price_forecast", {}).get("auction_enabled", False):
        return daily["prices"], {}
    selected_cutoff = price_information_cutoff(replay.config, daily, cutoff)
    starts = pd.DatetimeIndex(daily["grid"].starts)
    curves, evidence = {}, {}
    participating = {event.auction for event in daily["schedule"]}
    configured = replay.config.get("gme_oracle_experiment", {}).get("participating_auctions")
    if configured is not None:
        participating &= set(configured)
    # Known prices are authoritative even in the frozen-forecast ablation.
    for auction, actual in daily["prices"].items():
        if auction not in participating:
            continue
        if auction in replay.known[day]:
            curves[auction] = actual
            evidence[auction] = {"source": "published_actual_clearing_price"}
        else:
            # Expired products are excluded by the planning grid; do not ask the
            # forecaster to manufacture a past prediction at a later cutoff.
            eligible = starts >= selected_cutoff
            if auction == "IDA3":
                eligible &= starts.tz_convert(replay.prepared["calendar"].timezone).hour >= 12
            forecast = provider.predict(auction, selected_cutoff, starts[eligible], scenario_count=1)
            curves[auction] = np.zeros(len(starts))
            curves[auction][eligible] = np.asarray(forecast.point_eur_per_mwh, dtype=float)
            evidence[auction] = dict(forecast.provenance)
    return curves, {"auction_price_information": "causal_point_forecast",
        "price_forecast_cutoff_utc": selected_cutoff.isoformat(),
        "auction_price_forecasts": evidence}
