"""Retrospective conditional intraday price features and linear spread models.

Source prices retain their native products and unverified publication metadata.
Study auction clocks live in separate columns; they never become verified source
timestamps. All fits and artifacts from this module are operationally ineligible.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
from typing import Any, Mapping, Sequence
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .prices import FEATURES, PRECEDING, RidgePriceModel, fit_ridge


AUCTIONS = ("DA", "IDA1", "IDA2", "IDA3")
POLICY = "retrospective_latest_revision_with_assumed_auction_results"
DEFAULT_GATES = {"IDA1": "15:00", "IDA2": "22:00", "IDA3": "10:00"}
DEFAULT_RESULTS = {"DA": "13:30", "IDA1": "15:30", "IDA2": "22:30", "IDA3": "10:30"}


def _aware(values: Sequence[Any], label: str, *, missing: bool = False) -> pd.DatetimeIndex:
    """Normalize explicit aware clocks without localizing naive source labels."""
    stamps = [pd.Timestamp(value) for value in values]
    if any((pd.isna(value) and not missing) or (not pd.isna(value) and value.tzinfo is None) for value in stamps):
        raise ValueError(f"{label} must contain explicit timezone-aware timestamps")
    return pd.DatetimeIndex(pd.to_datetime(stamps, utc=True)).as_unit("ns")


def _settings(settings: Mapping[str, Any] | None) -> dict:
    """Keep configurable study clocks distinct from the verified market calendar."""
    result = dict(settings or {})
    result.setdefault("timezone", "Europe/Rome")
    ZoneInfo(result["timezone"])
    buffer = result.setdefault("decision_buffer_minutes", 5.0)
    if isinstance(buffer, bool) or not np.isfinite(buffer) or not 0 <= buffer < 60:
        raise ValueError("decision_buffer_minutes must lie in [0,60)")
    result["gates"] = {**DEFAULT_GATES, **result.get("gates", {})}
    result["result_times"] = {**DEFAULT_RESULTS, **result.get("result_times", {})}
    for value in [*result["gates"].values(), *result["result_times"].values()]:
        time.fromisoformat(value)
    if result.get("availability_policy", POLICY) != POLICY:
        raise ValueError("Intraday features require the explicit retrospective assumed-clock policy")
    result["availability_policy"] = POLICY
    return result


def _clock(day: str, auction: str, settings: Mapping[str, Any], *, results: bool) -> pd.Timestamp:
    """Construct a civil clock, preserving noon/gates across daylight saving."""
    if results:
        overrides = settings.get("result_overrides", {}).get(auction, {})
        if day in overrides:
            value = overrides[day]
            return pd.NaT if value is None else _aware([value], "result_overrides")[0]
    event_day = date.fromisoformat(day) - timedelta(days=int(auction != "IDA3"))
    clock = settings["result_times" if results else "gates"][auction]
    result = pd.Timestamp(datetime.combine(event_day, time.fromisoformat(clock)), tz=settings["timezone"])
    if not results:
        result -= pd.Timedelta(minutes=float(settings["decision_buffer_minutes"]))
    return result.tz_convert("UTC")


def _products(frame: pd.DataFrame, timezone_name: str) -> pd.DataFrame:
    """Validate actual native interval boundaries without interpolating products."""
    if not {"product_start", "product_end"}.issubset(frame):
        raise ValueError("Native price products require product_start and product_end")
    result = frame.copy().reset_index(drop=True)
    for name in ("product_start", "product_end"):
        result[name] = _aware(result[name], name)
    durations = (result.product_end - result.product_start).dt.total_seconds() / 60
    if not durations.isin([15, 30, 60]).all():
        raise ValueError("Native price products must last 15, 30 or 60 minutes")
    if "native_interval_minutes" in result and not np.equal(result.native_interval_minutes.to_numpy(float), durations).all():
        raise ValueError("Declared native duration differs from the source price product")
    result["native_interval_minutes"] = durations.astype(int)
    result["delivery_day"] = result.product_start.dt.tz_convert(timezone_name).dt.date.astype(str)
    return result


def _records(frame: pd.DataFrame, auction: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Select latest retrieval revisions explicitly, preserving unknown prices."""
    if frame.empty:
        return pd.DataFrame()
    result = _products(frame, settings["timezone"])
    if "auction" in result and not result.auction.eq(auction).all():
        raise ValueError("Intraday source auction disagrees with its mapping key")
    if "price_eur_mwh" not in result:
        raise ValueError("Intraday source records need actual native price_eur_mwh labels")
    result["price_eur_mwh"] = pd.to_numeric(result.price_eur_mwh, errors="raise").astype(float)
    if np.isinf(result.price_eur_mwh).any():
        raise ValueError("Intraday prices cannot be infinite; missing observations remain missing")
    if "unit" in result and not result.unit.eq("EUR/MWh").all():
        raise ValueError("Intraday price inputs must use EUR/MWh")
    if "zone" in result and result.zone.nunique(dropna=False) != 1:
        raise ValueError("Select one explicit market zone before constructing intraday features")
    keys = ["product_start", "product_end"]
    if result.duplicated(keys).any():
        if "retrieved_at" not in result:
            raise ValueError("Duplicate native prices require explicit retrieval revisions")
        result["retrieved_at"] = _aware(result.retrieved_at, "retrieved_at")
        conflicts = result.groupby([*keys, "retrieved_at"]).price_eur_mwh.nunique(dropna=False)
        if conflicts.gt(1).any():
            raise ValueError("Conflicting native prices at an identical retrieval revision")
        result = result.sort_values([*keys, "retrieved_at"]).drop_duplicates(keys, keep="last")
    result = result.sort_values("product_start").reset_index(drop=True)
    if (result.product_start.iloc[1:].to_numpy() < result.product_end.iloc[:-1].to_numpy()).any():
        raise ValueError("Native source price products overlap")
    assumed = {day: _clock(day, auction, settings, results=True) for day in result.delivery_day.unique()}
    result["_assumed_available_at"] = pd.to_datetime(result.delivery_day.map(assumed), utc=True)
    result["_study_available_at"] = result._assumed_available_at
    # A real later publication, if supplied, cannot be overruled by the study
    # clock. Its original source column is kept unchanged in the output.
    if "published_at" in result:
        recorded = _aware(result.published_at, "published_at", missing=True)
        known = ~recorded.isna()
        result.loc[known, "_study_available_at"] = recorded[known]
    result["auction"] = auction
    return result


def _ancestors(auction: str, settings: Mapping[str, Any]) -> tuple[str, ...]:
    """Permit only earlier auctions, always including the immediate predecessor."""
    if auction not in PRECEDING:
        raise ValueError("Intraday targets must be IDA1, IDA2 or IDA3")
    earlier = AUCTIONS[:AUCTIONS.index(auction)]
    return earlier if settings.get("include_earlier_auctions", True) else (PRECEDING[auction],)


def _align(source: pd.DataFrame, products: pd.DataFrame, auction: str,
           settings: Mapping[str, Any], *, enforce_clock: bool = True) -> pd.DataFrame:
    """Map native prices, or containing hourly DA rates as derived features only."""
    count = len(products)
    values = np.full(count, np.nan)
    starts = pd.Series(pd.NaT, index=range(count), dtype="datetime64[ns, UTC]")
    ends, available = starts.copy(), starts.copy()
    minutes = np.full(count, np.nan)
    policy = np.full(count, "missing_or_unavailable", dtype=object)
    if not source.empty:
        positions = np.searchsorted(source.product_start.array.asi8, products.product_start.array.asi8, side="right") - 1
        candidate = np.maximum(positions, 0)
        chosen = source.iloc[candidate].reset_index(drop=True)
        contained = ((positions >= 0) & (chosen.product_start <= products.product_start).to_numpy()
                     & (chosen.product_end >= products.product_end).to_numpy())
        exact = (chosen.product_start.eq(products.product_start) & chosen.product_end.eq(products.product_end)).to_numpy()
        transition = str(settings.get("da_quarter_hour_start", "2025-10-01"))
        hourly = (contained & chosen.native_interval_minutes.eq(60).to_numpy()
                  & products.native_interval_minutes.eq(15).to_numpy()
                  & products.delivery_day.lt(transition).to_numpy() & (auction == "DA")
                  & bool(settings.get("allow_hourly_da_features", True)))
        mapped = contained & (exact | hourly)
        clock_ok = chosen._study_available_at.notna().to_numpy()
        if enforce_clock:
            clock_ok = clock_ok & (chosen._study_available_at <= products.forecast_cutoff).to_numpy()
        valid = mapped & clock_ok
        values[valid] = chosen.loc[valid, "price_eur_mwh"]
        starts.loc[mapped], ends.loc[mapped] = chosen.loc[mapped, "product_start"], chosen.loc[mapped, "product_end"]
        available.loc[mapped] = chosen.loc[mapped, "_study_available_at"]
        minutes[mapped] = chosen.loc[mapped, "native_interval_minutes"]
        policy[valid & exact] = "exact_native_product"
        policy[valid & hourly] = "derived_containing_hourly_DA_price_feature_only"
    return pd.DataFrame({"price": values, "source_product_start": starts, "source_product_end": ends,
        "source_native_interval_minutes": minutes, "assumed_or_recorded_available_at": available,
        "feature_mapping": policy})


def _base(products: pd.DataFrame, history: pd.DataFrame, auction: str,
          settings: Mapping[str, Any]) -> pd.DataFrame:
    """Build calendar and own-auction past-price features on exact civil products."""
    frame = products.copy().reset_index(drop=True)
    timezone_name = settings["timezone"]
    clocks = [stamp.to_pydatetime().astimezone(ZoneInfo(timezone_name)) for stamp in frame.product_start]
    keys = [(stamp.date(), stamp.hour, stamp.minute, stamp.fold, int(minutes))
            for stamp, minutes in zip(clocks, frame.native_interval_minutes)]
    hours = np.array([stamp.hour + stamp.minute / 60 for stamp in clocks])
    for label, values, period in (("hour", hours, 24), ("weekday", np.array([stamp.weekday() for stamp in clocks]), 7),
                                  ("month", np.array([stamp.month for stamp in clocks]), 12)):
        frame[f"{label}_sin"], frame[f"{label}_cos"] = np.sin(2 * np.pi * values / period), np.cos(2 * np.pi * values / period)
    forecast_times = {day: _clock(day, auction, settings, results=False) for day in frame.delivery_day.unique()}
    frame["forecast_cutoff"] = pd.to_datetime(frame.delivery_day.map(forecast_times), utc=True)
    lookup = {}
    if not history.empty:
        history_clocks = [stamp.to_pydatetime().astimezone(ZoneInfo(timezone_name)) for stamp in history.product_start]
        lookup = {(stamp.date(), stamp.hour, stamp.minute, stamp.fold, int(minutes)): (float(value), available)
                  for stamp, minutes, value, available in zip(history_clocks, history.native_interval_minutes,
                      history.price_eur_mwh, history._study_available_at)}
    for lag in (1, 7):
        values, availability = [], []
        for (day, hour, minute, fold, minutes), cutoff in zip(keys, frame.forecast_cutoff):
            value, available = lookup.get((day - timedelta(days=lag), hour, minute, fold, minutes), (np.nan, pd.NaT))
            eligible = not pd.isna(available) and available <= cutoff
            values.append(value if eligible else np.nan)
            availability.append(available if eligible else pd.NaT)
        frame[f"lag_{lag}d"] = values
        frame[f"lag_{lag}d_study_available_at"] = pd.to_datetime(availability, utc=True)
    frame["static_feature_available_at"] = frame[["lag_1d_study_available_at", "lag_7d_study_available_at"]].max(axis=1)
    frame["previous_auction"] = PRECEDING[auction]
    frame["auction"] = auction
    frame["availability_policy"] = POLICY
    frame["operational_eligible"] = False
    return frame


def _attach(frame: pd.DataFrame, sources: Mapping[str, pd.DataFrame], auction: str,
            settings: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Attach only assumed/recorded earlier-auction prices at this decision gate."""
    names = list(FEATURES)
    available_columns = ["static_feature_available_at"]
    for ancestor in _ancestors(auction, settings):
        aligned = _align(sources.get(ancestor, pd.DataFrame()), frame, ancestor, settings)
        prefix = ancestor.lower()
        frame[f"{prefix}_price"] = aligned.price.to_numpy()
        for name in aligned.columns.drop("price"):
            frame[f"{prefix}_{name}"] = aligned[name].to_numpy()
        available_columns.append(f"{prefix}_assumed_or_recorded_available_at")
        if ancestor != PRECEDING[auction]:
            names.append(f"{prefix}_price")
    frame["previous_auction_price"] = frame[f"{PRECEDING[auction].lower()}_price"]
    names.append("previous_auction_price")
    frame["feature_complete"] = np.isfinite(frame[names].to_numpy(float)).all(axis=1)
    frame["assumed_feature_available_at"] = frame[available_columns].apply(lambda values: max(
        (pd.Timestamp(value) for value in values if not pd.isna(value)), default=pd.NaT), axis=1)
    frame.loc[~frame.feature_complete, "assumed_feature_available_at"] = pd.NaT
    frame["assumed_feature_available_at"] = pd.to_datetime(frame.assumed_feature_available_at, utc=True)
    frame.attrs.update(feature_names=names, study_settings=dict(settings), target_policy="own auction minus immediate previous auction")
    return frame, names


def _intraday_products(frame: pd.DataFrame, auction: str, settings: Mapping[str, Any]) -> pd.DataFrame:
    """Keep auction eligibility explicit: IDA3 has only noon-to-midnight labels."""
    if not frame.native_interval_minutes.eq(15).all():
        raise ValueError("Intraday target products must be native quarter-hours; hourly labels are not expanded")
    if auction == "IDA3" and (frame.product_start.dt.tz_convert(settings["timezone"]).dt.hour < 12).any():
        raise ValueError("IDA3 contains an ineligible morning product; do not fabricate morning labels")
    return frame


def _check_zones(reference: pd.DataFrame, sources: Mapping[str, pd.DataFrame], settings: Mapping[str, Any]) -> None:
    """Reject any explicit disagreement across target, history and ancestor zones."""
    zones = {settings["zone"]} if settings.get("zone") else set()
    for frame in (reference, *sources.values()):
        if "zone" in frame and not frame.empty:
            zones.update(frame.zone.dropna().astype(str))
    if len(zones) > 1:
        raise ValueError("Intraday target, history and ancestor prices belong to different zones")


def build_intraday_price_features(records_by_auction: Mapping[str, pd.DataFrame], auction: str,
                                  settings: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Construct observed native targets and eligible retrospective spread features."""
    policy = _settings(settings)
    ancestors = _ancestors(auction, policy)
    own = _records(records_by_auction.get(auction, pd.DataFrame()), auction, policy)
    if own.empty:
        raise ValueError("Actual native intraday target labels are required")
    _intraday_products(own, auction, policy)
    sources = {name: _records(records_by_auction.get(name, pd.DataFrame()), name, policy) for name in ancestors}
    _check_zones(own, sources, policy)
    frame, names = _attach(_base(own, own, auction, policy), sources, auction, policy)
    frame["actual"] = frame.price_eur_mwh
    frame["target"] = frame.actual - frame.previous_auction_price
    frame["assumed_result_at"] = frame._assumed_available_at
    frame["assumed_target_available_at"] = frame._study_available_at
    frame = frame.drop(columns=[name for name in frame if name.startswith("_")])
    return frame, names


def build_future_intraday_price_features(own_history: pd.DataFrame, future_products: pd.DataFrame,
                                         auction: str, known_antecedent_curves: Mapping[str, pd.DataFrame] | None = None,
                                         settings: Mapping[str, Any] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Build a target-free day using prior own history and known earlier curves.

    Missing, not-yet-observed ancestor curves remain missing until the conditional
    forecast rebuilds them from scenario history. No future own-price column or
    dummy target is created. Explicit ``scenario_grid_index`` columns are retained
    for callbacks whose ancestor vectors use a shorter residual physical grid.
    """
    policy = _settings(settings)
    ancestors = _ancestors(auction, policy)
    if set(future_products) & {"actual", "target", "price_eur_mwh"}:
        raise ValueError("Future intraday products must not contain own target price labels")
    products = _intraday_products(_products(future_products, policy["timezone"]), auction, policy)
    if products.empty or products.delivery_day.nunique() != 1:
        raise ValueError("Future intraday products must cover one nonempty local delivery day")
    if products.product_start.duplicated().any():
        raise ValueError("Future intraday products must be unique")
    history = _records(own_history, auction, policy)
    if not history.empty and history.delivery_day.ge(products.delivery_day.iloc[0]).any():
        raise ValueError("Own intraday history must contain strictly earlier delivery days")
    known = known_antecedent_curves or {}
    if set(known) - set(ancestors):
        raise ValueError("Future intraday features cannot use this or a later auction result")
    sources = {name: _records(value, name, policy) for name, value in known.items()}
    _check_zones(products, {"own_history": history, **sources}, policy)
    if "zone" not in products and not history.empty and "zone" in history:
        products["zone"] = history.zone.iloc[0]
    return _attach(_base(products, history, auction, policy), sources, auction, policy)


def _array_ancestor(value: Any, frame: pd.DataFrame, settings: Mapping[str, Any]) -> np.ndarray:
    """Map scenario curves to eligible products without padding actual price labels."""
    values = np.asarray(value, float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Ancestor scenario prices must be one finite native product vector")
    if "scenario_grid_index" in frame:
        raw = frame.scenario_grid_index.to_numpy(float)
        if not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all() or (raw < 0).any() or (raw >= len(values)).any():
            raise ValueError("scenario_grid_index must identify valid ancestor-vector products")
        return values[raw.astype(int)].copy()
    if len(values) == len(frame):
        return values.copy()
    if frame.delivery_day.nunique() != 1 or frame.native_interval_minutes.nunique() != 1:
        raise ValueError("Ancestor array mapping needs one native day or explicit scenario_grid_index")
    day = date.fromisoformat(str(frame.delivery_day.iloc[0]))
    start = pd.Timestamp(day, tz=settings["timezone"])
    end = pd.Timestamp(day + timedelta(days=1), tz=settings["timezone"])
    full = pd.date_range(start.tz_convert("UTC"), end.tz_convert("UTC"),
                         freq=f"{int(frame.native_interval_minutes.iloc[0])}min", inclusive="left")
    positions = full.get_indexer(pd.DatetimeIndex(frame.product_start))
    if len(values) != len(full) or (positions < 0).any():
        raise ValueError("Ancestor vector grid is ambiguous; supply native DataFrame bounds or scenario_grid_index")
    return values[positions].copy()


@dataclass(frozen=True)
class ConditionalSpreadPriceForecast:
    """A linear spread equation that rebuilds dynamic features from ancestor prices."""
    auction: str
    model: RidgePriceModel
    training_max_delivery_day: str
    settings: Mapping[str, Any] = field(default_factory=dict)
    operational_eligible: bool = False

    def __post_init__(self) -> None:
        """Prevent spread artifacts from acquiring unverified operational status."""
        if self.auction not in PRECEDING or self.operational_eligible:
            raise ValueError("Conditional intraday spread artifacts are retrospective IDA studies only")
        date.fromisoformat(self.training_max_delivery_day)
        _aware([self.model.fit_cutoff], "fit_cutoff")
        allowed = set(FEATURES) | {"previous_auction_price"} | {
            f"{name.lower()}_price" for name in _ancestors(self.auction, _settings(self.settings))}
        if not self.model.feature_names or set(self.model.feature_names) - allowed:
            raise ValueError("Conditional spread predictors cannot include own target or later-auction price labels")

    @property
    def fit_cutoff(self) -> str:
        """Expose the trained model boundary to scenario information checks."""
        return self.model.fit_cutoff

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict a spread from a frame whose predecessor is already observed."""
        return self.model.predict(frame)

    def predict_absolute(self, frame: pd.DataFrame, ancestor_prices: Mapping[str, Any]) -> np.ndarray:
        """Return previous price plus the spread conditional on sampled ancestors."""
        policy = _settings(self.settings)
        allowed = _ancestors(self.auction, policy)
        if set(ancestor_prices) - set(allowed):
            raise ValueError("Conditional spread prediction received this or a later auction result")
        rebuilt = _products(frame, policy["timezone"])
        for ancestor in allowed:
            column = f"{ancestor.lower()}_price"
            rebuilt[column] = np.nan  # Stored historical ancestors cannot leak into branch means.
            if ancestor not in ancestor_prices:
                continue
            value = ancestor_prices[ancestor]
            if isinstance(value, pd.DataFrame):
                source = _records(value, ancestor, policy)
                _check_zones(rebuilt, {ancestor: source}, policy)
                if "forecast_cutoff" not in rebuilt:
                    rebuilt["forecast_cutoff"] = [_clock(day, self.auction, policy, results=False) for day in rebuilt.delivery_day]
                rebuilt[column] = _align(source, rebuilt, ancestor, policy, enforce_clock=False).price.to_numpy()
            else:
                rebuilt[column] = _array_ancestor(value, rebuilt, policy)
        previous = PRECEDING[self.auction]
        rebuilt["previous_auction_price"] = rebuilt[f"{previous.lower()}_price"]
        if not np.isfinite(rebuilt.previous_auction_price).all():
            raise ValueError(f"Missing native {previous} predecessor curve for conditional spread prediction")
        return rebuilt.previous_auction_price.to_numpy(float) + self.predict(rebuilt)

    def to_dict(self) -> dict:
        """Serialize the offset equation and its explicit retrospective policy."""
        return {"schema_version": "conditional-intraday-spread-v1", "auction": self.auction,
            "previous_auction": PRECEDING[self.auction], "model": self.model.to_dict(),
            "training_max_delivery_day": self.training_max_delivery_day, "settings": dict(self.settings),
            "operational_eligible": False, "availability_policy": POLICY,
            "prediction_equation": "previous auction level + linear conditional spread; Gaussian innovations added separately"}

    def save(self, path: str | Path) -> None:
        """Save a reusable model without any future own-price labels."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=str, allow_nan=False) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> "ConditionalSpreadPriceForecast":
        """Restore the trained spread while keeping study assumptions explicit."""
        payload = json.loads(Path(path).read_text())
        if payload.get("schema_version") != "conditional-intraday-spread-v1" or payload.get("operational_eligible"):
            raise ValueError("Invalid or operationally promoted retrospective spread artifact")
        model = payload["model"]
        ridge = RidgePriceModel(tuple(model["feature_names"]), np.asarray(model["center"], float), np.asarray(model["scale"], float),
            np.asarray(model["coefficients"], float), float(model["intercept"]), float(model["alpha"]),
            model["fit_cutoff"], int(model["training_rows"]), model["target"])
        return cls(payload["auction"], ridge, payload["training_max_delivery_day"], payload.get("settings", {}))


def fit_intraday_ridge(rows: pd.DataFrame, auction: str, *, fit_cutoff: Any,
                       forecast_delivery_day: str | None = None, feature_names: Sequence[str] | None = None,
                       alpha: float = 1.0, allow_assumed_availability: bool = False) -> ConditionalSpreadPriceForecast:
    """Fit strictly earlier eligible spread labels under an explicit study opt-in."""
    if not allow_assumed_availability:
        raise ValueError("Unverified intraday vintages require allow_assumed_availability=True for a retrospective fit")
    policy = _settings(rows.attrs.get("study_settings"))
    _ancestors(auction, policy)
    cutoff = _aware([fit_cutoff], "fit_cutoff")[0]
    delivery = (date.fromisoformat(forecast_delivery_day) if forecast_delivery_day is not None else
        cutoff.tz_convert(policy["timezone"]).date() + timedelta(days=int(auction != "IDA3")))
    if cutoff > _clock(delivery.isoformat(), auction, policy, results=False):
        raise ValueError("Model fit occurs after the configured intraday decision cutoff")
    names = list(feature_names or rows.attrs.get("feature_names", []))
    allowed = set(FEATURES) | {"previous_auction_price"} | {f"{name.lower()}_price" for name in _ancestors(auction, policy)}
    if not names or set(names) - allowed:
        raise ValueError("Select documented calendar, own-history and earlier-auction features; target labels cannot be predictors")
    if not rows.auction.eq(auction).all() or not rows.availability_policy.eq(POLICY).all():
        raise ValueError("Training rows must match the requested auction and retrospective availability policy")
    feature_times = _aware(rows.assumed_feature_available_at, "assumed_feature_available_at", missing=True)
    label_times = _aware(rows.assumed_target_available_at, "assumed_target_available_at", missing=True)
    decisions = _aware(rows.forecast_cutoff, "forecast_cutoff")
    eligible = (rows.delivery_day.lt(delivery.isoformat()).to_numpy() & (feature_times <= cutoff)
        & (label_times <= cutoff) & (feature_times <= decisions)
        & np.isfinite(rows[names + ["target"]].to_numpy(float)).all(axis=1))
    train = rows.loc[eligible]
    if train.empty:
        raise ValueError("No strictly earlier, complete-feature intraday spread rows are available by the fit cutoff")
    model = fit_ridge(train[names], train.target, feature_available_at=feature_times[eligible],
        target_available_at=label_times[eligible], fit_cutoff=cutoff, alpha=alpha,
        target=f"{auction}_conditional_spread_retrospective_assumed_results")
    return ConditionalSpreadPriceForecast(auction, model, train.delivery_day.max(), policy)


def load_intraday_price_records(database_path: str | Path, *, provider: str = "gme", zone: str = "NORD",
                                 day_ahead_provider: str = "entsoe", start: Any = None, end: Any = None,
                                 settings: Mapping[str, Any] | None = None) -> tuple[dict[str, pd.DataFrame], list[dict]]:
    """Read DuckDB IDA labels and explicitly selected native DA source snapshots.

    Existing history downloads contain ENTSO-E DA and GME IDAs. ENTSO-E's
    provider grid may repeat historical hourly prices on quarters; declared
    native regimes recover the hourly products before they become features.
    Both providers' pinned snapshot identities and checksums remain in provenance.
    """
    from ..data.api_catalog import load_api_cache_records
    from .price_experiment import prepare_native_prices
    path = Path(database_path)
    if path.suffix.lower() != ".duckdb":
        raise ValueError("New intraday pipelines require a DuckDB database path")
    records, provenance = load_api_cache_records(path, provider=provider, zone=zone,
        start=start, end=end, verify_checksums=True)
    if not records.empty and "auction" not in records:
        raise ValueError("Database price records lack an explicit auction")
    prices = ({} if records.empty else {name: group.copy().reset_index(drop=True)
        for name, group in records.groupby("auction", sort=False) if name in AUCTIONS})
    if day_ahead_provider != provider:
        prices.pop("DA", None)
        da, da_sources = load_api_cache_records(path, provider=day_ahead_provider, dataset="day_ahead_prices",
            zone=zone, start=start, end=end, verify_checksums=True)
        provenance = [*provenance, *da_sources]
    else:
        da = prices.pop("DA", pd.DataFrame())
    if not da.empty:
        policy = _settings(settings)
        if "price_eur_mwh" not in da and {"value", "unit"}.issubset(da) and da.unit.eq("EUR/MWh").all():
            da = da.assign(price_eur_mwh=da.value)
        transition = str(policy.get("da_quarter_hour_start", "2025-10-01"))
        earliest = pd.to_datetime(da.product_start, utc=True).dt.tz_convert(policy["timezone"]).dt.date.min().isoformat()
        native_settings = {"timezone": policy["timezone"], "native_product_regimes": policy.get("native_product_regimes", [
            {"start": earliest, "end": transition, "minutes": 60},
            {"start": max(earliest, transition), "end": None, "minutes": 15}])}
        if earliest >= transition and "native_product_regimes" not in policy:
            native_settings["native_product_regimes"] = [{"start": earliest, "end": None, "minutes": 15}]
        prices["DA"] = prepare_native_prices(da, native_settings)
    return prices, provenance
