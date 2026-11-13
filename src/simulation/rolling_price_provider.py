"""Optional, publication-aware rolling auction-price provider.

The continuous oracle remains independent of this adapter.  A caller supplies a
gate-specific feature example and the curves actually published by that gate.
The rolling engine owns training and feature construction; this adapter owns the
once-per-delivery-day fitting lifecycle, safe fallback and restart state.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


AUCTIONS = ("DA", "IDA1", "IDA2", "IDA3")


def _utc(value: Any) -> pd.Timestamp:
    """Normalize an explicitly zoned time to UTC without assuming a timezone."""
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is None:
        raise ValueError("An explicit timezone-aware timestamp is required")
    return stamp.tz_convert("UTC")


def _digest(path: Path) -> str:
    """Fingerprint a serialized fit without loading it into memory."""
    checksum = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def _atomic_json(path: Path, data: Mapping[str, Any]) -> None:
    """Replace checkpoint metadata only after the entire payload is written."""
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(data, stream, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


@dataclass(frozen=True)
class PublishedAuctionCurve:
    """One actual clearing curve with its individual publication timestamp."""

    auction: str
    product_starts_utc: Sequence[Any]
    values_eur_per_mwh: Sequence[float]
    published_at: Any

    def as_mapping(self, cutoff: Any) -> dict[pd.Timestamp, float] | None:
        """Return aligned prices only after this clearing has been published."""
        if self.auction not in AUCTIONS:
            raise ValueError(f"Unknown auction: {self.auction}")
        if _utc(self.published_at) > _utc(cutoff):
            return None
        starts = [_utc(value) for value in self.product_starts_utc]
        prices = np.asarray(self.values_eur_per_mwh, dtype=float)
        if len(starts) != len(prices) or len(starts) != len(set(starts)):
            raise ValueError("Published products must have unique aligned prices")
        if not np.isfinite(prices).all():
            raise ValueError("Published prices must be finite")
        return dict(zip(starts, prices.tolist()))


def validate_replay_price_schedule(days_data: Mapping[str, Mapping[str, Any]],
                                   expected_auctions: Sequence[str]) -> None:
    """Reject hindsight-filtered cancellations until announcement clocks exist."""
    expected = set(expected_auctions)
    if not expected or expected - set(AUCTIONS):
        raise ValueError("Rolling replay requires explicitly configured auction names")
    for day, daily in days_data.items():
        skipped = daily.get("skipped")
        if skipped is None:
            raise ValueError(f"Rolling replay requires auction skip evidence for {day}")
        unavailable = [row for row in skipped if row.get("reason") != "not_participating"]
        if unavailable:
            raise ValueError(f"Rolling replay cannot use hindsight-filtered skipped auctions for {day}: "
                "cancellation announcement-time handling is not implemented")
        schedule = daily.get("schedule", ())
        available = {event.auction for event in schedule if event.available and event.results_at is not None}
        if available != expected:
            raise ValueError(f"Rolling replay requires the complete configured schedule on {day}; "
                "unavailable or partial auctions need announcement-time handling")


def _validate_gate_core(example: Mapping[str, Any], gate: str,
                        eligible: Mapping[str, tuple[pd.Timestamp, dict[pd.Timestamp, float]]],
                        starts: Sequence[pd.Timestamp], timezone: str) -> None:
    """Verify same-day price inputs against this replay's actual publications."""
    from ..forecasting.studies.rolling_price_data import pad_curve

    blocks = example.get("feature_blocks", {})
    core = np.asarray(blocks.get("core", ()), dtype=float)
    # Twelve 100-slot historical profiles precede three current-day profiles,
    # each followed by its cancellation flag in the rolling-data contract.
    offset = 4 * 3 * 100
    if core.ndim != 1 or len(core) < offset + 3 * 101:
        raise ValueError("Gate example lacks the required current-auction core profiles")
    earlier = AUCTIONS[:AUCTIONS.index(gate)]
    for position, market in enumerate(AUCTIONS[:3]):
        begin = offset + position * 101
        actual_values = core[begin:begin + 100]
        cancelled = core[begin + 100]
        if not np.isfinite(cancelled) or cancelled != 0:
            raise ValueError("Rolling replay cannot use cancellation flags without announcement-time evidence")
        if market not in earlier:
            if np.isfinite(actual_values).any():
                raise ValueError(f"Unpublished {market} clearing appears in gate features")
            continue
        if market not in eligible:
            raise ValueError(f"Cannot use own-gate model until {market} is actually published in this replay")
        mapping = eligible[market][1]
        ordered = sorted(mapping)
        expected = pad_curve(pd.DatetimeIndex(ordered), [mapping[stamp] for stamp in ordered], timezone)
        if not np.allclose(actual_values, expected, rtol=1e-6, atol=1e-4, equal_nan=True):
            raise ValueError(f"Gate feature {market} differs from the replay's published clearing")
    if earlier:
        latest = earlier[-1]
        mapping = eligible[latest][1]
        if any(start not in mapping for start in starts):
            raise ValueError("Latest published clearing does not cover all target products")
        reference = np.asarray(example.get("price_reference", ()), dtype=float)
        expected = np.asarray([mapping[start] for start in starts], dtype=float)
        if reference.shape != expected.shape or not np.allclose(reference, expected, rtol=1e-6, atol=1e-4):
            raise ValueError("Gate price reference differs from the replay's latest published clearing")


class RollingPriceProvider:
    """Cache one daily engine fit and provide gate-aware future-auction curves.

    ``engine`` must provide ``fit_delivery_day(day, cutoff)`` and
    ``predict(fitted, example, da_prediction=None)``.  Feature construction is
    the caller's responsibility and must use the stated cutoff.  Snapshot and
    model identities are required to prevent reusing unrelated checkpoints.
    """

    def __init__(self, engine: Any, *, snapshot_id: str, model_identity: str,
                 cache_directory: str | Path):
        if not snapshot_id or not model_identity:
            raise ValueError("Snapshot and model identities are required")
        self.engine = engine
        self.snapshot_id = str(snapshot_id)
        self.model_identity = str(model_identity)
        identity = json.dumps([self.snapshot_id, self.model_identity], separators=(",", ":"))
        self.root = Path(cache_directory) / hashlib.sha256(identity.encode()).hexdigest()
        self.root.mkdir(parents=True, exist_ok=True)
        self._models: dict[str, Any] = {}

    @contextmanager
    def _lock(self, day: str):
        """Serialize fits and checkpoint updates among farm processes."""
        directory = self.root / day
        directory.mkdir(parents=True, exist_ok=True)
        with (directory / ".lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield directory
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def _state(self, directory: Path, day: str) -> dict[str, Any]:
        """Load and validate the snapshot identity and fit checksum on restart."""
        manifest = directory / "state.json"
        if not manifest.exists():
            raise RuntimeError(f"Initialize {day} at its DA gate before forecasting")
        state = json.loads(manifest.read_text())
        if (state.get("snapshot_id") != self.snapshot_id or state.get("model_identity") != self.model_identity
                or state.get("delivery_day") != day):
            raise ValueError("Rolling checkpoint identity does not match the requested study")
        artifact = directory / "daily_models.joblib"
        if not artifact.is_file() or _digest(artifact) != state.get("models_sha256"):
            raise ValueError("Rolling fit is missing or its checksum changed")
        return state

    def begin_day(self, delivery_day: str, da_cutoff: Any) -> None:
        """Fit once at the DA cutoff or restore a matching durable daily fit."""
        day = pd.Timestamp(delivery_day).date().isoformat()
        cutoff = _utc(da_cutoff).isoformat()
        with self._lock(day) as directory:
            manifest = directory / "state.json"
            if manifest.exists():
                state = self._state(directory, day)
                if state["fit_cutoff_utc"] != cutoff:
                    raise ValueError("The requested DA cutoff differs from the persisted fit")
                if day not in self._models:
                    self._models[day] = joblib.load(directory / "daily_models.joblib")
                return
            # Model fitting uses the initial DA cutoff; later auction results cannot
            # enter the coefficients selected for this delivery day.
            fitted = self.engine.fit_delivery_day(day, pd.Timestamp(cutoff))
            with tempfile.NamedTemporaryFile(dir=directory, delete=False) as stream:
                temporary = Path(stream.name)
            try:
                joblib.dump(fitted, temporary, compress=3)
                os.replace(temporary, directory / "daily_models.joblib")
            finally:
                temporary.unlink(missing_ok=True)
            state = {"delivery_day": day, "snapshot_id": self.snapshot_id,
                "model_identity": self.model_identity, "fit_cutoff_utc": cutoff,
                "models_sha256": _digest(directory / "daily_models.joblib"),
                "da_gate_forecasts": {}}
            _atomic_json(manifest, state)
            self._models[day] = fitted

    def forecast(self, delivery_day: str, gate: str, target: str, cutoff: Any,
                 product_starts_utc: Sequence[Any], *, example: Mapping[str, Any] | None = None,
                 published_curves: Sequence[PublishedAuctionCurve] = ()) -> dict[str, Any]:
        """Predict at the DA or own gate, otherwise carry forward a known curve.

        Curves whose publication time follows the cutoff are never used.  A
        missing future-auction DA forecast falls back to the most recently
        published earlier-auction curve only when all requested products match.
        """
        day = pd.Timestamp(delivery_day).date().isoformat()
        if gate not in AUCTIONS or target not in AUCTIONS:
            raise ValueError("Unknown auction gate or target")
        now = _utc(cutoff)
        starts = [_utc(value) for value in product_starts_utc]
        if not starts or len(starts) != len(set(starts)):
            raise ValueError("A forecast needs unique delivery products")
        with self._lock(day) as directory:
            state = self._state(directory, day)
            if now < _utc(state["fit_cutoff_utc"]):
                raise ValueError("Forecast cutoff precedes the daily model fit")
            if gate == "DA" and now != _utc(state["fit_cutoff_utc"]):
                raise ValueError("DA forecast must use the cutoff used to fit daily models")
            if day not in self._models:
                self._models[day] = joblib.load(directory / "daily_models.joblib")
            # Build a map that cannot contain any unpublished clearing curve.
            eligible: dict[str, tuple[pd.Timestamp, dict[pd.Timestamp, float]]] = {}
            for curve in published_curves:
                mapped = curve.as_mapping(now)
                if mapped is not None:
                    if curve.auction in eligible:
                        raise ValueError(f"Multiple published curves for {curve.auction}")
                    if AUCTIONS.index(curve.auction) >= AUCTIONS.index(gate):
                        raise ValueError("A same- or later-gate actual clearing is not an eligible input")
                    eligible[curve.auction] = (_utc(curve.published_at), mapped)

            own_model = gate == "DA" or gate == target
            if own_model:
                if example is None:
                    raise ValueError("A gate-specific feature example is required")
                if str(example.get("delivery_day")) != day or example.get("market") != target or example.get("gate") != gate:
                    raise ValueError("Feature example does not match requested delivery, gate and auction")
                if _utc(example.get("decision_at")) > now:
                    raise ValueError("Feature example is from after the decision cutoff")
                example_starts = [_utc(value) for value in example.get("product_starts_utc", ())]
                if example_starts != starts:
                    raise ValueError("Example product intervals do not match the requested forecast")
                # Prebuilt inputs may reflect a publication the current replay
                # has not observed; compare every same-day profile before use.
                _validate_gate_core(example, gate, eligible, starts,
                    getattr(self.engine, "timezone", "Europe/Rome"))
                # Never pass the target outcome to the prediction implementation.
                safe_example = dict(example)
                safe_example.pop("target_values", None)
                da_prediction = None
                if gate == "DA" and target != "DA":
                    da = state["da_gate_forecasts"].get("DA")
                    if da is None:
                        raise ValueError("Forecast DA before DA-gate IDA spread curves")
                    by_start = {_utc(t): p for t, p in zip(da["product_starts_utc"], da["prediction"])}
                    if any(start not in by_start for start in starts):
                        raise ValueError("DA forecast does not cover all target products")
                    da_prediction = [by_start[start] for start in starts]
                    # DA-gate IDA forecasts may condition on the DA prediction,
                    # never on the as-yet unpublished DA outcome.
                    safe_example["price_reference"] = da_prediction
                result = self.engine.predict(self._models[day], safe_example, da_prediction=da_prediction)
                values = np.asarray(result["prediction"], dtype=float)
                if values.shape != (len(starts),) or not np.isfinite(values).all():
                    raise ValueError("Engine returned an invalid auction price curve")
                if [_utc(value) for value in result["product_starts_utc"]] != starts:
                    raise ValueError("Engine returned different product intervals")
                forecast = {"delivery_day": day, "auction": target, "gate": gate,
                    "forecast_issue_at": now.isoformat(), "product_starts_utc": [t.isoformat() for t in starts],
                    "prediction": values.tolist(), "method": "rolling_model", "snapshot_id": self.snapshot_id,
                    "fit_cutoff_utc": state["fit_cutoff_utc"], "model": result.get("model"),
                    "metadata": result.get("metadata", {})}
                if gate == "DA":
                    state["da_gate_forecasts"][target] = forecast
                    _atomic_json(directory / "state.json", state)
                return forecast

            # Choose the most recently available complete forecast or clearing;
            # newer actual clearings update later-auction fallback predictions.
            cached = state["da_gate_forecasts"].get(target)
            candidate = None
            if cached is not None:
                by_start = {_utc(t): p for t, p in zip(cached["product_starts_utc"], cached["prediction"])}
                if all(start in by_start for start in starts):
                    candidate = (_utc(cached["forecast_issue_at"]), "carry_da_gate_forecast", None, by_start)
            for earlier in reversed(AUCTIONS[:AUCTIONS.index(gate)]):
                if earlier not in eligible:
                    continue
                available_at, curve = eligible[earlier]
                if all(start in curve for start in starts) and (candidate is None or available_at > candidate[0]):
                    candidate = (available_at, "carry_published_clearing", earlier, curve)
            if candidate is None:
                raise ValueError("No eligible saved forecast or published clearing covers the requested products")
            available_at, method, source, values = candidate
            return {"delivery_day": day, "auction": target, "gate": gate,
                "forecast_issue_at": now.isoformat(), "product_starts_utc": [t.isoformat() for t in starts],
                "prediction": [values[start] for start in starts], "method": method,
                "source_auction": source, "source_available_at": available_at.isoformat(),
                "snapshot_id": self.snapshot_id, "fit_cutoff_utc": state["fit_cutoff_utc"]}


def attach_rolling_price_provider(prepared: dict, engine: Any, examples: Sequence[Mapping[str, Any]], *,
                                  snapshot_id: str, model_identity: str, cache_directory: str | Path) -> None:
    """Opt a prepared replay into rolling prices while preserving its oracle data.

    Call once after normal continuous-input preparation. The engine and examples
    must be constructed from the same pinned snapshot and gate contract.
    """
    indexed = {}
    for example in examples:
        key = (str(example["delivery_day"]), example["market"], example["gate"])
        if key in indexed:
            raise ValueError(f"Duplicate rolling price example: {key}")
        indexed[key] = example
    prepared["rolling_price_provider"] = RollingPriceProvider(
        engine, snapshot_id=snapshot_id, model_identity=model_identity, cache_directory=cache_directory)
    prepared["rolling_price_examples"] = indexed


def forecast_auction_book(provider: RollingPriceProvider, delivery_day: str, gate: str,
                          cutoff: Any, product_starts_utc: Sequence[Any], schedule: Sequence[Any],
                          published_prices: Mapping[str, Sequence[float]],
                          examples: Mapping[tuple[str, str, str], Mapping[str, Any]],
                          targets: Sequence[str] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build a whole-day price book with only published actuals and forecasts.

    The caller passes actual clearing curves solely for already published
    auctions. The provider is initialized at the original DA gate even when
    replay resumes at a later auction; this preserves the original fit cutoff.
    IDA3 values outside its eligible intervals are set to zero because no IDA3
    product is traded there.
    """
    starts = [_utc(value) for value in product_starts_utc]
    events = {event.auction: event for event in schedule if event.available}
    if "DA" not in events or gate not in events:
        raise ValueError("Rolling price replay requires DA and the current gate in the schedule")
    if _utc(events[gate].submission_at) != _utc(cutoff):
        raise ValueError("Price forecast cutoff must equal the auction submission gate")
    if set(published_prices) - set(events):
        raise ValueError("Published prices contain an auction outside the schedule")
    requested = set(events) if targets is None else set(targets)
    if not requested or requested - set(events):
        raise ValueError("Requested price targets must be scheduled auctions")
    provider.begin_day(delivery_day, events["DA"].submission_at)
    curves, records, published = {}, {}, []
    for auction, values in published_prices.items():
        event = events[auction]
        if _utc(event.results_at) > _utc(cutoff):
            raise ValueError(f"Actual {auction} price has not been published")
        array = np.asarray(values, dtype=float)
        if array.shape != (len(starts),) or not np.isfinite(array).all():
            raise ValueError(f"Published {auction} prices do not match the full delivery grid")
        positions = np.asarray(event.eligible_intervals, dtype=int)
        published.append(PublishedAuctionCurve(auction, [starts[i] for i in positions],
            array[positions], event.results_at))
        curves[auction] = array.copy()
        records[auction] = {"method": "published_clearing", "published_at": _utc(event.results_at).isoformat()}

    # Forecast in market order so DA-gate IDA heads condition on predicted DA.
    for auction in AUCTIONS:
        if auction not in requested or auction in curves:
            continue
        positions = np.asarray(events[auction].eligible_intervals, dtype=int)
        if (positions < 0).any() or (positions >= len(starts)).any() or len(set(positions)) != len(positions):
            raise ValueError(f"Invalid eligible intervals for {auction}")
        if auction != "IDA3" and len(positions) != len(starts):
            raise ValueError(f"Unexpected partial-day delivery window for {auction}")
        key = (str(delivery_day), auction, gate)
        example = examples.get(key)
        result = provider.forecast(delivery_day, gate, auction, cutoff,
            [starts[index] for index in positions], example=example, published_curves=published)
        values = np.asarray(result["prediction"], dtype=float)
        if values.shape != (len(positions),) or not np.isfinite(values).all():
            raise ValueError(f"Forecast prices for {auction} are incomplete")
        curve = np.zeros(len(starts), dtype=float)
        curve[positions] = values
        curves[auction] = curve
        records[auction] = {key: value for key, value in result.items() if key not in {"prediction", "product_starts_utc"}}
    return curves, records
