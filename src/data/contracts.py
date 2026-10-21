"""Small, solver-independent data contracts; all source timestamps are UTC."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


# UTC-naive source coordinates denote UTC rather than the host's local timezone.
def utc_timestamp(value: Any) -> pd.Timestamp:
    """Normalize one timestamp to timezone-aware UTC."""
    timestamp = pd.Timestamp(value)
    return timestamp.tz_localize("UTC") if timestamp.tzinfo is None else timestamp.tz_convert("UTC")


@dataclass(frozen=True)
class FarmMetadata:
    """Confirmed static fields; no module specification is inferred from type IDs."""
    farm_id: int
    zone: str
    ac_capacity_kw: float
    tracking_type_id: int
    surface_tilt_deg: float
    source_azimuth_deg: float
    elevation_m: float
    pv_type_id: float | None = None
    assumptions: dict[str, Any] = field(default_factory=dict)

    @property
    def ac_capacity_mw(self) -> float:
        """Express AC nameplate capacity in optimizer power units."""
        return self.ac_capacity_kw / 1000.0


@dataclass
class ObservationBundle:
    """Native observations and their separate derived values/evidence."""
    frame: pd.DataFrame
    provenance: dict[str, Any] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)

    def as_of(self, cutoff: Any, exclude_reviewed: bool = True) -> pd.DataFrame:
        """Expose only completed observations and transformations already justified."""
        cutoff_utc = utc_timestamp(cutoff)
        frame = self.frame.loc[self.frame.available_at <= cutoff_utc].copy()
        frame["value"] = frame.raw_value
        if "plateau_zero" in frame:
            eligible = frame.plateau_zero & (frame.plateau_evidence_at <= cutoff_utc)
            frame.loc[eligible, "value"] = 0.0
        if "measurement_interpolated" in frame:
            eligible = frame.measurement_interpolated & (frame.evidence_at <= cutoff_utc)
            frame.loc[eligible, "value"] = frame.loc[eligible, "treated_value"]
        if "night_window" in frame:
            frame.loc[frame.night_window & (frame.value > 0), "value"] = 0.0
        if exclude_reviewed and "review_excluded" in frame:
            frame.loc[frame.get("derived_review_excluded", frame.review_excluded), "value"] = np.nan
        return frame


@dataclass
class ForecastBundle:
    """One selected weather vintage, with a distinct explicit member axis."""
    farm_id: int
    forecast_vintage: pd.Timestamp
    available_at: pd.Timestamp
    values: pd.DataFrame
    units: dict[str, str]
    members: tuple[str, ...] = ("deterministic",)
    member_origin: str = "deterministic_weather_forecast"
    provenance: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject accidental conversion of forecast vintages into members."""
        self.forecast_vintage = utc_timestamp(self.forecast_vintage)
        self.available_at = utc_timestamp(self.available_at)
        if not self.members or len(set(self.members)) != len(self.members):
            raise ValueError("Forecast members must be explicit, nonempty and unique")
        if "forecast_vintage" in self.values and self.values.forecast_vintage.nunique() > 1:
            raise ValueError("A selected ForecastBundle must contain only one vintage")
