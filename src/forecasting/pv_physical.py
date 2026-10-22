"""Shared, cached pvlib forward model for operational forecasts and parameter fits."""
from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
import pvlib

from ..data import FarmMetadata, night_window_mask


def source_azimuth(angle, positive_direction):
    """Translate the declared south-zero source orientation to pvlib degrees."""
    if positive_direction not in {"east", "west"}:
        raise ValueError("Source azimuth positive direction must be explicitly east or west")
    return (180 + (-1 if positive_direction == "east" else 1) * np.asarray(angle)) % 360


def parameter_defaults(settings: dict) -> dict[str, float]:
    """Keep the independently fixed DC reference out of the fitted parameter vector."""
    output = settings["output"]
    coefficient = (output.get("calibrated_coefficient") if output["method"] == "calibrated_scale"
                   else output["pv_output_efficiency"])
    if coefficient is None:
        coefficient = output["pv_output_efficiency"]
    return {"coefficient": float(coefficient), "u0": float(settings["temperature"]["u0"]),
            "u1": float(settings["temperature"]["u1"]),
            "gcr": float(settings["tracker"]["gcr"]),
            "max_angle": float(settings["tracker"]["max_angle"])}


def parameter_settings(settings: dict, parameters: dict, *, transposition=None, bifacial=None) -> dict:
    """Create a portable override with the same meaning inside the production engine."""
    result = deepcopy(settings)
    result["parameter_registry"] = None
    result["output"].update(method="calibrated_scale", calibrated_coefficient=parameters["coefficient"])
    result["temperature"].update({key: parameters[key] for key in ("u0", "u1") if key in parameters})
    result["tracker"].update({key: parameters[key] for key in ("gcr", "max_angle") if key in parameters})
    if transposition is not None:
        result["transposition_model"] = transposition
    if bifacial is not None:
        result.setdefault("bifacial", {})["enabled"] = bifacial
    return result


class PhysicalPVModel:
    """Evaluate cached solar/radiation inputs without rereading weather or targets.

    The frame contains radiation diagnostics prepared by forecast_pv. The same
    class computes the final AC prediction in that function and during fitting.
    """

    def __init__(self, frame: pd.DataFrame, farm: FarmMetadata, settings: dict,
                 night_policy: dict):
        self.frame, self.farm, self.settings = frame, farm, settings
        self.zenith = frame.solar_zenith_deg.to_numpy(dtype=float)
        self.azimuth = frame.solar_azimuth_deg.to_numpy(dtype=float)
        self.ghi = frame.ghi_w_m2.to_numpy(dtype=float)
        self.dhi = frame.dhi_w_m2.to_numpy(dtype=float)
        self.dni = frame.dni_w_m2.to_numpy(dtype=float)
        self.air = frame.temperature_c.to_numpy(dtype=float)
        self.wind = frame.wind_speed_m_s.to_numpy(dtype=float)
        self.valid = (frame.pv_input_valid.to_numpy(dtype=bool) & np.isfinite(self.air)
                      & np.isfinite(self.wind))
        self.night = night_window_mask(frame.index, night_policy)
        self.extra = np.asarray(pvlib.irradiance.get_extra_radiation(
            pd.DatetimeIndex(frame.index) + pd.Timedelta(minutes=7.5)))
        self.airmass = np.asarray(pvlib.atmosphere.get_relative_airmass(self.zenith))
        limit = settings["output"].get("ac_limit_mw")
        self.ac_limit = min(float(limit), farm.ac_capacity_mw) if limit is not None else farm.ac_capacity_mw
        if self.ac_limit <= 0:
            raise ValueError("Configured AC limit must be positive")
        self._irradiance_cache: OrderedDict = OrderedDict()

    def irradiance(self, parameters: dict) -> tuple:
        """Cache geometry-dependent POA across repeated scale/thermal evaluations."""
        settings, farm = self.settings, self.farm
        tracker = settings["tracker"]
        rear = settings.get("bifacial", {})
        bifacial = bool(rear.get("enabled", False))
        if bifacial and farm.tracking_type_id != 3:
            raise ValueError("Explicit bifacial irradiance requires tracking_type_id 3")
        model = settings.get("transposition_model", "isotropic")
        if model not in {"isotropic", "haydavies", "perez-driesse"}:
            raise ValueError("Unsupported PV transposition model")
        key = (parameters["gcr"], parameters["max_angle"], model, bifacial,
               rear.get("bifaciality"), rear.get("height_to_pitch_ratio"), settings.get("albedo"))
        if key in self._irradiance_cache:
            return self._irradiance_cache[key]
        if farm.tracking_type_id == 0:
            tilt = np.full(len(self.ghi), farm.surface_tilt_deg)
            azimuth = np.full(len(self.ghi), source_azimuth(farm.source_azimuth_deg,
                settings.get("source_azimuth_positive", "unconfirmed")))
        elif farm.tracking_type_id in {1, 3}:
            if not tracker.get("assumptions_acknowledged", False):
                raise ValueError("Unresolved tracker axis geometry requires acknowledged configured assumptions")
            if farm.tracking_type_id == 3 and not bifacial and not settings.get("bifacial_front_side_only_enabled", False):
                raise ValueError("Bifacial farm requires explicit front-side-only approximation or rear-side modeling")
            orientation = pvlib.tracking.singleaxis(self.zenith, self.azimuth,
                axis_tilt=float(tracker["axis_tilt"]), axis_azimuth=float(tracker["axis_azimuth"]),
                max_angle=parameters["max_angle"], backtrack=bool(tracker["backtrack"]), gcr=parameters["gcr"])
            tilt = np.where(self.zenith >= 90, 0, np.asarray(orientation["surface_tilt"]))
            azimuth = np.where(self.zenith >= 90, float(tracker["axis_azimuth"]), np.asarray(orientation["surface_azimuth"]))
        else:
            raise ValueError(f"Unsupported tracking ID {farm.tracking_type_id}")
        if bifacial:
            if model not in {"isotropic", "haydavies"}:
                raise ValueError("Infinite-sheds irradiance supports isotropic or haydavies")
            height_ratio = float(rear["height_to_pitch_ratio"])
            if height_ratio <= parameters["gcr"] * np.sin(np.radians(parameters["max_angle"])) / 2:
                raise ValueError("Bifacial row geometry would intersect the ground")
            bifaciality = float(rear["bifaciality"])
            if not 0 <= bifaciality <= 1:
                raise ValueError("Bifaciality must be between zero and one")
            irradiance = pvlib.bifacial.infinite_sheds.get_irradiance(tilt, azimuth,
                self.zenith, self.azimuth, parameters["gcr"], height_ratio, 1.0,
                self.ghi, self.dhi, self.dni, float(settings["albedo"]), model=model,
                dni_extra=self.extra, bifaciality=bifaciality,
                shade_factor=float(rear.get("shade_factor", -0.02)),
                transmission_factor=float(rear.get("transmission_factor", 0)))
            poa = np.asarray(irradiance["poa_global"])
            front, back = np.asarray(irradiance["poa_front"]), np.asarray(irradiance["poa_back"])
            thermal_poa = front + back
        else:
            irradiance = pvlib.irradiance.get_total_irradiance(tilt, azimuth,
                self.zenith, self.azimuth, self.dni, self.ghi, self.dhi,
                dni_extra=self.extra, airmass=self.airmass, albedo=float(settings["albedo"]), model=model)
            poa = np.asarray(irradiance["poa_global"])
            front, back, thermal_poa = poa, np.zeros_like(poa), poa
        # Anisotropic models have undefined nighttime air mass; zero radiation
        # is still physically zero, whereas missing radiation remains missing.
        dark = (self.ghi == 0) & (self.dni == 0) & (self.dhi == 0)
        result = tuple(np.where(dark, 0, x) for x in (poa, thermal_poa, front, back)) + (tilt, azimuth)
        self._irradiance_cache[key] = result
        if len(self._irradiance_cache) > 2:
            self._irradiance_cache.popitem(last=False)
        return result

    def evaluate(self, parameters: dict | None = None, *, diagnostics=False):
        """Apply Faiman, PVWatts, the separate AC scale, clipping and night mask."""
        parameters = parameters or parameter_defaults(self.settings)
        settings = self.settings
        temperature, dc, output = settings["temperature"], settings["dc"], settings["output"]
        if temperature.get("model") != "faiman" or not temperature.get("wind_as_module_proxy_acknowledged"):
            raise ValueError("PV requires the acknowledged Faiman temperature model")
        if dc.get("model") != "pvwatts":
            raise ValueError("Only PVWatts DC is supported")
        reference = float(dc["reference_conversion_efficiency"])
        transformer = float(output["transformer_efficiency"])
        if not 0 < reference <= 1:
            raise ValueError("PV DC reference conversion efficiency requires 0<eta<=1")
        if not 0 < transformer <= 1 or not output.get("transformer_assumption_status"):
            raise ValueError("PV transformer efficiency requires 0<eta<=1 and assumption provenance")
        if parameters["coefficient"] <= 0 or parameters["u0"] <= 0 or parameters["u1"] < 0:
            raise ValueError("Invalid PV scale or thermal parameters")
        poa, thermal, front, back, tilt, azimuth = self.irradiance(parameters)
        cell = pvlib.temperature.faiman(thermal, self.air, self.wind,
            u0=parameters["u0"], u1=parameters["u1"])
        modeled = np.maximum(np.asarray(pvlib.pvsystem.pvwatts_dc(poa, cell,
            pdc0=self.farm.ac_capacity_mw / reference,
            gamma_pdc=float(dc["gamma_pdc_per_c"]), temp_ref=float(dc["reference_temperature_c"]))), 0)
        valid = self.valid & np.isfinite(modeled)
        modeled = np.where(valid, modeled, np.nan)
        raw = modeled * parameters["coefficient"] * transformer
        before_mask = np.minimum(raw, self.ac_limit)
        ac = np.where(self.night & (before_mask > 0), 0, before_mask)
        if not diagnostics:
            return ac
        return {"surface_tilt_deg": tilt, "surface_azimuth_deg": azimuth,
                "poa_w_m2": poa, "poa_front_w_m2": front, "poa_back_w_m2": back,
                "thermal_poa_w_m2": thermal, "cell_temperature_c": cell,
                "modeled_dc_mw": modeled, "pv_ac_before_night_mask_mw": before_mask,
                "pv_ac_mw": ac, "ac_limit_clipped": raw > self.ac_limit,
                "nighttime_zero": self.night & (before_mask > 0), "pv_input_valid": valid}
