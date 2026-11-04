"""Causal nominal continuation values for a PV battery's remaining flexibility.

The dynamic program conditions on forecasts available at the current decision.
It is an approximate terminal value, not an oracle, a lifetime model, or a new
market commitment. Piecewise-linear interpolation is explicit in the provenance.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..domain import DeliveryGrid, PlantParameters
from .energy_prices import _aware


@dataclass(frozen=True)
class ContinuationSettings:
    """Discretization and economic approximation used only after the MPC horizon."""
    energy_grid_points: int = 51
    throughput_cost_eur_per_mwh: float = 0.0
    import_fee_eur_per_mwh: float = 0.0
    final_inventory_value_eur_per_mwh: float = 0.0

    def __post_init__(self):
        if isinstance(self.energy_grid_points, bool) or not isinstance(self.energy_grid_points, int) or self.energy_grid_points < 2:
            raise ValueError("Continuation energy_grid_points must be an integer >= 2")
        for name in ("throughput_cost_eur_per_mwh", "import_fee_eur_per_mwh", "final_inventory_value_eur_per_mwh"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not np.isfinite(value):
                raise ValueError(f"Continuation {name} must be finite")
            if name != "final_inventory_value_eur_per_mwh" and value < 0:
                raise ValueError(f"Continuation {name} must be nonnegative")


def _dispatch_value(charge, discharge, pv, committed, duration, positive, negative, plant, settings):
    """Optimize curtailment and settlement exactly for each feasible battery action."""
    lower = np.maximum(discharge - charge, -plant.poi_import_mw)
    upper = np.minimum(discharge - charge + pv, plant.poi_export_mw)
    feasible = lower <= upper + 1e-12
    # A piecewise-linear settlement optimum lies at a bound or one of its kinks.
    candidates = np.stack([lower, upper, np.clip(np.full_like(lower, committed / duration), lower, upper),
                           np.clip(np.zeros_like(lower), lower, upper)])
    imbalance = candidates * duration - committed
    settlement = positive * np.maximum(imbalance, 0) - negative * np.maximum(-imbalance, 0)
    imports = settings.import_fee_eur_per_mwh * duration * np.maximum(-candidates, 0)
    wear = settings.throughput_cost_eur_per_mwh * duration * (charge + discharge)
    value = np.max(settlement - imports, axis=0) - wear
    return np.where(feasible, value, -np.inf)


def build_continuation_value(*, cutoff, available_at, grid: DeliveryGrid, plant: PlantParameters,
                             positive_eur_per_mwh, pv_ac_mw, committed_mwh,
                             negative_eur_per_mwh=None, settings=None,
                             provenance: Mapping[str, Any] | None = None):
    """Return an MPC TerminalValueCurve from causal conditional mean forecasts.

    The supplied grid starts at the end of the MPC horizon. Its PV and price
    curves must be forecasts available at ``cutoff``. Known commitments may enter;
    later auction opportunities require an explicit separate forecast of value.
    The state is stored energy only, so Xu cycle-memory effects are not modeled
    in this continuation approximation and must be disclosed by its caller.
    """
    from ..optimization.mpc import TerminalValueCurve

    settings = settings or ContinuationSettings()
    cutoff = _aware(cutoff, "continuation decision cutoff")
    available_at = _aware(available_at, "continuation input availability")
    if (provenance or {}).get("forecast_is_oracle") is True:
        raise ValueError("A causal continuation value cannot use disclosed oracle curves")
    if available_at > cutoff or pd.Timestamp(grid.starts[0]) < cutoff:
        raise ValueError("Continuation uses only forecasts available by cutoff and future products")
    count = len(grid.interval_ids)
    curves = {"positive_eur_per_mwh": positive_eur_per_mwh, "pv_ac_mw": pv_ac_mw,
              "committed_mwh": committed_mwh,
              "negative_eur_per_mwh": positive_eur_per_mwh if negative_eur_per_mwh is None else negative_eur_per_mwh}
    for name, values in curves.items():
        values = np.asarray(values, dtype=float)
        if values.shape != (count,) or not np.isfinite(values).all():
            raise ValueError(f"Continuation {name} must be a finite aligned vector")
        curves[name] = values
    if np.any(curves["pv_ac_mw"] < 0):
        raise ValueError("Continuation PV cannot be negative")
    battery = plant.battery
    if battery.maximum_energy <= battery.energy_min_mwh:
        raise ValueError("Continuation needs a positive battery operating energy range")
    energy = np.linspace(battery.energy_min_mwh, battery.maximum_energy, settings.energy_grid_points)
    future_value = energy * settings.final_inventory_value_eur_per_mwh
    # Work backward through the nominal forecast, vectorizing all energy transitions.
    for t in reversed(range(count)):
        duration = grid.delta_hours[t]
        decayed = energy * (1 - battery.self_discharge_per_hour) ** duration
        difference = energy[None, :] - decayed[:, None]
        charge = np.maximum(difference, 0) / (battery.eta_charge * duration)
        discharge = np.maximum(-difference, 0) * battery.eta_discharge / duration
        feasible = (charge <= battery.max_power_mw + 1e-12) & (discharge <= battery.max_power_mw + 1e-12)
        reward = _dispatch_value(charge, discharge, curves["pv_ac_mw"][t], curves["committed_mwh"][t],
                                 duration, curves["positive_eur_per_mwh"][t], curves["negative_eur_per_mwh"][t],
                                 plant, settings)
        candidate = np.where(feasible, reward + future_value[None, :], -np.inf)
        value = np.max(candidate, axis=1)
        # Idle remains available even when self-discharge leaves the discrete grid.
        idle_reward = _dispatch_value(np.zeros_like(energy), np.zeros_like(energy),
                                      curves["pv_ac_mw"][t], curves["committed_mwh"][t], duration,
                                      curves["positive_eur_per_mwh"][t], curves["negative_eur_per_mwh"][t],
                                      plant, settings)
        idle = idle_reward + np.interp(decayed, energy, future_value)
        idle[decayed < battery.energy_min_mwh - 1e-12] = -np.inf
        future_value = np.maximum(value, idle)
        if not np.isfinite(future_value).all():
            raise ValueError("Continuation energy grid contains infeasible states; refine grid or inspect physical inputs")
    evidence = dict(provenance or {})
    evidence.update({"method": "conditional_nominal_storage_dynamic_program",
                     "decision_cutoff_utc": cutoff.isoformat(),
                     "input_available_at_utc": available_at.isoformat(),
                     "continuation_start_utc": pd.Timestamp(grid.starts[0]).isoformat(),
                     "continuation_end_utc": pd.Timestamp(grid.ends[-1]).isoformat(),
                     "settings": asdict(settings), "plant": asdict(plant),
                     "forecast_is_oracle": False,
                     "uncertainty_approximation": "conditional_mean_curves_no_future_realizations",
                     "energy_approximation": "discrete_energy_transitions_plus_interpolated_idle",
                     "degradation_approximation": "linear_throughput_without_Xu_cycle_memory",
                     "future_auction_recourse": "none_only_supplied_fixed_commitments"})
    return TerminalValueCurve(energy_mwh=energy, value_eur=future_value,
                              available_at=cutoff, provenance=evidence)
