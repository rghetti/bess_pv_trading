"""Numerical comparison tolerances for continuous execution and reporting."""
from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class NumericalTolerances:
    """Absolute power/energy and money tolerances, plus a dimensionless relative tolerance."""
    physical_absolute: float = 1e-6
    money_absolute_eur: float = 1e-6
    relative: float = 1e-8

    def __post_init__(self):
        for name, value in (("physical_absolute", self.physical_absolute),
                            ("money_absolute_eur", self.money_absolute_eur), ("relative", self.relative)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ValueError(f"Numerical tolerance {name} must be finite and nonnegative")

    def close(self, left, right, *, monetary=False):
        """Compare with an absolute allowance plus a symmetric relative allowance."""
        left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
        absolute = self.money_absolute_eur if monetary else self.physical_absolute
        return np.isfinite(left) & np.isfinite(right) & (np.abs(left - right) <=
            absolute + self.relative * np.maximum(np.abs(left), np.abs(right)))


def continuous_tolerances(config):
    """Resolve opt-in overrides without changing saved configuration identities."""
    execution = config.get("execution", {})
    return NumericalTolerances(
        physical_absolute=execution.get("validation_physical_absolute_tolerance",
            execution.get("comparison_tolerance_mw", 1e-6)),
        money_absolute_eur=execution.get("validation_money_absolute_tolerance_eur", 1e-6),
        relative=execution.get("validation_relative_tolerance", 1e-8))
