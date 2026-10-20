"""Causal PV and battery auction trading, with separate original audit tools."""

from pathlib import Path

# Also expose the preserved audit when the package is imported via PYTHONPATH.
_audit_package = Path(__file__).resolve().parents[1] / "bess_pv_trading"
if _audit_package.is_dir():
    __path__.append(str(_audit_package))
__version__ = "0.2.0"
