"""Farm audit and trading application, runnable without an installation."""

from pathlib import Path

# Prefer the new application while retaining the original audit modules.
_source_package = Path(__file__).resolve().parents[1] / "src" / "bess_pv_trading"
if _source_package.is_dir():
    __path__.insert(0, str(_source_package))

__all__ = ["__version__"]
__version__ = "0.2.0"
