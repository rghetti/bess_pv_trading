"""Preserve package imports and CLI execution when src is on PYTHONPATH."""

import importlib.util as _import_util
from pathlib import Path as _Path
import sys as _sys


# Load the flat source directory as the public package, including audit tools.
_spec = _import_util.spec_from_file_location(
    "bess_pv_trading", _Path(__file__).with_name("__init__.py"))
_package = _import_util.module_from_spec(_spec)
_sys.modules[_spec.name] = _package
_spec.loader.exec_module(_package)

# Process this part of the workflow when `__name__ == '__main__'`.
if __name__ == "__main__":
    from bess_pv_trading.cli import main

    raise SystemExit(main())
