"""Read-only data adapters and separate causal derived-data policy."""
from .contracts import FarmMetadata, ForecastBundle, ObservationBundle, utc_timestamp
from .adapters import (
    align_forecast, apply_review_exclusions, config_path, forecast_valid_times,
    inspect_data, lead_timedeltas, load_metadata, load_observations, open_store,
    prepare_analysis_data, select_forecast_vintage, validate_source_config,
)
from .policy import clip_negative_forecasts, night_window_mask, plateau_runs, prepare_observations

# Keep the original audit imports and its review verdicts exactly compatible.
import importlib.util as _import_util
from pathlib import Path as _Path
import sys as _sys

_audit_source = _Path(__file__).resolve().parents[2] / "bess_pv_trading" / "data.py"
if _audit_source.is_file():
    _spec = _import_util.spec_from_file_location("bess_pv_trading._legacy_data", _audit_source)
    _legacy = _import_util.module_from_spec(_spec)
    _sys.modules[_spec.name] = _legacy
    _spec.loader.exec_module(_legacy)
    for _name in dir(_legacy):
        if not _name.startswith("_") and _name not in globals():
            globals()[_name] = getattr(_legacy, _name)
