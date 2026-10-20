"""Check public imports and CLI discovery outside the repository directory."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("import_root", [ROOT, ROOT / "src"], ids=["repository", "src"])
def test_public_imports_and_legacy_audit_from_another_directory(tmp_path, import_root):
    """Both import roots expose one set of application classes and audit exports."""
    environment = {**os.environ, "PYTHONPATH": str(import_root)}
    result = subprocess.run(
        [sys.executable, "-B", "-c", """
import json
from bess_pv_trading import domain
from bess_pv_trading.data import build_historical_context, select_forecast_vintage
from bess_pv_trading.forecasting import pv
from bess_pv_trading.optimization import model
from bess_pv_trading.simulation import replay
import bess_pv_trading.audit

assert model.OptimizationInput is domain.OptimizationInput
assert replay.PhysicalState is domain.PhysicalState
print(json.dumps({"domain_file": domain.__file__}))
"""],
        cwd=tmp_path, env=environment, text=True, capture_output=True, check=True)
    assert Path(json.loads(result.stdout)["domain_file"]).resolve() == ROOT / "src/domain.py"


@pytest.mark.parametrize("import_root", [ROOT, ROOT / "src"], ids=["repository", "src"])
def test_public_cli_from_another_directory(tmp_path, import_root):
    """The existing module command remains discoverable with either import root."""
    environment = {**os.environ, "PYTHONPATH": str(import_root)}
    result = subprocess.run(
        [sys.executable, "-B", "-m", "bess_pv_trading", "--help"],
        cwd=tmp_path, env=environment, text=True, capture_output=True, check=True)
    assert "experiment-gme-oracle" in result.stdout
    assert "--config" in result.stdout
