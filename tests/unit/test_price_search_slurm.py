"""Validate Slurm resource forwarding without launching a real cluster experiment."""

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]


def test_slurm_launcher_forwards_64_workers_and_caps_native_threads(tmp_path):
    """A fake srun records the real launcher arguments and environment, without executing fits."""
    bin_directory = tmp_path / "bin"
    bin_directory.mkdir()
    capture = tmp_path / "captured.json"
    executable = bin_directory / "srun"
    executable.write_text(f"#!{sys.executable}\nimport json,os,sys\nfrom pathlib import Path\n"
                          "Path(os.environ['SLURM_TEST_CAPTURE']).write_text(json.dumps({'args':sys.argv[1:],'omp':os.environ['OMP_NUM_THREADS'],'blas':os.environ['OPENBLAS_NUM_THREADS'],'cuda':os.environ['CUDA_VISIBLE_DEVICES']}))\n")
    executable.chmod(0o755)
    database = tmp_path / "market.duckdb"
    database.touch()
    environment = dict(os.environ, PATH=f"{bin_directory}:{os.environ['PATH']}", SLURM_SUBMIT_DIR=str(ROOT),
                       SLURM_CPUS_PER_TASK="64", PRICE_DB=str(database), PRICE_SEARCH_PYTHON=sys.executable,
                       PRICE_SEARCH_OUTPUT=str(tmp_path / "results"), SLURM_TEST_CAPTURE=str(capture))
    script = ROOT / "scripts/slurm/price_search_64.sbatch"
    subprocess.run(["bash", "-n", str(script)], check=True)
    subprocess.run(["bash", str(script)], check=True, env=environment)
    result = json.loads(capture.read_text())
    assert "--cpus-per-task=64" in result["args"]
    assert result["args"][result["args"].index("--workers") + 1] == "64"
    assert "--run" in result["args"]
    assert result["omp"] == result["blas"] == "1"
    assert result["cuda"] == ""
