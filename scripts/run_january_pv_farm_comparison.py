"""Run the matched January deterministic, scenario and perfect-PV comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

from bess_pv_trading.config import load_config
from compare_pv_deterministic_scenarios import compare


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = (
    ROOT / "configs/full_experiments/continuous_2026_01_pv_deterministic_four_farms.yaml",
    ROOT / "configs/full_experiments/continuous_2026_01_pv_scenarios_four_farms.yaml",
)


def main() -> None:
    """Run each replay once, then compare only completed and matched outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-id", type=int, required=True, choices=(1, 345, 901, 910))
    args = parser.parse_args()
    configs = [load_config(path, overrides={"farm_id": args.farm_id}) for path in CONFIGS]

    # Fail before any solve if the requested farm model is incomplete or corrupt.
    model = configs[0]["pv"]["forecast_replacement"]["models_by_farm"][str(args.farm_id)]
    if model is not None:
        manifest_path = ROOT / model["artifact_path"]
        manifest = json.loads(manifest_path.read_text())
        if manifest["farm_id"] != args.farm_id:
            raise ValueError("Forecast model belongs to another farm")
        for name, checksum in manifest["checksums"].items():
            artifact = manifest_path.parent / name
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != checksum:
                raise ValueError(f"Forecast model checksum mismatch: {artifact}")
    elif args.farm_id != 345:
        raise ValueError("No forecast model is configured for this farm")

    # Complete both independent trajectories before reading their comparison inputs.
    for path in CONFIGS:
        subprocess.run([sys.executable, "-m", "bess_pv_trading", "experiment-continuous",
                        "--config", str(path), "--farm-id", str(args.farm_id)],
                       cwd=ROOT, check=True)

    # Derive paths from the resolved configs so output settings have one source.
    roots = [ROOT / item["continuous_optimization"]["output_root"] / f"farm_{args.farm_id}"
             / item["continuous_optimization"]["run_id"] for item in configs]
    output = ROOT / "outputs/continuous_optimization/comparisons" / f"farm_{args.farm_id}" / "2026_01_mixed_models_pv_comparison"
    compare(roots[0], roots[1], output)
    print(f"Comparison saved to {output}", flush=True)


if __name__ == "__main__":
    main()
