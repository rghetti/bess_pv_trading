"""Launch the requested two-farm MPC study on a machine with its inputs installed."""
from __future__ import annotations

import argparse
import os
import signal
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
STUDIES = {
    "january-ensemble": "configs/research/ensemble_k25_january_2026.yaml",
    "january-august-deterministic": "configs/research/deterministic_k25_jan_aug_2026.yaml",
    "year-deterministic": "configs/research/deterministic_k25_full_year_2026.yaml",
}


def farm_command(study, farm, run_id, action):
    """Keep each farm independent and limit each spawned worker to one solver thread."""
    return [sys.executable, "-u", str(ROOT / "scripts/run_energy_research_study.py"),
        "--config", str(ROOT / STUDIES[study]), "--farm", str(farm), "--workers", "1",
        "--solver-threads", "1", "--run-id", f"{run_id}_farm{farm}", action]


def main(argv=None):
    """Launch two farm processes and return failure if either one fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=STUDIES, required=True)
    parser.add_argument("--run-id", required=True, help="Unique study and output identifier")
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--check-only", action="store_true")
    action.add_argument("--run", action="store_true")
    args = parser.parse_args(argv)
    if not args.run_id or not all(ch.isalnum() or ch in "_.-" for ch in args.run_id):
        parser.error("run-id must contain only letters, digits, period, underscore and hyphen")
    if args.run_id[0] in ".-":
        parser.error("run-id must begin with a letter or digit")
    # Distinct log directories prevent accidental replacement of previous evidence.
    log_root = ROOT / "outputs" / "k25_launches" / args.run_id
    try:
        log_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        parser.error(f"log directory exists; choose a new run ID: {log_root}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = "1"
    processes = []
    handles = []
    try:
        # Independent subprocesses ensure farms 1 and 345 actually run in parallel.
        for farm in (1, 345):
            log = log_root / f"farm_{farm}.log"
            handle = log.open("w")
            handles.append(handle)
            command = farm_command(args.study, farm, args.run_id, "--run" if args.run else "--check-only")
            processes.append((farm, subprocess.Popen(command, cwd=ROOT, env=env, stdout=handle,
                stderr=subprocess.STDOUT, start_new_session=True)))
            print(f"Farm {farm}: PID {processes[-1][1].pid}; log {log}", flush=True)
        failures = []
        for farm, process in processes:
            code = process.wait()
            print(f"Farm {farm}: exit {code}", flush=True)
            if code:
                failures.append(farm)
        return 1 if failures else 0
    except BaseException:
        # Stop only this launcher's children when interrupted or a launch fails.
        for _, process in processes:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
        for _, process in processes:
            process.wait()
        raise
    finally:
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
