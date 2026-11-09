"""Remove a completed K25 batch's temporary preflight cache, optionally waiting."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
import time

from k25_preflight_cache import clear_preflight_cache


def completed_cache_cleanup(batch):
    """Reject active or incomplete batches before removing only their temporary cache."""
    batch = Path(batch).resolve()
    manifest_path = batch / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    status = manifest.get("status")
    if status not in {"complete", "preflight_passed"}:
        raise ValueError(f"Batch is not successfully finished: {status}")
    plan = manifest.get("plan", {})
    if plan.get("scheduler") != "controller_trajectories":
        raise ValueError("Not a controller-trajectory batch")
    jobs = plan.get("jobs", [])
    expected = {(job["job_id"], controller) for job in jobs for controller in job["controllers"]}
    results = manifest.get("results", [])
    observed = {(item["job_id"], item["controller"]) for item in results}
    required_status = "complete" if status == "complete" else "preflight_passed"
    if not expected or observed != expected or len(results) != len(expected) or any(
            item.get("status") != required_status for item in results):
        raise ValueError("Batch has missing, duplicate or incomplete trajectories")
    if status == "complete" and manifest.get("reports", {}).get("status") != "complete":
        raise ValueError("Batch reports are not complete")
    result = clear_preflight_cache(batch)
    manifest["preflight_cache_cleanup"] = result
    # The completed manifest is replaced atomically after recording the cleanup.
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=batch,
                                     prefix=".cache-cleanup-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, manifest_path)
    return result


def main(argv=None):
    """Wait only while a named batch is running; never clean a failed batch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch", type=Path)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-interval-seconds", type=float, default=30.0)
    args = parser.parse_args(argv)
    if not 0 < args.poll_interval_seconds <= 60:
        parser.error("poll interval must be greater than zero and at most 60 seconds")
    while True:
        manifest = json.loads((args.batch / "manifest.json").read_text(encoding="utf-8"))
        status = manifest.get("status")
        if status in {"complete", "preflight_passed"}:
            result = completed_cache_cleanup(args.batch)
            print(f"Preflight cache {result['status']}; logical bytes: {result['logical_bytes']}", flush=True)
            return 0
        if status != "running" or not args.wait:
            print(f"Cache retained: batch status is {status}", flush=True)
            return 1
        time.sleep(args.poll_interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
