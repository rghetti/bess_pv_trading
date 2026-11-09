"""Share immutable prepared inputs across compatible trajectories in one batch."""

from __future__ import annotations

from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile


def preparation_key(config):
    """Identify equivalent preparation inputs, preserving all forecast and physical settings."""
    comparable = deepcopy(config)
    # Only the output identity and these objective-only settings differ between
    # the k25 cases; neither changes their weather or prepared physical inputs.
    settings = comparable.get("continuous_optimization", {})
    settings.pop("run_id", None)
    settings.pop("output_root", None)
    realtime = comparable.get("realtime_mpc", {})
    realtime.pop("first_interval_priority", None)
    realtime.pop("first_interval_weight", None)
    payload = json.dumps(comparable, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256(path):
    """Hash the complete serialized snapshot without keeping it in memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_shared(config, batch, prepare, progress):
    """Build one snapshot per input key; lock concurrent consumers until it is complete.

    Cache files are private to this fresh batch and are never accepted from a
    previous run. They contain Python objects and must remain trusted local
    files; a checksum detects accidental corruption, not malicious replacement.
    """
    cache_dir = Path(batch) / "preflight_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = preparation_key(config)
    data_path = cache_dir / f"{key}.pickle"
    meta_path = cache_dir / f"{key}.json"
    lock_path = cache_dir / f"{key}.lock"
    progress(f"Waiting for shared preflight cache {key[:12]}")
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if meta_path.exists():
            # A completed producer pins both the serialized artifact and the
            # original prepared-input fingerprint before another worker uses it.
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            if meta.get("key") != key or not data_path.is_file() or _sha256(data_path) != meta.get("sha256"):
                raise ValueError("Shared preflight cache is incomplete or checksum-mismatched")
            with data_path.open("rb") as stream:
                prepared = pickle.load(stream)
            if prepared.get("input_fingerprint") != meta.get("input_fingerprint"):
                raise ValueError("Shared preflight cache input fingerprint mismatch")
            progress(f"Reused shared preflight cache {key[:12]}")
            return prepared
        if data_path.exists():
            raise ValueError("Shared preflight cache has an incomplete producer artifact")
        progress(f"Building shared preflight cache {key[:12]}")
        prepared = prepare(config, progress=progress)
        if not prepared.get("input_fingerprint"):
            raise ValueError("Prepared inputs have no fingerprint")
        # Publish an atomic data file and then its metadata; readers hold the
        # same lock and cannot observe an in-progress snapshot.
        with tempfile.NamedTemporaryFile(dir=cache_dir, prefix=f"{key}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            try:
                pickle.dump(prepared, stream, protocol=pickle.HIGHEST_PROTOCOL)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        checksum = _sha256(temporary)
        os.replace(temporary, data_path)
        metadata = {"key": key, "sha256": checksum,
                    "input_fingerprint": prepared["input_fingerprint"], "schema": "batch_preflight_v1"}
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=cache_dir,
                                         prefix=f"{key}.", suffix=".json.tmp", delete=False) as stream:
            temporary_meta = Path(stream.name)
            json.dump(metadata, stream, sort_keys=True, indent=2)
            stream.write("\n")
        os.replace(temporary_meta, meta_path)
        progress(f"Published shared preflight cache {key[:12]}")
        return prepared
