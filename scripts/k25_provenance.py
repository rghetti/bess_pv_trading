"""Publish one immutable provenance record for each batch, farm and input fingerprint."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import tempfile

from bess_pv_trading.simulation.synthetic import json_default


def link_provenance(source, destination):
    """Expose a shared record at a compatible path without copying its contents."""
    source, destination = Path(source), Path(destination)
    try:
        os.link(source, destination)
    except FileExistsError:
        # A retry is safe only if the existing name already identifies this record.
        if not destination.samefile(source):
            raise ValueError(f"Provenance destination already points to different data: {destination}")


def publish_provenance(batch, farm, fingerprint, provenance, destination):
    """Atomically publish one complete record and hard-link each controller's path to it."""
    farm = str(farm)
    if not farm.isdecimal() or not isinstance(fingerprint, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", fingerprint):
        raise ValueError("Invalid farm or input fingerprint for shared provenance")
    directory = Path(batch) / "input_provenance" / f"farm_{farm}"
    directory.mkdir(parents=True, exist_ok=True)
    record = directory / f"{fingerprint}.json"
    # Producers serialize under one per-fingerprint lock; no reader can link a
    # partially written record, including when several workers start together.
    with (directory / f"{fingerprint}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not record.exists():
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=directory,
                                             prefix=".provenance-", suffix=".tmp", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    json.dump(provenance, stream, indent=2, default=json_default, allow_nan=False)
                    stream.write("\n")
                except BaseException:
                    temporary.unlink(missing_ok=True)
                    raise
            try:
                os.replace(temporary, record)
            finally:
                temporary.unlink(missing_ok=True)
        link_provenance(record, destination)
    return record
