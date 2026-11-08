"""Lightweight progress heartbeat used by the Slurm stall watchdog.

The heartbeat is deliberately local-file based so it works even when W&B or
network access is unavailable.  Calls are throttled to avoid filesystem churn.
"""
from __future__ import annotations

import json
import os
import socket
import time
from pathlib import Path
from typing import Any

_LAST_WRITE: dict[str, float] = {}


def heartbeat(
    path: str | None,
    stage: str,
    *,
    interval_seconds: float = 60.0,
    force: bool = False,
    **fields: Any,
) -> None:
    if not path:
        return
    path = os.path.abspath(os.path.expanduser(path))
    now = time.time()
    last = _LAST_WRITE.get(path, 0.0)
    if not force and now - last < float(interval_seconds):
        return

    payload = {
        "timestamp": now,
        "iso_time": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(now)),
        "stage": str(stage),
        "pid": os.getpid(),
        "host": socket.gethostname(),
        **fields,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, sort_keys=True, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, p)
    _LAST_WRITE[path] = now
