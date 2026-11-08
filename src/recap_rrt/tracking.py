"""Small Weights & Biases wrapper with a no-op disabled mode.

Only deviation from the reproduction copy: the training-batch counter is still
passed as W&B's internal step (so every run shares the same x-axis as the
single-GPU baselines) and is also logged as the plain metric ``train/step``, but
a step that would move backwards is clamped to the last step used.  The model's
diagnostic logs are emitted inside the forward pass with the pre-increment batch
counter, one behind the trainer, which W&B's monotone internal step rejected
("Tried to log to step N that is less than the current step N+1").
"""
from __future__ import annotations

import logging
import time
from typing import Any, Mapping, Sequence

try:
    import wandb as _wandb
except ImportError:  # allows unit/smoke tests with tracking disabled
    _wandb = None

_ACTIVE = False
_LAST_STEP = -1
STEP_KEY = "train/step"


def init(
    *,
    project: str,
    config: Mapping[str, Any],
    name: str | None = None,
    entity: str | None = None,
    mode: str = "online",
    tags: Sequence[str] | None = None,
    run_dir: str | None = None,
    group: str | None = None,
):
    """Initialize W&B. mode can be online, offline, or disabled."""
    global _ACTIVE
    mode = (mode or "online").lower()
    if mode == "disabled":
        _ACTIVE = False
        logging.info("W&B logging disabled.")
        return None
    if _wandb is None:
        raise ImportError("wandb is not installed. Run `pip install wandb` or use `--wandb-mode disabled`.")
    run = _wandb.init(project=project, entity=entity, name=name, config=dict(config), mode=mode,
                      tags=list(tags or []), dir=run_dir, group=group or None)
    _ACTIVE = run is not None
    global _LAST_STEP
    _LAST_STEP = -1
    return run


def log(data: Mapping[str, Any], step: int | None = None, commit: bool = True) -> None:
    global _LAST_STEP
    if _ACTIVE and _wandb is not None and _wandb.run is not None:
        payload = dict(data)
        if step is None:
            _wandb.log(payload, commit=commit)
            return
        # W&B closes a step once it is committed, so a later log at the same or an
        # earlier step is dropped with a warning; advance such logs to the next step.
        step = int(step) if int(step) > _LAST_STEP else _LAST_STEP + 1
        _LAST_STEP = step
        payload[STEP_KEY] = step
        _wandb.log(payload, step=step, commit=commit)


def summary_update(data: Mapping[str, Any]) -> None:
    if _ACTIVE and _wandb is not None and _wandb.run is not None:
        for key, value in data.items():
            _wandb.run.summary[key] = value


def watch(model, log: str = "gradients", log_freq: int = 1000) -> None:
    if _ACTIVE and _wandb is not None and _wandb.run is not None:
        _wandb.watch(model, log=log, log_freq=log_freq)


def run_identity() -> dict | None:
    """(entity, project, id) of the active run, for post-finish verification."""
    if _ACTIVE and _wandb is not None and _wandb.run is not None:
        r = _wandb.run
        return {"entity": r.entity, "project": r.project, "id": r.id}
    return None


def finish() -> None:
    global _ACTIVE
    if _ACTIVE and _wandb is not None and _wandb.run is not None:
        _wandb.finish()
    _ACTIVE = False


def verify_summary(identity: dict | None, expected: Mapping[str, Any], attempts: int = 4, wait: float = 20.0) -> dict:
    """After finish(): read the run back from the W&B server and rewrite any expected summary key
    that is missing or differs.  Returns {"missing_before": [...], "repaired": bool}.

    A run's final records can be lost if the process exits before the last upload completes;
    the server-side summary is what the UI shows, so it is checked explicitly.
    """
    if identity is None or _wandb is None or not expected:
        return {"missing_before": [], "repaired": False, "skipped": True}
    path = f"{identity['entity']}/{identity['project']}/{identity['id']}"
    missing = []
    for attempt in range(attempts):
        try:
            api = _wandb.Api(timeout=60)
            run = api.run(path)
            server = run.summary._json_dict if hasattr(run.summary, "_json_dict") else dict(run.summary)
            missing = [k for k, v in expected.items()
                       if k not in server or (isinstance(v, float) and abs(float(server[k]) - v) > 1e-9)]
            if not missing:
                return {"missing_before": [], "repaired": False}
            for k in missing:
                run.summary[k] = expected[k]
            run.summary.update()
            logging.warning("W&B summary repaired for %s: %s", path, missing)
            return {"missing_before": missing, "repaired": True}
        except Exception as exc:  # noqa: BLE001
            logging.warning("W&B summary verification attempt %d failed: %r", attempt + 1, exc)
            time.sleep(wait)
    return {"missing_before": missing, "repaired": False, "failed": True}


def is_active() -> bool:
    return _ACTIVE
