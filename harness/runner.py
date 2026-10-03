"""Walk-forward runner (owner: Shriker).

log_run is a shared starter helper used by everyone from day 1; the rest of
this file is built in step S9.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "results", "logs", "runs.jsonl")


def log_run(model_name: str, fold_id: int, seed: int, config: dict, metrics: dict,
            author: str, path: str = LOG_PATH) -> dict:
    """Append one experiment record to results/logs/runs.jsonl and return it."""
    record = {
        "time": datetime.now(timezone.utc).isoformat(),
        "author": author,
        "model": model_name,
        "fold_id": fold_id,
        "seed": seed,
        "config": config,
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    return record


# TODO (S9): run(models, folds, seeds) — fit, score, calibrate thresholds, compute metrics, log.
