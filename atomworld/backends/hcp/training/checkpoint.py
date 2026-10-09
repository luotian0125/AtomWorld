"""Atomic checkpoint and JSON-lines helpers for HCP training."""

from __future__ import annotations

import json
import os
from pathlib import Path

import torch


def append_jsonl(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_checkpoint(
    path,
    *,
    model,
    actor_optimizer,
    critic_optimizer,
    update,
    total_events,
    best_score,
    run_metadata,
):
    payload = {
        "schema_version": 1,
        "material_system": "hcp_zr_h",
        "clock_calibrated": False,
        "clock_owner": "none_clockless_training",
        "update": int(update),
        "total_events": int(total_events),
        "best_score": float(best_score),
        "model_state_dict": model.state_dict(),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        ),
        "run_metadata": run_metadata,
    }
    atomic_torch_save(payload, path)
    return payload


def load_checkpoint(path, *, model, actor_optimizer, critic_optimizer, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("material_system") != "hcp_zr_h":
        raise ValueError("checkpoint is not an HCP Zr-H training checkpoint")
    if payload.get("clock_calibrated") is not False:
        raise ValueError("unexpected clock calibration flag in clockless trainer")
    model.load_state_dict(payload["model_state_dict"], strict=True)
    actor_optimizer.load_state_dict(payload["actor_optimizer_state_dict"])
    critic_optimizer.load_state_dict(payload["critic_optimizer_state_dict"])
    torch.set_rng_state(payload["torch_rng_state"].cpu())
    if torch.cuda.is_available() and payload.get("cuda_rng_state_all") is not None:
        torch.cuda.set_rng_state_all(
            [state.cpu() for state in payload["cuda_rng_state_all"]]
        )
    return payload
