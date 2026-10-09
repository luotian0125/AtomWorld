#!/usr/bin/env python3
"""Run the HCP Experiment-1 fixed-event main-table comparison."""

from __future__ import annotations

import argparse
import csv
from dataclasses import fields
import json
from pathlib import Path
import shutil
import time

import numpy as np
import torch

from atomworld.backends.hcp.models import AtomWorldModelConfig, MemoryRestoredAtomWorld
from atomworld.backends.hcp.training.exp1_metrics import (
    BASELINE_METHODS,
    PersistentPrecursorDefinition,
    evaluate_exp1_episode,
    summarize_exp1_episodes,
)
from atomworld.backends.hcp.training.factory import HCPEnvironmentConfig, make_hcp_policy_environment
from atomworld.backends.hcp.training.reward import HCPRewardConfig, HCPRewardModel


METHOD_LABELS = {
    "classical_hcp_lkmc": "Classical HCP-LKMC",
    "amc_superbasin_reference": "AMC/Superbasin HCP-LKMC",
    "rate_greedy_tabu": "Rate-greedy KMC + reverse tabu",
    "energy_greedy_tabu": "Energy-greedy KMC + reverse tabu",
    "snapshot_only_atomworld": "Snapshot-only AtomWorld",
    "atomworld_no_short_memory": "AtomWorld w/o short memory",
    "atomworld_no_long_memory": "AtomWorld w/o long memory",
    "atomworld": "AtomWorld",
}


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:3")
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        metavar="METHOD=PATH",
        help="repeat for the four AtomWorld variants",
    )
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--sample-interval", type=int, default=16)
    parser.add_argument("--eval-seed", type=int, default=310000)
    parser.add_argument("--seed-stride", type=int, default=1009)
    parser.add_argument("--policy-temperature", type=float, default=1.0)
    parser.add_argument("--min-attractive-pairs-per-h", type=float, default=0.40)
    parser.add_argument("--min-largest-cluster-fraction", type=float, default=0.14)
    parser.add_argument("--min-nonisolated-fraction", type=float, default=0.70)
    parser.add_argument("--persistence-samples", type=int, default=4)
    return parser


def _checkpoint_map(values):
    checkpoints = {}
    for item in values:
        if "=" not in item:
            raise ValueError(f"checkpoint must be METHOD=PATH, got {item!r}")
        method, path = item.split("=", 1)
        method = method.strip()
        if method not in METHOD_LABELS or not method.startswith(("atomworld", "snapshot")):
            raise ValueError(f"unsupported learned method {method!r}")
        if method in checkpoints:
            raise ValueError(f"duplicate checkpoint for {method!r}")
        checkpoints[method] = Path(path).expanduser().resolve()
    required = {
        "snapshot_only_atomworld",
        "atomworld_no_short_memory",
        "atomworld_no_long_memory",
        "atomworld",
    }
    missing = required - set(checkpoints)
    if missing:
        raise ValueError(f"missing checkpoints: {sorted(missing)}")
    return checkpoints


def _dataclass_kwargs(dataclass_type, payload):
    accepted = {field.name for field in fields(dataclass_type)}
    return {key: value for key, value in payload.items() if key in accepted}


def _load_model_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("material_system") != "hcp_zr_h":
        raise ValueError(f"{path} is not an HCP Zr-H checkpoint")
    metadata = payload.get("run_metadata", {})
    model_config = AtomWorldModelConfig(**metadata["model"])
    model = MemoryRestoredAtomWorld(model_config).to(device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    finite = all(
        bool(torch.isfinite(value).all()) for value in model.state_dict().values()
    )
    if not finite:
        raise ValueError(f"checkpoint {path} contains non-finite model weights")
    return payload, model_config, model


def _write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _format_value(value, digits=3):
    if value is None:
        return "--"
    return f"{float(value):.{digits}f}"


def _write_markdown(path, rows):
    lines = [
        "# HCP Experiment 1 main table (clockless stage)",
        "",
        "| Method | Persistent transport AUC | Persistent precursor AUC | Persistent precursor hit | AF_prec | TT recross | DynErr | StructErr | ClockErr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        acceleration = _format_value(row["precursor_acceleration_factor"], 2)
        if row["precursor_acceleration_is_lower_bound"] and acceleration != "--":
            acceleration = ">=" + acceleration
        if acceleration != "--":
            acceleration += "x"
        lines.append(
            "| {label} | {transport} | {precursor} | {hit} | {af} | {recross} | -- | -- | -- |".format(
                label=METHOD_LABELS.get(row["method"], row["method"]),
                transport=_format_value(
                    row["persistent_transport_auc_normalized"], 2
                ),
                precursor=_format_value(
                    row["persistent_precursor_auc_normalized"], 2
                ),
                hit=_format_value(100.0 * row["persistent_precursor_hit_rate"], 1)
                + "%",
                af=acceleration,
                recross=_format_value(100.0 * row["tt_recross_fraction"], 2)
                + "%",
            )
        )
    lines.extend(
        [
            "",
            "DynErr, StructErr, and ClockErr are intentionally blank until a separately calibrated physical-time evaluation is available.",
            "AMC/Superbasin is an exact explicit-microstep reference projection in Experiment 1; it is not yet an optimized absorbing-chain implementation and makes no wall-clock acceleration claim.",
        ]
    )
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.episodes <= 0 or args.steps <= 0 or args.sample_interval <= 0:
        raise ValueError("episodes, steps, and sample interval must be positive")
    if args.sample_interval > args.steps:
        raise ValueError("sample interval cannot exceed steps")
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = _checkpoint_map(args.checkpoint)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but CUDA is unavailable")

    loaded = {}
    reference_environment = None
    reference_reward = None
    checkpoint_manifest = {}
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    for method, path in checkpoints.items():
        payload, model_config, model = _load_model_checkpoint(path, device)
        metadata = payload["run_metadata"]
        environment = metadata["environment"]
        reward = metadata["reward"]
        if reference_environment is None:
            reference_environment = environment
            reference_reward = reward
        elif environment != reference_environment or reward != reference_reward:
            raise ValueError(
                f"checkpoint {method} does not share the full environment/reward contract"
            )
        frozen_path = checkpoint_dir / f"{method}.pt"
        if not frozen_path.exists():
            shutil.copy2(path, frozen_path)
        checkpoint_manifest[method] = {
            "source": str(path),
            "frozen_copy": str(frozen_path),
            "update": int(payload["update"]),
            "total_events": int(payload["total_events"]),
            "best_score": float(payload["best_score"]),
            "use_short_memory": bool(model_config.use_short_memory),
            "use_long_memory": bool(model_config.use_long_memory),
            "clock_calibrated": bool(payload.get("clock_calibrated", False)),
        }
        loaded[method] = (model_config, model)

    env_config = HCPEnvironmentConfig(
        **_dataclass_kwargs(HCPEnvironmentConfig, reference_environment)
    )
    reward_config = HCPRewardConfig(
        **_dataclass_kwargs(HCPRewardConfig, reference_reward)
    )
    reward_model = HCPRewardModel(reward_config)
    environment_factory = lambda seed: make_hcp_policy_environment(
        env_config, seed=seed
    )
    precursor_definition = PersistentPrecursorDefinition(
        min_attractive_pairs_per_h=args.min_attractive_pairs_per_h,
        min_largest_cluster_fraction=args.min_largest_cluster_fraction,
        min_nonisolated_fraction=args.min_nonisolated_fraction,
        persistence_samples=args.persistence_samples,
    )
    seeds = [args.eval_seed + args.seed_stride * index for index in range(args.episodes)]
    method_order = list(BASELINE_METHODS) + [
        "snapshot_only_atomworld",
        "atomworld_no_short_memory",
        "atomworld_no_long_memory",
        "atomworld",
    ]
    episodes = []
    started = time.time()
    with torch.no_grad():
        for method in method_order:
            model_config, model = loaded.get(method, (None, None))
            for seed in seeds:
                episode = evaluate_exp1_episode(
                    environment_factory=environment_factory,
                    reward_model=reward_model,
                    seed=seed,
                    steps=args.steps,
                    sample_interval=args.sample_interval,
                    precursor_definition=precursor_definition,
                    method=method,
                    model=model,
                    model_config=model_config,
                    device=device,
                    policy_temperature=args.policy_temperature,
                )
                episodes.append(episode)
                print(
                    json.dumps(
                        {
                            "phase": "exp1_eval",
                            "method": method,
                            "seed": seed,
                            "transport_auc_A2": episode["transport_auc_A2"],
                            "precursor_auc": episode["precursor_auc"],
                            "hit": episode["persistent_precursor_hit"],
                            "tt_recross_fraction": episode["tt_recross_fraction"],
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
    rows = summarize_exp1_episodes(
        episodes, classical_method="classical_hcp_lkmc", steps=args.steps
    )
    rows_by_method = {row["method"]: row for row in rows}
    rows = [rows_by_method[method] for method in method_order]
    payload = {
        "schema_version": 1,
        "experiment": "hcp_exp1_fixed_event_main_table",
        "claim_boundary": (
            "fixed microscopic-event budget; persistent dissolved-H transport and "
            "precursor surrogate only; no physical-time or hydride-nucleation claim"
        ),
        "clock_calibrated": False,
        "environment": env_config.metadata(),
        "reward_training_metadata": reward_config.metadata(),
        "precursor_definition": precursor_definition.metadata(),
        "evaluation": {
            "episodes": args.episodes,
            "steps": args.steps,
            "sample_interval": args.sample_interval,
            "seeds": seeds,
            "policy_deterministic": True,
            "policy_temperature": args.policy_temperature,
            "wall_seconds": time.time() - started,
        },
        "amc_superbasin": {
            "implementation": "exact_explicit_microstep_reference_projection",
            "publication_wall_clock_claim_ready": False,
            "micro_event_budget_charged": True,
        },
        "checkpoints": checkpoint_manifest,
        "main_table": rows,
        "episodes": episodes,
    }
    (output_dir / "exp1_results.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    _write_csv(output_dir / "exp1_main_table.csv", rows)
    _write_markdown(output_dir / "exp1_main_table.md", rows)
    print(
        json.dumps(
            {
                "status": "complete",
                "output_dir": str(output_dir),
                "main_table_csv": str(output_dir / "exp1_main_table.csv"),
                "main_table_markdown": str(output_dir / "exp1_main_table.md"),
                "clock_calibrated": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

