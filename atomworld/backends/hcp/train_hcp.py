#!/usr/bin/env python3
"""End-to-end clockless PPO training for AtomWorld-HCP.

This entry point intentionally does not construct or calibrate a policy clock.
Checkpoints are valid for event selection, not physical-time claims.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from atomworld.backends.hcp.learning import HCPAtomWorldBinding
from atomworld.backends.hcp.models import AtomWorldModelConfig
from atomworld.backends.hcp.training.checkpoint import (
    append_jsonl,
    load_checkpoint,
    save_checkpoint,
)
from atomworld.backends.hcp.training.evaluation import EVALUATION_METHODS, evaluate_methods
from atomworld.backends.hcp.training.factory import HCPEnvironmentConfig, make_hcp_policy_environment
from atomworld.backends.hcp.training.ppo import ClocklessRolloutCollector, HCPPPOUpdater
from atomworld.backends.hcp.training.reward import HCPRewardConfig, HCPRewardModel


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train AtomWorld-HCP with a fixed-event, clockless PPO objective."
    )
    run = parser.add_argument_group("run")
    run.add_argument("--run-name", default="hcp_clockless_main")
    run.add_argument("--output-root", default="results/hcp_training")
    run.add_argument("--device", default="auto")
    run.add_argument("--seed", type=int, default=20260903)
    run.add_argument("--resume", default="auto", help="auto, none, or checkpoint path")
    run.add_argument("--updates", type=int, default=1000)
    run.add_argument("--log-interval", type=int, default=1)
    run.add_argument("--checkpoint-interval", type=int, default=10)
    run.add_argument("--eval-interval", type=int, default=25)
    run.add_argument("--eval-at-start", action="store_true")
    run.add_argument(
        "--best-metric",
        choices=("mean_return", "mean_objective_gain", "mean_final_objective"),
        default="mean_return",
    )

    physics = parser.add_argument_group("HCP physics")
    physics.add_argument("--size", nargs=3, type=int, default=(4, 4, 4))
    physics.add_argument("--n-hydrogen", type=int, default=8)
    physics.add_argument("--temperature-K", type=float, default=600.0)
    physics.add_argument("--parameter-file", default="")
    physics.add_argument(
        "--local-energy-file",
        default=str(Path(__file__).resolve().parent / "configs/hcp/zr_h_pair_screening_andolina2022.json"),
    )
    physics.add_argument("--pair-energy-scale", type=float, default=1.0)
    physics.add_argument("--initial-site-type", choices=("T", "O", "ANY"), default="T")
    physics.add_argument("--maximum-h-atomic-fraction", type=float, default=0.077)
    physics.add_argument("--allow-outside-source-range", action="store_true")

    observation = parser.add_argument_group("observation and memory")
    observation.add_argument("--history-len", type=int, default=5)
    observation.add_argument("--dense-max-neighbors", type=int, default=48)
    observation.add_argument("--sparse-max-tokens", type=int, default=24)
    observation.add_argument("--hidden-dim", type=int, default=128)
    observation.add_argument("--keyframe-dim", type=int, default=128)
    observation.add_argument("--memory-dim", type=int, default=128)
    observation.add_argument("--attention-heads", type=int, default=4)
    observation.add_argument("--dense-layers", type=int, default=2)
    observation.add_argument("--sparse-layers", type=int, default=1)
    observation.add_argument("--history-layers", type=int, default=1)
    observation.add_argument("--dropout", type=float, default=0.0)
    observation.add_argument("--disable-short-memory", action="store_true")
    observation.add_argument("--disable-long-memory", action="store_true")

    ppo = parser.add_argument_group("clockless PPO")
    ppo.add_argument("--rollout-steps", type=int, default=128)
    ppo.add_argument("--episode-length", type=int, default=512)
    ppo.add_argument("--ppo-epochs", type=int, default=4)
    ppo.add_argument("--minibatch-size", type=int, default=64)
    ppo.add_argument("--actor-lr", type=float, default=3.0e-4)
    ppo.add_argument("--critic-lr", type=float, default=3.0e-4)
    ppo.add_argument("--weight-decay", type=float, default=0.0)
    ppo.add_argument("--gamma", type=float, default=0.99)
    ppo.add_argument("--gae-lambda", type=float, default=0.95)
    ppo.add_argument("--clip-ratio", type=float, default=0.2)
    ppo.add_argument("--value-coefficient", type=float, default=0.5)
    ppo.add_argument("--entropy-coefficient", type=float, default=0.01)
    ppo.add_argument("--kmc-kl-coefficient", type=float, default=0.01)
    ppo.add_argument("--max-grad-norm", type=float, default=1.0)
    ppo.add_argument("--policy-temperature", type=float, default=1.0)
    ppo.add_argument("--linear-lr-decay", action="store_true")

    reward = parser.add_argument_group("degradation-surrogate reward")
    reward.add_argument("--reward-energy", type=float, default=1.0)
    reward.add_argument("--reward-attractive-pair", type=float, default=0.25)
    reward.add_argument("--reward-largest-cluster", type=float, default=1.0)
    reward.add_argument("--reward-nonisolated", type=float, default=0.5)
    reward.add_argument("--reward-trapped", type=float, default=1.0)
    reward.add_argument("--immediate-reverse-penalty", type=float, default=0.02)
    reward.add_argument("--keyframe-bonus", type=float, default=0.0)

    evaluation = parser.add_argument_group("periodic matched-seed comparison")
    evaluation.add_argument("--eval-episodes", type=int, default=4)
    evaluation.add_argument("--eval-steps", type=int, default=256)
    evaluation.add_argument("--eval-seed", type=int, default=91000)
    evaluation.add_argument(
        "--eval-methods",
        nargs="+",
        choices=EVALUATION_METHODS,
        default=EVALUATION_METHODS,
    )
    evaluation.add_argument("--eval-policy-temperature", type=float, default=1.0)
    return parser


def _positive_arguments(args):
    names = (
        "updates",
        "log_interval",
        "checkpoint_interval",
        "eval_interval",
        "rollout_steps",
        "episode_length",
        "ppo_epochs",
        "minibatch_size",
        "eval_episodes",
        "eval_steps",
    )
    for name in names:
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if "atomworld" not in args.eval_methods:
        raise ValueError(
            "--eval-methods must include atomworld for best-checkpoint selection"
        )


def resolve_device(requested):
    requested = str(requested).strip().lower()
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def environment_config_from_args(args):
    return HCPEnvironmentConfig(
        size=tuple(args.size),
        n_hydrogen=args.n_hydrogen,
        temperature_K=args.temperature_K,
        parameter_file=args.parameter_file,
        local_energy_file=args.local_energy_file,
        pair_energy_scale=args.pair_energy_scale,
        initial_site_type=args.initial_site_type,
        history_len=args.history_len,
        dense_max_neighbors=args.dense_max_neighbors,
        sparse_max_tokens=args.sparse_max_tokens,
        maximum_h_atomic_fraction=args.maximum_h_atomic_fraction,
        allow_outside_source_range=args.allow_outside_source_range,
    )


def model_config_from_args(args):
    return AtomWorldModelConfig(
        hidden_dim=args.hidden_dim,
        keyframe_dim=args.keyframe_dim,
        memory_dim=args.memory_dim,
        event_feature_dim=18,
        history_feature_dim=12,
        species_count=3,
        defect_type_count=4,
        attention_heads=args.attention_heads,
        dense_layers=args.dense_layers,
        sparse_layers=args.sparse_layers,
        history_layers=args.history_layers,
        dropout=args.dropout,
        mesoscale_dim=9,
        use_short_memory=not args.disable_short_memory,
        use_long_memory=not args.disable_long_memory,
    )


def reward_config_from_args(args):
    return HCPRewardConfig(
        energy_drop_weight_per_eV=args.reward_energy,
        attractive_pair_weight=args.reward_attractive_pair,
        largest_cluster_weight=args.reward_largest_cluster,
        nonisolated_fraction_weight=args.reward_nonisolated,
        trapped_fraction_weight=args.reward_trapped,
        immediate_reverse_penalty=args.immediate_reverse_penalty,
        keyframe_bonus=args.keyframe_bonus,
    )


def json_safe_args(args):
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in vars(args).items()
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    _positive_arguments(args)
    device = resolve_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    env_config = environment_config_from_args(args)
    model_config = model_config_from_args(args)
    reward_config = reward_config_from_args(args)
    reward_model = HCPRewardModel(reward_config)
    environment_factory = lambda seed: make_hcp_policy_environment(
        env_config, seed=seed
    )
    initial_env = environment_factory(args.seed)
    publication_ready = bool(
        initial_env.kmc.energy_model.provenance.get("publication_ready", False)
    )
    run_dir = Path(args.output_root).expanduser().resolve() / args.run_name
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    run_metadata = {
        "schema_version": 1,
        "material_system": "hcp_zr_h",
        "host_phase": "alpha_zr",
        "clock_calibrated": False,
        "clock_owner": "none_clockless_training",
        "claim_boundary": (
            "fixed-event dissolved-H precursor surrogate; no physical-time or "
            "hydride-nucleation claim"
        ),
        "publication_ready_parameters": publication_ready,
        "arguments": json_safe_args(args),
        "environment": env_config.metadata(),
        "model": asdict(model_config),
        "reward": reward_config.metadata(),
        "kinetic_parameters": initial_env.kmc.parameters.metadata(),
        "local_energy_model": initial_env.kmc.energy_model.metadata(),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    (run_dir / "run_config.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True), encoding="utf-8"
    )
    if not publication_ready:
        print(
            "WARNING: local-energy parameters are screening-grade; checkpoints are "
            "engineering artifacts, not publication-ready Zr-H predictions.",
            flush=True,
        )

    binding = HCPAtomWorldBinding(initial_env, config=model_config, device=device)
    actor_parameters = (
        list(binding.model.keyframes.parameters())
        + list(binding.model.memory.parameters())
        + list(binding.model.actor.parameters())
    )
    critic_parameters = list(binding.model.critic.parameters())
    actor_optimizer = torch.optim.AdamW(
        actor_parameters, lr=args.actor_lr, weight_decay=args.weight_decay
    )
    critic_optimizer = torch.optim.AdamW(
        critic_parameters, lr=args.critic_lr, weight_decay=args.weight_decay
    )
    collector = ClocklessRolloutCollector(
        binding,
        environment_factory=environment_factory,
        reward_model=reward_model,
        rollout_steps=args.rollout_steps,
        episode_length=args.episode_length,
        seed=args.seed,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        policy_temperature=args.policy_temperature,
    )
    updater = HCPPPOUpdater(
        binding.model,
        actor_optimizer,
        critic_optimizer,
        device=device,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        clip_ratio=args.clip_ratio,
        value_coefficient=args.value_coefficient,
        entropy_coefficient=args.entropy_coefficient,
        kmc_kl_coefficient=args.kmc_kl_coefficient,
        max_grad_norm=args.max_grad_norm,
    )

    latest_path = checkpoint_dir / "latest.pt"
    resume_path = None
    if args.resume.lower() == "auto" and latest_path.exists():
        resume_path = latest_path
    elif args.resume.lower() not in {"auto", "none"}:
        resume_path = Path(args.resume).expanduser().resolve()
    start_update = 1
    best_score = -math.inf
    if resume_path is not None:
        restored = load_checkpoint(
            resume_path,
            model=binding.model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            device=device,
        )
        previous_metadata = restored.get("run_metadata", {})
        for section, current in (
            ("environment", env_config.metadata()),
            ("model", asdict(model_config)),
            ("reward", reward_config.metadata()),
        ):
            if previous_metadata.get(section) != current:
                raise ValueError(
                    f"resume checkpoint {section} configuration differs from the "
                    "current run; use a new RUN_NAME or RESUME=none"
                )
        start_update = int(restored["update"]) + 1
        best_score = float(restored.get("best_score", -math.inf))
        collector.total_events = int(restored.get("total_events", 0))
        print(
            f"resumed {resume_path} at update={start_update} "
            f"events={collector.total_events}",
            flush=True,
        )
        if start_update > args.updates:
            print(
                json.dumps(
                    {
                        "status": "already_complete",
                        "checkpoint_update": int(restored["update"]),
                        "requested_updates": args.updates,
                        "checkpoint": str(resume_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            return

    eval_seeds = [args.eval_seed + 1009 * index for index in range(args.eval_episodes)]
    train_history = run_dir / "train_history.jsonl"
    eval_history = run_dir / "eval_history.jsonl"

    def save(tag_path, update):
        return save_checkpoint(
            tag_path,
            model=binding.model,
            actor_optimizer=actor_optimizer,
            critic_optimizer=critic_optimizer,
            update=update,
            total_events=collector.total_events,
            best_score=best_score,
            run_metadata=run_metadata,
        )

    def evaluate_and_maybe_save(update):
        nonlocal best_score
        started = time.time()
        result = evaluate_methods(
            binding.model,
            environment_factory=environment_factory,
            reward_model=reward_model,
            seeds=eval_seeds,
            steps=args.eval_steps,
            device=device,
            model_config=model_config,
            policy_temperature=args.eval_policy_temperature,
            methods=args.eval_methods,
        )
        result["update"] = int(update)
        result["total_events"] = int(collector.total_events)
        result["wall_seconds"] = time.time() - started
        policy_summary = result["methods"]["atomworld"]["summary"]
        score = float(policy_summary[args.best_metric])
        result["best_metric"] = args.best_metric
        result["selection_score"] = score
        result["previous_best_score"] = best_score
        improved = score > best_score
        result["new_best"] = improved
        if improved:
            best_score = score
            save(checkpoint_dir / "best.pt", update)
            save(
                checkpoint_dir / f"best_update_{update:06d}_score_{score:+.6f}.pt",
                update,
            )
        result["best_score"] = best_score
        append_jsonl(eval_history, result)
        print(
            json.dumps(
                {
                    "phase": "eval",
                    "update": update,
                    "score": score,
                    "best_score": best_score,
                    "new_best": improved,
                    "atomworld": policy_summary,
                    "comparisons": result.get("atomworld_comparisons", {}),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    if args.eval_at_start and start_update == 1:
        evaluate_and_maybe_save(0)

    for update in range(start_update, args.updates + 1):
        wall_start = time.time()
        if args.linear_lr_decay:
            fraction = 1.0 - (update - 1) / max(args.updates, 1)
            actor_optimizer.param_groups[0]["lr"] = args.actor_lr * fraction
            critic_optimizer.param_groups[0]["lr"] = args.critic_lr * fraction
        rollout = collector.collect()
        update_metrics = updater.update(rollout)
        row = {
            "phase": "train",
            "update": update,
            "wall_seconds": time.time() - wall_start,
            "actor_lr": actor_optimizer.param_groups[0]["lr"],
            "critic_lr": critic_optimizer.param_groups[0]["lr"],
            **rollout.metrics,
            **update_metrics,
        }
        append_jsonl(train_history, row)
        if update % args.log_interval == 0:
            print(json.dumps(row, sort_keys=True), flush=True)
        if update % args.checkpoint_interval == 0:
            save(latest_path, update)
            save(checkpoint_dir / f"update_{update:06d}.pt", update)
        if update % args.eval_interval == 0:
            evaluate_and_maybe_save(update)
            save(latest_path, update)

    save(latest_path, args.updates)
    if args.updates % args.eval_interval != 0:
        evaluate_and_maybe_save(args.updates)
        save(latest_path, args.updates)
    print(
        json.dumps(
            {
                "status": "complete",
                "updates": args.updates,
                "total_events": collector.total_events,
                "best_score": best_score,
                "checkpoint": str(latest_path),
                "clock_calibrated": False,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
