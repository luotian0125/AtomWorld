#!/usr/bin/env python
import os
from pathlib import Path
import sys
import socket

import numpy as np
import torch

from atomworld.backends.fcc.envs.kmc_env import KMCEnvWrap as KMCEnv
from atomworld.backends.fcc.config.parser import get_config


def make_train_env(all_args):
    env = KMCEnv(all_args)
    env.seed(all_args.seed)
    return env


def make_eval_env(all_args):
    env = KMCEnv(all_args)
    env.seed(all_args.seed * 500000)
    return env


def configure_algorithm_flags(all_args):
    if all_args.algorithm_name == "rmappo":
        print("Using rmappo: enabling recurrent policy.")
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        print("Using mappo: disabling recurrent policy.")
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        print("Using ippo: disabling centralized critic.")
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError(f"Unsupported algorithm_name={all_args.algorithm_name}")


def resolve_device(all_args):
    use_cuda = bool(all_args.cuda) and torch.cuda.is_available()
    if use_cuda:
        preferred = str(all_args.device).strip() if getattr(all_args, "device", None) else "cuda:0"
        if not preferred.startswith("cuda"):
            preferred = "cuda:0"
        device = torch.device(preferred)
        print(f"Using CUDA device: {device}")
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        device = torch.device("cpu")
        print("Using CPU device.")
    return device


def seed_everything(seed, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def build_run_dir(all_args):
    run_dir = Path(__file__).resolve().parent / "results" / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def maybe_init_wandb(all_args, run_dir):
    if not all_args.use_wandb:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        curr_run = 'run1' if len(exst_run_nums) == 0 else f"run{max(exst_run_nums) + 1}"
        local_run_dir = run_dir / curr_run
        local_run_dir.mkdir(parents=True, exist_ok=True)
        return local_run_dir

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError("--use_wandb requires the optional 'wandb' package") from exc

    wandb_kwargs = {
        "config": vars(all_args),
        "project": str(getattr(all_args, "wandb_project", "AtomWorld-FCC") or "AtomWorld-FCC"),
        "notes": socket.gethostname(),
        "name": all_args.train_name,
        "group": "AtomWorld-FCC",
        "dir": str(run_dir),
        "job_type": "evaluation",
        "reinit": True,
    }
    wandb_entity = str(getattr(all_args, "wandb_entity", "") or "").strip()
    if wandb_entity:
        wandb_kwargs["entity"] = wandb_entity
    wandb.init(**wandb_kwargs)
    return run_dir


def main(args):
    parser = get_config()
    all_args = parser.parse_args(args)
    all_args.use_eval = True
    configure_algorithm_flags(all_args)
    device = resolve_device(all_args)
    seed_everything(all_args.seed, device)

    run_dir = build_run_dir(all_args)
    run_dir = maybe_init_wandb(all_args, run_dir)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args)

    from atomworld.backends.fcc.runner.kmc_runner import KMCRunner as Runner

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
    }

    runner = Runner(all_args, config)

    if all_args.model_dir is not None:
        runner.restore4eval()
    else:
        print("Warning: model_dir is None, evaluation will use randomly initialized actor weights.")

    print("Starting deterministic evaluation...")
    runner.eval(0, 0)


if __name__ == "__main__":
    main(sys.argv[1:])
