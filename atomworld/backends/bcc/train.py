#!/usr/bin/env python
from pathlib import Path
import sys
import socket

import numpy as np
import torch

try:
    import wandb
except ImportError:
    wandb = None

from atomworld.backends.bcc.envs.kmc_env import KMCEnvWrap as KMCEnv
from atomworld.backends.bcc.config.parser import get_config


def make_train_env(all_args):
    env = KMCEnv(all_args)
    env.seed(all_args.seed)
    return env


def make_eval_env(all_args):
    env = KMCEnv(all_args)
    env.seed(all_args.seed * 50000)
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
    from atomworld.paths import training_root
    run_dir = training_root('bcc') / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def maybe_init_wandb(all_args, run_dir):
    if not all_args.use_wandb:
        exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if str(folder.name).startswith('run')]
        curr_run = 'run1' if len(exst_run_nums) == 0 else f"run{max(exst_run_nums) + 1}"
        local_run_dir = run_dir / curr_run
        local_run_dir.mkdir(parents=True, exist_ok=True)
        return local_run_dir

    if wandb is None:
        raise RuntimeError("Weights & Biases was requested, but wandb is not installed. Use --no_wandb or install wandb.")

    wandb_kwargs = {
        "config": vars(all_args),
        "project": str(getattr(all_args, "wandb_project", "AtomWorld-Mem") or "AtomWorld-Mem"),
        "notes": socket.gethostname(),
        "name": all_args.train_name,
        "group": "AtomWorld-Mem",
        "dir": str(run_dir),
        "job_type": "training",
        "reinit": True,
    }
    wandb_entity = str(getattr(all_args, "wandb_entity", "") or "").strip()
    if wandb_entity:
        wandb_kwargs["entity"] = wandb_entity
    wandb.init(**wandb_kwargs)
    return run_dir


def main(args):
    parser = get_config()
    all_args = parser.parse_known_args(args)[0]
    configure_algorithm_flags(all_args)
    device = resolve_device(all_args)
    seed_everything(all_args.seed, device)

    run_dir = build_run_dir(all_args)
    run_dir = maybe_init_wandb(all_args, run_dir)

    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None

    from atomworld.backends.bcc.runners.kmc import KMCRunner as Runner

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "device": device,
        "run_dir": run_dir,
    }

    runner = Runner(all_args, config)
    runner.run()


if __name__ == "__main__":
    main(sys.argv[1:])
