#!/usr/bin/env python
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from atomworld.backends.fcc.config.parser import get_config
from atomworld.backends.fcc.envs.kmc_env import KMCEnvWrap as KMCEnv
from atomworld.backends.fcc.PPO.rMAPPOPolicy import R_MAPPOPolicy

try:
    from scipy.stats import ttest_rel
except Exception:
    ttest_rel = None


ALL_MODES = [
    "traditional",
    "random_uniform",
    "rate_greedy",
    "energy_greedy",
    "rate_energy",
    "rl4kmc_stochastic",
    "rl4kmc_snapshot",
    "rl4kmc_no_sparse",
    "rl4kmc_greedy",
]
DEFAULT_MODES = ["traditional", "energy_greedy", "rate_energy", "rl4kmc_snapshot", "rl4kmc_no_sparse", "rl4kmc_stochastic"]
MAIN_MODES = ["traditional", "rl4kmc_stochastic"]
RL_MODES = {"rl4kmc_stochastic", "rl4kmc_snapshot", "rl4kmc_no_sparse", "rl4kmc_greedy"}
HEURISTIC_MODES = {"traditional", "random_uniform", "rate_greedy", "energy_greedy", "rate_energy"}
STRUCTURE_TIME_METRICS = [
    ("cluster_count", "Cu Cluster Count", "cluster_count_vs_time_main.png"),
    ("stable_cluster_count", "Stable Cu Cluster Count", "stable_cluster_count_vs_time_main.png"),
    ("aggregation_degree", "Aggregation Degree", "aggregation_degree_vs_time_main.png"),
    ("advancement_factor", "Advancement Factor zeta", "advancement_factor_vs_time_main.png"),
    ("largest_cluster_size", "Largest Cu Cluster Size", "largest_cluster_size_vs_time_main.png"),
    ("vacancy_contact_ratio", "Vacancy-Cluster Contact Ratio", "vacancy_contact_ratio_vs_time_main.png"),
    ("contacted_vacancy_ratio", "Ever-Contacted Vacancy Ratio", "contacted_vacancy_ratio_vs_time_main.png"),
    ("first_cluster_contact_time", "First Cluster-Contact Time", "first_cluster_contact_time_vs_time_main.png"),
    ("mean_first_arrival_time", "Mean First-Arrival Time", "mean_first_arrival_time_vs_time_main.png"),
]


def add_compare_args(parser):
    parser.add_argument(
        "--bench_step",
        type=int,
        default=4000,
        help="Run each benchmark rollout for exactly this many steps unless terminated earlier.",
    )
    parser.add_argument(
        "--energy_series_interval_steps",
        type=int,
        default=50,
        help="Save one CSV row every N steps. Summary metrics still use full per-step trajectories.",
    )
    parser.add_argument(
        "--structure_series_interval_steps",
        type=int,
        default=0,
        help="Save structure metrics every N steps. If <= 0, reuse energy_series_interval_steps.",
    )
    parser.add_argument(
        "--compare_seeds",
        type=int,
        nargs="+",
        default=list(range(1, 11)),
        help="Benchmark seeds. Each seed reuses the exact same initial state snapshot across all modes.",
    )
    parser.add_argument(
        "--compare_modes",
        type=str,
        nargs="+",
        default=DEFAULT_MODES,
        choices=ALL_MODES,
        help="Which rollout modes to compare. Includes KMC, simple heuristic baselines, and RL policy modes.",
    )
    parser.add_argument(
        "--compare_output_dir",
        type=str,
        default="results/compare_step_energy",
        help="Directory used to save CSVs, plots and summary files.",
    )
    parser.add_argument(
        "--plot_title",
        type=str,
        default="Step-Energy Comparison",
        help="Title prefix used in saved plots.",
    )
    parser.add_argument(
        "--plot_band",
        type=str,
        default="stderr",
        choices=["std", "stderr", "none"],
        help="Shaded band type used in plots.",
    )
    parser.add_argument(
        "--structure_plot_band",
        type=str,
        default="stderr",
        choices=["std", "stderr", "none"],
        help="Shaded band type used in kmc-time structure plots.",
    )
    parser.add_argument(
        "--structure_smooth_window",
        type=int,
        default=1,
        help="Centered moving-average window applied only to plotted structure-vs-time mean curves.",
    )
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--bootstrap_samples",
        type=int,
        default=5000,
        help="Number of bootstrap resamples used for paired confidence intervals.",
    )
    parser.add_argument(
        "--target_energy_drops",
        type=float,
        nargs="*",
        default=[10.0, 15.0, 18.0],
        help="Target energy-drop thresholds used for steps-to-target analysis.",
    )
    parser.add_argument(
        "--time_grid_points",
        type=int,
        default=200,
        help="Number of common kmc-time grid points used when plotting time-based structure curves.",
    )
    parser.add_argument(
        "--aligned_time_reference_mode",
        type=str,
        default="traditional",
        choices=ALL_MODES + ["auto"],
        help="Reference mode whose kmc_time is reused at the same step when generating forced-aligned time plots.",
    )
    parser.add_argument(
        "--rate_energy_beta",
        type=float,
        default=1.0,
        help="Beta for the Rate x Energy baseline: score = log(rate) + beta * max(energy_gain, 0).",
    )
    parser.add_argument(
        "--greedy_tie_atol",
        type=float,
        default=1e-12,
        help="Absolute tolerance used when randomly breaking ties in greedy baselines.",
    )
    parser.add_argument(
        "--energy_greedy_zero_gain_fallback",
        type=str,
        default="traditional",
        choices=["traditional", "none"],
        help=(
            "Energy-Greedy fallback when no legal event has positive immediate energy gain. "
            "traditional avoids deterministic zero-gain traps while preserving KMC physics."
        ),
    )
    parser.add_argument(
        "--cluster_min_size",
        type=int,
        default=4,
        help="Minimum Cu cluster size counted as a stable precipitate in structure metrics.",
    )
    parser.add_argument(
        "--metric_cluster_min_size",
        type=int,
        default=2,
        help="Minimum Cu cluster size used for time-based aggregation/contact metrics.",
    )
    return parser


def resolve_device(all_args) -> torch.device:
    use_cuda = bool(all_args.cuda) and torch.cuda.is_available()
    if not use_cuda:
        return torch.device("cpu")

    preferred = str(getattr(all_args, "device", "cuda:0") or "cuda:0").strip().lower()
    if preferred.isdigit():
        preferred = f"cuda:{preferred}"
    if preferred == "cuda":
        preferred = "cuda:0"
    if preferred == "cpu":
        return torch.device("cpu")
    if not preferred.startswith("cuda:"):
        raise ValueError(f"Unsupported --device value: {preferred}. Use cpu or cuda:<index>.")

    try:
        index = int(preferred.split(":", 1)[1])
    except Exception as exc:
        raise ValueError(f"Invalid CUDA device string: {preferred}") from exc

    n_cuda = torch.cuda.device_count()
    if index < 0 or index >= n_cuda:
        raise ValueError(f"Requested CUDA device {preferred}, but only {n_cuda} CUDA device(s) are available.")

    torch.cuda.set_device(index)
    if bool(getattr(all_args, "cuda_deterministic", False)):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    return torch.device(preferred)


def configure_runtime_args(all_args, device: torch.device) -> None:
    all_args.device = str(device)
    all_args.cuda = device.type == "cuda"
    all_args.use_wandb = False


def seed_everything(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def instantiate_env(all_args) -> KMCEnv:
    return KMCEnv(all_args)


def create_seed_snapshot(all_args, seed: int, device: torch.device) -> Dict:
    seed_everything(seed, device)
    env = instantiate_env(all_args)
    env.seed(seed)
    env.reset()
    return env.get_state_dict()


def restore_env_from_snapshot(all_args, seed: int, snapshot: Dict, device: torch.device) -> Tuple[KMCEnv, np.ndarray]:
    env = instantiate_env(all_args)
    env.set_state_dict(snapshot)
    # Re-seed after restoration so every mode starts from the same RNG state.
    seed_everything(seed, device)
    obs = env.env.get_combined_obs().astype(np.float32)
    return env, obs


def load_policy_if_needed(all_args, action_space, device: torch.device, compare_modes: Sequence[str]) -> Optional[R_MAPPOPolicy]:
    need_policy = any(mode in RL_MODES for mode in compare_modes)
    if not need_policy:
        return None

    if not all_args.model_dir:
        raise ValueError("--model_dir is required when compare_modes includes rl4kmc modes.")

    actor_path = Path(all_args.model_dir) / "actor_agent.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")

    policy = R_MAPPOPolicy(all_args, action_space, device=device)
    state_dict = torch.load(actor_path, map_location=device)
    msg = policy.actor.load_state_dict(state_dict, strict=False)
    if msg.missing_keys or msg.unexpected_keys:
        print(f"[Compare] actor partial load. missing={list(msg.missing_keys)} unexpected={list(msg.unexpected_keys)}")
    policy.actor.eval()
    policy.critic.eval()
    return policy


@torch.no_grad()
def sample_rl_action(
    policy: R_MAPPOPolicy,
    obs,
    rnn_states,
    masks,
    base_rates,
    event_features,
    sparse_defect_tokens,
    sparse_defect_masks,
    deterministic: bool,
):
    action, new_rnn_states, hazard_summary = policy.act(
        obs,
        rnn_states,
        masks,
        base_rates,
        event_features,
        sparse_defect_tokens=sparse_defect_tokens,
        sparse_defect_masks=sparse_defect_masks,
        deterministic=deterministic,
        return_hazard_summary=True,
    )
    return action.detach().cpu().numpy(), new_rnn_states.detach().cpu().numpy(), hazard_summary


def sample_traditional_action(base_rates: np.ndarray) -> Tuple[np.ndarray, bool]:
    base_rates = np.asarray(base_rates, dtype=np.float64)
    flat = base_rates.reshape(-1)
    legal = flat > 0.0
    if not np.any(legal):
        return np.array([[0]], dtype=np.int64), False

    legal_ids = np.where(legal)[0]
    probs = flat[legal_ids]
    probs = probs / np.maximum(probs.sum(), 1e-12)
    chosen = np.random.choice(legal_ids, p=probs)
    return np.array([[int(chosen)]], dtype=np.int64), True


def random_tie_argmax(ids: np.ndarray, values: np.ndarray, atol: float = 1e-12) -> int:
    ids = np.asarray(ids, dtype=np.int64)
    values = np.asarray(values, dtype=np.float64)
    if ids.size == 0:
        return 0
    best_value = float(np.max(values))
    tied = ids[np.isclose(values, best_value, rtol=0.0, atol=float(atol))]
    if tied.size == 0:
        tied = ids[values == best_value]
    return int(np.random.choice(tied))


def sample_heuristic_action(
    mode: str,
    base_rates: np.ndarray,
    event_features: np.ndarray,
    rate_energy_beta: float = 1.0,
    greedy_tie_atol: float = 1e-12,
    energy_greedy_zero_gain_fallback: str = "traditional",
) -> Tuple[np.ndarray, bool]:
    base_rates = np.asarray(base_rates, dtype=np.float64)
    flat_rates = base_rates.reshape(-1)
    legal = flat_rates > 0.0
    if not np.any(legal):
        return np.array([[0]], dtype=np.int64), False

    legal_ids = np.where(legal)[0]
    if mode == "traditional":
        probs = flat_rates[legal_ids]
        probs = probs / np.maximum(probs.sum(), 1e-12)
        chosen = np.random.choice(legal_ids, p=probs)
    elif mode == "random_uniform":
        chosen = np.random.choice(legal_ids)
    elif mode == "rate_greedy":
        legal_rates = flat_rates[legal_ids]
        chosen = random_tie_argmax(legal_ids, legal_rates, atol=float(greedy_tie_atol))
    elif mode == "energy_greedy":
        features = np.asarray(event_features, dtype=np.float64)
        if features.shape[:2] != base_rates.shape or features.shape[-1] <= 4:
            raise ValueError("energy_greedy requires event_features[..., 4] energy-gain values.")
        flat_gain = features[..., 4].reshape(-1)
        legal_gain = flat_gain[legal_ids]
        best_gain = np.max(legal_gain)
        if best_gain <= float(greedy_tie_atol) and str(energy_greedy_zero_gain_fallback) == "traditional":
            probs = flat_rates[legal_ids]
            probs = probs / np.maximum(probs.sum(), 1e-12)
            chosen = np.random.choice(legal_ids, p=probs)
            return np.array([[int(chosen)]], dtype=np.int64), True
        best_ids = legal_ids[np.isclose(legal_gain, best_gain, rtol=0.0, atol=float(greedy_tie_atol))]
        if best_ids.size == 1:
            chosen = best_ids[0]
        else:
            chosen = random_tie_argmax(best_ids, flat_rates[best_ids], atol=float(greedy_tie_atol))
    elif mode == "rate_energy":
        features = np.asarray(event_features, dtype=np.float64)
        if features.shape[:2] != base_rates.shape or features.shape[-1] <= 4:
            raise ValueError("rate_energy requires event_features[..., 4] energy-gain values.")
        flat_gain = np.maximum(features[..., 4].reshape(-1), 0.0)
        score = np.log(np.maximum(flat_rates, 1e-12)) + float(rate_energy_beta) * flat_gain
        score = np.where(legal, score, -np.inf)
        chosen = random_tie_argmax(legal_ids, score[legal_ids], atol=float(greedy_tie_atol))
    else:
        raise ValueError(f"Unsupported heuristic mode: {mode}")

    return np.array([[int(chosen)]], dtype=np.int64), True


def build_row(*, step: int, mode: str, seed: int, initial_energy: float, info: Dict, done: bool) -> Dict:
    total_energy = float(info.get("total_energy", initial_energy))
    selected_rate = float(info.get("selected_rate", 0.0))
    total_rate = float(info.get("total_rate", 0.0))
    pred_total_hazard = float(info.get("pred_total_hazard", np.nan))
    pred_expected_dt = float(info.get("pred_expected_dt", np.nan))
    pred_selected_hazard = float(info.get("pred_selected_hazard", np.nan))
    target_total_hazard = float(info.get("target_total_hazard", total_rate))
    target_expected_dt = float(1.0 / max(target_total_hazard, 1e-12)) if target_total_hazard > 0.0 else np.nan
    kmc_delta_t = float(info.get("kmc_delta_t", 0.0))
    if np.isfinite(pred_total_hazard) and pred_total_hazard > 0.0 and target_total_hazard > 0.0:
        log_total_hazard_abs_error = float(abs(np.log(pred_total_hazard) - np.log(target_total_hazard)))
        log_expected_dt_abs_error = float(abs(np.log(max(pred_expected_dt, 1e-12)) - np.log(max(target_expected_dt, 1e-12))))
        pred_time_nll = float(-np.log(max(pred_total_hazard, 1e-12)) + pred_total_hazard * kmc_delta_t)
        target_time_nll = float(-np.log(max(target_total_hazard, 1e-12)) + target_total_hazard * kmc_delta_t)
        waiting_time_nll_residual = float(pred_time_nll - target_time_nll)
    else:
        log_total_hazard_abs_error = np.nan
        log_expected_dt_abs_error = np.nan
        waiting_time_nll_residual = np.nan
    return {
        "step": int(step),
        "mode": mode,
        "seed": int(seed),
        "total_energy": total_energy,
        "energy_drop": float(initial_energy - total_energy),
        "global_energy_change": float(info.get("global_energy_change", 0.0)),
        "kmc_time": float(info.get("kmc_time", 0.0)),
        "kmc_delta_t": kmc_delta_t,
        "selected_rate": selected_rate,
        "total_rate": total_rate,
        "selected_rate_over_total": float(selected_rate / max(total_rate, 1e-12)),
        "selected_energy_gain": float(info.get("selected_energy_gain", np.nan)),
        "best_available_energy_gain": float(info.get("best_available_energy_gain", np.nan)),
        "missed_energy_gain": float(info.get("missed_energy_gain", np.nan)),
        "pred_total_hazard": pred_total_hazard,
        "pred_expected_dt": pred_expected_dt,
        "pred_selected_hazard": pred_selected_hazard,
        "target_total_hazard": target_total_hazard,
        "target_expected_dt": target_expected_dt,
        "log_total_hazard_abs_error": log_total_hazard_abs_error,
        "log_expected_dt_abs_error": log_expected_dt_abs_error,
        "waiting_time_nll_residual": waiting_time_nll_residual,
        "legal_event": int(bool(info.get("legal_event", False))),
        "event_id": int(info.get("event_id", -1)),
        "selected_vacancy": int(info.get("selected_vacancy", -1)),
        "selected_direction": int(info.get("selected_direction", -1)),
        "done": int(bool(done)),
    }


def build_structure_row(*, step: int, mode: str, seed: int, kmc_time: float, metrics: Dict, done: bool) -> Dict:
    return {
        "step": int(step),
        "mode": mode,
        "seed": int(seed),
        "kmc_time": float(kmc_time),
        "cu_cu_pair_count": int(metrics.get("cu_cu_pair_count", 0)),
        "cluster_count": int(metrics.get("cluster_count", 0)),
        "stable_cluster_count": int(metrics.get("stable_cluster_count", 0)),
        "total_cluster_count": int(metrics.get("total_cluster_count", 0)),
        "clustered_cu_atoms": int(metrics.get("clustered_cu_atoms", 0)),
        "free_cu_atoms": int(metrics.get("free_cu_atoms", 0)),
        "aggregation_degree": float(metrics.get("aggregation_degree", 0.0)),
        "advancement_factor": float(metrics.get("advancement_factor", 0.0)),
        "largest_cluster_size": int(metrics.get("largest_cluster_size", 0)),
        "vacancy_contact_count": int(metrics.get("vacancy_contact_count", 0)),
        "vacancy_contact_ratio": float(metrics.get("vacancy_contact_ratio", 0.0)),
        "contacted_vacancy_count": int(metrics.get("contacted_vacancy_count", 0)),
        "contacted_vacancy_ratio": float(metrics.get("contacted_vacancy_ratio", 0.0)),
        "first_cluster_contact_time": float(metrics.get("first_cluster_contact_time", np.nan)),
        "mean_first_arrival_time": float(metrics.get("mean_first_arrival_time", np.nan)),
        "done": int(bool(done)),
    }


def write_csv(rows: List[Dict], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _find_root(parent: np.ndarray, idx: int) -> int:
    while parent[idx] != idx:
        parent[idx] = parent[parent[idx]]
        idx = parent[idx]
    return int(idx)


def _union_sets(parent: np.ndarray, size: np.ndarray, a: int, b: int) -> None:
    root_a = _find_root(parent, a)
    root_b = _find_root(parent, b)
    if root_a == root_b:
        return
    if size[root_a] < size[root_b]:
        root_a, root_b = root_b, root_a
    parent[root_b] = root_a
    size[root_a] += size[root_b]


def compute_structure_metrics(
    env_core,
    stable_cluster_min_size: int,
    metric_cluster_min_size: int,
    initial_free_cu: int,
    initial_pair_count: int,
    first_contact_times: np.ndarray,
) -> Dict:
    cu_pos = np.asarray(env_core.cu_pos, dtype=np.int32)
    vac_pos = np.asarray(env_core.vacancy_pos, dtype=np.int32)
    nn1 = np.asarray(env_core.NN1, dtype=np.int32)
    dims = np.asarray(env_core.lattice.shape, dtype=np.int32)
    stable_cluster_min_size = max(int(stable_cluster_min_size), 2)
    metric_cluster_min_size = max(int(metric_cluster_min_size), 2)

    total_cu = int(cu_pos.shape[0])
    total_vac = int(vac_pos.shape[0])
    cu_cu_pair_count = 0

    metric_member_mask = np.zeros(total_cu, dtype=bool)
    stable_member_mask = np.zeros(total_cu, dtype=bool)
    cluster_sizes = np.empty(0, dtype=np.int32)

    if total_cu > 0:
        pos_to_idx = getattr(env_core, "cu_pos_index", None)
        if pos_to_idx is None or len(pos_to_idx) != total_cu:
            pos_to_idx = {tuple(map(int, pos)): idx for idx, pos in enumerate(cu_pos)}
        parent = np.arange(total_cu, dtype=np.int32)
        comp_size = np.ones(total_cu, dtype=np.int32)

        for idx, pos in enumerate(cu_pos):
            for offset in nn1:
                nbr = tuple(map(int, (pos + offset) % dims))
                nbr_idx = pos_to_idx.get(nbr)
                if nbr_idx is not None:
                    if int(nbr_idx) > idx:
                        cu_cu_pair_count += 1
                    _union_sets(parent, comp_size, idx, int(nbr_idx))

        root_to_members: Dict[int, List[int]] = defaultdict(list)
        for idx in range(total_cu):
            root = _find_root(parent, idx)
            root_to_members[root].append(idx)

        cluster_sizes = np.asarray([len(members) for members in root_to_members.values()], dtype=np.int32)
        for members in root_to_members.values():
            member_idx = np.asarray(members, dtype=np.int32)
            if len(members) >= metric_cluster_min_size:
                metric_member_mask[member_idx] = True
            if len(members) >= stable_cluster_min_size:
                stable_member_mask[member_idx] = True

    metric_cluster_sizes = (
        cluster_sizes[cluster_sizes >= metric_cluster_min_size] if cluster_sizes.size > 0 else np.empty(0, dtype=np.int32)
    )
    stable_cluster_sizes = (
        cluster_sizes[cluster_sizes >= stable_cluster_min_size] if cluster_sizes.size > 0 else np.empty(0, dtype=np.int32)
    )
    clustered_cu_atoms = int(metric_member_mask.sum())
    free_cu_atoms = int(total_cu - clustered_cu_atoms)
    aggregation_degree = float(clustered_cu_atoms / max(total_cu, 1))
    extra_pair_count = max(int(cu_cu_pair_count) - int(initial_pair_count), 0)
    advancement_factor = float((2.0 * extra_pair_count) / max(initial_free_cu, 1))
    advancement_factor = float(np.clip(advancement_factor, 0.0, 1.0))
    largest_cluster_size = int(cluster_sizes.max()) if cluster_sizes.size > 0 else 0

    vacancy_contacts = np.zeros(total_vac, dtype=bool)
    if total_vac > 0 and metric_member_mask.any():
        stable_cu_sites = {
            tuple(map(int, pos)) for pos in cu_pos[metric_member_mask]
        }
        for vac_idx, vac in enumerate(vac_pos):
            for offset in nn1:
                nbr = tuple(map(int, (vac + offset) % dims))
                if nbr in stable_cu_sites:
                    vacancy_contacts[vac_idx] = True
                    break

    current_time = float(env_core.time)
    newly_contacted = np.isnan(first_contact_times) & vacancy_contacts
    first_contact_times[newly_contacted] = current_time
    finite_contact_times = first_contact_times[np.isfinite(first_contact_times)]

    return {
        "cu_cu_pair_count": int(cu_cu_pair_count),
        "cluster_count": int(metric_cluster_sizes.size),
        "stable_cluster_count": int(stable_cluster_sizes.size),
        "total_cluster_count": int(cluster_sizes.size),
        "clustered_cu_atoms": clustered_cu_atoms,
        "free_cu_atoms": free_cu_atoms,
        "aggregation_degree": aggregation_degree,
        "advancement_factor": advancement_factor,
        "largest_cluster_size": largest_cluster_size,
        "vacancy_contact_count": int(vacancy_contacts.sum()),
        "vacancy_contact_ratio": float(vacancy_contacts.mean()) if total_vac > 0 else 0.0,
        "contacted_vacancy_count": int(np.isfinite(first_contact_times).sum()),
        "contacted_vacancy_ratio": float(np.isfinite(first_contact_times).mean()) if total_vac > 0 else 0.0,
        "first_cluster_contact_time": float(np.min(finite_contact_times)) if finite_contact_times.size > 0 else np.nan,
        "mean_first_arrival_time": float(np.mean(finite_contact_times)) if finite_contact_times.size > 0 else np.nan,
    }


def run_rollout(
    mode: str,
    all_args,
    device: torch.device,
    snapshot: Dict,
    seed: int,
    policy: Optional[R_MAPPOPolicy],
    output_dir: Path,
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    rollout_start = time.perf_counter()
    env, obs = restore_env_from_snapshot(all_args, seed, snapshot, device)
    initial_energy = float(env.env.energy_last)
    energy_interval = max(1, int(all_args.energy_series_interval_steps))
    structure_interval = int(getattr(all_args, "structure_series_interval_steps", 0))
    if structure_interval <= 0:
        structure_interval = energy_interval

    save_rows: List[Dict] = []
    full_rows: List[Dict] = []
    structure_rows: List[Dict] = []

    base_rates = np.asarray(env.get_base_rates_np(copy=False), dtype=np.float32)
    event_features = np.asarray(env.get_event_features_np(copy=False), dtype=np.float32)
    sparse_defect_tokens, sparse_defect_masks = env.get_sparse_defect_context_np(copy=False)
    initial_info = {
        "total_energy": initial_energy,
        "global_energy_change": 0.0,
        "kmc_time": float(env.env.time),
        "kmc_delta_t": 0.0,
        "selected_rate": 0.0,
        "total_rate": float(base_rates.sum(dtype=np.float64)),
        "target_total_hazard": float(base_rates.sum(dtype=np.float64)),
        "pred_total_hazard": np.nan,
        "pred_expected_dt": np.nan,
        "pred_selected_hazard": np.nan,
        "legal_event": False,
        "event_id": -1,
        "selected_vacancy": -1,
        "selected_direction": -1,
    }
    init_row = build_row(step=0, mode=mode, seed=seed, initial_energy=initial_energy, info=initial_info, done=False)
    init_row["wall_elapsed_seconds"] = 0.0
    save_rows.append(init_row)
    full_rows.append(init_row)

    first_contact_times = np.full(int(all_args.lattice_v_nums), np.nan, dtype=np.float64)
    initial_structure_metrics = compute_structure_metrics(
        env.env,
        stable_cluster_min_size=int(all_args.cluster_min_size),
        metric_cluster_min_size=int(all_args.metric_cluster_min_size),
        initial_free_cu=max(int(env.env.cu_pos.shape[0]), 1),
        initial_pair_count=0,
        first_contact_times=first_contact_times,
    )
    initial_free_cu = max(int(initial_structure_metrics["free_cu_atoms"]), 1)
    initial_pair_count = int(initial_structure_metrics["cu_cu_pair_count"])
    initial_structure_metrics["advancement_factor"] = 0.0
    structure_rows.append(
        build_structure_row(
            step=0,
            mode=mode,
            seed=seed,
            kmc_time=float(env.env.time),
            metrics=initial_structure_metrics,
            done=False,
        )
    )

    rnn_states = np.zeros((all_args.lattice_v_nums, all_args.recurrent_N, all_args.hidden_size), dtype=np.float32)
    masks = np.ones((1, 1), dtype=np.float32)
    done_flag = False
    setup_wall_seconds = float(time.perf_counter() - rollout_start)
    event_loop_start = time.perf_counter()

    for step in range(1, int(all_args.bench_step) + 1):
        has_legal = bool(np.any(base_rates > 0.0))
        hazard_info: Dict = {}

        if mode in HEURISTIC_MODES:
            action, _ = sample_heuristic_action(
                mode,
                base_rates,
                event_features,
                rate_energy_beta=float(getattr(all_args, "rate_energy_beta", 1.0)),
                greedy_tie_atol=float(getattr(all_args, "greedy_tie_atol", 1e-12)),
                energy_greedy_zero_gain_fallback=str(
                    getattr(all_args, "energy_greedy_zero_gain_fallback", "traditional")
                ),
            )
        else:
            if policy is None:
                raise RuntimeError("RL rollout requested but policy is None.")
            if not has_legal:
                action = np.array([[0]], dtype=np.int64)
            else:
                rl_obs = obs
                rl_rnn_states = rnn_states
                rl_sparse_tokens = sparse_defect_tokens
                rl_sparse_masks = sparse_defect_masks
                if mode == "rl4kmc_snapshot":
                    rl_obs = np.asarray(obs, dtype=np.float32).copy()
                    if rl_obs.shape[-1] > 50:
                        rl_obs[..., 50:] = -1.0
                    rl_rnn_states = np.zeros_like(rnn_states)
                if mode == "rl4kmc_no_sparse":
                    rl_sparse_tokens = np.zeros_like(sparse_defect_tokens, dtype=np.float32)
                    rl_sparse_tokens[..., 0] = 2.0
                    rl_sparse_masks = np.zeros_like(sparse_defect_masks, dtype=np.float32)

                old_recurrent_flag = None
                if mode == "rl4kmc_snapshot":
                    old_recurrent_flag = bool(getattr(policy.actor, "use_recurrent_memory", True))
                    policy.actor.use_recurrent_memory = False
                try:
                    action, new_rnn_states, hazard_summary = sample_rl_action(
                        policy,
                        rl_obs,
                        rl_rnn_states,
                        masks,
                        base_rates,
                        event_features,
                        rl_sparse_tokens,
                        rl_sparse_masks,
                        deterministic=(mode == "rl4kmc_greedy"),
                    )
                finally:
                    if old_recurrent_flag is not None:
                        policy.actor.use_recurrent_memory = old_recurrent_flag
                if mode == "rl4kmc_snapshot":
                    rnn_states = np.zeros_like(rnn_states)
                else:
                    rnn_states = new_rnn_states
                for key, value in hazard_summary.items():
                    arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
                    hazard_info[key] = float(np.asarray(arr).reshape(-1)[0])

        obs, _, _, _, dones, info = env.step(action)
        info.update(hazard_info)
        done_flag = bool(np.asarray(dones).reshape(-1)[0])
        masks = np.zeros((1, 1), dtype=np.float32) if done_flag else np.ones((1, 1), dtype=np.float32)
        base_rates = np.asarray(
            info.get("next_base_rates", env.get_base_rates_np(copy=False)),
            dtype=np.float32,
        )
        event_features = np.asarray(
            info.get("next_event_features", env.get_event_features_np(copy=False)),
            dtype=np.float32,
        )
        next_sparse_tokens = info.get("next_sparse_defect_tokens")
        next_sparse_masks = info.get("next_sparse_defect_masks")
        if next_sparse_tokens is None or next_sparse_masks is None:
            next_sparse_tokens, next_sparse_masks = env.get_sparse_defect_context_np(copy=False)
        sparse_defect_tokens = np.asarray(next_sparse_tokens, dtype=np.float32)
        sparse_defect_masks = np.asarray(next_sparse_masks, dtype=np.float32)

        row = build_row(step=step, mode=mode, seed=seed, initial_energy=initial_energy, info=info, done=done_flag)
        row["wall_elapsed_seconds"] = float(time.perf_counter() - event_loop_start)
        full_rows.append(row)

        should_save = (
            step % energy_interval == 0
            or done_flag
            or step == int(all_args.bench_step)
        )
        if should_save:
            save_rows.append(row)

        should_record_structure = (
            step % structure_interval == 0
            or done_flag
            or step == int(all_args.bench_step)
        )
        if should_record_structure:
            structure_metrics = compute_structure_metrics(
                env.env,
                stable_cluster_min_size=int(all_args.cluster_min_size),
                metric_cluster_min_size=int(all_args.metric_cluster_min_size),
                initial_free_cu=initial_free_cu,
                initial_pair_count=initial_pair_count,
                first_contact_times=first_contact_times,
            )
            structure_rows.append(
                build_structure_row(
                    step=step,
                    mode=mode,
                    seed=seed,
                    kmc_time=float(info.get("kmc_time", 0.0)),
                    metrics=structure_metrics,
                    done=done_flag,
                )
            )

        if done_flag:
            break

    wall_time_seconds = float(time.perf_counter() - rollout_start)
    event_loop_wall_seconds = float(time.perf_counter() - event_loop_start)
    final_row = full_rows[-1] if full_rows else {}
    final_step = int(final_row.get("step", 0))
    runtime_row = {
        "mode": str(mode),
        "seed": int(seed),
        "wall_time_seconds": wall_time_seconds,
        "setup_wall_seconds": setup_wall_seconds,
        "event_loop_wall_seconds": event_loop_wall_seconds,
        "final_step": final_step,
        "events_per_second": float(final_step / max(event_loop_wall_seconds, 1e-12)),
        "milliseconds_per_event": float(1000.0 * event_loop_wall_seconds / max(final_step, 1)),
        "final_energy_drop": float(final_row.get("energy_drop", 0.0)),
        "final_kmc_time": float(final_row.get("kmc_time", 0.0)),
    }

    seed_dir = Path(output_dir) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    write_csv(save_rows, seed_dir / f"{mode}_energy_series.csv")
    write_csv(structure_rows, seed_dir / f"{mode}_structure_series.csv")
    return save_rows, full_rows, structure_rows, runtime_row


def aggregate_rows(rows: List[Dict], value_key: str):
    bucket = defaultdict(list)
    for row in rows:
        bucket[(row["mode"], int(row["step"]))].append(float(row[value_key]))

    curves = defaultdict(list)
    for (mode, step), values in sorted(bucket.items(), key=lambda x: (x[0][0], x[0][1])):
        arr = np.asarray(values, dtype=np.float64)
        curves[mode].append(
            {
                "step": int(step),
                "mean": float(np.mean(arr)),
                "std": float(np.std(arr, ddof=0)),
                "stderr": float(np.std(arr, ddof=0) / np.sqrt(max(arr.size, 1))),
                "n": int(arr.size),
            }
        )
    return curves


def plot_curves(curves, ylabel: str, title: str, save_path: Path, plot_band: str, mode_order: Sequence[str], dpi: int) -> None:
    plt.figure(figsize=(9, 5.8))
    for mode in mode_order:
        stats = curves.get(mode)
        if not stats:
            continue
        steps = np.asarray([x["step"] for x in stats], dtype=np.int32)
        mean = np.asarray([x["mean"] for x in stats], dtype=np.float64)
        std = np.asarray([x["std"] for x in stats], dtype=np.float64)
        stderr = np.asarray([x["stderr"] for x in stats], dtype=np.float64)
        plt.plot(steps, mean, label=mode, linewidth=2)
        band = None
        if plot_band == "std":
            band = std
        elif plot_band == "stderr":
            band = stderr
        if band is not None:
            plt.fill_between(steps, mean - band, mean + band, alpha=0.18)

    plt.xlabel("Step")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def resolve_aligned_time_reference_mode(requested_mode: str, available_modes: Sequence[str]) -> Optional[str]:
    available = [str(mode) for mode in available_modes]
    if not available:
        return None

    requested = str(requested_mode).strip()
    if requested == "auto":
        if "traditional" in available:
            return "traditional"
        return available[0]

    if requested in available:
        return requested

    if "traditional" in available:
        return "traditional"
    return available[0]


def build_step_aligned_time_df(df: pd.DataFrame, reference_mode: str) -> pd.DataFrame:
    if df.empty or "kmc_time" not in df.columns or "step" not in df.columns or "seed" not in df.columns:
        return pd.DataFrame()

    ref_df = (
        df.loc[df["mode"] == reference_mode, ["seed", "step", "kmc_time"]]
        .sort_values(["seed", "step", "kmc_time"])
        .drop_duplicates(subset=["seed", "step"], keep="last")
        .rename(columns={"kmc_time": "aligned_kmc_time"})
    )
    if ref_df.empty:
        return pd.DataFrame()

    aligned_df = df.merge(ref_df, on=["seed", "step"], how="left")
    aligned_df["aligned_kmc_time"] = aligned_df["aligned_kmc_time"].where(
        aligned_df["aligned_kmc_time"].notna(),
        aligned_df["kmc_time"],
    )
    aligned_df["aligned_time_reference_mode"] = reference_mode
    return aligned_df


def aggregate_rows_by_time(df: pd.DataFrame, value_key: str, time_grid_points: int, time_key: str = "kmc_time"):
    if df.empty or value_key not in df.columns or time_key not in df.columns:
        return {"grid": np.empty(0, dtype=np.float64), "curves": {}}

    final_times = (
        df.groupby(["mode", "seed"], sort=False)[time_key]
        .max()
        .to_numpy(dtype=np.float64)
    )
    max_time = float(np.nanmax(final_times)) if final_times.size > 0 else 0.0
    if not np.isfinite(max_time) or max_time <= 0.0:
        grid = np.zeros(1, dtype=np.float64)
    else:
        grid = np.linspace(0.0, max_time, max(int(time_grid_points), 2), dtype=np.float64)

    curves = {}
    for mode, mode_df in df.groupby("mode", sort=False):
        samples = []
        for _, group in mode_df.groupby("seed", sort=False):
            group = group.sort_values([time_key, "step"])
            times = group[time_key].to_numpy(dtype=np.float64)
            values = group[value_key].to_numpy(dtype=np.float64)
            if times.size == 0:
                continue
            sample_idx = np.searchsorted(times, grid, side="right") - 1
            seed_samples = np.full(grid.shape, np.nan, dtype=np.float64)
            valid = (sample_idx >= 0) & (grid <= times[-1])
            if np.any(valid):
                seed_samples[valid] = values[np.clip(sample_idx[valid], 0, len(values) - 1)]
            samples.append(seed_samples)

        if not samples:
            continue

        arr = np.vstack(samples)
        finite_counts = np.sum(np.isfinite(arr), axis=0)
        mean = np.divide(
            np.nansum(arr, axis=0),
            np.maximum(finite_counts, 1),
            where=np.maximum(finite_counts, 1) > 0,
        )
        mean = mean.astype(np.float64, copy=False)
        mean[finite_counts == 0] = np.nan
        centered = np.where(np.isfinite(arr), arr - mean[None, :], 0.0)
        var = np.divide(
            np.sum(centered * centered, axis=0),
            np.maximum(finite_counts, 1),
            where=np.maximum(finite_counts, 1) > 0,
        )
        std = np.sqrt(var).astype(np.float64, copy=False)
        std[finite_counts == 0] = np.nan
        stderr = np.zeros_like(std, dtype=np.float64)
        np.divide(
            std,
            np.sqrt(np.maximum(finite_counts, 1)),
            out=stderr,
            where=np.maximum(finite_counts, 1) > 0,
        )
        stderr[finite_counts == 0] = np.nan
        curves[mode] = {
            "mean": mean,
            "std": std,
            "stderr": stderr,
            "n": int(arr.shape[0]),
        }

    return {"grid": grid, "curves": curves}


def _smooth_finite_series(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0 or int(window) <= 1:
        return values.copy()

    win = int(window)
    if win % 2 == 0:
        win += 1
    if values.size < win:
        win = int(values.size if values.size % 2 == 1 else values.size - 1)
    if win <= 1:
        return values.copy()

    mask = np.isfinite(values)
    if not np.any(mask):
        return values.copy()

    kernel = np.ones(win, dtype=np.float64)
    filled = np.where(mask, values, 0.0)
    smooth_num = np.convolve(filled, kernel, mode="same")
    smooth_den = np.convolve(mask.astype(np.float64), kernel, mode="same")
    out = values.copy()
    valid = smooth_den > 0.0
    out[valid] = smooth_num[valid] / smooth_den[valid]
    return out


def plot_time_curves(
    time_stats,
    ylabel: str,
    title: str,
    save_path: Path,
    plot_band: str,
    mode_order: Sequence[str],
    dpi: int,
    smooth_window: int = 1,
    x_label: str = "KMC Time",
) -> None:
    grid = np.asarray(time_stats.get("grid", []), dtype=np.float64)
    curves = time_stats.get("curves", {})
    if grid.size == 0 or not curves:
        return

    plt.figure(figsize=(9, 5.8))
    for mode in mode_order:
        stats = curves.get(mode)
        if not stats:
            continue
        mean = _smooth_finite_series(np.asarray(stats["mean"], dtype=np.float64), smooth_window)
        std = _smooth_finite_series(np.asarray(stats["std"], dtype=np.float64), smooth_window)
        stderr = _smooth_finite_series(np.asarray(stats["stderr"], dtype=np.float64), smooth_window)
        plt.plot(grid, mean, label=mode, linewidth=2.8)
        band = None
        if plot_band == "std":
            band = std
        elif plot_band == "stderr":
            band = stderr
        if band is not None:
            plt.fill_between(grid, mean - band, mean + band, alpha=0.24)

    plt.xlabel(x_label)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=dpi)
    plt.close()


def bootstrap_mean_ci(values: np.ndarray, n_boot: int = 5000, alpha: float = 0.05, seed: int = 12345):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = values.size
    for i in range(n_boot):
        sample = rng.choice(values, size=n, replace=True)
        means[i] = sample.mean()
    mean = float(values.mean())
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return mean, lo, hi


def paired_ttest(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size == 0 or b.size == 0 or a.size != b.size or ttest_rel is None:
        return np.nan
    try:
        return float(ttest_rel(a, b).pvalue)
    except Exception:
        return np.nan


def compute_auc(df_mode_seed: pd.DataFrame) -> float:
    x = df_mode_seed["step"].to_numpy(dtype=np.float64)
    y = df_mode_seed["energy_drop"].to_numpy(dtype=np.float64)
    if len(x) < 2:
        return float(y[-1]) if len(y) == 1 else 0.0
    integrate = getattr(np, "trapezoid", np.trapz)
    return float(integrate(y, x))


def steps_to_target(df_mode_seed: pd.DataFrame, target: float):
    hit = df_mode_seed[df_mode_seed["energy_drop"] >= target]
    if hit.empty:
        return np.nan
    return float(hit["step"].iloc[0])


def plot_paired_diff(values: np.ndarray, seeds: np.ndarray, title: str, ylabel: str, out_path: Path, dpi: int, ci_tuple=None) -> None:
    values = np.asarray(values, dtype=np.float64)
    seeds = np.asarray(seeds)
    if values.size == 0:
        return

    plt.figure(figsize=(8.5, 5.5))
    plt.axhline(0.0, color="black", linestyle="--", linewidth=1)
    plt.boxplot(values, positions=[1.0], widths=0.22, vert=True, patch_artist=True)
    jitter = np.linspace(-0.08, 0.08, len(values)) if len(values) > 1 else np.array([0.0])
    plt.scatter(np.ones_like(values) + jitter, values, s=55, zorder=3)
    for x, y, seed in zip(np.ones_like(values) + jitter, values, seeds):
        plt.text(x + 0.015, y, str(seed), fontsize=8, alpha=0.8)

    if ci_tuple is not None and np.isfinite(ci_tuple[0]):
        mean, lo, hi = ci_tuple
        plt.errorbar([1.28], [mean], yerr=[[mean - lo], [hi - mean]], fmt="o", capsize=5, label="Bootstrap 95% CI")
        plt.legend()

    plt.xlim(0.72, 1.42)
    plt.xticks([1.0], ["paired seed diff"])
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi)
    plt.close()


def summarize_final_per_seed(full_df: pd.DataFrame) -> pd.DataFrame:
    final_df = full_df.sort_values("step").groupby(["mode", "seed"], sort=False).tail(1).copy()
    final_df = final_df[["mode", "seed", "step", "total_energy", "energy_drop", "kmc_time"]].rename(
        columns={
            "step": "final_step",
            "total_energy": "final_total_energy",
            "energy_drop": "final_energy_drop",
            "kmc_time": "final_kmc_time",
        }
    )
    return final_df.sort_values(["mode", "seed"]).reset_index(drop=True)


def summarize_final_structure_per_seed(structure_df: pd.DataFrame) -> pd.DataFrame:
    if structure_df.empty:
        return pd.DataFrame()
    final_df = structure_df.sort_values(["kmc_time", "step"]).groupby(["mode", "seed"], sort=False).tail(1).copy()
    return final_df.sort_values(["mode", "seed"]).reset_index(drop=True)


def build_target_tables(full_df: pd.DataFrame, targets: Sequence[float]) -> pd.DataFrame:
    rows = []
    for (mode, seed), group in full_df.groupby(["mode", "seed"], sort=False):
        group = group.sort_values("step")
        row = {"mode": mode, "seed": int(seed)}
        for target in targets:
            row[f"steps_to_drop_{target:g}"] = steps_to_target(group, float(target))
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["mode", "seed"]).reset_index(drop=True)


def build_summary(
    full_df: pd.DataFrame,
    save_df: pd.DataFrame,
    structure_df: pd.DataFrame,
    output_dir: Path,
    all_args,
    main_modes: Sequence[str],
) -> Dict:
    summary: Dict = {
        "config": {
            "bench_step": int(all_args.bench_step),
            "energy_series_interval_steps": int(all_args.energy_series_interval_steps),
            "structure_series_interval_steps": int(
                getattr(all_args, "structure_series_interval_steps", 0) or all_args.energy_series_interval_steps
            ),
            "compare_modes": list(all_args.compare_modes),
            "main_plot_modes": list(main_modes),
            "compare_seeds": [int(s) for s in all_args.compare_seeds],
            "cluster_min_size": int(all_args.cluster_min_size),
            "metric_cluster_min_size": int(all_args.metric_cluster_min_size),
            "plot_band": str(all_args.plot_band),
            "structure_plot_band": str(all_args.structure_plot_band),
            "structure_smooth_window": int(all_args.structure_smooth_window),
            "target_energy_drops": [float(x) for x in all_args.target_energy_drops],
            "greedy_tie_atol": float(getattr(all_args, "greedy_tie_atol", 1e-12)),
            "energy_greedy_zero_gain_fallback": str(
                getattr(all_args, "energy_greedy_zero_gain_fallback", "traditional")
            ),
            "aligned_time_reference_mode": str(getattr(all_args, "aligned_time_reference_mode", "traditional")),
            "aligned_time_reference_mode_resolved": str(
                getattr(all_args, "aligned_time_reference_mode_resolved", "") or ""
            ),
            "device": str(all_args.device),
        },
        "per_mode_final_energy_drop": {},
        "per_mode_auc_energy_drop": {},
        "paired_statistics": {},
        "steps_to_target": {},
        "structure_final_metrics": {},
    }

    final_df = summarize_final_per_seed(full_df)
    final_df.to_csv(output_dir / "final_summary_per_seed.csv", index=False)
    final_structure_df = summarize_final_structure_per_seed(structure_df)
    if not final_structure_df.empty:
        final_structure_df.to_csv(output_dir / "final_structure_summary_per_seed.csv", index=False)
        for mode in sorted(final_structure_df["mode"].unique()):
            mode_df = final_structure_df.loc[final_structure_df["mode"] == mode]
            summary["structure_final_metrics"][mode] = {
                "cu_cu_pair_count_mean": float(mode_df["cu_cu_pair_count"].mean()),
                "cluster_count_mean": float(mode_df["cluster_count"].mean()),
                "stable_cluster_count_mean": float(mode_df["stable_cluster_count"].mean()),
                "aggregation_degree_mean": float(mode_df["aggregation_degree"].mean()),
                "advancement_factor_mean": float(mode_df["advancement_factor"].mean()),
                "largest_cluster_size_mean": float(mode_df["largest_cluster_size"].mean()),
                "first_cluster_contact_time_mean": float(mode_df["first_cluster_contact_time"].mean(skipna=True)),
                "mean_first_arrival_time_mean": float(mode_df["mean_first_arrival_time"].mean(skipna=True)),
                "n": int(mode_df.shape[0]),
            }

    auc_rows = []
    for (mode, seed), group in full_df.groupby(["mode", "seed"], sort=False):
        auc_rows.append({"mode": mode, "seed": int(seed), "auc_energy_drop": compute_auc(group.sort_values("step"))})
    auc_df = pd.DataFrame(auc_rows).sort_values(["mode", "seed"]).reset_index(drop=True)
    auc_df.to_csv(output_dir / "auc_summary_per_seed.csv", index=False)

    target_df = build_target_tables(full_df, all_args.target_energy_drops)
    target_df.to_csv(output_dir / "steps_to_target_by_seed.csv", index=False)

    for mode in sorted(full_df["mode"].unique()):
        final_vals = final_df.loc[final_df["mode"] == mode, "final_energy_drop"].to_numpy(dtype=np.float64)
        auc_vals = auc_df.loc[auc_df["mode"] == mode, "auc_energy_drop"].to_numpy(dtype=np.float64)
        summary["per_mode_final_energy_drop"][mode] = {
            "mean": float(np.mean(final_vals)) if final_vals.size else np.nan,
            "std": float(np.std(final_vals, ddof=1)) if final_vals.size > 1 else 0.0,
            "n": int(final_vals.size),
        }
        summary["per_mode_auc_energy_drop"][mode] = {
            "mean": float(np.mean(auc_vals)) if auc_vals.size else np.nan,
            "std": float(np.std(auc_vals, ddof=1)) if auc_vals.size > 1 else 0.0,
            "n": int(auc_vals.size),
        }

    if {"traditional", "rl4kmc_stochastic"}.issubset(set(full_df["mode"].unique())):
        final_pivot = final_df.pivot(index="seed", columns="mode", values="final_energy_drop")
        auc_pivot = auc_df.pivot(index="seed", columns="mode", values="auc_energy_drop")

        common_final = final_pivot[["traditional", "rl4kmc_stochastic"]].dropna().copy()
        common_auc = auc_pivot[["traditional", "rl4kmc_stochastic"]].dropna().copy()

        paired_df = pd.DataFrame(index=sorted(set(common_final.index).intersection(set(common_auc.index))))
        paired_df.index.name = "seed"
        paired_df["traditional_final_energy_drop"] = common_final.loc[paired_df.index, "traditional"]
        paired_df["rl4kmc_stochastic_final_energy_drop"] = common_final.loc[paired_df.index, "rl4kmc_stochastic"]
        paired_df["final_energy_drop_diff"] = (
            paired_df["rl4kmc_stochastic_final_energy_drop"] - paired_df["traditional_final_energy_drop"]
        )
        paired_df["traditional_auc_energy_drop"] = common_auc.loc[paired_df.index, "traditional"]
        paired_df["rl4kmc_stochastic_auc_energy_drop"] = common_auc.loc[paired_df.index, "rl4kmc_stochastic"]
        paired_df["auc_energy_drop_diff"] = (
            paired_df["rl4kmc_stochastic_auc_energy_drop"] - paired_df["traditional_auc_energy_drop"]
        )

        target_pivot_map = {}
        for target in all_args.target_energy_drops:
            col = f"steps_to_drop_{target:g}"
            pivot = target_df.pivot(index="seed", columns="mode", values=col)
            target_pivot_map[col] = pivot
            if {"traditional", "rl4kmc_stochastic"}.issubset(set(pivot.columns)):
                paired_df[f"traditional_{col}"] = pivot.loc[paired_df.index, "traditional"]
                paired_df[f"rl4kmc_stochastic_{col}"] = pivot.loc[paired_df.index, "rl4kmc_stochastic"]
                paired_df[f"{col}_diff"] = (
                    paired_df[f"traditional_{col}"] - paired_df[f"rl4kmc_stochastic_{col}"]
                )

        paired_df = paired_df.reset_index()
        paired_df.to_csv(output_dir / "paired_seed_differences.csv", index=False)

        final_vals = paired_df["final_energy_drop_diff"].dropna().to_numpy(dtype=np.float64)
        final_seeds = paired_df.loc[paired_df["final_energy_drop_diff"].notna(), "seed"].to_numpy(dtype=np.int64)
        final_ci = bootstrap_mean_ci(final_vals, n_boot=int(all_args.bootstrap_samples))
        plot_paired_diff(
            final_vals,
            final_seeds,
            "Paired Final Energy-Drop Difference by Seed",
            "Final Energy-Drop Difference (eV)",
            output_dir / "paired_final_energy_drop_diff.png",
            dpi=int(all_args.plot_dpi),
            ci_tuple=final_ci,
        )
        summary["paired_statistics"]["final_energy_drop_diff"] = {
            "mean": final_ci[0],
            "bootstrap_ci_95": [final_ci[1], final_ci[2]],
            "paired_ttest_pvalue": paired_ttest(
                paired_df["rl4kmc_stochastic_final_energy_drop"].dropna().to_numpy(dtype=np.float64),
                paired_df["traditional_final_energy_drop"].dropna().to_numpy(dtype=np.float64),
            ),
            "n": int(final_vals.size),
        }

        auc_vals = paired_df["auc_energy_drop_diff"].dropna().to_numpy(dtype=np.float64)
        auc_seeds = paired_df.loc[paired_df["auc_energy_drop_diff"].notna(), "seed"].to_numpy(dtype=np.int64)
        auc_ci = bootstrap_mean_ci(auc_vals, n_boot=int(all_args.bootstrap_samples), seed=22345)
        plot_paired_diff(
            auc_vals,
            auc_seeds,
            "Paired AUC Energy-Drop Difference by Seed",
            "AUC Energy-Drop Difference (eV·step)",
            output_dir / "paired_auc_energy_drop_diff.png",
            dpi=int(all_args.plot_dpi),
            ci_tuple=auc_ci,
        )
        summary["paired_statistics"]["auc_energy_drop_diff"] = {
            "mean": auc_ci[0],
            "bootstrap_ci_95": [auc_ci[1], auc_ci[2]],
            "paired_ttest_pvalue": paired_ttest(
                paired_df["rl4kmc_stochastic_auc_energy_drop"].dropna().to_numpy(dtype=np.float64),
                paired_df["traditional_auc_energy_drop"].dropna().to_numpy(dtype=np.float64),
            ),
            "n": int(auc_vals.size),
        }

        for target in all_args.target_energy_drops:
            key = f"steps_to_drop_{target:g}"
            mode_stats = []
            if key in target_df.columns:
                grouped = target_df.groupby("mode")[key]
                for mode, values in grouped:
                    vals = values.to_numpy(dtype=np.float64)
                    finite = vals[np.isfinite(vals)]
                    mode_stats.append(
                        {
                            "mode": mode,
                            "mean": float(np.mean(finite)) if finite.size else np.nan,
                            "std": float(np.std(finite, ddof=1)) if finite.size > 1 else (0.0 if finite.size == 1 else np.nan),
                            "n_hit": int(finite.size),
                            "n_total": int(vals.size),
                        }
                    )
                summary["steps_to_target"][key] = mode_stats

            diff_col = f"{key}_diff"
            if diff_col in paired_df.columns:
                diff_vals = paired_df[diff_col].dropna().to_numpy(dtype=np.float64)
                diff_seeds = paired_df.loc[paired_df[diff_col].notna(), "seed"].to_numpy(dtype=np.int64)
                diff_ci = bootstrap_mean_ci(diff_vals, n_boot=int(all_args.bootstrap_samples), seed=32345)
                plot_paired_diff(
                    diff_vals,
                    diff_seeds,
                    f"Paired Steps-to-{target:g}eV Difference by Seed",
                    f"Traditional - RL Stochastic Steps to {target:g} eV",
                    output_dir / f"paired_steps_to_drop_{target:g}_diff.png",
                    dpi=int(all_args.plot_dpi),
                    ci_tuple=diff_ci,
                )
                summary["paired_statistics"][diff_col] = {
                    "mean": diff_ci[0],
                    "bootstrap_ci_95": [diff_ci[1], diff_ci[2]],
                    "paired_ttest_pvalue": paired_ttest(
                        paired_df[f"traditional_{key}"].dropna().to_numpy(dtype=np.float64),
                        paired_df[f"rl4kmc_stochastic_{key}"].dropna().to_numpy(dtype=np.float64),
                    ),
                    "n": int(diff_vals.size),
                }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return summary


def main(argv: Sequence[str]) -> None:
    parser = get_config()
    parser = add_compare_args(parser)
    all_args = parser.parse_known_args(argv)[0]

    device = resolve_device(all_args)
    configure_runtime_args(all_args, device)

    output_dir = Path(all_args.compare_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    probe_env = instantiate_env(all_args)
    policy = load_policy_if_needed(all_args, probe_env.action_space, device, all_args.compare_modes)

    all_save_rows: List[Dict] = []
    all_full_rows: List[Dict] = []
    all_structure_rows: List[Dict] = []
    all_runtime_rows: List[Dict] = []

    for seed in all_args.compare_seeds:
        seed = int(seed)
        snapshot = create_seed_snapshot(all_args, seed, device)
        for mode in all_args.compare_modes:
            save_rows, full_rows, structure_rows, runtime_row = run_rollout(mode, all_args, device, snapshot, seed, policy, output_dir)
            all_save_rows.extend(save_rows)
            all_full_rows.extend(full_rows)
            all_structure_rows.extend(structure_rows)
            all_runtime_rows.append(runtime_row)

    write_csv(all_save_rows, output_dir / "merged_energy_series.csv")
    write_csv(all_full_rows, output_dir / "merged_energy_series_full_metrics.csv")
    write_csv(all_structure_rows, output_dir / "merged_structure_time_series.csv")
    write_csv(all_runtime_rows, output_dir / "runtime_measurements.csv")

    save_df = pd.DataFrame(all_save_rows)
    full_df = pd.DataFrame(all_full_rows)
    structure_df = pd.DataFrame(all_structure_rows)

    main_modes = [m for m in MAIN_MODES if m in set(all_args.compare_modes)]
    plot_modes = main_modes if main_modes else list(all_args.compare_modes)
    aligned_reference_mode = resolve_aligned_time_reference_mode(
        str(getattr(all_args, "aligned_time_reference_mode", "traditional")),
        plot_modes,
    )
    all_args.aligned_time_reference_mode_resolved = aligned_reference_mode or ""
    aligned_structure_df = pd.DataFrame()
    if aligned_reference_mode is not None:
        aligned_structure_df = build_step_aligned_time_df(
            structure_df.loc[structure_df["mode"].isin(plot_modes)].copy(),
            reference_mode=aligned_reference_mode,
        )
        if not aligned_structure_df.empty:
            aligned_structure_df.to_csv(output_dir / "merged_structure_time_series_aligned.csv", index=False)

    energy_curves_main = aggregate_rows(all_full_rows, "total_energy")
    drop_curves_main = aggregate_rows(all_full_rows, "energy_drop")

    plot_curves(
        energy_curves_main,
        ylabel="Total Energy (eV)",
        title=f"{all_args.plot_title}: Total Energy vs Step",
        save_path=output_dir / "total_energy_vs_step_main.png",
        plot_band=all_args.plot_band,
        mode_order=plot_modes,
        dpi=int(all_args.plot_dpi),
    )
    plot_curves(
        drop_curves_main,
        ylabel="Energy Drop from Initial State (eV)",
        title=f"{all_args.plot_title}: Energy Drop vs Step",
        save_path=output_dir / "energy_drop_vs_step_main.png",
        plot_band=all_args.plot_band,
        mode_order=plot_modes,
        dpi=int(all_args.plot_dpi),
    )

    if any(mode not in plot_modes for mode in all_args.compare_modes):
        plot_curves(
            energy_curves_main,
            ylabel="Total Energy (eV)",
            title=f"{all_args.plot_title}: Total Energy vs Step (debug all modes)",
            save_path=output_dir / "total_energy_vs_step_debug_all_modes.png",
            plot_band=all_args.plot_band,
            mode_order=all_args.compare_modes,
            dpi=int(all_args.plot_dpi),
        )
        plot_curves(
            drop_curves_main,
            ylabel="Energy Drop from Initial State (eV)",
            title=f"{all_args.plot_title}: Energy Drop vs Step (debug all modes)",
            save_path=output_dir / "energy_drop_vs_step_debug_all_modes.png",
            plot_band=all_args.plot_band,
            mode_order=all_args.compare_modes,
            dpi=int(all_args.plot_dpi),
        )

    for metric_key, ylabel, filename in STRUCTURE_TIME_METRICS:
        time_stats = aggregate_rows_by_time(
            structure_df.loc[structure_df["mode"].isin(plot_modes)].copy(),
            value_key=metric_key,
            time_grid_points=int(all_args.time_grid_points),
        )
        plot_time_curves(
            time_stats,
            ylabel=ylabel,
            title=f"{all_args.plot_title}: {ylabel} vs KMC Time",
            save_path=output_dir / filename,
            plot_band=all_args.structure_plot_band,
            mode_order=plot_modes,
            dpi=int(all_args.plot_dpi),
            smooth_window=int(all_args.structure_smooth_window),
        )
        if not aligned_structure_df.empty:
            aligned_time_stats = aggregate_rows_by_time(
                aligned_structure_df,
                value_key=metric_key,
                time_grid_points=int(all_args.time_grid_points),
                time_key="aligned_kmc_time",
            )
            aligned_filename = filename.replace("_vs_time_", "_vs_aligned_time_")
            plot_time_curves(
                aligned_time_stats,
                ylabel=ylabel,
                title=f"{all_args.plot_title}: {ylabel} vs Step-Aligned KMC Time (ref={aligned_reference_mode})",
                save_path=output_dir / aligned_filename,
                plot_band=all_args.structure_plot_band,
                mode_order=plot_modes,
                dpi=int(all_args.plot_dpi),
                smooth_window=int(all_args.structure_smooth_window),
                x_label=f"Aligned KMC Time (ref: {aligned_reference_mode})",
            )

    summary = build_summary(full_df, save_df, structure_df, output_dir, all_args, plot_modes)

    print(f"Using device: {device}")
    print(f"Saved interval CSV to: {output_dir / 'merged_energy_series.csv'}")
    print(f"Saved full-metrics CSV to: {output_dir / 'merged_energy_series_full_metrics.csv'}")
    print(f"Saved structure CSV to: {output_dir / 'merged_structure_time_series.csv'}")
    print(f"Saved runtime CSV to: {output_dir / 'runtime_measurements.csv'}")
    if not aligned_structure_df.empty:
        print(f"Saved aligned structure CSV to: {output_dir / 'merged_structure_time_series_aligned.csv'}")
        print(f"Aligned time reference mode: {aligned_reference_mode}")
    print(f"Saved plots and summary files to: {output_dir}")
    print(json.dumps(summary.get("paired_statistics", {}), indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
