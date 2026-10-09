#!/usr/bin/env python
"""Matched-initial-state comparison of classic KMC and FCC RL-KMC policies."""

import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from atomworld.backends.fcc.config.parser import get_config
from atomworld.backends.fcc.envs.fcc_lattice import AL, CR, NI, VACANCY
from atomworld.backends.fcc.envs.kmc_env import KMCEnvWrap
from atomworld.backends.fcc.PPO.rMAPPOPolicy import R_MAPPOPolicy
from atomworld.backends.fcc.fcc_l12 import L12DetectorConfig, analyze_l12_lattice


MODES = (
    "classic_kmc", "uniform_random", "rate_greedy", "energy_greedy",
    "rl_stochastic", "rl_greedy",
)
RL_MODES = {"rl_stochastic", "rl_greedy"}
COLORS = {
    "classic_kmc": "#202124",
    "uniform_random": "#7f8c8d",
    "rate_greedy": "#6a5acd",
    "energy_greedy": "#2e8b57",
    "rl_stochastic": "#d45500",
    "rl_greedy": "#c2185b",
}


def add_compare_args(parser):
    parser.add_argument("--bench_step", type=int, default=500)
    parser.add_argument("--energy_series_interval_steps", type=int, default=10)
    parser.add_argument("--structure_series_interval_steps", type=int, default=50)
    parser.add_argument("--compare_seeds", type=int, nargs="+", default=[1, 2, 3])
    parser.add_argument(
        "--compare_modes", nargs="+", choices=MODES,
        default=["classic_kmc", "rl_stochastic", "rl_greedy"],
    )
    parser.add_argument("--compare_output_dir", type=str, required=True)
    parser.add_argument("--plot_title", type=str, default="FCC Ni-Al-Cr: classic KMC vs RL-KMC")
    parser.add_argument("--plot_band", choices=("std", "stderr", "none"), default="stderr")
    parser.add_argument("--plot_dpi", type=int, default=180)
    parser.add_argument("--target_energy_drops", type=float, nargs="*", default=[])
    parser.add_argument("--rate_energy_beta", type=float, default=1.0)
    parser.add_argument("--bootstrap_samples", type=int, default=0)  # command compatibility
    return parser


def resolve_device(args):
    if not bool(args.cuda) or not torch.cuda.is_available():
        args.cuda = False
        args.device = "cpu"
        return torch.device("cpu")
    device = torch.device(str(args.device or "cuda:0"))
    torch.cuda.set_device(device)
    args.device = str(device)
    return device


def seed_everything(seed, device):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def write_csv(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def create_snapshot(args, seed, device):
    seed_everything(seed, device)
    env = KMCEnvWrap(args)
    env.seed(seed)
    env.reset()
    return env.get_state_dict()


def restore_snapshot(args, snapshot, seed, device):
    env = KMCEnvWrap(args)
    env.set_state_dict(snapshot)
    seed_everything(seed, device)
    obs = env.env.get_combined_obs().astype(np.float32)
    return env, obs


def load_policy(args, action_space, device):
    if not any(mode in RL_MODES for mode in args.compare_modes):
        return None
    if not args.model_dir:
        raise ValueError("--model_dir is required for RL comparison modes")
    actor_path = Path(args.model_dir) / "actor_agent.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor checkpoint not found: {actor_path}")
    policy = R_MAPPOPolicy(args, action_space, device=device)
    message = policy.actor.load_state_dict(torch.load(actor_path, map_location=device), strict=True)
    if message.missing_keys or message.unexpected_keys:
        raise RuntimeError(f"Actor checkpoint mismatch: {message}")
    policy.actor.eval()
    policy.critic.eval()
    return policy


def choose_non_rl_action(mode, rates, features):
    flat_rates = np.asarray(rates, dtype=np.float64).reshape(-1)
    legal_ids = np.flatnonzero(flat_rates > 0.0)
    if legal_ids.size == 0:
        return np.asarray([[0]], dtype=np.int64)
    if mode == "classic_kmc":
        probabilities = flat_rates[legal_ids] / flat_rates[legal_ids].sum()
        selected = int(np.random.choice(legal_ids, p=probabilities))
    elif mode == "uniform_random":
        selected = int(np.random.choice(legal_ids))
    elif mode == "rate_greedy":
        legal_values = flat_rates[legal_ids]
        selected = int(np.random.choice(legal_ids[np.isclose(legal_values, legal_values.max())]))
    elif mode == "energy_greedy":
        gains = np.asarray(features, dtype=np.float64)[..., 4].reshape(-1)
        legal_gains = gains[legal_ids]
        best_ids = legal_ids[np.isclose(legal_gains, legal_gains.max())]
        if legal_gains.max() <= 0.0:
            probabilities = flat_rates[legal_ids] / flat_rates[legal_ids].sum()
            selected = int(np.random.choice(legal_ids, p=probabilities))
        else:
            selected = int(best_ids[np.argmax(flat_rates[best_ids])])
    else:
        raise ValueError(f"Unsupported non-RL mode: {mode}")
    return np.asarray([[selected]], dtype=np.int64)


@torch.no_grad()
def choose_rl_action(policy, deterministic, obs, rnn_states, rates, features, tokens, token_masks):
    action, next_states, hazard = policy.act(
        obs,
        rnn_states,
        np.ones((1, 1), dtype=np.float32),
        rates,
        features,
        sparse_defect_tokens=tokens,
        sparse_defect_masks=token_masks,
        deterministic=deterministic,
        return_hazard_summary=True,
    )
    scalar_hazard = {}
    for key, value in hazard.items():
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        scalar_hazard[key] = float(np.asarray(array).reshape(-1)[0])
    return action.detach().cpu().numpy(), next_states.detach().cpu().numpy(), scalar_hazard


def _half_neighbor_vectors(nn1):
    selected = []
    for vector in np.asarray(nn1, dtype=np.int32):
        first_nonzero = vector[np.flatnonzero(vector)[0]]
        if first_nonzero > 0:
            selected.append(vector)
    return np.asarray(selected, dtype=np.int32)


def _cowley_alpha(lattice, coords, nn1, atom_types, center_type, neighbor_type):
    centers = coords[atom_types == center_type]
    nonvacancy_count = int(np.count_nonzero(atom_types != VACANCY))
    neighbor_count = int(np.count_nonzero(atom_types == neighbor_type))
    if centers.size == 0 or neighbor_count == 0 or nonvacancy_count == 0:
        return np.nan
    dims = np.asarray(lattice.shape, dtype=np.int32)
    neighbor_coords = (centers[:, None, :] + nn1[None, :, :]) % dims
    values = lattice[neighbor_coords[..., 0], neighbor_coords[..., 1], neighbor_coords[..., 2]]
    conditional = float(np.count_nonzero(values == neighbor_type) / values.size)
    concentration = float(neighbor_count / nonvacancy_count)
    return float(1.0 - conditional / max(concentration, 1.0e-12))


def _solute_clusters(lattice, solute_coords, half_nn1):
    count = int(len(solute_coords))
    if count == 0:
        return np.empty(0, dtype=np.int32)
    dims = np.asarray(lattice.shape, dtype=np.int32)
    linear = np.ravel_multi_index(solute_coords.T, lattice.shape)
    lookup = {int(site): index for index, site in enumerate(linear)}
    parent = np.arange(count, dtype=np.int32)
    sizes = np.ones(count, dtype=np.int32)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left, right):
        root_left, root_right = find(left), find(right)
        if root_left == root_right:
            return
        if sizes[root_left] < sizes[root_right]:
            root_left, root_right = root_right, root_left
        parent[root_right] = root_left
        sizes[root_left] += sizes[root_right]

    for index, position in enumerate(solute_coords):
        neighbors = (position[None, :] + half_nn1) % dims
        for site in np.ravel_multi_index(neighbors.T, lattice.shape):
            other = lookup.get(int(site))
            if other is not None:
                union(index, other)
    roots = np.asarray([find(index) for index in range(count)], dtype=np.int32)
    return np.unique(roots, return_counts=True)[1].astype(np.int32)


def compute_fcc_structure_metrics(env):
    lattice = env.lattice
    coords = np.asarray(env.coords, dtype=np.int32)
    atom_types = lattice[tuple(coords.T)]
    nn1 = np.asarray(env.NN1, dtype=np.int32)
    half_nn1 = _half_neighbor_vectors(nn1)
    dims = np.asarray(lattice.shape, dtype=np.int32)

    pair_counts = defaultdict(int)
    neighbors = (coords[:, None, :] + half_nn1[None, :, :]) % dims
    neighbor_types = lattice[neighbors[..., 0], neighbors[..., 1], neighbors[..., 2]]
    for left, right_row in zip(atom_types, neighbor_types):
        for right in right_row:
            pair_counts[tuple(sorted((int(left), int(right))))] += 1

    solute_coords = coords[(atom_types == AL) | (atom_types == CR)]
    cluster_sizes = _solute_clusters(lattice, solute_coords, half_nn1)
    clustered_atoms = int(cluster_sizes[cluster_sizes >= 2].sum()) if cluster_sizes.size else 0
    vacancy_coords = coords[atom_types == VACANCY]
    if vacancy_coords.size:
        vacancy_neighbors = (vacancy_coords[:, None, :] + nn1[None, :, :]) % dims
        vacancy_neighbor_types = lattice[
            vacancy_neighbors[..., 0], vacancy_neighbors[..., 1], vacancy_neighbors[..., 2]
        ]
        vacancy_contact = float(
            np.mean(np.any((vacancy_neighbor_types == AL) | (vacancy_neighbor_types == CR), axis=1))
        )
    else:
        vacancy_contact = np.nan

    metrics = {
        "al_al_pairs": int(pair_counts[(AL, AL)]),
        "cr_cr_pairs": int(pair_counts[(CR, CR)]),
        "al_cr_pairs": int(pair_counts[(AL, CR)]),
        "solute_nn_pairs": int(
            pair_counts[(AL, AL)] + pair_counts[(CR, CR)] + pair_counts[(AL, CR)]
        ),
        "solute_cluster_count": int(cluster_sizes.size),
        "largest_solute_cluster": int(cluster_sizes.max()) if cluster_sizes.size else 0,
        "clustered_solute_fraction": float(clustered_atoms / max(len(solute_coords), 1)),
        "alpha_ni_al": _cowley_alpha(lattice, coords, nn1, atom_types, NI, AL),
        "alpha_ni_cr": _cowley_alpha(lattice, coords, nn1, atom_types, NI, CR),
        "alpha_al_cr": _cowley_alpha(lattice, coords, nn1, atom_types, AL, CR),
        "vacancy_solute_contact_fraction": vacancy_contact,
    }
    metrics.update(analyze_l12_lattice(lattice, L12DetectorConfig()))
    return metrics


def energy_row(step, mode, seed, initial_energy, info, wall_seconds, event_totals=None):
    energy = float(info.get("total_energy", initial_energy))
    row = {
        "step": int(step),
        "mode": mode,
        "seed": int(seed),
        "total_energy": energy,
        "energy_drop": float(initial_energy - energy),
        "kmc_time": float(info.get("kmc_time", 0.0)),
        "kmc_delta_t": float(info.get("kmc_delta_t", 0.0)),
        "selected_rate": float(info.get("selected_rate", 0.0)),
        "total_rate": float(info.get("total_rate", 0.0)),
        "selected_energy_gain": float(info.get("selected_energy_gain", np.nan)),
        "selected_barrier": float(info.get("selected_barrier", np.nan)),
        "pred_total_hazard": float(info.get("pred_total_hazard", np.nan)),
        "pred_expected_dt": float(info.get("pred_expected_dt", np.nan)),
        "wall_seconds": float(wall_seconds),
        "done": int(bool(info.get("done", False))),
    }
    totals = event_totals or {}
    row.update({
        "cumulative_ni_moves": int(totals.get("ni", 0)),
        "cumulative_al_moves": int(totals.get("al", 0)),
        "cumulative_cr_moves": int(totals.get("cr", 0)),
        "cumulative_downhill_moves": int(totals.get("downhill", 0)),
        "cumulative_abs_energy_change": float(totals.get("abs_energy_change", 0.0)),
    })
    return row


def structure_row(step, mode, seed, kmc_time, metrics):
    return {"step": int(step), "mode": mode, "seed": int(seed), "kmc_time": float(kmc_time), **metrics}


def run_rollout(mode, args, device, snapshot, seed, policy, output_dir):
    started = time.perf_counter()
    env, obs = restore_snapshot(args, snapshot, seed, device)
    initial_energy = float(env.env.energy_last)
    rates = np.asarray(env.get_base_rates_np(copy=False), dtype=np.float32)
    features = np.asarray(env.get_event_features_np(copy=False), dtype=np.float32)
    tokens, token_masks = env.get_sparse_defect_context_np(copy=False)
    rnn_states = np.zeros(
        (args.lattice_v_nums, args.recurrent_N, args.hidden_size), dtype=np.float32
    )
    energy_rows = []
    structure_rows = []
    event_totals = {"ni": 0, "al": 0, "cr": 0, "downhill": 0, "abs_energy_change": 0.0}
    initial_info = {
        "total_energy": initial_energy,
        "kmc_time": float(env.env.time),
        "total_rate": float(rates.sum(dtype=np.float64)),
    }
    energy_rows.append(energy_row(0, mode, seed, initial_energy, initial_info, 0.0, event_totals))
    structure_rows.append(
        structure_row(0, mode, seed, float(env.env.time), compute_fcc_structure_metrics(env.env))
    )

    for step in range(1, int(args.bench_step) + 1):
        hazard_info = {}
        if mode in RL_MODES:
            action, rnn_states, hazard_info = choose_rl_action(
                policy, mode == "rl_greedy", obs, rnn_states, rates, features, tokens, token_masks
            )
        else:
            action = choose_non_rl_action(mode, rates, features)
        selected_event = int(np.asarray(action).reshape(-1)[0])
        flat_features = np.asarray(features).reshape(-1, np.asarray(features).shape[-1])
        selected_features = flat_features[selected_event]
        obs, _, _, _, dones, info = env.step(action)
        info["selected_barrier"] = float(selected_features[5])
        moving_type = int(info.get("moving_type", -1))
        if moving_type == NI:
            event_totals["ni"] += 1
        elif moving_type == AL:
            event_totals["al"] += 1
        elif moving_type == CR:
            event_totals["cr"] += 1
        gain = float(info.get("global_energy_change", 0.0))
        event_totals["downhill"] += int(gain > 0.0)
        event_totals["abs_energy_change"] += abs(gain)
        info.update(hazard_info)
        done = bool(np.asarray(dones).reshape(-1)[0])
        info["done"] = done
        rates = np.asarray(info["next_base_rates"], dtype=np.float32)
        features = np.asarray(info["next_event_features"], dtype=np.float32)
        tokens = np.asarray(info["next_sparse_defect_tokens"], dtype=np.float32)
        token_masks = np.asarray(info["next_sparse_defect_masks"], dtype=np.float32)

        if step % max(1, args.energy_series_interval_steps) == 0 or step == args.bench_step or done:
            energy_rows.append(
                energy_row(
                    step, mode, seed, initial_energy, info,
                    time.perf_counter() - started, event_totals,
                )
            )
        if step % max(1, args.structure_series_interval_steps) == 0 or step == args.bench_step or done:
            structure_rows.append(
                structure_row(step, mode, seed, float(info["kmc_time"]), compute_fcc_structure_metrics(env.env))
            )
        if done:
            break

    seed_dir = output_dir / f"seed_{seed}"
    write_csv(energy_rows, seed_dir / f"{mode}_energy.csv")
    write_csv(structure_rows, seed_dir / f"{mode}_structure.csv")
    return energy_rows, structure_rows


def _band(values, band):
    if values.shape[0] <= 1 or band == "none":
        return np.zeros(values.shape[1], dtype=np.float64)
    std = np.nanstd(values, axis=0, ddof=1)
    return std if band == "std" else std / np.sqrt(values.shape[0])


def aggregate_by_step(rows, metric):
    buckets = defaultdict(lambda: defaultdict(list))
    for row in rows:
        buckets[row["mode"]][int(row["step"])].append(float(row[metric]))
    result = {}
    for mode, step_map in buckets.items():
        steps = np.asarray(sorted(step_map), dtype=np.int64)
        values = [np.asarray(step_map[step], dtype=np.float64) for step in steps]
        result[mode] = {
            "x": steps,
            "mean": np.asarray([np.nanmean(item) for item in values]),
            "spread": np.asarray([
                (np.nanstd(item, ddof=1) if len(item) > 1 else 0.0) for item in values
            ]),
            "n": np.asarray([len(item) for item in values]),
        }
    return result


def plot_step(rows, metric, ylabel, output, args, ax=None):
    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(6.2, 4.0))
    for mode in args.compare_modes:
        stats = aggregate_by_step(rows, metric).get(mode)
        if stats is None:
            continue
        spread = stats["spread"]
        if args.plot_band == "stderr":
            spread = spread / np.sqrt(np.maximum(stats["n"], 1))
        if args.plot_band == "none":
            spread = np.zeros_like(spread)
        ax.plot(stats["x"], stats["mean"], label=mode, color=COLORS[mode], linewidth=2.0)
        ax.fill_between(stats["x"], stats["mean"] - spread, stats["mean"] + spread,
                        color=COLORS[mode], alpha=0.16)
    ax.set_xlabel("Executed vacancy events")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    if own_figure:
        ax.legend(frameon=False)
        ax.figure.tight_layout()
        ax.figure.savefig(output, dpi=args.plot_dpi, bbox_inches="tight")
        plt.close(ax.figure)


def plot_energy_vs_time(rows, output, args, ax=None):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["mode"], int(row["seed"]))].append(row)
    endpoints = [max(float(item["kmc_time"]) for item in series) for series in grouped.values()]
    common_end = min(endpoints) if endpoints else 0.0
    grid = np.linspace(0.0, common_end, 160)
    mode_curves = defaultdict(list)
    for (mode, _), series in grouped.items():
        ordered = sorted(series, key=lambda item: float(item["kmc_time"]))
        times = np.asarray([float(item["kmc_time"]) for item in ordered])
        drops = np.asarray([float(item["energy_drop"]) for item in ordered])
        unique_times, unique_index = np.unique(times, return_index=True)
        mode_curves[mode].append(np.interp(grid, unique_times, drops[unique_index]))
    own_figure = ax is None
    if own_figure:
        _, ax = plt.subplots(figsize=(6.2, 4.0))
    for mode in args.compare_modes:
        if not mode_curves.get(mode):
            continue
        values = np.asarray(mode_curves[mode], dtype=np.float64)
        mean = np.nanmean(values, axis=0)
        spread = _band(values, args.plot_band)
        ax.plot(grid, mean, label=mode, color=COLORS[mode], linewidth=2.0)
        ax.fill_between(grid, mean - spread, mean + spread, color=COLORS[mode], alpha=0.16)
    ax.set_xlabel("KMC physical time / s")
    ax.set_ylabel("Energy drop / eV")
    ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    ax.grid(alpha=0.2)
    if own_figure:
        ax.legend(frameon=False)
        ax.figure.tight_layout()
        ax.figure.savefig(output, dpi=args.plot_dpi, bbox_inches="tight")
        plt.close(ax.figure)


def compute_auc(series):
    ordered = sorted(series, key=lambda row: int(row["step"]))
    x = np.asarray([float(row["step"]) for row in ordered])
    y = np.asarray([float(row["energy_drop"]) for row in ordered])
    return float(np.trapezoid(y, x)) if len(x) > 1 else 0.0


def summarize(energy_rows, structure_rows, args, output_dir, parameter_metadata):
    energy_groups = defaultdict(list)
    structure_groups = defaultdict(list)
    for row in energy_rows:
        energy_groups[(row["mode"], int(row["seed"]))].append(row)
    for row in structure_rows:
        structure_groups[(row["mode"], int(row["seed"]))].append(row)

    final_rows = []
    for key, series in energy_groups.items():
        final = max(series, key=lambda row: int(row["step"]))
        final_rows.append({
            "mode": key[0], "seed": key[1], "final_step": int(final["step"]),
            "final_energy_drop": float(final["energy_drop"]),
            "final_kmc_time": float(final["kmc_time"]), "auc_energy_drop": compute_auc(series),
        })
    write_csv(final_rows, output_dir / "final_summary_per_seed.csv")

    summary = {
        "material_system": "fcc_ni_al_cr",
        "parameter_metadata": parameter_metadata,
        "config": {
            "bench_step": int(args.bench_step),
            "compare_seeds": [int(seed) for seed in args.compare_seeds],
            "compare_modes": list(args.compare_modes),
            "model_dir": str(args.model_dir or ""),
        },
        "per_mode": {},
        "paired_statistics": {},
    }
    for mode in args.compare_modes:
        rows = [row for row in final_rows if row["mode"] == mode]
        drops = np.asarray([row["final_energy_drop"] for row in rows], dtype=np.float64)
        aucs = np.asarray([row["auc_energy_drop"] for row in rows], dtype=np.float64)
        summary["per_mode"][mode] = {
            "final_energy_drop_mean": float(np.mean(drops)) if len(drops) else np.nan,
            "final_energy_drop_std": float(np.std(drops, ddof=1)) if len(drops) > 1 else 0.0,
            "auc_energy_drop_mean": float(np.mean(aucs)) if len(aucs) else np.nan,
            "auc_energy_drop_std": float(np.std(aucs, ddof=1)) if len(aucs) > 1 else 0.0,
            "n": len(rows),
        }

    if {"classic_kmc", "rl_stochastic"}.issubset(args.compare_modes):
        classic = {row["seed"]: row for row in final_rows if row["mode"] == "classic_kmc"}
        rl = {row["seed"]: row for row in final_rows if row["mode"] == "rl_stochastic"}
        common = sorted(set(classic).intersection(rl))
        paired_rows = []
        for seed in common:
            paired_rows.append({
                "seed": seed,
                "final_energy_drop_diff": rl[seed]["final_energy_drop"] - classic[seed]["final_energy_drop"],
                "auc_energy_drop_diff": rl[seed]["auc_energy_drop"] - classic[seed]["auc_energy_drop"],
            })
        write_csv(paired_rows, output_dir / "paired_seed_differences.csv")
        for metric in ("final_energy_drop_diff", "auc_energy_drop_diff"):
            values = np.asarray([row[metric] for row in paired_rows], dtype=np.float64)
            summary["paired_statistics"][metric] = {
                "mean": float(np.mean(values)) if len(values) else np.nan,
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "n": int(len(values)),
            }

    final_structure = []
    for (mode, seed), series in structure_groups.items():
        row = max(series, key=lambda item: int(item["step"]))
        final_structure.append(row)
    write_csv(final_structure, output_dir / "final_structure_summary_per_seed.csv")
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    return summary


def make_plots(energy_rows, structure_rows, args, output_dir):
    plot_step(energy_rows, "energy_drop", "Energy drop / eV",
              output_dir / "energy_drop_vs_step.png", args)
    plot_energy_vs_time(energy_rows, output_dir / "energy_drop_vs_kmc_time.png", args)
    plot_step(structure_rows, "largest_solute_cluster", "Largest solute cluster / atoms",
              output_dir / "largest_solute_cluster_vs_step.png", args)
    plot_step(structure_rows, "alpha_ni_al", "Warren-Cowley alpha(Ni-Al)",
              output_dir / "ni_al_sro_vs_step.png", args)

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.6))
    plot_step(energy_rows, "energy_drop", "Energy drop / eV", None, args, axes[0, 0])
    plot_energy_vs_time(energy_rows, None, args, axes[0, 1])
    plot_step(structure_rows, "largest_solute_cluster", "Largest solute cluster", None, args, axes[1, 0])
    plot_step(structure_rows, "alpha_ni_al", "Warren-Cowley alpha(Ni-Al)", None, args, axes[1, 1])
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle(args.plot_title, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_dir / "fcc_comparison_overview.png", dpi=args.plot_dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv):
    parser = add_compare_args(get_config())
    args = parser.parse_args(argv)
    args.use_wandb = False
    device = resolve_device(args)
    output_dir = Path(args.compare_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    probe = KMCEnvWrap(args)
    policy = load_policy(args, probe.action_space, device)
    parameter_metadata = dict(probe.env.parameter_metadata)
    all_energy_rows, all_structure_rows = [], []
    started = time.perf_counter()
    for seed in args.compare_seeds:
        snapshot = create_snapshot(args, int(seed), device)
        for mode in args.compare_modes:
            energy_rows, structure_rows = run_rollout(
                mode, args, device, snapshot, int(seed), policy, output_dir
            )
            all_energy_rows.extend(energy_rows)
            all_structure_rows.extend(structure_rows)
            print(f"[FCC Compare] seed={seed} mode={mode} final_drop={energy_rows[-1]['energy_drop']:.6f}")

    write_csv(all_energy_rows, output_dir / "merged_energy_series.csv")
    write_csv(all_structure_rows, output_dir / "merged_structure_series.csv")
    summary = summarize(all_energy_rows, all_structure_rows, args, output_dir, parameter_metadata)
    summary["runtime_seconds"] = float(time.perf_counter() - started)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=True)
    make_plots(all_energy_rows, all_structure_rows, args, output_dir)
    print(f"[FCC Compare] outputs: {output_dir}")


if __name__ == "__main__":
    main(sys.argv[1:])
