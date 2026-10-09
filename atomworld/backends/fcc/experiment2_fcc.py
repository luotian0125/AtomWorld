#!/usr/bin/env python
"""Resumable long-horizon FCC gamma-prime nucleation/growth experiment."""

import copy
import csv
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from atomworld.backends.fcc.compare_fcc_kmc import (
    choose_non_rl_action,
    choose_rl_action,
    create_snapshot,
    load_policy,
    resolve_device,
    restore_snapshot,
    write_csv,
)
from atomworld.backends.fcc.experiment1_fcc import _apply_checkpoint_metadata, _bootstrap_mean
from atomworld.backends.fcc.fcc_l12 import L12DetectorConfig, analyze_l12_lattice, validate_l12_detector
from atomworld.backends.fcc.config.parser import get_config
from atomworld.backends.fcc.envs.fcc_lattice import AL, CR, NI
from atomworld.backends.fcc.envs.kmc_env import KMCEnvWrap


MODE_LABELS = {
    "classic_kmc": "Classical FCC-LKMC",
    "rl_stochastic": "AtomWorld-FCC-Mem",
}
MODE_COLORS = {"classic_kmc": "#202124", "rl_stochastic": "#d45500"}
DEFAULT_DOSES = (0.0, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)
DEFAULT_SEEDS_20 = tuple(range(201, 211))


def add_exp2_args(parser):
    parser.add_argument("--exp2_output_dir", required=True)
    parser.add_argument("--exp2_sizes", type=int, nargs="+", default=[20])
    parser.add_argument("--exp2_seeds_20", type=int, nargs="+", default=list(DEFAULT_SEEDS_20))
    parser.add_argument("--exp2_seeds_40", type=int, nargs="+", default=[301])
    parser.add_argument("--exp2_modes", nargs="+", choices=tuple(MODE_LABELS),
                        default=list(MODE_LABELS))
    parser.add_argument("--dose_points", type=float, nargs="+", default=list(DEFAULT_DOSES))
    parser.add_argument("--energy_samples_per_dose", type=int, default=32)
    parser.add_argument("--resume_interval_steps", type=int, default=0,
                        help="Extra state-save interval; zero saves only at dose checkpoints.")
    parser.add_argument("--l12_radius_cells", type=int, default=1)
    parser.add_argument("--l12_order_threshold", type=float, default=0.30)
    parser.add_argument("--l12_min_al_fraction", type=float, default=0.11)
    parser.add_argument("--l12_max_al_fraction", type=float, default=0.35)
    parser.add_argument("--l12_min_ni_fraction", type=float, default=0.58)
    parser.add_argument("--l12_min_cluster_atoms", type=int, default=32)
    parser.add_argument("--skip_detector_preflight", action="store_true", default=False)
    return parser


def _l12_config(args):
    return L12DetectorConfig(
        radius_cells=args.l12_radius_cells,
        order_threshold=args.l12_order_threshold,
        min_al_fraction=args.l12_min_al_fraction,
        max_al_fraction=args.l12_max_al_fraction,
        min_ni_fraction=args.l12_min_ni_fraction,
        min_cluster_atoms=args.l12_min_cluster_atoms,
    )


def _composition_for_size(size):
    total_sites = 4 * int(size) ** 3
    al = int(round(0.075 * total_sites))
    cr = int(round(0.085 * total_sites))
    return total_sites, al, cr


def _append_csv(row, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _truncate_csv_after_step(path, resume_step):
    """Remove stale rows beyond the authoritative resumable state."""
    path = Path(path)
    if not path.exists() or path.stat().st_size == 0:
        return 0
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        rows = list(reader)
    if not fieldnames or "step" not in fieldnames:
        return 0
    kept = [row for row in rows if int(float(row["step"])) <= int(resume_step)]
    removed = len(rows) - len(kept)
    if removed <= 0:
        return 0
    backup = Path(str(path) + ".pre_resume_backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(kept)
    os.replace(temporary, path)
    return removed


def _atomic_species(env):
    coords = np.asarray(env.coords, dtype=np.int32)
    return np.asarray(env.lattice[tuple(coords.T)], dtype=np.int8)


def _save_atomic_snapshot(env, path, step, dose):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        species=_atomic_species(env),
        lattice_size=np.asarray(env.size, dtype=np.int32),
        step=np.asarray(step, dtype=np.int64),
        event_dose=np.asarray(dose, dtype=np.float64),
        kmc_time=np.asarray(env.time, dtype=np.float64),
        total_energy=np.asarray(env.energy_last, dtype=np.float64),
        species_order=np.asarray(["Ni", "Al", "Cr", "Vacancy"]),
        coordinate_order=np.asarray("fcc_lattice.Lattice.generate_lattice_coordinates"),
    )


def _slim_env_state(env):
    return {
        "schema_version": 2,
        "material_system": "fcc_ni_al_cr",
        "parameter_metadata": dict(env.parameter_metadata),
        "lattice": np.asarray(env.lattice, dtype=np.int32).copy(),
        "vacancy_pos": np.asarray(env.vacancy_pos, dtype=np.int32).copy(),
        "solute_pos": np.asarray(env.solute_pos, dtype=np.int32).copy(),
        "time": float(env.time),
        "energy_last": float(env.energy_last),
        "energy_history": [float(env.energy_last)],
        "time_history": [float(env.time)],
        "action_histories": np.asarray(env.action_history_array, dtype=np.int32).tolist(),
    }


def _save_resume(path, env, step, rnn_states, previous_stable_mask, status):
    payload = {
        "schema_version": 1,
        "step": int(step),
        "env_state": _slim_env_state(env),
        "rnn_states": np.asarray(rnn_states, dtype=np.float32),
        "previous_stable_mask": previous_stable_mask,
        "status": dict(status),
        "numpy_rng": np.random.get_state(),
        "python_rng": random.getstate(),
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    temporary = Path(str(path) + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_resume(path, args, device):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    env = KMCEnvWrap(args)
    env.set_state_dict(payload["env_state"])
    np.random.set_state(payload["numpy_rng"])
    random.setstate(payload["python_rng"])
    torch.set_rng_state(payload["torch_rng"])
    if device.type == "cuda" and payload.get("cuda_rng") is not None:
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    obs = env.env.get_combined_obs().astype(np.float32)
    return (
        env, obs, int(payload["step"]), np.asarray(payload["rnn_states"], dtype=np.float32),
        payload.get("previous_stable_mask"), dict(payload.get("status", {})),
    )


def _write_status(path, status):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(status, indent=2, allow_nan=True), encoding="utf-8")
    os.replace(temporary, path)


def _checkpoint_structure(env, config, step, total_sites, mode, seed, run_dir,
                          previous_stable_mask, status, wall_seconds):
    dose = float(step / total_sites)
    metrics, masks = analyze_l12_lattice(env.lattice, config, return_masks=True)
    current_stable = masks["stable_mask"]
    overlap = 0 if previous_stable_mask is None else int(
        np.count_nonzero(current_stable & np.asarray(previous_stable_mask, dtype=bool))
    )
    persistent = int(overlap >= config.min_cluster_cells)
    if persistent and status.get("first_persistent_nucleus_dose") is None:
        status["first_persistent_nucleus_dose"] = float(status.get("previous_dose", dose))
        status["first_persistent_nucleus_step"] = int(status.get("previous_step", step))
    row = {
        "size": int(round((total_sites / 4) ** (1.0 / 3.0))),
        "mode": mode, "seed": int(seed), "step": int(step), "event_dose": dose,
        "kmc_time": float(env.time), "total_energy": float(env.energy_last),
        "wall_seconds": float(wall_seconds), "persistent_gamma_prime": persistent,
        "persistent_overlap_cells": overlap, **metrics,
    }
    _append_csv(row, run_dir / "structure_series.csv")
    for component_cells in masks["stable_component_sizes_cells"]:
        _append_csv({
            "size": row["size"], "mode": mode, "seed": int(seed), "step": int(step),
            "event_dose": dose, "particle_cells": int(component_cells),
            "particle_atoms": int(component_cells) * 4,
            "equivalent_radius_cells": float((3.0 * component_cells / (4.0 * np.pi)) ** (1.0 / 3.0)),
        }, run_dir / "particle_size_distribution.csv")
    _save_atomic_snapshot(env, run_dir / "snapshots" / f"dose_{dose:07.3f}.npz", step, dose)
    status["previous_dose"] = dose
    status["previous_step"] = int(step)
    return current_stable, row


def run_one(mode, args, device, policy, size, seed, config, output_root):
    total_sites, al_count, cr_count = _composition_for_size(size)
    local_args = copy.deepcopy(args)
    local_args.lattice_size = [size, size, size]
    local_args.lattice_al_nums = al_count
    local_args.lattice_cr_nums = cr_count
    local_args.lattice_v_nums = 1
    local_args.skip_policy_context = mode == "classic_kmc"
    targets = sorted({int(round(float(dose) * total_sites)) for dose in args.dose_points})
    targets = [target for target in targets if target >= 0]
    max_step = max(targets)
    # The actor does not consume centralized system statistics during rollout.
    # Cache them instead of rebuilding a global cKDTree at every event.
    local_args.system_stats_interval = max(int(local_args.system_stats_interval), max_step + 1)
    run_dir = output_root / f"size_{size}" / f"seed_{seed}" / mode
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    resume_path = run_dir / "resume.pt"
    if status_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("complete") and int(status.get("step", -1)) >= max_step:
            print(f"[Exp2] skip completed size={size} seed={seed} mode={mode}", flush=True)
            return
    if resume_path.exists():
        env, obs, step, rnn_states, previous_stable, status = _load_resume(
            resume_path, local_args, device
        )
        for csv_name in (
            "energy_series.csv", "structure_series.csv", "particle_size_distribution.csv"
        ):
            removed = _truncate_csv_after_step(run_dir / csv_name, step)
            if removed:
                print(
                    f"[Exp2] removed {removed} stale rows beyond resume step={step} "
                    f"from {run_dir / csv_name}", flush=True,
                )
        print(f"[Exp2] resume size={size} seed={seed} mode={mode} step={step}", flush=True)
    else:
        initial_state = create_snapshot(local_args, int(seed), device)
        env, obs = restore_snapshot(local_args, initial_state, int(seed), device)
        step = 0
        rnn_states = np.zeros(
            (local_args.lattice_v_nums, local_args.recurrent_N, local_args.hidden_size),
            dtype=np.float32,
        )
        previous_stable = None
        status = {
            "complete": False, "size": int(size), "seed": int(seed), "mode": mode,
            "total_sites": int(total_sites), "al_count": int(al_count), "cr_count": int(cr_count),
            "first_persistent_nucleus_dose": None,
        }
    rates = np.asarray(env.get_base_rates_np(copy=False), dtype=np.float32)
    features = np.asarray(env.get_event_features_np(copy=False), dtype=np.float32)
    tokens, token_masks = env.get_sparse_defect_context_np(copy=False)
    started = time.perf_counter()
    wall_offset = float(status.get("wall_seconds", 0.0))
    energy_interval = max(1, total_sites // max(1, int(args.energy_samples_per_dose)))
    target_set = set(targets)

    if step == 0 and not (run_dir / "structure_series.csv").exists():
        previous_stable, _ = _checkpoint_structure(
            env.env, config, 0, total_sites, mode, seed, run_dir, None, status, wall_offset
        )
        _append_csv({
            "size": size, "mode": mode, "seed": seed, "step": 0, "event_dose": 0.0,
            "total_energy": float(env.env.energy_last), "energy_drop": 0.0,
            "kmc_time": float(env.env.time), "wall_seconds": wall_offset,
            "cumulative_ni_moves": 0, "cumulative_al_moves": 0, "cumulative_cr_moves": 0,
        }, run_dir / "energy_series.csv")
    initial_energy = float(status.get("initial_energy", env.env.energy_last))
    status["initial_energy"] = initial_energy
    counts = dict(status.get("event_counts", {"ni": 0, "al": 0, "cr": 0}))

    while step < max_step:
        if mode == "rl_stochastic":
            action, rnn_states, _ = choose_rl_action(
                policy, False, obs, rnn_states, rates, features, tokens, token_masks
            )
        else:
            action = choose_non_rl_action("classic_kmc", rates, features)
        obs, _, _, _, dones, info = env.step(action)
        step += 1
        moving = int(info.get("moving_type", -1))
        if moving == NI:
            counts["ni"] += 1
        elif moving == AL:
            counts["al"] += 1
        elif moving == CR:
            counts["cr"] += 1
        rates = np.asarray(info["next_base_rates"], dtype=np.float32)
        features = np.asarray(info["next_event_features"], dtype=np.float32)
        tokens = np.asarray(info["next_sparse_defect_tokens"], dtype=np.float32)
        token_masks = np.asarray(info["next_sparse_defect_masks"], dtype=np.float32)
        wall_seconds = wall_offset + time.perf_counter() - started

        if step % energy_interval == 0 or step in target_set or bool(np.asarray(dones).reshape(-1)[0]):
            _append_csv({
                "size": size, "mode": mode, "seed": seed, "step": step,
                "event_dose": float(step / total_sites),
                "total_energy": float(info["total_energy"]),
                "energy_drop": float(initial_energy - float(info["total_energy"])),
                "kmc_time": float(info["kmc_time"]), "wall_seconds": float(wall_seconds),
                "cumulative_ni_moves": int(counts["ni"]),
                "cumulative_al_moves": int(counts["al"]),
                "cumulative_cr_moves": int(counts["cr"]),
            }, run_dir / "energy_series.csv")
            # Avoid retaining millions of boxed Python floats in long runs.
            env.env.energy_history = [float(env.env.energy_last)]
            env.env.time_history = [float(env.env.time)]

        save_resume = args.resume_interval_steps > 0 and step % args.resume_interval_steps == 0
        if step in target_set:
            previous_stable, structure_row = _checkpoint_structure(
                env.env, config, step, total_sites, mode, seed, run_dir,
                previous_stable, status, wall_seconds,
            )
            print(
                f"[Exp2] size={size} seed={seed} mode={mode} dose={step / total_sites:.3f} "
                f"clusters={structure_row['gamma_prime_cluster_count']} "
                f"vf={structure_row['gamma_prime_volume_fraction']:.5f}", flush=True,
            )
            save_resume = True
        status.update({
            "step": int(step), "wall_seconds": float(wall_seconds), "event_counts": counts,
            "complete": bool(step >= max_step or bool(np.asarray(dones).reshape(-1)[0])),
        })
        if save_resume or status["complete"]:
            _save_resume(resume_path, env.env, step, rnn_states, previous_stable, status)
            _write_status(status_path, status)
        if bool(np.asarray(dones).reshape(-1)[0]):
            break

    status["complete"] = bool(step >= max_step)
    status["step"] = int(step)
    status["wall_seconds"] = float(wall_offset + time.perf_counter() - started)
    status["event_counts"] = counts
    _save_resume(resume_path, env.env, step, rnn_states, previous_stable, status)
    _write_status(status_path, status)


def _read_all_structure(output_root):
    rows = []
    for path in output_root.glob("size_*/seed_*/*/structure_series.csv"):
        with path.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                converted = dict(row)
                for key, value in list(converted.items()):
                    try:
                        converted[key] = float(value)
                    except (TypeError, ValueError):
                        pass
                rows.append(converted)
    return rows


def _first_row_at_or_above(rows, metric, target):
    for row in sorted(rows, key=lambda item: float(item["event_dose"])):
        if float(row[metric]) >= target:
            return row
    return None


def _first_persistent_dose_from_status(output_root, size, seed, mode):
    status_path = output_root / f"size_{size}" / f"seed_{seed}" / mode / "status.json"
    if not status_path.exists():
        return np.nan
    status = json.loads(status_path.read_text(encoding="utf-8"))
    value = status.get("first_persistent_nucleus_dose")
    return float(value) if value is not None else np.nan


def make_derived_tables(output_root):
    rows = _read_all_structure(output_root)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["size"]), int(row["seed"]), str(row["mode"]))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda item: float(item["event_dose"]))

    final_rows, acceleration_rows, coarsening_rows = [], [], []
    for (size, seed, mode), series in sorted(grouped.items()):
        final = series[-1]
        first_persistent = _first_persistent_dose_from_status(
            output_root, size, seed, mode
        )
        final_rows.append({
            "size": size, "seed": seed, "mode": mode,
            "first_persistent_nucleus_dose": first_persistent,
            "final_dose": float(final["event_dose"]),
            "final_gamma_prime_volume_fraction": float(final["gamma_prime_volume_fraction"]),
            "final_gamma_prime_cluster_count": int(final["gamma_prime_cluster_count"]),
            "final_gamma_prime_mean_radius_cells": float(final["gamma_prime_mean_radius_cells"]),
            "final_l12_order_mean": float(final["l12_order_mean"]),
        })
        fit_rows = [
            row for row in series
            if int(row["gamma_prime_cluster_count"]) > 0
            and float(row["event_dose"]) >= (first_persistent if np.isfinite(first_persistent) else np.inf)
        ]
        if len(fit_rows) >= 3:
            x = np.asarray([float(row["event_dose"]) for row in fit_rows])
            y = np.asarray([float(row["gamma_prime_mean_radius_cubed"]) for row in fit_rows])
            slope, intercept = np.polyfit(x, y, 1)
            prediction = slope * x + intercept
            denominator = float(np.sum((y - y.mean()) ** 2))
            r2 = 1.0 - float(np.sum((y - prediction) ** 2)) / denominator if denominator > 0 else np.nan
        else:
            slope = intercept = r2 = np.nan
        coarsening_rows.append({
            "size": size, "seed": seed, "mode": mode, "n_fit_points": len(fit_rows),
            "Kq_radius_cubed_per_dose": float(slope), "intercept": float(intercept),
            "r_squared": float(r2),
        })

    keys = sorted({(int(row["size"]), int(row["seed"])) for row in rows})
    for size, seed in keys:
        classic = grouped.get((size, seed, "classic_kmc"), [])
        learned = grouped.get((size, seed, "rl_stochastic"), [])
        if not classic or not learned:
            continue
        classic_final = classic[-1]
        target_vf = float(classic_final["gamma_prime_volume_fraction"])
        classic_hit = _first_row_at_or_above(classic, "gamma_prime_volume_fraction", target_vf) if target_vf > 0 else None
        learned_hit = _first_row_at_or_above(learned, "gamma_prime_volume_fraction", target_vf) if target_vf > 0 else None
        q_classic = float(classic_hit["event_dose"]) if classic_hit else np.nan
        q_learned = float(learned_hit["event_dose"]) if learned_hit else np.nan
        qn_classic = _first_persistent_dose_from_status(
            output_root, size, seed, "classic_kmc"
        )
        qn_learned = _first_persistent_dose_from_status(
            output_root, size, seed, "rl_stochastic"
        )
        acceleration_rows.append({
            "size": size, "seed": seed, "target_classic_final_volume_fraction": target_vf,
            "classic_target_dose": q_classic, "atomworld_target_dose": q_learned,
            "event_speedup_to_classic_final_structure": q_classic / q_learned
                if np.isfinite(q_classic) and np.isfinite(q_learned) and q_learned > 0 else np.nan,
            "classic_persistent_nucleus_dose": qn_classic,
            "atomworld_persistent_nucleus_dose": qn_learned,
            "nucleation_event_speedup": qn_classic / qn_learned
                if np.isfinite(qn_classic) and np.isfinite(qn_learned) and qn_learned > 0 else np.nan,
            "matched_progress_l12_order_abs_error": abs(
                float(learned_hit["l12_order_mean"]) - float(classic_hit["l12_order_mean"])
            ) if learned_hit and classic_hit else np.nan,
            "matched_progress_al_fraction_abs_error": abs(
                float(learned_hit["gamma_prime_al_fraction"]) - float(classic_hit["gamma_prime_al_fraction"])
            ) if learned_hit and classic_hit else np.nan,
            "matched_progress_cr_fraction_abs_error": abs(
                float(learned_hit["gamma_prime_cr_fraction"]) - float(classic_hit["gamma_prime_cr_fraction"])
            ) if learned_hit and classic_hit else np.nan,
        })
    write_csv(final_rows, output_root / "long_horizon_summary_per_run.csv")
    write_csv(acceleration_rows, output_root / "event_acceleration_paired.csv")
    write_csv(coarsening_rows, output_root / "coarsening_fit_per_run.csv")
    rng = np.random.default_rng(20260902)
    acceleration_summary = []
    for size in sorted({int(row["size"]) for row in acceleration_rows}):
        subset = [row for row in acceleration_rows if int(row["size"]) == size]
        for metric in ("event_speedup_to_classic_final_structure", "nucleation_event_speedup"):
            stats = _bootstrap_mean([row[metric] for row in subset], 5000, rng)
            acceleration_summary.append({"size": size, "metric": metric, **stats})
    write_csv(acceleration_summary, output_root / "event_acceleration_summary.csv")


def make_summary_plots(output_root):
    rows = _read_all_structure(output_root)
    if not rows:
        return
    metrics = (
        ("gamma_prime_volume_fraction", "$\\gamma'$ volume fraction"),
        ("gamma_prime_cluster_count", "$\\gamma'$ particle count"),
        ("gamma_prime_mean_radius_cubed", "$\\langle R^3\\rangle$ / cell$^3$"),
        ("l12_order_mean", "Local L1$_2$ order"),
    )
    for size in sorted({int(row["size"]) for row in rows}):
        fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.5))
        for ax, (metric, ylabel) in zip(axes.flat, metrics):
            for mode in MODE_LABELS:
                subset = [row for row in rows if int(row["size"]) == size and row["mode"] == mode]
                doses = sorted({float(row["event_dose"]) for row in subset})
                means, sems = [], []
                for dose in doses:
                    values = np.asarray([
                        float(row[metric]) for row in subset
                        if np.isclose(float(row["event_dose"]), dose)
                    ])
                    means.append(float(np.mean(values)))
                    sems.append(float(np.std(values, ddof=1) / np.sqrt(values.size)) if values.size > 1 else 0.0)
                means, sems = np.asarray(means), np.asarray(sems)
                ax.plot(doses, means, color=MODE_COLORS[mode], label=MODE_LABELS[mode], lw=2)
                ax.fill_between(doses, means - sems, means + sems, color=MODE_COLORS[mode], alpha=0.15)
            ax.set_xlabel("Event dose $q=N_{events}/N_{sites}$")
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
        fig.suptitle(f"Alloy A, 873 K, conventional FCC {size}$^3$", y=0.995)
        fig.tight_layout(rect=(0, 0, 1, 0.95))
        fig.savefig(output_root / f"gamma_prime_evolution_size_{size}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def main(argv=None):
    args = add_exp2_args(get_config()).parse_args(argv)
    args.use_wandb = False
    args.compare_modes = ["rl_stochastic"]
    output_root = Path(args.exp2_output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args)
    checkpoint_metadata = _apply_checkpoint_metadata(args, args.model_dir)
    config = _l12_config(args)
    validation = validate_l12_detector(config)
    (output_root / "l12_detector_preflight.json").write_text(
        json.dumps(validation, indent=2, allow_nan=True), encoding="utf-8"
    )
    if not args.skip_detector_preflight and not validation["passed"]:
        raise RuntimeError(
            "L1_2 detector preflight failed; inspect l12_detector_preflight.json or explicitly use "
            "--skip_detector_preflight for threshold-development runs."
        )

    probe = KMCEnvWrap(args)
    policy = load_policy(args, probe.action_space, device) if "rl_stochastic" in args.exp2_modes else None
    manifest = {
        "experiment": "exp2_gamma_prime_long_horizon",
        "material": "Ni-7.5Al-8.5Cr, FCC Mao-LKMC1", "temperature_K": float(args.temperature),
        "sizes": list(map(int, args.exp2_sizes)), "dose_points": list(map(float, args.dose_points)),
        "seeds": {
            "20": list(map(int, args.exp2_seeds_20)),
            "40": list(map(int, args.exp2_seeds_40)),
        },
        "modes": list(args.exp2_modes), "checkpoint": str(Path(args.model_dir).resolve()),
        "checkpoint_metadata": checkpoint_metadata, "l12_detector": config.__dict__,
        "clock_limitation": "Acceleration is reported against event dose, not physical seconds; AtomWorld lacks unbiased Poisson-network clock reconstruction.",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for size in args.exp2_sizes:
        seeds = args.exp2_seeds_20 if int(size) == 20 else args.exp2_seeds_40
        for seed in seeds:
            for mode in args.exp2_modes:
                run_one(mode, args, device, policy, int(size), int(seed), config, output_root)
    make_derived_tables(output_root)
    make_summary_plots(output_root)
    print(f"[Exp2] complete: {output_root}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
