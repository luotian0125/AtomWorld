#!/usr/bin/env python
"""In-domain FCC Ni-Al-Cr main-table experiment with paired statistics."""

import copy
import csv
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from atomworld.backends.fcc.compare_fcc_kmc import (
    add_compare_args,
    compute_auc,
    create_snapshot,
    load_policy,
    resolve_device,
    run_rollout,
    write_csv,
)
from atomworld.backends.fcc.config.parser import get_config
from atomworld.backends.fcc.envs.kmc_env import KMCEnvWrap


BASELINE_MODES = ("classic_kmc", "uniform_random", "rate_greedy", "energy_greedy")
MEM_MODE = "atomworld_mem"
SNAPSHOT_MODE = "snapshot_atomworld"
TABLE_MODES = (*BASELINE_MODES, SNAPSHOT_MODE, MEM_MODE)
DISPLAY_NAMES = {
    "classic_kmc": "Classical FCC-LKMC",
    "uniform_random": "Uniform-random KMC",
    "rate_greedy": "Rate-greedy KMC",
    "energy_greedy": "Energy-greedy KMC",
    SNAPSHOT_MODE: "Snapshot-only AtomWorld-FCC",
    MEM_MODE: "AtomWorld-FCC-Mem",
}
PLOT_COLORS = {
    "classic_kmc": "#202124",
    "uniform_random": "#7f8c8d",
    "rate_greedy": "#6a5acd",
    "energy_greedy": "#2e8b57",
    SNAPSHOT_MODE: "#2878b5",
    MEM_MODE: "#d45500",
}


def add_exp1_args(parser):
    parser.add_argument("--snapshot_model_dir", required=True)
    parser.add_argument("--exp1_bootstrap_samples", type=int, default=5000)
    parser.add_argument("--exp1_bootstrap_seed", type=int, default=20260901)
    parser.add_argument("--arrival_persistence_points", type=int, default=2)
    return parser


def _apply_checkpoint_metadata(args, model_dir):
    metadata_path = Path(model_dir) / "metadata.json"
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    fields = (
        "policy_logit_scale", "rate_prior_scale", "energy_prior_scale",
    )
    for field in fields:
        if field not in metadata:
            continue
        value = float(metadata[field])
        setattr(args, field, value)
        setattr(args, f"{field}_start", value)
        setattr(args, f"{field}_end", value)
        setattr(args, f"{field}_schedule", "constant")
    return metadata


def _rename_rows(rows, mode):
    renamed = []
    for row in rows:
        row = dict(row)
        row["mode"] = mode
        renamed.append(row)
    return renamed


def _group(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(str(row["mode"]), int(row["seed"]))].append(row)
    for key in grouped:
        grouped[key].sort(key=lambda row: int(row["step"]))
    return grouped


def _first_hit(series, target, x_key):
    for left, right in zip(series[:-1], series[1:]):
        y0, y1 = float(left["energy_drop"]), float(right["energy_drop"])
        if y0 >= target:
            return float(left[x_key])
        if y1 >= target and y1 != y0:
            fraction = float(np.clip((target - y0) / (y1 - y0), 0.0, 1.0))
            return float(left[x_key]) + fraction * (float(right[x_key]) - float(left[x_key]))
    if series and float(series[-1]["energy_drop"]) >= target:
        return float(series[-1][x_key])
    return np.nan


def _aligned_rows(series, reference):
    series_map = {int(row["step"]): row for row in series}
    reference_map = {int(row["step"]): row for row in reference}
    common = sorted(set(series_map).intersection(reference_map))
    return [(series_map[step], reference_map[step]) for step in common]


def _dynamics_error(series, reference):
    pairs = _aligned_rows(series, reference)
    if not pairs:
        return np.nan
    errors = []
    for row, ref in pairs:
        step = max(int(row["step"]), 1)
        for key in ("cumulative_ni_moves", "cumulative_al_moves", "cumulative_cr_moves",
                    "cumulative_downhill_moves"):
            errors.append(abs(float(row[key]) - float(ref[key])) / step)
    return float(np.mean(errors))


def _structure_error(series, reference, total_sites):
    pairs = _aligned_rows(series, reference)
    if not pairs:
        return np.nan
    specs = (
        ("alpha_ni_al", 1.0),
        ("alpha_ni_cr", 1.0),
        ("alpha_al_cr", 1.0),
        ("l12_order_mean", 1.0),
        ("gamma_prime_volume_fraction", 1.0),
        ("gamma_prime_largest_atoms", float(total_sites)),
    )
    values = []
    for row, ref in pairs:
        for key, scale in specs:
            left, right = float(row[key]), float(ref[key])
            if np.isfinite(left) and np.isfinite(right):
                values.append(abs(left - right) / scale)
    return float(np.mean(values)) if values else np.nan


def _arrival_hit(series, persistence):
    run = 0
    for row in series:
        run = run + 1 if int(float(row.get("gamma_prime_cluster_count", 0))) > 0 else 0
        if run >= persistence:
            return 1.0
    return 0.0


def _bootstrap_mean(values, samples, rng):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"mean": np.nan, "std": np.nan, "ci95_low": np.nan, "ci95_high": np.nan, "n": 0}
    if values.size == 1 or samples <= 0:
        low = high = float(values.mean())
    else:
        indices = rng.integers(0, values.size, size=(samples, values.size))
        boot = values[indices].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci95_low": float(low),
        "ci95_high": float(high),
        "n": int(values.size),
    }


def build_main_table(energy_rows, structure_rows, args, output_dir):
    energy = _group(energy_rows)
    structure = _group(structure_rows)
    seeds = [int(seed) for seed in args.compare_seeds]
    total_sites = 4 * int(np.prod(args.lattice_size))
    per_seed = []
    for seed in seeds:
        classic_energy = energy[("classic_kmc", seed)]
        classic_structure = structure[("classic_kmc", seed)]
        classic_auc = compute_auc(classic_energy)
        target = float(classic_energy[-1]["energy_drop"])
        classic_target_wall = _first_hit(classic_energy, target, "wall_seconds")
        for mode in TABLE_MODES:
            mode_energy = energy[(mode, seed)]
            mode_structure = structure[(mode, seed)]
            mode_auc = compute_auc(mode_energy)
            target_wall = _first_hit(mode_energy, target, "wall_seconds")
            per_seed.append({
                "mode": mode,
                "method": DISPLAY_NAMES[mode],
                "seed": seed,
                "energy_auc_ratio": mode_auc / classic_auc if classic_auc != 0.0 else np.nan,
                "final_energy_drop": float(mode_energy[-1]["energy_drop"]),
                "e2e_speedup": classic_target_wall / target_wall
                    if np.isfinite(target_wall) and target_wall > 0.0 else np.nan,
                "clock_error": 0.0 if mode == "classic_kmc" else np.nan,
                "dynamics_error": 0.0 if mode == "classic_kmc" else _dynamics_error(mode_energy, classic_energy),
                "structure_error": 0.0 if mode == "classic_kmc" else _structure_error(
                    mode_structure, classic_structure, total_sites
                ),
                "arrival_hit": _arrival_hit(mode_structure, args.arrival_persistence_points),
                "target_energy_drop": target,
                "target_reached": int(np.isfinite(target_wall)),
            })
    write_csv(per_seed, output_dir / "main_table_per_seed.csv")

    rng = np.random.default_rng(args.exp1_bootstrap_seed)
    aggregate = []
    metrics = (
        "energy_auc_ratio", "final_energy_drop", "e2e_speedup", "dynamics_error",
        "structure_error", "arrival_hit",
    )
    for mode in TABLE_MODES:
        rows = [row for row in per_seed if row["mode"] == mode]
        row = {
            "mode": mode,
            "method": DISPLAY_NAMES[mode],
            "clock": "Exact" if mode == "classic_kmc" else (
                "base-rate clock (biased path)" if mode in BASELINE_MODES else "N/A"
            ),
        }
        for metric in metrics:
            stats = _bootstrap_mean(
                [item[metric] for item in rows], args.exp1_bootstrap_samples, rng
            )
            for key, value in stats.items():
                row[f"{metric}_{key}"] = value
        aggregate.append(row)
    write_csv(aggregate, output_dir / "main_table.csv")

    paired = []
    classic_by_seed = {row["seed"]: row for row in per_seed if row["mode"] == "classic_kmc"}
    for row in per_seed:
        if row["mode"] == "classic_kmc":
            continue
        ref = classic_by_seed[row["seed"]]
        paired.append({
            "mode": row["mode"], "seed": row["seed"],
            "energy_auc_ratio_minus_classic": row["energy_auc_ratio"] - ref["energy_auc_ratio"],
            "final_energy_drop_minus_classic": row["final_energy_drop"] - ref["final_energy_drop"],
            "arrival_hit_minus_classic": row["arrival_hit"] - ref["arrival_hit"],
        })
    write_csv(paired, output_dir / "paired_seed_differences.csv")
    _write_latex_table(aggregate, output_dir / "main_table.tex")
    return aggregate


def _fmt(row, metric, suffix=""):
    mean = row.get(f"{metric}_mean", np.nan)
    std = row.get(f"{metric}_std", np.nan)
    if not np.isfinite(mean):
        return "N/A"
    return f"${mean:.3f} \\pm {std:.3f}{suffix}$"


def _write_latex_table(rows, path):
    lines = [
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Method & Energy AUC $\uparrow$ & E2E speedup $\uparrow$ & Clock & Dynamics $\downarrow$ & Structure $\downarrow$ & Arrival hit $\uparrow$ \\",
        r"\midrule",
    ]
    times_suffix = r"\times"
    for row in rows:
        arrival = row.get("arrival_hit_mean", np.nan)
        arrival_text = "N/A" if not np.isfinite(arrival) else f"{100.0 * arrival:.1f}\\%"
        lines.append(
            f"{row['method']} & {_fmt(row, 'energy_auc_ratio')} & {_fmt(row, 'e2e_speedup', times_suffix)} & "
            f"{row['clock']} & {_fmt(row, 'dynamics_error')} & {_fmt(row, 'structure_error')} & {arrival_text} \\\\"
        )
    lines.extend([
        r"\bottomrule", r"\end{tabular}", "",
        r"% Non-classical clocks are not unbiased passive-KMC physical times.",
        r"% AtomWorld rows therefore report N/A until Poisson-network clock reconstruction is implemented.",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_series(rows, metric, ylabel, output):
    grouped = _group(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for mode in TABLE_MODES:
        series = [value for (name, _), value in grouped.items() if name == mode]
        if not series:
            continue
        common_steps = sorted(set.intersection(*[{int(row["step"]) for row in item} for item in series]))
        values = np.asarray([
            [{int(row["step"]): float(row[metric]) for row in item}[step] for step in common_steps]
            for item in series
        ])
        mean = np.nanmean(values, axis=0)
        sem = np.nanstd(values, axis=0, ddof=1) / math.sqrt(values.shape[0]) if values.shape[0] > 1 else np.zeros_like(mean)
        ax.plot(common_steps, mean, label=DISPLAY_NAMES[mode], color=PLOT_COLORS[mode], lw=2)
        ax.fill_between(common_steps, mean - sem, mean + sem, color=PLOT_COLORS[mode], alpha=0.14)
    ax.set_xlabel("Executed vacancy events")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.2)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(argv=None):
    parser = add_exp1_args(add_compare_args(get_config()))
    args = parser.parse_args(argv)
    args.use_wandb = False
    args.compare_modes = list(BASELINE_MODES) + ["rl_stochastic"]
    output_dir = Path(args.compare_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args)

    mem_args = copy.deepcopy(args)
    mem_args.model_dir = str(Path(args.model_dir).resolve())
    mem_metadata = _apply_checkpoint_metadata(mem_args, mem_args.model_dir)
    probe = KMCEnvWrap(mem_args)
    mem_policy = load_policy(mem_args, probe.action_space, device)

    snapshot_args = copy.deepcopy(args)
    snapshot_args.model_dir = str(Path(args.snapshot_model_dir).resolve())
    snapshot_args.compare_modes = ["rl_stochastic"]
    snapshot_args.use_action_history = False
    snapshot_args.use_recurrent_memory = False
    snapshot_args.use_sparse_context = True
    snapshot_metadata = _apply_checkpoint_metadata(snapshot_args, snapshot_args.model_dir)
    snapshot_policy = load_policy(snapshot_args, probe.action_space, device)

    all_energy, all_structure = [], []
    started = time.perf_counter()
    for seed in args.compare_seeds:
        initial_state = create_snapshot(mem_args, int(seed), device)
        for mode in BASELINE_MODES:
            energy, structure = run_rollout(
                mode, mem_args, device, initial_state, int(seed), None,
                output_dir / "raw" / "baselines",
            )
            all_energy.extend(energy)
            all_structure.extend(structure)
        energy, structure = run_rollout(
            "rl_stochastic", mem_args, device, initial_state, int(seed), mem_policy,
            output_dir / "raw" / MEM_MODE,
        )
        all_energy.extend(_rename_rows(energy, MEM_MODE))
        all_structure.extend(_rename_rows(structure, MEM_MODE))
        energy, structure = run_rollout(
            "rl_stochastic", snapshot_args, device, initial_state, int(seed), snapshot_policy,
            output_dir / "raw" / SNAPSHOT_MODE,
        )
        all_energy.extend(_rename_rows(energy, SNAPSHOT_MODE))
        all_structure.extend(_rename_rows(structure, SNAPSHOT_MODE))
        print(f"[Exp1] completed paired seed {seed}", flush=True)

    write_csv(all_energy, output_dir / "merged_energy_series.csv")
    write_csv(all_structure, output_dir / "merged_structure_series.csv")
    table = build_main_table(all_energy, all_structure, args, output_dir)
    _plot_series(all_energy, "energy_drop", "Energy drop / eV", output_dir / "energy_drop_vs_events.png")
    _plot_series(all_structure, "l12_order_mean", "Local L1$_2$ order", output_dir / "l12_order_vs_events.png")
    _plot_series(all_structure, "gamma_prime_volume_fraction", "Detected $\\gamma'$ volume fraction", output_dir / "gamma_prime_fraction_vs_events.png")

    manifest = {
        "experiment": "exp1_in_domain_main_table",
        "material": "Ni-7.5Al-8.5Cr, FCC Mao-LKMC1",
        "temperature_K": float(args.temperature),
        "lattice_size_conventional_cells": list(map(int, args.lattice_size)),
        "site_count": 4 * int(np.prod(args.lattice_size)),
        "event_budget": int(args.bench_step),
        "held_out_seeds": list(map(int, args.compare_seeds)),
        "memory_checkpoint": mem_args.model_dir,
        "memory_checkpoint_metadata": mem_metadata,
        "snapshot_checkpoint": snapshot_args.model_dir,
        "snapshot_checkpoint_metadata": snapshot_metadata,
        "snapshot_definition": {
            "use_action_history": False, "use_recurrent_memory": False,
            "use_sparse_context": True,
        },
        "clock_limitation": "Only Classical FCC-LKMC has an exact passive-KMC path clock. AtomWorld physical-time clock is N/A without Poisson-network reconstruction.",
        "bootstrap_samples": int(args.exp1_bootstrap_samples),
        "runtime_seconds": float(time.perf_counter() - started),
        "table_rows": table,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(f"[Exp1] complete: {output_dir}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
