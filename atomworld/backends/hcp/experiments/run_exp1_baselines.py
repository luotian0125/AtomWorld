#!/usr/bin/env python3
"""Incremental accelerated-baseline experiments against a verified frozen Exp1."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from atomworld.backends.hcp.experiments.freeze_exp1 import sha256, verify_freeze
from atomworld.backends.hcp.physics.accelerated_baselines import AcceleratedHCP
from atomworld.backends.hcp.training.exp1_metrics import (
    PersistentPrecursorDefinition, evaluate_exp1_episode, precursor_coordinate,
    summarize_exp1_episodes,
)
from atomworld.backends.hcp.training.factory import HCPEnvironmentConfig, make_hcp_policy_environment
from atomworld.backends.hcp.training.reward import HCPRewardConfig, HCPRewardModel


ROOT = Path(__file__).resolve().parents[1]
METHODS = ("lkmc", "amc", "mrm", "askmc")
NEW_NAMES = {"amc": "amc_tt_pair_exact", "mrm": "mrm_tt_pair",
             "askmc": "askmc_tt_pair_count_based"}


def kwargs_for(cls, data):
    names = {field.name for field in fields(cls)}
    return {key: value for key, value in data.items() if key in names}


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def write_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        writer.writerows(rows)


def validate_legacy_code(frozen):
    """No new measurement can silently use a changed old physics/metric module."""
    for item in verify_freeze(frozen)["files"]:
        relative = Path(item["path"])
        if relative.parts[:2] in {("code_at_freeze", "RL4KMC"),
                                  ("code_at_freeze", "configs")}:
            current = ROOT / Path(*relative.parts[1:])
            if not current.is_file() or sha256(current) != item["sha256"]:
                raise ValueError(f"Legacy contract code/config changed: {current}")


def verify_initial_samples(legacy, episode):
    reference = [r for r in legacy["episodes"] if r["method"] == "classical_hcp_lkmc"
                 and r["seed"] == episode["seed"]]
    if len(reference) != 1 or reference[0]["samples"][0] != episode["samples"][0]:
        raise ValueError("Initial sample does not match the frozen LKMC protocol")


def run_compatible(legacy, config, reward, seed, as_options):
    engines = []
    def factory(value):
        env = make_hcp_policy_environment(config, seed=value)
        engine = AcceleratedHCP(env.kmc, "amc", seed=value+49979687, **as_options)
        env.kmc.step = engine.micro_step
        engines.append(engine)
        return env
    start = time.perf_counter()
    episode = evaluate_exp1_episode(
        environment_factory=factory, reward_model=reward, seed=seed,
        steps=legacy["evaluation"]["steps"],
        sample_interval=legacy["evaluation"]["sample_interval"],
        precursor_definition=PersistentPrecursorDefinition(**kwargs_for(
            PersistentPrecursorDefinition, legacy["precursor_definition"])),
        method="classical_hcp_lkmc")
    episode["wall_seconds"] = time.perf_counter()-start
    episode["method"] = NEW_NAMES["amc"]
    episode["adapter"] = "exact_absorbing_chain_with_full_micro_path_reconstruction"
    episode["counters"] = engines[0].stats
    verify_initial_samples(legacy, episode)
    return episode


def run_physical(config, reward, method, seed, horizon, as_options, max_iterations):
    start = time.perf_counter()
    env = make_hcp_policy_environment(config, seed=seed)
    initial_hash = hashlib.sha256(env.kmc.lattice.hydrogen_sites.tobytes()).hexdigest()
    engine = AcceleratedHCP(env.kmc, method, seed=seed+32452843, **as_options)
    counters = engine.run_until(horizon, max_iterations=max_iterations)
    snapshot = reward.snapshot(env.kmc)
    displacement = env.kmc.lattice.unwrapped_displacements_angstrom
    endpoint = {
        "basal_msd_A2": float(np.mean(np.sum(displacement[:, :2]**2, axis=1))),
        "c_msd_A2": float(np.mean(displacement[:, 2]**2)),
        "precursor_coordinate": precursor_coordinate(snapshot),
        "O_site_fraction": float(np.mean(
            env.kmc.lattice.site_types[env.kmc.lattice.hydrogen_sites] == 0)),
        **snapshot,
    }
    elapsed = time.perf_counter()-start
    return {
        "method": method, "seed": seed, "horizon_s": horizon,
        "initial_state_sha256": initial_hash, "wall_seconds": elapsed,
        "timing_scope": "environment+basin_build+simulation+endpoint_metrics; excludes imports/io",
        "time_sampling": "exact original CTMC" if method in {"lkmc", "amc"}
                         else "approximate dynamics; not certified exact",
        "endpoint_observation": "mean-residence projection on censored basin" if method == "mrm"
                                else "microstate at fixed horizon",
        "microscopic_events": counters["ordinary_events"] if method in {"lkmc", "askmc"} else None,
        "original_microscopic_budget_compatible": method == "lkmc",
        "tt_recross_fraction": None, "endpoint": endpoint, "counters": counters,
    }


def mean_se(values):
    a = np.asarray(values, dtype=float)
    return float(a.mean()), float(a.std(ddof=1)/np.sqrt(a.size)) if a.size > 1 else None


def summarize_cost(episodes):
    reference = {e["seed"]: e for e in episodes if e["method"] == "lkmc"}
    rows = []
    for method in METHODS:
        values = [e for e in episodes if e["method"] == method]
        if not values:
            continue
        if any(e["seed"] not in reference for e in values):
            raise ValueError("Cost speedup requires matched freshly timed LKMC seeds")
        for e in values:
            ref = reference[e["seed"]]
            if e["horizon_s"] != ref["horizon_s"] or e["initial_state_sha256"] != ref["initial_state_sha256"]:
                raise ValueError("Mismatched cost comparison")
        elapsed, elapsed_se = mean_se([e["wall_seconds"] for e in values])
        row = {"method": method, "n_episodes": len(values), "horizon_s": values[0]["horizon_s"],
               "mean_wall_seconds": elapsed, "wall_seconds_se": elapsed_se,
               "same_horizon_rollout_speedup": sum(reference[e["seed"]]["wall_seconds"] for e in values)
                    / sum(e["wall_seconds"] for e in values),
               "physical_accuracy_certified": False,
               "tt_recross_fraction": None,
               "endpoint_observation": values[0]["endpoint_observation"]}
        for key in ("basal_msd_A2", "c_msd_A2", "precursor_coordinate", "O_site_fraction"):
            row[key+"_mean"], row[key+"_se"] = mean_se([e["endpoint"][key] for e in values])
            differences = [e["endpoint"][key]-reference[e["seed"]]["endpoint"][key] for e in values]
            row[key+"_paired_delta"], row[key+"_paired_delta_se"] = mean_se(differences)
        for key in values[0]["counters"]:
            row[key+"_mean"] = float(np.mean([e["counters"][key] for e in values]))
        rows.append(row)
    return rows


def merge_rows(legacy, compatible):
    rows = []
    for original in legacy["main_table"]:
        if original["method"] == "amc_superbasin_reference":
            continue
        row = dict(original)  # All historical numeric cells are copied, not recomputed.
        row["data_origin"] = "frozen_legacy"
        checkpoint = legacy.get("checkpoints", {}).get(row["method"], {})
        row["checkpoint_update"] = checkpoint.get("update")
        row["caveat"] = ("initialization checkpoint, not trained ablation"
                         if checkpoint.get("update") == 0 else "legacy metrics and training budget retained")
        rows.append(row)
    expected = sorted(legacy["evaluation"]["seeds"])
    full = sorted(e["seed"] for e in compatible) == expected
    if compatible and full:
        references = [e for e in legacy["episodes"] if e["method"] == "classical_hcp_lkmc"]
        summary = summarize_exp1_episodes(references+compatible,
            classical_method="classical_hcp_lkmc", steps=legacy["evaluation"]["steps"])
        row = next(r for r in summary if r["method"] == NEW_NAMES["amc"])
        row.update(data_origin="new_compatible_micro_reconstruction", checkpoint_update=None,
                   caveat="TT-pair basin only; reconstruction cost not macro benchmark cost")
        rows.append(row)
    else:
        rows.append({"method": NEW_NAMES["amc"], "data_origin": "pending_or_subset",
                     "caveat": "No aggregate appended until ALL frozen evaluation seeds are complete"})
    for method in ("mrm", "askmc"):
        rows.append({"method": NEW_NAMES[method], "data_origin": "separate_physical_time_table",
                     "caveat": "N/A: no equivalent original-microevent path statistics"})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["all", "compatible", "physical"], default="all")
    parser.add_argument("--episodes", type=int, default=0, help="0 uses all frozen seeds; subset is pilot-only")
    parser.add_argument("--physical-time-s", type=float, default=5e-11)
    parser.add_argument("--as-nf", type=int, default=20)
    parser.add_argument("--as-alpha", type=float, default=1.5)
    parser.add_argument("--as-gamma", type=float, default=2.0)
    parser.add_argument("--max-iterations", type=int, default=1000000)
    args = parser.parse_args(argv)
    if args.episodes < 0 or args.physical_time_s <= 0 or not np.isfinite(args.physical_time_s):
        raise ValueError("Invalid episode count or physical horizon")
    frozen, output = Path(args.frozen_dir).resolve(), Path(args.output_dir).resolve()
    if frozen == output or frozen in output.parents:
        raise ValueError("Output must be outside frozen snapshot")
    validate_legacy_code(frozen)
    legacy = json.loads((frozen / "experiment/evaluation/exp1_results.json").read_text())
    config = HCPEnvironmentConfig(**kwargs_for(HCPEnvironmentConfig, legacy["environment"]))
    if config.metadata() != legacy["environment"]:
        raise ValueError("Environment serialization differs from frozen contract")
    reward = HCPRewardModel(HCPRewardConfig(**kwargs_for(HCPRewardConfig, legacy["reward_training_metadata"])))
    seeds = legacy["evaluation"]["seeds"]
    if args.episodes:
        seeds = seeds[:args.episodes]
    as_options = {"as_nf": args.as_nf, "as_alpha": args.as_alpha, "as_gamma": args.as_gamma}
    code_paths = [Path(__file__), ROOT/"RL4KMC/physics/accelerated_baselines.py",
                  ROOT/"experiments/freeze_exp1.py"]
    contract = {"schema_version": 1, "frozen_manifest_sha256": sha256(frozen/"freeze_manifest.json"),
                "seeds": seeds, "mode": args.mode, "physical_time_s": args.physical_time_s,
                "as_options": as_options, "max_iterations": args.max_iterations,
                "new_code_sha256": {p.relative_to(ROOT).as_posix(): sha256(p) for p in code_paths}}
    output.mkdir(parents=True, exist_ok=True)
    contract_path = output / "contract.json"
    if contract_path.exists():
        if json.loads(contract_path.read_text()) != contract:
            raise ValueError("Output already belongs to a different protocol/code; use a NEW output directory")
    elif any(output.iterdir()):
        raise ValueError("Refusing nonempty output directory without a contract")
    else:
        write_json(contract_path, contract)
        write_json(output/"host.json", {"hostname": platform.node(), "platform": platform.platform(),
                   "python": platform.python_version(), "numpy": np.__version__,
                   "started_utc": datetime.now(timezone.utc).isoformat(),
                   "cpu_count": os.cpu_count(), "thread_settings": {k: os.getenv(k) for k in
                    ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                   "shared_host_warning": "Not an isolated hardware benchmark; imports and I/O excluded"})
    episode_dir = output / "episodes"
    episode_dir.mkdir(exist_ok=True)
    compatible, physical = [], []
    for seed in seeds:
        if args.mode in {"all", "compatible"}:
            path = episode_dir / f"compatible_amc_{seed}.json"
            if not path.exists():
                write_json(path, run_compatible(legacy, config, reward, seed, as_options))
            episode = json.loads(path.read_text())
            compatible.append(episode)
            print(json.dumps({"phase": "compatible", "seed": seed,
                              "wall_seconds": episode["wall_seconds"]}), flush=True)
        if args.mode in {"all", "physical"}:
            # Deterministic balanced order avoids always timing one method first.
            order = np.random.default_rng(seed+15485863).permutation(METHODS)
            for method in order:
                path = episode_dir / f"physical_{method}_{seed}.json"
                if not path.exists():
                    write_json(path, run_physical(config, reward, str(method), seed,
                        args.physical_time_s, as_options, args.max_iterations))
                episode = json.loads(path.read_text())
                physical.append(episode)
                print(json.dumps({"phase": "physical", "method": str(method), "seed": seed,
                                  "wall_seconds": episode["wall_seconds"],
                                  "counters": episode["counters"]}), flush=True)
    rows = merge_rows(legacy, compatible)
    cost_rows = summarize_cost(physical)
    write_csv(output/"merged_legacy_table.csv", rows)
    write_csv(output/"physical_cost_fidelity_table.csv", cost_rows)
    write_json(output/"baseline_results.json", {
        "contract": contract, "legacy_numeric_cells_preserved": True,
        "publication_ready": False, "merged_main_table": rows,
        "physical_cost_fidelity_table": cost_rows, "compatible_episodes": compatible,
        "physical_episodes": physical,
        "limitations": ["Two-state TT basins only; not general many-H superbasins",
            "Legacy update-0 ablations and unequal training budgets retained and flagged",
            "Legacy transport/precursor proxy definitions unchanged, not hydride labels",
            "MRM endpoint projection approximate; AS count variant has no certified error bound",
            "Cost measurements do not retroactively supply historical AtomWorld timing",
            "No microbudget figures for MRM/AS and no zero-filled hidden TT recross",
            "Shared host timing; no automatic scientific-accuracy certification"]})
    (output/"README.md").write_text(
        "# Incremental HCP baselines\n\n"
        "`merged_legacy_table.csv`: historical numeric cells preserved; projection removed; "
        "AMC only appended with all frozen seeds. Empty cells mean unavailable, not zero.\n\n"
        "`physical_cost_fidelity_table.csv`: newly timed same-horizon runs. MSD is an endpoint "
        "observable, not an estimated diffusion coefficient. Paired deltas include Monte Carlo "
        "uncertainty; they are not certified DynErr/StructErr.\n\n"
        "All methods use TT-pair-only basins where applicable. Report basin/scaling coverage "
        "alongside speedup; a method may be slower or rarely activate.\n",
        encoding="utf-8")
    print(json.dumps({"status": "complete", "output_dir": str(output),
                      "publication_ready": False}), flush=True)


if __name__ == "__main__":
    main()
