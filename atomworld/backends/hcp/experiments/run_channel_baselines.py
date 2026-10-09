"""Append only event-budget-compatible HCP results to a verified frozen table."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import time
import numpy as np

from atomworld.backends.hcp.experiments.freeze_exp1 import sha256
from atomworld.backends.hcp.experiments.run_exp1_baselines import (
    ROOT, kwargs_for, validate_legacy_code, verify_initial_samples, write_csv, write_json)
from atomworld.backends.hcp.physics.channel_scaling import ChannelScaledHCP, ScalingConfig
from atomworld.backends.hcp.training.exp1_metrics import (
    PersistentPrecursorDefinition, evaluate_exp1_episode, summarize_exp1_episodes)
from atomworld.backends.hcp.training.factory import HCPEnvironmentConfig, make_hcp_policy_environment
from atomworld.backends.hcp.training.reward import HCPRewardConfig, HCPRewardModel

METHOD = "stiffness_scaling_prats2025_hcp"
AMC = "amc_tt_pair_exact"


def run_episode(legacy, seed, config_dict, verify_initial=True):
    config = HCPEnvironmentConfig(**kwargs_for(HCPEnvironmentConfig, legacy["environment"]))
    if config.metadata() != legacy["environment"]:
        raise ValueError("Environment differs from frozen metadata")
    reward = HCPRewardModel(HCPRewardConfig(**kwargs_for(
        HCPRewardConfig, legacy["reward_training_metadata"])))
    engines = []
    def factory(value):
        env = make_hcp_policy_environment(config, seed=value)
        engine = ChannelScaledHCP(env.kmc, seed=value+49979687,
                                 config=ScalingConfig(**config_dict))
        env.kmc.step = engine.micro_step
        engines.append(engine)
        return env
    start = time.perf_counter()
    episode = evaluate_exp1_episode(environment_factory=factory, reward_model=reward,
        seed=seed, steps=legacy["evaluation"]["steps"],
        sample_interval=legacy["evaluation"]["sample_interval"],
        precursor_definition=PersistentPrecursorDefinition(**kwargs_for(
            PersistentPrecursorDefinition, legacy["precursor_definition"])),
        method="classical_hcp_lkmc")
    engine = engines[0]
    engine.kmc.lattice.assert_consistent()
    if not np.isclose(engine.kmc.energy_eV, engine.kmc.calculate_energy_eV(), atol=1e-9):
        raise AssertionError("Energy conservation/accounting failure")
    if engine.kmc.step_count != episode["events"] or sum(episode["hop_counts"].values()) != episode["events"]:
        raise AssertionError("Every charged event must be an executed microscopic hop")
    if engine.kmc.time_s != 0:
        raise AssertionError("Scaled time must not leak into the original-physics clock")
    if verify_initial:
        verify_initial_samples(legacy, episode)
    episode.update(method=METHOD, wall_seconds=time.perf_counter()-start,
        diagnostics=engine.diagnostics(), adapter="one_executed_hop_per_microevent",
        physical_accuracy_validated=False)
    return episode


def validate_episode_set(legacy, episodes, method):
    expected = sorted(legacy["evaluation"]["seeds"])
    if sorted(e["seed"] for e in episodes) != expected:
        raise ValueError("Require every frozen seed exactly once; no subset/duplicate merge")
    axis = [s["event"] for s in next(e for e in legacy["episodes"]
        if e["method"] == "classical_hcp_lkmc")["samples"]]
    for episode in episodes:
        if episode["method"] != method or episode["events"] != legacy["evaluation"]["steps"]:
            raise ValueError("Wrong method or microscopic budget")
        if [s["event"] for s in episode["samples"]] != axis:
            raise ValueError("Sampling axis differs from frozen experiment")
        if (episode["tt_recrosses"] != episode["underlying_tt_recrosses"]
                or episode["tt_events"] != episode["underlying_tt_events"]):
            raise ValueError("Projected/hidden recrossing statistics are not compatible")
        if sum(episode["hop_counts"].values()) != episode["events"]:
            raise ValueError("Incomplete microscopic event accounting")
        verify_initial_samples(legacy, episode)


def merge(legacy, amc_episodes, new_episodes):
    validate_episode_set(legacy, amc_episodes, AMC)
    validate_episode_set(legacy, new_episodes, METHOD)
    rows = [dict(r) for r in legacy["main_table"] if r["method"] != "amc_superbasin_reference"]
    refs = [e for e in legacy["episodes"] if e["method"] == "classical_hcp_lkmc"]
    summaries = summarize_exp1_episodes(refs+amc_episodes+new_episodes,
        classical_method="classical_hcp_lkmc", steps=legacy["evaluation"]["steps"])
    for method in (AMC, METHOD):
        row = next(r for r in summaries if r["method"] == method)
        row["data_origin"] = "existing_exact_amc_micro_replay" if method == AMC else "new_channel_scaling"
        rows.append(row)
    # Historical values, including their uncertainty convention, are immutable.
    for original in legacy["main_table"]:
        if original["method"] != "amc_superbasin_reference":
            if next(r for r in rows if r["method"] == original["method"]) != original:
                raise AssertionError("An old numeric cell was changed")
    return rows


def display_table(rows):
    labels = {"classical_hcp_lkmc": "Classical HCP-LKMC",
        "rate_greedy_tabu": "Rate-greedy + reverse tabu",
        "energy_greedy_tabu": "Energy-greedy + reverse tabu",
        "snapshot_only_atomworld": "Snapshot-only AtomWorld",
        "atomworld_no_short_memory": "AtomWorld w/o short memory [legacy]",
        "atomworld_no_long_memory": "AtomWorld w/o long memory [legacy]",
        "atomworld": "AtomWorld", AMC: "AMC TT-pair (exact micro-path replay)",
        METHOD: "Stiffness-scaling KMC (Prats 2025, HCP adaptation)"}
    lines = ["# HCP event-budget comparison", "",
        "| Method | Event-budget MSD AUC (x) | H-enrichment AUC (x) | Sustained precursor hit | TT recross |",
        "|---|---:|---:|---:|---:|"]
    for r in rows:
        n = r["n_episodes"]
        lines.append(f"| {labels.get(r['method'], r['method'])} | "
            f"{r['persistent_transport_auc_normalized']:.4f} | "
            f"{r['persistent_precursor_auc_normalized']:.4f} | "
            f"{round(r['persistent_precursor_hit_rate']*n)}/{n} | "
            f"{100*r['tt_recross_fraction']:.3f}% |")
    lines += ["", "All old numeric cells are retained; only the old AMC projection is removed.",
        "2048 executed microscopic events; samples every 16 events; four-sample precursor criterion.",
        "AUC normalization uses the frozen LKMC mean. Raw CSV retains original 95% normal-approximation intervals.",
        "No macro exit is charged as a single microscopic hop. No zero-filled hidden TT recrossing.",
        "TT recross uses the frozen GLOBAL immediately preceding event definition (not per-H last-hop history).",
        "MSD is not persistence-filtered; enrichment is a pair-model surrogate, not hydride nucleation.",
        "Scaling changes event probabilities. These four columns do not validate physical kinetics or E2E speedup.",
        "Legacy no-short/no-long selected checkpoints were update 0; training budgets differ. No repaired ablation claim.",
        "Uncertainty and per-seed observations are in merged_main_table.csv / channel_results.json.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--amc-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--check-every", type=int, default=512)
    parser.add_argument("--pe-tolerance", type=float, default=.1)
    parser.add_argument("--pe-min-count", type=int, default=10)
    parser.add_argument("--min-separation", type=float, default=5.)
    parser.add_argument("--max-separation", type=float, default=10.)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("Workers must be 1--4")
    if os.getenv("SLURM_JOB_ID") and args.workers > int(os.getenv("SLURM_CPUS_PER_TASK", "1")):
        raise ValueError("Worker count exceeds Slurm allocation")
    frozen, amc, output = map(lambda p: Path(p).resolve(),
                             (args.frozen_dir, args.amc_dir, args.output_dir))
    if any(output == p or p in output.parents for p in (frozen, amc)):
        raise ValueError("Do not write into old results")
    validate_legacy_code(frozen)
    legacy = json.loads((frozen/"experiment/evaluation/exp1_results.json").read_text())
    amc_path = amc/"baseline_results.json"
    old_amc = json.loads(amc_path.read_text())
    if old_amc["contract"]["frozen_manifest_sha256"] != sha256(frozen/"freeze_manifest.json"):
        raise ValueError("AMC used a different frozen protocol")
    for path, digest in old_amc["contract"]["new_code_sha256"].items():
        if sha256(ROOT/path) != digest:
            raise ValueError(f"AMC provenance differs: {path}")
    amc_episodes = old_amc["compatible_episodes"]
    validate_episode_set(legacy, amc_episodes, AMC)
    options = asdict(ScalingConfig(check_every=args.check_every,
        pe_tolerance=args.pe_tolerance, pe_min_count=args.pe_min_count,
        min_separation=args.min_separation, max_separation=args.max_separation))
    paths = ["RL4KMC/physics/channel_scaling.py", "experiments/run_channel_baselines.py",
             "tests/test_channel_scaling.py", "tests/test_channel_protocol.py",
             "run_exp1HCP_strong_baselines.sh", "scripts/run_hcp_channels_slurm.sh",
             "HCP_CHANNEL_BASELINES.md", "experiments/run_exp1_baselines.py"]
    contract = {"schema": 1, "method": METHOD, "options": options,
        "frozen_manifest_sha256": sha256(frozen/"freeze_manifest.json"),
        "amc_results_sha256": sha256(amc_path), "evaluation": legacy["evaluation"],
        "development_seeds": [920000, 921009],
        "source_sha256": {p: sha256(ROOT/p) for p in paths},
        "parameter_selection": "a priori HCP short-budget settings; no test-set tuning",
        "physical_accuracy_validated": False}
    output.mkdir(parents=True, exist_ok=True)
    if (output/"contract.json").exists():
        if json.loads((output/"contract.json").read_text()) != contract:
            raise ValueError("Changed code/parameters: use a NEW output directory")
    elif any(output.iterdir()):
        raise ValueError("Nonempty output without contract")
    else:
        write_json(output/"contract.json", contract)
        for p in paths:
            dest = output/"source"/p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/p, dest)
    write_json(output/f"host-{os.getenv('SLURM_JOB_ID', 'local')}.json",
        {"hostname": platform.node(), "python": platform.python_version(),
         "numpy": np.__version__, "workers": args.workers,
         "slurm_job_id": os.getenv("SLURM_JOB_ID"),
         "started_utc": datetime.now(timezone.utc).isoformat()})
    # Development sanity checks precede evaluation, without selecting parameters.
    for seed in contract["development_seeds"]:
        if seed in legacy["evaluation"]["seeds"]:
            raise ValueError("Development/test overlap")
        path = output/f"development_{seed}.json"
        if not path.exists():
            write_json(path, run_episode(legacy, seed, options, False))
        d = json.loads(path.read_text())
        print(json.dumps({"phase": "development", "seed": seed,
            "scaled_events": d["diagnostics"]["scaled_events"]}), flush=True)
    episode_dir = output/"episodes"
    episode_dir.mkdir(exist_ok=True)
    pending = [seed for seed in legacy["evaluation"]["seeds"]
               if not (episode_dir/f"{seed}.json").exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_episode, legacy, seed, options): seed for seed in pending}
        for future in as_completed(futures):
            seed = futures[future]
            episode = future.result()
            write_json(episode_dir/f"{seed}.json", episode)
            print(json.dumps({"phase": "evaluation", "seed": seed,
                "wall_seconds": episode["wall_seconds"],
                "scaled_events": episode["diagnostics"]["scaled_events"]}), flush=True)
    episodes = [json.loads((episode_dir/f"{seed}.json").read_text())
                for seed in legacy["evaluation"]["seeds"]]
    rows = merge(legacy, amc_episodes, episodes)
    validate_legacy_code(frozen)
    for p, digest in contract["source_sha256"].items():
        if sha256(ROOT/p) != digest:
            raise ValueError("Source changed while running; refusing final table")
    write_csv(output/"merged_main_table.csv", rows)
    write_json(output/"channel_results.json", {"contract": contract,
        "merged_main_table": rows, "new_episodes": episodes,
        "amc_episodes": amc_episodes, "legacy_numeric_cells_preserved": True,
        "physical_accuracy_validated": False,
        "active_trajectories": sum(e["diagnostics"]["scaled_events"] > 0 for e in episodes)})
    (output/"merged_main_table.md").write_text(display_table(rows), encoding="utf-8")
    print(json.dumps({"status": "complete", "episodes": len(episodes),
                      "output": str(output)}), flush=True)


if __name__ == "__main__":
    main()
