"""Measure frozen stiffness-scaling against matched physical LKMC ensembles."""
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
from atomworld.backends.hcp.experiments.run_exp1_baselines import ROOT, kwargs_for, validate_legacy_code, write_json, write_csv
from atomworld.backends.hcp.physics.channel_scaling import ChannelScaledHCP, ScalingConfig
from atomworld.backends.hcp.physics.fidelity_metrics import (
    DYNAMIC_KEYS, STRUCTURE_KEYS, PreviousStateRecorder, bootstrap_errors)
from atomworld.backends.hcp.training.factory import HCPEnvironmentConfig, make_hcp_policy_environment
from atomworld.backends.hcp.training.reward import HCPRewardConfig, HCPRewardModel

METHODS = ("lkmc_reference", "lkmc_independent", "stiffness_scaling")


class ReachedCap(Exception):
    pass


def run_trajectory(legacy, options, seed, method, horizon, grid_points, slow_target, cap, max_events):
    start = time.perf_counter()
    config = HCPEnvironmentConfig(**kwargs_for(HCPEnvironmentConfig, legacy["environment"]))
    if config.metadata() != legacy["environment"]:
        raise ValueError("Changed frozen environment")
    kmc = make_hcp_policy_environment(config, seed=seed).kmc
    initial_sites = kmc.lattice.hydrogen_sites.tolist()
    reward = HCPRewardModel(HCPRewardConfig(**kwargs_for(
        HCPRewardConfig, legacy["reward_training_metadata"])))
    if method == "stiffness_scaling":
        engine = ChannelScaledHCP(kmc, seed=seed+49979687, config=ScalingConfig(**options))
        advance = engine.micro_step
        clock = lambda: engine.scaled_process_elapsed_s
    else:
        engine = None
        kmc.rng = np.random.default_rng(seed+(32452843 if method == "lkmc_reference" else 67867967))
        advance = kmc.step
        clock = lambda: kmc.time_s
    grid = np.linspace(0, horizon, grid_points)
    recorder = PreviousStateRecorder(grid)
    counters = {"slow_events": 0, "fpt_s": None, "censored": False}
    def observe():
        snapshot = reward.snapshot(kmc)
        displacement = kmc.lattice.unwrapped_displacements_angstrom
        return {"basal_msd_A2": float(np.mean(np.sum(displacement[:, :2]**2, axis=1))),
            "c_msd_A2": float(np.mean(displacement[:, 2]**2)),
            "O_site_fraction": float(np.mean(kmc.lattice.site_types[kmc.lattice.hydrogen_sites] == 0)),
            "largest_cluster_fraction": float(snapshot["largest_cluster_fraction"]),
            "nonisolated_fraction": float(snapshot["nonisolated_fraction"]),
            "attractive_pairs_per_h": float(snapshot["attractive_pairs_per_h"]),
            "energy_eV": float(kmc.energy_eV), "slow_events": counters["slow_events"]}
    apply = kmc.apply_event
    def recording_apply(eid, **kwargs):
        next_time = clock() if engine is not None else clock()+kwargs.get("delta_time_s", 0.)
        recorder.before_jump(min(next_time, cap), observe)
        if next_time > cap:
            recorder.finish(cap, observe)
            counters["censored"] = counters["fpt_s"] is None
            raise ReachedCap()
        transition = apply(eid, **kwargs)
        if transition.hop_kind != "TT":
            counters["slow_events"] += 1
            if counters["slow_events"] == slow_target:
                counters["fpt_s"] = float(next_time)
        return transition
    kmc.apply_event = recording_apply
    try:
        while clock() < horizon or counters["fpt_s"] is None:
            if kmc.step_count >= max_events:
                raise RuntimeError("Event safety cap reached; no incomplete result")
            advance()
    except ReachedCap:
        pass
    recorder.finish(min(clock(), cap), observe)
    if len(recorder.samples) != grid_points:
        raise AssertionError("Incomplete fixed-time grid")
    kmc.lattice.assert_consistent()
    if not np.isclose(kmc.energy_eV, kmc.calculate_energy_eV(), atol=1e-9):
        raise AssertionError("Energy mismatch")
    if sum(kmc.hop_counts.values()) != kmc.step_count:
        raise AssertionError("Lost microscopic event")
    if engine is not None and kmc.time_s != 0:
        raise AssertionError("Scaled clock falsely assigned as exact physical clock")
    return {"method": method, "seed": seed, "initial_sites": initial_sites,
        "configuration": config.metadata(), "time_grid_s": grid.tolist(),
        "samples": recorder.samples, "slow_target": slow_target, **counters,
        "microscopic_events": kmc.step_count, "hop_counts": dict(kmc.hop_counts),
        "recorded_time_axis": "candidate scaled-process time" if engine else "exact LKMC physical time",
        "scaled_events_including_possible_rejected_cap_proposal": engine.scaled_events if engine else 0,
        "scaling_history": engine.controller.history if engine else [],
        "physical_accuracy_certified": False, "wall_seconds": time.perf_counter()-start}


def aggregate(episodes, seeds, replicates):
    grouped = {}
    for method in METHODS:
        values = [e for e in episodes if e["method"] == method]
        if sorted(e["seed"] for e in values) != sorted(seeds):
            raise ValueError("Missing or duplicate initial-condition groups")
        grouped[method] = sorted(values, key=lambda e: e["seed"])
    ref = grouped["lkmc_reference"]
    for method, values in grouped.items():
        for r, c in zip(ref, values):
            for key in ("initial_sites", "configuration", "time_grid_s", "slow_target"):
                if r[key] != c[key]:
                    raise ValueError("Unmatched fidelity protocol: "+key)
    keys = DYNAMIC_KEYS+STRUCTURE_KEYS
    arrays = {m: np.array([[[s[k] for k in keys] for s in e["samples"]] for e in es])
              for m, es in grouped.items()}
    fpts = {m: None if any(e["fpt_s"] is None for e in es)
            else np.array([e["fpt_s"] for e in es]) for m, es in grouped.items()}
    summaries = {}
    for method in ("lkmc_independent", "stiffness_scaling"):
        summaries[method] = bootstrap_errors(arrays["lkmc_reference"], arrays[method],
            fpts["lkmc_reference"], fpts[method], replicates=replicates)
        summaries[method]["fpt_hits"] = sum(e["fpt_s"] is not None for e in grouped[method])
    summaries["lkmc_reference"] = {"fpt_hits": sum(e["fpt_s"] is not None for e in ref)}
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir", required=True)
    parser.add_argument("--channel-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--horizon-s", type=float, default=1e-10)
    parser.add_argument("--fpt-cap-s", type=float, default=5e-10)
    parser.add_argument("--slow-target", type=int, default=64)
    parser.add_argument("--grid-points", type=int, default=65)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    if not 0 < args.horizon_s < args.fpt_cap_s or not np.isfinite(args.fpt_cap_s):
        raise ValueError("Require finite 0 < horizon < cap")
    if args.slow_target < 1 or args.grid_points < 2 or args.bootstrap < 100:
        raise ValueError("Invalid measurement settings")
    if not 1 <= args.workers <= 4 or args.workers > int(os.getenv("SLURM_CPUS_PER_TASK", "4")):
        raise ValueError("Workers exceed allowed allocation")
    frozen, channels, output = [Path(p).resolve() for p in
        (args.frozen_dir, args.channel_dir, args.output_dir)]
    if any(output == p or p in output.parents for p in (frozen, channels)):
        raise ValueError("Old results are immutable")
    validate_legacy_code(frozen)
    legacy = json.loads((frozen/"experiment/evaluation/exp1_results.json").read_text())
    old = json.loads((channels/"channel_results.json").read_text())
    if old["contract"]["frozen_manifest_sha256"] != sha256(frozen/"freeze_manifest.json"):
        raise ValueError("Different frozen protocol")
    for path, digest in old["contract"]["source_sha256"].items():
        if sha256(ROOT/path) != digest:
            raise ValueError("Previous scaling implementation changed: "+path)
    options = old["contract"]["options"]
    if asdict(ScalingConfig(**options)) != options:
        raise ValueError("Changed scaling options")
    seeds = legacy["evaluation"]["seeds"]
    sources = ["RL4KMC/physics/fidelity_metrics.py", "experiments/run_channel_fidelity.py",
        "tests/test_fidelity_metrics.py", "tests/test_fidelity_protocol.py",
        "run_hcp_fidelity.sh", "scripts/run_hcp_fidelity_slurm.sh", "HCP_FIDELITY_V1.md",
        "RL4KMC/physics/channel_scaling.py", "experiments/run_exp1_baselines.py"]
    contract = {"metric_protocol": "hcp_fidelity_v1", "frozen_manifest_sha256": sha256(frozen/"freeze_manifest.json"),
        "channel_results_sha256": sha256(channels/"channel_results.json"),
        "options": options, "seeds": seeds, "horizon_s": args.horizon_s,
        "fpt_cap_s": args.fpt_cap_s, "slow_target_nonTT_events": args.slow_target,
        "grid_points": args.grid_points, "bootstrap_replicates": args.bootstrap,
        "max_events_safety": 50000, "source_sha256": {p: sha256(ROOT/p) for p in sources},
        "physical_accuracy_certified": False}
    output.mkdir(parents=True, exist_ok=True)
    if (output/"contract.json").exists():
        if json.loads((output/"contract.json").read_text()) != contract:
            raise ValueError("Changed protocol: use a NEW output")
    elif any(output.iterdir()):
        raise ValueError("Nonempty output without contract")
    else:
        write_json(output/"contract.json", contract)
        for p in sources:
            dest = output/"source"/p
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/p, dest)
    write_json(output/f"host-{os.getenv('SLURM_JOB_ID','local')}.json", {
        "host": platform.node(), "python": platform.python_version(), "numpy": np.__version__,
        "workers": args.workers, "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "started_utc": datetime.now(timezone.utc).isoformat()})
    trajectory_dir = output/"trajectories"
    trajectory_dir.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {}
        for seed in seeds:
            for method in METHODS:
                path = trajectory_dir/f"{method}_{seed}.json"
                if path.exists():
                    continue
                task = pool.submit(run_trajectory, legacy, options, seed, method,
                    args.horizon_s, args.grid_points, args.slow_target, args.fpt_cap_s, 50000)
                futures[task] = path
        for task in as_completed(futures):
            episode = task.result()
            write_json(futures[task], episode)
            print(json.dumps({k:episode[k] for k in
                ("method","seed","microscopic_events","fpt_s","wall_seconds")}), flush=True)
    episodes = [json.loads((trajectory_dir/f"{m}_{s}.json").read_text()) for s in seeds for m in METHODS]
    summaries = aggregate(episodes, seeds, args.bootstrap)
    validate_legacy_code(frozen)
    for p, digest in contract["source_sha256"].items():
        if sha256(ROOT/p) != digest:
            raise ValueError("Code changed during measurement")
    rows = []
    for method in ("lkmc_independent", "stiffness_scaling"):
        row = {"method": method, "metric_protocol": "hcp_fidelity_v1", "n": len(seeds)}
        for metric, stats in summaries[method]["metrics"].items():
            for field, value in stats.items():
                row[f"{metric}_{field}"] = value
        rows.append(row)
    write_csv(output/"fidelity_errors.csv", rows)
    write_json(output/"fidelity_results.json", {"contract": contract, "summaries": summaries,
        "episodes": episodes, "original_table_unchanged": True,
        "limitations": ["New explicit HCP definitions, not numerically interchangeable with FCC/BCC metrics",
            "Finite ensemble uncertainty retained; independent LKMC is a sampling-noise control",
            "Clock is slow-event FPT discrepancy, not Poisson loss or per-microevent dwell discrepancy",
            "Short 100 ps transport/structure interval does not establish long-time degradation fidelity"]})
    # Separate versioned table: only new HCP-v1 fields are added to the scaling row.
    augmented = [dict(r) for r in old["merged_main_table"]]
    row = next(r for r in augmented if r["method"] == "stiffness_scaling_prats2025_hcp")
    row["fidelity_metric_protocol"] = "hcp_fidelity_v1"
    for metric in ("ClockErr", "StructErr", "DynErr"):
        for field, value in summaries["stiffness_scaling"]["metrics"][metric].items():
            row[f"hcp_v1_{metric}_{field}"] = value
    write_csv(output/"main_table_with_fidelity_v1.csv", augmented)
    lines = ["# Stiffness-scaling HCP fidelity-v1", "",
        "| Method | ClockErr (95% CI) | StructErr (95% CI) | DynErr (95% CI) |",
        "|---|---:|---:|---:|"]
    for method in ("lkmc_independent", "stiffness_scaling"):
        cells = []
        for metric in ("ClockErr", "StructErr", "DynErr"):
            s = summaries[method]["metrics"][metric]
            cells.append("N/A (censored)" if s["estimate"] is None else
                f"{s['estimate']:.5f} [{s['ci95_low']:.5f}, {s['ci95_high']:.5f}]")
        lines.append("| "+method+" | "+" | ".join(cells)+" |")
    lines += ["", "20 paired initial conditions; 5000 whole-seed bootstrap draws.",
        "ClockErr: absolute relative ensemble-mean time difference to 64 non-TT events; cap 500 ps.",
        "Clock CI is transformed from the signed relative-error CI, retaining zero when appropriate.",
        "DynErr: average normalized RMSE of ensemble basal/c-axis MSD curves, reference-range denominator.",
        "StructErr: mean absolute difference of ensemble O occupancy / largest-cluster / nonisolated fractions.",
        "Transport/structure: right-continuous states on 65 common time points from 0 to 100 ps.",
        "Scaled-process time is a candidate physical-time approximation being tested, not declared exact.",
        "Independent LKMC control measures sampling discrepancy, not algorithm error. Do not subtract it.",
        "Finite-sample norm estimates/percentile CIs have a positive noise floor, especially near zero.",
        "Not directly comparable to untraced manuscript values or other metric definitions.", ""]
    (output/"fidelity_errors.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status":"complete","trajectories":len(episodes),"output":str(output)}), flush=True)


if __name__ == "__main__":
    main()
