"""Layered v2 validation/cost pilot; never appends new observables to the v1 table."""
from __future__ import annotations

import argparse
import csv
from dataclasses import fields, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np

from atomworld.backends.hcp.experiments.freeze_exp1 import verify_freeze, sha256
from atomworld.backends.hcp.physics.accelerated_baselines import AcceleratedHCP
from atomworld.backends.hcp.physics.multistate_baselines import MultistateHCP
from atomworld.backends.hcp.training.factory import HCPEnvironmentConfig, make_hcp_policy_environment
from atomworld.backends.hcp.training.reward import HCPRewardModel

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("lkmc", "amc", "mrm", "cached", "askmc")
CASES = ("dilute_1H", "coupled_3H", "main_27H")


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(data,indent=2,sort_keys=True,allow_nan=False),encoding="utf-8")
    temporary.replace(path)


def coupled_initialization(kmc):
    """Deterministic validation motif with genuine attractive H-H coupling.

    Anchor comes from the seeded initial state. Pick two distinct attractive
    T-neighbors; this is a validation preparation, NOT the main-table initial law.
    """
    anchor = int(kmc.lattice.hydrogen_sites[0])
    neighbors = [int(site) for site in np.flatnonzero(kmc.lattice.site_types == 1)
                 if site != anchor and kmc.energy_model.pair_energy_eV(kmc.lattice,anchor,int(site)) < 0]
    if len(neighbors) < 2:
        raise ValueError("Cannot construct the declared coupled validation motif")
    kmc.lattice.set_hydrogen_sites([anchor,*neighbors[:2]])
    kmc.energy_eV = kmc.calculate_energy_eV()
    if kmc.energy_model.total_extra_energy_eV(kmc.lattice) == 0:
        raise AssertionError("Coupled validation motif has no extra energy")


def case_config(base, case):
    if case == "dilute_1H":
        return replace(base,size=(4,4,4),n_hydrogen=1,local_energy_file="")
    if case == "coupled_3H":
        return replace(base,size=(4,4,4),n_hydrogen=3)
    if case == "main_27H":
        if base.n_hydrogen != 27:
            raise ValueError("Frozen main protocol does not contain 27 H")
        return base
    raise ValueError(case)


def run_episode(config,case,method,seed,horizon,options,max_iterations):
    start = time.perf_counter()
    kmc = make_hcp_policy_environment(config,seed=seed).kmc
    if case == "coupled_3H":
        coupled_initialization(kmc)
    initial_sites = kmc.lattice.hydrogen_sites.tolist()
    initial_extra_energy = kmc.energy_model.total_extra_energy_eV(kmc.lattice)
    engine = (AcceleratedHCP(kmc,"lkmc",seed=seed+15485863) if method == "lkmc"
              else MultistateHCP(kmc,method,seed=seed+15485863,**options))
    simulation_start = time.perf_counter()
    stats = engine.run_until(horizon,max_iterations=max_iterations)
    simulation_seconds = time.perf_counter()-simulation_start
    snapshot = HCPRewardModel().snapshot(kmc)
    d = kmc.lattice.unwrapped_displacements_angstrom
    observables = {
        "basal_msd_A2": float(np.mean(np.sum(d[:,:2]**2,axis=1))),
        "c_msd_A2": float(np.mean(d[:,2]**2)),
        "O_site_fraction": float(np.mean(kmc.lattice.site_types[kmc.lattice.hydrogen_sites] == 0)),
        "precursor_coordinate": .25*snapshot["attractive_pairs_per_h"]
             +snapshot["largest_cluster_fraction"]+.5*snapshot["nonisolated_fraction"],
        **snapshot,
    }
    if method == "askmc":
        activation = "scaled" if stats["as_scaling_updates"] else "inactive_no_scaling"
    else:
        activation = "not_applicable"
    return dict(case=case,method=method,seed=seed,horizon_s=horizon,
        initial_sites=initial_sites,initial_extra_energy_eV=initial_extra_energy,
        configuration=config.metadata(),endpoint=observables,counters=stats,
        wall_seconds=time.perf_counter()-start,simulation_seconds=simulation_seconds,
        as_activation=activation,slurm_job_id=os.environ.get("SLURM_JOB_ID"),
        endpoint_observation="mean_residence_projection_when_censored" if method == "mrm" else "microstate",
        microscopic_event_count=None if method in {"amc","mrm"} else stats["ordinary_events"],
        tt_recross=None,legacy_microbudget_compatible=False)


def mean_se(values):
    a = np.array(values,dtype=float)
    return float(a.mean()),float(a.std(ddof=1)/np.sqrt(len(a))) if len(a)>1 else None


def summarize(episodes):
    rows = []
    for case in CASES:
        reference = {e["seed"]:e for e in episodes if e["case"]==case and e["method"]=="lkmc"}
        for method in METHODS:
            values = [e for e in episodes if e["case"]==case and e["method"]==method]
            if not values:
                continue
            for e in values:
                ref = reference[e["seed"]]
                for key in ("initial_sites","configuration","horizon_s"):
                    if e[key] != ref[key]:
                        raise ValueError("Unmatched reference for "+key)
            wall,wall_se = mean_se([e["wall_seconds"] for e in values])
            row = dict(case=case,method=method,n=len(values),horizon_s=values[0]["horizon_s"],
                mean_wall_seconds=wall,wall_seconds_se=wall_se,
                rollout_speedup=sum(reference[e["seed"]]["wall_seconds"] for e in values)
                               /sum(e["wall_seconds"] for e in values),
                as_active_episodes=sum(e["as_activation"]=="scaled" for e in values),
                endpoint_observation=values[0]["endpoint_observation"],publication_ready=False)
            for observable in ("basal_msd_A2","c_msd_A2","O_site_fraction","precursor_coordinate"):
                row[observable],row[observable+"_se"] = mean_se([e["endpoint"][observable] for e in values])
                row[observable+"_paired_delta"],row[observable+"_paired_delta_se"] = mean_se(
                    [e["endpoint"][observable]-reference[e["seed"]]["endpoint"][observable] for e in values])
            for counter in sorted(set(k for e in values for k in e["counters"])):
                row[counter+"_mean"] = float(np.mean([e["counters"].get(counter,0) for e in values]))
            rows.append(row)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen-dir",required=True)
    parser.add_argument("--output-dir",required=True)
    parser.add_argument("--episodes",type=int,default=2)
    parser.add_argument("--seed-base",type=int,default=910000)
    parser.add_argument("--cases",nargs="+",choices=CASES,default=list(CASES))
    parser.add_argument("--max-states",type=int,default=16)
    parser.add_argument("--barrier-cutoff-eV",type=float,default=.25)
    parser.add_argument("--as-nf",type=int,default=20)
    parser.add_argument("--as-alpha",type=float,default=1.5)
    parser.add_argument("--as-gamma",type=float,default=2.)
    parser.add_argument("--dilute-horizon-s",type=float,default=2e-10)
    parser.add_argument("--coupled-horizon-s",type=float,default=2e-11)
    parser.add_argument("--main-horizon-s",type=float,default=2e-12)
    parser.add_argument("--max-iterations",type=int,default=1000000)
    args = parser.parse_args(argv)
    if args.episodes < 1 or len(set(args.cases)) != len(args.cases):
        raise ValueError("Positive episode count and unique cases required")
    horizons = dict(zip(CASES,(args.dilute_horizon_s,args.coupled_horizon_s,args.main_horizon_s)))
    if any(not np.isfinite(v) or v<=0 for v in horizons.values()):
        raise ValueError("Invalid horizons")
    frozen,output = Path(args.frozen_dir).resolve(),Path(args.output_dir).resolve()
    if frozen == output or frozen in output.parents:
        raise ValueError("Output must be outside frozen snapshot")
    manifest = verify_freeze(frozen)
    for item in manifest["files"]:
        p = Path(item["path"])
        if p.parts[:2] in {("code_at_freeze","RL4KMC"),("code_at_freeze","configs")}:
            if sha256(ROOT/Path(*p.parts[1:])) != item["sha256"]:
                raise ValueError("Original physics/metrics changed: "+str(p))
    legacy = json.loads((frozen/"experiment/evaluation/exp1_results.json").read_text())
    names = {f.name for f in fields(HCPEnvironmentConfig)}
    config = HCPEnvironmentConfig(**{k:v for k,v in legacy["environment"].items() if k in names})
    seeds = [args.seed_base+1009*i for i in range(args.episodes)]
    if set(seeds) & set(legacy["evaluation"]["seeds"]):
        raise ValueError("Development validation seeds must not overlap frozen test seeds")
    code = [Path(__file__),ROOT/"RL4KMC/physics/multistate_baselines.py",
            ROOT/"RL4KMC/physics/accelerated_baselines.py",ROOT/"experiments/freeze_exp1.py",
            ROOT/"tests/test_multistate_baselines.py"]
    contract = dict(schema_version=2,seeds=seeds,cases=args.cases,horizons=horizons,
        options=dict(max_states=args.max_states,barrier_cutoff_eV=args.barrier_cutoff_eV,
                     as_nf=args.as_nf,as_alpha=args.as_alpha,as_gamma=args.as_gamma),
        max_iterations=args.max_iterations,frozen_manifest_sha256=sha256(frozen/"freeze_manifest.json"),
        source_hashes={p.relative_to(ROOT).as_posix():sha256(p) for p in code},
        phase="development_validation_not_test_set_selection")
    output.mkdir(parents=True,exist_ok=True)
    if (output/"contract.json").exists():
        if json.loads((output/"contract.json").read_text()) != contract:
            raise ValueError("Protocol/code changed: use a NEW output directory")
    else:
        if any(output.iterdir()):
            raise ValueError("Output is nonempty without a contract")
        write_json(output/"contract.json",contract)
        for p in code:
            target = output/"source"/p.relative_to(ROOT)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,target)
    attempts = output/"attempts"
    attempts.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    write_json(attempts/(stamp+".json"),dict(job_id=os.getenv("SLURM_JOB_ID"),
        hostname=platform.node(),python=platform.python_version(),numpy=np.__version__,
        threads={k:os.getenv(k) for k in ("OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS")},
        timing_scope="environment+initial_preparation+basin+simulation+endpoint; excludes imports/io"))
    episode_dir = output/"episodes"
    episode_dir.mkdir(exist_ok=True)
    episodes = []
    for case in args.cases:
        for seed in seeds:
            for method in np.random.default_rng(seed+32452843).permutation(METHODS):
                path = episode_dir/f"{case}_{method}_{seed}.json"
                if not path.exists():
                    write_json(path,run_episode(case_config(config,case),case,str(method),seed,
                        horizons[case],contract["options"],args.max_iterations))
                episode = json.loads(path.read_text())
                if (episode["case"],episode["method"],episode["seed"]) != (case,method,seed):
                    raise ValueError("Cached episode identity mismatch")
                episodes.append(episode)
                print(json.dumps({k:episode[k] for k in
                    ("case","method","seed","wall_seconds","as_activation","counters")}),flush=True)
    rows = summarize(episodes)
    write_json(output/"multistate_results.json",dict(contract=contract,episodes=episodes,table=rows,
        publication_ready=False,legacy_results_modified=False,
        limitations=["Bounded full-configuration graph; no independence/factorization assumption",
            "AS graph-count variant, no original confidence calibration claimed",
            "MRM finite-horizon microstate projection is approximate",
            "Micro-event AUC/recross absent, not zero; no legacy table append",
            "Endpoint MSD is not a fitted diffusion coefficient; pilot SE is not equivalence certification",
            "AS inactivity and capacity hits must be disclosed; cached control separates cache effects"]))
    columns = list(dict.fromkeys(k for row in rows for k in row))
    with (output/"multistate_cost_fidelity.csv").open("w",newline="",encoding="utf-8") as stream:
        writer = csv.DictWriter(stream,columns)
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(dict(status="complete",output=str(output),publication_ready=False)),flush=True)


if __name__ == "__main__":
    main()
