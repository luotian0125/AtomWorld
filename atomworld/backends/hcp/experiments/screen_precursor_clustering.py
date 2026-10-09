#!/usr/bin/env python3
"""Matched-seed screening of dissolved-H precursor clustering in alpha-Zr."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from atomworld.backends.hcp.analysis.clusters import hydrogen_cluster_metrics
from atomworld.backends.hcp.envs.hcp_kmc import HCPHydrogenKMC
from atomworld.backends.hcp.physics.local_energy import (
    LocalEnergyModel,
    PairInteractionShell,
    load_local_energy_model,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", nargs=3, type=int, default=(4, 4, 4))
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=(400, 600, 800)
    )
    parser.add_argument("--hydrogen-counts", nargs="+", type=int, default=(2, 4, 8))
    parser.add_argument("--burn-in-steps", type=int, default=5000)
    parser.add_argument("--sample-interval", type=int, default=1000)
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--replicas", type=int, default=4)
    parser.add_argument("--seed", type=int, default=4211)
    parser.add_argument("--maximum-h-atomic-fraction", type=float, default=0.077)
    parser.add_argument(
        "--allow-outside-source-range",
        action="store_true",
        help="explicitly permit screening beyond the source study's H concentration range",
    )
    parser.add_argument(
        "--pair-energy-scales",
        nargs="+",
        type=float,
        default=(1.0,),
        help="multipliers for all literature-screening pair energies",
    )
    parser.add_argument(
        "--pair-config",
        type=Path,
        default=Path("configs/hcp/zr_h_pair_screening_andolina2022.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/screening/precursor_clustering.json"),
    )
    return parser.parse_args()


def _flatten_metrics(metrics):
    flat = {key: value for key, value in metrics.items() if key != "pair_counts"}
    for label, value in metrics["pair_counts"].items():
        flat[f"pair_count:{label}"] = value
    return flat


def _replica_summary(env, observable_model, args):
    env.run(args.burn_in_steps)
    observations = defaultdict(list)
    for _ in range(args.samples):
        env.run(args.sample_interval)
        metrics = _flatten_metrics(
            hydrogen_cluster_metrics(env.lattice, observable_model)
        )
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                observations[key].append(float(value))
    energy_residual = float(abs(env.energy_eV - env.calculate_energy_eV()))
    diagnostics = env.kinetic_diagnostics()
    for kind, value in diagnostics["hop_fractions"].items():
        observations[f"hop_fraction:{kind}"].append(float(value))
    observations["immediate_reverse_fraction"].append(
        float(diagnostics["immediate_reverse_fraction"])
    )
    for key, value in env.diffusion_coefficients_m2_per_s().items():
        if np.isfinite(value):
            observations[f"diffusion:{key}"].append(float(value))
    observations["elapsed_physical_time_s"].append(float(env.time_s))
    return {
        key: float(np.mean(values)) for key, values in observations.items()
    }, energy_residual


def _aggregate_replicas(replica_rows):
    keys = sorted(replica_rows[0])
    return {
        key: {
            "mean": float(np.mean([row[key] for row in replica_rows])),
            "std_across_replicas": (
                float(np.std([row[key] for row in replica_rows], ddof=1))
                if len(replica_rows) > 1
                else 0.0
            ),
        }
        for key in keys
    }


def _scaled_model(model, scale):
    scale = float(scale)
    if scale < 0.0 or not np.isfinite(scale):
        raise ValueError("pair-energy scales must be finite and non-negative")
    shells = [
        PairInteractionShell(
            minimum_distance_angstrom=shell.minimum_distance_angstrom,
            maximum_distance_angstrom=shell.maximum_distance_angstrom,
            interaction_energy_eV=scale * shell.interaction_energy_eV,
            source_site_type=shell.source_site_type,
            target_site_type=shell.target_site_type,
            label=shell.label,
        )
        for shell in model.pair_shells
    ]
    provenance = dict(model.provenance)
    provenance["pair_energy_scale"] = scale
    return LocalEnergyModel(
        trap_site_energies_eV=model.trap_site_energies_eV,
        pair_shells=shells,
        provenance=provenance,
    )


def run_case(
    case_name,
    kinetic_model,
    observable_model,
    temperature,
    n_hydrogen,
    args,
    pair_energy_scale,
):
    rows = []
    residuals = []
    for replica in range(args.replicas):
        env = HCPHydrogenKMC(
            size=args.size,
            n_hydrogen=n_hydrogen,
            temperature_K=temperature,
            seed=args.seed + replica,
            energy_model=kinetic_model,
        )
        row, residual = _replica_summary(env, observable_model, args)
        rows.append(row)
        residuals.append(residual)
    num_zr = 2 * int(np.prod(args.size))
    return {
        "case": case_name,
        "pair_energy_scale": float(pair_energy_scale),
        "temperature_K": float(temperature),
        "n_hydrogen": int(n_hydrogen),
        "n_zirconium": num_zr,
        "H_over_Zr": float(n_hydrogen / num_zr),
        "H_atomic_fraction": float(n_hydrogen / (n_hydrogen + num_zr)),
        "metrics": _aggregate_replicas(rows),
        "replica_metrics": rows,
        "maximum_energy_residual_eV": max(residuals),
        "quality_gate": {
            "energy_bookkeeping": bool(max(residuals) <= 1.0e-10),
        },
    }


def main():
    args = parse_args()
    if any(value < 3 for value in args.size):
        raise ValueError("all supercell dimensions must be at least 3")
    if any(value <= 0 for value in args.hydrogen_counts):
        raise ValueError("hydrogen counts must be positive")
    if (
        args.burn_in_steps < 0
        or min(args.sample_interval, args.samples, args.replicas) <= 0
    ):
        raise ValueError("sampling controls must be positive and burn-in non-negative")
    if not 0.0 < args.maximum_h_atomic_fraction < 1.0:
        raise ValueError(
            "maximum H atomic fraction must lie strictly between zero and one"
        )
    num_zr = 2 * int(np.prod(args.size))
    outside = [
        count
        for count in args.hydrogen_counts
        if count / (count + num_zr) > args.maximum_h_atomic_fraction
    ]
    if outside and not args.allow_outside_source_range:
        raise ValueError(
            f"hydrogen counts {outside} exceed the configured source-validity limit "
            f"of {args.maximum_h_atomic_fraction:.4f} atomic fraction; use "
            "--allow-outside-source-range only for an explicitly labeled stress test"
        )

    screening_model = load_local_energy_model(args.pair_config)
    dilute_model = LocalEnergyModel(
        provenance={"publication_ready": True, "model_name": "dilute_reference"}
    )
    results = []
    for temperature in args.temperatures:
        for n_hydrogen in args.hydrogen_counts:
            results.append(
                run_case(
                    "dilute",
                    dilute_model,
                    screening_model,
                    temperature,
                    n_hydrogen,
                    args,
                    0.0,
                )
            )
            for scale in args.pair_energy_scales:
                scaled_model = _scaled_model(screening_model, scale)
                results.append(
                    run_case(
                        f"andolina2022_pair_screening_x{scale:g}",
                        scaled_model,
                        screening_model,
                        temperature,
                        n_hydrogen,
                        args,
                        scale,
                    )
                )
    payload = {
        "schema_version": 1,
        "experiment": "hcp_zrh_dissolved_h_precursor_screening",
        "claim_boundary": "H-rich precursor connectivity, not hydride nucleation or phase fraction",
        "publication_ready_parameters": bool(
            screening_model.provenance.get("publication_ready", False)
        ),
        "pair_model": screening_model.metadata(),
        "pair_energy_scales": list(map(float, args.pair_energy_scales)),
        "supercell": list(map(int, args.size)),
        "burn_in_steps": args.burn_in_steps,
        "sample_interval": args.sample_interval,
        "samples_per_replica": args.samples,
        "replicas": args.replicas,
        "source_validity": {
            "maximum_h_atomic_fraction": args.maximum_h_atomic_fraction,
            "outside_range_allowed": bool(args.allow_outside_source_range),
        },
        "results": results,
        "quality_gate": {
            "passed": bool(
                all(row["quality_gate"]["energy_bookkeeping"] for row in results)
            )
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    for row in results:
        metrics = row["metrics"]
        print(
            f"{row['case']:32s} T={row['temperature_K']:6.1f} K "
            f"nH={row['n_hydrogen']:3d} "
            f"largest={metrics['largest_attractive_cluster_fraction']['mean']:.4f} "
            f"TTn2={metrics['pair_count:TTn2_attractive']['mean']:.4f}"
        )
    print(f"wrote {args.output.resolve()}")
    if not payload["quality_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
