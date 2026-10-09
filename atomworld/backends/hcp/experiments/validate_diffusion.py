#!/usr/bin/env python3
"""Validate passive HCP-LKMC diffusion against the analytical HCP random walk."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from atomworld.backends.hcp.envs.hcp_kmc import HCPHydrogenKMC
from atomworld.backends.hcp.lattices.hcp_interstitial import HOP_KINDS


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", nargs=3, type=int, default=(12, 12, 12))
    parser.add_argument(
        "--temperatures", nargs="+", type=float, default=(300, 400, 500, 600, 800, 1000)
    )
    parser.add_argument("--steps", type=int, default=50000)
    parser.add_argument(
        "--block-steps",
        type=int,
        default=1000,
        help="MSD sampling interval; block averaging reduces single-endpoint variance",
    )
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--seed", type=int, default=1701)
    parser.add_argument(
        "--max-component-relative-error",
        type=float,
        default=0.15,
        help="maximum accepted relative error for each of Da, Dc, and D3",
    )
    parser.add_argument(
        "--max-anisotropy-relative-error",
        type=float,
        default=0.20,
        help="maximum accepted relative error for Dc/Da",
    )
    parser.add_argument(
        "--max-site-fraction-absolute-error",
        type=float,
        default=0.02,
        help="maximum accepted absolute error in either T/O residence fraction",
    )
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="return a non-zero exit status when any validation gate fails",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/physics_validation/diffusion_validation.json"),
    )
    return parser.parse_args()


def run_temperature(temperature, args):
    basal_msd_sum = 0.0
    c_msd_sum = 0.0
    elapsed_sum = 0.0
    block_count = 0
    hop_totals = {kind: 0 for kind in HOP_KINDS}
    residence_totals = {"T": 0.0, "O": 0.0}
    analytical = None
    analytical_sites = None
    for replica in range(args.replicas):
        env = HCPHydrogenKMC(
            size=args.size,
            n_hydrogen=1,
            temperature_K=temperature,
            seed=args.seed + replica,
        )
        analytical = env.analytical_dilute_diffusivity_m2_per_s()
        analytical_sites = env.analytical_site_fractions()
        completed = 0
        while completed < args.steps:
            block_steps = min(args.block_steps, args.steps - completed)
            start_displacement = env.lattice.unwrapped_displacements_angstrom.copy()
            start_time = env.time_s
            env.run(block_steps)
            delta_m = (
                env.lattice.unwrapped_displacements_angstrom - start_displacement
            ) * 1.0e-10
            delta_time = env.time_s - start_time
            basal_msd_sum += float(np.mean(np.sum(delta_m[:, :2] ** 2, axis=1)))
            c_msd_sum += float(np.mean(delta_m[:, 2] ** 2))
            elapsed_sum += float(delta_time)
            block_count += 1
            completed += block_steps
        for kind in HOP_KINDS:
            hop_totals[kind] += int(env.hop_counts[kind])
        for site_type in ("T", "O"):
            residence_totals[site_type] += float(env.site_residence_time_s[site_type])
    d_a = basal_msd_sum / max(4.0 * elapsed_sum, 1.0e-300)
    d_c = c_msd_sum / max(2.0 * elapsed_sum, 1.0e-300)
    simulated = {
        "D_a": float(d_a),
        "D_c": float(d_c),
        "D_3d": float((2.0 * d_a + d_c) / 3.0),
        "D_c_over_D_a": float(d_c / d_a),
    }
    relative_error = {
        key: float(
            abs(simulated[key] - analytical[key]) / max(abs(analytical[key]), 1.0e-300)
        )
        for key in ("D_a", "D_c", "D_3d")
    }
    total_hops = sum(hop_totals.values())
    total_residence = sum(residence_totals.values())
    simulated_sites = {
        key: float(value / max(total_residence, 1.0e-300))
        for key, value in residence_totals.items()
    }
    anisotropy_relative_error = float(
        abs(simulated["D_c_over_D_a"] - analytical["D_c_over_D_a"])
        / max(abs(analytical["D_c_over_D_a"]), 1.0e-300)
    )
    site_fraction_absolute_error = {
        key: float(abs(simulated_sites[key] - analytical_sites[key]))
        for key in ("T", "O")
    }
    checks = {
        "diffusion_components": bool(
            max(relative_error.values()) <= args.max_component_relative_error
        ),
        "diffusion_anisotropy": bool(
            anisotropy_relative_error <= args.max_anisotropy_relative_error
        ),
        "site_residence": bool(
            max(site_fraction_absolute_error.values())
            <= args.max_site_fraction_absolute_error
        ),
    }
    return {
        "temperature_K": float(temperature),
        "steps_per_replica": int(args.steps),
        "replicas": int(args.replicas),
        "block_steps": int(args.block_steps),
        "block_count": int(block_count),
        "analytical_m2_per_s": analytical,
        "simulated_m2_per_s": simulated,
        "relative_error": relative_error,
        "anisotropy_relative_error": anisotropy_relative_error,
        "analytical_site_fractions": analytical_sites,
        "simulated_site_fractions": simulated_sites,
        "site_fraction_absolute_error": site_fraction_absolute_error,
        "hop_fractions": {
            key: float(value / max(total_hops, 1)) for key, value in hop_totals.items()
        },
        "quality_gate": {"passed": bool(all(checks.values())), "checks": checks},
    }


def main():
    args = parse_args()
    if args.steps <= 0 or args.replicas <= 0 or args.block_steps <= 0:
        raise ValueError("steps, replicas, and block-steps must be positive")
    tolerances = (
        args.max_component_relative_error,
        args.max_anisotropy_relative_error,
        args.max_site_fraction_absolute_error,
    )
    if any(value <= 0.0 for value in tolerances):
        raise ValueError("validation tolerances must be positive")
    results = [run_temperature(temperature, args) for temperature in args.temperatures]
    payload = {
        "schema_version": 1,
        "experiment": "hcp_zrh_dilute_diffusion_validation",
        "supercell": list(map(int, args.size)),
        "acceptance_thresholds": {
            "max_component_relative_error": args.max_component_relative_error,
            "max_anisotropy_relative_error": args.max_anisotropy_relative_error,
            "max_site_fraction_absolute_error": args.max_site_fraction_absolute_error,
        },
        "results": results,
        "quality_gate": {
            "passed": bool(all(row["quality_gate"]["passed"] for row in results))
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    for row in results:
        temperature = row["temperature_K"]
        simulated = row["simulated_m2_per_s"]
        analytical = row["analytical_m2_per_s"]
        print(
            f"T={temperature:7.1f} K  "
            f"D3 sim={simulated['D_3d']:.6e}  ref={analytical['D_3d']:.6e}  "
            f"Dc/Da sim={simulated['D_c_over_D_a']:.4f}  ref={analytical['D_c_over_D_a']:.4f}  "
            f"gate={'PASS' if row['quality_gate']['passed'] else 'FAIL'}"
        )
    print(f"wrote {args.output.resolve()}")
    if args.enforce and not payload["quality_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
