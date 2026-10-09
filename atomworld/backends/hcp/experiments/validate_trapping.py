#!/usr/bin/env python3
"""Validate the optional trap-rate correction against its exact CTMC state model.

This is a numerical mechanism test.  Its binding energy is intentionally
synthetic and must not be used as a zirconium-defect material parameter.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from atomworld.backends.hcp.envs.hcp_kmc import HCPHydrogenKMC
from atomworld.backends.hcp.lattices.hcp_interstitial import TETRAHEDRAL
from atomworld.backends.hcp.physics.local_energy import LocalEnergyModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", nargs=3, type=int, default=(3, 3, 3))
    parser.add_argument("--temperature", type=float, default=600.0)
    parser.add_argument("--binding-energy", type=float, default=0.15)
    parser.add_argument("--burn-in-steps", type=int, default=5000)
    parser.add_argument("--sample-steps", type=int, default=50000)
    parser.add_argument("--replicas", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2903)
    parser.add_argument("--max-trap-fraction-error", type=float, default=0.015)
    parser.add_argument("--max-site-type-fraction-error", type=float, default=0.02)
    parser.add_argument("--enforce", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/physics_validation/trapping_mechanism_validation.json"),
    )
    return parser.parse_args()


def exact_generator(env):
    lattice = env.lattice
    generator = np.zeros(
        (lattice.num_interstitial_sites, lattice.num_interstitial_sites),
        dtype=np.float64,
    )
    for source in range(lattice.num_interstitial_sites):
        lattice.set_hydrogen_sites([source])
        catalog = env.event_catalog()
        for local_event, rate in enumerate(catalog.rates[0]):
            if rate <= 0.0:
                continue
            target = int(lattice.event_targets[source, local_event])
            generator[source, target] += float(rate)
        generator[source, source] = -float(np.sum(generator[source]))
    return generator


def stationary_distribution(generator):
    matrix = generator.T.copy()
    rhs = np.zeros(generator.shape[0], dtype=np.float64)
    matrix[-1, :] = 1.0
    rhs[-1] = 1.0
    probability = np.linalg.solve(matrix, rhs)
    probability = np.maximum(probability, 0.0)
    return probability / probability.sum()


def simulate_residence(env, initial_site, burn_in_steps, sample_steps):
    env.lattice.set_hydrogen_sites([initial_site])
    env.energy_eV = env.calculate_energy_eV()
    env.run(burn_in_steps)
    residence = np.zeros(env.lattice.num_interstitial_sites, dtype=np.float64)
    for _ in range(sample_steps):
        source = int(env.lattice.hydrogen_sites[0])
        result = env.step()
        residence[source] += result.delta_time_s
    return residence


def main():
    args = parse_args()
    if args.temperature <= 0.0 or args.binding_energy <= 0.0:
        raise ValueError("temperature and binding energy must be positive")
    if args.burn_in_steps < 0 or args.sample_steps <= 0 or args.replicas <= 0:
        raise ValueError(
            "burn-in must be non-negative; sample-steps and replicas must be positive"
        )

    probe = HCPHydrogenKMC(
        size=args.size, temperature_K=args.temperature, seed=args.seed
    )
    trap_site = int(np.flatnonzero(probe.lattice.site_types == TETRAHEDRAL)[0])
    energy_model = LocalEnergyModel(
        trap_site_energies_eV={trap_site: -args.binding_energy},
        provenance={
            "publication_ready": False,
            "purpose": "synthetic numerical validation of trap kinetics",
        },
    )
    env = HCPHydrogenKMC(
        size=args.size,
        temperature_K=args.temperature,
        seed=args.seed,
        energy_model=energy_model,
    )
    generator = exact_generator(env)
    exact = stationary_distribution(generator)
    residence = np.zeros(env.lattice.num_interstitial_sites, dtype=np.float64)
    for replica in range(args.replicas):
        replica_env = HCPHydrogenKMC(
            size=args.size,
            temperature_K=args.temperature,
            seed=args.seed + replica,
            energy_model=energy_model,
        )
        residence += simulate_residence(
            replica_env, trap_site, args.burn_in_steps, args.sample_steps
        )
    simulated = residence / residence.sum()

    t_mask = env.lattice.site_types == TETRAHEDRAL
    exact_t = float(np.sum(exact[t_mask]))
    simulated_t = float(np.sum(simulated[t_mask]))
    trap_error = float(abs(simulated[trap_site] - exact[trap_site]))
    site_type_error = float(abs(simulated_t - exact_t))
    checks = {
        "trap_fraction": trap_error <= args.max_trap_fraction_error,
        "site_type_fraction": site_type_error <= args.max_site_type_fraction_error,
    }
    payload = {
        "schema_version": 1,
        "experiment": "synthetic_hcp_trapping_mechanism_validation",
        "publication_ready_parameters": False,
        "temperature_K": args.temperature,
        "binding_energy_eV": args.binding_energy,
        "supercell": list(map(int, args.size)),
        "trap_site": trap_site,
        "burn_in_steps": args.burn_in_steps,
        "sample_steps": args.sample_steps,
        "replicas": args.replicas,
        "total_sample_steps": args.sample_steps * args.replicas,
        "exact": {
            "trap_fraction": float(exact[trap_site]),
            "T_fraction": exact_t,
            "O_fraction": float(1.0 - exact_t),
        },
        "simulated": {
            "trap_fraction": float(simulated[trap_site]),
            "T_fraction": simulated_t,
            "O_fraction": float(1.0 - simulated_t),
        },
        "absolute_error": {
            "trap_fraction": trap_error,
            "site_type_fraction": site_type_error,
        },
        "quality_gate": {"passed": bool(all(checks.values())), "checks": checks},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        f"trap fraction sim={payload['simulated']['trap_fraction']:.6f} "
        f"exact={payload['exact']['trap_fraction']:.6f}; "
        f"T fraction sim={simulated_t:.6f} exact={exact_t:.6f}; "
        f"gate={'PASS' if payload['quality_gate']['passed'] else 'FAIL'}"
    )
    print(f"wrote {args.output.resolve()}")
    if args.enforce and not payload["quality_gate"]["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
