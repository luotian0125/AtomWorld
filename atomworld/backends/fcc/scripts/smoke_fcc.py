"""Small deterministic smoke run for the FCC Ni-Al-Cr physics kernel."""

import argparse
from types import SimpleNamespace

import numpy as np

from atomworld.backends.fcc.envs.kmc import KMC


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--parameter_set", default="mao2012_lkmc1")
    args = parser.parse_args()

    np.random.seed(args.seed)
    physics_args = SimpleNamespace(
        lattice_size=[8, 8, 8],
        lattice_al_nums=154,
        lattice_cr_nums=174,
        lattice_v_nums=1,
        temperature=873.0,
        attempt_frequency=None,
        min_barrier=0.0,
        fcc_parameter_set=args.parameter_set,
        fcc_parameter_file="",
        ni_migration_barrier=None,
        al_migration_barrier=None,
        cr_migration_barrier=None,
    )
    kmc = KMC(physics_args)
    initial_counts = np.bincount(kmc.lattice[tuple(kmc.coords.T)], minlength=4)
    initial_energy = float(kmc.calculate_system_energy())

    for _ in range(args.steps):
        if kmc.step() <= 0.0:
            raise RuntimeError("KMC became inactive before the requested event count")

    final_counts = np.bincount(kmc.lattice[tuple(kmc.coords.T)], minlength=4)
    result = {
        "steps": len(kmc.energy_history),
        "physical_time": float(kmc.time),
        "initial_energy": initial_energy,
        "final_energy": float(kmc.energy_history[-1]),
        "species_counts": final_counts.tolist(),
        "species_conserved": bool(np.array_equal(initial_counts, final_counts)),
        "rate_shape": list(kmc.calculate_diffusion_rate().shape),
    }
    print(result)
    if not result["species_conserved"]:
        raise RuntimeError("Species counts changed during vacancy exchange")


if __name__ == "__main__":
    main()
