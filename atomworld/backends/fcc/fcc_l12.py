#!/usr/bin/env python
"""L1_2 (gamma-prime) order and particle metrics for conventional-cell FCC data.

The detector works on the four simple-cubic basis grids of an FCC lattice.  It
is invariant to the four possible L1_2 variants: the locally Al-rich basis is
chosen independently in every conventional cell.  Spatial averaging and all
connected-component operations are periodic.
"""

from dataclasses import dataclass
from itertools import product

import numpy as np

from atomworld.backends.fcc.envs.fcc_lattice import AL, CR, NI, VACANCY


BASIS_SLICES = (
    (slice(0, None, 2), slice(0, None, 2), slice(0, None, 2)),
    (slice(0, None, 2), slice(1, None, 2), slice(1, None, 2)),
    (slice(1, None, 2), slice(0, None, 2), slice(1, None, 2)),
    (slice(1, None, 2), slice(1, None, 2), slice(0, None, 2)),
)


@dataclass(frozen=True)
class L12DetectorConfig:
    radius_cells: int = 1
    order_threshold: float = 0.30
    min_al_fraction: float = 0.11
    max_al_fraction: float = 0.35
    min_ni_fraction: float = 0.58
    min_cluster_atoms: int = 32

    @property
    def min_cluster_cells(self):
        return max(1, int(np.ceil(self.min_cluster_atoms / 4.0)))


def _basis_grids(lattice):
    lattice = np.asarray(lattice)
    if lattice.ndim != 3 or any(size % 2 for size in lattice.shape):
        raise ValueError("FCC dense lattice must be three-dimensional with even dimensions")
    grids = np.stack([lattice[index] for index in BASIS_SLICES], axis=0)
    if len(set(grid.shape for grid in grids)) != 1:
        raise ValueError("FCC basis grids have inconsistent shapes")
    return grids


def _periodic_box_mean(values, radius):
    values = np.asarray(values, dtype=np.float32)
    radius = max(0, int(radius))
    if radius == 0:
        return values.copy()
    total = np.zeros_like(values, dtype=np.float32)
    shifts = range(-radius, radius + 1)
    for shift in product(shifts, repeat=3):
        total += np.roll(values, shift=shift, axis=(-3, -2, -1))
    return total / float((2 * radius + 1) ** 3)


def _periodic_components(mask):
    """Return component labels and sizes for a periodic 6-neighbour cell mask."""
    mask = np.asarray(mask, dtype=bool)
    shape = mask.shape
    active = np.flatnonzero(mask.reshape(-1))
    if active.size == 0:
        return np.full(shape, -1, dtype=np.int32), np.empty(0, dtype=np.int32)

    lookup = {int(site): index for index, site in enumerate(active)}
    parent = np.arange(active.size, dtype=np.int32)
    sizes = np.ones(active.size, dtype=np.int32)

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = int(parent[index])
        return index

    def union(left, right):
        left, right = find(left), find(right)
        if left == right:
            return
        if sizes[left] < sizes[right]:
            left, right = right, left
        parent[right] = left
        sizes[left] += sizes[right]

    coords = np.column_stack(np.unravel_index(active, shape))
    dims = np.asarray(shape, dtype=np.int64)
    for axis in range(3):
        neighbours = coords.copy()
        neighbours[:, axis] = (neighbours[:, axis] + 1) % dims[axis]
        neighbour_linear = np.ravel_multi_index(neighbours.T, shape)
        for index, site in enumerate(neighbour_linear):
            other = lookup.get(int(site))
            if other is not None:
                union(index, other)

    roots = np.asarray([find(index) for index in range(active.size)], dtype=np.int32)
    unique_roots, inverse, component_sizes = np.unique(
        roots, return_inverse=True, return_counts=True
    )
    del unique_roots
    labels = np.full(int(np.prod(shape)), -1, dtype=np.int32)
    labels[active] = inverse.astype(np.int32)
    return labels.reshape(shape), component_sizes.astype(np.int32)


def analyze_l12_lattice(lattice, config=None, return_masks=False):
    """Measure variant-invariant local L1_2 order and periodic particles.

    A conventional cell is a gamma-prime candidate when its smoothed local
    composition is physically compatible with Ni3(Al,Cr) and its best L1_2
    sublattice contrast exceeds ``order_threshold``.  Particle sizes are
    reported in atoms (four FCC sites per conventional cell).
    """
    config = config or L12DetectorConfig()
    basis = _basis_grids(lattice)
    al = _periodic_box_mean(basis == AL, config.radius_cells)
    ni = _periodic_box_mean(basis == NI, config.radius_cells)
    cr = _periodic_box_mean(basis == CR, config.radius_cells)

    al_other = (al.sum(axis=0, keepdims=True) - al) / 3.0
    ni_other = (ni.sum(axis=0, keepdims=True) - ni) / 3.0
    variant_scores = 0.5 * ((al - al_other) + (ni_other - ni))
    best_variant = np.argmax(variant_scores, axis=0).astype(np.int8)
    best_score = np.max(variant_scores, axis=0)

    local_al = al.mean(axis=0)
    local_ni = ni.mean(axis=0)
    local_cr = cr.mean(axis=0)
    candidate = (
        (best_score >= float(config.order_threshold))
        & (local_al >= float(config.min_al_fraction))
        & (local_al <= float(config.max_al_fraction))
        & (local_ni >= float(config.min_ni_fraction))
    )

    labels, component_cells = _periodic_components(candidate)
    stable_ids = np.flatnonzero(component_cells >= config.min_cluster_cells)
    stable_mask = np.isin(labels, stable_ids) if stable_ids.size else np.zeros_like(candidate)
    stable_cells = component_cells[stable_ids] if stable_ids.size else np.empty(0, dtype=np.int32)
    stable_atoms = stable_cells.astype(np.int64) * 4
    radii = np.power(3.0 * stable_cells / (4.0 * np.pi), 1.0 / 3.0) if stable_cells.size else np.empty(0)

    stable_species = basis[:, stable_mask] if np.any(stable_mask) else np.empty((4, 0), dtype=basis.dtype)
    stable_site_count = max(int(stable_species.size), 1)
    metrics = {
        "l12_order_mean": float(np.mean(best_score)),
        "l12_order_max": float(np.max(best_score)),
        "l12_candidate_fraction": float(np.mean(candidate)),
        "gamma_prime_cluster_count": int(stable_cells.size),
        "gamma_prime_number_density_per_cell": float(stable_cells.size / candidate.size),
        "gamma_prime_volume_fraction": float(stable_cells.sum() / candidate.size) if stable_cells.size else 0.0,
        "gamma_prime_largest_atoms": int(stable_atoms.max()) if stable_atoms.size else 0,
        "gamma_prime_mean_atoms": float(stable_atoms.mean()) if stable_atoms.size else 0.0,
        "gamma_prime_mean_radius_cells": float(radii.mean()) if radii.size else 0.0,
        "gamma_prime_max_radius_cells": float(radii.max()) if radii.size else 0.0,
        "gamma_prime_mean_radius_cubed": float(np.mean(radii ** 3)) if radii.size else 0.0,
        "gamma_prime_ni_fraction": float(np.count_nonzero(stable_species == NI) / stable_site_count),
        "gamma_prime_al_fraction": float(np.count_nonzero(stable_species == AL) / stable_site_count),
        "gamma_prime_cr_fraction": float(np.count_nonzero(stable_species == CR) / stable_site_count),
        "gamma_prime_vacancy_fraction": float(np.count_nonzero(stable_species == VACANCY) / stable_site_count),
    }
    if return_masks:
        return metrics, {
            "candidate_mask": candidate,
            "stable_mask": stable_mask,
            "best_variant": best_variant,
            "best_score": best_score,
            "component_sizes_cells": component_cells,
            "stable_component_sizes_cells": stable_cells,
        }
    return metrics


def make_synthetic_l12_lattice(size, kind, seed=0, al_fraction=0.075, cr_fraction=0.085,
                               antisite_fraction=0.0):
    """Build compact synthetic structures used by the experiment preflight."""
    size = tuple(int(value) for value in size)
    rng = np.random.default_rng(seed)
    shape = tuple(2 * value for value in size)
    lattice = np.full(shape, -1, dtype=np.int32)
    grids = np.full((4, *size), NI, dtype=np.int32)
    if kind == "perfect_l12":
        grids[0] = AL
    elif kind == "partial_l12":
        grids[0] = AL
        if antisite_fraction > 0.0:
            flip_b = rng.random(size) < antisite_fraction
            grids[0][flip_b] = NI
            for basis_index in range(1, 4):
                flip_a = rng.random(size) < (antisite_fraction / 3.0)
                grids[basis_index][flip_a] = AL
    elif kind == "random_gamma":
        values = rng.choice(
            np.asarray([NI, AL, CR], dtype=np.int32),
            size=4 * int(np.prod(size)),
            p=[1.0 - al_fraction - cr_fraction, al_fraction, cr_fraction],
        )
        grids = values.reshape((4, *size))
    else:
        raise ValueError(f"Unknown synthetic structure kind: {kind}")
    for index, basis_slice in enumerate(BASIS_SLICES):
        lattice[basis_slice] = grids[index]
    return lattice


def validate_l12_detector(config=None, size=(12, 12, 12), random_seeds=range(8)):
    config = config or L12DetectorConfig()
    perfect = analyze_l12_lattice(make_synthetic_l12_lattice(size, "perfect_l12"), config)
    partial = analyze_l12_lattice(
        make_synthetic_l12_lattice(size, "partial_l12", seed=7, antisite_fraction=0.20), config
    )
    random_rows = [
        analyze_l12_lattice(make_synthetic_l12_lattice(size, "random_gamma", seed=seed), config)
        for seed in random_seeds
    ]
    false_positive = max(row["gamma_prime_volume_fraction"] for row in random_rows)
    return {
        "config": config.__dict__,
        "perfect_l12": perfect,
        "partial_l12_20pct_antisite": partial,
        "random_gamma_max_stable_volume_fraction": float(false_positive),
        "random_gamma_max_candidate_fraction": float(
            max(row["l12_candidate_fraction"] for row in random_rows)
        ),
        "false_positive_requirement": 0.01,
        "passed": bool(
            false_positive < 0.01
            and perfect["gamma_prime_volume_fraction"] > 0.99
            and partial["gamma_prime_volume_fraction"] > 0.50
        ),
    }
