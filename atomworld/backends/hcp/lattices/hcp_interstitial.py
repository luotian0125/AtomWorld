"""Periodic alpha-Zr host and its tetrahedral/octahedral H interstitial network."""

from __future__ import annotations

from itertools import product

import numpy as np


OCTAHEDRAL = 0
TETRAHEDRAL = 1
PADDING_SITE_TYPE = 2

HOP_KINDS = ("TT", "TO", "OT", "OO")
HOP_KIND_TO_ID = {name: index for index, name in enumerate(HOP_KINDS)}


class HCPInterstitialLattice:
    """Fixed HCP Zr host with an exclusive-occupation H interstitial lattice.

    Fractional coordinates use the primitive hexagonal vectors ``a1``, ``a2``,
    and ``c``.  The six-site interstitial basis is ordered O1, O2, T1--T4.
    Candidate topology is generated once from crystallographic nearest
    distances, then translated exactly over the periodic supercell.
    """

    basis_fractional = np.asarray(
        [
            (1.0 / 3.0, 2.0 / 3.0, 1.0 / 4.0),
            (1.0 / 3.0, 2.0 / 3.0, 3.0 / 4.0),
            (0.0, 0.0, 3.0 / 8.0),
            (0.0, 0.0, 5.0 / 8.0),
            (2.0 / 3.0, 1.0 / 3.0, 1.0 / 8.0),
            (2.0 / 3.0, 1.0 / 3.0, 7.0 / 8.0),
        ],
        dtype=np.float64,
    )
    basis_site_types = np.asarray(
        [OCTAHEDRAL, OCTAHEDRAL, TETRAHEDRAL, TETRAHEDRAL, TETRAHEDRAL, TETRAHEDRAL],
        dtype=np.int8,
    )
    zr_basis_fractional = np.asarray(
        [(0.0, 0.0, 0.0), (2.0 / 3.0, 1.0 / 3.0, 1.0 / 2.0)],
        dtype=np.float64,
    )
    events_per_hydrogen = 8

    def __init__(
        self,
        size=(6, 6, 6),
        *,
        lattice_a_angstrom=3.24,
        lattice_c_angstrom=5.18,
        n_hydrogen=1,
        seed=1,
        initial_site_type="T",
    ):
        self.size = tuple(int(value) for value in size)
        if len(self.size) != 3 or any(value < 3 for value in self.size):
            raise ValueError(
                "HCP supercell dimensions must each be at least 3 so periodic images do not alias "
                "distinct first-neighbor events"
            )
        self.lattice_a_angstrom = float(lattice_a_angstrom)
        self.lattice_c_angstrom = float(lattice_c_angstrom)
        if self.lattice_a_angstrom <= 0.0 or self.lattice_c_angstrom <= 0.0:
            raise ValueError("lattice constants must be positive")
        ratio = self.lattice_c_angstrom / self.lattice_a_angstrom
        if not 1.45 <= ratio <= 1.75:
            raise ValueError(
                f"unsupported HCP c/a ratio {ratio:.4f}; expected an alpha-Zr-like cell"
            )

        a = self.lattice_a_angstrom
        c = self.lattice_c_angstrom
        self.lattice_vectors_angstrom = np.asarray(
            [
                (a, 0.0, 0.0),
                (-0.5 * a, 0.5 * np.sqrt(3.0) * a, 0.0),
                (0.0, 0.0, c),
            ],
            dtype=np.float64,
        )
        self.num_cells = int(np.prod(self.size))
        self.num_zr_sites = 2 * self.num_cells
        self.num_interstitial_sites = 6 * self.num_cells
        self.num_tetrahedral_sites = 4 * self.num_cells
        self.num_octahedral_sites = 2 * self.num_cells

        self.cell_indices = np.indices(self.size, dtype=np.int32).reshape(3, -1).T
        self.site_cells = np.repeat(self.cell_indices, 6, axis=0)
        self.site_basis = np.tile(np.arange(6, dtype=np.int8), self.num_cells)
        self.site_types = self.basis_site_types[self.site_basis]
        self.site_fractional = (
            self.site_cells.astype(np.float64) + self.basis_fractional[self.site_basis]
        )
        self.site_positions_angstrom = (
            self.site_fractional @ self.lattice_vectors_angstrom
        )

        (
            self.basis_event_target,
            self.basis_event_cell_offset,
            self.basis_event_kind,
            self.basis_event_displacement_angstrom,
        ) = self._build_basis_event_templates()
        (
            self.event_targets,
            self.event_kinds,
            self.event_displacements_angstrom,
        ) = self._expand_event_templates()

        self.rng = np.random.default_rng(int(seed))
        self.n_hydrogen = int(n_hydrogen)
        if self.n_hydrogen <= 0:
            raise ValueError("n_hydrogen must be positive")
        initial_site_type = str(initial_site_type).strip().upper()
        if initial_site_type not in {"T", "O", "ANY"}:
            raise ValueError("initial_site_type must be 'T', 'O', or 'any'")
        if initial_site_type == "T":
            candidates = np.flatnonzero(self.site_types == TETRAHEDRAL)
        elif initial_site_type == "O":
            candidates = np.flatnonzero(self.site_types == OCTAHEDRAL)
        else:
            candidates = np.arange(self.num_interstitial_sites, dtype=np.int64)
        if self.n_hydrogen > len(candidates):
            raise ValueError(
                f"n_hydrogen={self.n_hydrogen} exceeds the {len(candidates)} available "
                f"{initial_site_type} sites"
            )

        self.hydrogen_sites = np.asarray(
            self.rng.choice(candidates, size=self.n_hydrogen, replace=False),
            dtype=np.int64,
        )
        self.occupancy = np.full(self.num_interstitial_sites, -1, dtype=np.int64)
        self.occupancy[self.hydrogen_sites] = np.arange(self.n_hydrogen, dtype=np.int64)
        self.unwrapped_displacements_angstrom = np.zeros(
            (self.n_hydrogen, 3), dtype=np.float64
        )

    def _build_basis_event_templates(self):
        target_basis = np.full((6, self.events_per_hydrogen), -1, dtype=np.int8)
        cell_offset = np.zeros((6, self.events_per_hydrogen, 3), dtype=np.int8)
        event_kind = np.full((6, self.events_per_hydrogen), -1, dtype=np.int8)
        displacement = np.zeros((6, self.events_per_hydrogen, 3), dtype=np.float64)

        for source_basis in range(6):
            source_type = int(self.basis_site_types[source_basis])
            if source_type == TETRAHEDRAL:
                groups = ((TETRAHEDRAL, 1, "TT"), (OCTAHEDRAL, 3, "TO"))
            else:
                groups = ((OCTAHEDRAL, 2, "OO"), (TETRAHEDRAL, 6, "OT"))
            slot = 0
            source = self.basis_fractional[source_basis]
            for requested_type, count, kind in groups:
                rows = []
                for candidate_basis in range(6):
                    if int(self.basis_site_types[candidate_basis]) != requested_type:
                        continue
                    for offset_tuple in product((-1, 0, 1), repeat=3):
                        if candidate_basis == source_basis and offset_tuple == (
                            0,
                            0,
                            0,
                        ):
                            continue
                        offset = np.asarray(offset_tuple, dtype=np.int8)
                        delta_fractional = (
                            self.basis_fractional[candidate_basis] + offset - source
                        )
                        delta_cartesian = (
                            delta_fractional @ self.lattice_vectors_angstrom
                        )
                        distance = float(np.linalg.norm(delta_cartesian))
                        rows.append(
                            (
                                distance,
                                int(candidate_basis),
                                tuple(map(int, offset_tuple)),
                                delta_cartesian,
                            )
                        )
                rows.sort(key=lambda row: (row[0], row[1], row[2]))
                selected = rows[:count]
                if len(selected) != count:
                    raise RuntimeError(f"could not construct {count} {kind} events")
                for _, candidate_basis, offset_tuple, delta_cartesian in selected:
                    target_basis[source_basis, slot] = candidate_basis
                    cell_offset[source_basis, slot] = offset_tuple
                    event_kind[source_basis, slot] = HOP_KIND_TO_ID[kind]
                    displacement[source_basis, slot] = delta_cartesian
                    slot += 1
        return target_basis, cell_offset, event_kind, displacement

    def _expand_event_templates(self):
        targets = np.full(
            (self.num_interstitial_sites, self.events_per_hydrogen), -1, dtype=np.int64
        )
        kinds = np.full_like(targets, -1, dtype=np.int8)
        displacements = np.zeros((*targets.shape, 3), dtype=np.float64)
        box = np.asarray(self.size, dtype=np.int32)
        for site in range(self.num_interstitial_sites):
            cell = self.site_cells[site]
            basis = int(self.site_basis[site])
            for slot in range(self.events_per_hydrogen):
                target_basis = int(self.basis_event_target[basis, slot])
                if target_basis < 0:
                    continue
                target_cell = (cell + self.basis_event_cell_offset[basis, slot]) % box
                targets[site, slot] = self.site_index(target_cell, target_basis)
                kinds[site, slot] = self.basis_event_kind[basis, slot]
                displacements[site, slot] = self.basis_event_displacement_angstrom[
                    basis, slot
                ]
        return targets, kinds, displacements

    def site_index(self, cell, basis):
        i, j, k = map(int, cell)
        nx, ny, nz = self.size
        if not (0 <= i < nx and 0 <= j < ny and 0 <= k < nz):
            raise IndexError(f"cell {(i, j, k)} is outside supercell {self.size}")
        basis = int(basis)
        if not 0 <= basis < 6:
            raise IndexError(f"basis={basis} is outside [0, 6)")
        return int(((i * ny + j) * nz + k) * 6 + basis)

    def decode_site(self, site):
        site = int(site)
        if not 0 <= site < self.num_interstitial_sites:
            raise IndexError(f"site={site} is outside the interstitial lattice")
        cell_flat, basis = divmod(site, 6)
        nx, ny, nz = self.size
        i, remainder = divmod(cell_flat, ny * nz)
        j, k = divmod(remainder, nz)
        return np.asarray((i, j, k), dtype=np.int32), int(basis)

    def site_type_name(self, site):
        return "T" if int(self.site_types[int(site)]) == TETRAHEDRAL else "O"

    def minimum_image_displacement_angstrom(self, source_site, target_site):
        """Return the shortest periodic displacement between arbitrary sites.

        Enumerating the 27 images around the component-wise wrapped image is
        inexpensive and remains correct for the non-orthogonal basal vectors.
        """
        source_site = int(source_site)
        target_site = int(target_site)
        if not 0 <= source_site < self.num_interstitial_sites:
            raise IndexError("source site is outside the interstitial lattice")
        if not 0 <= target_site < self.num_interstitial_sites:
            raise IndexError("target site is outside the interstitial lattice")
        delta = self.site_fractional[target_site] - self.site_fractional[source_site]
        box = np.asarray(self.size, dtype=np.float64)
        wrapped = delta - np.round(delta / box) * box
        candidates = np.asarray(
            [
                wrapped + np.asarray(shift, dtype=np.float64) * box
                for shift in product((-1, 0, 1), repeat=3)
            ]
        )
        cartesian = candidates @ self.lattice_vectors_angstrom
        return cartesian[int(np.argmin(np.linalg.norm(cartesian, axis=1)))].copy()

    def minimum_image_distance_angstrom(self, source_site, target_site):
        return float(
            np.linalg.norm(
                self.minimum_image_displacement_angstrom(source_site, target_site)
            )
        )

    def set_hydrogen_sites(self, sites, *, reset_displacements=True):
        """Set an explicit, validated H configuration for reproducible studies."""
        sites = np.asarray(sites, dtype=np.int64)
        if sites.shape != (self.n_hydrogen,):
            raise ValueError(
                f"expected {(self.n_hydrogen,)} hydrogen sites, got {sites.shape}"
            )
        if np.any(sites < 0) or np.any(sites >= self.num_interstitial_sites):
            raise IndexError("hydrogen site is outside the interstitial lattice")
        if np.unique(sites).size != self.n_hydrogen:
            raise ValueError(
                "multiple hydrogens cannot occupy the same interstitial site"
            )
        self.hydrogen_sites[...] = sites
        self.occupancy.fill(-1)
        self.occupancy[sites] = np.arange(self.n_hydrogen, dtype=np.int64)
        if reset_displacements:
            self.unwrapped_displacements_angstrom.fill(0.0)
        self.assert_consistent()

    def local_event_count(self, site):
        return int(np.count_nonzero(self.event_targets[int(site)] >= 0))

    def move_hydrogen(self, hydrogen_index, local_event_index):
        hydrogen_index = int(hydrogen_index)
        local_event_index = int(local_event_index)
        if not 0 <= hydrogen_index < self.n_hydrogen:
            raise IndexError("hydrogen index is outside the carrier table")
        if not 0 <= local_event_index < self.events_per_hydrogen:
            raise IndexError("local event index is outside [0, 8)")
        source = int(self.hydrogen_sites[hydrogen_index])
        target = int(self.event_targets[source, local_event_index])
        if target < 0:
            raise ValueError("candidate slot is padding, not a physical event")
        if self.occupancy[target] >= 0:
            raise ValueError("target interstitial site is already occupied")
        if self.occupancy[source] != hydrogen_index:
            raise RuntimeError("occupancy and hydrogen-site maps are inconsistent")
        self.occupancy[source] = -1
        self.occupancy[target] = hydrogen_index
        self.hydrogen_sites[hydrogen_index] = target
        displacement = self.event_displacements_angstrom[source, local_event_index]
        self.unwrapped_displacements_angstrom[hydrogen_index] += displacement
        return source, target, displacement.copy()

    def assert_consistent(self):
        occupied = np.flatnonzero(self.occupancy >= 0)
        if len(occupied) != self.n_hydrogen:
            raise RuntimeError("occupied-site count does not equal n_hydrogen")
        if np.unique(self.hydrogen_sites).size != self.n_hydrogen:
            raise RuntimeError("multiple hydrogens occupy the same site")
        if not np.array_equal(
            self.occupancy[self.hydrogen_sites],
            np.arange(self.n_hydrogen, dtype=np.int64),
        ):
            raise RuntimeError("occupancy and hydrogen-site maps disagree")
        return True
