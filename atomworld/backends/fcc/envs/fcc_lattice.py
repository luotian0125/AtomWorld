"""FCC lattice construction for vacancy-mediated Ni-Al-Cr LKMC.

Coordinates are stored in units of ``a / 2`` inside a dense periodic array.
Valid FCC sites have an even coordinate sum and correspond to the four basis
sites ``(0, 0, 0)``, ``(0, 1, 1)``, ``(1, 0, 1)``, and ``(1, 1, 0)``.
"""

from itertools import permutations, product

import numpy as np


NI = 0
AL = 1
CR = 2
VACANCY = 3
INVALID = -1
PADDING = 4


def _signed_permutations(values):
    """Return unique signed permutations of a non-negative integer tuple."""
    rows = set()
    for perm in set(permutations(values)):
        nonzero = [index for index, value in enumerate(perm) if value != 0]
        for signs in product((-1, 1), repeat=len(nonzero)):
            row = list(perm)
            for index, sign in zip(nonzero, signs):
                row[index] *= sign
            rows.add(tuple(row))
    return np.asarray(sorted(rows), dtype=np.int32)


class Lattice:
    """Periodic conventional-cell FCC lattice with Ni balance composition."""

    species_names = ("Ni", "Al", "Cr", "Vacancy")
    vacancy_type = VACANCY
    padding_type = PADDING

    def __init__(self, args):
        self.size = tuple(int(value) for value in args.lattice_size)
        self.Al_nums = int(getattr(args, "lattice_al_nums", 0))
        self.Cr_nums = int(getattr(args, "lattice_cr_nums", 0))
        self.V_nums = int(getattr(args, "lattice_v_nums", 1))
        # Compatibility with legacy utilities. This means Al + Cr, not Cu.
        self.Cu_nums = self.Al_nums + self.Cr_nums

        self.NN1 = _signed_permutations((1, 1, 0))  # 12, distance^2 = 2
        self.NN2 = _signed_permutations((2, 0, 0))  # 6, distance^2 = 4
        self.NN3 = _signed_permutations((2, 1, 1))  # 24, distance^2 = 6
        self.NN4 = _signed_permutations((2, 2, 0))  # 12, distance^2 = 8
        self.neighbor_shells = (self.NN1, self.NN2, self.NN3, self.NN4)
        self.events_per_agent = len(self.NN1)
        self.spatial_obs_dim = sum(len(shell) for shell in self.neighbor_shells)

        self.coords = self.generate_lattice_coordinates()
        self.lattice = None
        self.vacancy_pos = None
        self.al_pos = None
        self.cr_pos = None
        self.solute_pos = None
        self.solute_pos_index = None
        self.init_lattice()

    @property
    def total_sites(self):
        nx, ny, nz = self.size
        return 4 * nx * ny * nz

    def generate_lattice_coordinates(self):
        nx, ny, nz = self.size
        conventional = np.indices((nx, ny, nz)).reshape(3, -1).T * 2
        basis = np.asarray(
            ((0, 0, 0), (0, 1, 1), (1, 0, 1), (1, 1, 0)),
            dtype=np.int32,
        )
        return (conventional[:, None, :] + basis[None, :, :]).reshape(-1, 3)

    def _refresh_species_positions(self):
        atom_types = self.lattice[tuple(self.coords.T)]
        self.vacancy_pos = self.coords[atom_types == VACANCY].copy()
        self.al_pos = self.coords[atom_types == AL].copy()
        self.cr_pos = self.coords[atom_types == CR].copy()
        self.solute_pos = self.coords[(atom_types == AL) | (atom_types == CR)].copy()
        self.solute_pos_index = {
            tuple(map(int, position)): index for index, position in enumerate(self.solute_pos)
        }
        # Legacy aliases retained until plotting/reporting are renamed.
        self.cu_pos = self.solute_pos
        self.cu_pos_index = self.solute_pos_index

    def _build_cu_pos_index(self):
        """Backward-compatible alias for legacy state restoration."""
        self.solute_pos = np.asarray(self.cu_pos, dtype=np.int32)
        self.solute_pos_index = {
            tuple(map(int, position)): index for index, position in enumerate(self.solute_pos)
        }
        self.cu_pos = self.solute_pos
        self.cu_pos_index = self.solute_pos_index
        if self.lattice is not None:
            atom_types = self.lattice[tuple(self.coords.T)]
            self.al_pos = self.coords[atom_types == AL].copy()
            self.cr_pos = self.coords[atom_types == CR].copy()

    def update_cu_pos(self):
        """Backward-compatible alias; refreshes Al and Cr positions."""
        self._refresh_species_positions()

    def init_lattice(self):
        nx, ny, nz = self.size
        lattice_shape = (2 * nx, 2 * ny, 2 * nz)
        requested = self.Al_nums + self.Cr_nums + self.V_nums
        if requested > self.total_sites:
            raise ValueError(
                f"Al + Cr + vacancy count ({requested}) exceeds FCC site count ({self.total_sites})"
            )

        atom_list = np.asarray(
            [AL] * self.Al_nums
            + [CR] * self.Cr_nums
            + [VACANCY] * self.V_nums
            + [NI] * (self.total_sites - requested),
            dtype=np.int32,
        )
        np.random.shuffle(atom_list)

        lattice = np.full(lattice_shape, INVALID, dtype=np.int32)
        lattice[tuple(self.coords.T)] = atom_list
        self.lattice = lattice
        self._refresh_species_positions()

