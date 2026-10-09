"""HCP-specific observation contracts for the AtomWorld policy stack."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import numpy as np

from atomworld.backends.hcp.lattices.hcp_interstitial import PADDING_SITE_TYPE


@dataclass(frozen=True)
class HCPAtomicObservation:
    """Fixed-size H-centred interstitial graphs compatible with AtomWorld."""

    species: np.ndarray
    positions_angstrom: np.ndarray
    defect_types: np.ndarray
    connectivity: np.ndarray
    mask: np.ndarray


def _minimum_image_vectors(lattice, source_site):
    delta = lattice.site_fractional - lattice.site_fractional[int(source_site)]
    box = np.asarray(lattice.size, dtype=np.float64)
    wrapped = delta - np.round(delta / box) * box
    shifts = np.asarray(list(product((-1, 0, 1), repeat=3)), dtype=np.float64)
    candidates = wrapped[:, None, :] + shifts[None, :, :] * box
    cartesian = candidates @ lattice.lattice_vectors_angstrom
    distances2 = np.sum(cartesian * cartesian, axis=-1)
    best = np.argmin(distances2, axis=1)
    rows = np.arange(lattice.num_interstitial_sites)
    return cartesian[rows, best], distances2[rows, best]


def build_hcp_atomic_observation(lattice, *, max_neighbors=48):
    """Build dense local graphs around each mobile H carrier.

    Species encode O/T/padding.  Defect types encode empty/self/other-H/padding.
    Connectivity is the fraction of the site's crystallographic event slots
    whose target is currently empty.
    """

    max_neighbors = int(max_neighbors)
    if max_neighbors <= 0:
        raise ValueError("max_neighbors must be positive")
    agents = lattice.n_hydrogen
    species = np.full((agents, max_neighbors), PADDING_SITE_TYPE, dtype=np.int64)
    positions = np.zeros((agents, max_neighbors, 3), dtype=np.float32)
    defect_types = np.full((agents, max_neighbors), 3, dtype=np.int64)
    connectivity = np.zeros((agents, max_neighbors), dtype=np.float32)
    mask = np.zeros((agents, max_neighbors), dtype=bool)

    for agent, center in enumerate(lattice.hydrogen_sites):
        vectors, distances2 = _minimum_image_vectors(lattice, int(center))
        selected = np.argsort(distances2, kind="stable")[:max_neighbors]
        count = len(selected)
        species[agent, :count] = lattice.site_types[selected]
        positions[agent, :count] = vectors[selected]
        occupants = lattice.occupancy[selected]
        codes = np.zeros(count, dtype=np.int64)
        codes[occupants == agent] = 1
        codes[(occupants >= 0) & (occupants != agent)] = 2
        defect_types[agent, :count] = codes
        targets = lattice.event_targets[selected]
        valid = targets >= 0
        safe_targets = np.where(valid, targets, 0)
        empty = valid & (lattice.occupancy[safe_targets] < 0)
        connectivity[agent, :count] = empty.sum(axis=1) / float(
            lattice.events_per_hydrogen
        )
        mask[agent, :count] = True
    return HCPAtomicObservation(
        species=species,
        positions_angstrom=positions,
        defect_types=defect_types,
        connectivity=connectivity,
        mask=mask,
    )


def build_hcp_sparse_context(lattice, energy_model, *, max_tokens=24):
    """Encode other H atoms as sparse, periodic long-range context tokens.

    Token columns are ``[site_type, dx/a, dy/a, dz/c, distance^2/a^2,
    attractive_shell]``.  The last flag describes the current pair relation;
    it is an order parameter for dissolved-H connectivity, not hydride phase.
    """

    max_tokens = int(max_tokens)
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    agents = lattice.n_hydrogen
    tokens = np.zeros((agents, max_tokens, 6), dtype=np.float32)
    mask = np.zeros((agents, max_tokens), dtype=bool)
    a = float(lattice.lattice_a_angstrom)
    c = float(lattice.lattice_c_angstrom)
    for agent, source in enumerate(lattice.hydrogen_sites):
        rows = []
        source_type = lattice.site_type_name(int(source))
        for other, target in enumerate(lattice.hydrogen_sites):
            if other == agent:
                continue
            vector = lattice.minimum_image_displacement_angstrom(source, target)
            distance = float(np.linalg.norm(vector))
            target_type = lattice.site_type_name(int(target))
            attractive = any(
                shell.interaction_energy_eV < 0.0
                and shell.matches(source_type, target_type, distance)
                for shell in energy_model.pair_shells
            )
            rows.append(
                (
                    distance,
                    np.asarray(
                        [
                            lattice.site_types[int(target)],
                            vector[0] / a,
                            vector[1] / a,
                            vector[2] / c,
                            (distance / a) ** 2,
                            float(attractive),
                        ],
                        dtype=np.float32,
                    ),
                )
            )
        rows.sort(key=lambda row: row[0])
        for index, (_, token) in enumerate(rows[:max_tokens]):
            tokens[agent, index] = token
            mask[agent, index] = True
    return tokens, mask


def build_hcp_event_record(transition, catalog, decision):
    """Map an executed H event to the 12-channel short-memory contract."""

    agent = int(transition.hydrogen_index)
    event = int(transition.local_event_index)
    features = catalog.features[agent, event]
    record = np.zeros(12, dtype=np.float32)
    record[0:3] = np.asarray(transition.displacement_angstrom, dtype=np.float32)
    record[3:7] = features[1:5]
    record[7] = float(transition.delta_energy_eV)
    record[8] = float(catalog.rates[agent, event] / max(catalog.total_rate, 1.0e-300))
    record[9] = float(transition.info.get("immediate_reverse", False))
    record[10] = float(decision.is_keyframe)
    record[11] = 1.0
    return record


@dataclass
class HCPEventHistory:
    num_agents: int
    horizon: int = 5
    feature_dim: int = 12

    def __post_init__(self):
        if self.num_agents <= 0 or self.horizon <= 0:
            raise ValueError("history dimensions must be positive")
        if self.feature_dim != 12:
            raise ValueError("the HCP event-history schema has exactly 12 channels")
        self.values = np.zeros(
            (self.num_agents, self.horizon, self.feature_dim), dtype=np.float32
        )
        self.mask = np.zeros((self.num_agents, self.horizon), dtype=bool)

    def update(self, affected_agents, record):
        affected = np.unique(np.asarray(affected_agents, dtype=np.int64))
        if affected.size == 0:
            return
        if np.any(affected < 0) or np.any(affected >= self.num_agents):
            raise IndexError("affected H index is outside the history table")
        record = np.asarray(record, dtype=np.float32)
        if record.shape != (self.feature_dim,):
            raise ValueError("event-history record has the wrong shape")
        self.values[affected, :-1] = self.values[affected, 1:]
        self.mask[affected, :-1] = self.mask[affected, 1:]
        self.values[affected, -1] = record
        self.mask[affected, -1] = True

    def state_dict(self):
        return {"values": self.values.copy(), "mask": self.mask.copy()}

    def load_state_dict(self, state):
        values = np.asarray(state["values"], dtype=np.float32)
        mask = np.asarray(state["mask"], dtype=bool)
        if values.shape != self.values.shape or mask.shape != self.mask.shape:
            raise ValueError("history checkpoint shape mismatch")
        self.values[...] = values
        self.mask[...] = mask
