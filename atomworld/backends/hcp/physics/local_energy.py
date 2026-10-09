"""Optional local-energy terms for H trapping and H-rich precursors."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairInteractionShell:
    """Distance-binned pair term in a minimal H occupation cluster expansion."""

    minimum_distance_angstrom: float
    maximum_distance_angstrom: float
    interaction_energy_eV: float
    source_site_type: str = "ANY"
    target_site_type: str = "ANY"
    label: str = "pair_shell"

    def __post_init__(self):
        if not 0.0 <= self.minimum_distance_angstrom < self.maximum_distance_angstrom:
            raise ValueError("pair-shell distance bounds must satisfy 0 <= min < max")
        if not np.isfinite(self.interaction_energy_eV):
            raise ValueError("pair interaction energy must be finite")
        if not str(self.label).strip():
            raise ValueError("pair-shell label must be non-empty")
        allowed = {"T", "O", "ANY"}
        if (
            self.source_site_type.upper() not in allowed
            or self.target_site_type.upper() not in allowed
        ):
            raise ValueError("pair-shell site types must be T, O, or ANY")

    def matches(self, source_type, target_type, distance_angstrom):
        source_type = str(source_type).upper()
        target_type = str(target_type).upper()
        required_source = str(self.source_site_type).upper()
        required_target = str(self.target_site_type).upper()
        direct = (required_source in {"ANY", source_type}) and (
            required_target in {"ANY", target_type}
        )
        reverse = (required_source in {"ANY", target_type}) and (
            required_target in {"ANY", source_type}
        )
        type_match = direct or reverse
        return bool(
            type_match
            and self.minimum_distance_angstrom
            <= distance_angstrom
            < self.maximum_distance_angstrom
        )


class LocalEnergyModel:
    """Sparse trap field plus pair occupation terms.

    Trap energies and attractive pair energies are negative.  The default
    model contains no extra terms and therefore exactly recovers the dilute
    Zhang et al. rate table.  Parameter provenance is deliberately kept
    outside this numerical mechanism.
    """

    def __init__(self, *, trap_site_energies_eV=None, pair_shells=(), provenance=None):
        self.trap_site_energies_eV = {
            int(site): float(energy)
            for site, energy in (trap_site_energies_eV or {}).items()
        }
        if not all(np.isfinite(value) for value in self.trap_site_energies_eV.values()):
            raise ValueError("trap energies must be finite")
        self.pair_shells = tuple(pair_shells)
        if not all(
            isinstance(shell, PairInteractionShell) for shell in self.pair_shells
        ):
            raise TypeError("pair_shells must contain PairInteractionShell values")
        self.provenance = dict(provenance or {})
        self._pair_energy_cache = {}

    @property
    def enabled(self):
        return bool(self.trap_site_energies_eV or self.pair_shells)

    def trap_energy_eV(self, site):
        return float(self.trap_site_energies_eV.get(int(site), 0.0))

    @staticmethod
    def _lattice_signature(lattice):
        return (
            tuple(lattice.size),
            float(lattice.lattice_a_angstrom),
            float(lattice.lattice_c_angstrom),
            int(lattice.num_interstitial_sites),
        )

    def _uncached_pair_energy_eV(self, lattice, first_site, second_site):
        distance = lattice.minimum_image_distance_angstrom(first_site, second_site)
        first_type = lattice.site_type_name(first_site)
        second_type = lattice.site_type_name(second_site)
        return float(
            sum(
                shell.interaction_energy_eV
                for shell in self.pair_shells
                if shell.matches(first_type, second_type, distance)
            )
        )

    def pair_energy_eV(self, lattice, first_site, second_site):
        first_site = int(first_site)
        second_site = int(second_site)
        if first_site == second_site:
            raise ValueError("a hydrogen cannot form a pair with itself")
        cache = self._pair_energy_cache.setdefault(self._lattice_signature(lattice), {})
        key = (min(first_site, second_site), max(first_site, second_site))
        if key in cache:
            return cache[key]
        energy = self._uncached_pair_energy_eV(lattice, first_site, second_site)
        cache[key] = energy
        return energy

    def total_extra_energy_eV(self, lattice):
        sites = list(map(int, lattice.hydrogen_sites))
        energy = sum(self.trap_energy_eV(site) for site in sites)
        for first_index, first_site in enumerate(sites):
            for second_site in sites[first_index + 1 :]:
                energy += self.pair_energy_eV(lattice, first_site, second_site)
        return float(energy)

    def event_extra_energy_change_eV(self, lattice, hydrogen_index, target_site):
        hydrogen_index = int(hydrogen_index)
        target_site = int(target_site)
        source_site = int(lattice.hydrogen_sites[hydrogen_index])
        trap_delta = self.trap_energy_eV(target_site) - self.trap_energy_eV(source_site)
        pair_delta = 0.0
        coordination_target = 0
        for other_index, other_site in enumerate(lattice.hydrogen_sites):
            if other_index == hydrogen_index:
                continue
            other_site = int(other_site)
            source_pair = self.pair_energy_eV(lattice, source_site, other_site)
            target_pair = self.pair_energy_eV(lattice, target_site, other_site)
            pair_delta += target_pair - source_pair
            if target_pair != 0.0:
                coordination_target += 1
        return {
            "trap_delta_eV": float(trap_delta),
            "pair_delta_eV": float(pair_delta),
            "coordination_target": int(coordination_target),
        }

    def metadata(self):
        return {
            "enabled": self.enabled,
            "trap_site_count": len(self.trap_site_energies_eV),
            "pair_shell_count": len(self.pair_shells),
            "trap_site_energies_eV": {
                str(site): energy
                for site, energy in sorted(self.trap_site_energies_eV.items())
            },
            "pair_shells": [asdict(shell) for shell in self.pair_shells],
            "provenance": dict(self.provenance),
        }


def load_local_energy_model(path):
    """Load an auditable local-energy projection from JSON."""
    path = Path(path).expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("material_system") != "hcp_zr_h":
        raise ValueError("local-energy material_system must be 'hcp_zr_h'")
    if payload.get("host_phase") != "alpha_zr":
        raise ValueError("local-energy host_phase must be 'alpha_zr'")
    if payload.get("energy_unit") != "eV":
        raise ValueError("local-energy energy_unit must be 'eV'")
    shells = [
        PairInteractionShell(
            minimum_distance_angstrom=float(row["minimum_distance_angstrom"]),
            maximum_distance_angstrom=float(row["maximum_distance_angstrom"]),
            interaction_energy_eV=float(row["interaction_energy_eV"]),
            source_site_type=str(row.get("source_site_type", "ANY")),
            target_site_type=str(row.get("target_site_type", "ANY")),
            label=str(row.get("label", "pair_shell")),
        )
        for row in payload.get("pair_shells", ())
    ]
    provenance = dict(payload.get("provenance", {}))
    provenance.update(
        {
            "config_path": str(path),
            "publication_ready": bool(payload.get("publication_ready", False)),
            "model_name": str(payload.get("name", path.stem)),
        }
    )
    return LocalEnergyModel(
        trap_site_energies_eV=payload.get("trap_site_energies_eV", {}),
        pair_shells=shells,
        provenance=provenance,
    )
