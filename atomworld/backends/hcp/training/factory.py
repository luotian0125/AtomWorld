"""Validated construction of training and evaluation HCP environments."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

from atomworld.backends.hcp.envs.hcp_kmc import HCPHydrogenKMC
from atomworld.backends.hcp.envs.policy_env import HCPPolicyEnvironment
from atomworld.backends.hcp.physics.local_energy import LocalEnergyModel, load_local_energy_model


@dataclass(frozen=True)
class HCPEnvironmentConfig:
    size: tuple[int, int, int] = (4, 4, 4)
    n_hydrogen: int = 8
    temperature_K: float = 600.0
    parameter_file: str = ""
    local_energy_file: str = "configs/hcp/zr_h_pair_screening_andolina2022.json"
    pair_energy_scale: float = 1.0
    initial_site_type: str = "T"
    history_len: int = 5
    dense_max_neighbors: int = 48
    sparse_max_tokens: int = 24
    maximum_h_atomic_fraction: float = 0.077
    allow_outside_source_range: bool = False

    def __post_init__(self):
        if len(self.size) != 3 or any(int(value) < 3 for value in self.size):
            raise ValueError("HCP training size must contain three values >= 3")
        if self.n_hydrogen <= 0:
            raise ValueError("n_hydrogen must be positive")
        if self.temperature_K <= 0.0:
            raise ValueError("temperature_K must be positive")
        if self.pair_energy_scale <= 0.0:
            raise ValueError("pair_energy_scale must be positive")
        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        if self.dense_max_neighbors <= 0 or self.sparse_max_tokens <= 0:
            raise ValueError("observation sizes must be positive")
        if not 0.0 < self.maximum_h_atomic_fraction < 1.0:
            raise ValueError("maximum_h_atomic_fraction must lie in (0, 1)")
        if not self.allow_outside_source_range:
            fraction = self.h_atomic_fraction
            if fraction > self.maximum_h_atomic_fraction + 1.0e-12:
                raise ValueError(
                    f"H atomic fraction {fraction:.5f} exceeds the configured source "
                    f"boundary {self.maximum_h_atomic_fraction:.5f}"
                )

    @property
    def num_zr_sites(self):
        return 2 * int(self.size[0]) * int(self.size[1]) * int(self.size[2])

    @property
    def h_atomic_fraction(self):
        return self.n_hydrogen / (self.num_zr_sites + self.n_hydrogen)

    def metadata(self):
        payload = asdict(self)
        payload["size"] = list(self.size)
        payload["h_atomic_fraction"] = self.h_atomic_fraction
        return payload


def make_hcp_policy_environment(config, *, seed):
    if not isinstance(config, HCPEnvironmentConfig):
        raise TypeError("config must be an HCPEnvironmentConfig")
    energy_model = (
        load_local_energy_model(config.local_energy_file)
        if str(config.local_energy_file).strip()
        else LocalEnergyModel()
    )
    if energy_model.pair_shells and config.pair_energy_scale != 1.0:
        provenance = dict(energy_model.provenance)
        provenance["training_pair_energy_scale"] = config.pair_energy_scale
        energy_model = LocalEnergyModel(
            trap_site_energies_eV=energy_model.trap_site_energies_eV,
            pair_shells=tuple(
                replace(
                    shell,
                    interaction_energy_eV=(
                        shell.interaction_energy_eV * config.pair_energy_scale
                    ),
                )
                for shell in energy_model.pair_shells
            ),
            provenance=provenance,
        )
    kmc = HCPHydrogenKMC(
        size=config.size,
        n_hydrogen=config.n_hydrogen,
        temperature_K=config.temperature_K,
        seed=int(seed),
        parameter_file=config.parameter_file or None,
        initial_site_type=config.initial_site_type,
        energy_model=energy_model,
    )
    return HCPPolicyEnvironment(
        kmc,
        history_len=config.history_len,
        dense_max_neighbors=config.dense_max_neighbors,
        sparse_max_tokens=config.sparse_max_tokens,
    )
