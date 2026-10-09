"""Torch binding between HCP policy observations and the AtomWorld model."""

from __future__ import annotations

import numpy as np
import torch

from atomworld.backends.hcp.analysis.clusters import hydrogen_cluster_metrics
from atomworld.backends.hcp.envs.policy_env import HCPPolicyEnvironment
from atomworld.backends.hcp.models import AtomWorldModelConfig, MemoryRestoredAtomWorld


def hcp_mesoscale_descriptors(env):
    """Return nine auditable HCP descriptors for the centralized critic."""

    if not isinstance(env, HCPPolicyEnvironment):
        raise TypeError("env must be an HCPPolicyEnvironment")
    kmc = env.kmc
    clusters = hydrogen_cluster_metrics(kmc.lattice, kmc.energy_model)
    attractive_pairs = sum(
        count
        for shell, count in clusters["pair_counts"].items()
        if any(
            candidate.label == shell and candidate.interaction_energy_eV < 0.0
            for candidate in kmc.energy_model.pair_shells
        )
    )
    n_hydrogen = max(int(kmc.num_agents), 1)
    h_atomic_fraction = kmc.num_agents / (kmc.lattice.num_zr_sites + kmc.num_agents)
    diagnostics = kmc.kinetic_diagnostics()
    return np.asarray(
        [
            kmc.temperature_K / 1000.0,
            h_atomic_fraction,
            clusters["T_fraction"],
            clusters["O_fraction"],
            attractive_pairs / n_hydrogen,
            clusters["largest_attractive_cluster_fraction"],
            clusters["isolated_hydrogen_fraction"],
            diagnostics["immediate_reverse_fraction"],
            kmc.energy_eV / n_hydrogen,
        ],
        dtype=np.float32,
    )


class HCPAtomWorldBinding:
    """Tensorize HCP state and execute shared-memory global event arbitration."""

    def __init__(self, env, *, config=None, device="cpu", model=None):
        if not isinstance(env, HCPPolicyEnvironment):
            raise TypeError("env must be an HCPPolicyEnvironment")
        self.env = env
        self.device = torch.device(device)
        self.config = config or AtomWorldModelConfig(
            event_feature_dim=18,
            history_feature_dim=12,
            species_count=3,
            defect_type_count=4,
            mesoscale_dim=9,
        )
        if self.config.event_feature_dim != 18:
            raise ValueError("HCP event_feature_dim must be 18")
        if self.config.history_feature_dim != 12:
            raise ValueError("HCP history_feature_dim must be 12")
        if self.config.species_count < 3 or self.config.defect_type_count < 4:
            raise ValueError(
                "HCP observation vocabularies require 3 species and 4 defect types"
            )
        if self.config.mesoscale_dim != 9:
            raise ValueError("HCP mesoscale descriptor dimension must be 9")
        self.model = (
            MemoryRestoredAtomWorld(self.config) if model is None else model
        ).to(self.device)

    def _tensor(self, value, *, integer=False, boolean=False):
        if boolean:
            dtype = torch.bool
        elif integer:
            dtype = torch.long
        else:
            dtype = torch.float32
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def model_inputs(self, *, previous_long_state=None, update_mask=None):
        raw = self.env.get_model_inputs()
        agents = self.env.num_agents
        if previous_long_state is None:
            previous_long_state = self.model.memory.initial_state(
                1, agents, device=self.device, dtype=torch.float32
            )
        if update_mask is None:
            update_mask = np.ones(agents, dtype=bool)
        update_mask = np.asarray(update_mask, dtype=bool)
        if update_mask.shape != (agents,):
            raise ValueError(f"update_mask must have shape {(agents,)}")
        return {
            "dense_species": self._tensor(raw["dense_species"][None], integer=True),
            "dense_positions": self._tensor(raw["dense_positions"][None]),
            "dense_defect_types": self._tensor(
                raw["dense_defect_types"][None], integer=True
            ),
            "dense_connectivity": self._tensor(raw["dense_connectivity"][None]),
            "dense_mask": self._tensor(raw["dense_mask"][None], boolean=True),
            "sparse_tokens": self._tensor(raw["sparse_tokens"][None]),
            "sparse_mask": self._tensor(raw["sparse_mask"][None], boolean=True),
            "event_history": self._tensor(raw["event_history"][None]),
            "history_mask": self._tensor(raw["history_mask"][None], boolean=True),
            "previous_long_state": previous_long_state,
            "update_mask": self._tensor(update_mask[None], boolean=True),
            "event_features": self._tensor(raw["event_features"][None]),
            "legal_mask": self._tensor(raw["legal_mask"][None], boolean=True),
            "agent_mask": self._tensor(raw["agent_mask"][None], boolean=True),
            "energy": self._tensor([float(raw["energy_eV"])]),
            "mesoscale_descriptors": self._tensor(
                hcp_mesoscale_descriptors(self.env)[None]
            ),
        }

    def forward(
        self,
        *,
        previous_long_state=None,
        update_mask=None,
        policy_temperature=1.0,
        compute_value=True,
    ):
        inputs = self.model_inputs(
            previous_long_state=previous_long_state, update_mask=update_mask
        )
        return self.model(
            **inputs,
            policy_temperature=float(policy_temperature),
            compute_value=compute_value,
        )

    def select_event(
        self,
        *,
        previous_long_state=None,
        update_mask=None,
        deterministic=False,
        policy_temperature=1.0,
    ):
        with torch.no_grad():
            output = self.forward(
                previous_long_state=previous_long_state,
                update_mask=update_mask,
                policy_temperature=policy_temperature,
                compute_value=False,
            )
            if deterministic:
                event = torch.argmax(output.global_logits, dim=-1)
            else:
                event = output.distribution.sample()
        return int(event.item()), output.long_memory
