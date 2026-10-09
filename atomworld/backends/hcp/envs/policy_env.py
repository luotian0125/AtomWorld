"""Clock-safe AtomWorld adapter around the validated HCP KMC kernel."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from atomworld.backends.hcp.envs.hcp_kmc import HCPHydrogenKMC
from atomworld.backends.hcp.observations import (
    HCPEventHistory,
    build_hcp_atomic_observation,
    build_hcp_event_record,
    build_hcp_sparse_context,
)
from atomworld.backends.hcp.physics.clocks import PolicyPoissonClock
from atomworld.backends.hcp.physics.keyframes import HCPKeyframeCriterion, KeyframeDecision


@dataclass(frozen=True)
class HCPPolicyTransition:
    physical_transition: object
    keyframe: KeyframeDecision
    affected_agents: np.ndarray
    long_memory_update_mask: np.ndarray
    policy_time_increment_s: float | None
    policy_elapsed_time_s: float | None


class HCPPolicyEnvironment:
    """Expose HCP events to AtomWorld without borrowing the passive KMC clock."""

    def __init__(
        self,
        kmc: HCPHydrogenKMC,
        *,
        history_len=5,
        dense_max_neighbors=48,
        sparse_max_tokens=24,
        keyframe_criterion=None,
        policy_clock=None,
    ):
        if not isinstance(kmc, HCPHydrogenKMC):
            raise TypeError("kmc must be an HCPHydrogenKMC")
        if policy_clock is not None and not isinstance(
            policy_clock, PolicyPoissonClock
        ):
            raise TypeError("policy_clock must be a certified PolicyPoissonClock")
        self.kmc = kmc
        self.history = HCPEventHistory(kmc.num_agents, int(history_len))
        self.dense_max_neighbors = int(dense_max_neighbors)
        self.sparse_max_tokens = int(sparse_max_tokens)
        self.keyframe_criterion = keyframe_criterion or HCPKeyframeCriterion()
        self.policy_clock = policy_clock

    @property
    def num_agents(self):
        return self.kmc.num_agents

    @property
    def events_per_agent(self):
        return self.kmc.events_per_agent

    def enumerate_events(self):
        return self.kmc.event_catalog()

    def get_model_inputs(self):
        dense = build_hcp_atomic_observation(
            self.kmc.lattice, max_neighbors=self.dense_max_neighbors
        )
        sparse, sparse_mask = build_hcp_sparse_context(
            self.kmc.lattice,
            self.kmc.energy_model,
            max_tokens=self.sparse_max_tokens,
        )
        catalog = self.enumerate_events()
        return {
            "dense_species": dense.species,
            "dense_positions": dense.positions_angstrom,
            "dense_defect_types": dense.defect_types,
            "dense_connectivity": dense.connectivity,
            "dense_mask": dense.mask,
            "sparse_tokens": sparse,
            "sparse_mask": sparse_mask,
            "event_history": self.history.values.copy(),
            "history_mask": self.history.mask.copy(),
            "event_features": catalog.features,
            "legal_mask": catalog.legal_mask,
            "agent_mask": np.ones(self.num_agents, dtype=bool),
            "energy_eV": np.asarray(self.kmc.energy_eV, dtype=np.float32),
        }

    def _affected_agents(self, source_site, target_site, moved_agent):
        affected = {int(moved_agent)}
        if not self.kmc.energy_model.pair_shells:
            return np.asarray(sorted(affected), dtype=np.int64)
        cutoff = max(
            shell.maximum_distance_angstrom
            for shell in self.kmc.energy_model.pair_shells
        )
        for agent, site in enumerate(self.kmc.lattice.hydrogen_sites):
            if agent == moved_agent:
                continue
            if (
                min(
                    self.kmc.lattice.minimum_image_distance_angstrom(source_site, site),
                    self.kmc.lattice.minimum_image_distance_angstrom(target_site, site),
                )
                <= cutoff
            ):
                affected.add(int(agent))
        return np.asarray(sorted(affected), dtype=np.int64)

    def preview_event(self, event_id):
        """Return the exact successor contract without mutating simulator state."""

        state = self.kmc.state_dict()
        history_state = self.history.state_dict()
        clock_elapsed = (
            self.policy_clock.elapsed_s if self.policy_clock is not None else None
        )
        try:
            transition = self.execute_event(event_id)
            return {
                "transition": transition,
                "catalog": self.enumerate_events(),
                "model_inputs": self.get_model_inputs(),
            }
        finally:
            self.kmc.load_state_dict(state)
            self.history.load_state_dict(history_state)
            if self.policy_clock is not None:
                self.policy_clock.elapsed_s = float(clock_elapsed)

    def execute_event(self, event_id, *, predicted_time_increment_s=None):
        catalog = self.enumerate_events()
        agent, local_event = catalog.assert_legal(int(event_id))
        source = int(self.kmc.lattice.hydrogen_sites[agent])
        target = int(self.kmc.lattice.event_targets[source, local_event])
        affected = self._affected_agents(source, target, agent)

        policy_increment = None
        if predicted_time_increment_s is not None:
            if self.policy_clock is None:
                raise RuntimeError(
                    "a predicted policy-time increment requires a certified policy clock"
                )
            policy_increment = self.policy_clock.advance(predicted_time_increment_s)

        transition = self.kmc.apply_event(
            int(event_id),
            catalog=catalog,
            delta_time_s=0.0,
            clock_owner=(
                "policy_poisson" if policy_increment is not None else "unassigned"
            ),
        )
        decision = self.keyframe_criterion.classify(transition)
        record = build_hcp_event_record(transition, catalog, decision)
        self.history.update(affected, record)
        update_mask = np.zeros(self.num_agents, dtype=bool)
        if decision.is_keyframe:
            update_mask[affected] = True
        return HCPPolicyTransition(
            physical_transition=transition,
            keyframe=decision,
            affected_agents=affected,
            long_memory_update_mask=update_mask,
            policy_time_increment_s=policy_increment,
            policy_elapsed_time_s=(
                self.policy_clock.elapsed_s if self.policy_clock is not None else None
            ),
        )
