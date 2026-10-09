from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from .actor import LocalEventActor, arbitrate_global_events
from .config import AtomWorldModelConfig
from .critic import GlobalKineticCritic
from .keyframe import AtomicKeyframeEncoder
from .memory import KineticMemory


@dataclass
class AtomWorldOutput:
    distribution: torch.distributions.Categorical
    global_logits: torch.Tensor
    local_scores: torch.Tensor
    value: Optional[torch.Tensor]
    restored_states: torch.Tensor
    long_memory: torch.Tensor


class MemoryRestoredAtomWorld(nn.Module):
    """New-policy stack. Physical rates enter masks/regularizers, never its clock."""

    def __init__(self, config: AtomWorldModelConfig):
        super().__init__()
        self.config = config
        self.keyframes = AtomicKeyframeEncoder(config)
        self.memory = KineticMemory(config)
        self.actor = LocalEventActor(config)
        self.critic = GlobalKineticCritic(config)

    def forward(
        self,
        *,
        dense_species: torch.Tensor,
        dense_positions: torch.Tensor,
        dense_defect_types: torch.Tensor,
        dense_connectivity: torch.Tensor,
        dense_mask: torch.Tensor,
        sparse_tokens: torch.Tensor,
        sparse_mask: torch.Tensor,
        event_history: torch.Tensor,
        history_mask: torch.Tensor,
        previous_long_state: torch.Tensor,
        update_mask: torch.Tensor,
        event_features: torch.Tensor,
        legal_mask: torch.Tensor,
        agent_mask: torch.Tensor,
        energy: torch.Tensor,
        mesoscale_descriptors: torch.Tensor,
        policy_temperature: float = 1.0,
        compute_value: bool = True,
    ) -> AtomWorldOutput:
        keyframes = self.keyframes(
            dense_species,
            dense_positions,
            dense_defect_types,
            dense_connectivity,
            dense_mask,
            sparse_tokens,
            sparse_mask,
        )
        restored, long_memory = self.memory(
            keyframes,
            event_history,
            history_mask,
            previous_long_state,
            update_mask,
        )
        local_scores = self.actor(restored, event_features)
        distribution, logits = arbitrate_global_events(
            local_scores, legal_mask, policy_temperature
        )
        # CTDE value learning consumes the restored state but does not backpropagate
        # critic regression through the deployed actor/memory representation.
        value = (
            self.critic(restored.detach(), agent_mask, energy, mesoscale_descriptors)
            if compute_value
            else None
        )
        return AtomWorldOutput(
            distribution, logits, local_scores, value, restored, long_memory
        )
