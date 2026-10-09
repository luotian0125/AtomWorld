import torch
import torch.nn as nn

from .config import AtomWorldModelConfig


class GlobalKineticCritic(nn.Module):
    """Training-only value model over restored local states and global descriptors."""

    def __init__(self, config: AtomWorldModelConfig):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(config.memory_dim))
        nn.init.normal_(self.query, std=0.02)
        global_dim = config.memory_dim + 1 + config.mesoscale_dim
        self.value = nn.Sequential(
            nn.Linear(global_dim, 2 * config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * config.hidden_dim),
            nn.Linear(2 * config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self,
        restored_states: torch.Tensor,
        agent_mask: torch.Tensor,
        energy: torch.Tensor,
        mesoscale_descriptors: torch.Tensor,
    ) -> torch.Tensor:
        scores = torch.einsum("...nh,h->...n", restored_states, self.query)
        scores = scores.masked_fill(~agent_mask.bool(), -torch.inf)
        if bool((~agent_mask.bool().any(dim=-1)).any()):
            raise RuntimeError("critic requires at least one active local environment")
        weights = torch.softmax(scores, dim=-1)
        pooled = torch.einsum("...n,...nh->...h", weights, restored_states)
        energy = energy.reshape(*pooled.shape[:-1], 1).to(pooled.dtype)
        return self.value(
            torch.cat([pooled, energy, mesoscale_descriptors.to(pooled.dtype)], dim=-1)
        )
