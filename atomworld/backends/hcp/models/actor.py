import torch
import torch.nn as nn
from torch.distributions import Categorical

from .config import AtomWorldModelConfig


def arbitrate_global_events(
    local_scores: torch.Tensor, legal_mask: torch.Tensor, temperature: float = 1.0
):
    if local_scores.shape != legal_mask.shape:
        raise ValueError("local_scores and legal_mask must have identical shapes")
    if temperature <= 0.0:
        raise ValueError("policy temperature must be positive")
    flat_scores = local_scores.flatten(start_dim=-2) / float(temperature)
    flat_mask = legal_mask.bool().flatten(start_dim=-2)
    if bool((~flat_mask.any(dim=-1)).any()):
        raise RuntimeError("every state must expose at least one legal simulator event")
    logits = flat_scores.masked_fill(~flat_mask, -torch.inf)
    return Categorical(logits=logits), logits


class LocalEventActor(nn.Module):
    """Shared local scorer; it never defines rates or a physical clock."""

    def __init__(self, config: AtomWorldModelConfig):
        super().__init__()
        self.config = config
        self.event_encoder = nn.Sequential(
            nn.Linear(config.event_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        self.state_projection = nn.Linear(config.memory_dim, config.hidden_dim)
        self.score = nn.Sequential(
            nn.Linear(3 * config.hidden_dim, 2 * config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(2 * config.hidden_dim),
            nn.Linear(2 * config.hidden_dim, 1),
        )

    def forward(
        self, restored_states: torch.Tensor, event_features: torch.Tensor
    ) -> torch.Tensor:
        if event_features.shape[-1] != self.config.event_feature_dim:
            raise ValueError("unexpected event feature dimension")
        event = self.event_encoder(event_features.float())
        state = self.state_projection(restored_states).unsqueeze(-2).expand_as(event)
        interaction = state * event
        return self.score(torch.cat([state, event, interaction], dim=-1)).squeeze(-1)
