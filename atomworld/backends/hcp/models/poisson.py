import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AtomWorldModelConfig
from .keyframe import AtomicKeyframeEncoder


class PoissonNetwork(nn.Module):
    """Independent positive dimensionless Poisson potential u(s)=Gamma_total(s) tau(s)."""

    def __init__(self, patch_dim: int, hidden_dim: int = 128, depth: int = 3):
        super().__init__()
        if patch_dim <= 0 or hidden_dim <= 0 or depth < 1:
            raise ValueError("invalid Poisson network dimensions")
        layers = [nn.Linear(patch_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)]
        for _ in range(depth - 1):
            layers.extend(
                [nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.LayerNorm(hidden_dim)]
            )
        self.backbone = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self, local_patch: torch.Tensor, agent_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        encoded = self.backbone(local_patch.float())
        if agent_mask is not None:
            if local_patch.shape[:-1] != agent_mask.shape:
                raise ValueError(
                    "agent_mask must match the local-patch leading dimensions"
                )
            weights = agent_mask.to(encoded.dtype).unsqueeze(-1)
            encoded = (encoded * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(
                1.0
            )
        return F.softplus(self.output(encoded)).squeeze(-1)


class PoissonWorldModel(nn.Module):
    """Poisson branch with its own keyframe weights, separate from actor memory."""

    def __init__(self, config: AtomWorldModelConfig, depth: int = 3):
        super().__init__()
        self.keyframes = AtomicKeyframeEncoder(config)
        self.potential = PoissonNetwork(config.keyframe_dim, config.hidden_dim, depth)

    def forward(
        self,
        dense_species: torch.Tensor,
        dense_positions: torch.Tensor,
        dense_defect_types: torch.Tensor,
        dense_connectivity: torch.Tensor,
        dense_mask: torch.Tensor,
        sparse_tokens: torch.Tensor,
        sparse_mask: torch.Tensor,
        agent_mask: torch.Tensor,
    ) -> torch.Tensor:
        keyframes = self.keyframes(
            dense_species,
            dense_positions,
            dense_defect_types,
            dense_connectivity,
            dense_mask,
            sparse_tokens,
            sparse_mask,
        )
        return self.potential(keyframes, agent_mask=agent_mask)


def poisson_increment(
    current_u: torch.Tensor,
    next_u: torch.Tensor,
    current_total_rate: torch.Tensor,
    next_total_rate: torch.Tensor,
) -> torch.Tensor:
    current_total_rate = current_total_rate.clamp_min(1.0e-20)
    next_total_rate = next_total_rate.clamp_min(1.0e-20)
    return (
        current_u - (current_total_rate / next_total_rate) * next_u
    ) / current_total_rate


def poisson_residual_loss(
    current_u: torch.Tensor,
    successor_u: torch.Tensor,
    rates: torch.Tensor,
    current_total_rate: torch.Tensor,
    successor_total_rates: torch.Tensor,
    legal_mask: torch.Tensor,
    absorbing_mask: torch.Tensor | None = None,
    boundary_weight: float = 1.0,
):
    """Full legal-event Dynkin residual with an explicit absorbing boundary."""
    delta_tau = poisson_increment(
        current_u.unsqueeze(-1),
        successor_u,
        current_total_rate.unsqueeze(-1),
        successor_total_rates,
    )
    legal_rates = rates * legal_mask.to(rates.dtype)
    event_probabilities = legal_rates / current_total_rate.unsqueeze(-1).clamp_min(
        1.0e-20
    )
    dimensionless_progress = (
        current_u.unsqueeze(-1)
        - (
            current_total_rate.unsqueeze(-1).clamp_min(1.0e-20)
            / successor_total_rates.clamp_min(1.0e-20)
        )
        * successor_u
    )
    weighted_progress = (event_probabilities * dimensionless_progress).sum(dim=-1)
    residual = weighted_progress - 1.0
    if absorbing_mask is None:
        interior = torch.ones_like(residual, dtype=torch.bool)
    else:
        interior = ~absorbing_mask.bool()
    interior_loss = (
        residual[interior].square().mean()
        if bool(interior.any())
        else residual.sum() * 0.0
    )
    boundary_loss = torch.zeros(
        (), dtype=interior_loss.dtype, device=interior_loss.device
    )
    if absorbing_mask is not None and bool(absorbing_mask.bool().any()):
        boundary_loss = current_u[absorbing_mask.bool()].square().mean()
    total = interior_loss + float(boundary_weight) * boundary_loss
    return total, {
        "poisson_residual_loss": interior_loss.detach(),
        "poisson_boundary_loss": boundary_loss.detach(),
        "poisson_residual_abs": residual.abs().mean().detach(),
        "predicted_delta_tau_mean": delta_tau.mean().detach(),
    }
