import torch
import torch.nn as nn

from .config import AtomWorldModelConfig


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(values.dtype).unsqueeze(-1)
    return (values * weights).sum(dim=-2) / weights.sum(dim=-2).clamp_min(1.0)


def _safe_padding_mask(mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    valid = mask.bool()
    safe = valid.clone()
    empty = ~safe.any(dim=-1)
    if bool(empty.any()):
        safe[empty, 0] = True
    return valid, ~safe


class AtomicKeyframeEncoder(nn.Module):
    """Dense radius-graph observations plus sparse long-range defect context."""

    def __init__(self, config: AtomWorldModelConfig):
        super().__init__()
        self.config = config
        h = config.hidden_dim
        self.species_embedding = nn.Embedding(config.species_count, h)
        self.defect_embedding = nn.Embedding(config.defect_type_count, h)
        self.position_encoder = nn.Sequential(
            nn.Linear(5, h), nn.GELU(), nn.Linear(h, h)
        )

        dense_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.attention_heads,
            dim_feedforward=2 * h,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.dense_encoder = nn.TransformerEncoder(dense_layer, config.dense_layers)

        self.sparse_type_embedding = nn.Embedding(config.defect_type_count, h)
        self.sparse_geometry = nn.Sequential(
            nn.Linear(5, h), nn.GELU(), nn.Linear(h, h)
        )
        sparse_layer = nn.TransformerEncoderLayer(
            d_model=h,
            nhead=config.attention_heads,
            dim_feedforward=2 * h,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.sparse_encoder = nn.TransformerEncoder(sparse_layer, config.sparse_layers)
        self.fusion = nn.Sequential(
            nn.Linear(2 * h, 2 * h),
            nn.GELU(),
            nn.LayerNorm(2 * h),
            nn.Linear(2 * h, config.keyframe_dim),
            nn.LayerNorm(config.keyframe_dim),
        )

    def forward(
        self,
        dense_species: torch.Tensor,
        dense_positions: torch.Tensor,
        dense_defect_types: torch.Tensor,
        dense_connectivity: torch.Tensor,
        dense_mask: torch.Tensor,
        sparse_tokens: torch.Tensor,
        sparse_mask: torch.Tensor,
    ) -> torch.Tensor:
        if dense_positions.shape[-1] != 3:
            raise ValueError("dense_positions must end in xyz coordinates")
        if sparse_tokens.shape[-1] < 6:
            raise ValueError(
                "sparse_tokens must contain [type, dx, dy, dz, dist2, shell]"
            )

        leading_shape = dense_species.shape[:-1]
        dense_nodes = dense_species.shape[-1]
        sparse_nodes = sparse_tokens.shape[-2]
        dense_species = dense_species.reshape(-1, dense_nodes)
        dense_positions = dense_positions.reshape(-1, dense_nodes, 3)
        dense_defect_types = dense_defect_types.reshape(-1, dense_nodes)
        dense_connectivity = dense_connectivity.reshape(-1, dense_nodes, 1)
        dense_mask = dense_mask.reshape(-1, dense_nodes)
        sparse_tokens = sparse_tokens.reshape(-1, sparse_nodes, sparse_tokens.shape[-1])
        sparse_mask = sparse_mask.reshape(-1, sparse_nodes)

        dense_valid, dense_padding = _safe_padding_mask(dense_mask)
        radius = torch.linalg.vector_norm(dense_positions, dim=-1, keepdim=True)
        dense = (
            self.species_embedding(
                dense_species.long().clamp(0, self.config.species_count - 1)
            )
            + self.defect_embedding(
                dense_defect_types.long().clamp(0, self.config.defect_type_count - 1)
            )
            + self.position_encoder(
                torch.cat(
                    [dense_positions.float(), radius, dense_connectivity.float()],
                    dim=-1,
                )
            )
        )
        dense = self.dense_encoder(dense, src_key_padding_mask=dense_padding)
        dense_pooled = _masked_mean(dense, dense_valid)

        sparse_valid, sparse_padding = _safe_padding_mask(sparse_mask)
        sparse_type = (
            sparse_tokens[..., 0].long().clamp(0, self.config.defect_type_count - 1)
        )
        sparse_geometry = sparse_tokens[..., 1:6].float()
        sparse = self.sparse_type_embedding(sparse_type) + self.sparse_geometry(
            sparse_geometry
        )
        sparse = self.sparse_encoder(sparse, src_key_padding_mask=sparse_padding)
        sparse_pooled = _masked_mean(sparse, sparse_valid)
        keyframe = self.fusion(torch.cat([dense_pooled, sparse_pooled], dim=-1))
        return keyframe.reshape(*leading_shape, self.config.keyframe_dim)
