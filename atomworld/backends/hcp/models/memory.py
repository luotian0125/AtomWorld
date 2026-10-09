import torch
import torch.nn as nn

from .config import AtomWorldModelConfig


class ShortEventEncoder(nn.Module):
    def __init__(self, config: AtomWorldModelConfig):
        super().__init__()
        self.input = nn.Sequential(
            nn.Linear(config.history_feature_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.attention_heads,
            dim_feedforward=2 * config.hidden_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, config.history_layers)

    def forward(
        self, history: torch.Tensor, history_mask: torch.Tensor
    ) -> torch.Tensor:
        original_shape = history.shape[:-2]
        horizon, feature_dim = history.shape[-2:]
        flat_history = history.reshape(-1, horizon, feature_dim)
        valid = history_mask.reshape(-1, horizon).bool()
        safe_valid = valid.clone()
        empty = ~safe_valid.any(dim=-1)
        if bool(empty.any()):
            safe_valid[empty, 0] = True
        encoded = self.encoder(
            self.input(flat_history.float()), src_key_padding_mask=~safe_valid
        )
        weights = valid.to(encoded.dtype).unsqueeze(-1)
        pooled = (encoded * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return pooled.reshape(*original_shape, -1)


class KineticMemory(nn.Module):
    """Short event memory plus locally updated recurrent structural memory."""

    def __init__(self, config: AtomWorldModelConfig):
        super().__init__()
        self.config = config
        self.short = ShortEventEncoder(config)
        self.long_cell = nn.GRUCell(
            config.keyframe_dim + config.hidden_dim, config.memory_dim
        )
        self.restore = nn.Sequential(
            nn.Linear(
                config.keyframe_dim + config.hidden_dim + config.memory_dim,
                2 * config.memory_dim,
            ),
            nn.GELU(),
            nn.LayerNorm(2 * config.memory_dim),
            nn.Linear(2 * config.memory_dim, config.memory_dim),
            nn.LayerNorm(config.memory_dim),
        )

    def initial_state(
        self, batch_size: int, num_agents: int, *, device=None, dtype=None
    ) -> torch.Tensor:
        return torch.zeros(
            batch_size, num_agents, self.config.memory_dim, device=device, dtype=dtype
        )

    def forward(
        self,
        keyframes: torch.Tensor,
        event_history: torch.Tensor,
        history_mask: torch.Tensor,
        previous_long_state: torch.Tensor,
        update_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if keyframes.shape[:-1] != previous_long_state.shape[:-1]:
            raise ValueError(
                "keyframes and long-memory state must share batch/agent dimensions"
            )
        if self.config.use_short_memory:
            short_context = self.short(event_history, history_mask)
        else:
            short_context = torch.zeros(
                *keyframes.shape[:-1],
                self.config.hidden_dim,
                dtype=keyframes.dtype,
                device=keyframes.device,
            )
        update_input = torch.cat([keyframes, short_context], dim=-1)
        if self.config.use_long_memory:
            flat_candidate = self.long_cell(
                update_input.reshape(-1, update_input.shape[-1]),
                previous_long_state.reshape(-1, previous_long_state.shape[-1]),
            ).reshape_as(previous_long_state)
            mask = update_mask.bool().unsqueeze(-1)
            long_state = torch.where(mask, flat_candidate, previous_long_state)
        else:
            long_state = torch.zeros_like(previous_long_state)
        restored = self.restore(
            torch.cat([keyframes, short_context, long_state], dim=-1)
        )
        return restored, long_state
