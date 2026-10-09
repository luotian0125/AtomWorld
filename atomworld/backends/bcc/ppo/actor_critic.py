import numpy as np
import torch
import torch.nn as nn

from atomworld.backends.bcc.ppo.utils.rnn import RNNLayer
from atomworld.backends.bcc.ppo.utils.util import check
from atomworld.backends.bcc.ppo.utils.mymlp import ElegantMLP
from atomworld.backends.bcc.embedding.embedding import VacancyEmbedding, SparseDefectEncoder


class R_Actor(nn.Module):
    def __init__(self, args, action_space=None, device=torch.device("cpu")):
        super().__init__()
        self.args = args
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.hidden_size = int(args.hidden_size)
        self.history_len = int(getattr(args, "history_len", 5))
        self.dense_local_scale = float(getattr(args, "dense_local_scale", 1.0))
        self.use_sparse_context = bool(getattr(args, "use_sparse_context", True))
        self.use_action_history = bool(getattr(args, "use_action_history", True))
        self.use_recurrent_memory = bool(getattr(args, "use_recurrent_memory", True))
        self.sparse_feature_dim = 6
        self.sparse_max_tokens = int(getattr(args, "sparse_defect_max_tokens", 24))
        self.event_feature_dim = max(9, int(getattr(args, "event_feature_dim", 9)))
        self.use_direction_aware_head = bool(getattr(args, "use_direction_aware_head", True))
        self.direction_heads = int(getattr(args, "direction_attention_heads", 4))
        self.direction_dropout = float(getattr(args, "direction_attention_dropout", 0.05))
        assert self.hidden_size % self.direction_heads == 0, "hidden_size must be divisible by direction_attention_heads"

        self.embed = VacancyEmbedding(args, device=device)
        self.sparse_encoder = SparseDefectEncoder(args, device=device)

        self.space_gate_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        nn.init.constant_(self.space_gate_linear.bias, 2.0)
        self.space_gate = nn.Sequential(self.space_gate_linear, nn.Sigmoid())
        self.space_norm = nn.LayerNorm(self.hidden_size)

        self.action_embedding = nn.Embedding(num_embeddings=9, embedding_dim=32, padding_idx=0)
        self.history_pos_embed = nn.Parameter(torch.zeros(1, self.history_len, 32))
        nn.init.normal_(self.history_pos_embed, mean=0.0, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=4,
            dim_feedforward=64,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        self.history_proj = nn.Linear(32, self.hidden_size)

        self.fusion_gate_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        nn.init.constant_(self.fusion_gate_linear.bias, 3.0)
        self.fusion_gate = nn.Sequential(self.fusion_gate_linear, nn.Sigmoid())
        self.fusion_norm = nn.LayerNorm(self.hidden_size)

        self.rnn = RNNLayer(
            inputs_dim=self.hidden_size,
            outputs_dim=self.hidden_size,
            recurrent_N=args.recurrent_N,
            use_orthogonal=args.use_orthogonal,
        )

        self.base_action_head = nn.Linear(self.hidden_size, 8)
        self.direction_query_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_dense_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.context_sparse_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.direction_attn = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=self.direction_heads,
            dropout=self.direction_dropout,
            batch_first=True,
        )
        self.direction_ffn = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.direction_norm1 = nn.LayerNorm(self.hidden_size)
        self.direction_norm2 = nn.LayerNorm(self.hidden_size)
        self.direction_event_norm = nn.LayerNorm(self.hidden_size)
        self.direction_delta_head = nn.Linear(self.hidden_size, 1)
        nn.init.zeros_(self.direction_delta_head.weight)
        nn.init.zeros_(self.direction_delta_head.bias)

        self.event_feature_encoder = nn.Sequential(
            nn.Linear(self.event_feature_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.event_gate_linear = nn.Linear(self.hidden_size * 2, self.hidden_size)
        nn.init.constant_(self.event_gate_linear.bias, 1.0)
        self.event_delta_head = nn.Linear(self.hidden_size, 1)

        direction_vectors = self.embed.node_positions[:8].clone()
        direction_vectors = direction_vectors / torch.norm(direction_vectors, dim=-1, keepdim=True).clamp(min=1.0)
        self.register_buffer("direction_vectors", direction_vectors)
        self.direction_geom = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.direction_token = nn.Parameter(torch.zeros(8, self.hidden_size))
        nn.init.normal_(self.direction_token, mean=0.0, std=0.02)
        self.to(device)

    def _expand_masks(self, masks, num_agents):
        masks = check(masks).to(**self.tpdv)
        if masks.dim() == 1:
            masks = masks.unsqueeze(-1)
        if masks.shape[0] == 1:
            masks = masks.expand(num_agents, masks.shape[-1])
        return masks

    def _prepare_sparse_inputs(self, sparse_defect_tokens, sparse_defect_masks, num_entries):
        device = self.tpdv["device"]
        if sparse_defect_tokens is None:
            sparse_defect_tokens = torch.zeros(
                (num_entries, self.sparse_max_tokens, self.sparse_feature_dim),
                dtype=torch.float32,
                device=device,
            )
            sparse_defect_tokens[..., 0] = 2.0
        else:
            sparse_defect_tokens = torch.as_tensor(sparse_defect_tokens, dtype=torch.float32, device=device)
            if sparse_defect_tokens.dim() == 2:
                sparse_defect_tokens = sparse_defect_tokens.unsqueeze(0)
            sparse_defect_tokens = sparse_defect_tokens.reshape(num_entries, self.sparse_max_tokens, self.sparse_feature_dim)

        if sparse_defect_masks is None:
            sparse_defect_masks = torch.zeros((num_entries, self.sparse_max_tokens), dtype=torch.float32, device=device)
        else:
            sparse_defect_masks = torch.as_tensor(sparse_defect_masks, dtype=torch.float32, device=device)
            if sparse_defect_masks.dim() == 1:
                sparse_defect_masks = sparse_defect_masks.unsqueeze(0)
            sparse_defect_masks = sparse_defect_masks.reshape(num_entries, self.sparse_max_tokens)

        return sparse_defect_tokens, sparse_defect_masks

    def _flatten_leading(self, tensor, keep_last_dims=1):
        if keep_last_dims <= 0:
            raise ValueError("keep_last_dims must be >= 1")
        lead_shape = tuple(tensor.shape[:-keep_last_dims])
        flat = tensor.reshape(-1, *tensor.shape[-keep_last_dims:])
        return flat, lead_shape

    def _restore_leading(self, tensor, lead_shape):
        if len(lead_shape) == 0:
            return tensor
        return tensor.reshape(*lead_shape, *tensor.shape[1:])

    def _prepare_event_features(self, event_features, lead_shape):
        num_entries = int(np.prod(lead_shape)) if len(lead_shape) > 0 else 1
        if event_features is None:
            event_features = torch.zeros(
                (num_entries, 8, self.event_feature_dim),
                dtype=torch.float32,
                device=self.tpdv["device"],
            )
        else:
            event_features = torch.as_tensor(event_features, dtype=torch.float32, device=self.tpdv["device"])
            event_features = event_features.reshape(num_entries, 8, self.event_feature_dim)
        if len(lead_shape) == 0:
            return event_features
        return event_features.reshape(*lead_shape, 8, self.event_feature_dim)

    def encode_nonrecurrent(self, obs, sparse_defect_tokens=None, sparse_defect_masks=None):
        obs = check(obs).to(**self.tpdv)
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        flat_obs, lead_shape = self._flatten_leading(obs, keep_last_dims=1)
        num_entries = flat_obs.shape[0]

        spatial_obs = flat_obs[:, :50]
        history_obs = flat_obs[:, 50:].long() + 1
        if not self.use_action_history:
            history_obs = torch.zeros_like(history_obs)
        illegal_mask = flat_obs[:, 0:8] == 2

        dense_features, dense_node_features = self.embed(spatial_obs, return_node_features=True)
        if self.dense_local_scale != 1.0:
            dense_features = dense_features * self.dense_local_scale
            dense_node_features = dense_node_features * self.dense_local_scale

        sparse_defect_tokens, sparse_defect_masks = self._prepare_sparse_inputs(
            sparse_defect_tokens, sparse_defect_masks, num_entries
        )
        sparse_features, sparse_node_features, sparse_valid_mask = self.sparse_encoder(
            sparse_defect_tokens,
            sparse_defect_masks,
            return_token_features=True,
        )
        if not self.use_sparse_context:
            sparse_features = torch.zeros_like(sparse_features)
            sparse_node_features = torch.zeros_like(sparse_node_features)
            sparse_valid_mask = torch.zeros_like(sparse_valid_mask, dtype=torch.bool)

        spatial_gate = self.space_gate(torch.cat([dense_features, sparse_features], dim=-1))
        spatial_features = spatial_gate * dense_features + (1.0 - spatial_gate) * sparse_features
        spatial_features = self.space_norm(spatial_features)

        history_embed = self.action_embedding(history_obs)
        pos_embed = self.history_pos_embed[:, : history_embed.shape[1], :]
        history_input = history_embed + pos_embed
        history_features = self.history_proj(self.transformer(history_input).mean(dim=1))

        fused_input = torch.cat([spatial_features, history_features], dim=-1)
        gate = self.fusion_gate(fused_input)
        fused_features = gate * spatial_features + (1.0 - gate) * history_features
        fused_features = self.fusion_norm(fused_features)

        return {
            "fused_features": self._restore_leading(fused_features, lead_shape),
            "dense_node_features": self._restore_leading(dense_node_features, lead_shape),
            "sparse_node_features": self._restore_leading(sparse_node_features, lead_shape),
            "sparse_valid_mask": self._restore_leading(sparse_valid_mask, lead_shape),
            "illegal_mask": self._restore_leading(illegal_mask, lead_shape),
        }

    def forward_recurrent_from_encoded(self, encoded, rnn_states, masks, event_features=None, return_latent=False):
        fused_features = encoded["fused_features"]
        dense_node_features = encoded["dense_node_features"]
        sparse_node_features = encoded["sparse_node_features"]
        sparse_valid_mask = encoded["sparse_valid_mask"]
        illegal_mask = encoded["illegal_mask"]

        rnn_states = check(rnn_states).to(**self.tpdv)
        if not self.use_recurrent_memory:
            result = self.forward_snapshot_from_encoded(
                fused_features,
                dense_node_features,
                sparse_node_features,
                sparse_valid_mask,
                illegal_mask,
                rnn_states,
                event_features=event_features,
            )
            return (*result, fused_features) if return_latent else result

        if fused_features.dim() == 2:
            num_agents = fused_features.shape[0]
            masks = self._expand_masks(masks, num_agents)
            rnn_features, new_rnn_states = self.rnn(fused_features, rnn_states, masks)
            if self.use_direction_aware_head:
                local_logits = self._direction_aware_logits(
                    rnn_features,
                    dense_node_features,
                    sparse_node_features,
                    sparse_valid_mask,
                    event_features,
                )
            else:
                local_logits = self.base_action_head(rnn_features) + self._event_feature_logits(rnn_features, event_features)
            result = (local_logits, illegal_mask, new_rnn_states)
            return (*result, rnn_features) if return_latent else result

        if fused_features.dim() == 3:
            T, N, _ = fused_features.shape
            actor_state = rnn_states
            logits_seq = []
            latent_seq = []
            for t in range(T):
                step_masks = self._expand_masks(masks[t], N)
                rnn_features, actor_state = self.rnn(fused_features[t], actor_state, step_masks)
                if self.use_direction_aware_head:
                    step_logits = self._direction_aware_logits(
                        rnn_features,
                        dense_node_features[t],
                        sparse_node_features[t],
                        sparse_valid_mask[t],
                        event_features[t],
                    )
                else:
                    step_logits = self.base_action_head(rnn_features) + self._event_feature_logits(rnn_features, event_features[t])
                logits_seq.append(step_logits.unsqueeze(0))
                if return_latent:
                    latent_seq.append(rnn_features.unsqueeze(0))

            local_logits = torch.cat(logits_seq, dim=0)
            result = (local_logits, illegal_mask, actor_state)
            return (*result, torch.cat(latent_seq, dim=0)) if return_latent else result

        if fused_features.dim() != 4:
            raise ValueError(f"Unsupported fused_features shape: {tuple(fused_features.shape)}")

        T, B, N, hidden_dim = fused_features.shape
        if rnn_states.dim() == 3:
            rnn_states = rnn_states.unsqueeze(0)
        actor_state = rnn_states.reshape(B * N, *rnn_states.shape[-2:])
        sequence_masks = check(masks).to(**self.tpdv).reshape(T, B, -1)[..., :1]
        sequence_masks = sequence_masks.unsqueeze(2).expand(T, B, N, 1).reshape(T * B * N, 1)
        rnn_features, actor_state = self.rnn(
            fused_features.reshape(T * B * N, hidden_dim), actor_state, sequence_masks,
        )
        flat_event_features = (event_features.reshape(T * B * N, 8, event_features.shape[-1])
                               if event_features is not None else None)
        if self.use_direction_aware_head:
            flat_logits = self._direction_aware_logits(
                rnn_features,
                dense_node_features.reshape(T * B * N, *dense_node_features.shape[-2:]),
                sparse_node_features.reshape(T * B * N, *sparse_node_features.shape[-2:]),
                sparse_valid_mask.reshape(T * B * N, sparse_valid_mask.shape[-1]),
                flat_event_features,
            )
        else:
            flat_logits = self.base_action_head(rnn_features) + self._event_feature_logits(rnn_features, flat_event_features)
        local_logits = flat_logits.reshape(T, B, N, 8)
        result = (local_logits, illegal_mask, actor_state.reshape(B, N, *actor_state.shape[-2:]))
        return (*result, rnn_features.reshape(T, B, N, hidden_dim)) if return_latent else result

    def forward_snapshot_from_encoded(
        self,
        fused_features,
        dense_node_features,
        sparse_node_features,
        sparse_valid_mask,
        illegal_mask,
        rnn_states,
        event_features=None,
    ):
        new_rnn_states = torch.zeros_like(rnn_states)

        if fused_features.dim() == 2:
            if self.use_direction_aware_head:
                local_logits = self._direction_aware_logits(
                    fused_features,
                    dense_node_features,
                    sparse_node_features,
                    sparse_valid_mask,
                    event_features,
                )
            else:
                local_logits = self.base_action_head(fused_features) + self._event_feature_logits(fused_features, event_features)
            return local_logits, illegal_mask, new_rnn_states

        if fused_features.dim() == 3:
            logits_seq = []
            for t in range(fused_features.shape[0]):
                step_event_features = event_features[t] if event_features is not None else None
                if self.use_direction_aware_head:
                    step_logits = self._direction_aware_logits(
                        fused_features[t],
                        dense_node_features[t],
                        sparse_node_features[t],
                        sparse_valid_mask[t],
                        step_event_features,
                    )
                else:
                    step_logits = self.base_action_head(fused_features[t]) + self._event_feature_logits(fused_features[t], step_event_features)
                logits_seq.append(step_logits.unsqueeze(0))
            return torch.cat(logits_seq, dim=0), illegal_mask, new_rnn_states

        if fused_features.dim() != 4:
            raise ValueError(f"Unsupported fused_features shape: {tuple(fused_features.shape)}")

        T, B, N, hidden_dim = fused_features.shape
        flat_fused = fused_features.reshape(T * B * N, hidden_dim)
        flat_event_features = (event_features.reshape(T * B * N, 8, event_features.shape[-1])
                               if event_features is not None else None)
        if self.use_direction_aware_head:
            flat_logits = self._direction_aware_logits(
                flat_fused,
                dense_node_features.reshape(T * B * N, *dense_node_features.shape[-2:]),
                sparse_node_features.reshape(T * B * N, *sparse_node_features.shape[-2:]),
                sparse_valid_mask.reshape(T * B * N, sparse_valid_mask.shape[-1]),
                flat_event_features,
            )
        else:
            flat_logits = self.base_action_head(flat_fused) + self._event_feature_logits(flat_fused, flat_event_features)
        return flat_logits.reshape(T, B, N, 8), illegal_mask, new_rnn_states

    def _event_feature_logits(self, rnn_features, event_features):
        batch_size = rnn_features.shape[0]
        if event_features is None:
            event_features = torch.zeros((batch_size, 8, self.event_feature_dim), dtype=torch.float32, device=rnn_features.device)
        else:
            event_features = torch.as_tensor(event_features, dtype=torch.float32, device=rnn_features.device).reshape(batch_size, 8, self.event_feature_dim)
        phys_features = self.event_feature_encoder(event_features)
        return self.event_delta_head(phys_features).squeeze(-1)

    def _direction_aware_logits(self, rnn_features, dense_nodes, sparse_nodes, sparse_valid_mask, event_features):
        batch_size = rnn_features.shape[0]
        dense_context = self.context_dense_proj(dense_nodes)
        sparse_context = self.context_sparse_proj(sparse_nodes)
        context_tokens = torch.cat([dense_context, sparse_context], dim=1)
        dense_mask = torch.ones((batch_size, dense_nodes.shape[1]), dtype=torch.bool, device=rnn_features.device)
        context_mask = torch.cat([dense_mask, sparse_valid_mask], dim=1)

        base_query = self.direction_query_proj(rnn_features).unsqueeze(1)
        dir_token = self.direction_token.unsqueeze(0).expand(batch_size, -1, -1)
        dir_geom = self.direction_geom(self.direction_vectors).unsqueeze(0).expand(batch_size, -1, -1)
        queries = base_query + dir_token + dir_geom

        attn_out, _ = self.direction_attn(queries, context_tokens, context_tokens, key_padding_mask=~context_mask)
        dir_features = self.direction_norm1(attn_out + queries)
        dir_features = self.direction_norm2(self.direction_ffn(dir_features) + dir_features)

        if event_features is None:
            event_features = torch.zeros((batch_size, 8, self.event_feature_dim), dtype=torch.float32, device=rnn_features.device)
        else:
            event_features = torch.as_tensor(event_features, dtype=torch.float32, device=rnn_features.device).reshape(batch_size, 8, self.event_feature_dim)
        phys_features = self.event_feature_encoder(event_features)
        mix_gate = torch.sigmoid(self.event_gate_linear(torch.cat([dir_features, phys_features], dim=-1)))
        mixed_features = self.direction_event_norm(mix_gate * dir_features + (1.0 - mix_gate) * phys_features)
        delta_logits = self.direction_delta_head(mixed_features).squeeze(-1)
        phys_logits = self.event_delta_head(phys_features).squeeze(-1)
        return self.base_action_head(rnn_features) + delta_logits + phys_logits

    def get_local_logits(self, obs, rnn_states, masks, event_features=None, sparse_defect_tokens=None, sparse_defect_masks=None, return_latent=False):
        encoded = self.encode_nonrecurrent(
            obs,
            sparse_defect_tokens=sparse_defect_tokens,
            sparse_defect_masks=sparse_defect_masks,
        )
        lead_shape = tuple(encoded["fused_features"].shape[:-1])
        event_features = self._prepare_event_features(event_features, lead_shape)
        return self.forward_recurrent_from_encoded(
            encoded, rnn_states, masks, event_features=event_features, return_latent=return_latent,
        )


class R_Critic(nn.Module):
    def __init__(self, args, device=torch.device("cpu")):
        super().__init__()
        self.args = args
        self.hidden_size = int(args.hidden_size)
        self.num_agents = int(args.lattice_v_nums)
        self.history_len = int(getattr(args, "history_len", 5))
        self.dense_local_scale = float(getattr(args, "dense_local_scale", 1.0))
        self.use_action_history = bool(getattr(args, "use_action_history", True))
        self.use_recurrent_memory = bool(getattr(args, "use_recurrent_memory", True))
        self.recurrent_N = int(args.recurrent_N)
        self.spatial_obs_dim = 50
        self.obs_dim = self.spatial_obs_dim + self.history_len
        self.system_stats_dim = 9
        self.event_feature_dim = max(9, int(getattr(args, "event_feature_dim", 9)))
        self.tpdv = dict(dtype=torch.float32, device=device)

        type_embed_dim = max(16, min(64, self.hidden_size // 2))
        history_embed_dim = max(16, min(64, self.hidden_size // 2))

        self.local_type_embed = nn.Embedding(3, type_embed_dim)
        self.local_pos_embed = nn.Parameter(torch.zeros(1, 1, self.spatial_obs_dim, type_embed_dim))
        nn.init.normal_(self.local_pos_embed, mean=0.0, std=0.02)
        self.local_spatial_proj = nn.Sequential(
            nn.Linear(type_embed_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        self.history_action_embed = nn.Embedding(9, history_embed_dim, padding_idx=0)
        self.history_pos_embed = nn.Parameter(torch.zeros(1, 1, self.history_len, history_embed_dim))
        nn.init.normal_(self.history_pos_embed, mean=0.0, std=0.02)
        self.local_history_proj = nn.Sequential(
            nn.Linear(history_embed_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        self.local_fusion_gate = nn.Linear(self.hidden_size * 2, self.hidden_size)
        nn.init.constant_(self.local_fusion_gate.bias, 1.5)
        self.local_fusion_norm = nn.LayerNorm(self.hidden_size)
        self.local_encoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        self.event_encoder = nn.Sequential(
            nn.Linear(self.event_feature_dim, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
        )

        global_input_dim = self.hidden_size * 4 + self.system_stats_dim + 4
        self.global_encoder = ElegantMLP(
            input_dim=global_input_dim,
            output_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=max(1, int(args.layer_N)),
            activation="relu",
            use_layernorm=True,
            use_orthogonal=True,
        )
        self.rnn = RNNLayer(self.hidden_size, self.hidden_size, args.recurrent_N, args.use_orthogonal)
        self.restored_state_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.restored_state_norm = nn.LayerNorm(self.hidden_size)
        self.v_out = nn.Linear(self.hidden_size, 1)
        self.to(device)

    def _centralize_share_obs(self, share_obs):
        share_obs = torch.as_tensor(share_obs, **self.tpdv)
        if share_obs.dim() == 3:
            share_obs = share_obs[:, 0, :]
        elif share_obs.dim() == 1:
            share_obs = share_obs.unsqueeze(0)
        return share_obs

    def _prepare_event_inputs(self, base_rates, event_features, batch_size):
        if base_rates is None:
            base_rates = torch.zeros((batch_size, self.num_agents, 8), **self.tpdv)
        else:
            base_rates = torch.as_tensor(base_rates, **self.tpdv).reshape(batch_size, self.num_agents, 8)

        if event_features is None:
            event_features = torch.zeros((batch_size, self.num_agents, 8, self.event_feature_dim), **self.tpdv)
        else:
            event_features = torch.as_tensor(event_features, **self.tpdv).reshape(batch_size, self.num_agents, 8, self.event_feature_dim)
        return base_rates, event_features

    def forward(self, share_obs, rnn_states=None, masks=None, base_rates=None, event_features=None, return_state=False, restored_state=None):
        share_obs = self._centralize_share_obs(share_obs)
        expected_dim = self.num_agents * self.obs_dim + self.system_stats_dim
        if share_obs.shape[-1] != expected_dim:
            raise ValueError(
                f"Expected share_obs trailing dim {expected_dim} (= {self.num_agents} * {self.obs_dim} + {self.system_stats_dim}), "
                f"got {share_obs.shape[-1]}. Check history_len/obs_dim/share_obs_dim alignment."
            )

        vacancy_flat = share_obs[:, : self.num_agents * self.obs_dim]
        system_stats = share_obs[:, self.num_agents * self.obs_dim :]
        vacancy_obs = vacancy_flat.view(-1, self.num_agents, self.obs_dim)

        spatial_obs = vacancy_obs[:, :, : self.spatial_obs_dim].long().clamp_(0, 2)
        history_obs = vacancy_obs[:, :, self.spatial_obs_dim :].long() + 1
        history_obs = history_obs.clamp_(0, 8)
        if not self.use_action_history:
            history_obs = torch.zeros_like(history_obs)

        spatial_tokens = self.local_type_embed(spatial_obs) + self.local_pos_embed
        spatial_features = self.local_spatial_proj(spatial_tokens.mean(dim=2))
        if self.dense_local_scale != 1.0:
            spatial_features = spatial_features * self.dense_local_scale

        history_tokens = self.history_action_embed(history_obs) + self.history_pos_embed[:, :, : history_obs.shape[-1], :]
        history_features = self.local_history_proj(history_tokens.mean(dim=2))

        gate = torch.sigmoid(self.local_fusion_gate(torch.cat([spatial_features, history_features], dim=-1)))
        vacancy_features = gate * spatial_features + (1.0 - gate) * history_features
        vacancy_features = self.local_fusion_norm(vacancy_features)
        vacancy_features = self.local_encoder(vacancy_features)

        batch_size = share_obs.shape[0]
        base_rates, event_features = self._prepare_event_inputs(base_rates, event_features, batch_size)
        event_tokens = self.event_encoder(event_features.reshape(-1, self.event_feature_dim)).reshape(batch_size, self.num_agents * 8, self.hidden_size)
        legal_mask = event_features[..., 0].reshape(batch_size, self.num_agents * 8) > 0.5
        legal_float = legal_mask.unsqueeze(-1).float()
        legal_count = legal_float.sum(dim=1).clamp(min=1.0)
        event_mean = (event_tokens * legal_float).sum(dim=1) / legal_count
        masked_event_tokens = event_tokens.masked_fill(~legal_mask.unsqueeze(-1), -1e9)
        event_max = masked_event_tokens.max(dim=1).values
        has_legal = legal_mask.any(dim=1, keepdim=True)
        event_max = torch.where(has_legal, event_max, torch.zeros_like(event_max))

        rate_flat = base_rates.reshape(batch_size, self.num_agents * 8)
        legal_rate = (rate_flat > 0.0).float()
        legal_rate_count = legal_rate.sum(dim=1, keepdim=True).clamp(min=1.0)
        rate_summary = torch.cat(
            [
                torch.log1p(rate_flat.sum(dim=1, keepdim=True)),
                torch.log1p((rate_flat * legal_rate).sum(dim=1, keepdim=True) / legal_rate_count),
                torch.log1p(rate_flat.max(dim=1, keepdim=True).values),
                legal_rate.mean(dim=1, keepdim=True),
            ],
            dim=-1,
        )

        pooled_mean = vacancy_features.mean(dim=1)
        pooled_max = vacancy_features.max(dim=1).values
        global_features = self.global_encoder(
            torch.cat([pooled_mean, pooled_max, event_mean, event_max, system_stats, rate_summary], dim=-1)
        )
        if restored_state is not None:
            restored_state = torch.as_tensor(restored_state, **self.tpdv).reshape(
                batch_size, self.num_agents, self.hidden_size,
            )
            global_features = self.restored_state_norm(
                global_features + self.restored_state_proj(restored_state.mean(dim=1))
            )
        if rnn_states is None:
            rnn_states = torch.zeros(batch_size, self.recurrent_N, self.hidden_size, **self.tpdv)
        else:
            rnn_states = torch.as_tensor(rnn_states, **self.tpdv).reshape(-1, self.recurrent_N, self.hidden_size)
        if masks is None:
            masks = torch.ones(batch_size, 1, **self.tpdv)
        else:
            masks = torch.as_tensor(masks, **self.tpdv).reshape(batch_size, 1)
        if self.use_recurrent_memory:
            global_features, new_rnn_states = self.rnn(global_features, rnn_states, masks)
        else:
            new_rnn_states = torch.zeros_like(rnn_states)
        values = self.v_out(global_features)
        return (values, new_rnn_states) if return_state else values
