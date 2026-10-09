import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, radius_graph
from atomworld.backends.bcc.ppo.utils.mymlp import ElegantMLP
import numpy as np
from collections import OrderedDict


class VacancyEmbedding(nn.Module):
    def __init__(self, args, device=torch.device("cuda")):
        super().__init__()
        self.args = args
        self.device = device
        self.hidden_size = getattr(args, "hidden_size", 256)
        self.gnn_heads = getattr(args, "gnn_attention_heads", 4)
        self.dropout_rate = getattr(args, "gnn_attention_dropout", 0.1)
        self.num_gnn_layers = int(getattr(args, "gnn_layers", 2))

        assert self.hidden_size % self.gnn_heads == 0, (
            f"Hidden size ({self.hidden_size}) must be divisible by GNN heads ({self.gnn_heads})"
        )

        nn1 = np.array(
            [
                [1, 1, 1],
                [-1, -1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
            ]
        )
        nn2 = np.array([[2, 0, 0], [-2, 0, 0], [0, 2, 0], [0, -2, 0], [0, 0, 2], [0, 0, -2]])
        nn3 = np.array(
            [
                [2, 2, 0], [2, -2, 0], [-2, 2, 0], [-2, -2, 0],
                [2, 0, 2], [2, 0, -2], [-2, 0, 2], [-2, 0, -2],
                [0, 2, 2], [0, 2, -2], [0, -2, 2], [0, -2, -2],
            ]
        )
        nn4_list = []
        for x in [3, -3]:
            for y in [1, -1]:
                for z in [1, -1]:
                    nn4_list.extend([[x, y, z], [y, x, z], [y, z, x]])
        nn4 = np.array(sorted(list(set(map(tuple, nn4_list))), key=lambda x: (x[0], x[1], x[2])))

        node_positions = torch.tensor(np.concatenate([nn1, nn2, nn3, nn4]), dtype=torch.float32, device=device)
        self.register_buffer("node_positions", node_positions)
        self.num_nodes = self.node_positions.shape[0]
        assert self.num_nodes == 50

        self.neighbor_radius = float(getattr(args, "neighbor_radius", 4.0))
        requested_max_neighbors = int(getattr(args, "max_num_neighbors", self.num_nodes))
        self.max_num_neighbors = max(requested_max_neighbors, self.num_nodes)

        edge_index = radius_graph(
            self.node_positions,
            r=self.neighbor_radius,
            loop=True,
            max_num_neighbors=self.max_num_neighbors,
        ).to(device)
        self.register_buffer("edge_index", edge_index)
        self.num_edges = self.edge_index.size(1)

        self.type_embedding = nn.Embedding(num_embeddings=4, embedding_dim=self.hidden_size, padding_idx=3)
        nn.init.xavier_uniform_(self.type_embedding.weight[:3])

        self.gnn_layers = nn.ModuleList()
        self.gnn_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            self.gnn_layers.append(
                GATConv(
                    in_channels=self.hidden_size,
                    out_channels=self.hidden_size // self.gnn_heads,
                    heads=self.gnn_heads,
                    concat=True,
                    dropout=self.dropout_rate,
                    add_self_loops=False,
                )
            )
            self.gnn_norms.append(nn.LayerNorm(self.hidden_size))

        self.post_mlp = ElegantMLP(
            input_dim=self.hidden_size,
            output_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            activation="relu",
            use_layernorm=True,
            use_orthogonal=True,
        )
        self._cached_edge_index = {}
        self.inference_cache_size = max(0, int(getattr(args, "inference_cache_size", 128)))
        self._inference_cache = OrderedDict()
        self.to(device)

    def train(self, mode=True):
        # Encoded keyframes are valid only for the frozen weights that wrote them.
        if hasattr(self, "_inference_cache"):
            self._inference_cache.clear()
        return super().train(mode)

    def _validate_local_stats(self, local_stats: torch.Tensor) -> torch.Tensor:
        if not torch.isfinite(local_stats).all():
            raise ValueError("local_stats contains NaN or Inf values.")
        invalid_mask = (local_stats < 0) | (local_stats > 3)
        if invalid_mask.any():
            invalid_values = local_stats[invalid_mask][:10].detach().cpu().tolist()
            raise ValueError(f"Invalid atom types found in local_stats. Examples: {invalid_values}")
        return local_stats.long()

    def forward(self, local_stats: torch.Tensor, return_node_features: bool = False):
        cache_key = None
        if not self.training and not torch.is_grad_enabled() and self.inference_cache_size:
            values = local_stats.detach().cpu().numpy()
            cache_key = (values.shape, values.dtype.str, values.tobytes())
            if cache_key in self._inference_cache:
                pooled, node_feat = self._inference_cache.pop(cache_key)
                self._inference_cache[cache_key] = (pooled, node_feat)
                return (pooled, node_feat) if return_node_features else pooled
        batch_size = local_stats.size(0)
        local_stats = local_stats.to(self.device)
        x = self._validate_local_stats(local_stats)
        x_embed = self.type_embedding(x)
        x_flat = x_embed.view(-1, self.hidden_size)

        if batch_size not in self._cached_edge_index:
            shift = torch.arange(batch_size, device=self.device) * self.num_nodes
            shift = shift.repeat_interleave(self.num_edges)
            edge_index_batch = self.edge_index.repeat(1, batch_size) + shift.unsqueeze(0)
            self._cached_edge_index[batch_size] = edge_index_batch

        edge_index_batch = self._cached_edge_index[batch_size]
        feat = x_flat
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            feat_residual = feat
            feat = gnn(feat, edge_index_batch)
            feat = F.relu(feat)
            feat = norm(feat)
            feat = feat + feat_residual
            feat = F.dropout(feat, p=self.dropout_rate, training=self.training)

        node_feat = feat.view(batch_size, self.num_nodes, self.hidden_size)
        graph_feat = node_feat.mean(dim=1)
        pooled = self.post_mlp(graph_feat)
        if cache_key is not None:
            self._inference_cache[cache_key] = (pooled, node_feat)
            if len(self._inference_cache) > self.inference_cache_size:
                self._inference_cache.popitem(last=False)
        if return_node_features:
            return pooled, node_feat
        return pooled


class SparseDefectEncoder(nn.Module):
    """Encode sparse Cu/V tokens: [type_id, dx, dy, dz, dist2, shell_rank]."""

    def __init__(self, args, device=torch.device("cuda")):
        super().__init__()
        self.args = args
        self.device = device
        self.hidden_size = int(getattr(args, "hidden_size", 256))
        self.dropout_rate = float(getattr(args, "sparse_defect_dropout", 0.1))
        self.nhead = int(getattr(args, "sparse_defect_attention_heads", 4))
        self.num_layers = int(getattr(args, "sparse_defect_transformer_layers", 1))
        self.offset_scale = float(getattr(args, "sparse_defect_offset_scale", 8.0))
        self.max_shell_bins = int(max(8, getattr(args, "sparse_defect_shell_cutoff", 16) + 2))

        assert self.hidden_size % self.nhead == 0, (
            f"hidden_size ({self.hidden_size}) must be divisible by sparse attention heads ({self.nhead})."
        )

        self.type_embedding = nn.Embedding(3, self.hidden_size, padding_idx=2)
        self.shell_embedding = nn.Embedding(self.max_shell_bins, self.hidden_size, padding_idx=0)
        nn.init.xavier_uniform_(self.type_embedding.weight[:2])
        nn.init.xavier_uniform_(self.shell_embedding.weight[1:])

        self.offset_mlp = nn.Sequential(
            nn.Linear(4, self.hidden_size),
            nn.GELU(),
            nn.LayerNorm(self.hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_size,
            nhead=self.nhead,
            dim_feedforward=self.hidden_size * 2,
            dropout=self.dropout_rate,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        self.post = ElegantMLP(
            input_dim=self.hidden_size,
            output_dim=self.hidden_size,
            hidden_size=self.hidden_size,
            num_layers=2,
            activation="relu",
            use_layernorm=True,
            use_orthogonal=True,
        )
        self.to(device)

    def forward(self, tokens: torch.Tensor, token_mask: torch.Tensor, return_token_features: bool = False):
        tokens = torch.as_tensor(tokens, dtype=torch.float32, device=self.device)
        token_mask = torch.as_tensor(token_mask, dtype=torch.float32, device=self.device)
        if tokens.dim() == 2:
            tokens = tokens.unsqueeze(0)
            token_mask = token_mask.unsqueeze(0)

        batch_size, num_tokens = tokens.shape[:2]
        out = torch.zeros(batch_size, self.hidden_size, dtype=torch.float32, device=self.device)
        token_features_out = torch.zeros(batch_size, num_tokens, self.hidden_size, dtype=torch.float32, device=self.device)
        valid_rows = token_mask.sum(dim=1) > 0.0
        if not bool(valid_rows.any()):
            if return_token_features:
                return out, token_features_out, token_mask > 0.0
            return out

        tokens_v = tokens[valid_rows]
        mask_v = token_mask[valid_rows] > 0.0
        type_ids = tokens_v[..., 0].round().long().clamp(0, 2)
        offsets = tokens_v[..., 1:4] / max(self.offset_scale, 1.0)
        dist = torch.sqrt(torch.clamp(tokens_v[..., 4:5], min=0.0)) / max(self.offset_scale, 1.0)
        shell_ids = tokens_v[..., 5].round().long().clamp(0, self.max_shell_bins - 1)

        feat = self.type_embedding(type_ids) + self.shell_embedding(shell_ids) + self.offset_mlp(torch.cat([offsets, dist], dim=-1))
        feat = self.transformer(feat, src_key_padding_mask=~mask_v)
        denom = mask_v.float().sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (feat * mask_v.unsqueeze(-1).float()).sum(dim=1) / denom
        out[valid_rows] = self.post(pooled)
        token_features_out[valid_rows] = feat
        if return_token_features:
            return out, token_features_out, token_mask > 0.0
        return out
