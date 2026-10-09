"""FCC local vacancy embedding used by AtomWorld-FCC."""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

from atomworld.backends.fcc.PPO.utils.mymlp import ElegantMLP
from atomworld.backends.fcc.envs.fcc_lattice import PADDING, _signed_permutations


class FCCVacancyEmbedding(nn.Module):
    """Encode the 54-site 1NN--4NN FCC environment around a vacancy."""

    def __init__(self, args, device=torch.device("cuda")):
        super().__init__()
        self.args = args
        self.device = device
        self.hidden_size = int(getattr(args, "hidden_size", 256))
        self.gnn_heads = int(getattr(args, "gnn_attention_heads", 4))
        self.dropout_rate = float(getattr(args, "gnn_attention_dropout", 0.1))
        self.num_gnn_layers = int(getattr(args, "gnn_layers", 2))
        if self.hidden_size % self.gnn_heads != 0:
            raise ValueError("hidden_size must be divisible by gnn_attention_heads")

        shells = (
            _signed_permutations((1, 1, 0)),
            _signed_permutations((2, 0, 0)),
            _signed_permutations((2, 1, 1)),
            _signed_permutations((2, 2, 0)),
        )
        node_positions = torch.as_tensor(
            np.concatenate(shells, axis=0), dtype=torch.float32, device=device
        )
        self.register_buffer("node_positions", node_positions)
        self.num_nodes = int(node_positions.shape[0])
        if self.num_nodes != 54:
            raise RuntimeError(f"FCC template must contain 54 nodes, got {self.num_nodes}")

        radius = float(getattr(args, "neighbor_radius", 4.0))
        max_neighbors = max(int(getattr(args, "max_num_neighbors", self.num_nodes)), self.num_nodes)
        distances = torch.cdist(self.node_positions, self.node_positions)
        edge_index = torch.nonzero(distances <= radius, as_tuple=False).T.contiguous()
        if max_neighbors < self.num_nodes:
            raise ValueError(
                f"max_num_neighbors={max_neighbors} would truncate the fixed FCC template; "
                f"use at least {self.num_nodes}"
            )
        self.register_buffer("edge_index", edge_index)
        self.num_edges = int(edge_index.shape[1])

        self.type_embedding = nn.Embedding(5, self.hidden_size, padding_idx=PADDING)
        nn.init.xavier_uniform_(self.type_embedding.weight[:PADDING])
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
        self.to(device)

    def forward(self, local_stats, return_node_features=False):
        local_stats = torch.as_tensor(local_stats, device=self.device)
        if local_stats.shape[-1] != self.num_nodes:
            raise ValueError(
                f"FCC local observation must end in {self.num_nodes} nodes, got {local_stats.shape}"
            )
        invalid = (local_stats < 0) | (local_stats > PADDING)
        if bool(invalid.any()):
            raise ValueError("FCC local observation contains an invalid species id")
        batch_size = int(local_stats.shape[0])
        features = self.type_embedding(local_stats.long()).reshape(-1, self.hidden_size)
        if batch_size not in self._cached_edge_index:
            shift = torch.arange(batch_size, device=self.device) * self.num_nodes
            shift = shift.repeat_interleave(self.num_edges)
            self._cached_edge_index[batch_size] = self.edge_index.repeat(1, batch_size) + shift.unsqueeze(0)
        edge_index = self._cached_edge_index[batch_size]
        for gnn, norm in zip(self.gnn_layers, self.gnn_norms):
            residual = features
            features = norm(F.relu(gnn(features, edge_index))) + residual
            features = F.dropout(features, p=self.dropout_rate, training=self.training)
        node_features = features.reshape(batch_size, self.num_nodes, self.hidden_size)
        pooled = self.post_mlp(node_features.mean(dim=1))
        if return_node_features:
            return pooled, node_features
        return pooled
