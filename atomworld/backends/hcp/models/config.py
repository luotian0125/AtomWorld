from dataclasses import dataclass


@dataclass(frozen=True)
class AtomWorldModelConfig:
    hidden_dim: int = 128
    keyframe_dim: int = 128
    memory_dim: int = 128
    event_feature_dim: int = 9
    history_feature_dim: int = 12
    species_count: int = 4
    defect_type_count: int = 4
    attention_heads: int = 4
    dense_layers: int = 2
    sparse_layers: int = 1
    history_layers: int = 1
    dropout: float = 0.05
    system_stats_dim: int = 9
    mesoscale_dim: int = 9
    use_short_memory: bool = True
    use_long_memory: bool = True

    def __post_init__(self):
        if self.hidden_dim <= 0 or self.memory_dim <= 0 or self.keyframe_dim <= 0:
            raise ValueError("model dimensions must be positive")
        if self.hidden_dim % self.attention_heads != 0:
            raise ValueError("hidden_dim must be divisible by attention_heads")
