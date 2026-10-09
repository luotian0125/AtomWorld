from dataclasses import dataclass
from typing import Any, Dict

import numpy as np


@dataclass(frozen=True)
class EventCatalog:
    """Immutable rectangular catalogue of simulator candidate events."""

    rates: np.ndarray
    features: np.ndarray

    def __post_init__(self):
        rates = np.asarray(self.rates, dtype=np.float64)
        features = np.asarray(self.features, dtype=np.float32)
        if rates.ndim != 2:
            raise ValueError(
                f"rates must have shape [agents, events], got {rates.shape}"
            )
        if features.ndim != 3 or features.shape[:2] != rates.shape:
            raise ValueError(
                "features must have shape [agents, events, features] matching rates; "
                f"got rates={rates.shape}, features={features.shape}"
            )
        if np.any(rates < 0.0) or not np.all(np.isfinite(rates)):
            raise ValueError("event rates must be finite and non-negative")
        object.__setattr__(self, "rates", rates.copy())
        object.__setattr__(self, "features", features.copy())

    @property
    def legal_mask(self) -> np.ndarray:
        return (self.features[..., 0] > 0.5) & (self.rates > 0.0)

    @property
    def total_rate(self) -> float:
        return float(self.rates[self.legal_mask].sum(dtype=np.float64))

    @property
    def num_agents(self) -> int:
        return int(self.rates.shape[0])

    @property
    def events_per_agent(self) -> int:
        return int(self.rates.shape[1])

    def decode(self, event_id: int) -> tuple[int, int]:
        size = self.num_agents * self.events_per_agent
        if event_id < 0 or event_id >= size:
            raise IndexError(f"event_id={event_id} is outside [0, {size})")
        return divmod(int(event_id), self.events_per_agent)

    def assert_legal(self, event_id: int) -> tuple[int, int]:
        agent, event = self.decode(event_id)
        if not bool(self.legal_mask[agent, event]):
            raise ValueError(f"event_id={event_id} is not physically admissible")
        return agent, event


@dataclass(frozen=True)
class TransitionResult:
    event_id: int
    hydrogen_index: int
    local_event_index: int
    source_site: int
    target_site: int
    hop_kind: str
    delta_time_s: float
    energy_before_eV: float
    energy_after_eV: float
    displacement_angstrom: np.ndarray
    info: Dict[str, Any]

    @property
    def delta_energy_eV(self) -> float:
        return float(self.energy_after_eV - self.energy_before_eV)
