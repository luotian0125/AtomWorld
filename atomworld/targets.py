"""Absorbing targets. Persistence memory is part of the Poisson state."""
from copy import deepcopy
from dataclasses import dataclass, field
import numpy as np


@dataclass
class TargetMemory:
    phase: int = 0
    streak: int = 0
    previous_mask: np.ndarray | None = None
    hit: bool = False


@dataclass
class Target:
    kind: str
    options: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.kind not in {"site", "energy_drop", "bcc_arrival", "fcc_nucleus", "hcp_precursor"}:
            raise ValueError("unknown target kind")
        if int(self.options.get("sample_interval", 1)) < 1:
            raise ValueError("sample interval must be positive")
        if self.kind == "site" and "site" not in self.options:
            raise ValueError("site target needs a site index or coordinate")
        if self.kind == "energy_drop" and ("reference_eV" not in self.options or float(self.options.get("drop_eV", 0)) <= 0):
            raise ValueError("energy target needs fixed reference_eV and positive drop_eV")

    def initialize(self, adapter):
        memory = TargetMemory()
        self._observe(adapter, memory)
        return memory

    def advance(self, adapter, memory):
        memory = deepcopy(memory)
        if memory.hit:
            return memory
        interval = int(self.options.get("sample_interval", 1))
        memory.phase = (memory.phase + 1) % interval
        if memory.phase == 0:
            self._observe(adapter, memory)
        return memory

    def _observe(self, adapter, memory):
        k, cfg = adapter.kernel, self.options
        if self.kind == "site":
            site = np.asarray(cfg["site"])
            if adapter.material == "hcp":
                reached = int(k.lattice.hydrogen_sites[0]) == int(site)
            else:
                reached = np.array_equal(k.vacancy_pos[0], site)
        elif self.kind == "energy_drop":
            reached = adapter.energy_eV() <= float(cfg["reference_eV"]) - float(cfg["drop_eV"])
        elif self.kind == "bcc_arrival":
            if adapter.material != "bcc":
                raise ValueError("BCC arrival target requires BCC")
            positions = np.asarray(k.cu_pos)
            lookup = {tuple(p): i for i, p in enumerate(positions)}
            unseen, components = set(range(len(positions))), []
            while unseen:
                root = unseen.pop(); component = {root}; stack = [root]
                while stack:
                    current = stack.pop()
                    for d in k.NN1:
                        j = lookup.get(tuple((positions[current] + d) % k.lattice.shape))
                        if j in unseen:
                            unseen.remove(j); component.add(j)
                            stack.append(j)
                components.append(component)
            stable = set().union(*(c for c in components if len(c) >= int(cfg.get("min_cluster_size", 4))))
            reached = any(lookup.get(tuple((p+d) % k.lattice.shape)) in stable
                          for p in k.vacancy_pos for d in k.NN1)
        elif self.kind == "fcc_nucleus":
            if adapter.material != "fcc":
                raise ValueError("FCC nucleus target requires FCC")
            from .backends.fcc.fcc_l12 import L12DetectorConfig, analyze_l12_lattice
            detector = L12DetectorConfig(**cfg.get("detector", {}))
            _, masks = analyze_l12_lattice(k.lattice, detector, return_masks=True)
            stable = masks["stable_mask"]
            overlap = 0 if memory.previous_mask is None else np.count_nonzero(stable & memory.previous_mask)
            reached = overlap >= detector.min_cluster_cells
            memory.previous_mask = stable.copy()
        else:
            if adapter.material != "hcp":
                raise ValueError("HCP precursor target requires HCP")
            from .backends.hcp.analysis.clusters import hydrogen_cluster_metrics
            metrics = hydrogen_cluster_metrics(k.lattice, k.energy_model)
            attractive = {s.label for s in k.energy_model.pair_shells if s.interaction_energy_eV < 0}
            pairs = sum(v for label, v in metrics["pair_counts"].items() if label in attractive)
            reached = (pairs / max(k.num_agents, 1) >= cfg.get("min_pairs_per_h", .40)
                       and metrics["largest_attractive_cluster_fraction"] >= cfg.get("min_largest_fraction", .14)
                       and 1 - metrics["isolated_hydrogen_fraction"] >= cfg.get("min_nonisolated_fraction", .70))
        memory.streak = memory.streak + 1 if reached else 0
        memory.hit = memory.streak >= int(cfg.get("persistence_samples", 1))

    def features(self, adapter, memory):
        extra = [float(memory.phase), float(memory.streak), float(memory.hit)]
        if self.kind == "fcc_nucleus":
            shape = tuple(adapter.kernel.size)
            extra += [float(memory.previous_mask is not None)]
            mask = np.zeros(shape) if memory.previous_mask is None else memory.previous_mask
            extra.extend(np.asarray(mask).reshape(-1).astype(float))
        return np.concatenate([adapter.features(), np.asarray(extra)])

    def key(self, adapter, memory):
        return self.features(adapter, memory).tobytes()


def paper_target(material, adapter):
    """Prospective target choices traced to paper/code; not retrospective fits."""
    if material == "bcc":
        if len(adapter.kernel.cu_pos) < 4:
            raise ValueError("paper BCC arrival target requires at least 4 Cu atoms")
        return Target("bcc_arrival", {"min_cluster_size": 4})
    if material == "fcc":
        # The existing experiment observes at nonuniform event-dose checkpoints.
        # A fixed 0.25-dose cadence is an EXPLICIT prospective calibration choice.
        return Target("fcc_nucleus", {"sample_interval": max(1, int(len(adapter.kernel.coords) / 4))})
    if material == "hcp":
        if adapter.kernel.num_agents < 2 or not any(
                s.interaction_energy_eV < 0 for s in adapter.kernel.energy_model.pair_shells):
            raise ValueError("paper HCP precursor target requires multiple H and a sourced attractive pair model; supply --energy-file")
        return Target("hcp_precursor", {"sample_interval": 16, "persistence_samples": 4,
                                       "min_pairs_per_h": .40, "min_largest_fraction": .14,
                                       "min_nonisolated_fraction": .70})
    raise ValueError("unknown material")
