"""Prats-Li-Stamatakis (2025), DOI 10.1021/acs.jctc.5c01394, HCP adaptation.

Implements count-based partial-equilibrium detection and bounded bidirectional
channel scaling (sections 3.1.1--3.1.4). Every call executes ONE microscopic hop.
Only TT channels are eligible; other channels remain unscaled competitors.
Local endpoint energy environments define reversible channel pairs. No global
configuration graph, tabu mask, reward, checkpoint, or hidden-event projection.
The scaled-process clock is diagnostic only, NOT an original-CTMC physical clock.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import numpy as np

from atomworld.backends.hcp.physics.accelerated_baselines import weighted_choice


@dataclass(frozen=True)
class ScalingConfig:
    # A priori short-budget HCP adaptation, not the catalyst-paper defaults.
    check_every: int = 512
    pe_tolerance: float = 0.1
    pe_min_count: int = 10
    min_separation: float = 5.0
    max_separation: float = 10.0
    down_limit: float = 2.0
    up_limit: float = 2.0
    enabled: bool = True

    def __post_init__(self):
        if self.check_every < 1 or self.pe_min_count < 1:
            raise ValueError("Positive sampling window and minimum count required")
        if not 0 < self.pe_tolerance < 0.5:
            raise ValueError("PE tolerance must be between 0 and 0.5")
        if not 1 <= self.min_separation < self.max_separation:
            raise ValueError("Require 1 <= MinSep < MaxSep")
        if not all(math.isfinite(x) for x in asdict(self).values()):
            raise ValueError("Nonfinite scaling parameter")
        if self.down_limit <= 1 or self.up_limit <= 1:
            raise ValueError("Scaling limits must exceed one")
        if self.pe_min_count <= (self.pe_tolerance + .5)/(2*self.pe_tolerance):
            raise ValueError("Minimum count violates the paper's Eq. 20 guard")


class ChannelController:
    def __init__(self, config=ScalingConfig()):
        self.config = config
        self.scales = {}
        self.counts = defaultdict(lambda: np.zeros(2, dtype=int))
        self.eligible = {}
        self.windows = 0
        self.history = []

    def register(self, key, eligible):
        self.scales.setdefault(key, 1.)
        self.eligible[key] = bool(eligible)

    def observe(self, key, direction):
        self.counts[key][direction] += 1

    def update(self, event):
        self.windows += 1
        c = self.config
        n = {k: int(self.counts[k].max()) for k in self.scales}
        eq = {}
        for k in self.scales:
            counts = self.counts[k]
            # PE ratio = forward/(forward+reverse); no symmetry exemption.
            eq[k] = bool(self.eligible[k] and n[k] >= c.pe_min_count
                         and abs(counts[0]/max(1, counts.sum())-.5) <= c.pe_tolerance)
        fastest_noneq = max([n[k] for k in n if not eq[k]] or [0])
        floor = max(fastest_noneq, c.pe_min_count)
        before = dict(self.scales)
        recovery = [k for k in n if self.scales[k] < 1.
                    and (not eq[k] or n[k] < floor)]
        if self.windows > 1 and c.enabled:
            if recovery:
                # Recovery takes precedence; no other scaling in this window.
                for k in recovery:
                    self.scales[k] = min(1., self.scales[k]*c.up_limit)
            else:
                target = math.sqrt(c.min_separation*c.max_separation)*floor
                for k in n:
                    if not eq[k]:
                        continue
                    if n[k] > c.max_separation*floor:
                        self.scales[k] *= max(1/c.down_limit, target/n[k])
                    elif self.scales[k] < 1. and n[k] < c.min_separation*floor:
                        self.scales[k] = min(1., self.scales[k]*min(c.up_limit, target/n[k]))
        changes = [{"channel": repr(k), "before": before[k], "after": self.scales[k]}
                   for k in n if before[k] != self.scales[k]]
        self.history.append({"event": event, "equilibrated_channels": sum(eq.values()),
            "fastest_nonequilibrated_count": fastest_noneq,
            "recovery_priority": bool(recovery), "changes": changes,
            "counts": {repr(k): self.counts[k].tolist() for k in n}})
        self.counts.clear()


def endpoint_environment(kmc, site, hydrogen_index):
    """Moving H excluded, so reversing a hop swaps these exact descriptors.

Nonzero pair-energy multisets retain coordination (not merely net delta-E).
This is an energy-class channel partition, not an assertion that all spatial
environments in a channel have identical downstream escape kinetics.
"""
    model, lattice = kmc.energy_model, kmc.lattice
    energies = []
    for other_index, other in enumerate(lattice.hydrogen_sites):
        if other_index != hydrogen_index:
            value = round(model.pair_energy_eV(lattice, site, int(other)), 10)
            if value:
                energies.append(value)
    return (lattice.site_type_name(site), round(model.trap_energy_eV(site), 10),
            tuple(sorted(energies)))


def classify_channels(kmc, catalog):
    ids = np.flatnonzero(catalog.legal_mask.reshape(-1))
    result = []
    source_cache = {}
    for eid in ids:
        agent, slot = catalog.decode(int(eid))
        source = int(kmc.lattice.hydrogen_sites[agent])
        target = int(kmc.lattice.event_targets[source, slot])
        kind_id = int(kmc.lattice.event_kinds[source, slot])
        if kind_id != 0:
            # Keep TO and OT as separate nonequilibrated competitors.
            result.append((int(eid), ("unscaled", kind_id), 0, False))
            continue
        if agent not in source_cache:
            source_cache[agent] = endpoint_environment(kmc, source, agent)
        left = source_cache[agent]
        right = endpoint_environment(kmc, target, agent)
        delta = kmc.lattice.event_displacements_angstrom[source, slot]
        nonzero = np.flatnonzero(np.abs(delta) > 1e-9)
        if not len(nonzero):
            raise ValueError("Zero-displacement TT event")
        canonical_delta = delta * (1 if delta[nonzero[0]] > 0 else -1)
        geometry = tuple(np.round(canonical_delta, 8))
        direction = int(left > right) if left != right else int(delta[nonzero[0]] < 0)
        result.append((int(eid), ("TT", geometry, min(left, right), max(left, right)),
                       direction, True))
    return result


class ChannelScaledHCP:
    def __init__(self, kmc, *, seed, config=ScalingConfig()):
        self.kmc = kmc
        self.rng = np.random.default_rng(seed)
        self.controller = ChannelController(config)
        self.events = 0
        self.scaled_events = 0
        self.active_steps = 0
        self.first_scaled_event = None
        self.minimum_scale = 1.
        self.scaled_process_elapsed_s = 0.

    def micro_step(self):
        catalog = self.kmc.event_catalog()
        channels = classify_channels(self.kmc, catalog)
        rates = []
        for eid, key, _, eligible in channels:
            self.controller.register(key, eligible)
            rates.append(catalog.rates.reshape(-1)[eid]*self.controller.scales[key])
        rates = np.asarray(rates)
        index = weighted_choice(np.arange(len(channels)), rates, self.rng)
        eid, key, direction, _ = channels[index]
        self.scaled_process_elapsed_s += float(self.rng.exponential(1./rates.sum()))
        scale = self.controller.scales[key]
        self.active_steps += int(any(self.controller.scales[k] < 1. for _, k, _, _ in channels))
        self.scaled_events += int(scale < 1.)
        self.minimum_scale = min(self.minimum_scale, scale)
        self.events += 1
        if scale < 1. and self.first_scaled_event is None:
            self.first_scaled_event = self.events
        # Do not mislabel a scaled clock as passive KMC or calibrated Poisson.
        transition = self.kmc.apply_event(eid, catalog=catalog)
        self.controller.observe(key, direction)
        if self.events % self.controller.config.check_every == 0:
            self.controller.update(self.events)
        return transition

    def diagnostics(self):
        return {"microscopic_events": self.events, "scaled_events": self.scaled_events,
                "active_steps": self.active_steps, "first_scaled_event": self.first_scaled_event,
                "minimum_executed_scale": self.minimum_scale,
                "scaled_process_elapsed_s_not_physical": self.scaled_process_elapsed_s,
                "scaling_windows": self.controller.history}
