"""Concrete deployment of the independent potential on legal policy events."""
from dataclasses import dataclass
import numpy as np
import torch
from .cli import domain_metadata
from .clocks import PotentialProgress, PassiveKMCClock
from .network import load_potential


@dataclass(frozen=True)
class CalibratedTransition:
    event: int
    tau_before_s: float
    tau_after_s: float
    signed_progress_s: float
    accumulated_progress_s: float
    target_reached: bool
    physical_elapsed_time_s: None = None


class CalibratedPotentialSession:
    """A validated MFPT potential does not change actor logits or physical rates."""
    def __init__(self, adapter, target, checkpoint, *, require_validated=True):
        self.adapter, self.target = adapter, target
        self.domain = domain_metadata(adapter, target)
        self.model, self.validation = load_potential(checkpoint, self.domain, require_validated=require_validated)
        self.memory = target.initialize(adapter)
        self.diagnostic = PotentialProgress()

    @torch.no_grad()
    def tau_s(self):
        value = float(self.model(self.target.features(self.adapter, self.memory), np.asarray(self.memory.hit)))
        if not np.isfinite(value) or value < 0:
            raise ArithmeticError("invalid predicted remaining MFPT")
        return value

    def step(self, event):
        if self.memory.hit:
            raise RuntimeError("absorbing target already reached; start a new explicitly defined target/domain")
        before = self.tau_s()
        snapshot = self.adapter.snapshot()
        memory_before = self.memory
        try:
            self.adapter.apply(event)
            self.memory = self.target.advance(self.adapter, self.memory)
            after = self.tau_s()
        except Exception:
            self.adapter.restore(snapshot)
            self.memory = memory_before
            raise
        increment = self.diagnostic.advance(before, after)
        return CalibratedTransition(int(event), before, after, increment,
                                    self.diagnostic.progress_s, self.memory.hit)


class PassiveKMCSession:
    """Reference process: own physical event draw plus residence time together."""
    def __init__(self, adapter, target, seed=0):
        self.adapter, self.target = adapter, target
        self.memory = target.initialize(adapter)
        self.clock = PassiveKMCClock(seed)

    def step(self):
        if self.memory.hit:
            raise RuntimeError("absorbing target already reached")
        event, dt = self.clock.draw(self.adapter.rates())
        self.adapter.apply(event)
        self.memory = self.target.advance(self.adapter, self.memory)
        return {"event": event, "physical_delta_time_s": dt,
                "physical_elapsed_time_s": self.clock.elapsed_s,
                "target_reached": self.memory.hit, "clock_owner": "passive_kmc"}
