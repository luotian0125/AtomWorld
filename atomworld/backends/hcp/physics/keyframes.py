"""Physics-owned keyframe definitions for the HCP AtomWorld extension."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KeyframeDecision:
    is_keyframe: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class KeyframeInterval:
    microsteps: int
    elapsed_physical_time_s: float
    terminal_event_id: int
    terminal_hop_kind: str
    reasons: tuple[str, ...]


class HCPKeyframeCriterion:
    """Separate rapid T-T flicker from slow or topology-changing evolution.

    This criterion classifies passive-KMC transitions only.  It does not grant
    a physical clock to a policy-selected transition sequence.
    """

    def __init__(
        self,
        *,
        slow_hop_kinds=("TO", "OT", "OO"),
        trap_delta_threshold_eV=1.0e-12,
        pair_delta_threshold_eV=1.0e-12,
    ):
        self.slow_hop_kinds = frozenset(str(kind).upper() for kind in slow_hop_kinds)
        if not self.slow_hop_kinds:
            raise ValueError("at least one slow hop kind is required")
        self.trap_delta_threshold_eV = float(trap_delta_threshold_eV)
        self.pair_delta_threshold_eV = float(pair_delta_threshold_eV)
        if self.trap_delta_threshold_eV < 0.0 or self.pair_delta_threshold_eV < 0.0:
            raise ValueError("keyframe energy thresholds must be non-negative")

    def classify(self, transition):
        reasons = []
        if str(transition.hop_kind).upper() in self.slow_hop_kinds:
            reasons.append("slow_hop_kind")
        trap_delta = abs(float(transition.info.get("trap_delta_eV", 0.0)))
        pair_delta = abs(float(transition.info.get("pair_delta_eV", 0.0)))
        if trap_delta > self.trap_delta_threshold_eV:
            reasons.append("trap_state_change")
        if pair_delta > self.pair_delta_threshold_eV:
            reasons.append("precursor_topology_change")
        return KeyframeDecision(bool(reasons), tuple(reasons))


class KeyframeTracker:
    """Accumulate passive microsteps until the next physics keyframe."""

    def __init__(self, criterion=None):
        self.criterion = criterion or HCPKeyframeCriterion()
        self.microsteps_since_keyframe = 0
        self.time_since_keyframe_s = 0.0
        self.intervals = []

    def observe(self, transition):
        self.microsteps_since_keyframe += 1
        self.time_since_keyframe_s += float(transition.delta_time_s)
        decision = self.criterion.classify(transition)
        if not decision.is_keyframe:
            return None
        interval = KeyframeInterval(
            microsteps=self.microsteps_since_keyframe,
            elapsed_physical_time_s=self.time_since_keyframe_s,
            terminal_event_id=int(transition.event_id),
            terminal_hop_kind=str(transition.hop_kind),
            reasons=decision.reasons,
        )
        self.intervals.append(interval)
        self.microsteps_since_keyframe = 0
        self.time_since_keyframe_s = 0.0
        return interval
