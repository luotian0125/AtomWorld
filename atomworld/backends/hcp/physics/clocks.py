"""Separated passive and policy-clock mechanisms with explicit validation gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PoissonCalibrationCertificate:
    model_name: str
    validation_dataset: str
    context_domain: str
    held_out_gate_passed: bool
    policy_clock_validated: bool = False

    def authorize_policy_clock(self):
        raise RuntimeError("Legacy boolean certificate cannot authorize a policy physical clock; use the independent MFPT potential API")
        if not self.held_out_gate_passed:
            raise RuntimeError("Poisson held-out validation gate did not pass")
        if not self.policy_clock_validated:
            raise RuntimeError(
                "passive first-passage calibration does not authorize a policy clock"
            )


class PolicyPoissonClock:
    """Accumulate only increments from an independently certified policy clock."""

    def __init__(self, certificate):
        if not isinstance(certificate, PoissonCalibrationCertificate):
            raise TypeError("certificate must be a PoissonCalibrationCertificate")
        certificate.authorize_policy_clock()
        self.certificate = certificate
        self.elapsed_s = 0.0

    def advance(self, predicted_increment_s):
        increment = float(predicted_increment_s)
        if not np.isfinite(increment) or increment < 0.0:
            raise ValueError("policy Poisson increment must be finite and non-negative")
        self.elapsed_s += increment
        return increment
