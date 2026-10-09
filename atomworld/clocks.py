"""Clock ownership: KMC residence time requires physical event selection."""
import numpy as np
from .poisson import checked_rates, signed_progress


class PassiveKMCClock:
    def __init__(self, seed=0):
        self.rng = np.random.default_rng(seed)
        self.elapsed_s = 0.0

    def draw(self, rates):
        rates = checked_rates(rates)
        gamma = rates.sum()
        if gamma <= 0:
            raise ValueError("no legal positive-rate event")
        # This object owns BOTH draws. It does not accept policy-selected events.
        event = int(self.rng.choice(len(rates), p=rates / gamma))
        dt = float(self.rng.exponential(1.0 / gamma))
        self.elapsed_s += dt
        return event, dt


class PotentialProgress:
    """Signed diagnostic in seconds. It deliberately has no elapsed_s field."""
    def __init__(self):
        self.progress_s = 0.0

    def advance(self, current_tau_s, successor_tau_s):
        increment = float(signed_progress(current_tau_s, successor_tau_s))
        self.progress_s += increment
        return increment
