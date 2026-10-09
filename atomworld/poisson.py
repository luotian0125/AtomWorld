"""Backward Poisson equation, with explicit absorbing boundary and SI units.

The potential decrement is SIGNED progress, never a residence-time clock.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve


def checked_rates(rates):
    rates = np.asarray(rates, dtype=np.float64)
    if rates.ndim != 1 or not np.isfinite(rates).all() or np.any(rates < 0):
        raise ValueError("rates must be a finite nonnegative vector in s^-1")
    if not np.isfinite(rates.sum()):
        raise ValueError("total rate overflow")
    return rates


def signed_progress(current_tau_s, successor_tau_s):
    """Paper Eq. (9), evaluated as tau(s)-tau(s') including zero-rate targets."""
    values = np.asarray(current_tau_s) - np.asarray(successor_tau_s)
    if not np.isfinite(values).all():
        raise ValueError("nonfinite MFPT potential")
    return values


def normalized_residual(tau_s, successor_tau_s, rates):
    """Gamma-normalized residual: Gamma*(tau-E_p[tau'])-1."""
    rates = checked_rates(rates)
    gamma = rates.sum()
    if gamma <= 0:
        raise ValueError("nonabsorbing zero-rate state has infinite MFPT")
    return float(np.dot(rates, signed_progress(tau_s, successor_tau_s)) - 1.0)


@dataclass
class FiniteRateGraph:
    rates: list
    successors: list
    absorbing: np.ndarray
    features: np.ndarray | None = None

    def __post_init__(self):
        self.absorbing = np.asarray(self.absorbing, dtype=bool)
        n = len(self.absorbing)
        if not n or len(self.rates) != n or len(self.successors) != n or not self.absorbing.any():
            raise ValueError("graph requires an explicit, nonempty absorbing boundary")
        self.rates = [checked_rates(row) for row in self.rates]
        indices = []
        for i, (rates, successors) in enumerate(zip(self.rates, self.successors)):
            raw = np.asarray(successors)
            if raw.ndim != 1 or np.any(raw != raw.astype(np.int64)):
                raise ValueError("successors must be integer state indices")
            row = raw.astype(np.int64)
            if rates.shape != row.shape or np.any(row < 0) or np.any(row >= n):
                raise ValueError("invalid event successor catalogue")
            if not self.absorbing[i] and rates.sum() <= 0:
                raise ValueError("nonabsorbing dead state has infinite MFPT")
            indices.append(row)
        self.successors = indices
        # Every state must reach the target. In a finite chain this excludes
        # closed communicating classes outside the boundary and implies a.s. hit.
        reverse = [[] for _ in range(n)]
        for i, (rates, js) in enumerate(zip(self.rates, self.successors)):
            if not self.absorbing[i]:
                for j in js[rates > 0]:
                    reverse[int(j)].append(i)
        reachable = set(np.flatnonzero(self.absorbing).tolist())
        frontier = list(reachable)
        while frontier:
            for i in reverse[frontier.pop()]:
                if i not in reachable:
                    reachable.add(i)
                    frontier.append(i)
        if len(reachable) != n:
            raise ValueError("target is unreachable from a closed class; MFPT is infinite")
        if self.features is not None:
            self.features = np.asarray(self.features, dtype=np.float64)
            if self.features.ndim != 2 or self.features.shape[0] != n or not np.isfinite(self.features).all():
                raise ValueError("invalid graph features")

    def operator(self):
        interior = np.flatnonzero(~self.absorbing)
        lookup = {int(i): k for k, i in enumerate(interior)}
        rows, cols, values = [], [], []
        gamma = np.array([self.rates[i].sum() for i in interior])
        for k, i in enumerate(interior):
            rows.append(k); cols.append(k); values.append(1.0)
            for rate, j in zip(self.rates[i], self.successors[i]):
                if j in lookup and rate > 0:
                    rows.append(k); cols.append(lookup[int(j)]); values.append(-rate / gamma[k])
        matrix = sparse.coo_matrix((values, (rows, cols)), shape=(len(interior), len(interior))).tocsr()
        return interior, gamma, matrix

    def solve(self):
        """Exact first two hitting-time moments; no sampled/censored labels."""
        interior, gamma, matrix = self.operator()
        tau = np.zeros(len(self.absorbing), dtype=np.float64)
        second = tau.copy()
        if len(interior):
            tau[interior] = spsolve(matrix, 1.0 / gamma)
            second[interior] = spsolve(matrix, 2.0 * tau[interior] / gamma)
        if not np.isfinite(tau).all() or np.any(tau < 0):
            raise ArithmeticError("invalid MFPT solution; inspect conditioning and rates")
        residuals = self.residuals(tau)
        if np.max(np.abs(residuals), initial=0.0) > 1e-7:
            raise ArithmeticError("backward equation failed numerical verification")
        return tau, second

    def residuals(self, tau_s):
        tau_s = np.asarray(tau_s, dtype=np.float64)
        if tau_s.shape != self.absorbing.shape or not np.isfinite(tau_s).all():
            raise ValueError("invalid potential shape/value")
        return np.array([tau_s[i] if self.absorbing[i] else
                         normalized_residual(tau_s[i], tau_s[self.successors[i]], self.rates[i])
                         for i in range(len(tau_s))])

    def sample_hitting_time(self, start, rng, max_events=1000000):
        state = int(start)
        if not 0 <= state < len(self.absorbing):
            raise ValueError("invalid initial state")
        elapsed = 0.0
        for _ in range(max_events):
            if self.absorbing[state]:
                return elapsed
            rates = self.rates[state]
            gamma = rates.sum()
            elapsed += float(rng.exponential(1.0 / gamma))
            event = int(rng.choice(len(rates), p=rates / gamma))
            state = int(self.successors[state][event])
        raise RuntimeError("right-censored path: never treat its duration as a completed MFPT label")
