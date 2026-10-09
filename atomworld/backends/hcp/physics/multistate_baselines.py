"""Bounded full-configuration TT basins and reversible phase-type kinetics.

Separate from the frozen v1 implementation. All omitted edges remain exits.
AMC samples joint first-exit time/state by the killed CTMC semigroup; MRM uses
entry-conditioned mean residence weights. AS is explicitly a count-based graph
variant, NOT a claim to reproduce Chatterjee-Voter's confidence calibration.
"""
from __future__ import annotations

from dataclasses import dataclass
import time
import numpy as np

from atomworld.backends.hcp.physics.accelerated_baselines import weighted_choice
from atomworld.backends.hcp.envs.hcp_kmc import K_B_EV_PER_K


class UnsupportedBasin(ValueError):
    """Numerically unsafe or unsupported reduction: caller must fall back."""


class ReversibleKilledKernel:
    def __init__(self, internal_rates, exit_rates, stationary_weights):
        r = np.asarray(internal_rates, dtype=float)
        e = np.asarray(exit_rates, dtype=float)
        w = np.asarray(stationary_weights, dtype=float)
        n = len(e)
        if r.shape != (n, n) or w.shape != (n,) or n < 1:
            raise UnsupportedBasin("Inconsistent kernel dimensions")
        if not all(np.all(np.isfinite(x)) for x in (r, e, w)):
            raise UnsupportedBasin("Nonfinite kernel")
        if np.any(r < 0) or np.any(e < 0) or np.any(w <= 0) or np.any(np.diag(r) != 0):
            raise UnsupportedBasin("Invalid rates/weights")
        self.internal = r.copy()
        self.exits = e.copy()
        self.total = r.sum(axis=1) + e
        self.scale = float(self.total.max())
        if self.scale <= 0 or w.min()/w.max() < 1e-100:
            raise UnsupportedBasin("Degenerate rates or weight dynamic range")
        self.generator = r - np.diag(self.total)
        self.sqrt_weights = np.sqrt(w / w.max())
        symmetric = (self.sqrt_weights[:, None] * self.generator
                     / self.sqrt_weights[None, :]) / self.scale
        if not np.allclose(symmetric, symmetric.T, atol=2e-10, rtol=2e-8):
            raise UnsupportedBasin("Internal detailed balance not satisfied")
        self.eigenvalues, self.eigenvectors = np.linalg.eigh((symmetric+symmetric.T)/2)
        if self.eigenvalues[-1] >= -1e-13:
            raise UnsupportedBasin("Nonabsorbing or excessively ill-conditioned basin")
        self.fundamental = np.linalg.solve(-self.generator/self.scale, np.eye(n))/self.scale
        if np.min(self.fundamental) < -1e-10/self.scale:
            raise UnsupportedBasin("Invalid fundamental matrix")
        self.fundamental = np.maximum(self.fundamental, 0.)
        if not np.allclose(self.fundamental @ e, 1., atol=1e-7):
            raise UnsupportedBasin("Exit probability does not sum to one")

    def transient(self, elapsed, entry=0):
        if not np.isfinite(elapsed) or elapsed < 0:
            raise ValueError("Invalid elapsed time")
        row = ((self.eigenvectors[entry] * np.exp(self.eigenvalues*self.scale*elapsed))
               @ self.eigenvectors.T) * self.sqrt_weights/self.sqrt_weights[entry]
        if np.min(row) < -1e-8 or row.sum() > 1.+1e-7:
            raise ArithmeticError("Invalid spectral transient probability")
        return np.maximum(row, 0.)

    def sample_amc(self, rng, horizon, entry=0):
        """Return elapsed, pre-exit/transient state, censored; no hidden count fiction."""
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError("Positive finite horizon required")
        u = max(float(rng.random()), np.finfo(float).tiny)
        boundary = self.transient(horizon, entry)
        if u < boundary.sum():
            return horizon, weighted_choice(np.arange(len(boundary)), boundary, rng), True
        low, high = 0., horizon
        # Fixed-iteration inverse survival: retain time/exit-state correlation.
        for _ in range(64):
            middle = (low+high)/2
            if self.transient(middle, entry).sum() > u:
                low = middle
            else:
                high = middle
        elapsed = (low+high)/2
        flux = self.transient(elapsed, entry) * self.exits
        return elapsed, weighted_choice(np.arange(len(flux)), flux, rng), False

    def sample_mrm(self, rng, horizon, entry=0):
        residence = self.fundamental[entry]
        elapsed = float(rng.exponential(residence.sum()))
        if elapsed >= horizon:
            # Explicit approximate observation projection, not exact microstate law.
            return horizon, weighted_choice(np.arange(len(residence)), residence, rng), True
        return elapsed, weighted_choice(np.arange(len(residence)), residence*self.exits, rng), False


def configuration_key(kmc):
    return tuple(map(int, kmc.lattice.hydrogen_sites))


def destination_key(key, catalog, event_id, lattice):
    agent, slot = catalog.decode(int(event_id))
    result = list(key)
    result[agent] = int(lattice.event_targets[key[agent], slot])
    return tuple(result)


def eligible_tt(kmc, catalog, event_id, cutoff):
    agent, slot = catalog.decode(int(event_id))
    f = catalog.features[agent, slot]
    if f[1] < .5:
        return False
    # Both directional barriers must satisfy the cutoff. TT sites share the
    # same intrinsic energy; the extra-energy rule is base + max(0, deltaE).
    reverse_barrier = kmc.parameters.migration_barriers_eV["TT"] + max(0., -float(f[11]))
    return max(float(f[12]), reverse_barrier) <= cutoff + 1e-8


@dataclass
class ConfigurationBasin:
    snapshots: list
    catalogs: list
    internal_targets: list
    exit_ids: list
    rates: np.ndarray
    exit_rates: np.ndarray
    closed_fast_graph: bool
    capacity_reached: bool
    kernel: ReversibleKilledKernel | None
    kernel_failure: str | None

    @property
    def size(self):
        return len(self.snapshots)


def build_configuration_basin(kmc, *, max_states=32, barrier_cutoff_eV=.25):
    """BFS through reversible low-barrier TT edges; every catalog remains complete.

    Configurations are full labeled-H occupancy states, not independent-particle
    approximations. No cache survives an exit. A zero-winding check rejects any
    ambiguity in unwrapped endpoint displacements.
    """
    if not 2 <= max_states <= 256 or not np.isfinite(barrier_cutoff_eV) or barrier_cutoff_eV <= 0:
        raise ValueError("Require 2<=max_states<=256 and a positive finite cutoff")
    original = kmc.state_dict()
    snapshots, keys = [original], [configuration_key(kmc)]
    index = {keys[0]: 0}
    catalogs = []
    try:
        cursor = 0
        while cursor < len(snapshots):
            kmc.load_state_dict(snapshots[cursor])
            catalog = kmc.event_catalog()
            catalogs.append(catalog)
            ids = np.flatnonzero(catalog.legal_mask.reshape(-1))
            # Rate order is deterministic and independent of test outcomes.
            ids = sorted(ids, key=lambda eid: (-catalog.rates.reshape(-1)[eid], int(eid)))
            for eid in ids:
                if not eligible_tt(kmc, catalog, eid, barrier_cutoff_eV):
                    continue
                target = destination_key(keys[cursor], catalog, eid, kmc.lattice)
                if target in index or len(snapshots) >= max_states:
                    continue
                kmc.load_state_dict(snapshots[cursor])
                kmc.apply_event(int(eid), catalog=catalog)
                index[target] = len(snapshots)
                keys.append(target)
                snapshots.append(kmc.state_dict())
            cursor += 1
        n = len(snapshots)
        rates, exits = np.zeros((n,n)), np.zeros(n)
        mappings, exit_ids = [], []
        closed, capacity = True, False
        for i, (key, catalog) in enumerate(zip(keys, catalogs)):
            mapping, outside = {}, []
            kmc.load_state_dict(snapshots[i])
            for eid in np.flatnonzero(catalog.legal_mask.reshape(-1)):
                eid = int(eid)
                target = destination_key(key, catalog, eid, kmc.lattice)
                fast = eligible_tt(kmc, catalog, eid, barrier_cutoff_eV)
                j = index.get(target)
                if j is not None:
                    agent, slot = catalog.decode(eid)
                    displacement = snapshots[i]["unwrapped_displacements_angstrom"].copy()
                    displacement[agent] += kmc.lattice.event_displacements_angstrom[key[agent], slot]
                    if not np.allclose(displacement, snapshots[j]["unwrapped_displacements_angstrom"], atol=1e-10):
                        raise UnsupportedBasin("Winding state cannot be identified by occupancy alone")
                    mapping[eid] = j
                    rates[i,j] += catalog.rates.reshape(-1)[eid]
                    if not fast:
                        closed = False  # Conservative AS criterion.
                else:
                    outside.append(eid)
                    exits[i] += catalog.rates.reshape(-1)[eid]
                    if fast:
                        closed = False
                        capacity |= n >= max_states
            mappings.append(mapping)
            exit_ids.append(np.asarray(outside, dtype=int))
        energies = np.array([s["energy_eV"] for s in snapshots])
        weights = np.exp(-(energies-energies.min())/(K_B_EV_PER_K*kmc.temperature_K))
        failure = None
        try:
            kernel = ReversibleKilledKernel(rates, exits, weights)
        except UnsupportedBasin as error:
            kernel, failure = None, str(error)
        return ConfigurationBasin(snapshots, catalogs, mappings, exit_ids, rates, exits,
                                  closed, capacity, kernel, failure)
    finally:
        kmc.load_state_dict(original)


class GraphCountScaling:
    """Conservative graph-wide AS variant with explicit closure and visit gates."""
    def __init__(self, basin, *, nf=20, alpha=1.5, gamma=2.):
        if nf < 1 or not all(np.isfinite(v) and v > 1 for v in (alpha,gamma)):
            raise ValueError("Require Nf>=1, finite alpha>1, gamma>1")
        self.basin, self.nf, self.alpha, self.gamma = basin,nf,alpha,gamma
        self.counts = {(i,eid): 0 for i,mapping in enumerate(basin.internal_targets) for eid in mapping}
        self.factor = 1.
        self.updates = 0

    def maybe_scale(self):
        if not self.basin.closed_fast_graph or not self.counts or min(self.counts.values()) < self.nf:
            return False
        proposed = self.factor/self.alpha
        for i, catalog in enumerate(self.basin.catalogs):
            outside = self.basin.exit_ids[i]
            max_exit = float(catalog.rates.reshape(-1)[outside].max()) if len(outside) else 0.
            ids = list(self.basin.internal_targets[i])
            if ids and min(catalog.rates.reshape(-1)[ids])*proposed < self.gamma*max_exit:
                return False
        self.factor = proposed
        self.updates += 1
        self.counts = dict.fromkeys(self.counts, 0)
        return True

    def rates(self, state):
        result = self.basin.catalogs[state].rates.reshape(-1).copy()
        for eid in self.basin.internal_targets[state]:
            result[eid] *= self.factor
        return result


class MultistateHCP:
    """Physical-time endpoint engine. No fabricated microscopic counts/AUCs.

    Supports exact spectral AMC, approximate MRM, graph-count AS, and an exact
    unscaled cached-catalog control to separate caching from rate scaling.
    """
    def __init__(self, kmc, method="amc", *, seed=1, max_states=32,
                 barrier_cutoff_eV=.25, as_nf=20, as_alpha=1.5, as_gamma=2.):
        if method not in {"amc", "mrm", "askmc", "cached"}:
            raise ValueError(method)
        self.kmc,self.method,self.rng = kmc,method,np.random.default_rng(seed)
        self.max_states,self.cutoff = max_states,barrier_cutoff_eV
        self.as_options = dict(nf=as_nf, alpha=as_alpha, gamma=as_gamma)
        self.stats = dict(basins_built=0, catalog_builds=0, total_basin_states=0,
            largest_basin=0, capacity_hits=0, closed_fast_basins=0, numerical_fallbacks=0,
            ordinary_events=0, macro_exits=0, censored_basins=0, as_scaling_updates=0,
            as_scaled_events=0, build_seconds=0., kernel_sampling_seconds=0.)
        self.basin = None
        self.state = 0
        self.scaling = None

    def build(self):
        started = time.perf_counter()
        self.basin = build_configuration_basin(self.kmc, max_states=self.max_states,
                                               barrier_cutoff_eV=self.cutoff)
        self.stats["build_seconds"] += time.perf_counter()-started
        self.stats["basins_built"] += 1
        self.stats["catalog_builds"] += self.basin.size
        self.stats["total_basin_states"] += self.basin.size
        self.stats["largest_basin"] = max(self.stats["largest_basin"],self.basin.size)
        self.stats["capacity_hits"] += int(self.basin.capacity_reached)
        self.stats["closed_fast_basins"] += int(self.basin.closed_fast_graph)
        self.stats["numerical_fallbacks"] += int(self.basin.kernel is None)
        self.state = 0
        self.scaling = GraphCountScaling(self.basin, **self.as_options)

    def explicit_step(self, horizon):
        if self.method == "askmc":
            self.stats["as_scaling_updates"] += int(self.scaling.maybe_scale())
            rates = self.scaling.rates(self.state)
        else:
            rates = self.basin.catalogs[self.state].rates.reshape(-1)
        dt = float(self.rng.exponential(1./rates.sum()))
        if self.kmc.time_s+dt >= horizon:
            self.kmc.time_s = horizon
            return
        eid = weighted_choice(np.arange(len(rates)), rates, self.rng)
        self.kmc.apply_event(eid, catalog=self.basin.catalogs[self.state])
        self.kmc.time_s += dt
        self.stats["ordinary_events"] += 1
        self.stats["as_scaled_events"] += int(self.method == "askmc" and self.scaling.factor < 1.)
        target = self.basin.internal_targets[self.state].get(eid)
        if target is None:
            self.basin = None
            self.scaling = None  # Any exit invalidates ALL counts and altered rates.
        else:
            self.scaling.counts[self.state,eid] += 1
            self.state = target

    def macro_step(self, horizon):
        current_time = self.kmc.time_s
        started = time.perf_counter()
        sampler = self.basin.kernel.sample_amc if self.method == "amc" else self.basin.kernel.sample_mrm
        dt,state,censored = sampler(self.rng,horizon-current_time)
        self.stats["kernel_sampling_seconds"] += time.perf_counter()-started
        self.kmc.load_state_dict(self.basin.snapshots[state])
        if censored:
            self.stats["censored_basins"] += 1
        else:
            catalog = self.basin.catalogs[state]
            ids = self.basin.exit_ids[state]
            eid = weighted_choice(ids, catalog.rates.reshape(-1)[ids], self.rng)
            self.kmc.apply_event(eid,catalog=catalog)
            self.stats["macro_exits"] += 1
        self.kmc.time_s = horizon if censored else current_time+dt
        self.kmc.previous_transition = None
        self.basin = None

    def run_until(self, horizon, *, max_iterations=1000000):
        if not np.isfinite(horizon) or horizon <= self.kmc.time_s:
            raise ValueError("A later finite horizon is required")
        for _ in range(max_iterations):
            if self.kmc.time_s >= horizon:
                break
            if self.basin is None:
                self.build()
            if self.method in {"cached", "askmc"} or self.basin.kernel is None:
                self.explicit_step(horizon)
            else:
                self.macro_step(horizon)
        if self.kmc.time_s < horizon:
            raise RuntimeError("Iteration cap reached: incomplete trajectory")
        self.kmc.lattice.assert_consistent()
        if not np.isclose(self.kmc.energy_eV,self.kmc.calculate_energy_eV(),atol=1e-9):
            raise AssertionError("Endpoint energy mismatch")
        return dict(self.stats)
