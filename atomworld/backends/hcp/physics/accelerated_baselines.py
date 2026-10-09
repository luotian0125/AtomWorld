"""Auditable TT-pair accelerated KMC baselines; no changes to the base simulator.

Scope: a basin has TWO FULL SYSTEM configurations differing by one reversible
TT hop. Every other legal transition, including every other H hop, is an exit.
This is deliberately not a many-H hypercube/general basin implementation.

AMC samples the embedded absorbing chain analytically (geometric round trips),
then gamma-distributed dwell sums conditional on the sampled path. MRM uses
entry-conditioned mean residence weights to form effective exit rates.
AS-KMC is a restricted count-based barrier-raising variant, not an exact method
or a reproduction of a general-purpose basin discovery package.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass
class PairKernel:
    internal: np.ndarray
    exits: np.ndarray

    def __post_init__(self):
        self.internal = np.asarray(self.internal, dtype=float)
        self.exits = np.asarray(self.exits, dtype=float)
        if self.internal.shape != (2,) or self.exits.shape != (2,):
            raise ValueError("A pair kernel needs two internal and two exit rates")
        if not np.all(np.isfinite(np.r_[self.internal, self.exits])):
            raise ValueError("Nonfinite pair rates")
        if np.any(self.internal <= 0) or np.any(self.exits <= 0):
            raise ValueError("Both states must have positive internal and exit rates")
        self.total = self.internal + self.exits
        self.p = self.internal / self.total
        self.escape = self.exits / self.total
        # Stable evaluation of 1-p0*p1, including when p is close to one.
        self.round_trip_escape = self.escape[0] + self.p[0] * self.escape[1]
        if self.round_trip_escape < 1e-14:
            raise ValueError("Pair too stiff for reliable finite-precision sampling")
        self.mean_residence = np.array([1., self.p[0]]) / (
            self.round_trip_escape * self.total)
        self.mean_time = float(self.mean_residence.sum())
        self.exit_probabilities = self.mean_residence * self.exits
        self.exit_probabilities /= self.exit_probabilities.sum()

    def sample_path(self, rng):
        """Number of alternating internal hops before the first absorbing hop."""
        cycles = int(rng.geometric(self.round_trip_escape)) - 1
        parity = int(rng.random() >= self.escape[0] / self.round_trip_escape)
        return 2 * cycles + parity

    def sample_amc(self, rng):
        internal_hops = self.sample_path(rng)
        # Includes the holding time before the final exit. Holding times and
        # jump decisions are independent conditional on the visited states.
        visits = ((internal_hops // 2) + 1, (internal_hops + 1) // 2)
        elapsed = sum(float(rng.gamma(n, 1. / rate)) if n else 0.
                      for n, rate in zip(visits, self.total))
        return internal_hops, elapsed

    def sample_mrm(self, rng):
        state = int(rng.random() >= self.exit_probabilities[0])
        return state, float(rng.exponential(self.mean_time))

    def transient_probabilities(self, time_s):
        """Row 0 of exp(Q*t); stable symmetric similarity transform, NumPy only."""
        if not np.isfinite(time_s) or time_s < 0:
            raise ValueError("time must be finite and nonnegative")
        scale = float(self.total.max())
        q = np.array([[-self.total[0], np.sqrt(self.internal[0]*self.internal[1])],
                      [np.sqrt(self.internal[0]*self.internal[1]), -self.total[1]]]) / scale
        values, vectors = np.linalg.eigh(q)
        row = (vectors[0] * np.exp(values * (time_s * scale))) @ vectors.T
        row[1] *= np.sqrt(self.internal[0] / self.internal[1])
        row = np.maximum(row, 0.)
        if row.sum() > 1. + 1e-9:
            raise ArithmeticError("Invalid transient probability")
        return row


@dataclass
class TTPairBasin:
    catalogs: tuple
    internal_ids: tuple
    exit_ids: tuple
    kernel: PairKernel
    state: int = 0

    def exit_event(self, state, rng):
        ids = self.exit_ids[state]
        rates = self.catalogs[state].rates.reshape(-1)[ids]
        return weighted_choice(ids, rates, rng)


def weighted_choice(ids, rates, rng):
    rates = np.asarray(rates, dtype=float)
    total = float(rates.sum())
    if total <= 0 or not np.isfinite(total):
        raise RuntimeError("No finite positive rate")
    index = min(int(np.searchsorted(np.cumsum(rates), rng.random() * total,
                                   side="right")), len(rates)-1)
    return int(ids[index])


def build_tt_pair(kmc, catalog, stats):
    """Explore one reversible edge and restore ALL simulator state afterwards."""
    ids = np.flatnonzero(catalog.legal_mask.reshape(-1)
                         & (catalog.features[..., 1].reshape(-1) > .5))
    if not len(ids):
        return None
    eid = int(ids[np.argmax(catalog.rates.reshape(-1)[ids])])
    agent, slot = catalog.decode(eid)
    source = int(kmc.lattice.hydrogen_sites[agent])
    target = int(kmc.lattice.event_targets[source, slot])
    forward = kmc.lattice.event_displacements_angstrom[source, slot].copy()
    original = kmc.state_dict()
    try:
        kmc.apply_event(eid, catalog=catalog)
        reverse_slots = np.flatnonzero(
            (kmc.lattice.event_targets[target] == source)
            & (kmc.lattice.event_kinds[target] == 0))
        if len(reverse_slots) != 1:
            return None
        reverse_slot = int(reverse_slots[0])
        if not np.allclose(forward + kmc.lattice.event_displacements_angstrom[
                target, reverse_slot], 0., atol=1e-12):
            return None  # Exclude winding loops from the two-state reduction.
        second = kmc.event_catalog()
        stats["catalog_builds"] += 1
        rid = agent * second.events_per_agent + reverse_slot
        if not second.legal_mask.reshape(-1)[rid]:
            return None
        catalogs, internal_ids = (catalog, second), (eid, rid)
        exits = tuple(np.flatnonzero(c.legal_mask.reshape(-1)
                      & (np.arange(c.rates.size) != i)) for c, i in zip(catalogs, internal_ids))
        internal_rates = [c.rates.reshape(-1)[i] for c, i in zip(catalogs, internal_ids)]
        exit_rates = [c.rates.reshape(-1)[idx].sum() for c, idx in zip(catalogs, exits)]
        try:
            kernel = PairKernel(internal_rates, exit_rates)
        except ValueError:
            stats["numerical_fallbacks"] += 1
            return None
        stats["basins_built"] += 1
        return TTPairBasin(catalogs, internal_ids, exits, kernel)
    finally:
        kmc.load_state_dict(original)


class AcceleratedHCP:
    """Owns macro statistics and clocks separately from physical micro counters.

    `kmc.step_count`, hop counts and site residence counters are NOT valid after
    macro endpoint execution. Consumers must use this object's metadata; hidden
    TT recross statistics are unavailable in endpoint mode, never zero-filled.
    """
    def __init__(self, kmc, method="amc", *, seed=1, as_nf=20, as_alpha=1.5,
                 as_gamma=2.0):
        if method not in {"lkmc", "amc", "mrm", "askmc"}:
            raise ValueError(method)
        if as_nf < 1 or as_alpha <= 1 or as_gamma <= 1:
            raise ValueError("Require Nf>=1, alpha>1, gamma>1")
        self.kmc, self.method = kmc, method
        self.rng = np.random.default_rng(seed)
        self.as_nf, self.as_alpha, self.as_gamma = as_nf, as_alpha, as_gamma
        self.basin = None
        self.as_counts = np.zeros(2, dtype=int)
        self.as_scale = 1.
        self.stats = {name: 0 for name in (
            "catalog_builds", "basins_built", "numerical_fallbacks", "macro_exits",
            "ordinary_events", "sampled_hidden_events", "endpoint_moves",
            "as_scaling_updates", "as_scaled_events", "as_eligible_basins",
            "censored_basins", "reconstructed_events")}
        self._micro_iterator = None

    def catalog(self):
        self.stats["catalog_builds"] += 1
        return self.kmc.event_catalog()

    def new_basin(self, catalog):
        self.basin = build_tt_pair(self.kmc, catalog, self.stats)
        self.as_counts[:] = 0
        self.as_scale = 1.
        return self.basin

    def _ordinary(self, catalog, horizon):
        dt = float(self.rng.exponential(1. / catalog.total_rate))
        if self.kmc.time_s + dt >= horizon:
            self.kmc.time_s = horizon
            return
        ids = np.flatnonzero(catalog.legal_mask.reshape(-1))
        eid = weighted_choice(ids, catalog.rates.reshape(-1)[ids], self.rng)
        self.kmc.apply_event(eid, catalog=catalog)
        self.kmc.time_s += dt
        self.stats["ordinary_events"] += 1

    def _place_in_pair_state(self, basin, state):
        if state != basin.state:
            self.kmc.apply_event(basin.internal_ids[basin.state],
                                 catalog=basin.catalogs[basin.state])
            basin.state = state
            self.stats["endpoint_moves"] += 1

    def _macro(self, basin, horizon):
        kernel = basin.kernel
        if self.method == "amc":
            n_internal, dt = kernel.sample_amc(self.rng)
            state = n_internal % 2
        else:
            state, dt = kernel.sample_mrm(self.rng)
            n_internal = None
        remaining = horizon - self.kmc.time_s
        if dt >= remaining:
            if self.method == "amc":
                weights = kernel.transient_probabilities(remaining)
            else:
                # Explicit observation convention for the approximate macro
                # process, NOT an exact conditional microstate distribution.
                weights = kernel.mean_residence
            state = weighted_choice([0, 1], weights, self.rng)
            self._place_in_pair_state(basin, state)
            self.stats["censored_basins"] += 1
            self.kmc.time_s = horizon
        else:
            self._place_in_pair_state(basin, state)
            eid = basin.exit_event(state, self.rng)
            self.kmc.apply_event(eid, catalog=basin.catalogs[state])
            self.kmc.time_s += dt
            self.stats["macro_exits"] += 1
            if n_internal is not None:
                self.stats["sampled_hidden_events"] += n_internal
        self.kmc.previous_transition = None  # Do not misreport macro recrosses.
        self.basin = None

    def _as_step(self, basin, horizon):
        state = basin.state
        # Conservative closure check: EVERY outgoing competitor at BOTH states
        # must be slower. Other H TT events are NOT assumed equilibrated.
        largest_exit = np.array([c.rates.reshape(-1)[ids].max()
                                 for c, ids in zip(basin.catalogs, basin.exit_ids)])
        eligible = bool(np.all(basin.kernel.internal >= self.as_gamma * largest_exit))
        if not eligible:
            self._ordinary(basin.catalogs[state], horizon)
            self.basin = None
            return
        if self.as_counts.sum() == 0 and self.as_scale == 1.:
            self.stats["as_eligible_basins"] += 1
        if np.all(self.as_counts >= self.as_nf):
            proposed = self.as_scale / self.as_alpha
            if np.all(basin.kernel.internal * proposed >= self.as_gamma * largest_exit):
                self.as_scale = proposed
                self.stats["as_scaling_updates"] += 1
            self.as_counts[:] = 0
        catalog = basin.catalogs[state]
        rates = catalog.rates.reshape(-1).copy()
        rates[basin.internal_ids[state]] *= self.as_scale
        dt = float(self.rng.exponential(1. / rates.sum()))
        if self.kmc.time_s + dt >= horizon:
            self.kmc.time_s = horizon
            return
        eid = weighted_choice(np.arange(len(rates)), rates, self.rng)
        self.kmc.apply_event(eid, catalog=catalog)  # time is owned here, not LKMC
        self.kmc.time_s += dt
        self.stats["ordinary_events"] += 1
        self.stats["as_scaled_events"] += int(self.as_scale < 1.)
        if eid == basin.internal_ids[state]:
            self.as_counts[state] += 1
            basin.state = 1-state
        else:
            self.basin = None  # Restore original rates on ANY external event.

    def run_until(self, horizon_s, *, max_iterations=1000000):
        if not np.isfinite(horizon_s) or horizon_s <= self.kmc.time_s:
            raise ValueError("horizon must be finite and later than current time")
        for _ in range(max_iterations):
            if self.kmc.time_s >= horizon_s:
                break
            if self.basin is None:
                catalog = self.catalog()
                if self.method == "lkmc":
                    self._ordinary(catalog, horizon_s)
                    continue
                self.new_basin(catalog)
                if self.basin is None:
                    self._ordinary(catalog, horizon_s)
                    continue
            if self.method == "askmc":
                self._as_step(self.basin, horizon_s)
            else:
                self._macro(self.basin, horizon_s)
        if self.kmc.time_s < horizon_s:
            raise RuntimeError("Iteration cap reached: trajectory is incomplete, not publishable")
        self.kmc.lattice.assert_consistent()
        if not np.isclose(self.kmc.energy_eV, self.kmc.calculate_energy_eV(), atol=1e-9):
            raise AssertionError("Endpoint energy disagrees with recomputation")
        return dict(self.stats)

    def _amc_microsteps(self):
        """Exact absorbing-chain paths replayed for legacy path observables.

        This mode charges EVERY underlying event and includes reconstruction
        overhead. It is not used for the macro-only physical-time benchmark.
        """
        while True:
            catalog = self.catalog()
            basin = self.new_basin(catalog)
            if basin is None:
                ids = np.flatnonzero(catalog.legal_mask.reshape(-1))
                eid = weighted_choice(ids, catalog.rates.reshape(-1)[ids], self.rng)
                dt = float(self.rng.exponential(1. / catalog.total_rate))
                self.stats["ordinary_events"] += 1
                yield self.kmc.apply_event(eid, catalog=catalog, delta_time_s=dt,
                                           clock_owner="passive_kmc")
                continue
            n_internal = basin.kernel.sample_path(self.rng)
            for index in range(n_internal + 1):
                state = index % 2
                eid = (basin.internal_ids[state] if index < n_internal
                       else basin.exit_event(state, self.rng))
                dt = float(self.rng.exponential(1. / basin.kernel.total[state]))
                self.stats["reconstructed_events"] += 1
                if index == n_internal:
                    self.stats["macro_exits"] += 1
                else:
                    self.stats["sampled_hidden_events"] += 1
                yield self.kmc.apply_event(eid, catalog=basin.catalogs[state],
                                           delta_time_s=dt, clock_owner="passive_kmc")

    def micro_step(self):
        if self.method != "amc":
            raise ValueError("Only exact AMC has a legacy microscopic-budget adapter")
        if self._micro_iterator is None:
            self._micro_iterator = self._amc_microsteps()
        return next(self._micro_iterator)
