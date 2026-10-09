"""HCP fidelity-v1 estimands; time-grid ensembles and seed-cluster bootstrap.

These explicit definitions are NEW HCP metrics, not retroactive definitions of
untraced numbers in a manuscript. No same-microevent-count clock comparison.
"""
from __future__ import annotations

import numpy as np

DYNAMIC_KEYS = ("basal_msd_A2", "c_msd_A2")
STRUCTURE_KEYS = ("O_site_fraction", "largest_cluster_fraction", "nonisolated_fraction")


def ensemble_errors(reference, comparison, reference_fpt, comparison_fpt):
    """Arrays: (seed, time, observable), dynamics first, then structure.

ClockErr = |mean FPT_cmp / mean FPT_ref - 1|, to the same non-TT count.
DynErr = mean over basal/c-axis MSD ensemble-curve NRMSE, reference-range scale.
StructErr = mean absolute ensemble-curve difference over three bounded fractions.
Results are discrepancies of finite ensemble estimates, not pure algorithm bias.
"""
    r, c = np.asarray(reference, float), np.asarray(comparison, float)
    if r.shape != c.shape or r.ndim != 3 or r.shape[2] != 5 or r.shape[0] < 2:
        raise ValueError("Matched (n>=2,time,5) arrays required")
    if not np.all(np.isfinite(r)) or not np.all(np.isfinite(c)):
        raise ValueError("Nonfinite observable")
    if np.any(r[..., 2:] < -1e-12) or np.any(r[..., 2:] > 1+1e-12):
        raise ValueError("Reference structural fractions outside [0,1]")
    if np.any(c[..., 2:] < -1e-12) or np.any(c[..., 2:] > 1+1e-12):
        raise ValueError("Comparison structural fractions outside [0,1]")
    rm, cm = r.mean(axis=0), c.mean(axis=0)
    span = np.ptp(rm[:, :2], axis=0)
    if np.any(span <= 1e-12):
        raise ValueError("Reference transport range unresolved; extend horizon, do not invent denominator")
    dyn_components = np.sqrt(np.mean((cm[:, :2]-rm[:, :2])**2, axis=0))/span
    struct_components = np.mean(np.abs(cm[:, 2:]-rm[:, 2:]), axis=0)
    result = {"DynErr": float(dyn_components.mean()), "StructErr": float(struct_components.mean()),
        "ClockErr": None, "ClockSignedRelativeError": None,
        "dynamic_components": dict(zip(DYNAMIC_KEYS, map(float, dyn_components))),
        "structure_components": dict(zip(STRUCTURE_KEYS, map(float, struct_components))),
        "reference_dynamic_ranges": dict(zip(DYNAMIC_KEYS, map(float, span)))}
    if reference_fpt is not None and comparison_fpt is not None:
        rt, ct = np.asarray(reference_fpt, float), np.asarray(comparison_fpt, float)
        if rt.shape != (len(r),) or ct.shape != rt.shape:
            raise ValueError("One FPT per complete seed required")
        if not np.all(np.isfinite(rt)) or not np.all(np.isfinite(ct)) or min(rt.min(),ct.min()) <= 0:
            raise ValueError("Positive uncensored FPT values required; no dropping censored seeds")
        signed = float(ct.mean()/rt.mean()-1)
        result.update(ClockErr=abs(signed), ClockSignedRelativeError=signed)
    return result


def bootstrap_errors(reference, comparison, reference_fpt, comparison_fpt,
                     *, replicates=5000, seed=20260905):
    if replicates < 100:
        raise ValueError("At least 100 bootstrap replicates required")
    point = ensemble_errors(reference, comparison, reference_fpt, comparison_fpt)
    rng = np.random.default_rng(seed)
    n = len(reference)
    values = {key: [] for key in ("ClockErr", "ClockSignedRelativeError", "DynErr", "StructErr")}
    for _ in range(replicates):
        # Resample entire matched initial-condition groups, never time samples.
        ids = rng.integers(0, n, n)
        value = ensemble_errors(reference[ids], comparison[ids],
            None if reference_fpt is None else reference_fpt[ids],
            None if comparison_fpt is None else comparison_fpt[ids])
        for key in values:
            if value[key] is not None:
                values[key].append(value[key])
    rows = {}
    for key, samples in values.items():
        low, high = map(float, np.quantile(samples, [.025, .975])) if samples else (None, None)
        rows[key] = {"estimate": point[key], "ci95_low": low, "ci95_high": high}
    # A nonnegative clock discrepancy has a boundary at zero. Transform the
    # signed-ratio CI instead of reporting a spuriously positive lower bound.
    signed = rows["ClockSignedRelativeError"]
    if signed["ci95_low"] is not None:
        lo, hi = signed["ci95_low"], signed["ci95_high"]
        rows["ClockErr"]["ci95_low"] = 0. if lo <= 0 <= hi else min(abs(lo), abs(hi))
        rows["ClockErr"]["ci95_high"] = max(abs(lo), abs(hi))
    return {"metrics": rows, "components": point, "n_initial_conditions": n,
            "bootstrap_replicates": replicates, "bootstrap_seed": seed,
            "bootstrap_unit": "whole paired initial-condition group",
            "interpretation": "finite-ensemble discrepancy; retains sampling noise; not certified pure bias"}


class PreviousStateRecorder:
    """Right-continuous CTMC samples: capture pre-jump state for grid < jump time."""
    def __init__(self, grid):
        self.grid = np.asarray(grid, float)
        if self.grid.ndim != 1 or len(self.grid) < 2 or self.grid[0] != 0 or np.any(np.diff(self.grid) <= 0):
            raise ValueError("Strictly increasing grid starting at zero required")
        self.samples = []

    def before_jump(self, jump_time, observe):
        index = len(self.samples)
        end = int(np.searchsorted(self.grid, jump_time, side="left"))
        if end > index:
            state = observe()
            self.samples.extend([dict(state) for _ in range(end-index)])

    def finish(self, current_time, observe):
        self.before_jump(np.nextafter(current_time, np.inf), observe)
