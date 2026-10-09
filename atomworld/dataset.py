"""Policy-independent full-event data and exhaustive small-system graphs."""
from copy import deepcopy
import json
import numpy as np
from .poisson import FiniteRateGraph


def full_event_row(adapter, target, memory):
    saved = adapter.snapshot()
    rates = adapter.rates()
    events = np.flatnonzero(rates > 0)
    x = target.features(adapter, memory)
    successors, boundary = [], []
    try:
        for event in events:
            adapter.restore(saved)
            adapter.apply(int(event))
            next_memory = target.advance(adapter, memory)
            successors.append(target.features(adapter, next_memory))
            boundary.append(next_memory.hit)
    finally:
        adapter.restore(saved)
    return {"x": x, "absorbing": memory.hit, "rates": rates[events], "events": events,
            "successor_x": np.asarray(successors).reshape(len(events), len(x)),
            "successor_absorbing": np.asarray(boundary, dtype=bool)}


def enumerate_graph(adapter, target, max_states=5000):
    """BFS over every legal event. Truncation never becomes an absorbing boundary."""
    original = adapter.snapshot()
    initial_memory = target.initialize(adapter)
    states = [(original, initial_memory)]
    keys = {target.key(adapter, initial_memory): 0}
    rates, successors, boundaries, features = [], [], [], []
    try:
        index = 0
        while index < len(states):
            snapshot, memory = states[index]
            adapter.restore(snapshot)
            features.append(target.features(adapter, memory))
            boundaries.append(memory.hit)
            row_rates, row_successors = [], []
            if not memory.hit:
                catalogue = adapter.rates()
                for event in np.flatnonzero(catalogue > 0):
                    adapter.restore(snapshot)
                    adapter.apply(int(event))
                    next_memory = target.advance(adapter, memory)
                    key = target.key(adapter, next_memory)
                    if key not in keys:
                        if len(states) >= max_states:
                            raise RuntimeError("state-space limit exceeded; partial graph is NOT a calibrated MFPT")
                        keys[key] = len(states)
                        states.append((adapter.snapshot(), deepcopy(next_memory)))
                    row_successors.append(keys[key])
                    row_rates.append(catalogue[event])
            rates.append(np.asarray(row_rates)); successors.append(np.asarray(row_successors, dtype=np.int64))
            index += 1
    finally:
        adapter.restore(original)
    return FiniteRateGraph(rates, successors, np.asarray(boundaries), np.asarray(features))


def collect_rows(adapter, target, *, steps=100, seed=1, reference_samples=0, reference_max_events=100000):
    """Visit with the passive embedded chain; evaluate ALL outgoing events."""
    if steps <= 0:
        raise ValueError("steps must be positive")
    rng = np.random.default_rng(seed)
    initial = adapter.snapshot()
    memory = target.initialize(adapter)
    rows = []
    for _ in range(steps):
        row = full_event_row(adapter, target, memory)
        if reference_samples:
            # Each reference rollout has an independent stream; draws cannot
            # perturb the embedded-chain visitation stream above.
            row.update(passive_reference(adapter, target, memory, reference_samples,
                                         seed + 104729 * (len(rows) + 1), reference_max_events))
        rows.append(row)
        if memory.hit:
            adapter.restore(initial)
            memory = target.initialize(adapter)
            continue
        rates = adapter.rates()
        if rates.sum() <= 0:
            raise ValueError("nonabsorbing dead state has infinite MFPT")
        event = int(rng.choice(len(rates), p=rates / rates.sum()))
        adapter.apply(event)
        memory = target.advance(adapter, memory)
    return rows


def passive_reference(adapter, target, memory, samples, seed, max_events):
    """Complete first-passage replicates; any censoring invalidates the label."""
    if samples < 2 or max_events <= 0:
        raise ValueError("MFPT references require at least two replicates and a positive budget")
    if memory.hit:
        return {"reference_mean_s": 0., "reference_se_s": 0., "reference_censored": 0}
    saved = adapter.snapshot()
    rng = np.random.default_rng(seed)
    durations, censored = [], 0
    try:
        for _ in range(samples):
            adapter.restore(saved)
            progress = deepcopy(memory)
            elapsed = 0.
            for _ in range(max_events):
                rates = adapter.rates()
                gamma = rates.sum()
                if gamma <= 0:
                    raise ValueError("passive reference encounters a nonabsorbing dead state")
                elapsed += float(rng.exponential(1. / gamma))
                event = int(rng.choice(len(rates), p=rates / gamma))
                adapter.apply(event)
                progress = target.advance(adapter, progress)
                if progress.hit:
                    durations.append(elapsed)
                    break
            else:
                censored += 1
    finally:
        adapter.restore(saved)
    # A mean over only successful paths would be downward biased.
    return {"reference_mean_s": float(np.mean(durations)) if not censored else float("nan"),
            "reference_se_s": float(np.std(durations, ddof=1) / np.sqrt(samples)) if not censored else float("nan"),
            "reference_censored": censored}


def save_rows(path, rows, metadata):
    """Non-pickle NPZ; trajectory/seed splits belong to separate files."""
    max_events = max(len(row["rates"]) for row in rows)
    feature_dim = len(rows[0]["x"])
    rates = np.zeros((len(rows), max_events))
    successor_x = np.zeros((len(rows), max_events, feature_dim))
    successor_absorbing = np.ones((len(rows), max_events), dtype=bool)
    for i, row in enumerate(rows):
        count = len(row["rates"])
        rates[i, :count] = row["rates"]
        successor_x[i, :count] = row["successor_x"]
        successor_absorbing[i, :count] = row["successor_absorbing"]
    references = {}
    if "reference_mean_s" in rows[0]:
        references = {k: np.asarray([row[k] for row in rows])
                      for k in ("reference_mean_s", "reference_se_s", "reference_censored")}
    np.savez_compressed(path, x=np.asarray([row["x"] for row in rows]),
                        absorbing=np.asarray([row["absorbing"] for row in rows]), rates=rates,
                        successor_x=successor_x, successor_absorbing=successor_absorbing,
                        metadata=np.asarray(json.dumps(metadata, sort_keys=True)), **references)
