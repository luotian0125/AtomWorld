import numpy as np
from atomworld.bcc_clusters import CuClusters


def brute_components(positions, shape, offsets):
    c = CuClusters(positions, shape, offsets)
    return sorted(sorted(part) for part in c.members.values()), c.metrics(), c.pairs


def test_incremental_components_match_fresh_graph_under_splits_merges_and_wraps():
    shape = np.array([8, 8, 8])
    offsets = np.array([[i, j, k] for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    coords = np.array([(i, j, k) for i in range(8) for j in range(8) for k in range(8)
                       if i % 2 == j % 2 == k % 2])
    rng = np.random.default_rng(517)
    positions = coords[rng.choice(len(coords), 40, replace=False)]
    c = CuClusters(positions, shape, offsets)
    for _ in range(250):
        node = int(rng.integers(len(positions)))
        old = tuple(positions[node]); new = (positions[node] + offsets[rng.integers(8)]) % shape
        if tuple(new) in c.lookup: continue
        c.move(old, new); positions[node] = new
        expected, metrics, pairs = brute_components(positions, shape, offsets)
        assert sorted(sorted(part) for part in c.members.values()) == expected
        assert c.metrics() == metrics
        assert c.pairs == pairs
        fresh = CuClusters(positions, shape, offsets)
        for minimum in (2, 4):
            assert np.array_equal(c.vacancy_contacts(coords[:15], minimum),
                                  fresh.vacancy_contacts(coords[:15], minimum))
