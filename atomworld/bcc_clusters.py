"""Exact incremental periodic 1NN Cu components, retaining Cu identities."""
from collections import Counter
import numpy as np


class CuClusters:
    def __init__(self, positions, shape, offsets):
        self.shape = np.asarray(shape)
        self.offsets = np.asarray(offsets)
        self.positions = [tuple(map(int, p)) for p in positions]
        self.lookup = {p: i for i, p in enumerate(self.positions)}
        self.adj = [set(self.lookup[p] for p in self.neighbors(pos) if p in self.lookup)
                    for pos in self.positions]
        self.root = {}; self.members = {}; self.sizes = Counter()
        remaining = set(range(len(self.positions)))
        while remaining:
            part = self._component(next(iter(remaining)), remaining)
            self._add(part)
            remaining -= part
        self.pairs = sum(map(len, self.adj)) // 2

    def neighbors(self, pos):
        return [tuple(map(int, p)) for p in (np.asarray(pos) + self.offsets) % self.shape]

    def _component(self, start, allowed):
        found = {start}; pending = [start]
        while pending:
            node = pending.pop()
            for other in self.adj[node]:
                if other in allowed and other not in found:
                    found.add(other); pending.append(other)
        return found

    def _add(self, part):
        root = min(part); self.members[root] = part; self.sizes[len(part)] += 1
        for node in part:
            self.root[node] = root

    def _remove(self, root):
        part = self.members.pop(root); self.sizes[len(part)] -= 1
        if not self.sizes[len(part)]:
            del self.sizes[len(part)]
        return part

    def move(self, old, new):
        old, new = tuple(map(int, old)), tuple(map(int, new))
        node = self.lookup.pop(old)
        affected = self._remove(self.root[node])
        self.pairs -= len(self.adj[node])
        for other in self.adj[node]:
            self.adj[other].remove(node)
        self.adj[node].clear()
        self.lookup[new] = node; self.positions[node] = new
        # Split the old component after deletion, including isolated moved Cu.
        remaining = set(affected)
        while remaining:
            part = self._component(next(iter(remaining)), remaining)
            self._add(part); remaining -= part
        connected = {self.lookup[p] for p in self.neighbors(new) if p in self.lookup}
        roots = {self.root[node]} | {self.root[n] for n in connected}
        merged = set()
        for root in roots:
            merged.update(self._remove(root))
        for other in connected:
            self.adj[node].add(other); self.adj[other].add(node)
        self.pairs += len(connected)
        self._add(merged)

    def metrics(self):
        n = len(self.positions)
        clustered = sum(size * count for size, count in self.sizes.items() if size >= 2)
        stable = sum(count for size, count in self.sizes.items() if size >= 4)
        largest = max(self.sizes, default=0)
        return clustered / n if n else 0., largest / n if n else 0., stable, largest

    def vacancy_contacts(self, vacancies, minimum):
        result = []
        for pos in vacancies:
            hit = False
            for neighbor in self.neighbors(pos):
                node = self.lookup.get(neighbor)
                if node is not None and len(self.members[self.root[node]]) >= minimum:
                    hit = True; break
            result.append(hit)
        return np.asarray(result, dtype=bool)
