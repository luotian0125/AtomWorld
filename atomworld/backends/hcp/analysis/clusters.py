"""Order parameters for dissolved-H enrichment and precursor connectivity."""

from __future__ import annotations

from collections import Counter

import numpy as np

from atomworld.backends.hcp.lattices.hcp_interstitial import OCTAHEDRAL, TETRAHEDRAL


def _connected_components(adjacency):
    remaining = set(adjacency)
    components = []
    while remaining:
        root = remaining.pop()
        component = {root}
        frontier = [root]
        while frontier:
            node = frontier.pop()
            unseen = adjacency[node] & remaining
            remaining.difference_update(unseen)
            component.update(unseen)
            frontier.extend(unseen)
        components.append(component)
    return components


def hydrogen_cluster_metrics(lattice, energy_model):
    """Measure occupied pair shells and attractive-shell connected clusters.

    Connectivity is defined only by shells with negative interaction energy.
    It is therefore a dissolved-H precursor order parameter, not a hydride
    phase classifier.
    """
    sites = list(map(int, lattice.hydrogen_sites))
    adjacency = {hydrogen_index: set() for hydrogen_index in range(len(sites))}
    pair_counts = Counter({shell.label: 0 for shell in energy_model.pair_shells})
    pair_energy_eV = 0.0
    for first_index, first_site in enumerate(sites):
        first_type = lattice.site_type_name(first_site)
        for second_index in range(first_index + 1, len(sites)):
            second_site = sites[second_index]
            second_type = lattice.site_type_name(second_site)
            distance = lattice.minimum_image_distance_angstrom(first_site, second_site)
            attractive = False
            for shell in energy_model.pair_shells:
                if shell.matches(first_type, second_type, distance):
                    pair_counts[shell.label] += 1
                    pair_energy_eV += shell.interaction_energy_eV
                    attractive = attractive or shell.interaction_energy_eV < 0.0
            if attractive:
                adjacency[first_index].add(second_index)
                adjacency[second_index].add(first_index)

    components = _connected_components(adjacency)
    component_sizes = sorted((len(component) for component in components), reverse=True)
    occupied_types = lattice.site_types[lattice.hydrogen_sites]
    n_hydrogen = max(len(sites), 1)
    return {
        "n_hydrogen": len(sites),
        "T_fraction": float(
            np.count_nonzero(occupied_types == TETRAHEDRAL) / n_hydrogen
        ),
        "O_fraction": float(
            np.count_nonzero(occupied_types == OCTAHEDRAL) / n_hydrogen
        ),
        "pair_counts": dict(pair_counts),
        "pair_energy_eV": float(pair_energy_eV),
        "cluster_count": len(components),
        "largest_attractive_cluster": component_sizes[0] if component_sizes else 0,
        "largest_attractive_cluster_fraction": float(
            component_sizes[0] / n_hydrogen if component_sizes else 0.0
        ),
        "isolated_hydrogen_fraction": float(
            sum(size == 1 for size in component_sizes) / n_hydrogen
        ),
    }
