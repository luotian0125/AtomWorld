"""Minimal species-correct plotting for FCC Ni-Al-Cr lattices."""

from math import ceil

import matplotlib.pyplot as plt
import numpy as np

from atomworld.backends.fcc.envs.fcc_lattice import AL, CR, NI, VACANCY


class FCCPlotter:
    styles = {
        NI: dict(color="#8f98a3", label="Ni", marker="o", alpha=0.15),
        AL: dict(color="#2b74c7", label="Al", marker="o", alpha=0.85),
        CR: dict(color="#d26a22", label="Cr", marker="^", alpha=0.88),
        VACANCY: dict(color="#202124", label="Vacancy", marker="x", alpha=1.0),
    }

    def _draw(self, ax, lattice, constant=1.0, title=None, show_vacancy=False, show_matrix=False):
        species = ([NI] if show_matrix else []) + [AL, CR] + ([VACANCY] if show_vacancy else [])
        for atom_type in species:
            coords = np.argwhere(lattice == atom_type).astype(np.float64) * float(constant)
            if coords.size == 0:
                continue
            style = self.styles[atom_type]
            size = 7.0 if atom_type != VACANCY else 28.0
            ax.scatter(
                coords[:, 0], coords[:, 1], coords[:, 2], s=size,
                c=style["color"], marker=style["marker"], alpha=style["alpha"],
                label=style["label"], depthshade=False,
            )
        extents = (np.asarray(lattice.shape, dtype=float) - 1.0) * float(constant)
        ax.set_xlim(0.0, extents[0])
        ax.set_ylim(0.0, extents[1])
        ax.set_zlim(0.0, extents[2])
        ax.set_box_aspect(np.maximum(extents, 1.0))
        ax.set_xlabel("x / (a/2)")
        ax.set_ylabel("y / (a/2)")
        ax.set_zlabel("z / (a/2)")
        if title:
            ax.set_title(title)
        ax.legend(loc="upper right", frameon=False, fontsize=8)

    def plot_lattice(self, lattice, constant=1.0, title=None, save_path=None, show_vacancy=False, **_):
        fig = plt.figure(figsize=(5.0, 4.4))
        ax = fig.add_subplot(111, projection="3d")
        self._draw(ax, lattice, constant, title, show_vacancy)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax

    def plot_lattice_snapshots(
        self, lattices, constant=1.0, titles=None, save_path=None,
        show_vacancy=False, suptitle=None, **_,
    ):
        count = len(lattices)
        if count == 0:
            raise ValueError("lattices must contain at least one snapshot")
        titles = list(titles) if titles is not None else [None] * count
        if len(titles) != count:
            raise ValueError("titles must match the number of lattices")
        cols = min(count, 4)
        rows = ceil(count / cols)
        fig = plt.figure(figsize=(4.2 * cols, 3.8 * rows))
        axes = []
        for index, (lattice, title) in enumerate(zip(lattices, titles), start=1):
            ax = fig.add_subplot(rows, cols, index, projection="3d")
            axes.append(ax)
            self._draw(ax, lattice, constant, title, show_vacancy)
        if suptitle:
            fig.suptitle(suptitle)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, axes

    def plot_energy_evolution(self, energy_history, time_history, save_path=None):
        fig, ax = plt.subplots(figsize=(5.2, 3.5))
        ax.plot(time_history, energy_history, linewidth=1.5)
        ax.set_xlabel("KMC physical time / s")
        ax.set_ylabel("Hamiltonian / eV")
        ax.grid(alpha=0.2)
        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig, ax
