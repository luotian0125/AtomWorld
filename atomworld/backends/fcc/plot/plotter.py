from typing import Iterable, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import proj3d
import numpy as np


class Plotter:
    def __init__(self):
        self.atom_styles = {
            1: {
                "color": "#b012b0",
                "label": "Cu",
                "marker": "o",
                "alpha": 0.9,
            },
            2: {
                "color": "#2b2b2b",
                "label": "Vacancy",
                "marker": "x",
                "alpha": 0.95,
            },
        }
        self.cluster_cmap = LinearSegmentedColormap.from_list(
            "cu_cluster",
            ["#fff1d6", "#f5b36f", "#e7774a", "#c33a2b", "#8f1416"],
            N=256,
        )
        self.default_view = {"elev": 22, "azim": -58}
        self.inset_view = {"elev": 22, "azim": -58}
        self.paper_figsize = (6.9, 2.55)
        self.box_edges = [
            (0, 1),
            (0, 2),
            (0, 4),
            (1, 3),
            (1, 5),
            (2, 3),
            (2, 6),
            (3, 7),
            (4, 5),
            (4, 6),
            (5, 7),
            (6, 7),
        ]
        self.nn1_offsets = np.array(
            [
                [1, 1, 1],
                [-1, -1, -1],
                [1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [-1, 1, -1],
                [-1, 1, 1],
                [1, -1, -1],
            ],
            dtype=np.int32,
        )

    def _extract_coords(self, lattice: np.ndarray, atom_type: int, constant: float) -> np.ndarray:
        coords = np.argwhere(lattice == atom_type)
        if coords.size == 0:
            return np.empty((0, 3), dtype=np.float32)
        return coords.astype(np.float32) * float(constant)

    def _cube_extents(self, lattice: np.ndarray, constant: float) -> np.ndarray:
        return np.maximum((np.asarray(lattice.shape, dtype=np.float32) - 1.0) * float(constant), 1.0)

    def _auto_point_size(self, num_points: int, is_vacancy: bool = False) -> float:
        if is_vacancy:
            return 24.0
        if num_points <= 0:
            return 3.0
        return float(np.clip(1800.0 / num_points, 0.75, 5.0))

    def _box_vertices(self, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
        x0, y0, z0 = lower.tolist()
        x1, y1, z1 = upper.tolist()
        return np.array(
            [
                [x0, y0, z0],
                [x1, y0, z0],
                [x0, y1, z0],
                [x1, y1, z0],
                [x0, y0, z1],
                [x1, y0, z1],
                [x0, y1, z1],
                [x1, y1, z1],
            ],
            dtype=np.float32,
        )

    def _draw_box(
        self,
        ax: plt.Axes,
        lower: np.ndarray,
        upper: np.ndarray,
        color: str = "#202428",
        linewidth: float = 0.65,
        alpha: float = 0.82,
        linestyle: str = "-",
        zorder: int = 3,
    ) -> None:
        vertices = self._box_vertices(lower, upper)
        for start, end in self.box_edges:
            segment = vertices[[start, end]]
            ax.plot(
                segment[:, 0],
                segment[:, 1],
                segment[:, 2],
                color=color,
                linewidth=linewidth,
                alpha=alpha,
                linestyle=linestyle,
                zorder=zorder,
            )

    def _draw_cube(self, ax: plt.Axes, extents: np.ndarray, **kwargs) -> None:
        self._draw_box(ax, np.zeros(3, dtype=np.float32), extents, **kwargs)

    def _hide_3d_axes(self, ax: plt.Axes) -> None:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel("")
        ax.set_ylabel("")
        ax.set_zlabel("")
        ax.grid(False)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis.pane.set_edgecolor((1.0, 1.0, 1.0, 0.0))
            axis._axinfo["grid"]["linewidth"] = 0.0
            axis.line.set_color((1.0, 1.0, 1.0, 0.0))

    def _style_snapshot_axes(
        self,
        ax: plt.Axes,
        lattice: np.ndarray,
        constant: float,
        title: Optional[str],
        title_loc: str = "center",
        title_fontsize: float = 9.0,
    ) -> None:
        extents = self._cube_extents(lattice, constant)
        margin = max(float(np.max(extents)) * 0.025, 0.4)

        ax.set_xlim(-margin, extents[0] + margin)
        ax.set_ylim(-margin, extents[1] + margin)
        ax.set_zlim(-margin, extents[2] + margin)
        ax.set_box_aspect(tuple(extents.tolist()))
        ax.view_init(**self.default_view)
        ax.set_proj_type("persp")
        ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
        self._hide_3d_axes(ax)

        if title:
            ax.set_title(title, loc=title_loc, pad=0.0, fontsize=title_fontsize, fontweight="bold")

        self._draw_cube(ax, extents, linewidth=0.62, alpha=0.82, zorder=4)

    def _style_zoom_axes(
        self,
        ax: plt.Axes,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        lower, upper = bounds
        side = upper - lower
        margin = max(float(np.max(side)) * 0.05, 0.1)

        ax.set_xlim(lower[0] - margin, upper[0] + margin)
        ax.set_ylim(lower[1] - margin, upper[1] + margin)
        ax.set_zlim(lower[2] - margin, upper[2] + margin)
        ax.set_box_aspect(tuple(np.maximum(side, 1e-6).tolist()))
        ax.view_init(**self.inset_view)
        ax.set_proj_type("persp")
        ax.set_facecolor((1.0, 1.0, 1.0, 0.0))
        self._hide_3d_axes(ax)
        self._draw_box(ax, lower, upper, linewidth=0.55, alpha=0.78, zorder=6)

    def _find_root(self, parent: np.ndarray, idx: int) -> int:
        while parent[idx] != idx:
            parent[idx] = parent[parent[idx]]
            idx = int(parent[idx])
        return int(idx)

    def _union_sets(self, parent: np.ndarray, size: np.ndarray, a: int, b: int) -> None:
        root_a = self._find_root(parent, a)
        root_b = self._find_root(parent, b)
        if root_a == root_b:
            return
        if size[root_a] < size[root_b]:
            root_a, root_b = root_b, root_a
        parent[root_b] = root_a
        size[root_a] += size[root_b]

    def _clustered_cu_points(self, lattice: np.ndarray, constant: float) -> Tuple[np.ndarray, np.ndarray, int]:
        coords = np.argwhere(lattice == 1).astype(np.int32)
        if coords.size == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0,), dtype=np.float32), 1

        dims = np.asarray(lattice.shape, dtype=np.int32)
        pos_to_idx = {tuple(map(int, pos)): idx for idx, pos in enumerate(coords)}
        parent = np.arange(coords.shape[0], dtype=np.int32)
        size = np.ones(coords.shape[0], dtype=np.int32)

        for idx, pos in enumerate(coords):
            for offset in self.nn1_offsets:
                nbr = tuple(map(int, (pos + offset) % dims))
                nbr_idx = pos_to_idx.get(nbr)
                if nbr_idx is not None:
                    self._union_sets(parent, size, idx, int(nbr_idx))

        cluster_values = np.empty(coords.shape[0], dtype=np.float32)
        for idx in range(coords.shape[0]):
            cluster_values[idx] = float(size[self._find_root(parent, idx)])

        return coords.astype(np.float32) * float(constant), cluster_values, int(np.max(cluster_values))

    def _largest_cu_cluster_coords(self, lattice: np.ndarray, constant: float) -> np.ndarray:
        coords = np.argwhere(lattice == 1).astype(np.int32)
        if coords.size == 0:
            return np.empty((0, 3), dtype=np.float32)

        dims = np.asarray(lattice.shape, dtype=np.int32)
        pos_to_idx = {tuple(map(int, pos)): idx for idx, pos in enumerate(coords)}
        parent = np.arange(coords.shape[0], dtype=np.int32)
        size = np.ones(coords.shape[0], dtype=np.int32)

        for idx, pos in enumerate(coords):
            for offset in self.nn1_offsets:
                nbr = tuple(map(int, (pos + offset) % dims))
                nbr_idx = pos_to_idx.get(nbr)
                if nbr_idx is not None:
                    self._union_sets(parent, size, idx, int(nbr_idx))

        roots = np.array([self._find_root(parent, idx) for idx in range(coords.shape[0])], dtype=np.int32)
        unique_roots, counts = np.unique(roots, return_counts=True)
        largest_root = int(unique_roots[int(np.argmax(counts))])
        return coords[roots == largest_root].astype(np.float32) * float(constant)

    def _select_zoom_bounds(
        self,
        lattice: np.ndarray,
        constant: float,
        zoom_fraction: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        extents = self._cube_extents(lattice, constant)
        max_extent = float(np.max(extents))
        min_extent = float(np.min(extents))
        side = min(max(max_extent * float(zoom_fraction), float(constant) * 4.0), min_extent)
        box_size = np.full(3, max(side, 1.0), dtype=np.float32)
        return self._select_cluster_zoom_bounds(lattice, constant, box_size)

    def _select_shared_zoom_bounds(
        self,
        lattices: Sequence[np.ndarray],
        constant: float,
        zoom_fraction: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        reference_lattice = lattices[-1]
        extents = self._cube_extents(reference_lattice, constant)
        max_extent = float(np.max(extents))
        min_extent = float(np.min(extents))
        side = min(max(max_extent * float(zoom_fraction), float(constant) * 4.0), min_extent)
        box_size = np.full(3, max(side, 1.0), dtype=np.float32)
        return self._select_cluster_zoom_bounds(reference_lattice, constant, box_size)

    def _select_cluster_zoom_bounds(
        self,
        lattice: np.ndarray,
        constant: float,
        box_size: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        extents = self._cube_extents(lattice, constant)
        box_size = np.minimum(np.asarray(box_size, dtype=np.float32), extents)
        coords, cluster_values, _ = self._clustered_cu_points(lattice, constant)
        if coords.size == 0:
            lower = np.maximum((extents - box_size) * 0.5, 0.0)
            return lower.astype(np.float32), (lower + box_size).astype(np.float32)

        largest_cluster = self._largest_cu_cluster_coords(lattice, constant)
        lower_limit = np.zeros(3, dtype=np.float32)
        upper_limit = np.maximum(extents - box_size, 0.0)
        if len(largest_cluster) >= 4:
            center = np.mean(largest_cluster, axis=0)
            lower = np.clip(center.astype(np.float32) - box_size * 0.5, lower_limit, upper_limit)
            upper = np.minimum(lower + box_size, extents)
            lower = np.maximum(upper - box_size, 0.0)
            return lower.astype(np.float32), upper.astype(np.float32)

        max_candidates = min(512, len(coords))
        ranked_indices = np.argsort(cluster_values)[-max_candidates:]
        candidate_centers = coords[ranked_indices]

        best_score = -float("inf")
        best_lower = np.clip(np.mean(candidate_centers, axis=0) - box_size * 0.5, lower_limit, upper_limit)
        for center in candidate_centers:
            lower = np.clip(center - box_size * 0.5, lower_limit, upper_limit)
            upper = lower + box_size
            mask = np.all((coords >= lower) & (coords <= upper), axis=1)
            if not np.any(mask):
                continue
            local_values = cluster_values[mask].astype(np.float64)
            score = float(np.sum(local_values * local_values) + 0.2 * local_values.size * local_values.size)
            if score > best_score:
                best_score = score
                best_lower = lower

        for _ in range(2):
            best_upper = best_lower + box_size
            mask = np.all((coords >= best_lower) & (coords <= best_upper), axis=1)
            if not np.any(mask):
                break
            local_coords = coords[mask].astype(np.float64)
            local_values = np.maximum(cluster_values[mask].astype(np.float64), 1.0)
            center = np.average(local_coords, axis=0, weights=local_values * local_values)
            best_lower = np.clip(center.astype(np.float32) - box_size * 0.5, lower_limit, upper_limit)

        best_upper = np.minimum(best_lower + box_size, extents)
        best_lower = np.maximum(best_upper - box_size, 0.0)
        return best_lower.astype(np.float32), best_upper.astype(np.float32)

    def _plot_cu_clusters(
        self,
        ax: plt.Axes,
        lattice: np.ndarray,
        constant: float,
        norm: Normalize,
        point_size: Optional[float] = None,
        alpha: float = 0.78,
        zoom_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> int:
        coords, cluster_values, cmax = self._clustered_cu_points(lattice, constant)
        if coords.size == 0:
            return cmax

        if zoom_bounds is not None:
            lower, upper = zoom_bounds
            mask = np.all((coords >= lower) & (coords <= upper), axis=1)
            coords = coords[mask]
            cluster_values = cluster_values[mask]
            if coords.size == 0:
                return cmax

        size = point_size if point_size is not None else self._auto_point_size(len(coords))
        ax.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=size,
            c=cluster_values,
            cmap=self.cluster_cmap,
            norm=norm,
            marker="o",
            alpha=alpha,
            linewidths=0.0,
            edgecolors="none",
            depthshade=False,
            rasterized=True,
        )
        return cmax

    def _plot_species(
        self,
        ax: plt.Axes,
        lattice: np.ndarray,
        constant: float,
        atom_types: Iterable[int],
        cu_point_size: Optional[float] = None,
        vacancy_point_size: Optional[float] = None,
        cluster_norm: Optional[Normalize] = None,
        zoom_bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    ) -> None:
        for atom_type in atom_types:
            if atom_type == 1 and cluster_norm is not None:
                self._plot_cu_clusters(
                    ax,
                    lattice,
                    constant,
                    norm=cluster_norm,
                    point_size=cu_point_size,
                    zoom_bounds=zoom_bounds,
                )
                continue

            coords = self._extract_coords(lattice, atom_type, constant)
            if coords.size == 0:
                continue

            if zoom_bounds is not None:
                lower, upper = zoom_bounds
                mask = np.all((coords >= lower) & (coords <= upper), axis=1)
                coords = coords[mask]
                if coords.size == 0:
                    continue

            style = self.atom_styles[atom_type]
            is_vacancy = atom_type == 2
            size = vacancy_point_size if is_vacancy else cu_point_size
            if size is None:
                size = self._auto_point_size(len(coords), is_vacancy=is_vacancy)

            ax.scatter(
                coords[:, 0],
                coords[:, 1],
                coords[:, 2],
                s=size,
                c=style["color"],
                marker=style["marker"],
                alpha=style["alpha"],
                linewidths=0.0 if not is_vacancy else 0.7,
                depthshade=False,
                label=style["label"],
            )

    def _project_box_to_figure(
        self,
        fig: plt.Figure,
        ax: plt.Axes,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> np.ndarray:
        vertices = self._box_vertices(bounds[0], bounds[1])
        projected = []
        projection = ax.get_proj()
        for point in vertices:
            x2, y2, _ = proj3d.proj_transform(float(point[0]), float(point[1]), float(point[2]), projection)
            display_xy = ax.transData.transform((x2, y2))
            projected.append(fig.transFigure.inverted().transform(display_xy))
        return np.asarray(projected, dtype=np.float32)

    def _projected_side_pair(self, points: np.ndarray, side: str) -> Tuple[np.ndarray, np.ndarray]:
        if side == "right":
            candidates = points[np.argsort(points[:, 0])[-4:]]
        elif side == "left":
            candidates = points[np.argsort(points[:, 0])[:4]]
        else:
            raise ValueError(f"Unsupported projected side: {side}")
        return candidates[np.argmax(candidates[:, 1])], candidates[np.argmin(candidates[:, 1])]

    def _add_zoom_inset(
        self,
        fig: plt.Figure,
        parent_ax: plt.Axes,
        lattice: np.ndarray,
        constant: float,
        norm: Normalize,
        bounds: Tuple[np.ndarray, np.ndarray],
        show_vacancy: bool,
        inset_size: float,
        cu_point_size: Optional[float],
        vacancy_point_size: Optional[float],
    ) -> plt.Axes:
        bbox = parent_ax.get_position()
        fig_w, fig_h = fig.get_size_inches()
        inset_w = min(float(inset_size) / fig_w, bbox.width * 0.36)
        inset_h = min(float(inset_size) / fig_h, bbox.height * 0.36)
        left = bbox.x1 - inset_w * 0.02
        bottom = bbox.y0 + bbox.height * 0.04

        inset_ax = fig.add_axes([left, bottom, inset_w, inset_h], projection="3d")
        inset_cu_size = max((cu_point_size or 2.0) * 2.35, 3.0)
        atom_types = [1, 2] if show_vacancy else [1]
        self._plot_species(
            inset_ax,
            lattice,
            constant,
            atom_types=atom_types,
            cu_point_size=inset_cu_size,
            vacancy_point_size=vacancy_point_size,
            cluster_norm=norm,
            zoom_bounds=bounds,
        )
        self._style_zoom_axes(inset_ax, bounds)
        return inset_ax

    def _add_zoom_connectors(
        self,
        fig: plt.Figure,
        parent_ax: plt.Axes,
        inset_ax: plt.Axes,
        bounds: Tuple[np.ndarray, np.ndarray],
    ) -> None:
        source_points = self._project_box_to_figure(fig, parent_ax, bounds)
        target_points = self._project_box_to_figure(fig, inset_ax, bounds)
        source_top, source_bottom = self._projected_side_pair(source_points, "right")
        target_top, target_bottom = self._projected_side_pair(target_points, "left")

        for source, target in ((source_top, target_top), (source_bottom, target_bottom)):
            connector = Line2D(
                [source[0], target[0]],
                [source[1], target[1]],
                transform=fig.transFigure,
                color="#9aa2a6",
                linewidth=0.38,
                alpha=0.34,
                zorder=2,
                clip_on=False,
            )
            fig.add_artist(connector)

    def _paper_layout(self, ncols: int):
        if ncols == 2:
            return {
                "panels": [[0.045, 0.16, 0.305, 0.64], [0.445, 0.16, 0.305, 0.64]],
                "titles": [(0.030, 0.885), (0.430, 0.885)],
                "colorbar": [0.905, 0.25, 0.012, 0.48],
            }
        return {
            "panels": [[0.070, 0.16, 0.40, 0.64]],
            "titles": [(0.045, 0.885)],
            "colorbar": [0.720, 0.25, 0.012, 0.48],
        }

    def _plot_initial_final_snapshots(
        self,
        lattices: Sequence[np.ndarray],
        constant: float,
        titles: Optional[Sequence[str]],
        save_path: Optional[str],
        show_vacancy: bool,
        suptitle: Optional[str],
        cu_point_size: Optional[float],
        vacancy_point_size: Optional[float],
        figsize: Optional[Tuple[float, float]],
        zoom_fraction: float,
        inset_size: float,
        dpi: int,
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        if len(lattices) >= 2:
            selected_lattices = [lattices[0], lattices[-1]]
            selected_titles = ["Initial", "Final"]
        else:
            selected_lattices = [lattices[0]]
            selected_titles = [titles[0] if titles is not None else "Initial"]

        cmax = 1
        for lattice in selected_lattices:
            _, _, lattice_cmax = self._clustered_cu_points(lattice, constant)
            cmax = max(cmax, lattice_cmax)
        norm = Normalize(vmin=1.0, vmax=float(max(cmax, 2)))

        fig = plt.figure(figsize=figsize or self.paper_figsize, facecolor="white")
        layout = self._paper_layout(len(selected_lattices))
        axes = []
        inset_axes = []
        zoom_bounds = []
        atom_types = [1, 2] if show_vacancy else [1]
        shared_zoom_bounds = self._select_shared_zoom_bounds(selected_lattices, constant, zoom_fraction)

        for idx, (lattice, title) in enumerate(zip(selected_lattices, selected_titles)):
            ax = fig.add_axes(layout["panels"][idx], projection="3d")
            axes.append(ax)
            self._plot_species(
                ax,
                lattice,
                constant,
                atom_types=atom_types,
                cu_point_size=cu_point_size,
                vacancy_point_size=vacancy_point_size,
                cluster_norm=norm,
            )
            self._style_snapshot_axes(ax, lattice, constant, title=None)
            fig.text(
                layout["titles"][idx][0],
                layout["titles"][idx][1],
                f"({chr(ord('a') + idx)}) {title}",
                ha="left",
                va="top",
                fontsize=9.2,
                fontweight="bold",
                color="#202327",
            )

            bounds = shared_zoom_bounds
            zoom_bounds.append(bounds)
            self._draw_box(
                ax,
                bounds[0],
                bounds[1],
                color="#5f6669",
                linewidth=0.52,
                alpha=0.68,
                linestyle=(0, (4, 2)),
                zorder=7,
            )

        cbar_ax = fig.add_axes(layout["colorbar"])
        sm = ScalarMappable(norm=norm, cmap=self.cluster_cmap)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.outline.set_linewidth(0.4)
        cbar.set_ticks([])
        cbar_ax.text(
            1.35,
            1.0,
            r"$C_{\max}$",
            transform=cbar_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.3,
            color="#202327",
        )
        cbar_ax.text(
            1.35,
            0.0,
            r"$C_1$" + "\n(single Cu atom)",
            transform=cbar_ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=6.6,
            color="#202327",
            linespacing=0.95,
        )

        if suptitle:
            fig.suptitle(suptitle, y=0.985, fontsize=8.5, color="#202327")

        fig.canvas.draw()
        for ax, lattice, bounds in zip(axes, selected_lattices, zoom_bounds):
            inset_ax = self._add_zoom_inset(
                fig,
                ax,
                lattice,
                constant,
                norm=norm,
                bounds=bounds,
                show_vacancy=show_vacancy,
                inset_size=inset_size,
                cu_point_size=cu_point_size,
                vacancy_point_size=vacancy_point_size,
            )
            inset_axes.append(inset_ax)
            fig.canvas.draw()
            self._add_zoom_connectors(fig, ax, inset_ax, bounds)

        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02, facecolor="white")

        return fig, axes

    def plot_lattice(
        self,
        lattice: np.ndarray,
        constant: float,
        title: Optional[str] = None,
        save_path: Optional[str] = None,
        show_vacancy: bool = False,
        figsize: Tuple[float, float] = (4.2, 4.2),
        cu_point_size: Optional[float] = None,
        vacancy_point_size: Optional[float] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot a single BCC lattice snapshot in a paper-style 3D view."""
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")

        _, _, cmax = self._clustered_cu_points(lattice, constant)
        norm = Normalize(vmin=1.0, vmax=float(max(cmax, 2)))
        atom_types = [1, 2] if show_vacancy else [1]
        self._plot_species(
            ax,
            lattice,
            constant,
            atom_types=atom_types,
            cu_point_size=cu_point_size,
            vacancy_point_size=vacancy_point_size,
            cluster_norm=norm,
        )
        self._style_snapshot_axes(ax, lattice, constant, title=title)

        if show_vacancy:
            ax.legend(loc="upper right", frameon=False, fontsize=8)

        fig.tight_layout(pad=0.5)
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")

        return fig, ax

    def plot_lattice_snapshots(
        self,
        lattices: Sequence[np.ndarray],
        constant: float,
        titles: Optional[Sequence[str]] = None,
        save_path: Optional[str] = None,
        show_vacancy: bool = False,
        suptitle: Optional[str] = None,
        cu_point_size: Optional[float] = None,
        vacancy_point_size: Optional[float] = None,
        figsize: Optional[Tuple[float, float]] = None,
        paper_style: bool = True,
        zoom_fraction: float = 0.26,
        inset_size: float = 0.72,
        dpi: int = 600,
    ) -> Tuple[plt.Figure, Sequence[plt.Axes]]:
        """Plot lattice snapshots.

        The default paper style renders Initial/Final snapshots with a shared
        Cu-cluster colorbar and a fixed-size zoom inset in each panel.
        """
        if len(lattices) == 0:
            raise ValueError("lattices must contain at least one snapshot")

        if titles is not None and len(titles) != len(lattices):
            raise ValueError("titles must have the same length as lattices")

        if paper_style:
            return self._plot_initial_final_snapshots(
                lattices=lattices,
                constant=constant,
                titles=titles,
                save_path=save_path,
                show_vacancy=show_vacancy,
                suptitle=suptitle,
                cu_point_size=cu_point_size,
                vacancy_point_size=vacancy_point_size,
                figsize=figsize,
                zoom_fraction=zoom_fraction,
                inset_size=inset_size,
                dpi=dpi,
            )

        ncols = len(lattices)
        fig = plt.figure(figsize=figsize or (3.9 * ncols, 4.2))
        axes = []
        atom_types = [1, 2] if show_vacancy else [1]
        cmax = 1
        for lattice in lattices:
            _, _, lattice_cmax = self._clustered_cu_points(lattice, constant)
            cmax = max(cmax, lattice_cmax)
        norm = Normalize(vmin=1.0, vmax=float(max(cmax, 2)))

        for idx, lattice in enumerate(lattices):
            ax = fig.add_subplot(1, ncols, idx + 1, projection="3d")
            axes.append(ax)
            self._plot_species(
                ax,
                lattice,
                constant,
                atom_types=atom_types,
                cu_point_size=cu_point_size,
                vacancy_point_size=vacancy_point_size,
                cluster_norm=norm,
            )
            title = titles[idx] if titles is not None else None
            self._style_snapshot_axes(ax, lattice, constant, title=title)

        if suptitle:
            fig.suptitle(suptitle, y=0.97, fontsize=12)
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93), pad=0.6, w_pad=0.3)
        else:
            fig.tight_layout(pad=0.6, w_pad=0.3)
        if save_path:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig, axes

    def plot_energy_evolution(
        self,
        energies: Sequence[float],
        times: Sequence[float],
        save_path: Optional[str] = None,
    ) -> Tuple[plt.Figure, plt.Axes]:
        """Plot system energy evolution over time."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, energies, "-o")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("System Energy (eV)")
        ax.set_title("Energy Evolution")

        fig.tight_layout()
        if save_path:
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        return fig, ax
