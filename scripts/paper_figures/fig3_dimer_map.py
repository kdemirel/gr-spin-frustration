#!/usr/bin/env python3
"""Figure 3 of the PRL manuscript: correlation map in the frame of a dimer (single column).

<u_i . u_k> for a spin i of a mutual-nearest-neighbour dimer and the surrounding spins k, as a
function of the position of k in the dimer frame: z along the axis from i to its partner j
(horizontal, [-2.5, 2.5] r_nn) and rho transverse (vertical, mirrored to [-2.5, 2.5]).  The dimer
is drawn (i at the origin, j at the mean partner distance x_partner_mean), cells with fewer than
min_count pairs are light grey, colormap RdBu_r with symmetric limits +-0.6.

Data: <root>/M/analysis/dimer_map.csv (z_lo, z_hi, z_center, rho_lo, rho_hi, rho_center, n, C,
sem_C, P, sem_P) and summary.json ("dimer_map": min_count, x_partner_mean).  ``--with-P`` adds the
P map as a lower panel (b).  Usage: python3 fig3_dimer_map.py [--results-root R] [--out DIR]
[--entry-map M=...] [--with-P]
"""

from __future__ import annotations

import _style as st
import matplotlib.pyplot as plt
import numpy as np
from _style import BLACK, FigureDataError

ENTRY = "M"
VMAX = 0.6
MASK_COLOR = "#E4E4E4"
MAP_WIDTH = 2.35  # inches of the square map axes


def load_map(root: st.Root, entry: str, min_count: int | None = None):
    df = root.csv(entry, "dimer_map.csv")
    for col in ("z_lo", "z_hi", "rho_lo", "rho_hi", "n", "C", "P"):
        if col not in df.columns:
            raise FigureDataError(f"{entry}: dimer_map.csv has no column {col!r}")
    info = root.summary(entry).get("dimer_map") or {}
    if min_count is None:
        min_count = int(info.get("min_count", 20))
    x_partner = float(info.get("x_partner_mean") or 0.84)
    z_edges = np.unique(np.concatenate([df["z_lo"].to_numpy(float), df["z_hi"].to_numpy(float)]))
    rho_edges = np.unique(np.concatenate([df["rho_lo"].to_numpy(float), df["rho_hi"].to_numpy(float)]))
    nz, nr = len(z_edges) - 1, len(rho_edges) - 1
    iz = np.searchsorted(z_edges, df["z_lo"].to_numpy(float) + 1e-9) - 1
    ir = np.searchsorted(rho_edges, df["rho_lo"].to_numpy(float) + 1e-9) - 1
    grids = {}
    for obs in ("C", "P"):
        g = np.full((nr, nz), np.nan)
        g[ir, iz] = df[obs].to_numpy(float)
        grids[obs] = g
    counts = np.zeros((nr, nz))
    counts[ir, iz] = df["n"].to_numpy(float)
    for obs in grids:
        grids[obs][counts < min_count] = np.nan
    return z_edges, rho_edges, grids, counts, min_count, x_partner, info


def draw_map(ax, z_edges, rho_edges, grid, x_partner, label):
    # mirror in rho: rows from -rho_max to +rho_max
    full = np.vstack([grid[::-1], grid])
    r_edges = np.concatenate([-rho_edges[::-1][:-1], rho_edges])
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad(MASK_COLOR)
    mesh = ax.pcolormesh(
        z_edges, r_edges, np.ma.masked_invalid(full), cmap=cmap, vmin=-VMAX, vmax=VMAX, rasterized=True, shading="flat"
    )
    # the dimer: i at the origin, j at the mean partner distance on the positive axis
    ax.plot([0.0, x_partner], [0.0, 0.0], color=BLACK, lw=1.2, solid_capstyle="round", zorder=4)
    ax.plot([0.0], [0.0], marker="o", ms=4.5, color=BLACK, zorder=5)
    ax.plot([x_partner], [0.0], marker="o", ms=4.5, mfc="white", mec=BLACK, mew=0.9, zorder=5)
    ax.annotate("$i$", (0.0, 0.0), xytext=(-2, -5), textcoords="offset points", ha="right", va="top", fontsize=st.SMALL)
    ax.annotate("$j$", (x_partner, 0.0), xytext=(2, -5), textcoords="offset points", ha="left", va="top", fontsize=st.SMALL)
    ax.set_xlim(z_edges[0], z_edges[-1])
    ax.set_ylim(r_edges[0], r_edges[-1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-2, -1, 0, 1, 2])
    ax.set_yticks([-2, -1, 0, 1, 2])
    ax.tick_params(top=False, right=False)
    ax.set_xlabel(r"$z/\bar{r}_{\mathrm{nn}}$ (along the dimer axis)")
    ax.set_ylabel(r"$\varrho/\bar{r}_{\mathrm{nn}}$")
    ax.text(0.98, 0.97, label, transform=ax.transAxes, ha="right", va="top", fontsize=st.SMALL, zorder=6)
    return mesh


def describe(z_edges, rho_edges, grid, counts, min_count, x_partner, info, root):
    zc = 0.5 * (z_edges[:-1] + z_edges[1:])
    rc = 0.5 * (rho_edges[:-1] + rho_edges[1:])
    Z, R = np.meshgrid(zc, rc)
    dist = np.hypot(Z, R)
    cos = np.where(dist > 0, Z / np.maximum(dist, 1e-12), 0.0)

    def cell_range(mask):
        v = grid[mask & np.isfinite(grid)]
        return f"[{v.min():+.3f}, {v.max():+.3f}] ({v.size} cells)" if v.size else "no cells"

    beside_i = (np.abs(cos) < 0.25) & (dist >= 0.6) & (dist < 1.5)
    beside_j = (cos >= 0.5) & (cos < 0.8) & (dist >= 0.6) & (dist < 1.5)
    on_axis = (np.abs(cos) > 0.8) & (dist >= 0.75) & (dist < 2.5)
    behind = (cos < -0.8) & (dist < 0.75)
    st.report(
        "Fig. 3 dimer map " + ENTRY + root.label_suffix(ENTRY),
        {
            "cells": f"{grid.shape[1]} (z) x {grid.shape[0]} (rho), masked (n < {min_count}): {int(np.sum(counts < min_count))}",
            "source / n_real / n_pairs": f"{info.get('source')} / {info.get('n_real')} / {info.get('n_pairs')}",
            "x_partner_mean": f"{x_partner:.3f}",
            "C beside i (|cos|<0.25, 0.6<r<1.5)": cell_range(beside_i),
            "C beside j (0.5<cos<0.8, 0.6<r<1.5)": cell_range(beside_j),
            "C on axis (|cos|>0.8, r>0.75)": cell_range(on_axis),
            "C behind i on axis (cos<-0.8, r<0.75)": cell_range(behind),
            "C extremes": f"{np.nanmin(grid):+.3f} / {np.nanmax(grid):+.3f}" if np.isfinite(grid).any() else "all cells masked",
        },
    )


def build(root: st.Root, args):
    z_edges, rho_edges, grids, counts, min_count, x_partner, info = load_map(root, ENTRY, args.min_count)
    if not np.isfinite(grids["C"]).any():
        print(f"fig3_dimer_map: warning: every cell has fewer than {min_count} pairs; the map is empty")
    describe(z_edges, rho_edges, grids["C"], counts, min_count, x_partner, info, root)
    panels = [("C", r"$\langle\hat{\mathbf{u}}_i\cdot\hat{\mathbf{u}}_k\rangle$")]
    if args.with_P:
        panels.append(("P", r"$\langle(\hat{\mathbf{u}}_i\cdot\hat{\mathbf{r}})(\hat{\mathbf{u}}_k\cdot\hat{\mathbf{r}})\rangle$"))
    n = len(panels)
    height = 0.55 + n * (MAP_WIDTH + 0.12)
    fig, axes = plt.subplots(n, 1, figsize=(st.SINGLE, height), constrained_layout=True, squeeze=False)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, wspace=0.0, hspace=0.02)
    for k, (ax, (obs, cblabel)) in enumerate(zip(axes[:, 0], panels, strict=False)):
        mesh = draw_map(ax, z_edges, rho_edges, grids[obs], x_partner, "")
        cb = fig.colorbar(mesh, ax=ax, fraction=0.05, pad=0.03, aspect=22)
        cb.set_label(cblabel)
        cb.ax.tick_params(labelsize=st.SMALL)
        cb.set_ticks([-0.6, -0.3, 0.0, 0.3, 0.6])
        if n > 1:
            st.panel_label(ax, f"({'ab'[k]})")
        if k < n - 1:
            ax.set_xlabel("")
    return fig


def main():
    st.setup_rc()
    parser = st.make_parser(__doc__)
    parser.add_argument("--with-P", action="store_true", help="add the P map as panel (b)")
    parser.add_argument(
        "--min-count", type=int, default=None, help="mask cells with fewer pairs (default: min_count of summary.json, 20)"
    )
    args = parser.parse_args()
    try:
        root = st.Root.from_args(args)
        fig = build(root, args)
    except FigureDataError as exc:
        raise SystemExit(f"fig3_dimer_map: cannot draw the figure\n{exc}") from None
    st.finish(fig, args.out, "fig3_dimer_map", dpi=args.dpi)


if __name__ == "__main__":
    main()
