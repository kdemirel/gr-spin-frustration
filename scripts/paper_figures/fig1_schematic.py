#!/usr/bin/env python3
"""Figure 1 of the PRL manuscript: schematic of the gravitational spin-spin interaction.

Pure matplotlib, no data.  Three panels in one row at PRL column width:

  (a) two-body minimum: antiparallel spins along the bond, U = -2 G s_i s_j / r^3;
  (b) the two channels of the tensor: bond-parallel components antiparallel
      (weight 2), transverse components parallel (weight 1);
  (c) a dimer and its surroundings: a spin beside member i (transverse to the
      axis) aligns parallel to S_i, a spin beside member j aligns parallel to S_j;
      no on-axis neighbour is drawn because the measured on-axis correlations are
      weak beyond 0.75 r_nn.

Usage:  python3 fig1_schematic.py [OUTDIR]

OUTDIR defaults to ../../../spin_spin_paper_for_PRL/figures relative to this
script.  Writes fig1_schematic.pdf (vector) and fig1_schematic.png (200 dpi).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Circle, Polygon  # noqa: E402

# ----------------------------------------------------------------------------
# Layout constants (all lengths in inches; the drawing axes uses inch units)
# ----------------------------------------------------------------------------
WIDTH = 3.375  # PRL column width
HEIGHT = 1.85  # the drawing needs ~1.6 in; a taller figure only adds white space
FONTSIZE = 8.0

# panel columns: left edge and width
COLS = {"a": (0.000, 0.890), "b": (0.890, 1.005), "c": (1.895, 1.480)}
# common height of the bond axis in every panel (keeps the drawing centred if HEIGHT changes)
Y_AXIS = 0.62 + (HEIGHT - 1.85) / 2

BOND = 0.55  # bond length r
ARROW = 0.24  # spin-arrow length (tail at the body centre)
RADIUS = 0.04  # body radius

# colours (Okabe-Ito blue/orange + greys)
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREY = "#8A8A8A"  # bonds
LIGHT = "#A6A6A6"  # dotted field-transmitting bonds
BODY = "#4A4A4A"  # bodies
INK = "black"

Z_BOND, Z_COMP, Z_SPIN, Z_BODY, Z_TEXT = 1, 2, 3, 4, 5

DOTTED = (0, (1.0, 1.6))
DASHED = (0, (3.0, 2.0))


def setup_rc() -> None:
    # DejaVu Sans ships with matplotlib (regular/bold/oblique and the ∥, ⊥ glyphs),
    # so the figure is identical on every machine.
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "font.size": FONTSIZE,
            "mathtext.fontset": "dejavusans",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


# ----------------------------------------------------------------------------
# Drawing primitives
# ----------------------------------------------------------------------------
def line(ax, p0, p1, color, lw=0.7, ls="-", z=Z_BOND):
    ax.add_line(
        Line2D(
            [p0[0], p1[0]],
            [p0[1], p1[1]],
            color=color,
            lw=lw,
            ls=ls,
            solid_capstyle="butt",
            dash_capstyle="butt",
            zorder=z,
        )
    )


def arrow(ax, p0, p1, color, lw=1.0, ls="-", head_len=0.075, head_hw=0.032, z=Z_SPIN):
    """Arrow from p0 to p1 with a filled triangular head of exact size (inches)."""
    (x0, y0), (x1, y1) = p0, p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length
    bx, by = x1 - head_len * ux, y1 - head_len * uy  # base of the head
    line(ax, (x0, y0), (bx, by), color, lw=lw, ls=ls, z=z)
    head = Polygon(
        [
            [x1, y1],
            [bx - head_hw * uy, by + head_hw * ux],
            [bx + head_hw * uy, by - head_hw * ux],
        ],
        closed=True,
        facecolor=color,
        edgecolor="none",
        zorder=z,
    )
    ax.add_patch(head)


def body(ax, x, y):
    ax.add_patch(Circle((x, y), RADIUS, facecolor=BODY, edgecolor="none", zorder=Z_BODY))


def text(ax, x, y, s, ha="center", va="center", **kw):
    kw.setdefault("fontsize", FONTSIZE)
    kw.setdefault("color", INK)
    kw.setdefault("linespacing", 1.15)
    return ax.text(x, y, s, ha=ha, va=va, zorder=Z_TEXT, **kw)


def panel_label(ax, x, s):
    text(ax, x, HEIGHT - 0.04, s, ha="left", va="top", fontweight="bold")


# ----------------------------------------------------------------------------
# Panels
# ----------------------------------------------------------------------------
def panel_a(ax, x0, w, y0):
    """Two-body minimum: spins along the bond, antiparallel."""
    xc = x0 + w / 2
    xi, xj = xc - BOND / 2, xc + BOND / 2
    line(ax, (xi, y0), (xj, y0), GREY)
    arrow(ax, (xi, y0), (xi + ARROW, y0), BLUE)  # S_i points towards j
    arrow(ax, (xj, y0), (xj - ARROW, y0), ORANGE)  # S_j points towards i
    body(ax, xi, y0)
    body(ax, xj, y0)
    text(ax, xi + ARROW / 2, y0 + 0.075, r"$\mathbf{S}_i$", va="bottom")
    text(ax, xj - ARROW / 2, y0 + 0.075, r"$\mathbf{S}_j$", va="bottom")
    text(ax, xc, y0 - 0.14, r"$U=-2Gs_is_j/r^3$", va="top")


def panel_b(ax, x0, w, y0):
    """Two channels: longitudinal components antiparallel, transverse parallel."""
    xc = x0 + w / 2
    xi, xj = xc - BOND / 2, xc + BOND / 2
    theta = math.radians(60.0)  # tilt of the spins from the bond
    L = 0.38
    px, py = L * math.cos(theta), L * math.sin(theta)
    comp = dict(lw=0.75, ls=DASHED, head_len=0.06, head_hw=0.026, z=Z_COMP)

    line(ax, (xi, y0), (xj, y0), GREY)
    # components, head to tail: S = S_par (along the bond) + S_perp (transverse)
    arrow(ax, (xi, y0), (xi + px, y0), BLUE, **comp)
    arrow(ax, (xi + px, y0), (xi + px, y0 + py), BLUE, **comp)
    arrow(ax, (xj, y0), (xj - px, y0), ORANGE, **comp)
    arrow(ax, (xj - px, y0), (xj - px, y0 + py), ORANGE, **comp)
    # full spins
    arrow(ax, (xi, y0), (xi + px, y0 + py), BLUE)
    arrow(ax, (xj, y0), (xj - px, y0 + py), ORANGE)
    body(ax, xi, y0)
    body(ax, xj, y0)

    text(ax, xi + 0.07, y0 + 0.24, r"$\mathbf{S}_i$", ha="right")
    text(ax, xj - 0.07, y0 + 0.24, r"$\mathbf{S}_j$", ha="left")
    text(ax, xc, y0 + py + 0.07, "$\\perp$: parallel\nweight 1", va="bottom")
    text(ax, xc, y0 - 0.11, "$\\parallel$: antiparallel\nweight 2", va="top")


def panel_c(ax, x0, w, y0):
    """Dimer (i, j) with two transverse neighbours: k beside i, k' beside j.

    Each neighbour sees the nearer dimer member as a transverse field and aligns
    parallel to it (k parallel to S_i, k' parallel to S_j).  On the axis itself the
    measured correlations are weak, so no on-axis neighbour is drawn.
    """
    xi = x0 + 0.22 + RADIUS  # dimer placed left of centre to leave room for the k' label
    xj = xi + BOND
    off = 0.30  # |k - i| = |k' - j| = BOND, displaced towards the far ends of the dimer
    xk = xi + off * BOND
    yk = y0 + BOND * math.sqrt(1 - off**2)
    xk2 = xj - off * BOND
    yk2 = y0 - BOND * math.sqrt(1 - off**2)

    line(ax, (xi, y0), (xj, y0), GREY)  # dimer bond
    line(ax, (xk, yk), (xi, y0), LIGHT, ls=DOTTED)  # k - i (transverse to the dimer)
    line(ax, (xk2, yk2), (xj, y0), LIGHT, ls=DOTTED)  # k' - j (transverse to the dimer)

    arrow(ax, (xi, y0), (xi + ARROW, y0), BLUE)  # S_i
    arrow(ax, (xj, y0), (xj - ARROW, y0), ORANGE)  # S_j antiparallel to S_i
    arrow(ax, (xk, yk), (xk + ARROW, yk), BLUE)  # S_k parallel to S_i
    arrow(ax, (xk2, yk2), (xk2 - ARROW, yk2), ORANGE)  # S_k' parallel to S_j
    for x, y in ((xi, y0), (xj, y0), (xk, yk), (xk2, yk2)):
        body(ax, x, y)

    text(ax, xi, y0 - 0.08, "$i$", va="top")
    text(ax, xj, y0 + 0.08, "$j$", va="bottom")
    text(ax, xk - 0.08, yk, "$k$", ha="right")
    text(ax, xk2 - 0.03, yk2 + 0.07, "$k'$", ha="right", va="bottom")
    text(ax, xk + ARROW / 2, yk + 0.085, "beside $i$: $\\parallel\\mathbf{S}_i$", va="bottom")
    text(ax, xk2 + 0.10, yk2 + 0.02, "beside $j$: $\\parallel\\mathbf{S}_j$", ha="left", va="bottom")


def make_figure():
    setup_rc()
    fig = plt.figure(figsize=(WIDTH, HEIGHT))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(0, HEIGHT)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")

    for key, fn in (("a", panel_a), ("b", panel_b), ("c", panel_c)):
        x0, w = COLS[key]
        panel_label(ax, x0 + 0.02, f"({key})")
        fn(ax, x0, w, Y_AXIS)
    return fig


def main(argv=None, dpi=200):
    args = sys.argv[1:] if argv is None else list(argv)
    script_dir = Path(__file__).resolve().parent
    default_out = (script_dir / "../../../spin_spin_paper_for_PRL/figures").resolve()
    outdir = Path(args[0]).expanduser().resolve() if args else default_out
    outdir.mkdir(parents=True, exist_ok=True)

    fig = make_figure()
    pdf = outdir / "fig1_schematic.pdf"
    png = outdir / "fig1_schematic.png"
    fig.savefig(pdf)
    fig.savefig(png, dpi=dpi)
    plt.close(fig)
    print(f"wrote {pdf}\nwrote {png}")


if __name__ == "__main__":
    main()
