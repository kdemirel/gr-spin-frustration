#!/usr/bin/env python3
"""Figure S4: dependence on the distribution of spin magnitudes.

  (a) C(r) (lines with markers) and C_perp(r) (dashed, no markers) for U(0, S) (M), equal
      magnitudes (S4_equal), U(S/2, 3S/2) (S4_uniform_half) and log-normal (S4_lognormal);
  (b) Q(r) for the same ensembles (Q is 0 at short distance for equal magnitudes).

Data: <root>/<ENTRY>/analysis/ensemble_table.csv and summary.json of M, S4_equal,
S4_uniform_half, S4_lognormal.
"""

from __future__ import annotations

import _style as st
import matplotlib.pyplot as plt
import numpy as np
from _style import BLACK, BLUE, GREEN, GREY, ORANGE

RUNS = [
    ("M", "$U(0,S)$", BLACK, "o"),
    ("S4_equal", "equal", BLUE, "s"),
    ("S4_uniform_half", "$U(S/2,3S/2)$", ORANGE, "^"),
    ("S4_lognormal", "log-normal", GREEN, "v"),
]


def build(root: st.Root, args):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(st.SINGLE, 1.8), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02, wspace=0.04, hspace=0.0)
    st.zero_line(ax_a)
    st.zero_line(ax_b, 1.0 / 3.0, ls=st.DOTTED, lw=0.7)
    numbers = {}
    for entry, label, color, marker in RUNS:
        df = root.table(entry)
        lm = root.landmarks(entry)
        x, c, sc = st.series(df, "C")
        _, cp, scp = st.series(df, "Cperp")
        xq, q, sq = st.series(df, "Q")
        st.curve(ax_a, x, c, sc, color=color, marker=marker, ms=2.5, label=label)
        st.curve(ax_a, x, cp, scp, color=color, marker=None, ls=st.DASHED, lw=0.9)
        st.curve(ax_b, xq, q, sq, color=color, marker=marker, ms=2.5, label=label)
        numbers[f"{entry}{root.label_suffix(entry)}"] = (
            f"r_0 = {st.fmt(*lm.get('x_zero_C', (np.nan, np.nan)), digits=3)}, "
            f"C_max = {st.fmt(*lm.get('C_peak', (np.nan, np.nan)), digits=3)}, "
            f"Cperp_max = {st.fmt(*lm.get('Cperp_peak', (np.nan, np.nan)), digits=3)}, "
            f"Q_min = {st.fmt(*lm.get('Q_min', (np.nan, np.nan)), digits=3)} at {st.fmt(*lm.get('x_min_Q', (np.nan, np.nan)), digits=2)}, "
            f"Q = 1/3 at {st.fmt(*lm.get('x_cross_Q_third', (np.nan, np.nan)), digits=3)}, "
            f"Q first bin = {q[0]:.3f}, min P = {np.nanmin(st.series(df, 'P')[1]):+.3f}, min Cperp = {np.nanmin(cp):+.3f}"
        )
    st.xaxis_r(ax_a, 4.0)
    st.xaxis_r(ax_b, 4.0)
    ax_a.set_ylim(-1.05, 0.75)
    ax_a.set_yticks([-1.0, -0.5, 0.0, 0.5])
    ax_a.set_ylabel(r"$C$ (solid), $C_\perp$ (dashed)")
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("$Q$")
    ax_b.text(3.9, 1.0 / 3.0 + 0.01, "1/3", ha="right", va="bottom", fontsize=st.SMALL, color=GREY)
    st.legend(ax_a, loc="lower right", handlelength=1.3)
    st.legend(ax_b, loc="upper right", handlelength=1.3)
    st.panel_label(ax_a, "(a)")
    st.panel_label(ax_b, "(b)")
    st.report("Fig. S4", numbers)
    return fig


if __name__ == "__main__":
    st.run_main(build, __doc__, "figS4_magnitudes")
