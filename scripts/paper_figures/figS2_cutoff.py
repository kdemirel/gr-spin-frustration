#!/usr/bin/env python3
"""Figure S2: cutoff dependence of C(r) and the far tail.

  (a) C(r) on [0, 4] for r_cut = 3, 4, 5, 7.5, 10 (M), 12, 15 r_nn (S2_rc*, M; viridis);
  (b) far tail of C on [0, 10] for T_N10000_rc7, _rc10, _rc15 (and T_N20000_rc10, dashed) with
      y in [-0.03, 0.05], and the paired difference C(r_cut = 15) - C(r_cut = 10) as a thin line
      (from _comparisons/cmp_far_tail/paired_C_r_cut_10_minus_r_cut_15.csv, sign reversed) when
      that file exists.

Data: <root>/<ENTRY>/analysis/ensemble_table.csv of S2_rc3, S2_rc4, S2_rc5, S2_rc7p5, M,
S2_rc12, S2_rc15, T_N10000_rc7, T_N10000_rc10, T_N10000_rc15, T_N20000_rc10 (missing T entries are
skipped with a message; the figure needs at least M and one S2 entry) and
<root>/_comparisons/cmp_far_tail/.
"""

from __future__ import annotations

import _style as st
import matplotlib.pyplot as plt
import numpy as np
from _style import BLACK, VERMILION

CUTOFFS = [("S2_rc3", 3.0), ("S2_rc4", 4.0), ("S2_rc5", 5.0), ("S2_rc7p5", 7.5), ("M", 10.0), ("S2_rc12", 12.0), ("S2_rc15", 15.0)]
TAIL = [("T_N10000_rc7", 7.0, 10000, "-"), ("T_N10000_rc10", 10.0, 10000, "-"), ("T_N10000_rc15", 15.0, 10000, "-"), ("T_N20000_rc10", 10.0, 20000, st.DASHED)]
TAIL_COLORS = {7.0: "#440154", 10.0: "#21918C", 15.0: "#FDE725"}


def panel_a(ax, root: st.Root):
    cmap = plt.get_cmap("viridis")
    st.zero_line(ax)
    available = [(e, rc) for e, rc in CUTOFFS if root.has_entry(e)]
    missing = [e for e, _ in CUTOFFS if not root.has_entry(e)]
    if missing:
        print(f"figS2_cutoff (a): entries without ensemble_table.csv skipped: {missing}")
    if len(available) < 2 or not any(e == "M" for e, _ in available):
        raise st.FigureDataError(f"panel (a) needs M and at least one S2_rc* entry; found {available}")
    numbers = {}
    for k, (entry, rc) in enumerate(available):
        df = root.table(entry)
        lm = root.landmarks(entry)
        x, c, sc = st.series(df, "C")
        color = cmap(0.1 + 0.8 * k / max(len(available) - 1, 1))
        st.curve(
            ax, x, c, sc, color=color, marker="o" if entry == "M" else None, ms=2.2, lw=1.0,
            label=rf"$r_{{\rm cut}}={rc:g}$" + (" (M)" if entry == "M" else ""), zorder=4 if entry == "M" else 3,
        )
        imin = np.argmin(c[x > 1.5]) if (x > 1.5).any() else None
        numbers[f"r_cut = {rc:g} ({entry}{root.label_suffix(entry)})"] = (
            f"r_0 = {st.fmt(*lm.get('x_zero_C', (np.nan, np.nan)), digits=3)}, "
            f"C_max = {st.fmt(*lm.get('C_peak', (np.nan, np.nan)), digits=3)}, "
            f"min C beyond 1.5 = {c[x > 1.5][imin]:+.3f} at {x[x > 1.5][imin]:.2f}"
        )
    st.xaxis_r(ax, 4.0)
    ax.set_ylim(-1.05, 0.35)
    ax.set_ylabel("$C$")
    st.legend(ax, loc="lower right", ncol=2, columnspacing=0.7, handlelength=1.3)
    st.report("Fig. S2(a)", numbers)


def panel_b(ax, root: st.Root) -> bool:
    available = [(e, rc, n, ls) for e, rc, n, ls in TAIL if root.has_entry(e)]
    missing = [e for e, *_ in TAIL if not root.has_entry(e)]
    if missing:
        print(f"figS2_cutoff (b): far-tail entries without ensemble_table.csv skipped: {missing}")
    if not available:
        print("figS2_cutoff (b): no far-tail entry (T_*) analysed; panel (b) omitted")
        return False
    st.zero_line(ax)
    numbers = {}
    xmax = 4.0
    for entry, rc, n, ls in available:
        df = root.table(entry)
        x, c, sc = st.series(df, "C")
        xmax = max(xmax, float(x.max()) + 0.05)
        label = rf"$r_{{\rm cut}}={rc:g}$, $N={n // 1000}$k"
        st.curve(ax, x, c, sc, color=TAIL_COLORS.get(rc, BLACK), marker=None, ls=ls, lw=1.0 if ls == "-" else 0.9, label=label)
        far = x > 4.0
        if far.any():
            i = np.argmin(c[far])
            numbers[f"{entry}{root.label_suffix(entry)}"] = (
                f"x_max = {x.max():.2f}; min C beyond 4 = {c[far][i]:+.4f} ± {sc[far][i]:.4f} at {x[far][i]:.2f}; "
                f"C(4) = {st.interp_at(x, c, 4.0):+.4f}, C(x_max) = {c[-1]:+.4f}"
            )
        else:
            numbers[f"{entry}{root.label_suffix(entry)}"] = f"x_max = {x.max():.2f} (no far tail in this table)"
    f = root.paired_file("cmp_far_tail", "C", "r_cut = 10", "r_cut = 15", required=False)
    if f is not None:
        d = st.pd.read_csv(f)
        xd, diff, sem = d["x_center"].to_numpy(float), -d["diff"].to_numpy(float), d["sem"].to_numpy(float)
        ax.plot(xd, diff, color=VERMILION, lw=0.7, ls=st.DASHDOT, zorder=5, label=r"$C_{15}-C_{10}$ (paired)")
        i = np.nanargmax(np.abs(diff))
        numbers["paired C(15) - C(10)"] = (
            f"max |diff| = {diff[i]:+.4f} ± {sem[i]:.4f} at {xd[i]:.2f}; max |diff| for x <= 3.5 = {np.nanmax(np.abs(diff[xd <= 3.5])):.4f}"
        )
    else:
        print(f"figS2_cutoff (b): {root.comparison_dir('cmp_far_tail')}/paired_C_r_cut_10_minus_r_cut_15.csv not found; difference omitted")
    st.xaxis_r(ax, xmax)
    ax.set_ylim(-0.03, 0.05)
    ax.set_ylabel("$C$")
    st.legend(ax, loc="upper right", bbox_to_anchor=(1.0, 1.0), ncol=2, columnspacing=0.7, handlelength=1.4, framealpha=0.9)
    st.report("Fig. S2(b)", numbers)
    return True


def build(root: st.Root, args):
    has_b = any(root.has_entry(e) for e, *_ in TAIL)
    nrow = 2 if has_b else 1
    fig, axes = plt.subplots(nrow, 1, figsize=(st.SINGLE, 1.9 * nrow + 0.1), constrained_layout=True, squeeze=False)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.04, wspace=0.0, hspace=0.02)
    panel_a(axes[0, 0], root)
    st.panel_label(axes[0, 0], "(a)")
    if has_b:
        panel_b(axes[1, 0], root)
        st.panel_label(axes[1, 0], "(b)")
    else:
        panel_b(axes[0, 0], root)
    return fig


if __name__ == "__main__":
    st.run_main(build, __doc__, "figS2_cutoff")
