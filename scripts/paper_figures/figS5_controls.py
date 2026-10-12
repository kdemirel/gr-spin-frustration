#!/usr/bin/env python3
"""Figure S5: channel decomposition for the control couplings, and the channel family.

  (a) longitudinal channel P (lines with markers) and transverse channel C_perp (dashed) for the
      GR tensor (M), the isotropic 1/r^3 antiferromagnet (S6_isotropic_afm) and the sign-reversed
      (dipolar) tensor (S6_dipolar);
  (b) Q(r) for the same three couplings, with the isotropic value 1/3;
  (c) C(r) of the channel family gamma = 1 (S7_gamma1, transverse only), 3 (M), 12 (S7_gamma12)
      and 100 (S7_gamma100, longitudinal only); drawn only when at least one S7 entry has an
      ensemble_table.csv.  The landmarks of _comparisons/cmp_channels/gamma_family.csv are
      printed when that file exists.

Data: <root>/<ENTRY>/analysis/ensemble_table.csv and summary.json of M, S6_isotropic_afm,
S6_dipolar (required) and S7_gamma1, S7_gamma12, S7_gamma100 (optional).
"""

from __future__ import annotations

import _style as st
import matplotlib.pyplot as plt
import numpy as np
from _style import BLACK, BLUE, GREEN, GREY, ORANGE, VERMILION
from matplotlib.gridspec import GridSpec

CONTROLS = [("M", "GR", BLACK, "o"), ("S6_isotropic_afm", "isotropic AFM", GREEN, "s"), ("S6_dipolar", "dipolar", VERMILION, "^")]
FAMILY = [("S7_gamma1", 1, BLUE, "v"), ("M", 3, BLACK, "o"), ("S7_gamma12", 12, ORANGE, "s"), ("S7_gamma100", 100, VERMILION, "^")]


def panels_ab(ax_a, ax_b, root: st.Root):
    st.zero_line(ax_a)
    st.zero_line(ax_b, 1.0 / 3.0, ls=st.DOTTED, lw=0.7)
    numbers = {}
    for entry, label, color, marker in CONTROLS:
        df = root.table(entry)
        lm = root.landmarks(entry)
        x, p, sp = st.series(df, "P")
        _, cp, scp = st.series(df, "Cperp")
        _, c, _ = st.series(df, "C")
        xq, q, sq = st.series(df, "Q")
        st.curve(ax_a, x, p, sp, color=color, marker=marker, ms=2.5, label=label)
        st.curve(ax_a, x, cp, scp, color=color, marker=None, ls=st.DASHED, lw=0.9)
        st.curve(ax_b, xq, q, sq, color=color, marker=marker, ms=2.5, label=label)
        iext = np.nanargmax(np.abs(cp))
        numbers[f"{entry}{root.label_suffix(entry)}"] = (
            f"C first bin = {c[0]:+.3f}; r_0 = {st.fmt(*lm.get('x_zero_C', (np.nan, np.nan)), digits=3)}; "
            f"C_max = {st.fmt(*lm.get('C_peak', (np.nan, np.nan)), digits=3)} at {st.fmt(*lm.get('x_peak_C', (np.nan, np.nan)), digits=2)}; "
            f"Cperp extreme = {cp[iext]:+.3f} at {x[iext]:.2f}; P range [{np.nanmin(p):+.3f}, {np.nanmax(p):+.3f}]; "
            f"Cperp range [{np.nanmin(cp):+.3f}, {np.nanmax(cp):+.3f}]; Q range [{np.nanmin(q):.3f}, {np.nanmax(q):.3f}]; "
            f"max |P - C/3| = {np.nanmax(np.abs(p - c / 3)):.3f}"
        )
    for ax in (ax_a, ax_b):
        st.xaxis_r(ax, 4.0)
    ax_a.set_ylim(-1.05, 1.05)
    ax_a.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax_a.set_ylabel(r"$P$ (solid), $C_\perp$ (dashed)")
    ax_b.set_ylim(0.0, 1.0)
    ax_b.set_ylabel("$Q$")
    ax_b.text(3.9, 1.0 / 3.0 + 0.01, "1/3", ha="right", va="bottom", fontsize=st.SMALL, color=GREY)
    st.legend(ax_a, loc="upper right", handlelength=1.3)
    st.legend(ax_b, loc="upper right", handlelength=1.3)
    st.report("Fig. S5(a,b)", numbers)


def panel_c(ax, root: st.Root):
    st.zero_line(ax)
    numbers = {}
    for entry, gamma, color, marker in FAMILY:
        if not root.has_entry(entry):
            print(f"figS5_controls (c): {entry} has no ensemble_table.csv; skipped")
            continue
        df = root.table(entry)
        lm = root.landmarks(entry)
        x, c, sc = st.series(df, "C")
        _, cp, _ = st.series(df, "Cperp")
        st.curve(ax, x, c, sc, color=color, marker=marker, ms=2.5, label=rf"$\gamma={gamma}$" + (" (GR)" if gamma == 3 else ""))
        numbers[f"gamma = {gamma} ({entry}{root.label_suffix(entry)})"] = (
            f"r_0 = {st.fmt(*lm.get('x_zero_C', (np.nan, np.nan)), digits=3)}, C_max = {st.fmt(*lm.get('C_peak', (np.nan, np.nan)), digits=3)} "
            f"at {st.fmt(*lm.get('x_peak_C', (np.nan, np.nan)), digits=2)}, Cperp_max = {st.fmt(*lm.get('Cperp_peak', (np.nan, np.nan)), digits=3)}, "
            f"Q_min = {st.fmt(*lm.get('Q_min', (np.nan, np.nan)), digits=3)}, C(1.2) = {st.interp_at(x, c, 1.2):+.3f}, "
            f"C first bin = {c[0]:+.3f}"
        )
    st.xaxis_r(ax, 4.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_ylabel("$C$")
    st.legend(ax, loc="lower right", ncol=2, handlelength=1.3, columnspacing=0.7)
    f = root.comparison_file("cmp_channels", "gamma_family.csv", required=False)
    if f is not None:
        g = st.pd.read_csv(f)
        for _, row in g.iterrows():
            numbers[f"gamma_family.csv: {row['run']}"] = (
                f"x_zero_C = {st.fmt(row.get('x_zero_C'), row.get('x_zero_C_err'))}, C_peak = {st.fmt(row.get('C_peak'), row.get('C_peak_err'))}, "
                f"Cperp_peak = {st.fmt(row.get('Cperp_peak'), row.get('Cperp_peak_err'))}, Q_min = {st.fmt(row.get('Q_min'), row.get('Q_min_err'))}, "
                f"C({row.get('x_sign')}) = {st.fmt(row.get('C_at_x_sign'), row.get('sem_C_at_x_sign'))}"
            )
    st.report("Fig. S5(c) channel family", numbers)


def build(root: st.Root, args):
    family = [e for e, *_ in FAMILY if e != "M" and root.has_entry(e)]
    has_c = bool(family)
    if not has_c:
        print("figS5_controls: no S7_gamma* entry analysed; panel (c) omitted")
    height = 1.8 + (1.55 if has_c else 0.0)
    fig = plt.figure(figsize=(st.SINGLE, height), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.04, wspace=0.04, hspace=0.02)
    if has_c:
        gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.85])
        ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, :])
    else:
        gs = GridSpec(1, 2, figure=fig)
        ax_a, ax_b, ax_c = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), None
    panels_ab(ax_a, ax_b, root)
    st.panel_label(ax_a, "(a)")
    st.panel_label(ax_b, "(b)")
    if ax_c is not None:
        panel_c(ax_c, root)
        st.panel_label(ax_c, "(c)")
    return fig


if __name__ == "__main__":
    st.run_main(build, __doc__, "figS5_controls")
