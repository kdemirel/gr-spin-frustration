#!/usr/bin/env python3
"""Figure 2 of the PRL manuscript: pair correlations of the relaxed state (double column).

  (a) M: C (black), P (orange), C_perp (blue) versus r/r_nn on [0, 4]; the exact dimer
      fraction -f_m(r) = -exp(-9 pi rho r^3/4) (thin grey line); the initial C (light grey);
      r_0 marked.
  (b) M: B/2 (vermilion), A (green), Q (purple); dotted line at 1/3; dashed line
      (A + P)/(1 + C) for x >= 1.3 (where A = 1/3), the unit-spin estimate of Q from the channels.
  (c) C(r) for M (black), S6_isotropic_afm (green) and S6_dipolar (vermilion).

Data: <root>/<ENTRY>/analysis/ensemble_table.csv and summary.json of M, S6_isotropic_afm,
S6_dipolar.  Usage: python3 fig2_correlations.py [--results-root R] [--out DIR]
[--entry-map M=S3_N5000_periodic ...]
"""

from __future__ import annotations

import _style as st
import matplotlib.pyplot as plt
import numpy as np
from _style import BLACK, BLUE, GREEN, GREY, LIGHT, ORANGE, PURPLE, VERMILION

WIDTH, HEIGHT = st.DOUBLE, 2.2
ENTRIES = {"M": "M", "afm": "S6_isotropic_afm", "dip": "S6_dipolar"}


def panel_a(ax, root: st.Root):
    df = root.table("M")
    lm = root.landmarks("M")
    x, c, sc = st.series(df, "C")
    _, p, sp = st.series(df, "P")
    _, cp, scp = st.series(df, "Cperp")
    xi, ci, _ = st.series(df, "C", state="initial")

    st.zero_line(ax)
    xs = np.linspace(0.0, 4.0, 400)
    ax.plot(xs, -st.dimer_fraction(xs), color=GREY, lw=0.7, zorder=2, label=r"$-f_m(r)$")
    ax.plot(xi, ci, color=LIGHT, lw=0.8, zorder=2, label="initial $C$")
    st.curve(ax, x, cp, scp, color=BLUE, marker="^", label=r"$C_\perp$")
    st.curve(ax, x, p, sp, color=ORANGE, marker="s", label="$P$")
    st.curve(ax, x, c, sc, color=BLACK, marker="o", label="$C$", zorder=4)

    x0, x0_err = lm.get("x_zero_C", (np.nan, np.nan))
    if np.isfinite(x0):
        ax.plot([x0, x0], [-0.07, 0.07], color=BLACK, lw=0.8, zorder=5)
        ax.annotate(
            "$r_0$", (x0, -0.09), xytext=(3, 0), textcoords="offset points", ha="left", va="top", fontsize=st.SMALL
        )

    st.xaxis_r(ax, 4.0)
    ax.set_ylim(-1.05, 0.75)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5])
    ax.set_ylabel(r"$C,\ P,\ C_\perp$")
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(k) for k in ("$C$", "$P$", r"$C_\perp$", r"$-f_m(r)$", "initial $C$")]
    st.legend(ax, handles=[handles[i] for i in order], labels=[labels[i] for i in order], loc="upper right", ncol=2)

    # numbers for the caption / text
    cperp_pk = lm.get("Cperp_peak", (np.nan, np.nan))
    st.report(
        "Fig. 2(a) M" + root.label_suffix("M"),
        {
            "bins": f"{len(x)} on [0, {x.max() + (x[1] - x[0]) / 2:.2f}] (width {x[1] - x[0]:.3g} r_nn)",
            "r_0 (zero of C)": st.fmt(*lm.get("x_zero_C", (np.nan, np.nan)), digits=4),
            "C_max": st.fmt(*lm.get("C_peak", (np.nan, np.nan)), digits=4),
            "x at C_max": st.fmt(*lm.get("x_peak_C", (np.nan, np.nan)), digits=3),
            "Cperp_max": st.fmt(*cperp_pk, digits=4),
            "x at Cperp_max": st.fmt(*lm.get("x_peak_Cperp", (np.nan, np.nan)), digits=3),
            "P at Cperp_max": st.fmt(*lm.get("P_at_Cperp_peak", (np.nan, np.nan)), digits=3),
            "C at 4 r_nn (last bin)": f"{c[-1]:.4f} ± {sc[-1]:.4f}",
            "C first bin": f"{c[0]:.4f}",
            "1+C vs 1-f_m at x=0.3": f"{1 + st.interp_at(x, c, 0.3):.3f} vs {1 - st.dimer_fraction(0.3):.3f}",
        },
    )


def panel_b(ax, root: st.Root):
    df = root.table("M")
    lm = root.landmarks("M")
    x, c, _ = st.series(df, "C")
    _, p, _ = st.series(df, "P")
    _, b, sb = st.series(df, "B")
    _, a, sa = st.series(df, "A")
    xq, q, sq = st.series(df, "Q")

    st.zero_line(ax, 1.0 / 3.0, ls=st.DOTTED, color=GREY, lw=0.7)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_implied = (a + p) / (1.0 + c)
    # the unit-spin estimate is meaningful only where single spins are isotropic (A = 1/3);
    # below ~1.3 r_nn the dimers (A -> 1, C -> -1) make it a 0/0 form, so it is not drawn there
    ok = np.isfinite(q_implied) & (np.abs(1.0 + c) > 1e-3) & (x >= 1.3)
    ax.plot(x[ok], q_implied[ok], color=BLACK, ls=st.DASHED, lw=0.9, zorder=5, label=r"$(A+P)/(1+C)$")
    st.curve(ax, x, b / 2.0, sb / 2.0, color=VERMILION, marker="o", label="$B/2$")
    st.curve(ax, x, a, sa, color=GREEN, marker="s", label="$A$")
    st.curve(ax, xq, q, sq, color=PURPLE, marker="^", label="$Q$")

    st.xaxis_r(ax, 4.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel(r"$B/2,\ A,\ Q$")
    ax.text(0.1, 1.0 / 3.0 + 0.01, "1/3", ha="left", va="bottom", fontsize=st.SMALL, color=GREY)
    st.legend(ax, loc="upper right")

    st.report(
        "Fig. 2(b) M" + root.label_suffix("M"),
        {
            "B/2 at x=0 (first bin)": f"{b[0] / 2:.3f}",
            "B/2 at r_0": f"{st.interp_at(x, b / 2, lm.get('x_zero_C', (np.nan,))[0]):.3f}",
            "B/2 at 2, 4 r_nn": f"{st.interp_at(x, b / 2, 2.0):.3f}, {b[-1] / 2:.3f}",
            "B/2 min over bins": f"{(b / 2).min():.4f} (positive everywhere: {bool((b > 0).all())})",
            "A first bin": f"{a[0]:.3f}",
            "x where A first < 0.34": f"{x[np.argmax(a < 0.34)]:.2f}" if (a < 0.34).any() else "never",
            "Q first bin": f"{q[0]:.3f}",
            "Q crosses 1/3 at": st.fmt(*lm.get("x_cross_Q_third", (np.nan, np.nan)), digits=4),
            "Q_min": st.fmt(*lm.get("Q_min", (np.nan, np.nan)), digits=4),
            "x at Q_min": st.fmt(*lm.get("x_min_Q", (np.nan, np.nan)), digits=3),
            "Q at 4 r_nn": f"{q[-1]:.3f}",
            "Q vs (A+P)/(1+C) at x=0.5,1,1.5,2,3": ", ".join(
                f"{st.interp_at(xq, q, v):.3f}/{st.interp_at(x[ok], q_implied[ok], v):.3f}" for v in (0.5, 1.0, 1.5, 2.0, 3.0)
            ),
            "max |Q - (A+P)/(1+C)| for x>1": f"{np.nanmax(np.abs(q[ok & (x > 1.0)] - q_implied[ok & (x > 1.0)])):.3f}",
        },
    )


def panel_c(ax, root: st.Root):
    st.zero_line(ax)
    specs = [("M", BLACK, "o", "GR tensor"), ("afm", GREEN, "s", "isotropic AFM"), ("dip", VERMILION, "^", "dipolar")]
    for key, color, marker, label in specs:
        entry = ENTRIES[key]
        df = root.table(entry)
        lm = root.landmarks(entry)
        x, c, sc = st.series(df, "C")
        st.curve(ax, x, c, sc, color=color, marker=marker, label=label, zorder=4 if key == "M" else 3)
        _, cp, _ = st.series(df, "Cperp")
        _, p, _ = st.series(df, "P")
        _, q, _ = st.series(df, "Q")
        st.report(
            f"Fig. 2(c) {entry}" + root.label_suffix(entry),
            {
                "C first bin": f"{c[0]:.3f}",
                "r_0": st.fmt(*lm.get("x_zero_C", (np.nan, np.nan)), digits=3),
                "C_max": st.fmt(*lm.get("C_peak", (np.nan, np.nan)), digits=3)
                + f" at x = {st.fmt(*lm.get('x_peak_C', (np.nan, np.nan)), digits=2)}",
                "Cperp extreme": f"{cp[np.nanargmax(np.abs(cp))]:+.3f} at x = {x[np.nanargmax(np.abs(cp))]:.2f}",
                "max |P - C/3|": f"{np.nanmax(np.abs(p - c / 3)):.3f}",
                "Q range": f"[{np.nanmin(q):.3f}, {np.nanmax(q):.3f}]",
                "sign changes of C in [0,4]": int(np.sum(np.diff(np.sign(c)) != 0)),
            },
        )
    st.xaxis_r(ax, 4.0)
    ax.set_ylim(-1.05, 1.05)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax.set_ylabel("$C$")
    st.legend(ax, loc="upper right")


def build(root: st.Root, args):
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, HEIGHT), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.03, wspace=0.03, hspace=0.0)
    panel_a(axes[0], root)
    panel_b(axes[1], root)
    panel_c(axes[2], root)
    for ax, lab in zip(axes, "abc", strict=False):
        st.panel_label(ax, f"({lab})")
    return fig


if __name__ == "__main__":
    st.run_main(build, __doc__, "fig2_correlations")
