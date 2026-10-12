#!/usr/bin/env python3
"""Figure S6: nearest-neighbour diagnostics of the main ensemble.

  (a) fraction of pairs at separation r that are mutual nearest neighbours, with
      f_m(r) = exp(-9 pi rho r^3/4) and the one-sided estimate exp(-4 pi rho r^3/3);
  (b) C(r) restricted to mutual pairs, one-way pairs and all other ("far") pairs;
  (c) correlation C_k of a spin with its k-th nearest neighbour; the mean distance of that
      neighbour is given on the upper axis.

Data: <root>/M/analysis/nn_pairs.csv (x_center, f_mutual, sem_f_mutual, C_mutual, sem_C_mutual,
C_oneway, sem_C_oneway, C_far, sem_C_far, f_mutual_theory, f_isolated_sphere_theory) and
nn_ranks.csv (k, mean_r_over_rnn, C_k, sem_C_k), written by grspin analyze for runs with the
nearest-neighbour diagnostics.
"""

from __future__ import annotations

import _style as st
import matplotlib.pyplot as plt
import numpy as np
from _style import BLACK, BLUE, GREEN, GREY, ORANGE
from matplotlib.gridspec import GridSpec

ENTRY = "M"
XMAX = 2.0


def panel_a(ax, root: st.Root, nn):
    x = nn["x_center"].to_numpy(float)
    f, sf = nn["f_mutual"].to_numpy(float), nn["sem_f_mutual"].to_numpy(float)
    xs = np.linspace(0.0, XMAX, 300)
    ax.plot(xs, st.dimer_fraction(xs), color=GREY, lw=0.8, label=r"$f_m=e^{-9\pi\rho r^3/4}$", zorder=2)
    ax.plot(xs, st.isolated_sphere(xs), color=GREY, lw=0.8, ls=st.DASHED, label=r"$e^{-4\pi\rho r^3/3}$", zorder=2)
    ok = np.isfinite(f)
    st.curve(ax, x[ok], f[ok], sf[ok], color=BLACK, marker="o", ms=2.5, label="measured", zorder=4)
    st.xaxis_r(ax, XMAX, label=False)
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("mutual fraction")
    st.legend(ax, loc="upper right", handlelength=1.3)
    vals = {}
    for xv in (0.25, 0.45, 0.65, 0.85, 1.05):
        i = np.argmin(np.abs(x - xv))
        if abs(x[i] - xv) < 0.06:
            vals[f"x = {x[i]:.2f}"] = f"measured {f[i]:.3f}, f_m {st.dimer_fraction(x[i]):.3f}, one-sided {st.isolated_sphere(x[i]):.3f}"
    st.report("Fig. S6(a) " + ENTRY + root.label_suffix(ENTRY), vals)


def panel_b(ax, root: st.Root, nn):
    x = nn["x_center"].to_numpy(float)
    st.zero_line(ax)
    vals = {}
    for col, color, marker, label in (("C_mutual", BLUE, "o", "mutual"), ("C_oneway", ORANGE, "s", "one-way"), ("C_far", GREEN, "^", "other")):
        y, s = nn[col].to_numpy(float), nn[f"sem_{col}"].to_numpy(float)
        ok = np.isfinite(y)
        st.curve(ax, x[ok], y[ok], s[ok], color=color, marker=marker, ms=2.5, label=label)
        vals[col] = ", ".join(f"{x[i]:.2f}: {y[i]:+.3f}" for i in range(len(x)) if ok[i] and x[i] <= 1.3)
    if "C_all" in nn.columns:
        y = nn["C_all"].to_numpy(float)
        ok = np.isfinite(y)
        ax.plot(x[ok], y[ok], color=BLACK, lw=0.7, ls=st.DOTTED, label="all pairs", zorder=2)
    st.xaxis_r(ax, XMAX)
    ax.set_ylim(-1.05, 0.55)
    ax.set_ylabel("$C$ by pair class")
    st.legend(ax, loc="lower right", handlelength=1.3, ncol=2, columnspacing=0.6)
    st.report("Fig. S6(b) " + ENTRY, vals)


def panel_c(ax, root: st.Root, ranks):
    k = ranks["k"].to_numpy(float)
    ck, sk = ranks["C_k"].to_numpy(float), ranks["sem_C_k"].to_numpy(float)
    dist = ranks["mean_r_over_rnn"].to_numpy(float)
    st.zero_line(ax)
    st.curve(ax, k, ck, sk, color=BLACK, marker="o", label="$C_k$")
    ax.set_xlabel("neighbour rank $k$")
    ax.set_ylabel("$C_k$")
    ax.set_xlim(0.5, k.max() + 0.5)
    ax.set_xticks(k[:: 2 if len(k) > 8 else 1].astype(int))
    top = ax.secondary_xaxis("top")
    sel = slice(None, None, 2 if len(k) > 6 else 1)
    top.set_xticks(k[sel])
    top.set_xticklabels([f"{d:.2f}" for d in dist[sel]], fontsize=st.SMALL - 0.5)
    top.set_xlabel(r"mean $r_k/\bar{r}_{\mathrm{nn}}$", fontsize=st.SMALL)
    top.tick_params(direction="in", length=2.5, width=0.6)
    ax.tick_params(top=False)
    lo = min(-0.05, np.nanmin(ck - np.nan_to_num(sk)) - 0.02)
    hi = max(0.25, np.nanmax(ck + np.nan_to_num(sk)) + 0.02)
    ax.set_ylim(lo, hi)
    st.report(
        "Fig. S6(c) " + ENTRY,
        {f"k = {int(kk)}": f"C_k = {c:+.3f} ± {s:.3f}, mean r = {d:.2f}" for kk, c, s, d in zip(k, ck, sk, dist, strict=True)},
    )


def build(root: st.Root, args):
    nn = root.csv(ENTRY, "nn_pairs.csv")
    ranks = root.csv(ENTRY, "nn_ranks.csv")
    for col in ("x_center", "f_mutual", "C_mutual", "C_oneway", "C_far"):
        if col not in nn.columns:
            raise st.FigureDataError(f"{ENTRY}: nn_pairs.csv has no column {col!r}")
    for col in ("k", "mean_r_over_rnn", "C_k", "sem_C_k"):
        if col not in ranks.columns:
            raise st.FigureDataError(f"{ENTRY}: nn_ranks.csv has no column {col!r}")
    fig = plt.figure(figsize=(st.SINGLE, 4.4), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02, hspace=0.03)
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.8, 1.0, 1.0])
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0], sharex=ax_a)
    ax_c = fig.add_subplot(gs[2, 0])
    panel_a(ax_a, root, nn)
    ax_a.tick_params(labelbottom=False)
    ax_a.set_xlabel("")
    panel_b(ax_b, root, nn)
    panel_c(ax_c, root, ranks)
    st.panel_label(ax_a, "(a)")
    st.panel_label(ax_b, "(b)")
    ax_c.annotate("(c)", (0.0, 1.0), xycoords="axes fraction", xytext=(-2, 14), textcoords="offset points",
                  fontweight="bold", fontsize=st.FONTSIZE, ha="right", va="bottom", annotation_clip=False)
    return fig


if __name__ == "__main__":
    st.run_main(build, __doc__, "figS6_nn")
