#!/usr/bin/env python3
"""Figure S3: finite size and boundary conditions.

  (a) r_0 (top) and C_max (bottom) versus N (log axis) for periodic and open boundaries, with
      bootstrap errors, from comparison_landmarks.csv of cmp_size_periodic and cmp_size_open;
  (b) paired difference open - periodic of C(r) and Q(r) at N = 5000 from
      cmp_boundary_N5000/paired_{C,Q}_open_minus_periodic.csv.

Data: <root>/_comparisons/cmp_size_periodic/comparison_landmarks.csv,
<root>/_comparisons/cmp_size_open/comparison_landmarks.csv,
<root>/_comparisons/cmp_boundary_N5000/paired_C_open_minus_periodic.csv and paired_Q_....
"""

from __future__ import annotations

import _style as st
import matplotlib.pyplot as plt
import numpy as np
from _style import BLACK, BLUE, ORANGE, PURPLE
from matplotlib.gridspec import GridSpec

SIZE_CMPS = {"periodic": "cmp_size_periodic", "open": "cmp_size_open"}
BOUNDARY_CMP = "cmp_boundary_N5000"
STYLE = {"periodic": dict(color=BLUE, marker="o"), "open": dict(color=ORANGE, marker="s", mfc="white", mew=0.8)}


def load_landmarks(root: st.Root, cmp: str):
    df = st.pd.read_csv(root.comparison_file(cmp, "comparison_landmarks.csv"))
    for col in ("n_particles", "x_zero_C", "x_zero_C_err", "C_peak", "C_peak_err"):
        if col not in df.columns:
            raise st.FigureDataError(f"{cmp}: comparison_landmarks.csv has no column {col!r}")
    return df.sort_values("n_particles")


def panel_a(ax_top, ax_bot, root: st.Root):
    numbers = {}
    for boundary, cmp in SIZE_CMPS.items():
        df = load_landmarks(root, cmp)
        n = df["n_particles"].to_numpy(float)
        kw = dict(STYLE[boundary])
        color = kw.pop("color")
        for ax, col in ((ax_top, "x_zero_C"), (ax_bot, "C_peak")):
            y, e = df[col].to_numpy(float), df[f"{col}_err"].to_numpy(float)
            ax.errorbar(n, y, yerr=e, color=color, ls="-", lw=0.8, elinewidth=st.ELW, capsize=1.5, ms=st.MS, label=boundary if ax is ax_top else None, zorder=3, **kw)
        numbers[f"{boundary}: N"] = ", ".join(f"{int(v)}" for v in n)
        numbers[f"{boundary}: r_0"] = ", ".join(f"{v:.3f}({int(round(e * 1000)):d})" for v, e in zip(df["x_zero_C"], df["x_zero_C_err"], strict=True))
        numbers[f"{boundary}: C_max"] = ", ".join(f"{v:.3f}({int(round(e * 1000)):d})" for v, e in zip(df["C_peak"], df["C_peak_err"], strict=True))
        if "Q_min" in df.columns:
            numbers[f"{boundary}: Q_min"] = ", ".join(f"{v:.3f}" for v in df["Q_min"])
    for ax in (ax_top, ax_bot):
        ax.set_xscale("log")
    ax_top.set_ylabel(r"$r_0/\bar{r}_{\mathrm{nn}}$")
    ax_bot.set_ylabel(r"$C_{\max}$")
    ax_bot.set_xlabel("$N$")
    ax_top.tick_params(labelbottom=False)
    lo, hi = ax_top.get_ylim()
    ax_top.set_ylim(lo - 0.35 * (hi - lo), hi)
    st.legend(ax_top, loc="lower right", ncol=2, handlelength=1.2, columnspacing=0.6)
    st.report("Fig. S3(a)", numbers)


def panel_b(ax, root: st.Root):
    st.zero_line(ax)
    numbers = {}
    for obs, color, marker, label in (("C", BLACK, "o", r"$\Delta C$"), ("Q", PURPLE, "s", r"$\Delta Q$")):
        f = root.paired_file(BOUNDARY_CMP, obs, "open", "periodic")
        d = st.pd.read_csv(f)
        x, diff, sem = d["x_center"].to_numpy(float), d["diff"].to_numpy(float), d["sem"].to_numpy(float)
        st.curve(ax, x, diff, sem, color=color, marker=marker, label=label)
        sig = np.isfinite(sem) & (np.abs(diff) > 2 * sem)
        i = np.nanargmax(np.abs(diff))
        numbers[f"{obs}: max |diff|"] = f"{diff[i]:+.4f} ± {sem[i]:.4f} at x = {x[i]:.2f}; significant (>2 sem) in {int(sig.sum())} of {len(x)} bins"
        if obs == "C":
            numbers["C: diff at 0.85 / 1.25"] = f"{st.interp_at(x, diff, 0.85):+.4f} / {st.interp_at(x, diff, 1.25):+.4f}"
    st.xaxis_r(ax, 4.0)
    ax.set_ylabel("open $-$ periodic")
    st.legend(ax, loc="lower right")
    st.report("Fig. S3(b) " + BOUNDARY_CMP, numbers)


def build(root: st.Root, args):
    fig = plt.figure(figsize=(st.SINGLE, 2.2), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02, wspace=0.05, hspace=0.02)
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 1.15])
    ax_top = fig.add_subplot(gs[0, 0])
    ax_bot = fig.add_subplot(gs[1, 0], sharex=ax_top)
    ax_b = fig.add_subplot(gs[:, 1])
    panel_a(ax_top, ax_bot, root)
    panel_b(ax_b, root)
    st.panel_label(ax_top, "(a)")
    st.panel_label(ax_b, "(b)")
    return fig


if __name__ == "__main__":
    st.run_main(build, __doc__, "figS3_size_boundary")
