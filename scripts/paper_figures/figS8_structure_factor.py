#!/usr/bin/env python3
"""Figure S8: shell-averaged spin structure factor S(k) of the unit spins.

S(k) of M before (light grey) and after (black) relaxation, and the final S(k) of the far-tail
entries T_N10000_rc7, T_N10000_rc10, T_N10000_rc15 (colours by r_cut) and T_N20000_rc10 (dashed)
on the same axes; x = k r_nn (log), y = S(k), horizontal line at S = 1.  The k = 0 values (first
row of structure_factor.csv) are drawn as markers at the left edge of the axis (tick "0").

Data: <root>/<ENTRY>/analysis/structure_factor.csv (k_rnn_lo, k_rnn_hi, k_rnn_mean, n_vectors,
wavelength_rnn, S_final, sem_S_final, S_initial, sem_S_initial); M is required, the T entries are
optional (skipped with a message).  The table of _comparisons/cmp_structure_factor/
structure_factor_vs_cutoff.csv is printed when present.
"""

from __future__ import annotations

import _style as st
import matplotlib.pyplot as plt
import numpy as np
from _style import BLACK, GREY, LIGHT

RUNS = [
    ("M", "M (final)", BLACK, "-", "o"),
    ("T_N10000_rc7", r"$r_{\rm cut}=7$, $N=10$k", "#440154", "-", "s"),
    ("T_N10000_rc10", r"$r_{\rm cut}=10$, $N=10$k", "#21918C", "-", "^"),
    ("T_N10000_rc15", r"$r_{\rm cut}=15$, $N=10$k", "#B5DE2B", "-", "v"),
    ("T_N20000_rc10", r"$r_{\rm cut}=10$, $N=20$k", "#21918C", st.DASHED, "D"),
]


def load_sk(root: st.Root, entry: str):
    df = root.csv(entry, "structure_factor.csv")
    for col in ("k_rnn_mean", "S_final"):
        if col not in df.columns:
            raise st.FigureDataError(f"{entry}: structure_factor.csv has no column {col!r}")
    k = df["k_rnn_mean"].to_numpy(float)
    zero = k <= 0
    return df, k, zero


def build(root: st.Root, args):
    fig, ax = plt.subplots(figsize=(st.SINGLE, 2.1), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02)
    st.zero_line(ax, 1.0)
    available = [r for r in RUNS if root.has_entry(r[0], "structure_factor.csv")]
    missing = [r[0] for r in RUNS if r not in available]
    if "M" in missing:
        raise st.FigureDataError(f"entry M{root.label_suffix('M')}: missing {root.analysis_dir('M') / 'structure_factor.csv'}")
    if missing:
        print(f"figS8_structure_factor: entries without structure_factor.csv skipped: {missing}")
    kmins = []
    for entry, *_ in available:
        _, k, zero = load_sk(root, entry)
        if (~zero).any():
            kmins.append(k[~zero].min())
    k0 = 0.5 * min(kmins) if kmins else 0.1  # position of the k = 0 markers
    numbers = {}
    for entry, label, color, ls, marker in available:
        df, k, zero = load_sk(root, entry)
        sf, ssf = df["S_final"].to_numpy(float), df["sem_S_final"].to_numpy(float) if "sem_S_final" in df.columns else np.full(len(df), np.nan)
        if entry == "M" and "S_initial" in df.columns:
            si, ssi = df["S_initial"].to_numpy(float), df["sem_S_initial"].to_numpy(float) if "sem_S_initial" in df.columns else np.full(len(df), np.nan)
            st.curve(ax, k[~zero], si[~zero], ssi[~zero], color=LIGHT, marker="o", ms=2.3, label="M (initial)", zorder=2)
            if zero.any():
                st.curve(ax, [k0], si[zero][:1], ssi[zero][:1], color=LIGHT, marker="o", ms=2.3, zorder=2)
        st.curve(ax, k[~zero], sf[~zero], ssf[~zero], color=color, marker=marker, ms=2.3, ls=ls, label=label, zorder=4 if entry == "M" else 3)
        if zero.any():
            st.curve(ax, [k0], sf[zero][:1], ssf[zero][:1], color=color, marker=marker, ms=2.3, zorder=4 if entry == "M" else 3)
        imax = np.nanargmax(sf[~zero]) if (~zero).any() else None
        numbers[f"{entry}{root.label_suffix(entry)}"] = (
            f"S(0) = {st.fmt(sf[zero][0], ssf[zero][0]) if zero.any() else 'n/a'}"
            + (f", S_initial(0) = {st.fmt(si[zero][0], ssi[zero][0])}" if entry == "M" and zero.any() and "S_initial" in df.columns else "")
            + (f", first shell k r_nn = {k[~zero][0]:.3f}: S = {st.fmt(sf[~zero][0], ssf[~zero][0])}" if (~zero).any() else "")
            + (f", max S(k>0) = {sf[~zero][imax]:.3f} at k r_nn = {k[~zero][imax]:.3f}" if imax is not None else "")
            + f", shells = {int((~zero).sum())}"
        )
    ax.set_xscale("log")
    ax.set_xlabel(r"$k\,\bar{r}_{\mathrm{nn}}$")
    ax.set_ylabel("$S(k)$")
    ax.axvline(k0 * 1.35, color=GREY, lw=0.4, ls=st.DOTTED, zorder=1)
    ax.set_xlim(k0 * 0.7, ax.get_xlim()[1])
    ticks = [t for t in (0.2, 0.5, 1, 2, 5) if k0 * 1.35 < t < ax.get_xlim()[1]]
    ax.set_xticks([k0] + ticks)
    ax.set_xticklabels(["0"] + [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_ylim(0.0, max(1.5, ax.get_ylim()[1]))
    st.legend(ax, loc="upper left", bbox_to_anchor=(1.01, 1.0), handlelength=1.4, ncol=1, borderaxespad=0.0)
    f = root.comparison_file("cmp_structure_factor", "structure_factor_vs_cutoff.csv", required=False)
    if f is not None:
        t = st.pd.read_csv(f)
        for _, row in t.iterrows():
            numbers[f"vs_cutoff: {row['label']}"] = f"S0 = {st.fmt(row.get('S0'), row.get('sem_S0'))}, S0_initial = {st.fmt(row.get('S0_initial'))}, k1 = {row.get('k1_rnn'):.3f}, S1 = {st.fmt(row.get('S1'), row.get('sem_S1'))}, S_max = {st.fmt(row.get('S_max'))} at {row.get('k_max_rnn'):.3f}"
    st.report("Fig. S8 structure factor", numbers)
    return fig


if __name__ == "__main__":
    st.run_main(build, __doc__, "figS8_structure_factor")
