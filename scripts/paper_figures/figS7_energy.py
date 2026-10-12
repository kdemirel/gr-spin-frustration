#!/usr/bin/env python3
"""Figure S7: energy budget of the stored relaxed states for all couplings.

Reads <root>/<ENTRY>/analysis/energy_budget.csv (one row per stored state, METHODS 8.8) for the
gravitational tensor (M), the controls (S6), the channel family (S7) and the hard-core ensemble
(S8), and draws per ensemble the per-state values, their median and the interquartile range of
  (a) the frustration index E / sum of the two-body minima of all bonds within r_cut;
  (b) the realized fraction of the two-body minima of the non-dimer bonds, eta_nd.
Ensembles whose energy_budget.csv is missing are skipped with a message.
"""

from __future__ import annotations

import _style as st
import matplotlib.pyplot as plt
import numpy as np
from _style import BLACK, BLUE, GREEN, GREY, ORANGE, PURPLE, VERMILION

ENSEMBLES = [
    ("M", "GR", BLACK),
    ("S6_dipolar", "dipolar", VERMILION),
    ("S6_isotropic_afm", "iso. AFM", GREEN),
    ("S7_gamma1", r"$\gamma=1$", BLUE),
    ("S7_gamma12", r"$\gamma=12$", ORANGE),
    ("S7_gamma100", r"$\gamma=100$", PURPLE),
    ("S8_hardcore05", "hard core", GREY),
]
PANELS = [
    ("frustration_index", r"$E/\sum_{i<j}U^{\min}_{ij}$"),
    ("realized_fraction_nondimer", r"$\eta_{\rm nd}$ (non-dimer bonds)"),
]


def build(root: st.Root, args):
    data = []
    for entry, label, color in ENSEMBLES:
        f = root.analysis_file(entry, "energy_budget.csv", required=False)
        if f is None:
            print(f"figS7_energy: no energy_budget.csv for {entry}, skipped")
            continue
        data.append((entry, label, color, st.pd.read_csv(f)))
    if not data:
        raise st.FigureDataError("no energy_budget.csv found for any ensemble")

    fig, axes = plt.subplots(2, 1, figsize=(st.SINGLE, 3.0), sharex=True, constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.02, h_pad=0.02, hspace=0.03)
    rng = np.random.default_rng(0)
    numbers = {}
    for ax, (col, ylabel) in zip(axes, PANELS, strict=True):
        for i, (entry, _label, color, eb) in enumerate(data):
            v = eb[col].to_numpy(float)
            v = v[np.isfinite(v)]
            jitter = (rng.random(len(v)) - 0.5) * 0.35
            ax.plot(i + jitter, v, ls="none", marker="o", ms=1.8, color=color, alpha=0.55, zorder=2)
            q25, med, q75 = np.percentile(v, [25, 50, 75])
            ax.plot([i - 0.28, i + 0.28], [med, med], color=color, lw=1.4, zorder=4)
            ax.add_patch(
                plt.Rectangle((i - 0.28, q25), 0.56, q75 - q25, facecolor="none", edgecolor=color, lw=0.7, zorder=3)
            )
            numbers[f"{col} {entry}"] = f"median {med:.3f}, IQR [{q25:.3f}, {q75:.3f}], range [{v.min():.3f}, {v.max():.3f}], n = {len(v)}"
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        ax.tick_params(top=False)
    axes[-1].set_xticks(range(len(data)))
    axes[-1].set_xticklabels([d[1] for d in data], fontsize=st.SMALL, rotation=30, ha="right")
    axes[-1].set_xlim(-0.6, len(data) - 0.4)
    st.panel_label(axes[0], "(a)")
    st.panel_label(axes[1], "(b)")
    st.report("Fig. S7 energy budget", numbers)
    return fig


if __name__ == "__main__":
    st.run_main(build, __doc__, "figS7_energy")
