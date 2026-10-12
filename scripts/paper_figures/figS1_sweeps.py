#!/usr/bin/env python3
"""Figure S1: convergence of the relaxation (ensemble S1, 500 sweeps).

  (a) median relative excess energy (E_t - E_500)/|E_500| and mean largest single-spin
      rotation per sweep versus sweep (log-log), from histories.csv;
  (b) landmarks r_0, x_peak, C_max, Q_min versus the number of sweeps from
      sweep_convergence.csv (snapshots 1 ... 500); error bars are the bootstrap errors of the
      final-sweep landmarks (summary.json), which apply to every snapshot of the same
      realizations.  The vertical line marks the 100 production sweeps.  Panel (b) is omitted
      with a message when sweep_convergence.csv does not exist.

Data: <root>/S1/analysis/histories.csv, sweep_convergence.csv, summary.json.
"""

from __future__ import annotations

import _style as st
import matplotlib.pyplot as plt
import numpy as np
from _style import BLACK, BLUE, GREEN, GREY, ORANGE, PURPLE, VERMILION

ENTRY = "S1"
PRODUCTION_SWEEPS = 100


def panel_a(ax, root: st.Root):
    h = root.csv(ENTRY, "histories.csv")
    for col in ("sweep", "median_excess_energy", "mean_max_direction_change"):
        if col not in h.columns:
            raise st.FigureDataError(f"{ENTRY}: histories.csv has no column {col!r}")
    sweep = h["sweep"].to_numpy(float)
    exc = h["median_excess_energy"].to_numpy(float)
    rot = h["mean_max_direction_change"].to_numpy(float)
    n_final = int(sweep.max())
    ok_e = (sweep > 0) & (exc > 0)
    ok_r = (sweep > 0) & np.isfinite(rot) & (rot > 0)
    ax.axvline(PRODUCTION_SWEEPS, color=GREY, lw=0.6, ls=st.DOTTED, zorder=1)
    ax.plot(sweep[ok_e], exc[ok_e], color=BLUE, lw=st.LW, label=rf"$(E_t-E_{{{n_final}}})/|E_{{{n_final}}}|$")
    ax.plot(sweep[ok_r], rot[ok_r], color=ORANGE, lw=st.LW, label="largest $|\\Delta\\hat{\\mathbf{u}}|$")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("sweep $t$")
    ax.set_xlim(1, n_final)
    ax.set_ylim(min(exc[ok_e].min(), rot[ok_r].min()) * 0.5, 3.0)
    st.legend(ax, loc="lower left", handlelength=1.2)

    def at(arr, ok, t):
        i = np.where(sweep == t)[0]
        return f"{arr[i[0]]:.2e}" if len(i) and ok[i[0]] else "n/a"

    st.report(
        "Fig. S1(a) " + ENTRY + root.label_suffix(ENTRY),
        {
            "final sweep": n_final,
            "median excess energy at 10/50/100/200/400": ", ".join(at(exc, ok_e, t) for t in (10, 50, 100, 200, 400)),
            "mean max rotation at 10/100/500": ", ".join(at(rot, ok_r, t) for t in (10, 100, 500)),
        },
    )


def panel_b(ax, root: st.Root) -> bool:
    f = root.analysis_file(ENTRY, "sweep_convergence.csv", required=False)
    if f is None:
        print(f"figS1_sweeps: {root.analysis_dir(ENTRY) / 'sweep_convergence.csv'} not found; panel (b) omitted")
        return False
    sc = st.pd.read_csv(f)
    lm = root.landmarks(ENTRY)
    sweep = sc["sweep"].to_numpy(float)
    specs = [
        ("x_zero_C", "$r_0$", BLACK, "o"),
        ("x_peak_C", r"$r$ at $C_{\max}$", GREEN, "s"),
        ("C_peak", r"$C_{\max}$", VERMILION, "^"),
        ("Q_min", r"$Q_{\min}$", PURPLE, "v"),
    ]
    ax.axvline(PRODUCTION_SWEEPS, color=GREY, lw=0.6, ls=st.DOTTED, zorder=1)
    numbers = {}
    for col, label, color, marker in specs:
        if col not in sc.columns:
            print(f"figS1_sweeps: sweep_convergence.csv has no column {col!r}; skipped")
            continue
        y = sc[col].to_numpy(float)
        err = lm.get(col, (np.nan, np.nan))[1]
        sem = np.full_like(y, err if np.isfinite(err) else np.nan)
        ok = sweep > 0
        st.curve(ax, sweep[ok], y[ok], sem[ok], color=color, marker=marker, label=label)
        i100 = np.where(sweep == PRODUCTION_SWEEPS)[0]
        ifin = np.argmax(sweep)
        if len(i100):
            numbers[f"{col}: at 100 / at {int(sweep[ifin])} / shift / err"] = (
                f"{y[i100[0]]:.4f} / {y[ifin]:.4f} / {y[ifin] - y[i100[0]]:+.4f} / {err:.4f}"
            )
    ax.set_xscale("log")
    ax.set_xlabel("sweeps")
    ax.set_xlim(1, sweep.max())
    ax.set_ylim(0.0, 1.45)
    st.legend(ax, loc="center", ncol=2, columnspacing=0.6, bbox_to_anchor=(0.5, 0.30))
    st.report("Fig. S1(b) " + ENTRY, numbers)
    return True


def build(root: st.Root, args):
    has_b = root.analysis_file(ENTRY, "sweep_convergence.csv", required=False) is not None
    ncol = 2 if has_b else 1
    fig, axes = plt.subplots(1, ncol, figsize=(st.SINGLE, 1.75), constrained_layout=True, squeeze=False)
    fig.set_constrained_layout_pads(w_pad=0.03, h_pad=0.02, wspace=0.04, hspace=0.0)
    panel_a(axes[0, 0], root)
    st.panel_label(axes[0, 0], "(a)")
    if has_b:
        panel_b(axes[0, 1], root)
        st.panel_label(axes[0, 1], "(b)")
    else:
        panel_b(axes[0, 0], root)  # prints the message
    return fig


if __name__ == "__main__":
    st.run_main(build, __doc__, "figS1_sweeps")
