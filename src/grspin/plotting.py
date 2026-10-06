"""Publication-style figures.

All functions return :class:`matplotlib.figure.Figure` objects that are *not* registered with
``pyplot`` (no global state, no backend requirements); save them with :func:`save_figure` or
display them in a notebook. Sizes follow the APS column widths and the smallest text is 8 pt, which
keeps capital letters above the 2 mm required by the journal when the figure is not rescaled.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import ticker
from matplotlib.figure import Figure

from . import theory
from .analysis import EnsembleData
from .fit import FitResult

SINGLE_COLUMN = 3.4  # inches
DOUBLE_COLUMN = 7.0  # inches

STYLE = {
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 8,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 7.5,
    "legend.frameon": False,
    "legend.handlelength": 1.6,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.2,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

COLOR_RELAXED = "#1f77b4"
COLOR_INITIAL = "0.6"
COLOR_NN = "#d62728"
COLOR_FIT = "#2ca02c"
CYCLE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]

X_LABEL = r"$r/\bar{r}_{\rm nn}$"
OBS_LABEL = {
    "C": r"$C(r)=\langle\hat{S}_i\cdot\hat{S}_j\rangle$",
    "Q": r"$Q(r)=\langle\cos^2\varphi\rangle$",
    "A": r"$A(r)=\langle(\hat{S}\cdot\hat{r})^2\rangle$",
    "P": r"$P(r)=\langle(\hat{S}_i\cdot\hat{r})(\hat{S}_j\cdot\hat{r})\rangle$",
    "B": r"$B(r)=C-3P$",
}
OBS_BASELINE = {"C": 0.0, "Q": 1.0 / 3.0, "A": 1.0 / 3.0, "P": 0.0, "B": 0.0}


@contextmanager
def _style():
    with mpl.rc_context(STYLE):
        yield


def save_figure(fig: Figure, path: str | Path) -> Path:
    """Write ``fig`` to ``path`` (format from the suffix; ``.pdf`` and ``.png`` are typical)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _style():
        fig.savefig(path)
    return path


def _log_axis(ax) -> None:
    """Logarithmic y axis whose minor ticks are labelled when the data span < 1 decade."""
    ax.set_yscale("log")
    ax.yaxis.set_minor_formatter(
        ticker.LogFormatter(labelOnlyBase=False, minor_thresholds=(2, 0.4))
    )


# --------------------------------------------------------------------------------------
# Single-run panels
# --------------------------------------------------------------------------------------
def draw_observable(
    ax,
    data: EnsembleData,
    obs: str,
    rebin: int = 1,
    label: str = "relaxed",
    color: str = COLOR_RELAXED,
    show_initial: bool = True,
    show_baseline: bool = True,
) -> None:
    """Mean and standard error of ``obs`` (initial and relaxed state) on ``ax``."""
    t = data.table(rebin)
    if show_baseline:
        ax.axhline(OBS_BASELINE[obs], ls=":", lw=0.9, color="0.4")
    if show_initial:
        ax.errorbar(
            t["x_center"],
            t[f"mean_{obs}_initial"],
            t[f"sem_{obs}_initial"],
            fmt="o",
            ms=2.2,
            lw=0.8,
            color=COLOR_INITIAL,
            label="initial (random)",
        )
    ax.errorbar(
        t["x_center"],
        t[f"mean_{obs}_final"],
        t[f"sem_{obs}_final"],
        fmt="o",
        ms=3,
        lw=0.9,
        color=color,
        label=label,
    )
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel(OBS_LABEL[obs])
    ax.set_xlim(0.0, data.config.x_max)


def draw_C(
    ax,
    data: EnsembleData,
    rebin: int = 1,
    show_initial: bool = True,
    show_nn: bool = True,
    fit: FitResult | None = None,
) -> None:
    """``C(r)`` with the nearest-neighbour references and an optional damped-oscillatory fit."""
    draw_observable(ax, data, "C", rebin, show_initial=show_initial)
    xx = np.linspace(1e-3, data.config.x_max, 400)
    if show_nn:
        ax.plot(
            xx,
            -theory.mutual_nn_probability(xx),
            "--",
            color=COLOR_NN,
            lw=1.0,
            label=r"$-e^{-9\pi\rho r^{3}/4}$ (mutual NN)",
        )
        ax.plot(
            xx,
            -theory.isolated_sphere_probability(xx),
            ":",
            color=COLOR_NN,
            lw=1.0,
            label=r"$-e^{-4\pi\rho r^{3}/3}$",
        )
    if fit is not None and fit.success:
        ax.plot(xx, fit.curve(xx), "-", color=COLOR_FIT, lw=1.1, label="damped-oscillatory fit")
    t = data.table(rebin)
    top = max(0.35, float(np.nanmax(t["mean_C_final"] + t["sem_C_final"].fillna(0))) + 0.08)
    ax.set_ylim(-1.05, top)


def draw_Q(ax, data: EnsembleData, rebin: int = 1, show_initial: bool = True) -> None:
    draw_observable(ax, data, "Q", rebin, show_initial=show_initial)
    ax.set_ylim(0.0, 1.05)


def draw_A(ax, data: EnsembleData, rebin: int = 1, show_initial: bool = True) -> None:
    draw_observable(ax, data, "A", rebin, show_initial=show_initial)
    ax.set_ylim(0.2, 1.05)


def draw_B(ax, data: EnsembleData, rebin: int = 1, show_initial: bool = True) -> None:
    """Tensor (pair-energy) correlation; a perfect dimer has ``B = 2``."""
    draw_observable(ax, data, "B", rebin, show_initial=show_initial)
    ax.axhline(2.0, ls="--", lw=0.8, color="0.6")
    ax.set_ylim(-0.25, 2.1)


def figure_C(data: EnsembleData, rebin: int = 1, fit: FitResult | None = None, **kw) -> Figure:
    with _style():
        fig = Figure(figsize=(SINGLE_COLUMN, 2.6), layout="constrained")
        ax = fig.subplots()
        draw_C(ax, data, rebin, fit=fit, **kw)
        ax.legend(loc="lower right")
    return fig


def figure_QA(data: EnsembleData, rebin: int = 1) -> Figure:
    """``Q(r)`` (top) and ``A(r)`` (bottom) in one single-column figure."""
    with _style():
        fig = Figure(figsize=(SINGLE_COLUMN, 4.2), layout="constrained")
        ax_q, ax_a = fig.subplots(2, 1, sharex=True)
        draw_Q(ax_q, data, rebin)
        ax_q.set_xlabel("")
        ax_q.legend(loc="upper right")
        draw_A(ax_a, data, rebin, show_initial=data.has("init_sum_A"))
    return fig


def figure_overview(data: EnsembleData, rebin: int = 1, fit: FitResult | None = None) -> Figure:
    """Double-column overview with panels ``C, Q, A, B``."""
    with _style():
        fig = Figure(figsize=(DOUBLE_COLUMN, 5.4), layout="constrained")
        axes = fig.subplots(2, 2)
        draw_C(axes[0, 0], data, rebin, fit=fit)
        axes[0, 0].legend(loc="lower right")
        draw_Q(axes[0, 1], data, rebin)
        axes[0, 1].legend(loc="upper right")
        if data.has("fin_sum_A"):
            draw_A(axes[1, 0], data, rebin)
            draw_B(axes[1, 1], data, rebin)
        else:
            axes[1, 0].axis("off")
            axes[1, 1].axis("off")
        for label, ax in zip("abcd", axes.ravel(), strict=True):
            ax.set_title(f"({label})", loc="left", pad=3)
    return fig


def figure_histories(data: EnsembleData) -> Figure:
    """Relaxation history: energy excess, largest direction change per sweep, total spin."""
    h = data.histories()
    n = data.config.n_particles
    with _style():
        fig = Figure(figsize=(DOUBLE_COLUMN, 2.4), layout="constrained")
        ax_e, ax_d, ax_s = fig.subplots(1, 3)
        good = (h["mean_excess_energy"] > 0) & np.isfinite(h["mean_excess_energy"])
        ax_e.errorbar(
            h["sweep"][good],
            h["mean_excess_energy"][good],
            h["sem_excess_energy"][good],
            fmt="o-",
            ms=2.5,
            lw=0.8,
            color=COLOR_RELAXED,
        )
        _log_axis(ax_e)
        ax_e.set_xlabel("sweep")
        ax_e.set_ylabel(r"$(E_t-E_{\rm final})/|E_{\rm final}|$")
        d = h["mean_max_direction_change"]
        ok = d > 0
        ax_d.plot(h["sweep"][ok], d[ok], "o-", ms=2.5, lw=0.8, color=COLOR_RELAXED)
        ax_d.plot(
            h["sweep"][ok],
            h["max_max_direction_change"][ok],
            "s--",
            ms=2.0,
            lw=0.7,
            color="0.5",
            label="worst realization",
        )
        _log_axis(ax_d)
        ax_d.set_xlabel("sweep")
        ax_d.set_ylabel(r"max. change of a unit spin")
        ax_d.legend()
        ax_s.errorbar(
            h["sweep"],
            h["mean_total_spin"] / np.sqrt(n),
            h["sem_total_spin"] / np.sqrt(n),
            fmt="o-",
            ms=2.5,
            lw=0.8,
            color=COLOR_RELAXED,
        )
        ax_s.set_xlabel("sweep")
        ax_s.set_ylabel(r"$|\sum_i \mathbf{S}_i|/\sqrt{N}$")
    return fig


def figure_sweep_convergence(data: EnsembleData, rebin: int = 1) -> Figure:
    """``C(r)`` at the recorded sweeps (needs ``snapshots`` in the configuration)."""
    sweeps = [int(s) for s in data.arrays["snap_sweeps"][0]]
    x = data.x_center(rebin)
    cmap = mpl.colormaps["viridis"]
    with _style():
        fig = Figure(figsize=(SINGLE_COLUMN, 2.6), layout="constrained")
        ax = fig.subplots()
        from .analysis import mean_sem

        for index, sweep in enumerate(sweeps):
            mean = mean_sem(data.realization_means("C", rebin=rebin, snapshot=index))[0]
            ax.plot(
                x, mean, "-", lw=1.0, color=cmap(index / max(len(sweeps) - 1, 1)), label=f"{sweep}"
            )
        ax.plot(
            x,
            mean_sem(data.realization_means("C", "final", rebin))[0],
            "k--",
            lw=1.0,
            label=f"final ({data.config.n_sweeps})",
        )
        ax.axhline(0.0, ls=":", lw=0.8, color="0.4")
        ax.set_xlabel(X_LABEL)
        ax.set_ylabel(OBS_LABEL["C"])
        ax.set_xlim(0.0, data.config.x_max)
        ax.legend(title="sweeps", ncol=2, loc="lower right", fontsize=7)
    return fig


def figure_nn(data: EnsembleData, rebin: int = 1) -> Figure:
    """Dimer picture: mutual-NN fraction versus ``exp(-9 pi rho r^3 / 4)`` and the pair decomposition."""
    t = data.nn_table(rebin)
    x = t["x_center"]
    with _style():
        fig = Figure(figsize=(DOUBLE_COLUMN, 2.7), layout="constrained")
        ax_f, ax_c = fig.subplots(1, 2)
        ax_f.errorbar(
            x,
            t["f_mutual"],
            t["sem_f_mutual"],
            fmt="o",
            ms=3,
            color=COLOR_RELAXED,
            label="measured: mutual-NN pairs / all pairs",
        )
        xx = np.linspace(1e-3, data.config.x_max, 300)
        ax_f.plot(
            xx,
            theory.mutual_nn_probability(xx),
            "--",
            color=COLOR_NN,
            label=r"$e^{-9\pi\rho r^{3}/4}$",
        )
        ax_f.plot(
            xx,
            theory.isolated_sphere_probability(xx),
            ":",
            color=COLOR_NN,
            label=r"$e^{-4\pi\rho r^{3}/3}$",
        )
        ax_f.set_xlabel(X_LABEL)
        ax_f.set_ylabel("fraction of pairs that are dimers")
        ax_f.set_xlim(0.0, min(data.config.x_max, 2.5))
        ax_f.legend(loc="upper right")
        for key, label, color in (
            ("C_all", "all pairs", "k"),
            ("C_mutual", "mutual-NN pairs", COLOR_NN),
            ("C_non_mutual", "other pairs", COLOR_RELAXED),
        ):
            ax_c.errorbar(
                x, t[key], t[f"sem_{key}"], fmt="o-", ms=2.5, lw=0.8, color=color, label=label
            )
        ax_c.axhline(0.0, ls=":", lw=0.8, color="0.4")
        ax_c.set_xlabel(X_LABEL)
        ax_c.set_ylabel(OBS_LABEL["C"])
        ax_c.set_xlim(0.0, min(data.config.x_max, 2.5))
        ax_c.legend()
    return fig


# --------------------------------------------------------------------------------------
# Multi-run comparison
# --------------------------------------------------------------------------------------
def figure_compare(
    runs: Mapping[str, EnsembleData],
    observables: tuple[str, ...] = ("C", "Q", "A"),
    rebin: int = 1,
    show_nn: bool = True,
) -> Figure:
    """Overlay of the ensemble-mean curves of several runs (one panel per observable)."""
    first = next(iter(runs.values()))
    with _style():
        fig = Figure(
            figsize=(DOUBLE_COLUMN if len(observables) > 1 else SINGLE_COLUMN, 2.7),
            layout="constrained",
        )
        axes = np.atleast_1d(fig.subplots(1, len(observables)))
        for ax, obs in zip(axes, observables, strict=True):
            ax.axhline(OBS_BASELINE[obs], ls=":", lw=0.9, color="0.4")
            for index, (label, data) in enumerate(runs.items()):
                if not data.has(f"fin_sum_{obs}"):
                    continue
                t = data.table(rebin)
                ax.errorbar(
                    t["x_center"],
                    t[f"mean_{obs}_final"],
                    t[f"sem_{obs}_final"],
                    fmt="o-",
                    ms=2.5,
                    lw=0.8,
                    color=CYCLE[index % len(CYCLE)],
                    label=label,
                )
            if obs == "C" and show_nn:
                xx = np.linspace(1e-3, first.config.x_max, 300)
                ax.plot(
                    xx,
                    -theory.mutual_nn_probability(xx),
                    "--",
                    color="0.5",
                    lw=0.9,
                    label=r"$-e^{-9\pi\rho r^{3}/4}$",
                )
            ax.set_xlabel(X_LABEL)
            ax.set_ylabel(OBS_LABEL[obs])
            ax.set_xlim(0.0, first.config.x_max)
        axes[0].legend(fontsize=7, loc="lower right" if observables[0] == "C" else "best")
    return fig


def figure_protocol(curves_C: pd.DataFrame, curves_Q: pd.DataFrame) -> Figure:
    """Protocol check: curves of all preparations (top) and paired differences to the reference."""
    from .protocols import VARIANTS

    with _style():
        fig = Figure(figsize=(DOUBLE_COLUMN, 4.6), layout="constrained")
        axes = fig.subplots(2, 2, sharex=True)
        for col, (curves, obs) in enumerate(((curves_C, "C"), (curves_Q, "Q"))):
            x = curves["x_center"]
            for index, name in enumerate(VARIANTS):
                color = CYCLE[index]
                axes[0, col].errorbar(
                    x,
                    curves[f"mean_{name}"],
                    curves[f"sem_{name}"],
                    fmt="o-",
                    ms=2.2,
                    lw=0.7,
                    color=color,
                    label=name.replace("_", " "),
                )
                if name != "sequential":
                    axes[1, col].errorbar(
                        x,
                        curves[f"diff_{name}"],
                        curves[f"sem_diff_{name}"],
                        fmt="o-",
                        ms=2.2,
                        lw=0.7,
                        color=color,
                        label=name.replace("_", " "),
                    )
            axes[0, col].set_ylabel(OBS_LABEL[obs])
            axes[1, col].set_ylabel(rf"${obs}(r)-{obs}_{{\rm seq}}(r)$")
            axes[1, col].axhline(0.0, ls=":", lw=0.8, color="0.4")
            axes[1, col].set_xlabel(X_LABEL)
        axes[0, 0].legend(fontsize=7)
        axes[1, 0].legend(fontsize=7)
    return fig
