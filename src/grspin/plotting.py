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
from typing import Any

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
COLOR_P = "#e6550d"  # longitudinal channel (orange)
COLOR_PERP = "#31a354"  # transverse channel (green)
CYCLE = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b", "#17becf", "#7f7f7f"]

X_LABEL = r"$r/\bar{r}_{\rm nn}$"
OBS_LABEL = {
    "C": r"$C(r)=\langle\hat{S}_i\cdot\hat{S}_j\rangle$",
    "Q": r"$Q(r)=\langle\cos^2\varphi\rangle$",
    "A": r"$A(r)=\langle(\hat{S}\cdot\hat{r})^2\rangle$",
    "P": r"$P(r)=\langle(\hat{S}_i\cdot\hat{r})(\hat{S}_j\cdot\hat{r})\rangle$",
    "Cperp": r"$C_\perp(r)=C-P$",
    "B": r"$B(r)=C-3P$",
}
OBS_BASELINE = {"C": 0.0, "Q": 1.0 / 3.0, "A": 1.0 / 3.0, "P": 0.0, "Cperp": 0.0, "B": 0.0}
#: Channel decomposition ``C = P + C_perp``: observable, colour, marker, legend label.
CHANNELS = (
    ("C", COLOR_RELAXED, "o", r"$C$"),
    ("P", COLOR_P, "s", r"$P$ (longitudinal)"),
    ("Cperp", COLOR_PERP, "^", r"$C_\perp=C-P$ (transverse)"),
)


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
    show_nn: bool = False,
    fit: FitResult | None = None,
) -> None:
    """``C(r)``, optionally with the exact dimer-fraction reference and a damped-oscillatory fit.

    ``show_nn`` draws ``-f_m(x) = -exp(-9 pi rho r^3 / 4)``, the probability that a pair at
    separation ``r`` is a mutual nearest-neighbour pair (a dimer with ``C = -1``): the exact
    ``r -> 0`` limit of ``C(r)`` for Poisson positions. The fit is drawn only when passed explicitly;
    it is a phenomenological guide and not used for quantitative statements.
    """
    draw_observable(ax, data, "C", rebin, show_initial=show_initial)
    xx = np.linspace(1e-3, data.config.x_max, 400)
    if show_nn:
        ax.plot(
            xx,
            -theory.mutual_nn_probability(xx),
            "--",
            color=COLOR_NN,
            lw=1.0,
            label=r"dimer fraction, $-f_m(x)$",
        )
    if fit is not None and fit.success:
        ax.plot(xx, fit.curve(xx), "-", color=COLOR_FIT, lw=1.1, label="damped-oscillatory fit")
    t = data.table(rebin)
    top = max(0.35, float(np.nanmax(t["mean_C_final"] + t["sem_C_final"].fillna(0))) + 0.08)
    ax.set_ylim(-1.05, top)


def draw_Q(ax, data: EnsembleData, rebin: int = 1, show_initial: bool = True) -> None:
    draw_observable(ax, data, "Q", rebin, show_initial=show_initial)
    ax.set_ylim(0.0, 1.05)


def _ylim_from_data(
    t: pd.DataFrame,
    observables: tuple[str, ...],
    include: tuple[float, ...] = (),
    show_initial: bool = True,
    pad: float = 0.06,
) -> tuple[float, float]:
    """Axis limits enclosing mean +- SEM of the given observables (and the values ``include``)."""
    lows, highs = list(include), list(include)
    states = ("final", "initial") if show_initial else ("final",)
    for obs in observables:
        for state in states:
            if f"mean_{obs}_{state}" not in t:
                continue
            mean = t[f"mean_{obs}_{state}"].to_numpy(float)
            sem = t[f"sem_{obs}_{state}"].fillna(0.0).to_numpy(float)
            if np.isfinite(mean).any():
                lows.append(float(np.nanmin(mean - sem)))
                highs.append(float(np.nanmax(mean + sem)))
    if not lows:
        return -1.0, 1.0
    lo, hi = min(lows), max(highs)
    span = (hi - lo) or 1.0
    return lo - pad * span, hi + pad * span


def draw_A(ax, data: EnsembleData, rebin: int = 1, show_initial: bool = True) -> None:
    """Single-spin axis locking; the limits follow the data and include the isotropic ``1/3``."""
    draw_observable(ax, data, "A", rebin, show_initial=show_initial)
    ax.set_ylim(*_ylim_from_data(data.table(rebin), ("A",), (1.0 / 3.0,), show_initial))


def draw_B(ax, data: EnsembleData, rebin: int = 1, show_initial: bool = True) -> None:
    """Tensor (pair-energy) correlation; a perfect dimer has ``B = 2`` (dashed reference).

    The limits follow the data (the dipolar control has ``B < 0``), always including zero.
    """
    draw_observable(ax, data, "B", rebin, show_initial=show_initial)
    ax.axhline(2.0, ls="--", lw=0.8, color="0.6")
    ax.set_ylim(*_ylim_from_data(data.table(rebin), ("B",), (0.0,), show_initial))


def draw_decomposition(
    ax,
    data: EnsembleData,
    rebin: int = 1,
    x_min: float = 0.0,
    show_initial: bool = False,
    legend: bool = True,
) -> None:
    """Channel decomposition ``C = P + C_perp`` of the relaxed state (mean +- SEM, zero line).

    ``P`` is the longitudinal (bond-projected) channel, ``C_perp = C - P`` the transverse one. With
    ``x_min > 0`` only the tail ``x >= x_min`` is shown with limits that follow those bins.
    """
    t = data.table(rebin)
    sel = (t["x_center"] >= x_min).to_numpy()
    ax.axhline(0.0, ls=":", lw=0.9, color="0.4")
    for obs, color, marker, label in CHANNELS:
        if f"mean_{obs}_final" not in t:
            continue
        if show_initial and f"mean_{obs}_initial" in t:
            ax.errorbar(
                t["x_center"][sel],
                t[f"mean_{obs}_initial"][sel],
                t[f"sem_{obs}_initial"][sel],
                fmt=marker,
                ms=2.0,
                lw=0.7,
                color=COLOR_INITIAL,
                mfc="none",
            )
        ax.errorbar(
            t["x_center"][sel],
            t[f"mean_{obs}_final"][sel],
            t[f"sem_{obs}_final"][sel],
            fmt=f"{marker}-",
            ms=2.8,
            lw=0.9,
            color=color,
            label=label,
        )
    ax.set_xlabel(X_LABEL)
    ax.set_ylabel("pair correlation")
    ax.set_xlim(x_min, data.config.x_max)
    present = tuple(obs for obs, *_ in CHANNELS if f"mean_{obs}_final" in t)
    ax.set_ylim(*_ylim_from_data(t[sel], present, (0.0,), show_initial))
    if legend:
        ax.legend(loc="best", fontsize=7)


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
        draw_A(ax_a, data, rebin, show_initial=data.has_observable("A", "initial"))
    return fig


def figure_decomposition(data: EnsembleData, rebin: int = 1, tail: bool = True) -> Figure:
    """``C``, ``P`` and ``C_perp`` versus ``x`` (top) and a zoom into the tail ``x >= x_max/2``.

    The zoom shows that the anisotropic channels ``P < 0`` and ``C_perp > 0`` persist where their
    sum ``C`` has nearly cancelled.
    """
    with _style():
        fig = Figure(figsize=(SINGLE_COLUMN, 4.2 if tail else 2.6), layout="constrained")
        if tail:
            ax_full, ax_tail = fig.subplots(2, 1)
            draw_decomposition(ax_full, data, rebin)
            draw_decomposition(ax_tail, data, rebin, x_min=0.5 * data.config.x_max, legend=False)
            ax_tail.set_title("tail", loc="right", pad=3)
        else:
            draw_decomposition(fig.subplots(), data, rebin)
    return fig


def figure_overview(data: EnsembleData, rebin: int = 1, fit: FitResult | None = None) -> Figure:
    """Double-column overview: ``C``, the channel decomposition, ``Q``; ``A``, ``B`` and the tail.

    Panels: (a) ``C(r)`` (with the fit only if ``fit`` is passed), (b) ``C = P + C_perp``,
    (c) ``Q(r)``, (d) ``A(r)``, (e) ``B(r)`` and (f) the decomposition in the tail ``x >= x_max/2``.
    Panels that need the extra observables are left blank when the run did not record them.
    """
    with _style():
        fig = Figure(figsize=(DOUBLE_COLUMN, 4.8), layout="constrained")
        axes = fig.subplots(2, 3)
        draw_C(axes[0, 0], data, rebin, fit=fit)
        axes[0, 0].legend(loc="lower right")
        draw_Q(axes[0, 2], data, rebin)
        axes[0, 2].legend(loc="upper right")
        if data.has_observable("Cperp"):
            draw_decomposition(axes[0, 1], data, rebin)
            draw_decomposition(axes[1, 2], data, rebin, x_min=0.5 * data.config.x_max, legend=False)
        else:
            axes[0, 1].axis("off")
            axes[1, 2].axis("off")
        if data.has_observable("A"):
            draw_A(axes[1, 0], data, rebin)
        else:
            axes[1, 0].axis("off")
        if data.has_observable("B"):
            draw_B(axes[1, 1], data, rebin)
        else:
            axes[1, 1].axis("off")
        for label, ax in zip("abcdef", axes.ravel(), strict=True):
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
# Dimer-frame map and structure factor
# --------------------------------------------------------------------------------------
def _mirrored(cells: np.ndarray) -> np.ndarray:
    """``(n_z, n_rho)`` cell values as an image over ``z`` (columns) and ``±rho`` (rows)."""
    return np.concatenate([cells[:, ::-1], cells], axis=1).T


def _draw_dimer(ax, x_partner: float) -> None:
    """Spin ``i`` at the origin and its partner ``j`` on the positive axis."""
    if np.isfinite(x_partner):
        ax.plot([0.0, x_partner], [0.0, 0.0], "-", color="k", lw=1.4, solid_capstyle="round")
        ax.plot(x_partner, 0.0, "o", mfc="w", mec="k", mew=0.9, ms=5, zorder=4)
        ax.annotate("j", (x_partner, 0.0), xytext=(3, 4), textcoords="offset points", fontsize=7)
    ax.plot(0.0, 0.0, "o", color="k", mec="w", mew=0.6, ms=5, zorder=4)
    ax.annotate("i", (0.0, 0.0), xytext=(-8, 4), textcoords="offset points", fontsize=7)


def figure_dimer_map(dmap: Mapping[str, Any], vlim: float = 0.6) -> Figure:
    """Dimer-frame maps of ``C = <u_i . u_k>`` (left) and ``P = <(u_i . r)(u_k . r)>`` (right).

    The cylindrical map ``(z, rho)`` of :meth:`grspin.analysis.EnsembleData.dimer_map` is mirrored
    in ``rho`` so that the dimer axis runs horizontally through the centre; spin ``i`` sits at the
    origin and its partner ``j`` at the mean partner distance on the positive axis. Diverging
    colours on a symmetric scale ``±vlim`` (red: parallel to ``u_i``, blue: antiparallel; white:
    uncorrelated); cells with fewer than ``min_count`` pooled pairs are grey.
    """
    r_max = float(dmap["r_max"])
    extent = (-r_max, r_max, -r_max, r_max)
    cmap = mpl.colormaps["RdBu_r"].with_extremes(bad="0.82")
    panels = (
        ("C", r"$\langle\hat{S}_i\cdot\hat{S}_k\rangle$"),
        ("P", r"$\langle(\hat{S}_i\cdot\hat{r}_{ik})(\hat{S}_k\cdot\hat{r}_{ik})\rangle$"),
    )
    with _style():
        fig = Figure(figsize=(DOUBLE_COLUMN, 3.4), layout="constrained")
        axes = fig.subplots(1, 2, sharey=True)
        for ax, (obs, label), panel in zip(axes, panels, "ab", strict=True):
            image = np.ma.masked_invalid(_mirrored(np.asarray(dmap[obs], dtype=float)))
            im = ax.imshow(
                image,
                origin="lower",
                extent=extent,
                cmap=cmap,
                vmin=-vlim,
                vmax=vlim,
                interpolation="nearest",
            )
            _draw_dimer(ax, float(dmap.get("x_partner_mean", np.nan)))
            ax.set_title(f"({panel}) {label}", loc="left", pad=3)
            ax.set_xlabel(r"$z/\bar{r}_{\rm nn}$ (along the dimer axis $i\to j$)")
            fig.colorbar(im, ax=ax, shrink=0.85, pad=0.03)
        axes[0].set_ylabel(r"$\rho/\bar{r}_{\rm nn}$ (transverse, mirrored)")
        axes[0].text(
            0.02,
            0.98,
            f"{dmap.get('n_real', 0)} realizations, {dmap.get('n_pairs', 0)} pairs",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=7,
        )
    return fig


def figure_structure_factor(table: pd.DataFrame) -> Figure:
    """Spherically averaged ``S(k)`` of the unit spins, initial and relaxed (mean ± SEM).

    ``table`` is the frame of :func:`grspin.analysis.structure_factor_from_states` (errors over
    the states); the first point is ``k = 0``.
    """
    with _style():
        fig = Figure(figsize=(SINGLE_COLUMN, 2.6), layout="constrained")
        ax = fig.subplots()
        ax.axhline(1.0, ls=":", lw=0.9, color="0.4", label="uncorrelated directions")
        for state, color, label in (
            ("initial", COLOR_INITIAL, "initial (random)"),
            ("final", COLOR_RELAXED, "relaxed"),
        ):
            ax.errorbar(
                table["k_rnn_mean"],
                table[f"S_{state}"],
                table[f"sem_S_{state}"].fillna(0.0),
                fmt="o-",
                ms=2.5,
                lw=0.8,
                color=color,
                label=label,
            )
        ax.set_xlabel(r"$k\,\bar{r}_{\rm nn}$")
        ax.set_ylabel(r"$S(k)=|\sum_i \hat{S}_i\,e^{i\mathbf{k}\cdot\mathbf{r}_i}|^2/N$")
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="upper right")
    return fig


def figure_structure_factor_comparison(tables: Mapping[str, pd.DataFrame]) -> Figure:
    """Relaxed ``S(k)`` of several runs (e.g. different cutoffs) on one logarithmic ``k`` axis.

    ``tables`` maps a label to the frame of :func:`grspin.analysis.structure_factor_from_states`;
    the shells with ``k > 0`` are drawn with their errors over the states, the dotted line marks
    uncorrelated directions (``S = 1``) and the legend quotes ``S(0)`` of every run, which a
    logarithmic axis cannot show.
    """
    with _style():
        fig = Figure(figsize=(SINGLE_COLUMN, 2.8), layout="constrained")
        ax = fig.subplots()
        ax.axhline(1.0, ls=":", lw=0.9, color="0.4", label="uncorrelated directions")
        for index, (label, table) in enumerate(tables.items()):
            shells = table[table["k_rnn_mean"] > 0]
            zero = table.iloc[0]
            s0 = f"{zero['S_final']:.2f}"
            if np.isfinite(zero["sem_S_final"]):
                s0 += f" ± {zero['sem_S_final']:.2f}"
            ax.errorbar(
                shells["k_rnn_mean"],
                shells["S_final"],
                shells["sem_S_final"].fillna(0.0),
                fmt="o-",
                ms=2.5,
                lw=0.8,
                color=CYCLE[index % len(CYCLE)],
                label=f"{label}: $S(0)$ = {s0}",
            )
        ax.set_xscale("log")
        ax.set_xlabel(r"$k\,\bar{r}_{\rm nn}$")
        ax.set_ylabel(r"$S(k)$ of the relaxed unit spins")
        ax.set_ylim(bottom=0.0)
        ax.legend(loc="best", fontsize=6.5)
    return fig


# --------------------------------------------------------------------------------------
# Multi-run comparison
# --------------------------------------------------------------------------------------
def figure_compare(
    runs: Mapping[str, EnsembleData],
    observables: tuple[str, ...] = ("C", "Q", "A"),
    rebin: int = 1,
    show_nn: bool = False,
) -> Figure:
    """Overlay of the ensemble-mean curves of several runs (one panel per observable).

    ``observables`` may contain any entry of :data:`grspin.analysis.OBSERVABLES`, including the
    channels ``P`` and ``Cperp``; runs lacking an observable are skipped in that panel.
    """
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
                if not data.has_observable(obs):
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
                    label=r"dimer fraction, $-f_m(x)$",
                )
            ax.set_xlabel(X_LABEL)
            ax.set_ylabel(OBS_LABEL[obs])
            ax.set_xlim(0.0, first.config.x_max)
        axes[0].legend(fontsize=7, loc="lower right" if observables[0] == "C" else "best")
    return fig


def figure_gamma_family(runs: Mapping[str, EnsembleData], rebin: int = 1) -> Figure:
    """``C``, ``P`` and ``Cperp`` of the channel (``gamma``) family, one panel each.

    The runs are drawn in order of increasing ``gamma`` (the longitudinal weight ``gamma - 1``
    grows from left to right in the legend), labelled ``gamma = <value>`` followed by the given
    label: ``transverse_only`` (1), ``gr`` (3), ``gamma12`` (12), ``longitudinal_only`` (100).
    Runs with ``sigma < 0`` are not part of the family and raise ``ValueError``.
    """
    from .analysis import gamma_family_members

    order = gamma_family_members(dict(runs))
    if order is None:
        raise ValueError("the runs do not form the channel family (sigma = +1, several gamma)")
    ordered = {
        rf"$\gamma = {runs[label].config.sigma_gamma[1]:g}$: {label}": runs[label]
        for label in order
    }
    fig = figure_compare(ordered, ("C", "P", "Cperp"), rebin)
    with _style():
        fig.axes[0].legend(fontsize=6.5, loc="lower right")
        fig.suptitle(
            r"channel family: $U_{ij} = -(G/r^3)[S_{\perp,i}\cdot S_{\perp,j}"
            r" - (\gamma-1) S_{r,i} S_{r,j}]$",
            fontsize=8,
        )
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
