"""Turn finished runs into tables, JSON summaries, a short text report and figures."""

from __future__ import annotations

import dataclasses
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from . import plotting
from .analysis import EnsembleData, landmark_table, paired_difference
from .fit import FitResult, fit_ensemble
from .protocols import ProtocolData


def to_jsonable(obj: Any) -> Any:
    """Recursively convert numpy/pandas/dataclass objects to plain JSON types (NaN -> ``null``)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, Mapping):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return to_jsonable(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if not math.isfinite(value) else value
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, Path):
        return str(obj)
    return obj


def safe_name(label: str) -> str:
    """File-name friendly version of a run label."""
    cleaned = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in label)
    while "__" in cleaned:
        cleaned = cleaned.replace("__", "_")
    return cleaned.strip("_") or "run"


def write_json(path: Path, obj: Any) -> Path:
    path.write_text(json.dumps(to_jsonable(obj), indent=2) + "\n")
    return path


def _fmt(value: Any, digits: int = 4) -> str:
    return (
        "n/a"
        if value is None or (isinstance(value, float) and not math.isfinite(value))
        else f"{value:.{digits}g}"
    )


def _mark(marks: dict[str, Any], key: str) -> str:
    info = marks.get(key)
    if not info:
        return "n/a"
    return f"{_fmt(info['value'])} ± {_fmt(info['err'], 2)}"


def summary_markdown(summary: dict[str, Any]) -> str:
    """Human-readable digest of :func:`analyze_run`'s ``summary.json``."""
    cfg, d, marks, sc = (
        summary["config"],
        summary["derived"],
        summary["landmarks"],
        summary["scalars"],
    )
    lines = [
        f"# Analysis of `{summary['run_dir']}`",
        "",
        f"- realizations analysed: **{summary['n_realizations']}** (of {cfg['n_realizations']} planned)",
        f"- N = {cfg['n_particles']}, boundary = {cfg['boundary']}, coupling = {cfg['coupling']}, "
        f"spin distribution = {cfg['spin_distribution']}",
        f"- box L = {_fmt(d['box_length'])}, density = {_fmt(d['density'])}, "
        f"r_nn = {_fmt(d['r_nn'])}, r_cut = {_fmt(d['r_cut_nn'])} r_nn "
        f"({_fmt(d['neighbors_per_particle'], 3)} neighbours per particle)",
        f"- relaxation: {cfg['n_sweeps']} sweeps ({cfg['update_order']} order, {cfg['protocol']})",
        f"- radial bins: {summary['rebin_n_bins']} on [0, {cfg['x_max']}] r_nn "
        f"(stored {cfg['n_bins']}, rebin factor {summary['rebin']})",
        "",
        "## Model-independent landmarks (bootstrap errors over realizations)",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| C in the first bin | {_mark(marks, 'C_first_bin')} |",
        f"| first zero of C, x0 | {_mark(marks, 'x_zero_C')} |",
        f"| position of the positive peak of C | {_mark(marks, 'x_peak_C')} |",
        f"| height of the positive peak of C | {_mark(marks, 'C_peak')} |",
        f"| Q in the first bin | {_mark(marks, 'Q_first_bin')} |",
        f"| Q crosses 1/3 downward at x | {_mark(marks, 'x_cross_Q_third')} |",
        f"| minimum of Q | {_mark(marks, 'Q_min')} at x = {_mark(marks, 'x_min_Q')} |",
        f"| A in the first bin | {_mark(marks, 'A_first_bin')} |",
        f"| A crosses 1/2 downward at x | {_mark(marks, 'x_cross_A_half')} |",
        "",
        "## Global diagnostics (mean ± standard error)",
        "",
    ]
    for key, stat in sc.items():
        lines.append(f"- {key}: {_fmt(stat['mean'])} ± {_fmt(stat['sem'], 2)}")
    fit = summary.get("fit")
    if fit and fit["fit"]["success"]:
        f = fit["fit"]
        lines += [
            "",
            "## Damped-oscillatory fit (phenomenological)",
            "",
            f"C_fit = -exp(-alpha u^3) cos(k u + phi), u = rho^(1/3) r, range x in [{_fmt(f['x_min'])}, {_fmt(f['x_max'])}]",
            "",
            f"- alpha = {_fmt(f['alpha'])}, k = {_fmt(f['k'])}, phi = {_fmt(f['phi'])}",
            f"- chi2/dof = {_fmt(f['chi2'] / max(f['dof'], 1))}, rms residual = {_fmt(f['rms_residual'])}, "
            f"max |residual| = {_fmt(f['max_abs_residual'])}",
        ]
        for name in ("alpha", "k", "phi"):
            boot = fit.get(f"{name}_bootstrap")
            if boot:
                lines.append(
                    f"- bootstrap {name}: median {_fmt(boot['median'])} (16-84%: {_fmt(boot['p16'])} … {_fmt(boot['p84'])})"
                )
        lines.append(
            "- note: the fit parameters are only weakly determined; quote the landmarks above instead."
        )
    return "\n".join(lines) + "\n"


def analyze_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    rebin: int = 1,
    n_boot: int = 200,
    fit: bool = True,
    fit_range: tuple[float, float] | None = None,
    plots: bool = True,
    formats: tuple[str, ...] = ("pdf",),
) -> dict[str, Path]:
    """Analyse one finished run; write everything into ``<run_dir>/analysis`` (or ``out_dir``).

    Returns the written files keyed by a short name.
    """
    run_dir = Path(run_dir)
    data = EnsembleData.load(run_dir)
    out = Path(out_dir) if out_dir is not None else run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    table = data.table(rebin)
    table.to_csv(out / "ensemble_table.csv", index=False)
    written["ensemble_table"] = out / "ensemble_table.csv"
    data.histories().to_csv(out / "histories.csv", index=False)
    written["histories"] = out / "histories.csv"

    if data.has("snap_sweeps"):
        data.sweep_convergence(rebin).to_csv(out / "sweep_convergence.csv", index=False)
        written["sweep_convergence"] = out / "sweep_convergence.csv"
    has_nn = data.has("nn_cnt_mutual")
    if has_nn:
        for name, frame in (
            ("nn_pairs", data.nn_table(rebin)),
            ("nn_ranks", data.nn_rank_table()),
            ("nn_distance", data.nn_distance_table()),
            ("small_r", data.small_r_table()),
        ):
            frame.to_csv(out / f"{name}.csv", index=False)
            written[name] = out / f"{name}.csv"

    marks = data.landmarks(rebin, n_boot)
    fit_info = fit_result = None
    if fit:
        fit_info = fit_ensemble(data, rebin, fit_range, n_boot)
        if fit_info["fit"]["success"]:
            fit_result = FitResult(**fit_info["fit"])
        written["fit"] = write_json(out / "fit.json", fit_info)

    cfg = data.config
    summary = {
        "run_dir": str(run_dir),
        "n_realizations": data.n_real,
        "rebin": rebin,
        "rebin_n_bins": cfg.n_bins // rebin,
        "config": cfg.to_dict(),
        "derived": {
            "box_length": cfg.box,
            "density": cfg.density_effective,
            "r_nn": cfg.r_nn,
            "r_cut": cfg.r_cut,
            "r_cut_nn": cfg.r_cut_nn_resolved,
            "neighbors_per_particle": cfg.neighbors_per_particle,
        },
        "scalars": data.scalar_summary(),
        "landmarks": marks,
        "fit": fit_info,
    }
    written["summary_json"] = write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(summary_markdown(to_jsonable(summary)))
    written["summary_md"] = out / "summary.md"

    if plots:
        figures = {
            "C_of_r": plotting.figure_C(data, rebin, fit_result),
            "Q_A_of_r": plotting.figure_QA(data, rebin) if data.has("fin_sum_A") else None,
            "overview": plotting.figure_overview(data, rebin, fit_result),
            "histories": plotting.figure_histories(data),
            "sweep_convergence": plotting.figure_sweep_convergence(data, rebin)
            if data.has("snap_sweeps")
            else None,
            "nn_diagnostics": plotting.figure_nn(data, rebin) if has_nn else None,
        }
        for name, fig in figures.items():
            if fig is None:
                continue
            for suffix in formats:
                written[f"{name}.{suffix}"] = plotting.save_figure(fig, out / f"{name}.{suffix}")
    return written


def analyze_protocol(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    rebin: int = 1,
    plots: bool = True,
    formats: tuple[str, ...] = ("pdf",),
) -> dict[str, Path]:
    """Summarize a protocol check (see :mod:`grspin.protocols`)."""
    run_dir = Path(run_dir)
    data = ProtocolData.load(run_dir)
    out = Path(out_dir) if out_dir is not None else run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    summary = data.summary(rebin)
    summary.to_csv(out / "protocol_summary.csv", index=False)
    written["protocol_summary"] = out / "protocol_summary.csv"
    curves_c, curves_q = data.curves("C", rebin), data.curves("Q", rebin)
    curves_c.to_csv(out / "protocol_curves_C.csv", index=False)
    curves_q.to_csv(out / "protocol_curves_Q.csv", index=False)
    written["protocol_curves_C"], written["protocol_curves_Q"] = (
        out / "protocol_curves_C.csv",
        out / "protocol_curves_Q.csv",
    )
    if plots:
        fig = plotting.figure_protocol(curves_c, curves_q)
        for suffix in formats:
            written[f"protocol.{suffix}"] = plotting.save_figure(fig, out / f"protocol.{suffix}")
    return written


def compare_runs(
    runs: Mapping[str, EnsembleData],
    out_dir: str | Path,
    observables: tuple[str, ...] = ("C", "Q", "A"),
    rebin: int = 1,
    n_boot: int = 200,
    formats: tuple[str, ...] = ("pdf",),
) -> dict[str, Path]:
    """Compare several runs: landmark table, overlay figure and paired differences to the first run."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = list(runs)
    written: dict[str, Path] = {}
    landmark_table(dict(runs), rebin, n_boot).to_csv(out / "comparison_landmarks.csv", index=False)
    written["comparison_landmarks"] = out / "comparison_landmarks.csv"
    usable = tuple(o for o in observables if all(d.has(f"fin_sum_{o}") for d in runs.values()))
    if usable:
        fig = plotting.figure_compare(runs, usable, rebin)
        for suffix in formats:
            written[f"comparison.{suffix}"] = plotting.save_figure(
                fig, out / f"comparison.{suffix}"
            )
    reference = runs[labels[0]]
    for label in labels[1:]:
        for obs in usable:
            try:
                frame = paired_difference(reference, runs[label], obs, rebin)
            except ValueError:
                continue  # not paired (different seeds/binning): the overlay figure still applies
            path = out / f"paired_{obs}_{safe_name(label)}_minus_{safe_name(labels[0])}.csv"
            frame.to_csv(path, index=False)
            written[f"paired_{obs}_{label}"] = path
    return written
