"""Turn finished runs into tables, JSON summaries, a short text report and figures."""

from __future__ import annotations

import dataclasses
import json
import math
import warnings
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import plotting
from .analysis import (
    EnsembleData,
    aggregate_dimer_map,
    curve_value_at,
    dimer_map_from_states,
    energy_budget_from_states,
    gamma_family_members,
    landmark_difference_table,
    landmark_error_budget,
    landmark_table,
    paired_difference,
    structure_factor_from_states,
    summarize_energy_budget,
    tabulate_dimer_angular,
    tabulate_dimer_field,
    tabulate_dimer_map,
)
from .config import SimulationConfig
from .ensemble import CONFIG_FILENAME, SEEDS_DIRNAME, STATES_DIRNAME, read_config
from .fit import FitResult, fit_ensemble
from .observables import DIMER_ANG_COS_EDGES, DIMER_ANG_X_EDGES
from .protocols import SPIN_CLASSES, VARIANTS, ProtocolData

#: Configuration fields that may differ between a run and the sibling runs of its error budget
#: (``n_sweeps`` for the relaxation check, ``r_cut_nn`` for the cutoff check; the other two only
#: change what is recorded, not the relaxed state).
_BUDGET_FREE_KEYS = ("n_sweeps", "r_cut_nn", "snapshots", "nn_diagnostics")


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


def _signed(value: Any, digits: int = 3) -> str:
    return (
        "n/a"
        if value is None or (isinstance(value, float) and not math.isfinite(value))
        else f"{value:+.{digits}g}"
    )


def _pm(value: Any, err: Any, digits: int = 3) -> str:
    """``value ± err`` (signed value), or ``n/a`` when the value is missing."""
    return "n/a" if _signed(value) == "n/a" else f"{_signed(value, digits)} ± {_fmt(err, 2)}"


def _mark(marks: dict[str, Any], key: str) -> str:
    info = marks.get(key)
    if not info:
        return "n/a"
    return f"{_fmt(info['value'])} ± {_fmt(info['err'], 2)}"


def _val_err(value: Any, err: Any, digits: int = 4) -> str:
    """``value ± err`` (unsigned value), or ``n/a`` when the value is missing."""
    return "n/a" if _fmt(value) == "n/a" else f"{_fmt(value, digits)} ± {_fmt(err, 2)}"


# --------------------------------------------------------------------------------------
# summary.md
# --------------------------------------------------------------------------------------
def _landmark_lines(marks: dict[str, Any]) -> list[str]:
    meta = marks.get("_meta") or {}
    where = (
        "the pair-weighted mean distance `x_mean` of each bin"
        if meta.get("abscissa", "x_center") == "x_mean"
        else "the bin centres"
    )
    return [
        "## Model-independent landmarks (bootstrap errors over realizations)",
        "",
        f"Evaluated on {where} at a bin width of {_fmt(meta.get('bin_width'), 3)} r_nn "
        f"(rebin factor {meta.get('rebin', 'n/a')}, {meta.get('n_bins', 'n/a')} bins); "
        f"errors are the spread over {meta.get('n_boot', 'n/a')} bootstrap resamples of the "
        "realizations (statistical only; see the error budget below).",
        "",
        "| quantity | value |",
        "|---|---|",
        f"| C in the first bin | {_mark(marks, 'C_first_bin')} |",
        f"| first zero of C, x0 | {_mark(marks, 'x_zero_C')} |",
        f"| position of the positive peak of C | {_mark(marks, 'x_peak_C')} |",
        f"| height of the positive peak of C | {_mark(marks, 'C_peak')} |",
        f"| maximum of the transverse channel C_perp = C - P | {_mark(marks, 'Cperp_peak')} "
        f"at x = {_mark(marks, 'x_peak_Cperp')} |",
        f"| longitudinal channel P at the C_perp maximum | {_mark(marks, 'P_at_Cperp_peak')} |",
        f"| Q in the first bin | {_mark(marks, 'Q_first_bin')} |",
        f"| Q crosses 1/3 downward at x | {_mark(marks, 'x_cross_Q_third')} |",
        f"| minimum of Q | {_mark(marks, 'Q_min')} at x = {_mark(marks, 'x_min_Q')} |",
        f"| A in the first bin | {_mark(marks, 'A_first_bin')} |",
        f"| A crosses 1/2 downward at x | {_mark(marks, 'x_cross_A_half')} |",
    ]


def _budget_lines(budget: dict[str, Any]) -> list[str]:
    meta, rows = budget["meta"], budget["rows"]
    rebins, widths = meta["rebins"], meta["bin_widths"]
    parts = ["`stat`: bootstrap over realizations"]
    if len(rebins) > 1:
        parts.append(
            f"`binning`: |rebin {rebins[0]} - rebin {rebins[1]}| (bin widths {_fmt(widths[0], 3)} "
            f"and {_fmt(widths[1], 3)} r_nn)"
        )
    else:
        parts.append("`binning`: not available (a single rebin factor)")
    for name, what, unit, need in (
        ("sweeps", "sweeps", "", "a longer relaxation"),
        ("cutoff", "r_cut", " r_nn", "a larger cutoff"),
    ):
        info = meta.get(name)
        if info:
            parts.append(
                f"`{name}`: paired shift for {what} = {info['run']:g}{unit} instead of "
                f"{info['main']:g}{unit} on {info['n_common']} common seeds (same positions and "
                "initial directions; the sign is run minus main)"
            )
        else:
            parts.append(f"`{name}`: not available (no paired run with {need})")
    parts.append(
        "`total`: square root of the sum of squares, missing components counting as zero; "
        "`rounded`: value ± total at the precision of the total"
    )
    lines = [
        "## Error budget of the landmarks",
        "",
        f"Landmarks at rebin {rebins[0]} on `{meta['abscissa']}`; " + "; ".join(parts) + ".",
        "",
        "| landmark | value | stat | binning | sweeps | cutoff | total | rounded |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['landmark']} | {_fmt(row['value'])} | {_fmt(row['stat'], 2)} | "
            f"{_fmt(row['binning'], 2)} | {_pm(row['sweeps'], row['sweeps_err'])} | "
            f"{_pm(row['cutoff'], row['cutoff_err'])} | {_fmt(row['total'], 2)} | {row['rounded']} |"
        )
    return lines


def _energy_lines(energy: dict[str, Any], n_sweeps: int) -> list[str]:
    lines = [
        "## Energies (final state)",
        "",
        "The ensemble mean of the energy is dominated by the closest pairs: for Poisson positions "
        "the pair energy 2 G s_i s_j / r^3 of the nearest pair of a realization has a heavy tail, so "
        "the mean fluctuates strongly between realizations and is not a converged ensemble number. "
        "The median and the interquartile range are the robust values; only paired differences "
        "between runs on the same seeds are precise (the close pairs cancel).",
        "",
        f"- energy per spin E/N (units G S^2 rho): mean {_fmt(energy['per_spin_mean'])} ± "
        f"{_fmt(energy['per_spin_sem'], 2)}, median {_fmt(energy['per_spin_median'])}, "
        f"interquartile range [{_fmt(energy['per_spin_q25'])}, {_fmt(energy['per_spin_q75'])}], "
        f"extremes [{_fmt(energy['per_spin_min'])}, {_fmt(energy['per_spin_max'])}]",
    ]
    if energy.get("excess_sweep") is not None:
        lines.append(
            f"- relative excess energy (E_t - E_final)/|E_final| at sweep "
            f"{energy['excess_sweep']:.0f} of {n_sweeps}: mean {_fmt(energy['excess_mean'], 3)}, "
            f"median {_fmt(energy['excess_median'], 3)}"
        )
    if energy.get("dmax_last_mean") is not None:
        lines.append(
            f"- largest change of a unit spin in the last sweep: mean "
            f"{_fmt(energy['dmax_last_mean'], 3)}, median {_fmt(energy['dmax_last_median'], 3)}, "
            f"worst realization {_fmt(energy['dmax_last_max'], 3)}"
        )
    return lines


def _fit_lines(fit: dict[str, Any] | None) -> list[str]:
    if not fit or not fit["fit"]["success"]:
        return []
    f = fit["fit"]
    lines = [
        "## Damped-oscillatory fit: phenomenological guide (not used for quantitative statements)",
        "",
        f"C_fit = -exp(-alpha u^3) cos(k u + phi), u = rho^(1/3) r, range x in [{_fmt(f['x_min'])}, "
        f"{_fmt(f['x_max'])}], abscissa `{fit.get('abscissa', 'x_center')}`",
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
        "- note: the form reproduces truncated-interaction curves but not the full one (the "
        "relaxed C(r) has no oscillation); the parameters are only weakly determined. The fit is not "
        "drawn in the figures unless requested; quote the landmarks above instead."
    )
    return lines


def _angular_table_lines(
    rows: list[dict[str, Any]], key: str, with_counts: bool
) -> tuple[list[str], list[tuple[float, float, str]]]:
    """Markdown table of ``key ± sem_key`` per angular cell; returns the lines and the cos bins."""
    x_bins = list(dict.fromkeys((r["x_lo"], r["x_hi"]) for r in rows))
    cos_bins = list(dict.fromkeys((r["cos_lo"], r["cos_hi"], r["region"]) for r in rows))
    lookup = {(r["x_lo"], r["x_hi"], r["cos_lo"], r["cos_hi"]): r for r in rows}
    lines = [
        "| x_ik \\ cos(theta) | "
        + " | ".join(f"[{lo:g}, {hi:g})" for lo, hi, _ in cos_bins)
        + " |",
        "|---|" + "---|" * len(cos_bins),
    ]
    for x_lo, x_hi in x_bins:
        cells = []
        for lo, hi, _ in cos_bins:
            cell = lookup[(x_lo, x_hi, lo, hi)]
            text = _pm(cell.get(key), cell.get(f"sem_{key}"))
            cells.append(f"{text} (n={cell['n']})" if with_counts else text)
        lines.append(f"| [{x_lo:g}, {x_hi:g}) | " + " | ".join(cells) + " |")
    return lines, cos_bins


def _source_sentence(info: dict[str, Any]) -> str:
    if info.get("source") == "states":
        return (
            f"Computed from the {info.get('n_states', info.get('n_real'))} stored states of the "
            "run (its realization files predate this diagnostic)."
        )
    return "Computed from the per-realization records."


def _dimer_lines(info: dict[str, Any]) -> list[str]:
    rows = info["angular"]
    shape = info["shape"]
    lines = [
        "## Dimer-frame correlation map",
        "",
        "For every spin i of a mutual-nearest-neighbour pair (axis a from i to its partner j; "
        f"{info['n_real']} realizations, mean partner distance {_fmt(info['x_partner_mean'], 3)} "
        f"r_nn) and every other spin k within {_fmt(info['r_max_nn'], 2)} r_nn ({info['n_pairs']} "
        "pairs): C = <u_i . u_k> and P = <(u_i . rhat_ik)(u_k . rhat_ik)> on a "
        f"{shape[0]} x {shape[1]} grid in (z = a . r_ik, rho) (`dimer_map.csv`, `dimer_map.pdf`; "
        f"cells with fewer than {info['min_count']} pooled pairs are masked) and in bins of "
        "x_ik = r_ik / r_nn and cos(theta) = a . rhat_ik (`dimer_angular.csv`). The table gives C "
        "± standard error over realizations with the pooled number of pairs n; cos(theta) > 0 is "
        "the side of the partner. " + _source_sentence(info),
        "",
    ]
    table, cos_bins = _angular_table_lines(rows, "C", with_counts=True)
    lines += table
    lines += [
        "",
        "Directions: "
        + "; ".join(f"[{lo:g}, {hi:g}) {region}" for lo, hi, region in cos_bins)
        + ".",
    ]
    return lines


def _dimer_field_lines(info: dict[str, Any], angular: list[dict[str, Any]] | None) -> list[str]:
    lines = [
        "## Dimer-field test",
        "",
        "For every pair (i, k) of the dimer-frame map the field of the two dimer members alone, "
        "h = T_ki S_i + T_kj S_j with T_ab = w_ab (1 - gamma rhat_ab rhat_ab^T), w = 1/r^3 and the "
        f"actual spin vectors S = s u (sigma = {_fmt(info['sigma'], 3)}, gamma = "
        f"{_fmt(info['gamma'], 3)}), predicts the direction sigma h/|h| of the third spin k. "
        "`agree` is the fraction of third spins with u_k . sigma h > 0 (random directions: 0.5, "
        "a perfect prediction: 1), `cos` the mean of u_k . sigma h/|h| (random: 0); both are the "
        f"mean over {info['n_real']} realizations ± standard error, `pooled` counts all pairs "
        "together (`dimer_field.csv`). " + _source_sentence(info),
        "",
        "| x_ik | pairs | agree | agree (pooled) | cos |",
        "|---|---|---|---|---|",
    ]
    rows = info["bins"]
    for index, r in enumerate(rows):
        label = (
            f"all within {r['x_hi']:g} r_nn"
            if index == len(rows) - 1
            else f"[{r['x_lo']:g}, {r['x_hi']:g})"
        )
        lines.append(
            f"| {label} | {r['n']} | {_val_err(r['agree'], r['sem_agree'], 3)} | "
            f"{_fmt(r['agree_pooled'], 3)} | {_val_err(r['cos'], r['sem_cos'], 3)} |"
        )
    if angular and any(r.get("C_pred") is not None for r in angular):
        lines += [
            "",
            "Dimer-only prediction of the angular table: <(sigma h/|h|) . u_i> per cell (column "
            "`C_pred` of `dimer_angular.csv`), to be compared cell by cell with the measured C of "
            "the dimer-frame map above.",
            "",
        ]
        lines += _angular_table_lines(angular, "C_pred", with_counts=False)[0]
    return lines


def _energy_budget_lines(info: dict[str, Any]) -> list[str]:
    meta = info["meta"]

    def stat(key: str, digits: int = 3) -> str:
        s = info[key]
        err = s.get("sem")
        if err is not None and math.isfinite(err):
            text = _val_err(s["mean"], err, digits)
        else:  # a single state has no spread
            text = _fmt(s["mean"], digits)
        if info["n_states"] > 1:
            text += f" (range {_fmt(s['min'], 3)} to {_fmt(s['max'], 3)})"
        return text

    return [
        "## Energy budget of the stored states",
        "",
        f"Energies of the {info['n_states']} stored relaxed states with the run's coupling "
        f"`{meta['coupling']}` (sigma = {_fmt(meta['sigma'], 3)}, gamma = {_fmt(meta['gamma'], 3)}): "
        "every bond within r_cut is compared with the minimum of an isolated pair, "
        f"-f G s_i s_j / r^3 with f = max(1, |gamma - 1|) = {_fmt(meta['two_body_factor'], 3)}. "
        "`energy_budget.csv` has one row per state, `energy_budget.json` the statistics below "
        "(mean ± standard error over the states).",
        "",
        f"- dimer share: energy of the mutual-nearest-neighbour bonds / total energy = "
        f"{stat('dimer_share')}; non-dimer share {stat('nondimer_share')}",
        f"- realized fraction: dimer bonds / sum of their two-body minima = "
        f"{stat('realized_fraction_dimer')}; non-dimer bonds {stat('realized_fraction_nondimer')}",
        f"- frustration index: total energy / sum of the two-body minima of all bonds within r_cut = "
        f"{stat('frustration_index')} (1 if every bond sat at its own minimum; the stored initial "
        f"state gives {stat('frustration_index_initial')})",
        f"- energies per spin (units G S^2 rho): total {stat('energy_total_per_spin', 4)}, dimer "
        f"bonds {stat('energy_dimer_per_spin', 4)}, non-dimer bonds "
        f"{stat('energy_nondimer_per_spin', 4)}, sum of two-body minima "
        f"{stat('minimum_total_per_spin', 4)}",
    ]


def _structure_factor_lines(info: dict[str, Any]) -> list[str]:
    lines = [
        "## Spin structure factor of the stored states",
        "",
        "S(k) = |sum_i u_i exp(i k . r_i)|^2 / N of the unit spins on the reciprocal lattice of the "
        f"box (|n| <= {info['n_max']}, {info['n_vectors']} vectors up to k r_nn = "
        f"{_fmt(info['k_max_rnn'], 3)}), averaged over shells of |k| and over {info['n_states']} "
        "stored states (`structure_factor.csv`, `structure_factor.pdf`). Uncorrelated directions "
        "give S = 1; S < 1 at small k means that long-wavelength fluctuations of the spin texture "
        "are suppressed.",
        "",
        f"- S(0): relaxed {_val_err(info['S0_final'], info['sem_S0_final'])}, initial "
        f"{_val_err(info['S0_initial'], info['sem_S0_initial'])}",
    ]
    if info.get("k_first_rnn") is not None:
        lines.append(
            f"- smallest shell (k r_nn = {_fmt(info['k_first_rnn'], 3)}): relaxed "
            f"{_val_err(info['S_first_final'], info['sem_S_first_final'])}, initial "
            f"{_val_err(info['S_first_initial'], info['sem_S_first_initial'])}"
        )
        lines.append(
            f"- maximum of the relaxed S(k) over the shells with k > 0: "
            f"{_fmt(info['S_max_final'])} at k r_nn = {_fmt(info['k_at_max_rnn'], 3)}"
        )
    return lines


def _positions_text(cfg: Mapping[str, Any] | SimulationConfig) -> str:
    """``Poisson`` or ``hard core <x> r_nn`` (section 3 of the methods) for a configuration."""
    x_min = (
        cfg.min_separation_nn
        if isinstance(cfg, SimulationConfig)
        else cfg.get("min_separation_nn", 0.0)
    )
    if x_min:
        return f"hard core {float(x_min):g} r_nn (Poisson references do not apply)"
    return "Poisson"


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
        f"spin distribution = {cfg['spin_distribution']}, positions = {_positions_text(cfg)}",
        f"- box L = {_fmt(d['box_length'])}, density = {_fmt(d['density'])}, "
        f"r_nn = {_fmt(d['r_nn'])}, r_cut = {_fmt(d['r_cut_nn'])} r_nn "
        f"({_fmt(d['neighbors_per_particle'], 3)} neighbours per particle)",
        f"- relaxation: {cfg['n_sweeps']} sweeps ({cfg['update_order']} order, {cfg['protocol']})",
        f"- radial bins of the tables and figures: {summary['rebin_n_bins']} on [0, {cfg['x_max']}] "
        f"r_nn (stored {cfg['n_bins']}, rebin factor {summary['rebin']}); channels C = P + C_perp "
        "in `ensemble_table.csv` (columns mean_P_*, mean_Cperp_*)",
        "",
        *_landmark_lines(marks),
        "",
    ]
    if summary.get("error_budget"):
        lines += [*_budget_lines(summary["error_budget"]), ""]
    lines += ["## Global diagnostics (mean ± standard error)", ""]
    for key, stat in sc.items():
        lines.append(f"- {key}: {_fmt(stat['mean'])} ± {_fmt(stat['sem'], 2)}")
    if summary.get("energy"):
        lines += ["", *_energy_lines(summary["energy"], cfg["n_sweeps"])]
    if summary.get("energy_budget"):
        lines += ["", *_energy_budget_lines(summary["energy_budget"])]
    if summary.get("dimer_map"):
        lines += ["", *_dimer_lines(summary["dimer_map"])]
    if summary.get("dimer_field"):
        angular = (summary.get("dimer_map") or {}).get("angular")
        lines += ["", *_dimer_field_lines(summary["dimer_field"], angular)]
    if summary.get("structure_factor"):
        lines += ["", *_structure_factor_lines(summary["structure_factor"])]
    fit_lines = _fit_lines(summary.get("fit"))
    if fit_lines:
        lines += ["", *fit_lines]
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------------------
# Error budget helpers
# --------------------------------------------------------------------------------------
def _budget_key(config: SimulationConfig) -> dict[str, Any]:
    key = config.compat_key()
    for name in _BUDGET_FREE_KEYS:
        key.pop(name, None)
    return key


def find_budget_runs(run_dir: str | Path) -> dict[str, Path]:
    """Sibling run directories usable for the error budget of ``run_dir``.

    A sibling qualifies when its configuration equals that of ``run_dir`` (same system, seeds and
    binning) except for a longer relaxation at the same cutoff (``"sweeps_run"``) or a larger cutoff
    at the same number of sweeps (``"cutoff_run"``); snapshots and nearest-neighbour diagnostics do
    not matter. The longest relaxation and the largest cutoff win; directories with fewer than two
    finished realizations are ignored. Returns an empty mapping when nothing qualifies.
    """
    run_dir = Path(run_dir)
    main = read_config(run_dir)
    key = _budget_key(main)
    best: dict[str, tuple[float, Path]] = {}
    siblings = sorted(run_dir.parent.iterdir()) if run_dir.parent.is_dir() else []
    for sibling in siblings:
        if sibling.resolve() == run_dir.resolve() or not (sibling / CONFIG_FILENAME).is_file():
            continue
        try:
            meta = json.loads((sibling / CONFIG_FILENAME).read_text())
            cfg = SimulationConfig.from_dict(meta["config"])
        except (ValueError, KeyError, TypeError):
            continue
        if meta.get("kind", "ensemble") != "ensemble" or _budget_key(cfg) != key:
            continue
        if len(list((sibling / SEEDS_DIRNAME).glob("seed_*.npz"))) < 2:
            continue
        same_cut = cfg.r_cut_nn_resolved == main.r_cut_nn_resolved
        if same_cut and cfg.n_sweeps > max(main.n_sweeps, best.get("sweeps_run", (0,))[0]):
            best["sweeps_run"] = (float(cfg.n_sweeps), sibling)
        larger_cut = cfg.r_cut_nn_resolved > max(
            main.r_cut_nn_resolved, best.get("cutoff_run", (0.0,))[0]
        )
        if cfg.n_sweeps == main.n_sweeps and larger_cut:
            best["cutoff_run"] = (cfg.r_cut_nn_resolved, sibling)
    return {name: path for name, (_, path) in best.items()}


def _load_run(run: str | Path | EnsembleData | None, role: str) -> EnsembleData | None:
    """``EnsembleData`` from a path (partial runs allowed), passed through, or ``None``."""
    if run is None or isinstance(run, EnsembleData):
        return run
    path = Path(run)
    if not (path / CONFIG_FILENAME).is_file():
        warnings.warn(f"{role}: {path} is not a run directory, skipped", stacklevel=3)
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return EnsembleData.load(path)


def _budget_rebins(n_bins: int, rebin: int) -> tuple[int, ...]:
    return (rebin, 2 * rebin) if n_bins % (2 * rebin) == 0 else (rebin,)


# --------------------------------------------------------------------------------------
# Dimer-frame map and structure factor summaries
# --------------------------------------------------------------------------------------
def _has_states(run_dir: Path) -> bool:
    return any((Path(run_dir) / STATES_DIRNAME).glob("seed_*.npz"))


def _dimer_grid(dmap: dict[str, Any]) -> dict[str, Any]:
    """Grid parameters of the stored map (for ``summary.json``'s ``derived`` section)."""
    return {
        "r_max_nn": dmap["r_max"],
        "shape": list(np.asarray(dmap["count"]).shape),
        "x_edges": DIMER_ANG_X_EDGES.tolist(),
        "cos_edges": DIMER_ANG_COS_EDGES.tolist(),
        "min_count": dmap["min_count"],
    }


def _dimer_summary(
    dmap: dict[str, Any], angular: pd.DataFrame, source: str, n_states: int | None
) -> dict[str, Any]:
    return {
        **_dimer_grid(dmap),
        "n_real": dmap["n_real"],
        "n_pairs": dmap["n_pairs"],
        "x_partner_mean": dmap["x_partner_mean"],
        "source": source,
        "n_states": n_states,
        "angular": angular.to_dict("records"),
    }


def _dimer_field_summary(field: pd.DataFrame, source: str, n_states: int | None) -> dict[str, Any]:
    return {
        **field.attrs["meta"],
        "source": source,
        "n_states": n_states,
        "bins": field.to_dict("records"),
    }


def _structure_factor_summary(table: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = dict(table.attrs["meta"])
    zero = table.iloc[0]  # the k = 0 row
    for state in ("final", "initial"):
        out[f"S0_{state}"] = float(zero[f"S_{state}"])
        out[f"sem_S0_{state}"] = float(zero[f"sem_S_{state}"])
    shells = table[table["k_rnn_mean"] > 0]
    if len(shells):
        first = shells.iloc[0]
        out["k_first_rnn"] = float(first["k_rnn_mean"])
        for state in ("final", "initial"):
            out[f"S_first_{state}"] = float(first[f"S_{state}"])
            out[f"sem_S_first_{state}"] = float(first[f"sem_S_{state}"])
        best = shells.loc[shells["S_final"].idxmax()]
        out["S_max_final"] = float(best["S_final"])
        out["k_at_max_rnn"] = float(best["k_rnn_mean"])
    return out


# --------------------------------------------------------------------------------------
# Per-run analysis
# --------------------------------------------------------------------------------------
def analyze_run(
    run_dir: str | Path,
    out_dir: str | Path | None = None,
    rebin: int = 1,
    n_boot: int = 200,
    fit: bool = True,
    fit_range: tuple[float, float] | None = None,
    plots: bool = True,
    formats: tuple[str, ...] = ("pdf",),
    landmark_rebin: int = 1,
    show_fit: bool = False,
    sweeps_run: str | Path | EnsembleData | None = None,
    cutoff_run: str | Path | EnsembleData | None = None,
    error_budget: bool = True,
    structure_factor: bool = True,
    structure_factor_n_max: int = 12,
    energy_budget: bool = True,
    dimer_map_states: bool = True,
    max_states: int | None = None,
) -> dict[str, Path]:
    """Analyse one finished run; write everything into ``<run_dir>/analysis`` (or ``out_dir``).

    Tables and figures use ``rebin``. The quoted landmarks use ``landmark_rebin`` (default 1, the
    stored bins) on the pair-weighted mean distance of each bin. ``error_budget.csv`` combines their
    bootstrap error with the binning difference (``landmark_rebin`` versus twice that) and, when a
    paired run is available, with the shifts against a longer relaxation (``sweeps_run``) and a
    larger cutoff (``cutoff_run``); runs not given are searched among the sibling directories of
    ``run_dir`` (:func:`find_budget_runs`), and missing ones simply leave their column empty. The
    damped-oscillatory fit is written to ``fit.json`` but drawn only with ``show_fit=True``.

    Runs with ``nn_diagnostics`` also get the dimer-frame correlation map (``dimer_map.csv``,
    ``dimer_angular.csv`` with the dimer-only prediction ``C_pred``, ``dimer_map.pdf``) and the
    dimer-field test (``dimer_field.csv``). Runs whose realization files predate these
    diagnostics get them from their stored states when ``dimer_map_states`` is true. Runs with
    stored states also get the spin structure factor (``structure_factor.csv`` and ``.pdf``,
    reciprocal vectors up to ``|n| <= structure_factor_n_max``) unless ``structure_factor=False``,
    and the energy budget with the run's ``(sigma, gamma)`` (``energy_budget.csv`` and ``.json``:
    dimer share, realized fraction, frustration index) unless ``energy_budget=False``;
    ``max_states`` limits the states used by these three. Everything is skipped silently when the
    data are absent (older runs without states).

    Returns the written files keyed by a short name.
    """
    run_dir = Path(run_dir)
    data = EnsembleData.load(run_dir)
    out = Path(out_dir) if out_dir is not None else run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    cfg = data.config
    has_states = _has_states(run_dir)

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
    dimer = angular = field = None
    dimer_source = field_source = None
    states_arrays: dict[str, np.ndarray] | None = None
    n_states: int | None = None

    def from_states() -> dict[str, np.ndarray]:
        nonlocal states_arrays, n_states
        if states_arrays is None:
            states_arrays = dimer_map_from_states(run_dir, max_states=max_states)
            n_states = int(states_arrays["seed"].shape[0])
        return states_arrays

    if data.has("dimer_map_cnt"):
        dimer_arrays, dimer_source = data.arrays, "records"
    elif dimer_map_states and has_states:
        dimer_arrays, dimer_source = from_states(), "states"
    else:
        dimer_arrays = None
    if dimer_arrays is not None:
        dimer = aggregate_dimer_map(dimer_arrays)
        tabulate_dimer_map(dimer_arrays).to_csv(out / "dimer_map.csv", index=False)
        angular = tabulate_dimer_angular(dimer_arrays)
        if "dimer_field_cnt" in dimer_arrays:
            field_arrays, field_source = dimer_arrays, dimer_source
        elif dimer_map_states and has_states:  # records with the map but without the field test
            field_arrays, field_source = from_states(), "states"
        else:
            field_arrays = None
        if field_arrays is not None:
            if "C_pred" not in angular:
                predicted = tabulate_dimer_angular(field_arrays)
                angular["C_pred"] = predicted["C_pred"]
                angular["sem_C_pred"] = predicted["sem_C_pred"]
            field = tabulate_dimer_field(field_arrays)
            field.to_csv(out / "dimer_field.csv", index=False)
            written["dimer_field"] = out / "dimer_field.csv"
        angular.to_csv(out / "dimer_angular.csv", index=False)
        written["dimer_map"] = out / "dimer_map.csv"
        written["dimer_angular"] = out / "dimer_angular.csv"
    sk = None
    if structure_factor and has_states:
        sk = structure_factor_from_states(
            run_dir, n_max=structure_factor_n_max, max_states=max_states
        )
        sk.to_csv(out / "structure_factor.csv", index=False)
        written["structure_factor"] = out / "structure_factor.csv"
    budget_info = None
    if energy_budget and has_states:
        budget_frame = energy_budget_from_states(run_dir, max_states=max_states)
        budget_frame.to_csv(out / "energy_budget.csv", index=False)
        written["energy_budget"] = out / "energy_budget.csv"
        budget_info = summarize_energy_budget(budget_frame)
        written["energy_budget_json"] = write_json(
            out / "energy_budget.json", {**budget_info, "rows": budget_frame.to_dict("records")}
        )

    marks = data.landmarks(landmark_rebin, n_boot)
    budget = None
    if error_budget:
        found = find_budget_runs(run_dir) if sweeps_run is None or cutoff_run is None else {}
        sweeps = _load_run(
            sweeps_run if sweeps_run is not None else found.get("sweeps_run"), "sweeps_run"
        )
        cutoff = _load_run(
            cutoff_run if cutoff_run is not None else found.get("cutoff_run"), "cutoff_run"
        )
        budget = landmark_error_budget(
            data, sweeps, cutoff, _budget_rebins(cfg.n_bins, landmark_rebin), n_boot
        )
        budget.to_csv(out / "error_budget.csv", index=False)
        written["error_budget"] = out / "error_budget.csv"

    fit_info = fit_result = None
    if fit:
        fit_info = fit_ensemble(data, rebin, fit_range, n_boot)
        if fit_info["fit"]["success"]:
            fit_result = FitResult(**fit_info["fit"])
        written["fit"] = write_json(out / "fit.json", fit_info)

    summary = {
        "run_dir": str(run_dir),
        "n_realizations": data.n_real,
        "rebin": rebin,
        "rebin_n_bins": cfg.n_bins // rebin,
        "landmark_rebin": landmark_rebin,
        "config": cfg.to_dict(),
        "derived": {
            "box_length": cfg.box,
            "density": cfg.density_effective,
            "r_nn": cfg.r_nn,
            "r_cut": cfg.r_cut,
            "r_cut_nn": cfg.r_cut_nn_resolved,
            "neighbors_per_particle": cfg.neighbors_per_particle,
            "dimer_map": _dimer_grid(dimer) if dimer else None,
        },
        "scalars": data.scalar_summary(),
        "energy": data.energy_summary(),
        "energy_budget": budget_info,
        "landmarks": marks,
        "error_budget": (
            {"meta": budget.attrs["meta"], "rows": budget.to_dict("records")}
            if budget is not None
            else None
        ),
        "dimer_map": _dimer_summary(dimer, angular, dimer_source, n_states) if dimer else None,
        "dimer_field": (
            _dimer_field_summary(field, field_source, n_states) if field is not None else None
        ),
        "structure_factor": _structure_factor_summary(sk) if sk is not None else None,
        "fit": fit_info,
    }
    written["summary_json"] = write_json(out / "summary.json", summary)
    (out / "summary.md").write_text(summary_markdown(to_jsonable(summary)))
    written["summary_md"] = out / "summary.md"

    if plots:
        shown_fit = fit_result if show_fit else None
        figures = {
            "C_of_r": plotting.figure_C(data, rebin, shown_fit),
            "decomposition": plotting.figure_decomposition(data, rebin)
            if data.has_observable("Cperp")
            else None,
            "Q_A_of_r": plotting.figure_QA(data, rebin) if data.has_observable("A") else None,
            "overview": plotting.figure_overview(data, rebin, shown_fit),
            "histories": plotting.figure_histories(data),
            "sweep_convergence": plotting.figure_sweep_convergence(data, rebin)
            if data.has("snap_sweeps")
            else None,
            "nn_diagnostics": plotting.figure_nn(data, rebin) if has_nn else None,
            "dimer_map": plotting.figure_dimer_map(dimer) if dimer else None,
            "structure_factor": (plotting.figure_structure_factor(sk) if sk is not None else None),
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
    """Summarize a protocol check (see :mod:`grspin.protocols`).

    Writes ``protocol_summary.csv`` (one row per variant: energies, the energy differences relative
    to the total and to the non-dimer energy, overlaps overall and split into dimer and interstitial
    spins, curve differences, acceptance), its digest ``protocol_summary.md``, the curves
    ``protocol_curves_C.csv`` / ``protocol_curves_Q.csv`` and the figure.
    """
    run_dir = Path(run_dir)
    data = ProtocolData.load(run_dir)
    out = Path(out_dir) if out_dir is not None else run_dir / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    summary = data.summary(rebin)
    summary.to_csv(out / "protocol_summary.csv", index=False)
    written["protocol_summary"] = out / "protocol_summary.csv"
    (out / "protocol_summary.md").write_text(protocol_markdown(run_dir, data, summary))
    written["protocol_summary_md"] = out / "protocol_summary.md"
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


# --------------------------------------------------------------------------------------
# Comparisons
# --------------------------------------------------------------------------------------
_SHIFT_KEYS = (
    "x_zero_C",
    "x_peak_C",
    "C_peak",
    "x_cross_Q_third",
    "Q_min",
    "x_peak_Cperp",
    "Cperp_peak",
)


#: Landmarks tabulated against ``gamma`` for the channel family (``gamma_family.csv``).
_GAMMA_FAMILY_KEYS = ("x_zero_C", "C_peak", "x_peak_C", "Cperp_peak", "x_peak_Cperp", "Q_min")
#: Distance (units of ``r_nn``) at which the sign of ``C`` is tabulated for the channel family: in
#: the positive shell of the GR ensemble (``x_0 = 0.85``, peak at 1.2-1.3 r_nn).
GAMMA_FAMILY_X_SIGN = 1.2


def gamma_family_table(
    runs: Mapping[str, EnsembleData],
    landmarks: pd.DataFrame,
    rebin: int = 1,
    x_sign: float = GAMMA_FAMILY_X_SIGN,
) -> pd.DataFrame | None:
    """Landmarks against ``gamma`` for the channel family, one row per run in order of ``gamma``.

    Columns: ``run``, ``coupling``, ``sigma``, ``gamma``, ``n_real``, the landmarks
    :data:`_GAMMA_FAMILY_KEYS` with their bootstrap errors (taken from ``landmarks``, the table
    of :func:`grspin.analysis.landmark_table`), ``C_at_x_sign`` and ``sem_C_at_x_sign`` (the mean
    curve interpolated at ``x_sign`` on ``x_mean``, SEM over realizations), ``sign_C_at_x_sign``
    (``+1``, ``-1`` or ``0`` when the value is within one SEM of zero) and ``x_sign``. ``None``
    when the runs are not the channel family (:func:`grspin.analysis.gamma_family_members`).
    """
    order = gamma_family_members(dict(runs))
    if order is None:
        return None
    marks = landmarks.set_index("run")
    rows = []
    for label in order:
        data = runs[label]
        sigma, gamma = data.config.sigma_gamma
        value, sem = curve_value_at(data, "C", x_sign, rebin)
        if not math.isfinite(value):
            sign = float("nan")
        else:
            sign = (
                0.0
                if abs(value) <= (sem if math.isfinite(sem) else 0.0)
                else math.copysign(1.0, value)
            )
        row: dict[str, Any] = {
            "run": label,
            "coupling": data.config.coupling,
            "sigma": sigma,
            "gamma": gamma,
            "n_real": data.n_real,
        }
        for key in _GAMMA_FAMILY_KEYS:
            row[key] = float(marks.loc[label, key]) if key in marks.columns else float("nan")
            err = f"{key}_err"
            row[err] = float(marks.loc[label, err]) if err in marks.columns else float("nan")
        row.update(
            {
                "C_at_x_sign": value,
                "sem_C_at_x_sign": sem,
                "sign_C_at_x_sign": sign,
                "x_sign": x_sign,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def gamma_family_markdown_lines(frame: pd.DataFrame) -> list[str]:
    """Markdown section of :func:`gamma_family_table` for ``comparison_summary.md``."""
    x_sign = float(frame["x_sign"].iloc[0])
    lines = [
        "## Channel family: landmarks against gamma",
        "",
        "sigma = +1 throughout; gamma - 1 is the weight of the (antiferromagnetic) longitudinal "
        "channel relative to the (ferromagnetic) transverse one (`gamma_family.csv`, "
        "`gamma_family.pdf`). Landmarks with bootstrap errors on x_mean; "
        f"C({x_sign:g} r_nn) is the mean curve interpolated at {x_sign:g} r_nn ± SEM over "
        "realizations and `sign` its sign (0: within one SEM of zero).",
        "",
        "| run | gamma | x_0 | C_peak | x_peak | Cperp_max | x(Cperp_max) | Q_min | "
        f"C({x_sign:g} r_nn) | sign |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in frame.iterrows():
        cells = [_val_err(r[k], r[f"{k}_err"]) for k in _GAMMA_FAMILY_KEYS]
        sign = r["sign_C_at_x_sign"]
        sign_text = (
            "n/a" if not math.isfinite(sign) else "0" if sign == 0 else ("+" if sign > 0 else "-")
        )
        lines.append(
            f"| {r['run']} | {r['gamma']:g} | " + " | ".join(cells) + f" | "
            f"{_val_err(r['C_at_x_sign'], r['sem_C_at_x_sign'])} | {sign_text} |"
        )
    return lines


def comparison_markdown(
    runs: Mapping[str, EnsembleData],
    differences: pd.DataFrame,
    observables: tuple[str, ...],
    rebin: int,
    landmark_rebin: int,
    extra_sections: Sequence[Sequence[str]] = (),
) -> str:
    """Digest of :func:`compare_runs`: which members are paired with the reference, landmark shifts
    and any ``extra_sections`` (lists of Markdown lines, e.g. the channel-family table)."""
    labels = list(runs)
    reference = safe_name(labels[0])
    lines = [
        f"# Comparison with the reference `{labels[0]}`",
        "",
        f"- bin-wise differences `<kind>_<obs>_<run>_minus_{reference}.csv` for "
        f"{', '.join(observables) or 'no common observable'} at rebin {rebin}: `paired` = the "
        "difference is formed per common seed (identical positions and initial directions, most "
        "sampling noise cancels); `unpaired` = difference of the two ensemble means over all "
        "realizations of both runs, error sqrt(sem_a^2 + sem_b^2).",
        f"- landmarks (`comparison_landmarks.csv`) and their shifts "
        "(`comparison_landmark_differences.csv`, run minus reference, paired where possible) at "
        f"rebin {landmark_rebin} on the pair-weighted mean distance `x_mean`.",
        "",
        "| run | N | boundary | coupling | spins | positions | r_cut / r_nn | sweeps | "
        "realizations | comparison |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for label in labels:
        data, c = runs[label], runs[label].config
        if label == labels[0]:
            kind = "reference"
        else:
            row = differences[differences["run"] == label].iloc[0]
            kind = (
                f"paired ({int(row['n_common'])} common seeds)"
                if bool(row["paired"])
                else "unpaired (all realizations)"
            )
        positions = f"hard core {c.min_separation_nn:g} r_nn" if c.min_separation_nn else "Poisson"
        lines.append(
            f"| {label} | {c.n_particles} | {c.boundary} | {c.coupling} | {c.spin_distribution} | "
            f"{positions} | {c.r_cut_nn_resolved:g} | {c.n_sweeps} | {data.n_real} | {kind} |"
        )
    keys = [k for k in _SHIFT_KEYS if f"d_{k}" in differences.columns]
    if keys and len(differences):
        lines += [
            "",
            "## Landmark shifts (run minus reference, bootstrap errors)",
            "",
            "| run | " + " | ".join(keys) + " |",
            "|---|" + "---|" * len(keys),
        ]
        for _, row in differences.iterrows():
            cells = [_pm(float(row[f"d_{k}"]), float(row[f"d_{k}_err"])) for k in keys]
            lines.append(f"| {row['run']} | " + " | ".join(cells) + " |")
    for section in extra_sections:
        lines += ["", *section]
    return "\n".join(lines) + "\n"


def compare_runs(
    runs: Mapping[str, EnsembleData],
    out_dir: str | Path,
    observables: tuple[str, ...] = ("C", "Q", "A"),
    rebin: int = 1,
    n_boot: int = 200,
    formats: tuple[str, ...] = ("pdf",),
    landmark_rebin: int = 1,
) -> dict[str, Path]:
    """Compare several runs: landmark table, overlay figure and bin-wise differences to the first.

    ``observables`` may include the channels ``P`` and ``Cperp``. A member that shares the system
    and seeds with the reference is compared *paired* (``paired_<obs>_<run>_minus_<ref>.csv``),
    any other member *unpaired* over all realizations of both runs (``unpaired_...csv``);
    ``comparison_summary.md`` states which applies, and ``comparison_landmark_differences.csv``
    holds the landmark shifts with (paired) bootstrap errors at ``landmark_rebin``.

    When the runs form the channel (``gamma``) family (all ``sigma = +1``, several ``gamma``; the
    plan's ``cmp_channels``) the comparison also writes ``gamma_family.csv`` (landmarks and the
    sign of ``C`` at :data:`GAMMA_FAMILY_X_SIGN` against ``gamma``, :func:`gamma_family_table`),
    ``gamma_family.<fmt>`` (``C``, ``P``, ``Cperp`` of the members in order of ``gamma``) and a
    section of ``comparison_summary.md``.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    labels = list(runs)
    written: dict[str, Path] = {}
    landmarks = landmark_table(dict(runs), landmark_rebin, n_boot)
    landmarks.to_csv(out / "comparison_landmarks.csv", index=False)
    written["comparison_landmarks"] = out / "comparison_landmarks.csv"
    differences = landmark_difference_table(dict(runs), landmark_rebin, n_boot)
    differences.to_csv(out / "comparison_landmark_differences.csv", index=False)
    written["comparison_landmark_differences"] = out / "comparison_landmark_differences.csv"
    usable = tuple(o for o in observables if all(d.has_observable(o) for d in runs.values()))
    if usable:
        fig = plotting.figure_compare(runs, usable, rebin)
        for suffix in formats:
            written[f"comparison.{suffix}"] = plotting.save_figure(
                fig, out / f"comparison.{suffix}"
            )
    extra: list[list[str]] = []
    family = gamma_family_table(runs, landmarks, landmark_rebin)
    if family is not None:
        family.to_csv(out / "gamma_family.csv", index=False)
        written["gamma_family"] = out / "gamma_family.csv"
        extra.append(gamma_family_markdown_lines(family))
        if all(d.has_observable("Cperp") for d in runs.values()):
            fig = plotting.figure_gamma_family(runs, rebin)
            for suffix in formats:
                written[f"gamma_family.{suffix}"] = plotting.save_figure(
                    fig, out / f"gamma_family.{suffix}"
                )
    reference = runs[labels[0]]
    for label in labels[1:]:
        for obs in usable:
            try:
                frame = paired_difference(reference, runs[label], obs, rebin)
            except ValueError:
                continue  # different binning: the overlay figure still applies
            kind = "paired" if bool(frame["paired"].iloc[0]) else "unpaired"
            path = out / f"{kind}_{obs}_{safe_name(label)}_minus_{safe_name(labels[0])}.csv"
            frame.to_csv(path, index=False)
            written[f"{kind}_{obs}_{label}"] = path
    (out / "comparison_summary.md").write_text(
        comparison_markdown(runs, differences, usable, rebin, landmark_rebin, extra)
    )
    written["comparison_summary"] = out / "comparison_summary.md"
    return written


# --------------------------------------------------------------------------------------
# Protocol-check digest
# --------------------------------------------------------------------------------------
def protocol_markdown(run_dir: str | Path, data: ProtocolData, summary: pd.DataFrame) -> str:
    """Digest of :meth:`grspin.protocols.ProtocolData.summary` as Markdown tables."""
    cfg = data.config
    a = cfg.anneal
    rows = summary.set_index("variant")

    def cell(name: str, key: str, digits: int = 3, signed: bool = False) -> str:
        if key not in rows.columns:
            return "n/a"
        value, err = float(rows.loc[name, key]), rows.get(f"sem_{key}")
        err = float(err.loc[name]) if err is not None else float("nan")
        return _pm(value, err, digits) if signed else _val_err(value, err, digits)

    fraction = float(rows["dimer_spin_fraction"].iloc[0]) if "dimer_spin_fraction" in rows else None
    lines = [
        f"# Protocol check `{run_dir}`",
        "",
        f"- realizations: **{data.n_real}**; N = {cfg.n_particles}, r_cut = "
        f"{cfg.r_cut_nn_resolved:g} r_nn, {cfg.n_sweeps} sweeps; annealing from T = {a.t_start:g} "
        f"to {a.t_end:g} (units G S^2 rho) over {a.n_temperatures} temperatures with "
        f"{a.sweeps_per_temperature} Metropolis sweeps each, followed by the same quench",
        "- variants on identical positions and spin lengths: `sequential` (production), "
        "`random_order` (fresh permutation every sweep), `other_init` (independent initial "
        "directions), `anneal`; all differences are paired by seed (mean ± standard error over "
        "realizations)",
    ]
    if fraction is not None and math.isfinite(fraction):
        lines.append(
            f"- dimer spins (members of a mutual nearest-neighbour pair): {_fmt(fraction, 3)} of "
            "the spins; the remaining spins are called interstitial"
        )
    else:
        lines.append(
            "- the files predate the dimer / interstitial split (no dimer-bond energies, no split "
            "overlaps)"
        )
    lines += [
        "",
        "## Energies (differences to the production state; negative = lower energy)",
        "",
        "| variant | E/N | dE / |E_seq| (%) | dE / |E_nondimer,seq| (%) | dE_dimer / |E_dimer,seq| (%) "
        "| dimer energy share |",
        "|---|---|---|---|---|---|",
    ]
    for name in VARIANTS:
        if name not in rows.index:
            continue
        lines.append(
            f"| {name} | {cell(name, 'energy_per_spin', 4)} | "
            f"{cell(name, 'delta_energy_percent', 3, True)} | "
            f"{cell(name, 'delta_energy_nondimer_percent', 3, True)} | "
            f"{cell(name, 'delta_energy_dimer_percent', 3, True)} | "
            f"{cell(name, 'dimer_energy_share', 3)} |"
        )
    lines += [
        "",
        "dE is the same total energy difference in the second and third column; the third divides "
        "it by the energy of the non-dimer bonds of the production state (E_nondimer = E - sum over "
        "mutual-NN pairs of their pair energy), which is the part a protocol can change. The fourth "
        "column is the relative change of the dimer-bond energy itself.",
        "",
        "## Overlaps with the production state",
        "",
        "q1 = <u . u'>, q2 = <(u . u')^2> over the spins of nonzero length (random axes give "
        "q2 = 1/3, identical states up to a global flip give 1), overall and split into the dimer "
        "and the interstitial spins.",
        "",
        "| variant | q1 | q2 | " + " | ".join(f"q1 {c} | q2 {c}" for c in SPIN_CLASSES) + " |",
        "|---|---|---|" + "---|---|" * len(SPIN_CLASSES),
    ]
    for name in VARIANTS:
        if name not in rows.index:
            continue
        cells = [cell(name, "overlap_q1", 3, True), cell(name, "overlap_q2", 3)]
        for c in SPIN_CLASSES:
            cells += [cell(name, f"overlap_q1_{c}", 3, True), cell(name, f"overlap_q2_{c}", 3)]
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Correlation curves and landmarks",
        "",
        "| variant | max |dC| | max |dC| / sem | max |dQ| | C_peak | x_peak_C | Q_min | "
        "acceptance first / final |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name in VARIANTS:
        if name not in rows.index:
            continue
        r = rows.loc[name]
        acceptance = (
            f"{_row_value(r, 'anneal_acceptance_first')} / {_row_value(r, 'anneal_acceptance_final')}"
            if name == "anneal"
            else "-"
        )
        lines.append(
            f"| {name} | {_row_value(r, 'max_abs_diff_C')} | "
            f"{_row_value(r, 'max_abs_diff_C_over_sem')} | {_row_value(r, 'max_abs_diff_Q')} | "
            f"{_row_value(r, 'C_peak', 4)} | {_row_value(r, 'x_peak_C', 4)} | "
            f"{_row_value(r, 'Q_min', 4)} | {acceptance} |"
        )
    return "\n".join(lines) + "\n"


def _row_value(row: pd.Series, key: str, digits: int = 3) -> str:
    """One scalar of a summary row as text (``n/a`` when the column does not exist)."""
    return _fmt(float(row[key]), digits) if key in row.index else "n/a"


# --------------------------------------------------------------------------------------
# Structure factor versus cutoff (several runs)
# --------------------------------------------------------------------------------------
def _structure_factor_row(label: str, cfg: SimulationConfig, table: pd.DataFrame) -> dict[str, Any]:
    zero = table.iloc[0]
    shells = table[table["k_rnn_mean"] > 0].reset_index(drop=True)
    row: dict[str, Any] = {
        "label": label,
        "n_particles": cfg.n_particles,
        "r_cut_nn": cfg.r_cut_nn_resolved,
        "box_rnn": cfg.box / cfg.r_nn,
        "k_min_rnn": 2.0 * math.pi * cfg.r_nn / cfg.box,
        "n_states": table.attrs["meta"]["n_states"],
        "S0": float(zero["S_final"]),
        "sem_S0": float(zero["sem_S_final"]),
        "S0_initial": float(zero["S_initial"]),
    }
    for m in (1, 2, 3):
        has = len(shells) >= m
        row[f"k{m}_rnn"] = float(shells.loc[m - 1, "k_rnn_mean"]) if has else float("nan")
        row[f"S{m}"] = float(shells.loc[m - 1, "S_final"]) if has else float("nan")
        row[f"sem_S{m}"] = float(shells.loc[m - 1, "sem_S_final"]) if has else float("nan")
    if len(shells):
        best = shells.loc[shells["S_final"].idxmax()]
        row["k_max_rnn"], row["S_max"] = float(best["k_rnn_mean"]), float(best["S_final"])
        row["sem_S_max"] = float(best["sem_S_final"])
    else:
        row["k_max_rnn"] = row["S_max"] = row["sem_S_max"] = float("nan")
    row["n_max"] = table.attrs["meta"]["n_max"]
    row["n_shells"] = table.attrs["meta"]["n_shells"]
    return row


def structure_factor_comparison_markdown(frame: pd.DataFrame) -> str:
    """Digest of :func:`structure_factor_comparison` (one table row per run)."""
    lines = [
        "# Spin structure factor versus cutoff and system size",
        "",
        "S(k) = |sum_i u_i exp(i k . r_i)|^2 / N of the relaxed unit spins, averaged over shells of "
        "|k| on the reciprocal lattice of each box and over the stored states (± standard error "
        "over the states; uncorrelated directions give 1). S(0) and the first three shells of each "
        "run (`structure_factor_vs_cutoff.csv`, `structure_factor_comparison.pdf`). The smallest "
        "wavevector 2 pi / L depends on N, so the shells of different box sizes sit at different "
        "k r_nn.",
        "",
        "| run | N | r_cut / r_nn | L / r_nn | states | S(0) | S(k_1) at k_1 r_nn | S(k_2) at k_2 r_nn "
        "| S(k_3) at k_3 r_nn | max S(k>0) at k r_nn |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in frame.iterrows():
        shells = [
            f"{_val_err(r[f'S{m}'], r[f'sem_S{m}'])} at {_fmt(r[f'k{m}_rnn'], 3)}"
            for m in (1, 2, 3)
        ]
        lines.append(
            f"| {r['label']} | {int(r['n_particles'])} | {r['r_cut_nn']:g} | {_fmt(r['box_rnn'], 3)} | "
            f"{int(r['n_states'])} | {_val_err(r['S0'], r['sem_S0'])} | "
            + " | ".join(shells)
            + f" | {_val_err(r['S_max'], r['sem_S_max'])} at {_fmt(r['k_max_rnn'], 3)} |"
        )
    return "\n".join(lines) + "\n"


def structure_factor_comparison(
    run_dirs: Sequence[str | Path],
    out_dir: str | Path,
    labels: Sequence[str] | None = None,
    n_max: int = 12,
    n_shells: int = 24,
    max_states: int | None = None,
    formats: tuple[str, ...] = ("pdf",),
) -> dict[str, Path]:
    """Shell-averaged ``S(k)`` of several runs from their stored states, side by side.

    For every run (labels default to the directory names) :func:`grspin.analysis.structure_factor_from_states`
    is evaluated (``k = 0`` first, then ``n_shells`` shells up to ``|n| <= n_max``) and one row of
    ``structure_factor_vs_cutoff.csv`` records ``label``, ``n_particles``, ``r_cut_nn``,
    ``box_rnn``, ``k_min_rnn``, ``n_states``, ``S0`` and ``sem_S0`` (and ``S0_initial``), the first
    three shells ``k{m}_rnn, S{m}, sem_S{m}`` and the maximum of the relaxed ``S(k > 0)`` with its
    ``k`` (``k_max_rnn, S_max, sem_S_max``). ``structure_factor_comparison.pdf`` (or ``formats``)
    draws ``S(k)`` of all runs on one logarithmic ``k r_nn`` axis with the line ``S = 1``, and
    ``structure_factor_comparison.md`` is the Markdown digest. Runs without stored states are
    skipped with a warning; at least one is required. Returns the written files.
    """
    paths = [Path(r) for r in run_dirs]
    names = list(labels) if labels is not None else [p.name for p in paths]
    if len(names) != len(paths):
        raise ValueError("labels must have one entry per run")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tables: dict[str, pd.DataFrame] = {}
    rows = []
    for label, path in zip(names, paths, strict=True):
        if not (path / CONFIG_FILENAME).is_file() or not _has_states(path):
            warnings.warn(f"{label}: {path} has no stored states, skipped", stacklevel=2)
            continue
        table = structure_factor_from_states(
            path, n_shells=n_shells, n_max=n_max, max_states=max_states
        )
        tables[label] = table
        rows.append(_structure_factor_row(label, read_config(path), table))
    if not tables:
        raise FileNotFoundError("none of the runs has stored states")
    frame = pd.DataFrame(rows)
    written: dict[str, Path] = {}
    frame.to_csv(out / "structure_factor_vs_cutoff.csv", index=False)
    written["structure_factor_vs_cutoff"] = out / "structure_factor_vs_cutoff.csv"
    fig = plotting.figure_structure_factor_comparison(tables)
    for suffix in formats:
        written[f"structure_factor_comparison.{suffix}"] = plotting.save_figure(
            fig, out / f"structure_factor_comparison.{suffix}"
        )
    (out / "structure_factor_comparison.md").write_text(structure_factor_comparison_markdown(frame))
    written["structure_factor_comparison_md"] = out / "structure_factor_comparison.md"
    return written
