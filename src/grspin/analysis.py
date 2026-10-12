"""Ensemble analysis: re-binning, statistics over realizations, bootstrap, landmarks, diagnostics.

The statistics follow the definitions of the manuscript: per realization the binned mean of an
observable is formed over all pairs in the bin; the ensemble mean is the average of these means over
realizations and the error bar is the standard error over realizations
(``std(ddof=1) / sqrt(n)``, ignoring realizations without pairs in a bin). Since the runner stores
sums and counts per bin, the data can be re-binned exactly (any integer divisor of ``n_bins``).

Observables. ``C``, ``Q``, ``A``, ``P`` and ``B`` are stored as per-bin sums (see
:mod:`grspin.observables`). The transverse channel ``Cperp = C - P`` is *derived* from the stored
sums of ``C`` and ``P`` (same pairs, same counts), so it is available for every run that recorded
``P``. With it the GR pair energy splits into a longitudinal antiferromagnetic channel (``P``,
weight 2) and a transverse ferromagnetic channel (``Cperp``, weight 1): ``B = C - 3P = Cperp - 2P``.

Landmarks are evaluated on the pair-weighted mean distance ``x_mean`` of each bin (pooled sums of
``r / r_nn`` over counts) at the stored bin width by default; the bin centres remain available
(``abscissa="x_center"``).

Diagnostics of the relaxed states are aggregated here as well: the **dimer-frame correlation
map** with the **dimer-field test** (``dimer_map_*`` / ``dimer_ang_*`` / ``dimer_field_*`` record
keys of runs with ``nn_diagnostics``, see :func:`grspin.observables.dimer_frame_map`;
:func:`aggregate_dimer_map`, :func:`tabulate_dimer_angular`, :func:`tabulate_dimer_field`, or
:func:`dimer_map_from_states` for older runs), the **spin structure factor** of the stored states
(:func:`structure_factor_from_states`) and the **energy budget** of the stored states
(:func:`energy_budget_from_states`: dimer share, realized fraction and the frustration index
``E / sum of two-body minima`` with the run's ``(sigma, gamma)``).
"""

from __future__ import annotations

import math
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import theory
from .config import SimulationConfig
from .ensemble import (
    SEEDS_DIRNAME,
    STATES_DIRNAME,
    load_record,
    read_config,
    seed_path,
    state_path,
)
from .neighbors import build_tree
from .nn import NN_HIST_EDGES_X, mutual_nn_partners
from .observables import (
    DIMER_ANG_COS_EDGES,
    DIMER_ANG_X_EDGES,
    DIMER_MAP_R_NN,
    dimer_frame_map,
    energy_budget,
    reciprocal_lattice_vectors,
    spin_structure_factor,
)

OBSERVABLES = ("C", "Q", "A", "P", "Cperp", "B")
#: Derived observables as linear combinations of stored sums (they share the counts of ``C``).
_DERIVED: dict[str, tuple[tuple[str, float], ...]] = {"Cperp": (("C", 1.0), ("P", -1.0))}
_COUNT_OF = {"C": "C", "Q": "Q", "A": "C", "P": "C", "Cperp": "C", "B": "C"}
_STATE_PREFIX = {"initial": "init", "final": "fin"}
#: Observables that enter the landmarks (``C`` and ``Q`` are required, the others are optional).
LANDMARK_OBSERVABLES = ("C", "Q", "A", "P", "Cperp")
#: Abscissae on which curves and landmarks can be evaluated.
ABSCISSAE = ("x_mean", "x_center")
#: Configuration fields that fix the positions, spin lengths and initial directions of a seed
#: (:func:`grspin.initial.generate_initial_state`). Runs that agree on them and share seeds are
#: *paired*: coupling, cutoff, boundary condition, number of sweeps and protocol may differ.
PAIRING_KEYS = (
    "n_particles",
    "density",
    "box_length",
    "spin_distribution",
    "spin_scale",
    "lognormal_sigma",
    "min_separation_nn",  # hard-core positions differ from the Poisson ones of the same seed
)
#: Names of the angular bins of the dimer frame (:data:`grspin.observables.DIMER_ANG_COS_EDGES`),
#: from the far side of ``i`` (``cos(theta) = -1``) to the direction beyond its partner (``+1``).
DIMER_ANG_REGIONS = (
    "collinear, far side of i",
    "oblique, far side",
    "near transverse, far side",
    "transverse",
    "near transverse, partner side",
    "oblique, partner side",
    "collinear, beyond the partner",
)


# --------------------------------------------------------------------------------------
# Generic statistics helpers
# --------------------------------------------------------------------------------------
def mean_sem(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """NaN-aware mean, standard error (``ddof=1``) and sample size along axis 0."""
    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    n = finite.sum(axis=0)
    filled = np.where(finite, values, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, filled.sum(axis=0) / n, np.nan)
        dev = np.where(finite, values - mean, 0.0)
        var = (dev**2).sum(axis=0) / (n - 1)
        sem = np.where(n > 1, np.sqrt(var / n), np.nan)
    return mean, sem, n


def median_iqr(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NaN-aware median and interquartile range (``q75 - q25``) along axis 0."""
    values = np.asarray(values, dtype=float)
    with warnings.catch_warnings():  # all-NaN columns
        warnings.simplefilter("ignore", RuntimeWarning)
        q25, med, q75 = np.nanpercentile(values, [25.0, 50.0, 75.0], axis=0)
    return med, q75 - q25


def _resampled_means(values: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Means of ``values`` (shape ``(R, n_bins)``) over the resampled rows ``idx`` (``(n_boot, R)``)."""
    sample = values[idx]  # (n_boot, R, n_bins)
    finite = np.isfinite(sample)
    n = finite.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(finite, sample, 0.0).sum(axis=1) / n, np.nan)


def bootstrap_means(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Means over resampled rows (realizations) of ``values`` (shape ``(R, n_bins)``)."""
    r = values.shape[0]
    return _resampled_means(values, rng.integers(0, r, size=(n_boot, r)))


def round_to_error(value: float, err: float) -> str:
    """``value ± err`` with the error rounded to one significant digit (two if it starts with 1)."""
    if not np.isfinite(value):
        return "n/a"
    if not np.isfinite(err) or err <= 0.0:
        return f"{value:.4g}"
    digits = -math.floor(math.log10(err))
    if err * 10.0**digits < 2.0:
        digits += 1
    digits = max(int(digits), 0)
    return f"{value:.{digits}f} ± {err:.{digits}f}"


# --------------------------------------------------------------------------------------
# Landmarks of mean curves
# --------------------------------------------------------------------------------------
def _first_upward_zero(x: np.ndarray, y: np.ndarray) -> float:
    for k in range(len(y) - 1):
        if np.isfinite(y[k]) and np.isfinite(y[k + 1]) and y[k] < 0.0 <= y[k + 1]:
            return float(x[k] + (0.0 - y[k]) * (x[k + 1] - x[k]) / (y[k + 1] - y[k]))
    return float("nan")


def _first_downward_crossing(x: np.ndarray, y: np.ndarray, level: float) -> float:
    for k in range(len(y) - 1):
        if np.isfinite(y[k]) and np.isfinite(y[k + 1]) and y[k] > level >= y[k + 1]:
            return float(x[k] + (level - y[k]) * (x[k + 1] - x[k]) / (y[k + 1] - y[k]))
    return float("nan")


def _parabolic_extremum(x: np.ndarray, y: np.ndarray, k: int) -> tuple[float, float]:
    """Vertex of the parabola through bins ``k-1, k, k+1`` (the bin itself at the edges).

    The three abscissae need not be equally spaced (pair-weighted mean distances are not).
    """
    if k <= 0 or k >= len(y) - 1:
        return float(x[k]), float(y[k])
    if not (np.all(np.isfinite(y[k - 1 : k + 2])) and np.all(np.isfinite(x[k - 1 : k + 2]))):
        return float(x[k]), float(y[k])
    x0, x1, x2 = (float(v) for v in x[k - 1 : k + 2])
    y0, y1, y2 = (float(v) for v in y[k - 1 : k + 2])
    d01 = (y1 - y0) / (x1 - x0)
    d12 = (y2 - y1) / (x2 - x1)
    curvature = (d12 - d01) / (x2 - x0)  # leading coefficient of the interpolating parabola
    if curvature == 0.0:
        return x1, y1
    slope = d01 + curvature * (x1 - x0)  # derivative of the parabola at x1
    return x1 - slope / (2.0 * curvature), y1 - slope**2 / (4.0 * curvature)


def _interpolate(x: np.ndarray, y: np.ndarray, x0: float) -> float:
    """Linear interpolation of ``y(x)`` at ``x0`` through the finite points (NaN outside them)."""
    ok = np.isfinite(x) & np.isfinite(y)
    if not np.isfinite(x0) or ok.sum() < 2 or not (x[ok][0] <= x0 <= x[ok][-1]):
        return float("nan")
    return float(np.interp(x0, x[ok], y[ok]))


def landmark_values(
    x: np.ndarray,
    c: np.ndarray,
    q: np.ndarray,
    a: np.ndarray | None = None,
    p: np.ndarray | None = None,
    cperp: np.ndarray | None = None,
) -> dict[str, float]:
    """Model-independent features of the mean curves on the abscissa ``x``.

    ``C``: value in the first bin, first upward zero ``x_zero_C`` (linear interpolation), position and
    height of the positive peak beyond it (parabola through the highest bin and its neighbours).
    ``Q``: first bin, minimum (parabola) and the downward crossing of ``1/3``. ``A`` (optional):
    first bin and the downward crossing of ``1/2``. ``Cperp`` (optional): position and height of its
    maximum (parabola) and, with ``p``, the longitudinal channel ``P`` interpolated at that position.
    """
    out: dict[str, float] = {}
    out["C_first_bin"] = float(c[np.isfinite(c)][0]) if np.isfinite(c).any() else float("nan")
    x_zero = _first_upward_zero(x, c)
    out["x_zero_C"] = x_zero
    peak = float("nan"), float("nan")
    if np.isfinite(x_zero):
        cand = np.where(np.isfinite(c) & (x >= x_zero), c, -np.inf)
        if np.isfinite(cand).any() and cand.max() > -np.inf:
            peak = _parabolic_extremum(x, c, int(np.argmax(cand)))
    out["x_peak_C"], out["C_peak"] = peak
    if np.isfinite(q).any():
        k = int(np.nanargmin(q))
        out["x_min_Q"], out["Q_min"] = _parabolic_extremum(x, q, k)
        out["Q_first_bin"] = float(q[np.isfinite(q)][0])
        out["x_cross_Q_third"] = _first_downward_crossing(x, q, 1.0 / 3.0)
    if a is not None and np.isfinite(a).any():
        out["A_first_bin"] = float(a[np.isfinite(a)][0])
        out["x_cross_A_half"] = _first_downward_crossing(x, a, 0.5)
    if cperp is not None and np.isfinite(cperp).any():
        k = int(np.nanargmax(cperp))
        out["x_peak_Cperp"], out["Cperp_peak"] = _parabolic_extremum(x, cperp, k)
        if p is not None:
            out["P_at_Cperp_peak"] = _interpolate(x, p, out["x_peak_Cperp"])
    return out


def _features(x: np.ndarray, means: dict[str, np.ndarray]) -> dict[str, float]:
    """:func:`landmark_values` from a dictionary of mean curves keyed by observable."""
    return landmark_values(
        x, means["C"], means["Q"], means.get("A"), means.get("P"), means.get("Cperp")
    )


def _with_bootstrap_errors(
    central: dict[str, float], samples: list[dict[str, float]]
) -> dict[str, Any]:
    """Central values with the spread of the bootstrap ``samples`` (NaN if the value is NaN)."""
    result: dict[str, Any] = {}
    for key, value in central.items():
        vals = np.array([s.get(key, np.nan) for s in samples], dtype=float)
        finite = vals[np.isfinite(vals)]
        ok = bool(np.isfinite(value)) and finite.size > 2
        result[key] = {
            "value": float(value),
            "err": float(np.std(finite, ddof=1)) if ok else float("nan"),
            "p16": float(np.percentile(finite, 16)) if ok else float("nan"),
            "p84": float(np.percentile(finite, 84)) if ok else float("nan"),
            "n_valid_bootstrap": int(finite.size),
        }
    return result


# --------------------------------------------------------------------------------------
# Ensemble container
# --------------------------------------------------------------------------------------
class EnsembleData:
    """Stacked per-realization arrays of one run (leading axis: realization)."""

    def __init__(self, config: SimulationConfig, arrays: dict[str, np.ndarray]):
        self.config = config
        self.arrays = arrays
        self.seeds = np.asarray(arrays["seed"]).astype(int)

    # -- construction ----------------------------------------------------------------
    @classmethod
    def from_records(cls, config: SimulationConfig, records: list[dict[str, np.ndarray]]):
        """Stack per-realization records along a new leading axis.

        An ensemble extended after the code gained new record keys (such as the dimer-frame map)
        holds them only in part of its files. Such keys are filled with ``0`` (integer counts) or
        NaN (floating-point sums) in the records that lack them, which the aggregations treat as
        "no pairs", and a warning names them.
        """
        if not records:
            raise ValueError("no realizations")
        keys = list(dict.fromkeys(key for record in records for key in record))
        arrays: dict[str, np.ndarray] = {}
        partial: list[str] = []
        for key in keys:
            present = [np.asarray(r[key]) for r in records if key in r]
            if len(present) == len(records):
                arrays[key] = np.stack(present)
                continue
            template = present[0]
            fill = np.nan if np.issubdtype(template.dtype, np.floating) else 0
            blank = np.full(template.shape, fill, dtype=template.dtype)
            arrays[key] = np.stack([np.asarray(r[key]) if key in r else blank for r in records])
            partial.append(f"{key} ({len(present)} of {len(records)})")
        if partial:
            warnings.warn(
                "keys missing from some realizations are treated as empty: " + ", ".join(partial),
                stacklevel=2,
            )
        return cls(config, arrays)

    @classmethod
    def load(cls, run_dir: str | Path, max_realizations: int | None = None) -> EnsembleData:
        """Load all finished realizations of ``run_dir`` (those listed in its ``config.json``)."""
        run_dir = Path(run_dir)
        config = read_config(run_dir)
        seeds = [s for s in config.seeds if seed_path(run_dir, s).exists()]
        if max_realizations is not None:
            seeds = seeds[:max_realizations]
        if not seeds:
            raise FileNotFoundError(f"no realizations found in {run_dir / SEEDS_DIRNAME}")
        if len(seeds) < config.n_realizations and max_realizations is None:
            warnings.warn(
                f"{run_dir}: only {len(seeds)} of {config.n_realizations} realizations are finished",
                stacklevel=2,
            )
        return cls.from_records(config, [load_record(seed_path(run_dir, s)) for s in seeds])

    # -- basic info ------------------------------------------------------------------
    @property
    def n_real(self) -> int:
        return int(self.seeds.shape[0])

    def has(self, key: str) -> bool:
        return key in self.arrays

    def has_observable(self, obs: str, state: str = "final") -> bool:
        """Whether ``obs`` (stored, or derived such as ``Cperp``) can be formed for ``state``."""
        prefix = _STATE_PREFIX[state]
        parts = _DERIVED.get(obs, ((obs, 1.0),))
        return all(self.has(f"{prefix}_sum_{name}") for name, _ in parts)

    def _rebin(self, arr: np.ndarray, factor: int) -> np.ndarray:
        if factor == 1:
            return arr
        n_bins = arr.shape[-1]
        if factor < 1 or n_bins % factor:
            raise ValueError(f"rebin factor {factor} must divide n_bins = {n_bins}")
        return arr.reshape(*arr.shape[:-1], n_bins // factor, factor).sum(axis=-1)

    def x_edges(self, rebin: int = 1) -> np.ndarray:
        edges = self.config.x_edges
        if self.config.n_bins % rebin:
            raise ValueError(f"rebin factor {rebin} must divide n_bins = {self.config.n_bins}")
        return edges[::rebin]

    def x_center(self, rebin: int = 1) -> np.ndarray:
        e = self.x_edges(rebin)
        return 0.5 * (e[:-1] + e[1:])

    def x_mean(
        self,
        rebin: int = 1,
        state: str = "final",
        snapshot: int | None = None,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        """Pair-weighted mean distance ``sum_x / cnt_C`` of each bin, pooled over ``rows``.

        For Poisson positions it exceeds the bin centre ``c`` by about ``h^2 / (6c)`` (bin width
        ``h``). Falls back to the bin centres if the run stored no ``sum_x``.
        """
        prefix = "snap" if snapshot is not None else _STATE_PREFIX[state]
        if not self.has(f"{prefix}_sum_x"):
            return self.x_center(rebin)
        sums, counts = self.arrays[f"{prefix}_sum_x"], self.arrays[f"{prefix}_cnt_C"]
        if snapshot is not None:
            sums, counts = sums[:, snapshot, :], counts[:, snapshot, :]
        if rows is not None:
            sums, counts = sums[rows], counts[rows]
        s, c = self._rebin(sums, rebin).sum(axis=0), self._rebin(counts, rebin).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / c, np.nan)

    def abscissa(
        self, rebin: int = 1, kind: str = "x_mean", rows: np.ndarray | None = None
    ) -> np.ndarray:
        """Bin abscissa: ``"x_mean"`` (pair-weighted mean distance) or ``"x_center"``."""
        if kind not in ABSCISSAE:
            raise ValueError(f"abscissa must be one of {ABSCISSAE}")
        return self.x_mean(rebin, rows=rows) if kind == "x_mean" else self.x_center(rebin)

    def abscissa_used(self, kind: str) -> str:
        """The abscissa actually available for ``kind`` (``x_mean`` needs stored ``sum_x``)."""
        return kind if kind == "x_center" or self.has("fin_sum_x") else "x_center"

    def bin_width(self, rebin: int = 1) -> float:
        return float(self.config.x_max / self.config.n_bins * rebin)

    # -- per-realization means and ensemble statistics --------------------------------
    def _sums_counts(self, obs: str, state: str, rebin: int, snapshot: int | None):
        prefix = "snap" if snapshot is not None else _STATE_PREFIX[state]
        parts = _DERIVED.get(obs, ((obs, 1.0),))
        sums = sum(coef * self.arrays[f"{prefix}_sum_{name}"] for name, coef in parts)
        counts = self.arrays[f"{prefix}_cnt_{_COUNT_OF[obs]}"]
        if snapshot is not None:
            sums, counts = sums[:, snapshot, :], counts[:, snapshot, :]
        return self._rebin(sums, rebin), self._rebin(counts, rebin)

    def realization_means(
        self, obs: str, state: str = "final", rebin: int = 1, snapshot: int | None = None
    ) -> np.ndarray:
        """Per-realization bin means, shape ``(R, n_bins // rebin)`` (NaN in empty bins).

        ``obs`` is one of :data:`OBSERVABLES`; ``Cperp = C - P`` is formed from the stored sums.
        """
        sums, counts = self._sums_counts(obs, state, rebin, snapshot)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, sums / counts, np.nan)

    def pooled_means(
        self,
        obs: str,
        state: str = "final",
        rebin: int = 1,
        rows: np.ndarray | None = None,
    ) -> np.ndarray:
        """Pair-pooled means (sum of sums / sum of counts) over ``rows`` (default: all)."""
        sums, counts = self._sums_counts(obs, state, rebin, None)
        if rows is not None:
            sums, counts = sums[rows], counts[rows]
        s, c = sums.sum(axis=0), counts.sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c > 0, s / c, np.nan)

    def available_observables(self) -> list[str]:
        return [o for o in OBSERVABLES if self.has_observable(o)]

    def table(self, rebin: int = 1) -> pd.DataFrame:
        """Ensemble table (one row per radial bin); column names extend the prototype's CSV.

        For every available observable (including ``Cperp``) the columns ``mean_<obs>_<state>`` and
        ``sem_<obs>_<state>`` hold the mean over realizations and its standard error.
        """
        edges = self.x_edges(rebin)
        x = self.x_center(rebin)
        n_bins = len(x)
        cfg = self.config
        cnt_c = self._rebin(self.arrays["fin_cnt_C"], rebin)
        data: dict[str, Any] = {
            "bin": np.arange(n_bins),
            "x_left": edges[:-1],
            "x_right": edges[1:],
            "x_center": x,
            "x_mean": self.x_mean(rebin),
            "r_center": x * cfg.r_nn,
            "count_C_total_final": cnt_c.sum(axis=0),
            "count_Q_total_final": self._rebin(self.arrays["fin_cnt_Q"], rebin).sum(axis=0),
        }
        for obs in self.available_observables():
            for state in ("initial", "final"):
                if not self.has_observable(obs, state):
                    continue
                mean, sem, n = mean_sem(self.realization_means(obs, state, rebin))
                data[f"mean_{obs}_{state}"] = mean
                data[f"sem_{obs}_{state}"] = sem
                if state == "final" and obs in ("C", "Q"):
                    data[f"n_real_{obs}_final"] = n
        data["C_NN_isolated"] = -theory.isolated_sphere_probability(x)
        data["C_NN_mutual"] = -theory.mutual_nn_probability(x)
        data["Q_isotropic"] = np.full(n_bins, 1.0 / 3.0)
        data["A_isotropic"] = np.full(n_bins, 1.0 / 3.0)
        return pd.DataFrame(data)

    # -- scalar diagnostics ------------------------------------------------------------
    def scalar_summary(self) -> dict[str, dict[str, float]]:
        """Mean and standard error over realizations of the global spin diagnostics."""
        n = self.config.n_particles
        out: dict[str, dict[str, float]] = {}

        def add(name: str, values: np.ndarray) -> None:
            m, s, _ = mean_sem(np.asarray(values, dtype=float)[:, None])
            out[name] = {"mean": float(m[0]), "sem": float(s[0])}

        add("total_spin_over_sqrtN_final", self.arrays["fin_total_spin"] / np.sqrt(n))
        add("total_spin_over_sqrtN_initial", self.arrays["init_total_spin"] / np.sqrt(n))
        add("S0_unit_final", self.arrays["fin_s0_unit"])
        add("S0_unit_initial", self.arrays["init_s0_unit"])
        add("energy_drift", np.abs(self.arrays["energy_drift"]))
        add("elapsed_s", self.arrays["elapsed_s"])
        if self.has("nn_frac_mutual"):
            add("mutual_nn_particle_fraction", self.arrays["nn_frac_mutual"])
        return out

    def energy_summary(self) -> dict[str, float]:
        """Final energy per spin (in units of ``G S^2 rho``) and the state of the last sweeps.

        The ensemble *mean* of the energy is dominated by the closest pair of each realization: for
        Poisson positions the pair energy ``2 G s_i s_j / r^3`` of the nearest pair has a heavy tail,
        so the mean fluctuates strongly between realizations and only the median and the
        interquartile range are robust ensemble numbers (paired differences between runs on the
        same seeds are unaffected because the close pairs cancel).
        """
        energy = self.arrays["energy"]
        per_spin = energy[:, -1] / self.config.n_particles
        q25, med, q75 = np.percentile(per_spin, [25.0, 50.0, 75.0])
        m, s, _ = mean_sem(per_spin[:, None])
        out = {
            "per_spin_mean": float(m[0]),
            "per_spin_sem": float(s[0]),
            "per_spin_median": float(med),
            "per_spin_q25": float(q25),
            "per_spin_q75": float(q75),
            "per_spin_min": float(per_spin.min()),
            "per_spin_max": float(per_spin.max()),
        }
        n_sweeps = energy.shape[1] - 1
        if n_sweeps >= 2:
            half = n_sweeps // 2
            with np.errstate(invalid="ignore", divide="ignore"):
                excess = (energy[:, half] - energy[:, -1]) / np.abs(energy[:, -1])
            out["excess_sweep"] = float(half)
            out["excess_mean"] = float(np.nanmean(excess))
            out["excess_median"] = float(np.nanmedian(excess))
            dmax = self.arrays["dmax"][:, -1]
            if np.isfinite(dmax).any():
                out["dmax_last_mean"] = float(np.nanmean(dmax))
                out["dmax_last_median"] = float(np.nanmedian(dmax))
                out["dmax_last_max"] = float(np.nanmax(dmax))
        return out

    def histories(self) -> pd.DataFrame:
        """Convergence history over realizations: mean (with SEM), median and IQR per sweep.

        Energies are heavy-tailed over realizations (see :meth:`energy_summary`), hence the median
        and interquartile range of the energy per spin and of the relative excess energy are given
        next to the means; the largest direction change of a unit spin per sweep is the robust
        convergence metric.
        """
        energy = self.arrays["energy"]
        n = self.config.n_particles
        final = energy[:, -1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            excess = (energy - final) / np.abs(final)
        m_e, s_e, _ = mean_sem(energy)
        med_e, iqr_e = median_iqr(energy / n)
        m_x, s_x, _ = mean_sem(excess)
        med_x, iqr_x = median_iqr(excess)
        dmax = self.arrays["dmax"]
        m_d, _, _ = mean_sem(dmax)
        med_d, _ = median_iqr(dmax)
        m_s, s_s, _ = mean_sem(self.arrays["total_spin_hist"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_d = np.nanmax(dmax, axis=0) if energy.shape[1] > 1 else np.array([np.nan])
        return pd.DataFrame(
            {
                "sweep": np.arange(energy.shape[1]),
                "mean_energy": m_e,
                "sem_energy": s_e,
                "mean_energy_per_spin": m_e / n,
                "median_energy_per_spin": med_e,
                "iqr_energy_per_spin": iqr_e,
                "mean_excess_energy": m_x,
                "sem_excess_energy": s_x,
                "median_excess_energy": med_x,
                "iqr_excess_energy": iqr_x,
                "mean_max_direction_change": m_d,
                "median_max_direction_change": med_d,
                "max_max_direction_change": max_d,
                "mean_total_spin": m_s,
                "sem_total_spin": s_s,
            }
        )

    # -- bootstrap features ----------------------------------------------------------
    def _landmark_curves(self, rebin: int, rows: np.ndarray | None = None) -> dict[str, np.ndarray]:
        """Per-realization curves of the available landmark observables (``rows``: realizations)."""
        curves: dict[str, np.ndarray] = {}
        for obs in LANDMARK_OBSERVABLES:
            if self.has_observable(obs):
                values = self.realization_means(obs, "final", rebin)
                curves[obs] = values if rows is None else values[rows]
        return curves

    def landmarks(
        self, rebin: int = 1, n_boot: int = 200, seed: int = 12345, abscissa: str = "x_mean"
    ) -> dict[str, Any]:
        """Model-independent features with bootstrap errors (resampling realizations).

        The mean curves are evaluated on ``abscissa`` (default: the pair-weighted mean distance of
        each bin, pooled over all realizations and kept fixed during the bootstrap). All observables
        are resampled with the same realization indices. ``_meta`` records the abscissa, the rebin
        factor and the bin width that the values refer to.
        """
        x = self.abscissa(rebin, abscissa)
        curves = self._landmark_curves(rebin)
        central = _features(x, {o: mean_sem(v)[0] for o, v in curves.items()})
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, self.n_real, size=(n_boot, self.n_real))
        boot = {o: _resampled_means(v, idx) for o, v in curves.items()}
        samples = [_features(x, {o: b[k] for o, b in boot.items()}) for k in range(n_boot)]
        result = _with_bootstrap_errors(central, samples)
        result["_meta"] = {
            "n_real": self.n_real,
            "rebin": rebin,
            "n_bins": len(x),
            "n_boot": n_boot,
            "abscissa": self.abscissa_used(abscissa),
            "bin_width": self.bin_width(rebin),
        }
        return result

    def bootstrap_curves(
        self, obs: str = "C", rebin: int = 1, n_boot: int = 200, seed: int = 12345
    ) -> np.ndarray:
        """Bootstrap ensemble means of ``obs`` (shape ``(n_boot, n_bins)``)."""
        rng = np.random.default_rng(seed)
        return bootstrap_means(self.realization_means(obs, "final", rebin), n_boot, rng)

    # -- sweep convergence ---------------------------------------------------------------
    def sweep_convergence(self, rebin: int = 1, abscissa: str = "x_mean") -> pd.DataFrame:
        """Deviation of the mean ``C(x)`` at recorded sweeps from the final one (needs snapshots).

        The landmark columns (``x_zero_C``, ``x_peak_C``, ``C_peak``, ``Q_min``) use the same
        parabolic refinement as :meth:`landmarks`.
        """
        if not self.has("snap_sweeps"):
            raise ValueError("this run recorded no snapshots (set `snapshots=` in the config)")
        sweeps = [int(s) for s in self.arrays["snap_sweeps"][0]]
        x = self.abscissa(rebin, abscissa)
        ref_mean, ref_sem, _ = mean_sem(self.realization_means("C", "final", rebin))
        energy = self.arrays["energy"]
        final_e = energy[:, -1]
        rows = []
        for index, sweep in enumerate(sweeps):
            mean, sem, _ = mean_sem(self.realization_means("C", rebin=rebin, snapshot=index))
            mean_q = mean_sem(self.realization_means("Q", rebin=rebin, snapshot=index))[0]
            marks = landmark_values(x, mean, mean_q)
            diff = mean - ref_mean
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.abs(diff) / ref_sem
                excess = (energy[:, sweep] - final_e) / np.abs(final_e)
            rows.append(
                {
                    "sweep": sweep,
                    "max_abs_dC": float(np.nanmax(np.abs(diff))),
                    "rms_dC": float(np.sqrt(np.nanmean(diff**2))),
                    "max_dC_over_sem": float(np.nanmax(ratio)),
                    "x_zero_C": marks["x_zero_C"],
                    "x_peak_C": marks["x_peak_C"],
                    "C_peak": marks["C_peak"],
                    "Q_min": marks.get("Q_min", float("nan")),
                    "mean_excess_energy": float(np.mean(excess)),
                    "median_excess_energy": float(np.median(excess)),
                }
            )
        return pd.DataFrame(rows)

    # -- nearest-neighbour diagnostics -------------------------------------------------
    def nn_table(self, rebin: int = 1, n_boot: int = 100, seed: int = 12345) -> pd.DataFrame:
        """Dimer decomposition ``C = f_mut C_mut + (1 - f_mut) C_non`` (pooled over all pairs).

        Bootstrap errors (over realizations) are NaN for bins where a category holds fewer than two
        pairs or is fed by fewer than two realizations; resampling cannot estimate them there.
        """
        if not self.has("nn_cnt_mutual"):
            raise ValueError("run was made without `nn_diagnostics=True`")
        x = self.x_center(rebin)
        cnt_all = self._rebin(self.arrays["fin_cnt_C"], rebin)
        sum_all = self._rebin(self.arrays["fin_sum_C"], rebin)
        parts = {
            name: (
                self._rebin(self.arrays[f"nn_cnt_{name}"], rebin),
                self._rebin(self.arrays[f"nn_sum_{name}"], rebin),
            )
            for name in ("mutual", "oneway", "far")
        }
        counts_of = {
            "C_all": cnt_all,
            "f_mutual": cnt_all,
            "C_mutual": parts["mutual"][0],
            "C_non_mutual": cnt_all - parts["mutual"][0],
            "f_oneway": cnt_all,
            "C_oneway": parts["oneway"][0],
            "f_far": cnt_all,
            "C_far": parts["far"][0],
        }

        def estimate(rows: np.ndarray) -> dict[str, np.ndarray]:
            ca, sa = cnt_all[rows].sum(0), sum_all[rows].sum(0)
            cm, sm = (parts["mutual"][0][rows].sum(0), parts["mutual"][1][rows].sum(0))
            co, so = (parts["oneway"][0][rows].sum(0), parts["oneway"][1][rows].sum(0))
            cf, sf = (parts["far"][0][rows].sum(0), parts["far"][1][rows].sum(0))
            with np.errstate(invalid="ignore", divide="ignore"):
                return {
                    "C_all": sa / ca,
                    "f_mutual": cm / ca,
                    "C_mutual": sm / cm,
                    "C_non_mutual": (sa - sm) / (ca - cm),
                    "f_oneway": co / ca,
                    "C_oneway": so / co,
                    "f_far": cf / ca,
                    "C_far": sf / cf,
                }

        rows_all = np.arange(self.n_real)
        central = estimate(rows_all)
        rng = np.random.default_rng(seed)
        boots = [estimate(rng.integers(0, self.n_real, self.n_real)) for _ in range(n_boot)]
        data: dict[str, Any] = {"x_center": x}
        for key, value in central.items():
            data[key] = value
            if n_boot > 1:
                counts = counts_of[key]
                reliable = (counts.sum(axis=0) >= 2) & ((counts > 0).sum(axis=0) >= 2)
                with warnings.catch_warnings():  # bins without pairs of a kind are all-NaN
                    warnings.simplefilter("ignore", RuntimeWarning)
                    spread = np.nanstd([b[key] for b in boots], axis=0, ddof=1)
                data[f"sem_{key}"] = np.where(reliable, spread, np.nan)
        data["f_mutual_theory"] = theory.mutual_nn_probability(x)
        data["f_isolated_sphere_theory"] = theory.isolated_sphere_probability(x)
        return pd.DataFrame(data)

    def nn_rank_table(self) -> pd.DataFrame:
        """Correlation between a spin and its ``k``-th nearest neighbour (mean over realizations)."""
        if not self.has("nn_Ck"):
            raise ValueError("run was made without `nn_diagnostics=True`")
        ck_m, ck_s, _ = mean_sem(self.arrays["nn_Ck"])
        rk_m, _, _ = mean_sem(self.arrays["nn_rk"])
        return pd.DataFrame(
            {
                "k": np.arange(1, len(ck_m) + 1),
                "mean_r_over_rnn": rk_m,
                "C_k": ck_m,
                "sem_C_k": ck_s,
            }
        )

    def nn_distance_table(self) -> pd.DataFrame:
        """Histogram of first-neighbour distances versus the Poisson expectation."""
        if not self.has("nn_hist"):
            raise ValueError("run was made without `nn_diagnostics=True`")
        counts = self.arrays["nn_hist"].sum(axis=0).astype(float)
        edges = NN_HIST_EDGES_X
        centers = 0.5 * (edges[:-1] + edges[1:])
        total = counts.sum()
        density = counts / total / np.diff(edges)
        return pd.DataFrame(
            {
                "x_center": centers,
                "pdf_measured": density,
                "pdf_poisson": theory.nn_distance_pdf(centers),
            }
        )

    def small_r_table(self) -> pd.DataFrame:
        """Small-distance law of ``1 + C`` in log-spaced bins (pooled), versus the NN estimates."""
        if not self.has("sr_cnt"):
            raise ValueError("run was made without `nn_diagnostics=True`")
        cnt = self.arrays["sr_cnt"].sum(axis=0).astype(float)
        with np.errstate(invalid="ignore", divide="ignore"):
            one_plus_c = self.arrays["sr_sum1pC"].sum(axis=0) / cnt
            x_mean = self.arrays["sr_sumx"].sum(axis=0) / cnt
        iso = 1.0 - theory.isolated_sphere_probability(x_mean)
        mut = 1.0 - theory.mutual_nn_probability(x_mean)
        slope = np.full_like(x_mean, np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            slope[1:] = np.log(one_plus_c[1:] / one_plus_c[:-1]) / np.log(x_mean[1:] / x_mean[:-1])
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = one_plus_c / iso
        return pd.DataFrame(
            {
                "x_mean": x_mean,
                "n_pairs": cnt,
                "one_plus_C": one_plus_c,
                "isolated_sphere_1_minus_P": iso,
                "mutual_nn_1_minus_P": mut,
                "ratio_to_isolated_sphere": ratio,
                "local_log_slope": slope,
            }
        )

    # -- dimer-frame correlation map -------------------------------------------------------
    def dimer_map(self, min_count: int = 20) -> dict[str, Any]:
        """Mean dimer-frame map of ``C`` and ``P`` (:func:`aggregate_dimer_map`)."""
        return aggregate_dimer_map(self.arrays, min_count)

    def dimer_map_table(self, min_count: int = 20) -> pd.DataFrame:
        """The map in long format, one row per cell (:func:`tabulate_dimer_map`)."""
        return tabulate_dimer_map(self.arrays, min_count)

    def dimer_angular_table(self) -> pd.DataFrame:
        """``C`` and ``P`` per bin of ``x_ik`` times ``cos(theta)`` (:func:`tabulate_dimer_angular`)."""
        return tabulate_dimer_angular(self.arrays)

    def dimer_field_table(self) -> pd.DataFrame:
        """Agreement with the dimer-field prediction per bin of ``x_ik`` (:func:`tabulate_dimer_field`)."""
        return tabulate_dimer_field(self.arrays)


# --------------------------------------------------------------------------------------
# Dimer-frame correlation map (aggregation of the stored records or of stored states)
# --------------------------------------------------------------------------------------
def _cell_means(counts: np.ndarray, sums: np.ndarray) -> np.ndarray:
    """Per-realization means ``sums / counts`` (NaN where a realization has no pairs)."""
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(counts > 0, sums / counts, np.nan)


def _require_dimer_map(arrays: dict[str, np.ndarray]) -> None:
    if "dimer_map_cnt" not in arrays:
        raise ValueError(
            "run was made without `nn_diagnostics=True` (or before the dimer-frame map was "
            "recorded; use dimer_map_from_states on its stored states)"
        )


def dimer_map_grid(arrays: dict[str, np.ndarray]) -> tuple[float, np.ndarray, np.ndarray]:
    """``(R, z_edges, rho_edges)`` of the stored map in units of ``r_nn``."""
    n_z, n_rho = arrays["dimer_map_cnt"].shape[-2:]
    r_max = DIMER_MAP_R_NN
    if "dimer_map_r_max" in arrays:
        r_max = float(np.nanmax(arrays["dimer_map_r_max"]))
    return r_max, np.linspace(-r_max, r_max, n_z + 1), np.linspace(0.0, r_max, n_rho + 1)


def dimer_partner_distance(arrays: dict[str, np.ndarray]) -> float:
    """Mean distance ``r_ij / r_nn`` of the dimer spins to their partners, pooled (NaN if unknown).

    For Poisson positions the expected value is :data:`grspin.theory.MUTUAL_PAIR_MEAN_DISTANCE_X`.
    """
    if "dimer_map_n_spins" not in arrays or "dimer_map_sum_x_partner" not in arrays:
        return float("nan")
    n = np.nansum(arrays["dimer_map_n_spins"])
    return float(np.nansum(arrays["dimer_map_sum_x_partner"]) / n) if n > 0 else float("nan")


def aggregate_dimer_map(arrays: dict[str, np.ndarray], min_count: int = 20) -> dict[str, Any]:
    """Mean dimer-frame map over realizations: ``C``, ``P`` with standard errors, pooled counts.

    Each cell holds the mean over realizations of the per-realization cell mean (NaN-aware, like
    the radial observables) and ``sem_<obs>`` its standard error; cells with fewer than
    ``min_count`` pooled pairs are NaN. ``z_edges`` and ``rho_edges`` give the grid in units of
    ``r_nn``, ``x_partner_mean`` the mean partner distance (where to draw ``j``), ``n_real`` the
    number of realizations with pairs and ``n_pairs`` the pooled count.
    """
    _require_dimer_map(arrays)
    cnt = arrays["dimer_map_cnt"]
    r_max, z_edges, rho_edges = dimer_map_grid(arrays)
    pooled = cnt.sum(axis=0)
    sparse = pooled < min_count
    out: dict[str, Any] = {
        "r_max": r_max,
        "z_edges": z_edges,
        "rho_edges": rho_edges,
        "count": pooled,
        "min_count": int(min_count),
        "n_real": int(np.count_nonzero(cnt.sum(axis=(1, 2)))),
        "n_pairs": int(pooled.sum()),
        "x_partner_mean": dimer_partner_distance(arrays),
    }
    for obs in ("C", "P"):
        mean, sem, _ = mean_sem(_cell_means(cnt, arrays[f"dimer_map_sum_{obs}"]))
        out[obs] = np.where(sparse, np.nan, mean)
        out[f"sem_{obs}"] = np.where(sparse, np.nan, sem)
    return out


def tabulate_dimer_map(arrays: dict[str, np.ndarray], min_count: int = 20) -> pd.DataFrame:
    """:func:`aggregate_dimer_map` in long format: one row per cell ``(z, rho)``."""
    m = aggregate_dimer_map(arrays, min_count)
    z, rho = m["z_edges"], m["rho_edges"]
    iz, ir = np.meshgrid(np.arange(len(z) - 1), np.arange(len(rho) - 1), indexing="ij")
    iz, ir = iz.ravel(), ir.ravel()
    return pd.DataFrame(
        {
            "z_lo": z[iz],
            "z_hi": z[iz + 1],
            "z_center": 0.5 * (z[iz] + z[iz + 1]),
            "rho_lo": rho[ir],
            "rho_hi": rho[ir + 1],
            "rho_center": 0.5 * (rho[ir] + rho[ir + 1]),
            "n": m["count"].ravel(),
            "C": m["C"].ravel(),
            "sem_C": m["sem_C"].ravel(),
            "P": m["P"].ravel(),
            "sem_P": m["sem_P"].ravel(),
        }
    )


def tabulate_dimer_angular(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    """Angular table of the dimer frame: one row per bin of ``x_ik`` times signed ``cos(theta)``.

    Columns ``x_lo, x_hi, cos_lo, cos_hi, region`` (:data:`DIMER_ANG_REGIONS`), ``n`` (pooled
    pairs), ``C, sem_C, P, sem_P`` (mean over realizations of the per-realization bin means and its
    standard error) and, for records with the dimer-field test, ``C_pred, sem_C_pred``: the same
    statistic of ``(sigma h / |h|) . u_i``, i.e. ``C`` as predicted by the field of the two dimer
    members alone (:func:`grspin.observables.dimer_field_prediction`).
    """
    _require_dimer_map(arrays)
    cnt = arrays["dimer_ang_cnt"]
    n_x, n_c = cnt.shape[-2:]
    x_edges, c_edges = DIMER_ANG_X_EDGES, DIMER_ANG_COS_EDGES
    if (n_x, n_c) != (len(x_edges) - 1, len(c_edges) - 1):
        raise ValueError(f"stored angular table of shape {(n_x, n_c)} does not match the bin edges")
    ix, ic = np.meshgrid(np.arange(n_x), np.arange(n_c), indexing="ij")
    ix, ic = ix.ravel(), ic.ravel()
    data: dict[str, Any] = {
        "x_lo": x_edges[ix],
        "x_hi": x_edges[ix + 1],
        "cos_lo": c_edges[ic],
        "cos_hi": c_edges[ic + 1],
        "region": np.array(DIMER_ANG_REGIONS)[ic],
        "n": cnt.sum(axis=0).ravel(),
    }
    for obs in ("C", "P"):
        mean, sem, _ = mean_sem(_cell_means(cnt, arrays[f"dimer_ang_sum_{obs}"]))
        data[obs], data[f"sem_{obs}"] = mean.ravel(), sem.ravel()
    if "dimer_ang_sum_Cpred" in arrays:
        mean, sem, _ = mean_sem(_cell_means(cnt, arrays["dimer_ang_sum_Cpred"]))
        data["C_pred"], data["sem_C_pred"] = mean.ravel(), sem.ravel()
    return pd.DataFrame(data)


def _require_dimer_field(arrays: dict[str, np.ndarray]) -> None:
    if "dimer_field_cnt" not in arrays:
        raise ValueError(
            "run has no dimer-field test (recorded with the nn diagnostics since version 0.3; "
            "use dimer_map_from_states on its stored states)"
        )


def tabulate_dimer_field(arrays: dict[str, np.ndarray]) -> pd.DataFrame:
    """Agreement of the third spins with the dimer-field prediction, per bin of ``x_ik``.

    For every pair ``(i, k)`` of the dimer-frame map the field of the two dimer members alone
    predicts the direction ``sigma h / |h|`` of ``k`` (:func:`grspin.observables.dimer_field_prediction`).
    One row per bin of the angular table (:data:`DIMER_ANG_X_EDGES`) plus a last row for all pairs
    within the map radius (``x_lo = 0``, ``x_hi = R``, which also holds the pairs below the first
    edge). Columns: ``n`` and ``n_agree`` (pooled pairs and agreements), ``agree`` and
    ``sem_agree`` (mean over realizations of the per-realization agreement fraction and its
    standard error), ``agree_pooled = n_agree / n``, ``cos`` and ``sem_cos`` (the same statistic
    of ``u_k . sigma h / |h|``; not stored for the pooled row) and ``n_real`` (realizations with
    pairs in the bin). Random directions give ``agree = 0.5`` and ``cos = 0``, a perfect
    prediction ``1`` and ``1``. ``frame.attrs["meta"]`` holds ``sigma``, ``gamma``, the map radius
    and the pooled totals.
    """
    _require_dimer_field(arrays)
    cnt = np.asarray(arrays["dimer_field_cnt"])
    agree = np.asarray(arrays["dimer_field_agree"])
    edges = DIMER_ANG_X_EDGES
    if cnt.shape[-1] != len(edges) - 1:
        raise ValueError(
            f"stored dimer-field table of {cnt.shape[-1]} bins does not match the edges"
        )
    r_max = DIMER_MAP_R_NN
    if "dimer_map_r_max" in arrays:
        r_max = float(np.nanmax(arrays["dimer_map_r_max"]))
    m_agree, s_agree, n_real = mean_sem(_cell_means(cnt, agree.astype(float)))
    m_cos, s_cos, _ = mean_sem(_cell_means(cnt, arrays["dimer_field_sum_cos"]))
    n_all = np.asarray(arrays["dimer_field_n"]).reshape(-1, 1)
    a_all = np.asarray(arrays["dimer_field_n_agree"]).reshape(-1, 1)
    m_all, s_all, r_all = mean_sem(_cell_means(n_all, a_all.astype(float)))
    pooled_n = np.concatenate([cnt.sum(axis=0), n_all.sum(axis=0)]).astype(np.int64)
    pooled_a = np.concatenate([agree.sum(axis=0), a_all.sum(axis=0)]).astype(np.int64)
    with np.errstate(invalid="ignore", divide="ignore"):
        pooled = np.where(pooled_n > 0, pooled_a / pooled_n, np.nan)
    frame = pd.DataFrame(
        {
            "x_lo": np.concatenate([edges[:-1], [0.0]]),
            "x_hi": np.concatenate([edges[1:], [r_max]]),
            "n": pooled_n,
            "n_agree": pooled_a,
            "agree": np.concatenate([m_agree, m_all]),
            "sem_agree": np.concatenate([s_agree, s_all]),
            "agree_pooled": pooled,
            "cos": np.concatenate([m_cos, [np.nan]]),
            "sem_cos": np.concatenate([s_cos, [np.nan]]),
            "n_real": np.concatenate([n_real, r_all]).astype(int),
        }
    )

    def first_finite(key: str) -> float:
        values = np.asarray(arrays.get(key, np.nan), dtype=float).ravel()
        finite = values[np.isfinite(values)]
        return float(finite[0]) if finite.size else float("nan")

    frame.attrs["meta"] = {
        "sigma": first_finite("dimer_field_sigma"),
        "gamma": first_finite("dimer_field_gamma"),
        "r_max_nn": r_max,
        "n_pairs": int(pooled_n[-1]),
        "n_agree": int(pooled_a[-1]),
        "agree_pooled": float(pooled[-1]),
        "agree_mean": float(m_all[0]),
        "agree_sem": float(s_all[0]),
        "n_real": int(r_all[0]),
    }
    return frame


def _state_seeds(run_dir: Path, config: SimulationConfig, max_states: int | None) -> list[int]:
    seeds = [s for s in config.seeds if state_path(run_dir, s).exists()]
    if not seeds:
        raise FileNotFoundError(f"no stored states in {run_dir / STATES_DIRNAME}")
    return seeds[:max_states]


def dimer_map_from_states(
    run_dir: str | Path, max_states: int | None = None
) -> dict[str, np.ndarray]:
    """Dimer-frame map records computed from the stored states of ``run_dir`` (stacked arrays).

    For runs made before the map (or the dimer-field test) was recorded per realization. Returns
    the arrays of :func:`grspin.observables.dimer_frame_map`, including the dimer-field test with
    the run's ``(sigma, gamma)``, stacked over the states (plus ``seed``), ready for
    :func:`aggregate_dimer_map`, :func:`tabulate_dimer_map`, :func:`tabulate_dimer_angular` and
    :func:`tabulate_dimer_field`.
    """
    run_dir = Path(run_dir)
    config = read_config(run_dir)
    box = config.box if config.boundary == "periodic" else None
    records = []
    for seed in _state_seeds(run_dir, config, max_states):
        state = load_record(state_path(run_dir, seed))
        positions = state["positions"]
        tree = build_tree(positions, box)
        partner = mutual_nn_partners(positions, tree)
        record = dimer_frame_map(
            positions,
            state["spins_final"],
            tree,
            box,
            config.r_nn,
            partner,
            sigma_gamma=config.sigma_gamma,
        )
        record["seed"] = np.array(seed)
        records.append(record)
    return {key: np.stack([r[key] for r in records]) for key in records[0]}


# --------------------------------------------------------------------------------------
# Energy budget of the stored states
# --------------------------------------------------------------------------------------
#: Ratios of the energy budget (all independent of ``G``); see :func:`energy_budget_from_states`.
ENERGY_BUDGET_RATIOS = (
    "dimer_share",
    "nondimer_share",
    "realized_fraction_dimer",
    "realized_fraction_nondimer",
    "frustration_index",
    "frustration_index_initial",
)


def energy_budget_from_states(run_dir: str | Path, max_states: int | None = None) -> pd.DataFrame:
    """Energy budget of the stored states of ``run_dir``, one row per state.

    With the run's ``(sigma, gamma)`` and cutoff (:func:`grspin.observables.energy_budget`):
    ``energy_total`` (the energy of all bonds within ``r_cut``, with ``G``; it reproduces the
    final energy of the relaxation), ``energy_dimer`` (the bonds of the mutual-nearest-neighbour
    pairs), ``energy_nondimer = energy_total - energy_dimer``, and the sums of the isolated-pair
    minima ``-f G w s_i s_j`` (``f`` = :func:`grspin.theory.two_body_minimum_factor`) over the
    same sets, ``minimum_total`` and ``minimum_dimer``. Ratios:

    * ``dimer_share = energy_dimer / energy_total`` and ``nondimer_share = 1 - dimer_share``;
    * ``realized_fraction_dimer = energy_dimer / minimum_dimer``: how close the dimers are to the
      energy of isolated pairs (``1``: exactly there); ``realized_fraction_nondimer`` the same for
      the remaining bonds;
    * ``frustration_index = energy_total / minimum_total``: the energy in units of the sum of the
      two-body minima of all bonds (``1`` if every bond were at its own minimum, ``0`` for
      uncorrelated spins; ``frustration_index_initial`` is the value of the stored initial state).

    Also ``n_pairs``, ``n_dimers`` and ``n_dimer_spins``. ``frame.attrs["meta"]`` records the
    coupling, ``sigma``, ``gamma``, ``two_body_factor``, ``r_cut_nn`` and the seeds.
    """
    run_dir = Path(run_dir)
    config = read_config(run_dir)
    seeds = _state_seeds(run_dir, config, max_states)
    box = config.box if config.boundary == "periodic" else None
    sigma, gamma = config.sigma_gamma
    g = float(config.G)
    rows = []
    for seed in seeds:
        state = load_record(state_path(run_dir, seed))
        positions = state["positions"]
        tree = build_tree(positions, box)
        partner = mutual_nn_partners(positions, tree)
        final = energy_budget(
            positions, state["spins_final"], tree, box, config.r_cut, sigma, gamma, partner
        )
        initial = energy_budget(
            positions, state["spins_initial"], tree, box, config.r_cut, sigma, gamma, partner
        )
        e, e_d = final["energy_total"], final["energy_dimer"]
        m, m_d = final["minimum_total"], final["minimum_dimer"]
        with np.errstate(invalid="ignore", divide="ignore"):
            rows.append(
                {
                    "seed": int(seed),
                    "n_pairs": final["n_pairs"],
                    "n_dimers": final["n_dimers"],
                    "n_dimer_spins": int(np.count_nonzero(partner >= 0)),
                    "energy_total": g * e,
                    "energy_dimer": g * e_d,
                    "energy_nondimer": g * (e - e_d),
                    "minimum_total": g * m,
                    "minimum_dimer": g * m_d,
                    "energy_total_initial": g * initial["energy_total"],
                    "dimer_share": e_d / e,
                    "nondimer_share": (e - e_d) / e,
                    "realized_fraction_dimer": e_d / m_d,
                    "realized_fraction_nondimer": (e - e_d) / (m - m_d),
                    "frustration_index": e / m,
                    "frustration_index_initial": initial["energy_total"] / m,
                }
            )
    frame = pd.DataFrame(rows)
    frame.attrs["meta"] = {
        "n_states": len(seeds),
        "seeds": [int(s) for s in seeds],
        "coupling": config.coupling,
        "sigma": float(sigma),
        "gamma": float(gamma),
        "two_body_factor": theory.two_body_minimum_factor(sigma, gamma),
        "r_cut_nn": config.r_cut_nn_resolved,
        "n_particles": config.n_particles,
        "G": g,
    }
    return frame


def summarize_energy_budget(frame: pd.DataFrame) -> dict[str, Any]:
    """Mean, standard error, minimum and maximum over the states of the energy-budget ratios.

    Also the energies per spin (``energy_total``, ``energy_dimer``, ``energy_nondimer``,
    ``minimum_total`` divided by ``N``) with the same statistics, the number of states and the
    metadata of :func:`energy_budget_from_states`.
    """
    meta = dict(frame.attrs.get("meta", {}))
    n = float(meta.get("n_particles", 1))
    out: dict[str, Any] = {"meta": meta, "n_states": int(len(frame))}
    columns = [*ENERGY_BUDGET_RATIOS, "energy_total", "energy_dimer", "energy_nondimer"]
    columns.append("minimum_total")
    for column in columns:
        values = frame[column].to_numpy(dtype=float)
        if column.startswith(("energy", "minimum")):
            values = values / n
            name = f"{column}_per_spin"
        else:
            name = column
        mean, sem, _ = mean_sem(values[:, None])
        out[name] = {
            "mean": float(mean[0]),
            "sem": float(sem[0]),
            "min": float(np.nanmin(values)) if values.size else float("nan"),
            "max": float(np.nanmax(values)) if values.size else float("nan"),
        }
    return out


# --------------------------------------------------------------------------------------
# Spin structure factor of the stored states
# --------------------------------------------------------------------------------------
def structure_factor_from_states(
    run_dir: str | Path,
    n_shells: int = 24,
    n_max: int = 12,
    max_per_shell: int | None = None,
    seed: int = 12345,
    max_states: int | None = None,
) -> pd.DataFrame:
    """Spherically averaged spin structure factor of the stored states of ``run_dir``.

    ``S(k) = |sum_i u_i exp(i k . r_i)|^2 / N`` of the unit spins is evaluated at the reciprocal
    vectors ``k = 2 pi n / L`` of the box with ``0 < |n| <= n_max`` (one of each ``±k`` pair, about
    ``2 pi n_max^3 / 3`` vectors; ``max_per_shell`` limits them to a seeded random subset per
    shell for large ``n_max``), averaged over ``n_shells`` equal shells of ``|k|`` up to
    ``2 pi n_max / L`` and over the states for the initial and the final spins. The first row is
    ``k = 0`` (``s0_unit``); empty shells are dropped. Uncorrelated directions give ``S = 1``.

    Columns: ``k_rnn_lo``, ``k_rnn_hi``, ``k_rnn_mean`` (``|k| r_nn``), ``wavelength_rnn``,
    ``n_vectors``, ``S_final``, ``sem_S_final``, ``S_initial``, ``sem_S_initial`` (standard errors
    over the states, NaN with a single state). ``frame.attrs["meta"]`` records the run parameters,
    the seeds of the states and the number of vectors.
    """
    run_dir = Path(run_dir)
    config = read_config(run_dir)
    seeds = _state_seeds(run_dir, config, max_states)
    box, r_nn = config.box, config.r_nn
    _, kvec = reciprocal_lattice_vectors(box, n_max)
    k_abs = np.linalg.norm(kvec, axis=1)
    k_max = 2.0 * np.pi * n_max / box
    shell = 1 + np.clip(np.floor(k_abs / k_max * n_shells), 0, n_shells - 1).astype(int)
    if max_per_shell is not None:
        rng = np.random.default_rng(seed)
        chosen = []
        for s in range(1, n_shells + 1):
            members = np.flatnonzero(shell == s)
            if members.size > max_per_shell:
                members = np.sort(rng.choice(members, max_per_shell, replace=False))
            chosen.append(members)
        pick = np.concatenate(chosen)
        kvec, k_abs, shell = kvec[pick], k_abs[pick], shell[pick]
    kvec = np.vstack([np.zeros((1, 3)), kvec])  # k = 0 first, in a shell of its own
    k_abs = np.concatenate([[0.0], k_abs])
    shell = np.concatenate([[0], shell])
    n_rows = n_shells + 1
    n_vec = np.bincount(shell, minlength=n_rows)
    with np.errstate(invalid="ignore", divide="ignore"):
        k_mean = np.bincount(shell, k_abs, minlength=n_rows) / n_vec
    per_state: dict[str, list[np.ndarray]] = {"initial": [], "final": []}
    for s in seeds:
        state = load_record(state_path(run_dir, s))
        both = np.stack([state["spins_initial"], state["spins_final"]])
        values = spin_structure_factor(state["positions"], both, kvec)
        for row, name in enumerate(("initial", "final")):
            with np.errstate(invalid="ignore", divide="ignore"):
                per_state[name].append(np.bincount(shell, values[row], minlength=n_rows) / n_vec)
    edges = np.linspace(0.0, k_max, n_shells + 1)
    data: dict[str, Any] = {
        "k_rnn_lo": np.concatenate([[0.0], edges[:-1]]) * r_nn,
        "k_rnn_hi": np.concatenate([[0.0], edges[1:]]) * r_nn,
        "k_rnn_mean": k_mean * r_nn,
        "n_vectors": n_vec,
    }
    with np.errstate(divide="ignore"):
        data["wavelength_rnn"] = np.where(k_mean > 0, 2.0 * np.pi / (k_mean * r_nn), np.nan)
    for name in ("final", "initial"):
        mean, sem, _ = mean_sem(np.array(per_state[name]))
        data[f"S_{name}"], data[f"sem_S_{name}"] = mean, sem
    frame = pd.DataFrame(data)[n_vec > 0].reset_index(drop=True)
    frame.attrs["meta"] = {
        "n_states": len(seeds),
        "seeds": [int(s) for s in seeds],
        "n_max": int(n_max),
        "n_shells": int(n_shells),
        "max_per_shell": max_per_shell,
        "n_vectors": int(n_vec.sum() - 1),
        "k_max_rnn": float(k_max * r_nn),
        "box": float(box),
        "r_nn": float(r_nn),
    }
    return frame


# --------------------------------------------------------------------------------------
# Comparing runs
# --------------------------------------------------------------------------------------
def same_system(a: SimulationConfig, b: SimulationConfig) -> bool:
    """Whether two configurations give identical positions, spin lengths and initial directions.

    Only the fields in :data:`PAIRING_KEYS` enter the random stream of a seed; runs that agree on
    them (and share seeds) can be compared pairwise even if coupling, cutoff, boundary condition,
    number of sweeps or protocol differ.
    """
    return all(getattr(a, key) == getattr(b, key) for key in PAIRING_KEYS)


def common_rows(reference: EnsembleData, other: EnsembleData) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of the seeds present in both runs, in the same (ascending seed) order."""
    common = np.intersect1d(reference.seeds, other.seeds)

    def rows(seeds: np.ndarray) -> np.ndarray:
        return np.array([int(np.where(seeds == s)[0][0]) for s in common], dtype=int)

    return rows(reference.seeds), rows(other.seeds)


def is_paired(reference: EnsembleData, other: EnsembleData) -> bool:
    """Runs are paired when they share the system (:func:`same_system`) and at least two seeds."""
    return (
        same_system(reference.config, other.config) and common_rows(reference, other)[0].size >= 2
    )


def paired_difference(
    reference: EnsembleData,
    other: EnsembleData,
    obs: str = "C",
    rebin: int = 1,
    paired: bool | None = None,
) -> pd.DataFrame:
    """Difference ``obs_other - obs_reference`` per bin, paired by seed when the runs allow it.

    *Paired* (:func:`is_paired`: same system, common seeds): only the common seeds are used and the
    difference is formed per realization before averaging. Since paired realizations start from
    identical positions and initial directions, most of the sampling noise cancels. Otherwise an
    *unpaired* difference of the two ensemble means over all realizations of both runs is returned
    with the error ``sqrt(sem_reference^2 + sem_other^2)``; ``paired=True`` raises instead. The
    column ``paired`` records which estimate the frame holds (``n_common`` is 0 when unpaired).
    """
    if (
        reference.config.n_bins != other.config.n_bins
        or reference.config.x_max != other.config.x_max
    ):
        raise ValueError("runs must use the same radial binning to be compared")
    can_pair = is_paired(reference, other)
    if paired is None:
        paired = can_pair
    elif paired and not can_pair:
        raise ValueError("runs are not paired (different system or fewer than two common seeds)")
    a = reference.realization_means(obs, "final", rebin)
    b = other.realization_means(obs, "final", rebin)
    if paired:
        ia, ib = common_rows(reference, other)
        mean, sem, n_common = mean_sem(b[ib] - a[ia])
        n_a = n_b = n_common
    else:
        mean_a, sem_a, n_a = mean_sem(a)
        mean_b, sem_b, n_b = mean_sem(b)
        mean = mean_b - mean_a
        sem = np.sqrt(sem_a**2 + sem_b**2)
        n_common = np.zeros_like(n_a)
    with np.errstate(invalid="ignore", divide="ignore"):
        significance = np.abs(mean) / sem
    return pd.DataFrame(
        {
            "x_center": reference.x_center(rebin),
            "x_mean": reference.x_mean(rebin),
            "diff": mean,
            "sem": sem,
            "abs_diff_over_sem": significance,
            "n_common": n_common,
            "n_reference": n_a,
            "n_other": n_b,
            "paired": paired,
        }
    )


def landmark_difference(
    reference: EnsembleData,
    other: EnsembleData,
    rebin: int = 1,
    n_boot: int = 200,
    seed: int = 12345,
    abscissa: str = "x_mean",
) -> dict[str, Any]:
    """Landmarks of ``other`` minus those of ``reference`` with bootstrap errors.

    Paired runs (:func:`is_paired`) are restricted to their common seeds and resampled with the same
    realization indices, so the error is that of the *shift*; otherwise both runs are resampled
    independently over all their realizations. Each key holds ``value`` (the difference), ``err``,
    percentiles and the two individual values ``reference`` and ``other``; ``_meta`` records whether
    the comparison was paired and how many realizations entered.
    """
    paired = is_paired(reference, other)
    if paired:
        ia, ib = common_rows(reference, other)
    else:
        ia, ib = np.arange(reference.n_real), np.arange(other.n_real)
    curves_a = reference._landmark_curves(rebin, ia)
    curves_b = other._landmark_curves(rebin, ib)
    keys = [o for o in curves_a if o in curves_b]
    x_a = reference.abscissa(rebin, abscissa, rows=ia)
    x_b = other.abscissa(rebin, abscissa, rows=ib)
    f_a = _features(x_a, {o: mean_sem(curves_a[o])[0] for o in keys})
    f_b = _features(x_b, {o: mean_sem(curves_b[o])[0] for o in keys})
    central = {k: f_b[k] - f_a[k] for k in f_a if k in f_b}
    rng = np.random.default_rng(seed)
    idx_a = rng.integers(0, ia.size, size=(n_boot, ia.size))
    idx_b = idx_a if paired else rng.integers(0, ib.size, size=(n_boot, ib.size))
    boot_a = {o: _resampled_means(curves_a[o], idx_a) for o in keys}
    boot_b = {o: _resampled_means(curves_b[o], idx_b) for o in keys}
    samples = []
    for k in range(n_boot):
        s_a = _features(x_a, {o: boot_a[o][k] for o in keys})
        s_b = _features(x_b, {o: boot_b[o][k] for o in keys})
        samples.append({key: s_b[key] - s_a[key] for key in central})
    result = _with_bootstrap_errors(central, samples)
    for key in central:
        result[key]["reference"] = float(f_a[key])
        result[key]["other"] = float(f_b[key])
    result["_meta"] = {
        "paired": paired,
        "n_common": int(ia.size) if paired else 0,
        "n_reference": int(ia.size),
        "n_other": int(ib.size),
        "rebin": rebin,
        "n_boot": n_boot,
        "abscissa": reference.abscissa_used(abscissa),
    }
    return result


def landmark_table(
    runs: dict[str, EnsembleData],
    rebin: int = 1,
    n_boot: int = 200,
    seed: int = 12345,
    abscissa: str = "x_mean",
) -> pd.DataFrame:
    """Landmarks (value and bootstrap error) of several runs, one row per run."""
    rows = []
    for label, data in runs.items():
        marks = data.landmarks(rebin, n_boot, seed, abscissa)
        row: dict[str, Any] = {
            "run": label,
            "n_particles": data.config.n_particles,
            "boundary": data.config.boundary,
            "coupling": data.config.coupling,
            "spin_distribution": data.config.spin_distribution,
            "r_cut_nn": data.config.r_cut_nn_resolved,
            "n_sweeps": data.config.n_sweeps,
            "n_real": data.n_real,
            "rebin": rebin,
            "abscissa": marks["_meta"]["abscissa"],
        }
        for key, info in marks.items():
            if key.startswith("_"):
                continue
            row[key] = info["value"]
            row[f"{key}_err"] = info["err"]
        for name, stat in data.scalar_summary().items():
            if name in ("total_spin_over_sqrtN_final", "S0_unit_final"):
                row[name] = stat["mean"]
                row[f"{name}_err"] = stat["sem"]
        rows.append(row)
    return pd.DataFrame(rows)


def landmark_difference_table(
    runs: dict[str, EnsembleData],
    rebin: int = 1,
    n_boot: int = 200,
    seed: int = 12345,
    abscissa: str = "x_mean",
) -> pd.DataFrame:
    """Landmark shifts of every run against the first one (``d_<key>``, ``d_<key>_err``).

    The column ``paired`` says whether the shift and its error come from the common seeds
    (:func:`landmark_difference`).
    """
    labels = list(runs)
    reference = runs[labels[0]]
    rows = []
    for label in labels[1:]:
        diff = landmark_difference(reference, runs[label], rebin, n_boot, seed, abscissa)
        meta = diff["_meta"]
        row: dict[str, Any] = {
            "run": label,
            "reference": labels[0],
            "paired": meta["paired"],
            "n_common": meta["n_common"],
            "n_reference": meta["n_reference"],
            "n_other": meta["n_other"],
        }
        for key, info in diff.items():
            if key.startswith("_"):
                continue
            row[f"d_{key}"] = info["value"]
            row[f"d_{key}_err"] = info["err"]
        rows.append(row)
    return pd.DataFrame(rows)


def curve_value_at(
    data: EnsembleData, obs: str, x: float, rebin: int = 1, abscissa: str = "x_mean"
) -> tuple[float, float]:
    """Ensemble mean of ``obs`` (final state) interpolated linearly at ``x`` and its standard error.

    The mean over realizations and its SEM (:func:`mean_sem` of the per-realization bin means) are
    both interpolated on ``abscissa`` (``x_mean`` or ``x_center``). ``(nan, nan)`` outside the
    tabulated range or when fewer than two bins are defined.
    """
    mean, sem, _ = mean_sem(data.realization_means(obs, "final", rebin))
    xs = data.abscissa(rebin, data.abscissa_used(abscissa))
    ok = np.isfinite(xs) & np.isfinite(mean)
    if ok.sum() < 2 or not (xs[ok].min() <= x <= xs[ok].max()):
        return float("nan"), float("nan")
    order = np.argsort(xs[ok])
    xs, mean, sem = xs[ok][order], mean[ok][order], sem[ok][order]
    return float(np.interp(x, xs, mean)), float(np.interp(x, xs, sem))


def gamma_family_members(runs: dict[str, EnsembleData]) -> list[str] | None:
    """Labels of ``runs`` ordered by ``gamma`` when they form the channel (``gamma``) family.

    The family consists of couplings with ``sigma = +1`` (:data:`grspin.config.COUPLINGS`:
    ``transverse_only``, ``gr``, ``gamma12``, ``longitudinal_only``) and at least two distinct
    values of ``gamma``; ``None`` otherwise (e.g. the controls ``S6`` with ``sigma = -1``, or a
    comparison of one coupling at several cutoffs).
    """
    gammas = {}
    for label, data in runs.items():
        sigma, gamma = data.config.sigma_gamma
        if sigma <= 0.0:
            return None
        gammas[label] = float(gamma)
    if len(set(gammas.values())) < 2:
        return None
    return sorted(gammas, key=lambda label: (gammas[label], list(runs).index(label)))


def landmark_error_budget(
    main: EnsembleData,
    sweeps_run: EnsembleData | None = None,
    cutoff_run: EnsembleData | None = None,
    rebins: tuple[int, ...] = (1, 2),
    n_boot: int = 200,
    seed: int = 12345,
    abscissa: str = "x_mean",
) -> pd.DataFrame:
    """Statistical and systematic uncertainties of the landmarks of ``main``, one row per landmark.

    Columns: ``value`` and ``stat`` are the landmark at ``rebins[0]`` on ``abscissa`` and its
    bootstrap error; ``binning`` is ``|value(rebins[0]) - value(rebins[1])|`` (NaN with a single
    rebin); ``sweeps`` and ``sweeps_err`` are the paired shift ``landmark(sweeps_run) -
    landmark(main)`` on the common seeds (``sweeps_run`` relaxes the same realizations for more
    sweeps) and its paired bootstrap error; ``cutoff`` and ``cutoff_err`` the same for a run with a
    larger interaction cutoff; ``total = sqrt(stat^2 + binning^2 + sweeps^2 + cutoff^2)`` with
    missing components counting as zero; ``rounded`` is ``value ± total`` at the precision of
    ``total``. A run that is not paired with ``main``, or that does not relax longer / cut farther,
    raises ``ValueError``; ``None`` skips a component. ``frame.attrs["meta"]`` records the runs and
    seeds used.
    """
    rebin = rebins[0]
    marks = main.landmarks(rebin, n_boot, seed, abscissa)
    coarse = main.landmarks(rebins[1], n_boot, seed, abscissa) if len(rebins) > 1 else None
    meta: dict[str, Any] = {
        "rebins": list(rebins),
        "bin_widths": [main.bin_width(r) for r in rebins],
        "abscissa": marks["_meta"]["abscissa"],
        "n_real": main.n_real,
        "n_boot": n_boot,
    }
    shifts: dict[str, dict[str, Any]] = {}
    for name, run, attr, need in (
        ("sweeps", sweeps_run, "n_sweeps", "a longer relaxation (more sweeps)"),
        ("cutoff", cutoff_run, "r_cut_nn_resolved", "a larger cutoff"),
    ):
        if run is None:
            continue
        if not getattr(run.config, attr) > getattr(main.config, attr):
            raise ValueError(
                f"{name}_run must have {need} than the main run ({attr}: "
                f"{getattr(run.config, attr):g} versus {getattr(main.config, attr):g})"
            )
        diff = landmark_difference(main, run, rebin, n_boot, seed, abscissa)
        if not diff["_meta"]["paired"]:
            raise ValueError(f"{name}_run is not paired with the main run (system or seeds differ)")
        shifts[name] = diff
        meta[name] = {
            "main": float(getattr(main.config, attr)),
            "run": float(getattr(run.config, attr)),
            "n_common": diff["_meta"]["n_common"],
        }
    rows = []
    for key, info in marks.items():
        if key.startswith("_"):
            continue
        value, stat = info["value"], info["err"]
        row: dict[str, Any] = {"landmark": key, "value": value, "stat": stat}
        row["binning"] = (
            abs(value - coarse[key]["value"]) if coarse and key in coarse else float("nan")
        )
        for name in ("sweeps", "cutoff"):
            shift = shifts.get(name, {}).get(key)
            row[name] = shift["value"] if shift else float("nan")
            row[f"{name}_err"] = shift["err"] if shift else float("nan")
        parts = np.array([stat, row["binning"], row["sweeps"], row["cutoff"]], dtype=float)
        finite = np.isfinite(value) and np.isfinite(stat)
        row["total"] = float(np.sqrt(np.nansum(parts**2))) if finite else float("nan")
        row["rounded"] = round_to_error(value, row["total"])
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.attrs["meta"] = meta
    return frame
