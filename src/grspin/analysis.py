"""Ensemble analysis: re-binning, statistics over realizations, bootstrap, landmarks, diagnostics.

The statistics follow the definitions of the manuscript: per realization the binned mean of an
observable is formed over all pairs in the bin; the ensemble mean is the average of these means over
realizations and the error bar is the standard error over realizations
(``std(ddof=1) / sqrt(n)``, ignoring realizations without pairs in a bin). Since the runner stores
sums and counts per bin, the data can be re-binned exactly (any integer divisor of ``n_bins``).
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import theory
from .config import SimulationConfig
from .ensemble import SEEDS_DIRNAME, load_record, read_config, seed_path
from .nn import NN_HIST_EDGES_X

OBSERVABLES = ("C", "Q", "A", "P", "B")
_COUNT_OF = {"C": "C", "Q": "Q", "A": "C", "P": "C", "B": "C"}
_STATE_PREFIX = {"initial": "init", "final": "fin"}


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


def bootstrap_means(values: np.ndarray, n_boot: int, rng: np.random.Generator) -> np.ndarray:
    """Means over resampled rows (realizations) of ``values`` (shape ``(R, n_bins)``)."""
    r = values.shape[0]
    idx = rng.integers(0, r, size=(n_boot, r))
    sample = values[idx]  # (n_boot, R, n_bins)
    finite = np.isfinite(sample)
    n = finite.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, np.where(finite, sample, 0.0).sum(axis=1) / n, np.nan)


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
    """Vertex of the parabola through bins ``k-1, k, k+1`` (the bin itself at the edges)."""
    if k <= 0 or k >= len(y) - 1 or not np.all(np.isfinite(y[k - 1 : k + 2])):
        return float(x[k]), float(y[k])
    y0, y1, y2 = y[k - 1], y[k], y[k + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom == 0.0:
        return float(x[k]), float(y1)
    h = x[k] - x[k - 1]
    return float(x[k] + 0.5 * h * (y0 - y2) / denom), float(y1 - 0.125 * (y0 - y2) ** 2 / denom)


def landmark_values(
    x: np.ndarray, c: np.ndarray, q: np.ndarray, a: np.ndarray | None = None
) -> dict[str, float]:
    """Model-independent features of the mean curves ``C(x)``, ``Q(x)`` (and ``A(x)``)."""
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
    return out


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
        if not records:
            raise ValueError("no realizations")
        keys = records[0].keys()
        return cls(config, {k: np.stack([np.asarray(r[k]) for r in records]) for k in keys})

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

    # -- per-realization means and ensemble statistics --------------------------------
    def _sums_counts(self, obs: str, state: str, rebin: int, snapshot: int | None):
        prefix = "snap" if snapshot is not None else _STATE_PREFIX[state]
        sums = self.arrays[f"{prefix}_sum_{obs}"]
        counts = self.arrays[f"{prefix}_cnt_{_COUNT_OF[obs]}"]
        if snapshot is not None:
            sums, counts = sums[:, snapshot, :], counts[:, snapshot, :]
        return self._rebin(sums, rebin), self._rebin(counts, rebin)

    def realization_means(
        self, obs: str, state: str = "final", rebin: int = 1, snapshot: int | None = None
    ) -> np.ndarray:
        """Per-realization bin means, shape ``(R, n_bins // rebin)`` (NaN in empty bins)."""
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
        return [o for o in OBSERVABLES if self.has(f"fin_sum_{o}")]

    def table(self, rebin: int = 1) -> pd.DataFrame:
        """Ensemble table (one row per radial bin); column names extend the prototype's CSV."""
        edges = self.x_edges(rebin)
        x = self.x_center(rebin)
        n_bins = len(x)
        cfg = self.config
        cnt_c = self._rebin(self.arrays["fin_cnt_C"], rebin)
        sum_x = self._rebin(self.arrays["fin_sum_x"], rebin)
        with np.errstate(invalid="ignore", divide="ignore"):
            x_mean = np.where(cnt_c.sum(axis=0) > 0, sum_x.sum(axis=0) / cnt_c.sum(axis=0), np.nan)
        data: dict[str, Any] = {
            "bin": np.arange(n_bins),
            "x_left": edges[:-1],
            "x_right": edges[1:],
            "x_center": x,
            "x_mean": x_mean,
            "r_center": x * cfg.r_nn,
            "count_C_total_final": cnt_c.sum(axis=0),
            "count_Q_total_final": self._rebin(self.arrays["fin_cnt_Q"], rebin).sum(axis=0),
        }
        for obs in self.available_observables():
            for state in ("initial", "final"):
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

    def histories(self) -> pd.DataFrame:
        """Convergence history averaged over realizations."""
        energy = self.arrays["energy"]
        final = energy[:, -1:]
        with np.errstate(invalid="ignore", divide="ignore"):
            excess = (energy - final) / np.abs(final)
        m_e, s_e, _ = mean_sem(energy)
        m_x, s_x, _ = mean_sem(excess)
        m_d, _, _ = mean_sem(self.arrays["dmax"])
        m_s, s_s, _ = mean_sem(self.arrays["total_spin_hist"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            max_d = (
                np.nanmax(self.arrays["dmax"], axis=0)
                if energy.shape[1] > 1
                else np.array([np.nan])
            )
        return pd.DataFrame(
            {
                "sweep": np.arange(energy.shape[1]),
                "mean_energy": m_e,
                "sem_energy": s_e,
                "mean_excess_energy": m_x,
                "sem_excess_energy": s_x,
                "mean_max_direction_change": m_d,
                "max_max_direction_change": max_d,
                "mean_total_spin": m_s,
                "sem_total_spin": s_s,
            }
        )

    # -- bootstrap features ----------------------------------------------------------
    def landmarks(self, rebin: int = 1, n_boot: int = 200, seed: int = 12345) -> dict[str, Any]:
        """Model-independent features with bootstrap errors (resampling realizations)."""
        x = self.x_center(rebin)
        curves = {o: self.realization_means(o, "final", rebin) for o in ("C", "Q")}
        if self.has("fin_sum_A"):
            curves["A"] = self.realization_means("A", "final", rebin)

        def features(rows_c, rows_q, rows_a):
            return landmark_values(x, rows_c, rows_q, rows_a)

        central = features(
            mean_sem(curves["C"])[0],
            mean_sem(curves["Q"])[0],
            mean_sem(curves["A"])[0] if "A" in curves else None,
        )
        rng = np.random.default_rng(seed)
        boot = {o: bootstrap_means(v, n_boot, rng) for o, v in curves.items()}
        samples = [
            features(boot["C"][b], boot["Q"][b], boot["A"][b] if "A" in boot else None)
            for b in range(n_boot)
        ]
        result: dict[str, Any] = {}
        for key, value in central.items():
            vals = np.array([s.get(key, np.nan) for s in samples], dtype=float)
            finite = vals[np.isfinite(vals)]
            result[key] = {
                "value": value,
                "err": float(np.std(finite, ddof=1)) if finite.size > 2 else float("nan"),
                "p16": float(np.percentile(finite, 16)) if finite.size else float("nan"),
                "p84": float(np.percentile(finite, 84)) if finite.size else float("nan"),
                "n_valid_bootstrap": int(finite.size),
            }
        result["_meta"] = {
            "n_real": self.n_real,
            "rebin": rebin,
            "n_bins": len(x),
            "n_boot": n_boot,
        }
        return result

    def bootstrap_curves(
        self, obs: str = "C", rebin: int = 1, n_boot: int = 200, seed: int = 12345
    ) -> np.ndarray:
        """Bootstrap ensemble means of ``obs`` (shape ``(n_boot, n_bins)``)."""
        rng = np.random.default_rng(seed)
        return bootstrap_means(self.realization_means(obs, "final", rebin), n_boot, rng)

    # -- sweep convergence ---------------------------------------------------------------
    def sweep_convergence(self, rebin: int = 1) -> pd.DataFrame:
        """Deviation of the mean ``C(x)`` at recorded sweeps from the final one (needs snapshots)."""
        if not self.has("snap_sweeps"):
            raise ValueError("this run recorded no snapshots (set `snapshots=` in the config)")
        sweeps = [int(s) for s in self.arrays["snap_sweeps"][0]]
        ref_mean, ref_sem, _ = mean_sem(self.realization_means("C", "final", rebin))
        energy = self.arrays["energy"]
        final_e = energy[:, -1]
        rows = []
        for index, sweep in enumerate(sweeps):
            mean, sem, _ = mean_sem(self.realization_means("C", rebin=rebin, snapshot=index))
            diff = mean - ref_mean
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio = np.abs(diff) / ref_sem
            rows.append(
                {
                    "sweep": sweep,
                    "max_abs_dC": float(np.nanmax(np.abs(diff))),
                    "rms_dC": float(np.sqrt(np.nanmean(diff**2))),
                    "max_dC_over_sem": float(np.nanmax(ratio)),
                    "C_peak": float(np.nanmax(mean)),
                    "mean_excess_energy": float(
                        np.mean((energy[:, sweep] - final_e) / np.abs(final_e))
                    ),
                }
            )
        return pd.DataFrame(rows)

    # -- nearest-neighbour diagnostics -------------------------------------------------
    def nn_table(self, rebin: int = 1, n_boot: int = 100, seed: int = 12345) -> pd.DataFrame:
        """Dimer decomposition ``C = f_mut C_mut + (1 - f_mut) C_non`` (pooled over all pairs)."""
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
                with warnings.catch_warnings():  # bins without pairs of a kind are all-NaN
                    warnings.simplefilter("ignore", RuntimeWarning)
                    data[f"sem_{key}"] = np.nanstd([b[key] for b in boots], axis=0, ddof=1)
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


# --------------------------------------------------------------------------------------
# Comparing runs
# --------------------------------------------------------------------------------------
def paired_difference(
    reference: EnsembleData,
    other: EnsembleData,
    obs: str = "C",
    rebin: int = 1,
) -> pd.DataFrame:
    """Mean and standard error of ``obs_other - obs_reference`` per bin, paired by seed.

    Only realizations present in both runs are used. Runs that share ``base_seed`` and the system
    (``n_particles``, boundary) start from identical positions and initial directions, so most of
    the sampling noise cancels in the difference.
    """
    if (
        reference.config.n_bins != other.config.n_bins
        or reference.config.x_max != other.config.x_max
    ):
        raise ValueError("runs must use the same radial binning to be compared")
    common = np.intersect1d(reference.seeds, other.seeds)
    if common.size < 2:
        raise ValueError("fewer than two common seeds: runs are not paired")
    ia = (
        np.searchsorted(reference.seeds, common)
        if np.all(np.diff(reference.seeds) > 0)
        else np.array([int(np.where(reference.seeds == s)[0][0]) for s in common])
    )
    ib = (
        np.searchsorted(other.seeds, common)
        if np.all(np.diff(other.seeds) > 0)
        else np.array([int(np.where(other.seeds == s)[0][0]) for s in common])
    )
    diff = (
        other.realization_means(obs, "final", rebin)[ib]
        - reference.realization_means(obs, "final", rebin)[ia]
    )
    mean, sem, n = mean_sem(diff)
    with np.errstate(invalid="ignore", divide="ignore"):
        significance = np.abs(mean) / sem
    return pd.DataFrame(
        {
            "x_center": reference.x_center(rebin),
            "diff": mean,
            "sem": sem,
            "abs_diff_over_sem": significance,
            "n_common": n,
        }
    )


def landmark_table(
    runs: dict[str, EnsembleData], rebin: int = 1, n_boot: int = 200, seed: int = 12345
) -> pd.DataFrame:
    """Landmarks (value and bootstrap error) of several runs, one row per run."""
    rows = []
    for label, data in runs.items():
        marks = data.landmarks(rebin, n_boot, seed)
        row: dict[str, Any] = {
            "run": label,
            "n_particles": data.config.n_particles,
            "boundary": data.config.boundary,
            "r_cut_nn": data.config.r_cut_nn_resolved,
            "n_sweeps": data.config.n_sweeps,
            "n_real": data.n_real,
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
