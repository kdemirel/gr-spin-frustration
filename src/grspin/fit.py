"""Damped-oscillatory description of ``C(r)`` and its bootstrap uncertainty.

``C_fit(r) = -exp(-alpha u^3) cos(k u + phi)`` with ``u = rho^(1/3) r = RNN_UNIT_DENSITY * x`` and
``x = r / r_nn``. This is a *phenomenological* form: it captures the negative core, the sign change
and the positive lobe, but with sizeable systematic residuals, and its parameters are not
individually well determined (the ``chi^2`` surface has several nearby minima). Hence the fit uses a
multi-start search and reports bootstrap percentiles; model-independent landmarks
(:meth:`grspin.analysis.EnsembleData.landmarks`) should be preferred for quantitative statements.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import curve_fit

from .analysis import EnsembleData, bootstrap_means, mean_sem
from .config import RNN_UNIT_DENSITY

_BOUNDS = ([0.0, 0.0, -2.0 * np.pi], [50.0, 50.0, 2.0 * np.pi])
_GRID_STARTS = [
    (a, k, p)
    for a in (0.5, 1.0, 2.0, 3.0, 5.0)
    for k in (1.0, 2.0, 3.0, 4.0, 6.0)
    for p in (-0.5, 0.0, 0.5)
]
_BOOTSTRAP_EXTRA_STARTS = [
    (1.0, 3.0, 0.0),
    (3.0, 3.0, 0.0),
    (2.6, 2.8, 0.2),
    (3.5, 3.0, 0.0),
    (2.0, 2.0, 0.3),
]


def u_of_x(x: np.ndarray | float) -> np.ndarray | float:
    """``u = rho^(1/3) r`` for ``x = r / r_nn``."""
    return RNN_UNIT_DENSITY * np.asarray(x, dtype=float)


def damped_oscillatory(u: np.ndarray, alpha: float, k: float, phi: float) -> np.ndarray:
    """``-exp(-alpha u^3) cos(k u + phi)``."""
    return -np.exp(-alpha * u**3) * np.cos(k * u + phi)


@dataclass
class FitResult:
    success: bool
    alpha: float
    k: float
    phi: float
    alpha_err: float
    k_err: float
    phi_err: float
    chi2: float
    dof: int
    rms_residual: float
    max_abs_residual: float
    n_points: int
    x_min: float
    x_max: float

    def as_dict(self) -> dict:
        return asdict(self)

    def curve(self, x: np.ndarray) -> np.ndarray:
        return damped_oscillatory(u_of_x(x), self.alpha, self.k, self.phi)


def _weights(sem: np.ndarray) -> np.ndarray:
    good = np.isfinite(sem) & (sem > 0.0)
    fallback = float(np.median(sem[good])) if good.any() else 1.0
    return np.where(good, sem, fallback)


def fit_damped_oscillatory(
    x: np.ndarray,
    y: np.ndarray,
    sem: np.ndarray | None = None,
    x_range: tuple[float, float] | None = None,
    starts: list[tuple[float, float, float]] | None = None,
) -> FitResult:
    """Weighted least-squares fit with a multi-start search (lowest ``chi^2`` wins)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    sigma = _weights(np.full_like(y, np.nan) if sem is None else np.asarray(sem, dtype=float))
    lo, hi = (x.min(), x.max()) if x_range is None else x_range
    mask = np.isfinite(y) & (x >= lo) & (x <= hi)
    fail = FitResult(
        False,
        *(float("nan"),) * 6,
        float("nan"),
        0,
        float("nan"),
        float("nan"),
        int(mask.sum()),
        lo,
        hi,
    )
    if mask.sum() < 5:
        return fail
    u = u_of_x(x[mask])
    best = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for p0 in starts or _GRID_STARTS:
            try:
                popt, pcov = curve_fit(
                    damped_oscillatory,
                    u,
                    y[mask],
                    p0=p0,
                    sigma=sigma[mask],
                    absolute_sigma=True,
                    bounds=_BOUNDS,
                    maxfev=20000,
                )
            except (RuntimeError, ValueError):
                continue
            chi2 = float(np.sum(((y[mask] - damped_oscillatory(u, *popt)) / sigma[mask]) ** 2))
            if best is None or chi2 < best[0]:
                best = (chi2, popt, pcov)
    if best is None:
        return fail
    chi2, popt, pcov = best
    resid = y[mask] - damped_oscillatory(u, *popt)
    err = np.sqrt(np.diag(pcov)) if np.all(np.isfinite(pcov)) else np.full(3, np.nan)
    return FitResult(
        True,
        *map(float, popt),
        *map(float, err),
        chi2,
        int(mask.sum() - 3),
        float(np.sqrt(np.mean(resid**2))),
        float(np.max(np.abs(resid))),
        int(mask.sum()),
        float(lo),
        float(hi),
    )


def fit_ensemble(
    data: EnsembleData,
    rebin: int = 1,
    x_range: tuple[float, float] | None = None,
    n_boot: int = 200,
    seed: int = 12345,
) -> dict:
    """Fit the ensemble-mean ``C(x)`` and bootstrap the parameters over realizations."""
    x = data.x_center(rebin)
    per_real = data.realization_means("C", "final", rebin)
    mean, sem, _ = mean_sem(per_real)
    best = fit_damped_oscillatory(x, mean, sem, x_range)
    result: dict = {"fit": best.as_dict(), "rebin": rebin, "n_real": data.n_real, "n_boot": n_boot}
    if not best.success or n_boot < 2:
        return result
    starts = [(best.alpha, best.k, best.phi), *_BOOTSTRAP_EXTRA_STARTS]
    rng = np.random.default_rng(seed)
    boot_means = bootstrap_means(per_real, n_boot, rng)
    samples = []
    for b in range(n_boot):
        sem_b = sem  # weights fixed to the full-sample standard errors
        fb = fit_damped_oscillatory(x, boot_means[b], sem_b, x_range, starts)
        if fb.success:
            samples.append((fb.alpha, fb.k, fb.phi, fb.chi2 / max(fb.dof, 1)))
    samples_arr = np.array(samples)
    if len(samples_arr) > 2:
        for i, name in enumerate(("alpha", "k", "phi")):
            col = samples_arr[:, i]
            result[f"{name}_bootstrap"] = {
                "median": float(np.median(col)),
                "p16": float(np.percentile(col, 16)),
                "p84": float(np.percentile(col, 84)),
                "std": float(np.std(col, ddof=1)),
            }
    return result
