"""Damped-oscillatory fit: recovery of known parameters, failure modes, bootstrap output."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import synthetic_ensemble

from grspin.fit import FitResult, damped_oscillatory, fit_damped_oscillatory, fit_ensemble, u_of_x


def test_fit_recovers_known_parameters_from_noisy_data():
    x = np.linspace(0.05, 3.95, 40)
    truth = (2.7, 2.9, 0.1)
    rng = np.random.default_rng(5)
    sem = np.full_like(x, 0.003)
    y = damped_oscillatory(u_of_x(x), *truth) + sem * rng.normal(size=x.size)
    fit = fit_damped_oscillatory(x, y, sem)
    assert fit.success and fit.n_points == 40
    for value, expected, err in (
        (fit.alpha, truth[0], fit.alpha_err),
        (fit.k, truth[1], fit.k_err),
        (fit.phi, truth[2], fit.phi_err),
    ):
        assert abs(value - expected) < 4 * err + 1e-3
    assert 0.4 < fit.chi2 / fit.dof < 2.0  # the model is exact here
    assert np.allclose(fit.curve(x), damped_oscillatory(u_of_x(x), fit.alpha, fit.k, fit.phi))


def test_fit_range_and_missing_points_are_respected():
    x = np.linspace(0.05, 3.95, 40)
    y = damped_oscillatory(u_of_x(x), 3.0, 3.0, 0.0)
    y[5] = np.nan
    fit = fit_damped_oscillatory(x, y, None, x_range=(0.3, 3.0))
    assert fit.success and fit.n_points == int(np.sum((x >= 0.3) & (x <= 3.0)) - 1)
    assert fit.x_min == 0.3 and fit.x_max == 3.0
    assert fit.alpha == pytest.approx(3.0, abs=1e-3) and fit.k == pytest.approx(3.0, abs=1e-3)


def test_fit_fails_gracefully_with_too_few_points():
    x = np.linspace(0.1, 1.0, 4)
    fit = fit_damped_oscillatory(x, np.sin(x))
    assert isinstance(fit, FitResult) and not fit.success and np.isnan(fit.alpha)
    assert fit.as_dict()["n_points"] == 4


def test_ensemble_fit_with_bootstrap_on_a_synthetic_ensemble():
    data, _, _ = synthetic_ensemble(n_real=30, noise=0.01, params=(2.7, 2.9, 0.1))
    result = fit_ensemble(data, n_boot=40)
    fit = result["fit"]
    assert fit["success"] and result["n_real"] == 30
    assert fit["alpha"] == pytest.approx(2.7, abs=0.15) and fit["k"] == pytest.approx(2.9, abs=0.15)
    for name in ("alpha", "k", "phi"):
        boot = result[f"{name}_bootstrap"]
        assert boot["p16"] <= boot["median"] <= boot["p84"] and boot["std"] >= 0.0
    restricted = fit_ensemble(data, rebin=2, x_range=(0.2, 3.0), n_boot=0)
    assert restricted["fit"]["success"] and "alpha_bootstrap" not in restricted
