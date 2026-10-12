"""Exact Poisson-process results used as references."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy.spatial import cKDTree

from grspin import theory
from grspin.config import RNN_UNIT_DENSITY


def test_union_volume_of_two_balls_gives_the_mutual_nn_exponent():
    """Two balls of radius r with centres r apart: union = (9 pi / 4) r^3."""
    r = 1.3
    lens = np.pi * (4 * r + r) * (2 * r - r) ** 2 / 12.0  # lens volume for centre distance d = r
    union = 2 * (4 * np.pi / 3) * r**3 - lens
    assert union == pytest.approx(9 * np.pi / 4 * r**3)
    # the same number from numerical integration of the cap volumes
    cap, _ = integrate.quad(lambda z: np.pi * (r**2 - z**2), r / 2, r)
    assert 2 * cap == pytest.approx(lens)


def test_nn_distance_pdf_is_normalized_with_unit_mean():
    value, _ = integrate.quad(theory.nn_distance_pdf, 0, np.inf)
    mean, _ = integrate.quad(lambda x: x * theory.nn_distance_pdf(x), 0, np.inf)
    assert value == pytest.approx(1.0, abs=1e-9)
    assert mean == pytest.approx(1.0, abs=1e-9)  # r_nn is *defined* as the mean NN distance
    x = np.linspace(0, 3, 7)  # P(no neighbour within r) = 1 - CDF of the NN distance
    cdf = 1 - theory.isolated_sphere_probability(x)
    for a, b, increment in zip(x[:-1], x[1:], np.diff(cdf), strict=True):
        assert integrate.quad(theory.nn_distance_pdf, a, b)[0] == pytest.approx(increment, rel=1e-8)


def test_probabilities_are_ordered_and_limits_are_correct():
    x = np.array([0.0, 0.5, 1.0, 2.0])
    iso, mut = theory.isolated_sphere_probability(x), theory.mutual_nn_probability(x)
    assert iso[0] == mut[0] == 1.0 and np.all(np.diff(iso) < 0) and np.all(mut[1:] < iso[1:])
    assert theory.MUTUAL_PARTICLE_FRACTION == pytest.approx(16 / 27)


def test_mean_nearest_neighbour_distance_of_a_simulated_poisson_process():
    rng = np.random.default_rng(0)
    n = 20000
    box = n ** (1 / 3)  # unit density
    tree = cKDTree(rng.uniform(0, box, size=(n, 3)), boxsize=box)
    d = tree.query(tree.data, k=2)[0][:, 1]
    assert d.mean() == pytest.approx(RNN_UNIT_DENSITY, rel=0.01)


def test_two_body_minimum_factor_is_the_config_pair_minimum_factor():
    """One source of truth: ``theory.two_body_minimum_factor`` delegates to ``config``."""
    from grspin.config import COUPLINGS, pair_minimum_factor

    for name, (sigma, gamma) in COUPLINGS.items():
        assert theory.two_body_minimum_factor(sigma, gamma) == pair_minimum_factor(name)
    assert (
        theory.two_body_minimum_factor(-1.0, 7.0) == theory.two_body_minimum_factor(1.0, 7.0) == 6.0
    )
    assert theory.two_body_minimum_factor(1.0, 1.5) == 1.0  # transverse branch
    assert "Poisson" in theory.mutual_nn_probability.__doc__
    assert "Poisson" in theory.nn_distance_pdf.__doc__
