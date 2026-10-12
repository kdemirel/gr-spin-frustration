"""Configuration: the coupling family and its two-body minima, the hard core, (de)serialization."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import minimize

from grspin import kernels
from grspin.config import (
    CHANNEL_FAMILY,
    COUPLINGS,
    MAX_MIN_SEPARATION_NN,
    SimulationConfig,
    pair_minimum_factor,
)
from grspin.neighbors import build_neighbor_list

RHAT = np.array([0.0, 0.0, 1.0])
EXPECTED_FACTORS = {
    "gr": 2.0,
    "isotropic_afm": 1.0,
    "dipolar": 2.0,
    "transverse_only": 1.0,
    "gamma12": 11.0,
    "longitudinal_only": 99.0,
}


# --------------------------------------------------------------------------------------
# Couplings
# --------------------------------------------------------------------------------------
def test_coupling_table_and_channel_family():
    assert set(COUPLINGS) == set(EXPECTED_FACTORS)
    assert COUPLINGS["gr"] == (1.0, 3.0) and COUPLINGS["isotropic_afm"] == (-1.0, 0.0)
    assert COUPLINGS["dipolar"] == (-1.0, 3.0)
    assert COUPLINGS["transverse_only"] == (1.0, 1.0)
    assert COUPLINGS["gamma12"] == (1.0, 12.0)
    assert COUPLINGS["longitudinal_only"] == (1.0, 100.0)
    # the channel family keeps sigma = +1 and increases the longitudinal weight gamma - 1
    assert CHANNEL_FAMILY == ("transverse_only", "gr", "gamma12", "longitudinal_only")
    gammas = [COUPLINGS[name][1] for name in CHANNEL_FAMILY]
    assert all(COUPLINGS[name][0] == 1.0 for name in CHANNEL_FAMILY)
    assert gammas == sorted(gammas) == [1.0, 3.0, 12.0, 100.0]
    # gamma = 100: the transverse channel is 1 % of the longitudinal one
    assert 1.0 / (COUPLINGS["longitudinal_only"][1] - 1.0) == pytest.approx(0.0101, abs=2e-4)
    for name, factor in EXPECTED_FACTORS.items():
        assert pair_minimum_factor(name) == factor == pair_minimum_factor(COUPLINGS[name])
    with pytest.raises(KeyError):
        pair_minimum_factor("heisenberg")


def _pair_energy(u_i: np.ndarray, u_j: np.ndarray, sigma: float, gamma: float) -> float:
    """Pair energy of unit spins at unit distance without ``G``: ``-sigma u_i^T M u_j``."""
    return -sigma * (u_i @ u_j - gamma * (u_i @ RHAT) * (u_j @ RHAT))


def _fibonacci_sphere(n: int) -> np.ndarray:
    k = np.arange(n) + 0.5
    polar = np.arccos(1.0 - 2.0 * k / n)
    azimuth = math.pi * (1.0 + math.sqrt(5.0)) * k
    return np.stack(
        [np.cos(azimuth) * np.sin(polar), np.sin(azimuth) * np.sin(polar), np.cos(polar)], axis=1
    )


def _angles_to_unit(angles: np.ndarray) -> np.ndarray:
    theta, phi = angles
    return np.array(
        [math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi), math.cos(theta)]
    )


@pytest.mark.parametrize("name", sorted(COUPLINGS))
def test_two_body_minimum_is_minus_the_spectral_norm_of_the_pair_tensor(name):
    """``min_{u_i, u_j} -sigma u_i^T M u_j = -max(1, |gamma - 1|)`` for either sign of sigma."""
    sigma, gamma = COUPLINGS[name]
    m = np.eye(3) - gamma * np.outer(RHAT, RHAT)
    expected = -pair_minimum_factor(name)
    # (1) for a fixed u_i the exact inner minimum over u_j is -|M u_i| (u_j = sigma M u_i / |M u_i|);
    #     check it against a dense grid over u_j for a few random u_i
    grid = _fibonacci_sphere(40000)
    rng = np.random.default_rng(0)
    for u_i in rng.normal(size=(5, 3)):
        u_i /= np.linalg.norm(u_i)
        inner = np.min(-sigma * grid @ (m @ u_i))
        assert inner == pytest.approx(-np.linalg.norm(m @ u_i), rel=3e-4)  # grid: ~0.01 rad
    # (2) the outer minimum over u_i on the grid, then polished over all four angles
    outer = -np.linalg.norm(grid @ m, axis=1)  # -|M u_i| for every grid direction (M symmetric)
    assert outer.min() == pytest.approx(expected, rel=2e-3)
    best = grid[np.argmin(outer)]
    u_j = sigma * (m @ best) / np.linalg.norm(m @ best)
    start = np.concatenate(
        [
            [math.acos(best[2]), math.atan2(best[1], best[0])],
            [math.acos(u_j[2]), math.atan2(u_j[1], u_j[0])],
        ]
    )
    result = minimize(
        lambda a: _pair_energy(_angles_to_unit(a[:2]), _angles_to_unit(a[2:]), sigma, gamma),
        start,
        method="Nelder-Mead",
        options={"xatol": 1e-10, "fatol": 1e-12, "maxiter": 4000},
    )
    assert result.fun == pytest.approx(expected, rel=1e-8)
    # a random search never goes below the bound
    pairs = rng.normal(size=(20000, 2, 3))
    pairs /= np.linalg.norm(pairs, axis=2, keepdims=True)
    energies = -sigma * np.einsum("ni,ij,nj->n", pairs[:, 0], m, pairs[:, 1])
    assert energies.min() >= expected - 1e-12


def _relax_two_body(sigma: float, gamma: float, sweeps: int = 80):
    """Greedy relaxation of two spins (lengths 0.7 and 1.3, distance 2) with the production kernel."""
    rng = np.random.default_rng(3)
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    nlist = build_neighbor_list(positions, r_cut=10.0, box=None)
    smag = np.array([0.7, 1.3])
    v = rng.normal(size=(2, 3))
    spins = smag[:, None] * v / np.linalg.norm(v, axis=1, keepdims=True)
    order = np.arange(2, dtype=np.int64)
    args = (nlist.indptr, nlist.indices, nlist.weight, nlist.rhat)
    for _ in range(sweeps):
        kernels.sweep(*args, spins, smag, order, sigma, gamma)
    return spins / smag[:, None], kernels.energy(*args, spins, sigma, gamma), smag


@pytest.mark.parametrize("name", sorted(COUPLINGS))
def test_two_body_relaxation_reaches_the_pair_minimum(name):
    sigma, gamma = COUPLINGS[name]
    u, energy, smag = _relax_two_body(sigma, gamma)
    assert energy == pytest.approx(
        -pair_minimum_factor(name) * smag[0] * smag[1] / 2.0**3, rel=1e-9
    )
    along = np.abs(u @ RHAT)
    if name == "transverse_only":  # parallel and transverse to the bond
        assert u[0] @ u[1] == pytest.approx(1.0, abs=1e-9) and np.all(along < 1e-9)
    elif name == "isotropic_afm":  # antiparallel along any axis
        assert u[0] @ u[1] == pytest.approx(-1.0, abs=1e-9)
    elif name == "dipolar":  # parallel along the bond
        assert u[0] @ u[1] == pytest.approx(1.0, abs=1e-9) and np.all(along > 1 - 1e-9)
    else:  # gr, gamma12, longitudinal_only: antiparallel along the bond
        assert u[0] @ u[1] == pytest.approx(-1.0, abs=1e-9) and np.all(along > 1 - 1e-9)


def test_gamma_100_is_the_longitudinal_field_up_to_a_one_percent_admixture():
    """``h(gamma) = T - gamma L``: for gamma = 100 the transverse sum ``T`` enters at the 1 % level."""
    cfg = SimulationConfig(n_particles=200, boundary="open", n_sweeps=0, n_bins=4, x_max=1.0)
    from grspin.initial import generate_initial_state

    positions, spins, _ = generate_initial_state(cfg, seed=5)
    nl = build_neighbor_list(positions, cfg.r_cut, None)
    args = (nl.indptr, nl.indices, nl.weight, nl.rhat, spins)
    h0, h1, h100 = (kernels.local_fields(*args, g) for g in (0.0, 1.0, 100.0))
    transverse_sum, longitudinal_sum = h0, h0 - h1  # T = sum w S_j, L = sum w (S_j . rhat) rhat
    assert np.allclose(h100, transverse_sum - 100.0 * longitudinal_sum, rtol=1e-12, atol=1e-12)
    ratio = np.linalg.norm(transverse_sum, axis=1) / (
        100.0 * np.linalg.norm(longitudinal_sum, axis=1)
    )
    assert np.median(ratio) < 0.02  # the admixture is of order 1 % of the longitudinal field
    cosine = np.einsum("ij,ij->i", h100, -longitudinal_sum) / (
        np.linalg.norm(h100, axis=1) * np.linalg.norm(longitudinal_sum, axis=1)
    )
    assert np.median(cosine) > 1.0 - 1e-3  # the gamma = 100 field points along -L to about 1 degree


# --------------------------------------------------------------------------------------
# Hard core
# --------------------------------------------------------------------------------------
def test_min_separation_is_validated_and_converted_to_absolute_units():
    cfg = SimulationConfig(n_particles=1000, min_separation_nn=0.5, r_cut_nn=5.0)
    assert cfg.min_separation == pytest.approx(0.5 * cfg.r_nn)
    assert SimulationConfig(n_particles=1000, r_cut_nn=5.0).min_separation == 0.0
    assert MAX_MIN_SEPARATION_NN == 0.7
    SimulationConfig(n_particles=1000, min_separation_nn=MAX_MIN_SEPARATION_NN, r_cut_nn=5.0)  # ok
    for bad in (-0.1, MAX_MIN_SEPARATION_NN + 0.01, float("nan")):
        with pytest.raises(ValueError, match="min_separation_nn"):
            SimulationConfig(n_particles=1000, min_separation_nn=bad, r_cut_nn=5.0)
    # the excluded volume stays small, so sequential addition terminates quickly (config docstring)
    excluded = 4.0 / 3.0 * math.pi * (MAX_MIN_SEPARATION_NN * cfg.r_nn) ** 3 * cfg.density_effective
    assert 0.2 < excluded < 0.25
    assert math.pi / 6.0 * (MAX_MIN_SEPARATION_NN * cfg.r_nn) ** 3 * cfg.density_effective < 0.035


def test_min_separation_round_trips_and_is_part_of_the_compat_key():
    cfg = SimulationConfig(n_particles=1000, min_separation_nn=0.5, r_cut_nn=5.0, snapshots=(3, 1))
    d = cfg.to_dict()
    assert d["min_separation_nn"] == 0.5 and d["coupling"] == "gr"
    assert SimulationConfig.from_dict(d) == cfg
    assert "min_separation_nn" in cfg.compat_key() and "min_separation_nn" in cfg.compat_key(
        "protocol"
    )
    poisson = cfg.replace(min_separation_nn=0.0)
    assert poisson.compat_key() != cfg.compat_key()
    assert cfg.replace(n_realizations=3, save_states=1).compat_key() == cfg.compat_key()
    # a config.json written before the field existed loads as the ideal Poisson process ...
    old = poisson.to_dict()
    del old["min_separation_nn"]
    loaded = SimulationConfig.from_dict(old)
    assert loaded == poisson and loaded.min_separation_nn == 0.0
    # ... and keeps matching the key of the same configuration written today
    assert loaded.compat_key() == poisson.compat_key()
    assert loaded.compat_key("protocol") == poisson.compat_key("protocol")


def test_cost_model_charges_the_sequential_addition():
    base = SimulationConfig(n_particles=5000, r_cut_nn=10.0)
    hard = base.replace(min_separation_nn=0.5)
    extra = hard.estimate_cost()["seconds"] - base.estimate_cost()["seconds"]
    assert 0.1 < extra < 0.3  # measured: ~0.18 s at N = 5000 (O(N^2))
    assert hard.estimate_cost()["memory_mb"] == base.estimate_cost()["memory_mb"]
    assert hard.estimate_cost()["edges"] == base.estimate_cost()["edges"]
    big = base.replace(n_particles=20000, min_separation_nn=0.5)
    assert big.estimate_cost()["seconds"] - base.replace(n_particles=20000).estimate_cost()[
        "seconds"
    ] == pytest.approx(16.0 * extra)
