"""Pair observables against brute-force O(N^2) references."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import brute_force_pairs

from grspin.config import SimulationConfig
from grspin.initial import generate_initial_state
from grspin.neighbors import build_neighbor_list
from grspin.observables import pair_statistics, spin_summary
from grspin.relax import relax


def legacy_pair_observables(positions, spins, r_bins):
    """Verbatim logic of the prototype ``compute_pair_observables_binned`` (all pairs)."""
    n = len(positions)
    nbins = len(r_bins) - 1
    i_idx, j_idx = np.triu_indices(n, k=1)
    r_vecs = positions[j_idx] - positions[i_idx]
    distances = np.linalg.norm(r_vecs, axis=1)
    valid_r = distances > 0.0
    r_hat = np.zeros_like(r_vecs)
    r_hat[valid_r] = r_vecs[valid_r] / distances[valid_r, None]
    spin_norms = np.linalg.norm(spins, axis=1)
    unit_spins = np.zeros_like(spins)
    nonzero = spin_norms > 0.0
    unit_spins[nonzero] = spins[nonzero] / spin_norms[nonzero, None]
    valid_pairs = nonzero[i_idx] & nonzero[j_idx]
    cos_theta = np.full(len(i_idx), np.nan)
    raw = np.einsum("ij,ij->i", unit_spins[i_idx], unit_spins[j_idx])
    cos_theta[valid_pairs] = np.clip(raw[valid_pairs], -1.0, 1.0)
    total = spins[i_idx] + spins[j_idx]
    tn = np.linalg.norm(total, axis=1)
    valid_total = valid_r & (tn > 0.0)
    cos_phi = np.full(len(i_idx), np.nan)
    num = np.einsum("ij,ij->i", total, r_hat)
    cos_phi[valid_total] = np.clip(num[valid_total] / tn[valid_total], -1.0, 1.0)
    cos_phi_sq = cos_phi**2
    cat = np.digitize(distances, r_bins) - 1

    def binned(values):
        ok = np.isfinite(values) & (cat >= 0) & (cat < nbins)
        cnt = np.bincount(cat[ok], minlength=nbins)
        sm = np.bincount(cat[ok], weights=values[ok], minlength=nbins)
        return cnt, sm

    return binned(cos_theta), binned(cos_phi_sq)


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_pair_statistics_match_the_bruteforce_prototype_logic(boundary):
    cfg = SimulationConfig(n_particles=90, boundary=boundary, n_bins=10, x_max=3.0, n_sweeps=6)
    positions, spins0, smag = generate_initial_state(cfg, seed=8)
    box = cfg.box if boundary == "periodic" else None
    nl = build_neighbor_list(positions, cfg.r_cut, box)
    spins = relax(spins0, smag, nl, cfg, seed=8).spins
    stats = pair_statistics(positions, spins, nl.tree, box, cfg.r_nn, cfg.x_edges, extras=True)

    r_bins = cfg.x_edges * cfg.r_nn
    if boundary == "open":
        (cnt_c, sum_c), (cnt_q, sum_q) = legacy_pair_observables(positions, spins, r_bins)
        assert np.array_equal(stats["cnt_C"], cnt_c) and np.array_equal(stats["cnt_Q"], cnt_q)
        assert np.allclose(stats["sum_C"], sum_c, rtol=1e-12, atol=1e-12)
        assert np.allclose(stats["sum_Q"], sum_q, rtol=1e-12, atol=1e-12)

    # independent evaluation of every observable (also periodic) from brute-force pairs
    i, j, d, r = brute_force_pairs(positions, box, r_bins[-1])
    rhat = d / r[:, None]
    unit = spins / np.linalg.norm(spins, axis=1, keepdims=True)
    cos_t = np.einsum("ij,ij->i", unit[i], unit[j])
    ai, aj = np.einsum("ij,ij->i", unit[i], rhat), np.einsum("ij,ij->i", unit[j], rhat)
    tot = spins[i] + spins[j]
    cphi2 = (np.einsum("ij,ij->i", tot, rhat) / np.linalg.norm(tot, axis=1)) ** 2
    cat = np.digitize(r, r_bins) - 1
    ok = (cat >= 0) & (cat < cfg.n_bins)

    def binned(values):
        return np.bincount(cat[ok], weights=values[ok], minlength=cfg.n_bins)

    assert np.array_equal(stats["cnt_C"], np.bincount(cat[ok], minlength=cfg.n_bins))
    assert np.allclose(stats["sum_C"], binned(cos_t), atol=1e-12)
    assert np.allclose(stats["sum_Q"], binned(cphi2), atol=1e-12)
    assert np.allclose(stats["sum_A"], binned(0.5 * (ai**2 + aj**2)), atol=1e-12)
    assert np.allclose(stats["sum_P"], binned(ai * aj), atol=1e-12)
    assert np.allclose(stats["sum_B"], binned(cos_t - 3 * ai * aj), atol=1e-12)
    assert np.allclose(stats["sum_x"], binned(r / cfg.r_nn), atol=1e-12)


def test_perfect_dimer_has_the_expected_observables():
    """Two anti-parallel spins along the bond: C = -1, P = -1, A = 1, B = 2, Q = 1 (unequal lengths)."""
    cfg = SimulationConfig(n_particles=2, box_length=1.0, boundary="open", n_bins=1, x_max=10.0)
    positions = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.8]])
    spins = np.array([[0.0, 0.0, 0.6], [0.0, 0.0, -1.1]])
    nl = build_neighbor_list(positions, r_cut=2.0, box=None)
    st = pair_statistics(
        positions, spins, nl.tree, None, cfg.r_nn, np.array([0.0, 10.0]), extras=True
    )
    assert st["cnt_C"][0] == 1
    assert st["sum_C"][0] == pytest.approx(-1.0)
    assert st["sum_P"][0] == pytest.approx(-1.0)
    assert st["sum_A"][0] == pytest.approx(1.0)
    assert st["sum_B"][0] == pytest.approx(2.0)
    assert st["sum_Q"][0] == pytest.approx(1.0)


def test_equal_length_dimer_has_undefined_total_spin_direction():
    cfg = SimulationConfig(n_particles=2, box_length=1.0, boundary="open", n_bins=1, x_max=10.0)
    positions = np.array([[0.1, 0.2, 0.3], [0.1, 0.2, 0.8]])
    spins = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]])
    nl = build_neighbor_list(positions, r_cut=2.0, box=None)
    st = pair_statistics(positions, spins, nl.tree, None, cfg.r_nn, np.array([0.0, 10.0]))
    assert st["cnt_C"][0] == 1 and st["cnt_Q"][0] == 0  # pair excluded from Q (zero total spin)


def test_random_isotropic_spins_have_the_isotropic_baselines():
    cfg = SimulationConfig(n_particles=1200, n_bins=4, x_max=4.0)
    positions, spins, _ = generate_initial_state(cfg, seed=21)
    tree = build_neighbor_list(positions, cfg.r_cut, cfg.box).tree
    st = pair_statistics(positions, spins, tree, cfg.box, cfg.r_nn, cfg.x_edges, extras=True)
    n = st["cnt_C"].sum()
    assert n > 10000
    assert abs(st["sum_C"].sum() / n) < 0.02
    assert abs(st["sum_Q"].sum() / st["cnt_Q"].sum() - 1 / 3) < 0.02
    assert abs(st["sum_A"].sum() / n - 1 / 3) < 0.01
    assert abs(st["sum_P"].sum() / n) < 0.01 and abs(st["sum_B"].sum() / n) < 0.02


def test_spin_summary_ferromagnet_and_random():
    ferro = np.tile([0.0, 0.0, 2.0], (100, 1))
    s = spin_summary(ferro)
    assert s["total_spin"] == pytest.approx(200.0) and s["s0_unit"] == pytest.approx(100.0)
    # S(0) of a single random configuration fluctuates by ~0.8 (a scaled chi^2 with 3 degrees of
    # freedom); only its mean over configurations equals 1 for uncorrelated directions.
    rng = np.random.default_rng(1)
    values = []
    for _ in range(400):
        v = rng.normal(size=(50, 3))
        values.append(spin_summary(v / np.linalg.norm(v, axis=1, keepdims=True))["s0_unit"])
    assert abs(np.mean(values) - 1.0) < 0.12
