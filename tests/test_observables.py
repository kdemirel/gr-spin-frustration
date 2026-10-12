"""Pair observables against brute-force O(N^2) references."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import brute_force_pairs
from scipy.optimize import minimize

from grspin import kernels
from grspin.config import SimulationConfig
from grspin.initial import generate_initial_state
from grspin.neighbors import build_neighbor_list, build_tree, minimum_image
from grspin.nn import mutual_nn_partners, mutual_partners
from grspin.observables import (
    DIMER_ANG_COS_EDGES,
    DIMER_ANG_X_EDGES,
    DIMER_MAP_SHAPE,
    dimer_field_prediction,
    dimer_frame_map,
    energy_budget,
    mutual_pairs,
    pair_energies,
    pair_statistics,
    reciprocal_lattice_vectors,
    spin_structure_factor,
    spin_summary,
    unit_vectors,
)
from grspin.relax import relax
from grspin.theory import two_body_minimum_factor


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


# --------------------------------------------------------------------------------------
# Dimer-frame correlation map
# --------------------------------------------------------------------------------------
def test_dimer_frame_map_of_an_isolated_dimer_with_third_spins():
    """i at the origin, partner j on +z at 0.55; k transverse at 1.03, m on the axis at 1.63.

    Third spins parallel to u_i give C = +1 in i's frame and C = -1 in j's frame (u_j = -u_i); P is
    +1 for m (collinear) and 0 for k (transverse) in i's frame. Units: r_nn = 1.
    """
    positions = np.array([[0, 0, 0], [0, 0, 0.55], [1.03, 0, 0], [0, 0, 1.63]], float) + 5.0
    spins = np.array([[0, 0, 0.7], [0, 0, -0.9], [0, 0, 0.5], [0, 0, 0.6]])
    partner = np.array([1, 0, -1, -1])
    tree = build_tree(positions, None)
    out = dimer_frame_map(positions, spins, tree, None, 1.0, partner)
    cnt, sum_c, sum_p = out["dimer_map_cnt"], out["dimer_map_sum_C"], out["dimer_map_sum_P"]
    assert cnt.shape == DIMER_MAP_SHAPE and cnt.sum() == 4  # i sees k and m, so does j
    # frame of i (axis +z, cells of 0.1): k at z = 0, rho = 1.03 -> (25, 10); m at z = 1.63 -> (41, 0)
    assert cnt[25, 10] == 1 and sum_c[25, 10] == pytest.approx(1.0)
    assert sum_p[25, 10] == pytest.approx(0.0)
    assert cnt[41, 0] == 1 and sum_c[41, 0] == pytest.approx(1.0)
    assert sum_p[41, 0] == pytest.approx(1.0)
    # frame of j (axis -z): k at z = 0.55, rho = 1.03 -> (30, 10); m at z = -1.08 -> (14, 0)
    assert cnt[30, 10] == 1 and sum_c[30, 10] == pytest.approx(-1.0)
    assert cnt[14, 0] == 1 and sum_c[14, 0] == pytest.approx(-1.0)
    assert sum_p[14, 0] == pytest.approx(-1.0)
    # angular table: x bins [0.5, 0.75, 1, 1.25, 1.5, 2, 2.5] times
    # cos bins [-1, -0.8, -0.5, -0.25, 0.25, 0.5, 0.8, 1]
    ang = out["dimer_ang_cnt"]
    assert ang.shape == (len(DIMER_ANG_X_EDGES) - 1, len(DIMER_ANG_COS_EDGES) - 1) == (6, 7)
    assert ang.sum() == 4
    assert ang[2, 3] == 1  # i-k: x = 1.03, transverse
    assert ang[4, 6] == 1 and out["dimer_ang_sum_P"][4, 6] == pytest.approx(1.0)  # i-m: cos = +1
    assert ang[2, 4] == 1  # j-k: x = 1.168, cos = 0.47
    assert ang[2, 0] == 1 and out["dimer_ang_sum_C"][2, 0] == pytest.approx(-1.0)  # j-m: cos = -1
    assert out["dimer_ang_sum_C"].sum() == pytest.approx(sum_c.sum())  # the same four pairs
    assert out["dimer_map_n_spins"] == 2 and out["dimer_map_r_max"] == 2.5
    assert out["dimer_map_sum_x_partner"] == pytest.approx(1.1)
    # antiparallel third spins flip the signs of C and of P (both are odd in u_k)
    flipped = spins * np.array([[1.0], [1.0], [-1.0], [-1.0]])
    again = dimer_frame_map(positions, flipped, tree, None, 1.0, partner)
    assert again["dimer_map_sum_C"][25, 10] == pytest.approx(-1.0)
    assert again["dimer_map_sum_C"][41, 0] == pytest.approx(-1.0)
    assert again["dimer_map_sum_P"][41, 0] == pytest.approx(-1.0)
    assert again["dimer_map_sum_P"][25, 10] == pytest.approx(0.0)  # transverse: no projection
    # a zero-length third spin is skipped
    zero = spins.copy()
    zero[2] = 0.0
    assert dimer_frame_map(positions, zero, tree, None, 1.0, partner)["dimer_map_cnt"].sum() == 2


def test_dimer_frame_map_uses_the_minimum_image():
    box = 10.0  # r_nn = 1: the third spin sits 0.85 across the periodic boundary from i
    positions = np.array([[0.2, 5.0, 5.0], [0.2, 5.0, 5.55], [9.35, 5.0, 5.0]])
    spins = np.array([[0, 0, 1.0], [0, 0, -1.0], [0, 0, 1.0]])
    partner = np.array([1, 0, -1])
    out = dimer_frame_map(positions, spins, build_tree(positions, box), box, 1.0, partner)
    cnt = out["dimer_map_cnt"]
    assert cnt.sum() == 2 and cnt[25, 8] == 1 and cnt[30, 8] == 1
    assert out["dimer_map_sum_C"][25, 8] == pytest.approx(1.0)
    assert out["dimer_map_sum_C"][30, 8] == pytest.approx(-1.0)
    assert out["dimer_ang_cnt"][1, 3] == 1  # x = 0.85 in [0.75, 1), transverse to i
    # with open boundaries the same positions are 9.15 apart: nothing within 2.5
    open_ = dimer_frame_map(positions, spins, build_tree(positions, None), None, 1.0, partner)
    assert open_["dimer_map_cnt"].sum() == 0 and open_["dimer_map_n_spins"] == 2


def _brute_force_dimer_map(positions, spins, box, r_nn, partner, r_max_nn=2.5, shape=(50, 25)):
    unit, _ = unit_vectors(spins)
    n_x, n_c = len(DIMER_ANG_X_EDGES) - 1, len(DIMER_ANG_COS_EDGES) - 1
    out = {
        "dimer_map_cnt": np.zeros(shape, dtype=np.int64),
        "dimer_map_sum_C": np.zeros(shape),
        "dimer_map_sum_P": np.zeros(shape),
        "dimer_ang_cnt": np.zeros((n_x, n_c), dtype=np.int64),
        "dimer_ang_sum_C": np.zeros((n_x, n_c)),
        "dimer_ang_sum_P": np.zeros((n_x, n_c)),
    }
    r_max = r_max_nn * r_nn
    for i in np.flatnonzero(partner >= 0):
        j = partner[i]
        axis = minimum_image(positions[j] - positions[i], box)
        axis = axis / np.linalg.norm(axis)
        for k in range(len(positions)):
            if k in (i, j):
                continue
            d = minimum_image(positions[k] - positions[i], box)
            r = np.linalg.norm(d)
            if r > r_max:
                continue
            z = axis @ d
            rho = np.sqrt(max(r * r - z * z, 0.0))
            c = unit[i] @ unit[k]
            p = (unit[i] @ d / r) * (unit[k] @ d / r)
            iz = min(int((z + r_max) / (2 * r_max) * shape[0]), shape[0] - 1)
            ir = min(int(rho / r_max * shape[1]), shape[1] - 1)
            out["dimer_map_cnt"][iz, ir] += 1
            out["dimer_map_sum_C"][iz, ir] += c
            out["dimer_map_sum_P"][iz, ir] += p
            x, cos = r / r_nn, z / r
            if x < DIMER_ANG_X_EDGES[0] or x > DIMER_ANG_X_EDGES[-1]:
                continue
            ix = min(np.searchsorted(DIMER_ANG_X_EDGES, x, side="right") - 1, n_x - 1)
            ic = min(np.searchsorted(DIMER_ANG_COS_EDGES, cos, side="right") - 1, n_c - 1)
            out["dimer_ang_cnt"][ix, ic] += 1
            out["dimer_ang_sum_C"][ix, ic] += c
            out["dimer_ang_sum_P"][ix, ic] += p
    return out


@pytest.mark.parametrize("boundary", ["open", "periodic"])
def test_dimer_frame_map_matches_a_brute_force_reference(boundary):
    cfg = SimulationConfig(n_particles=220, boundary=boundary, n_bins=8, x_max=3.0, n_sweeps=5)
    positions, spins0, smag = generate_initial_state(cfg, seed=4)
    box = cfg.box if boundary == "periodic" else None
    nl = build_neighbor_list(positions, cfg.r_cut, box)
    spins = relax(spins0, smag, nl, cfg, seed=4).spins
    partner = mutual_nn_partners(positions, nl.tree)
    mutual = partner >= 0
    assert mutual.sum() > 50 and (partner[partner[mutual]] == np.flatnonzero(mutual)).all()
    assert np.array_equal(partner, mutual_partners(nl.tree.query(positions, k=2)[1][:, 1]))
    out = dimer_frame_map(positions, spins, nl.tree, box, cfg.r_nn, partner)
    ref = _brute_force_dimer_map(positions, spins, box, cfg.r_nn, partner)
    assert out["dimer_map_cnt"].sum() > 500
    for key, value in ref.items():
        if key.endswith("cnt"):
            assert np.array_equal(out[key], value), key
        else:
            assert np.allclose(out[key], value, atol=1e-12), key
    assert out["dimer_map_n_spins"] == mutual.sum()


# --------------------------------------------------------------------------------------
# Spin structure factor
# --------------------------------------------------------------------------------------
def test_reciprocal_lattice_vectors_cover_half_of_the_ball():
    n, k = reciprocal_lattice_vectors(box=10.0, n_max=4)
    assert n.shape == k.shape and len(n) == 128  # 257 lattice points in the ball, minus the origin
    assert (np.einsum("ij,ij->i", n, n) <= 16).all() and not (n == 0).all(axis=1).any()
    as_set = {tuple(v) for v in n}
    assert not any(tuple(-v) in as_set for v in n)  # one of each ±n pair
    assert np.allclose(k, 2 * np.pi / 10.0 * n)
    # every lattice vector of the ball appears either as itself or as its negative
    full = [(a, b, c) for a in range(-4, 5) for b in range(-4, 5) for c in range(-4, 5)]
    full = [v for v in full if 0 < sum(x * x for x in v) <= 16]
    assert all(v in as_set or tuple(-x for x in v) in as_set for v in full)
    assert len(full) == 2 * len(n)


def test_spin_structure_factor_of_random_spins_a_spin_wave_and_k_zero():
    rng = np.random.default_rng(3)
    n = 1500
    box = n ** (1 / 3)
    positions = rng.uniform(0.0, box, (n, 3))
    v = rng.normal(size=(n, 3))
    random = 0.5 * v / np.linalg.norm(v, axis=1, keepdims=True)  # lengths do not matter
    nvec, kvec = reciprocal_lattice_vectors(box, 5)
    s = spin_structure_factor(positions, random, kvec)
    assert s.shape == (len(kvec),) and abs(s.mean() - 1.0) < 0.1 and (s >= 0).all()
    k0 = kvec[(nvec == [1, 0, 0]).all(axis=1)][0]
    theta = positions @ k0
    wave = np.stack([np.cos(theta), np.sin(theta), np.zeros(n)], axis=1)
    s_wave = spin_structure_factor(positions, wave, np.stack([k0, -k0, np.zeros(3)]))
    assert s_wave[0] == pytest.approx(n / 2, abs=5.0) and s_wave[1] == pytest.approx(s_wave[0])
    assert s_wave[2] == pytest.approx(spin_summary(wave)["s0_unit"])
    ferro = np.tile([0.0, 0.0, 1.0], (n, 1))
    assert spin_structure_factor(positions, ferro, np.zeros((1, 3)))[0] == pytest.approx(n)
    # several configurations on the same positions share the phases
    both = spin_structure_factor(positions, np.stack([random, wave]), kvec, chunk=100)
    assert both.shape == (2, len(kvec)) and np.allclose(both[0], s)
    assert np.allclose(both[1], spin_structure_factor(positions, wave, kvec))


# --------------------------------------------------------------------------------------
# Two-body minimum for general (sigma, gamma)
# --------------------------------------------------------------------------------------
def _unit(theta: float, phi: float) -> np.ndarray:
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


@pytest.mark.parametrize(
    "sigma, gamma, along_bond",
    [
        (1.0, 3.0, True),  # GR: antiparallel along the bond, f = 2
        (-1.0, 0.0, False),  # isotropic antiferromagnet, f = 1
        (-1.0, 3.0, True),  # dipolar: parallel along the bond, f = 2
        (1.0, 1.0, False),  # transverse only: parallel and transverse, f = 1
        (1.0, 12.0, True),  # f = 11
        (1.0, 100.0, True),  # f = 99
        (1.0, 0.5, False),  # |gamma - 1| < 1: transverse, f = 1
        (-1.0, 1.5, False),
        (1.0, 2.0, True),  # degenerate: both give f = 1
    ],
)
def test_two_body_minimum_factor_is_the_minimum_over_all_directions(sigma, gamma, along_bond):
    """min over u_i, u_j of -sigma [u_i.u_j - gamma (u_i.r)(u_j.r)] equals -max(1, |gamma - 1|)."""
    f = two_body_minimum_factor(sigma, gamma)
    assert f == max(1.0, abs(gamma - 1.0))
    rhat = np.array([0.0, 0.0, 1.0])

    def reduced_energy(ui: np.ndarray, uj: np.ndarray) -> float:
        return float(-sigma * (ui @ uj - gamma * (ui @ rhat) * (uj @ rhat)))

    # the bound is never violated by random directions ...
    rng = np.random.default_rng(int(10 * gamma) + (sigma > 0))
    u = rng.normal(size=(2, 100_000, 3))
    u /= np.linalg.norm(u, axis=2, keepdims=True)
    sampled = -sigma * (np.einsum("ij,ij->i", u[0], u[1]) - gamma * u[0][:, 2] * u[1][:, 2])
    assert sampled.min() >= -f - 1e-12
    # ... and a numerical minimisation over the four angles reaches it
    best = np.inf
    for start in rng.uniform(0.0, np.pi, size=(12, 4)):
        res = minimize(
            lambda p: reduced_energy(_unit(p[0], p[1]), _unit(p[2], p[3])),
            start,
            method="Nelder-Mead",
            options={"xatol": 1e-10, "fatol": 1e-13, "maxiter": 20000, "maxfev": 20000},
        )
        best = min(best, float(res.fun))
    assert best == pytest.approx(-f, abs=1e-6)
    # the analytic minimiser attains the value exactly
    if along_bond:
        ui, uj = rhat, np.sign(sigma * (1.0 - gamma)) * rhat
    else:
        ui, uj = np.array([1.0, 0.0, 0.0]), np.sign(sigma) * np.array([1.0, 0.0, 0.0])
    assert reduced_energy(ui, uj) == pytest.approx(-f)
    # the sign of the coupling does not change the value
    assert two_body_minimum_factor(-sigma, gamma) == f


def test_pair_energies_and_mutual_pairs():
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5], [1.0, 0.0, 0.0]])
    spins = np.array([[0.0, 0.0, 0.6], [0.0, 0.0, -1.1], [0.3, 0.0, 0.0]])
    i, j = np.array([0, 0]), np.array([1, 2])
    energy, minimum, r = pair_energies(positions, spins, i, j, None, 1.0, 3.0)
    assert r.tolist() == [0.5, 1.0]
    # (0, 1): a perfect GR dimer sits at its two-body minimum -2 w s s'
    assert energy[0] == pytest.approx(-2.0 * 0.6 * 1.1 / 0.125)
    assert minimum[0] == pytest.approx(energy[0])
    # (0, 2): S_0 transverse to the bond, S_2 along it: zero energy, minimum -2 w s s'
    assert energy[1] == pytest.approx(0.0) and minimum[1] == pytest.approx(-2.0 * 0.6 * 0.3)
    # other couplings: isotropic (sigma = -1, gamma = 0) and dipolar (-1, 3)
    iso, iso_min, _ = pair_energies(positions, spins, i, j, None, -1.0, 0.0)
    assert iso[0] == pytest.approx(-0.66 / 0.125) and iso_min[0] == pytest.approx(-0.66 / 0.125)
    dip, dip_min, _ = pair_energies(positions, spins, i, j, None, -1.0, 3.0)
    assert dip[0] == pytest.approx(+2.0 * 0.66 / 0.125) and dip_min[0] == pytest.approx(
        -2.0 * 0.66 / 0.125
    )
    # periodic boundaries use the minimum image
    box = 10.0
    far = positions.copy()
    far[1] = [0.0, 0.0, 9.5]  # 0.5 below spin 0 across the boundary
    e_per, m_per, r_per = pair_energies(far, spins, i[:1], j[:1], box, 1.0, 3.0)
    assert r_per[0] == pytest.approx(0.5) and e_per[0] == pytest.approx(energy[0])
    assert m_per[0] == pytest.approx(minimum[0])
    # mutual pairs are listed once, with i < j
    pi, pj = mutual_pairs(np.array([1, 0, -1, 4, 3, -1]))
    assert pi.tolist() == [0, 3] and pj.tolist() == [1, 4]
    assert mutual_pairs(np.array([-1, -1]))[0].size == 0


@pytest.mark.parametrize(
    "boundary, coupling", [("open", "gr"), ("periodic", "dipolar"), ("periodic", "isotropic_afm")]
)
def test_energy_budget_matches_the_kernel_energy_and_a_brute_force_sum(boundary, coupling):
    cfg = SimulationConfig(
        n_particles=150, boundary=boundary, coupling=coupling, n_sweeps=4, r_cut_nn=2.5
    )
    positions, spins0, smag = generate_initial_state(cfg, seed=2)
    box = cfg.box if boundary == "periodic" else None
    nl = build_neighbor_list(positions, cfg.r_cut, box)
    spins = relax(spins0, smag, nl, cfg, seed=2).spins
    partner = mutual_nn_partners(positions, nl.tree)
    sigma, gamma = cfg.sigma_gamma
    b = energy_budget(positions, spins, nl.tree, box, cfg.r_cut, sigma, gamma, partner)
    direct = kernels.energy(nl.indptr, nl.indices, nl.weight, nl.rhat, spins, sigma, gamma)
    assert b["energy_total"] == pytest.approx(float(direct), rel=1e-12)
    i, j, d, r = brute_force_pairs(positions, box, cfg.r_cut)
    rhat = d / r[:, None]
    w = r**-3
    proj = np.einsum("ij,ij->i", spins[i], rhat) * np.einsum("ij,ij->i", spins[j], rhat)
    u_bf = -sigma * w * (np.einsum("ij,ij->i", spins[i], spins[j]) - gamma * proj)
    m_bf = -two_body_minimum_factor(sigma, gamma) * w * smag[i] * smag[j]
    assert b["n_pairs"] == len(i) and b["energy_total"] == pytest.approx(u_bf.sum())
    assert b["minimum_total"] == pytest.approx(m_bf.sum())
    mutual = partner[i] == j
    assert b["n_dimers"] == mutual.sum() == (partner >= 0).sum() // 2 > 10
    assert b["energy_dimer"] == pytest.approx(u_bf[mutual].sum())
    assert b["minimum_dimer"] == pytest.approx(m_bf[mutual].sum())
    # no bond can lie below its own two-body minimum
    assert np.all(u_bf >= m_bf - 1e-12)
    assert b["energy_total"] >= b["minimum_total"] and b["energy_dimer"] >= b["minimum_dimer"]
    assert b["energy_total"] < 0.0  # relaxed


def test_two_isolated_dimers_have_no_non_dimer_energy():
    positions = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.5], [9.0, 9.0, 9.0], [9.0, 9.0, 9.4]])
    spins = np.array([[0.0, 0.0, 0.8], [0.0, 0.0, -0.5], [0.0, 0.0, -0.7], [0.0, 0.0, 0.9]])
    tree = build_tree(positions, None)
    partner = mutual_nn_partners(positions, tree)
    assert partner.tolist() == [1, 0, 3, 2]
    # a cutoff below the inter-dimer distance: only the two dimer bonds exist
    b = energy_budget(positions, spins, tree, None, 5.0, 1.0, 3.0, partner)
    assert b["n_pairs"] == b["n_dimers"] == 2
    assert b["energy_total"] - b["energy_dimer"] == 0.0
    assert b["energy_dimer"] == pytest.approx(-2.0 * (0.8 * 0.5 / 0.125 + 0.7 * 0.9 / 0.064))
    assert b["energy_dimer"] == pytest.approx(b["minimum_dimer"])  # perfect dimers
    # with all six pairs the cross terms (r ~ 13.9, w ~ 4e-4) are negligible
    full = energy_budget(positions, spins, tree, None, 50.0, 1.0, 3.0, partner)
    assert full["n_pairs"] == 6 and full["n_dimers"] == 2
    assert abs(full["energy_total"] - full["energy_dimer"]) < 1e-3 * abs(full["energy_total"])
    assert full["energy_dimer"] == b["energy_dimer"]


# --------------------------------------------------------------------------------------
# Dimer-field test
# --------------------------------------------------------------------------------------
def _tensor_field(d: np.ndarray, s: np.ndarray, gamma: float) -> np.ndarray:
    r = np.linalg.norm(d)
    rh = d / r
    return (s - gamma * (s @ rh) * rh) / r**3


def test_dimer_field_prediction_of_a_synthetic_dimer_decides_the_agreement():
    """Dimer i (origin, S up) and j (on +z at 0.55, S down); third spin k transverse at 1.03 r_nn.

    The prediction is the field of the two members alone; a third spin set along it agrees
    (agree = 1, cos = 1), one set against it does not (agree = 0, cos = -1).
    """
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.55], [1.03, 0.0, 0.0]]) + 5.0
    spins = np.array([[0.0, 0.0, 0.7], [0.0, 0.0, -0.9], [0.2, 0.0, 0.4]])
    partner = np.array([1, 0, -1])
    tree = build_tree(positions, None)
    h = _tensor_field(positions[2] - positions[0], spins[0], 3.0) + _tensor_field(
        positions[2] - positions[1], spins[1], 3.0
    )
    d_ik = (positions[2] - positions[0])[None]
    d_ij = (positions[1] - positions[0])[None]
    pred, defined = dimer_field_prediction(
        spins, np.array([0]), np.array([2]), partner, d_ik, d_ij, None, 1.0, 3.0
    )
    assert defined.all() and np.allclose(pred[0], h / np.linalg.norm(h))
    assert abs(pred[0, 0]) > 0.5 and pred[0, 1] == 0.0  # the field is tilted, not along z
    flipped, _ = dimer_field_prediction(
        spins, np.array([0]), np.array([2]), partner, d_ik, d_ij, None, -1.0, 3.0
    )
    assert np.allclose(flipped, -pred)  # sigma flips the prediction
    # seen from j (partner i) the same two members give the same field
    from_j, _ = dimer_field_prediction(
        spins,
        np.array([1]),
        np.array([2]),
        partner,
        (positions[2] - positions[1])[None],
        (positions[0] - positions[1])[None],
        None,
        1.0,
        3.0,
    )
    assert np.allclose(from_j, pred)
    c_pred_i = float(pred[0] @ [0.0, 0.0, 1.0])  # prediction dotted with u_i
    for sign, expected in ((1.0, 1), (-1.0, 0)):
        aligned = spins.copy()
        aligned[2] = 0.3 * sign * pred[0]
        out = dimer_frame_map(positions, aligned, tree, None, 1.0, partner, sigma_gamma=(1.0, 3.0))
        # i-k at x = 1.03 and j-k at x = 1.168: both in the third x bin
        assert out["dimer_field_cnt"].tolist() == [0, 0, 2, 0, 0, 0]
        assert out["dimer_field_agree"].tolist() == [0, 0, 2 * expected, 0, 0, 0]
        assert out["dimer_field_n"] == 2 and out["dimer_field_n_agree"] == 2 * expected
        assert out["dimer_field_sum_cos"][2] == pytest.approx(2.0 * sign)
        assert out["dimer_ang_sum_Cpred"][2, 3] == pytest.approx(c_pred_i)  # i-k: transverse
        assert out["dimer_ang_sum_Cpred"][2, 4] == pytest.approx(-c_pred_i)  # j-k: u_j = -u_i
        assert out["dimer_ang_sum_Cpred"].sum() == pytest.approx(0.0)
        assert out["dimer_field_sigma"] == 1.0 and out["dimer_field_gamma"] == 3.0
        # the measured C of the aligned third spin equals the prediction (or its negative)
        assert out["dimer_ang_sum_C"][2, 3] == pytest.approx(sign * c_pred_i)
    # without the coupling no field keys are produced
    plain = dimer_frame_map(positions, spins, tree, None, 1.0, partner)
    assert not any(k.startswith("dimer_field") for k in plain)
    assert "dimer_ang_sum_Cpred" not in plain
    # a zero-length partner leaves the field of the other member
    half = spins.copy()
    half[1] = 0.0
    out = dimer_frame_map(positions, half, tree, None, 1.0, partner, sigma_gamma=(1.0, 3.0))
    h_i = _tensor_field(positions[2] - positions[0], spins[0], 3.0)
    assert out["dimer_field_cnt"].sum() == 1  # j has zero length: only i's frame is mapped
    assert out["dimer_ang_sum_Cpred"][2, 3] == pytest.approx(h_i[2] / np.linalg.norm(h_i))


def _brute_force_dimer_field(positions, spins, box, r_nn, partner, sigma, gamma, r_max_nn=2.5):
    unit, smag = unit_vectors(spins)
    n_x, n_c = len(DIMER_ANG_X_EDGES) - 1, len(DIMER_ANG_COS_EDGES) - 1
    out = {
        "dimer_field_cnt": np.zeros(n_x, dtype=np.int64),
        "dimer_field_agree": np.zeros(n_x, dtype=np.int64),
        "dimer_field_sum_cos": np.zeros(n_x),
        "dimer_ang_sum_Cpred": np.zeros((n_x, n_c)),
        "dimer_field_n": 0,
        "dimer_field_n_agree": 0,
    }
    r_max = r_max_nn * r_nn
    for i in np.flatnonzero(partner >= 0):
        j = partner[i]
        if smag[i] == 0.0:
            continue
        d_ij = minimum_image(positions[j] - positions[i], box)
        axis = d_ij / np.linalg.norm(d_ij)
        for k in range(len(positions)):
            if k in (i, j) or smag[k] == 0.0:
                continue
            d = minimum_image(positions[k] - positions[i], box)
            r = np.linalg.norm(d)
            if r > r_max:
                continue
            d_jk = minimum_image(positions[k] - positions[j], box)
            h = _tensor_field(d, spins[i], gamma) + _tensor_field(d_jk, spins[j], gamma)
            if np.linalg.norm(h) == 0.0:
                continue
            pred = sigma * h / np.linalg.norm(h)
            cos = unit[k] @ pred
            out["dimer_field_n"] += 1
            out["dimer_field_n_agree"] += int(cos > 0.0)
            x, cos_theta = r / r_nn, axis @ d / r
            if x < DIMER_ANG_X_EDGES[0] or x > DIMER_ANG_X_EDGES[-1]:
                continue
            ix = min(np.searchsorted(DIMER_ANG_X_EDGES, x, side="right") - 1, n_x - 1)
            ic = min(np.searchsorted(DIMER_ANG_COS_EDGES, cos_theta, side="right") - 1, n_c - 1)
            out["dimer_field_cnt"][ix] += 1
            out["dimer_field_agree"][ix] += int(cos > 0.0)
            out["dimer_field_sum_cos"][ix] += cos
            out["dimer_ang_sum_Cpred"][ix, ic] += pred @ unit[i]
    return out


@pytest.mark.parametrize(
    "boundary, coupling", [("open", "gr"), ("periodic", "dipolar"), ("periodic", "gr")]
)
def test_dimer_field_test_matches_a_brute_force_reference(boundary, coupling):
    cfg = SimulationConfig(
        n_particles=220, boundary=boundary, coupling=coupling, n_bins=8, x_max=3.0, n_sweeps=5
    )
    positions, spins0, smag = generate_initial_state(cfg, seed=4)
    box = cfg.box if boundary == "periodic" else None
    nl = build_neighbor_list(positions, cfg.r_cut, box)
    spins = relax(spins0, smag, nl, cfg, seed=4).spins
    partner = mutual_nn_partners(positions, nl.tree)
    sigma, gamma = cfg.sigma_gamma
    out = dimer_frame_map(
        positions, spins, nl.tree, box, cfg.r_nn, partner, sigma_gamma=(sigma, gamma)
    )
    ref = _brute_force_dimer_field(positions, spins, box, cfg.r_nn, partner, sigma, gamma)
    assert ref["dimer_field_n"] > 500
    for key, value in ref.items():
        if key in ("dimer_field_cnt", "dimer_field_agree", "dimer_field_n", "dimer_field_n_agree"):
            assert np.array_equal(out[key], value), key
        else:
            assert np.allclose(out[key], value, atol=1e-12), key
    assert out["dimer_field_cnt"].sum() == out["dimer_ang_cnt"].sum()  # every prediction defined
    assert out["dimer_field_sigma"] == sigma and out["dimer_field_gamma"] == gamma
    # a relaxed state follows the dimer field far more often than chance
    assert out["dimer_field_n_agree"] / out["dimer_field_n"] > 0.6


def test_dimer_field_agreement_is_one_half_for_random_directions():
    cfg = SimulationConfig(n_particles=3000, n_bins=4, x_max=2.0)
    positions, spins, _ = generate_initial_state(cfg, seed=5)
    tree = build_tree(positions, cfg.box)
    partner = mutual_nn_partners(positions, tree)
    out = dimer_frame_map(
        positions, spins, tree, cfg.box, cfg.r_nn, partner, sigma_gamma=(1.0, 3.0)
    )
    n = int(out["dimer_field_n"])
    assert n > 10000
    # u_k is independent of the field of i and j: agreement 1/2, mean cosine 0 ...
    assert abs(out["dimer_field_n_agree"] / n - 0.5) < 0.02
    assert abs(out["dimer_field_sum_cos"].sum() / out["dimer_field_cnt"].sum()) < 0.02
    assert abs(out["dimer_ang_sum_C"].sum() / out["dimer_ang_cnt"].sum()) < 0.02
    # ... whereas the prediction itself is built from S_i and S_j, so pred . u_i is not centred
    # on zero for random members (the longitudinal part of T_ki S_i has weight 2 and |h| in the
    # normalisation is larger when it dominates): a small positive bias, not a measurement of u_k
    c_pred = out["dimer_ang_sum_Cpred"].sum() / out["dimer_ang_cnt"].sum()
    assert 0.0 < c_pred < 0.15
