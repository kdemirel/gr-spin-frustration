"""Kernels, neighbour lists, relaxation and configuration handling."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import brute_force_pairs

from grspin import kernels
from grspin.config import COUPLINGS, AnnealConfig, SimulationConfig
from grspin.initial import generate_initial_state
from grspin.neighbors import build_neighbor_list
from grspin.relax import relax, total_energy


def _two_body(spins0, s1=0.7, s2=1.3, distance=2.0):
    positions = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, distance]])
    nlist = build_neighbor_list(positions, r_cut=10.0, box=None)
    return positions, nlist, np.array(spins0, dtype=float), np.array([s1, s2])


def _relax_two_body(sigma, gamma, sweeps=60):
    rng = np.random.default_rng(3)
    s = np.array([0.7, 1.3])
    v = rng.normal(size=(2, 3))
    spins = s[:, None] * v / np.linalg.norm(v, axis=1, keepdims=True)
    _, nl, spins, smag = _two_body(spins)
    order = np.arange(2, dtype=np.int64)
    args = (nl.indptr, nl.indices, nl.weight, nl.rhat)
    for _ in range(sweeps):
        kernels.sweep(*args, spins, smag, order, sigma, gamma)
    return spins, kernels.energy(*args, spins, sigma, gamma), 2.0


def test_two_body_gr_ground_state_is_antiparallel_along_the_axis():
    spins, e, r = _relax_two_body(*COUPLINGS["gr"])
    zhat = np.array([0.0, 0.0, 1.0])
    assert abs(abs(spins[0] @ zhat) - 0.7) < 1e-10 and abs(abs(spins[1] @ zhat) - 1.3) < 1e-10
    assert spins[0] @ spins[1] == pytest.approx(-0.7 * 1.3, rel=1e-10)
    assert e == pytest.approx(-2.0 * 0.7 * 1.3 / r**3, rel=1e-10)


def test_two_body_dipolar_control_is_parallel_along_the_axis():
    spins, e, r = _relax_two_body(*COUPLINGS["dipolar"])
    assert spins[0] @ spins[1] == pytest.approx(+0.7 * 1.3, rel=1e-10)
    assert e == pytest.approx(-2.0 * 0.7 * 1.3 / r**3, rel=1e-10)


def test_two_body_isotropic_afm_control_is_antiparallel():
    spins, e, r = _relax_two_body(*COUPLINGS["isotropic_afm"])
    assert spins[0] @ spins[1] == pytest.approx(-0.7 * 1.3, rel=1e-10)
    assert e == pytest.approx(-0.7 * 1.3 / r**3, rel=1e-10)


@pytest.mark.parametrize("order", ["sequential", "random"])
@pytest.mark.parametrize("boundary", ["periodic", "open"])
def test_energy_is_monotone_and_incremental_energy_is_exact(order, boundary):
    cfg = SimulationConfig(
        n_particles=200, boundary=boundary, n_sweeps=12, update_order=order, n_bins=4, x_max=1.0
    )
    positions, spins0, smag = generate_initial_state(cfg, seed=11)
    box = cfg.box if boundary == "periodic" else None
    nl = build_neighbor_list(positions, cfg.r_cut, box)
    res = relax(spins0, smag, nl, cfg, seed=11)
    assert res.energy_drift < 1e-12
    assert res.energy[-1] == pytest.approx(total_energy(res.spins, nl, cfg), rel=1e-12)
    assert np.all(np.diff(res.energy) <= 1e-9 * np.abs(res.energy[:-1]))
    assert np.allclose(np.linalg.norm(res.spins, axis=1), smag)  # magnitudes are conserved
    assert res.energy[0] > res.energy[-1]


def test_relaxed_state_is_a_fixed_point_of_the_alignment_rule():
    cfg = SimulationConfig(n_particles=40, boundary="open", n_sweeps=600, n_bins=4, x_max=1.0)
    positions, spins0, smag = generate_initial_state(cfg, seed=2)
    nl = build_neighbor_list(positions, cfg.r_cut, None)
    res = relax(spins0, smag, nl, cfg, seed=2)
    h = kernels.local_fields(nl.indptr, nl.indices, nl.weight, nl.rhat, res.spins, 3.0)
    cosine = np.einsum("ij,ij->i", h, res.spins) / (
        np.linalg.norm(h, axis=1) * np.linalg.norm(res.spins, axis=1)
    )
    assert res.dmax[-1] < 1e-6
    assert np.all(cosine > 1 - 1e-9)


def test_relaxation_is_invariant_under_rescaling_of_length_spin_and_coupling():
    base = dict(n_particles=120, boundary="open", n_sweeps=10, n_bins=4, x_max=1.0)
    finals = []
    for box_length, spin_scale, g in [(20.0, 1.0, 1.0), (3.7, 250.0, 1e3)]:
        cfg = SimulationConfig(box_length=box_length, spin_scale=spin_scale, G=g, **base)
        positions, spins0, smag = generate_initial_state(cfg, seed=4)
        nl = build_neighbor_list(positions, cfg.r_cut, None)
        res = relax(spins0, smag, nl, cfg, seed=4)
        finals.append(res.spins / smag[:, None])
    assert np.max(np.abs(finals[0] - finals[1])) < 1e-8


def test_periodic_neighbor_list_matches_brute_force_minimum_image():
    rng = np.random.default_rng(0)
    box, r_cut = 4.0, 1.9
    positions = rng.uniform(0, box, size=(70, 3))
    nl = build_neighbor_list(positions, r_cut, box)
    i, j, d, r = brute_force_pairs(positions, box, r_cut)
    expected = {(a, b) for a, b in zip(i, j, strict=True)}
    found = set()
    for a in range(nl.n_particles):
        for e in range(nl.indptr[a], nl.indptr[a + 1]):
            b = int(nl.indices[e])
            if a < b:
                found.add((a, b))
                dvec = positions[b] - positions[a]
                dvec -= box * np.round(dvec / box)
                rr = np.linalg.norm(dvec)
                assert nl.weight[e] == pytest.approx(1.0 / rr**3, rel=1e-12)
                assert abs(nl.rhat[e] @ dvec / rr) == pytest.approx(1.0, abs=1e-12)
    assert found == expected
    assert nl.n_edges == 2 * len(expected)


def test_open_neighbor_list_does_not_wrap():
    positions = np.array([[0.05, 0.5, 0.5], [3.95, 0.5, 0.5], [0.4, 0.5, 0.5]])
    nl = build_neighbor_list(positions, r_cut=1.0, box=None)
    assert set(nl.neighbors_of(0)) == {2}
    assert len(nl.neighbors_of(1)) == 0
    nl_p = build_neighbor_list(positions, r_cut=1.0, box=4.0)
    assert set(nl_p.neighbors_of(0)) == {1, 2}


def test_periodic_cutoff_larger_than_half_box_is_rejected():
    with pytest.raises(ValueError, match="r_cut <= L/2"):
        SimulationConfig(n_particles=300, r_cut_nn=10.0)  # L/2 is only ~6 r_nn here
    cfg = SimulationConfig(n_particles=300)  # automatic cutoff respects L/2
    assert cfg.r_cut_nn_resolved == pytest.approx(0.5 * cfg.box / cfg.r_nn)
    assert SimulationConfig(n_particles=5000).r_cut_nn_resolved == 10.0


def test_config_validation_roundtrip_and_compat_key():
    cfg = SimulationConfig(
        n_particles=500, n_sweeps=20, snapshots=(5, 1, 20), anneal=AnnealConfig(t_start=3.0)
    )
    assert cfg.snapshots == (1, 5, 20)
    again = SimulationConfig.from_dict(cfg.to_dict())
    assert again == cfg
    assert cfg.replace(n_realizations=7, save_states=2).compat_key() == cfg.compat_key()
    assert cfg.replace(n_sweeps=21).compat_key() != cfg.compat_key()
    for bad in (
        dict(n_particles=1),
        dict(boundary="torus"),
        dict(snapshots=(0,)),
        dict(x_max=-1.0),
    ):
        with pytest.raises(ValueError):
            cfg.replace(**bad)


def test_initial_state_reproduces_the_prototype_random_stream():
    cfg = SimulationConfig(n_particles=5, box_length=1000.0, spin_scale=1000.0, boundary="open")
    positions, spins, mags = generate_initial_state(cfg, seed=1)
    rng = np.random.default_rng(1)  # order of draws used by the prototype scripts
    ref_pos = rng.uniform(0.0, 1000.0, size=(5, 3))
    rng.uniform(0.0, 1.0, size=5)
    v = rng.normal(size=(5, 3))
    ref_mag = rng.uniform(0.0, 1000.0, size=(5, 1))
    assert np.array_equal(positions, ref_pos)
    assert np.allclose(spins, ref_mag * v / np.linalg.norm(v, axis=1, keepdims=True), rtol=1e-14)
    assert np.allclose(np.linalg.norm(spins, axis=1), mags)


@pytest.mark.parametrize("kind", ["uniform", "uniform_half", "equal", "lognormal"])
def test_spin_magnitude_distributions(kind):
    cfg = SimulationConfig(
        n_particles=4000, spin_distribution=kind, spin_scale=2.0, boundary="open"
    )
    _, spins, mags = generate_initial_state(cfg, seed=1)
    assert np.allclose(np.linalg.norm(spins, axis=1), mags)
    if kind == "uniform":
        assert 0 <= mags.min() and mags.max() <= 2.0 and abs(mags.mean() - 1.0) < 0.05
    elif kind == "uniform_half":
        assert 1.0 <= mags.min() and mags.max() <= 3.0
    elif kind == "equal":
        assert np.all(mags == 2.0)
    else:
        assert abs(np.median(mags) - 2.0) < 0.15


def test_annealing_protocol_reaches_low_energies_and_keeps_magnitudes():
    base = dict(n_particles=150, n_sweeps=20, n_bins=4, x_max=1.0)
    quench = SimulationConfig(**base)
    anneal = SimulationConfig(
        protocol="anneal", anneal=AnnealConfig(n_temperatures=12, sweeps_per_temperature=4), **base
    )
    positions, spins0, smag = generate_initial_state(quench, seed=3)
    nl = build_neighbor_list(positions, quench.r_cut, quench.box)
    res_q = relax(spins0, smag, nl, quench, seed=3)
    res_a = relax(spins0, smag, nl, anneal, seed=3)
    assert np.allclose(np.linalg.norm(res_a.spins, axis=1), smag)
    acc = res_a.anneal_acceptance_per_T
    assert acc.shape == (12,) and np.all((acc > 0.0) & (acc < 1.0))
    assert acc[0] > acc[-1] == res_a.anneal_acceptance_final  # hot: nearly all accepted
    # the annealed state is already deep before the final greedy stage, and the two protocols
    # end at comparable (not identical) low energies
    assert res_a.energy[0] < 0.8 * res_q.energy[-1] < 0.0
    assert abs(res_a.energy[-1] - res_q.energy[-1]) < 0.2 * abs(res_q.energy[-1])
    assert res_a.energy_drift < 1e-12 and np.all(
        np.diff(res_a.energy) <= 1e-9 * np.abs(res_a.energy[:-1])
    )
