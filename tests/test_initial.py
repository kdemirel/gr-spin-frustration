"""Initial state: Poisson and hard-core positions, reproducibility and the historic random stream."""

from __future__ import annotations

import numpy as np
import pytest
from conftest import brute_force_pairs
from scipy.spatial import cKDTree

from grspin.config import SimulationConfig
from grspin.ensemble import run_realization
from grspin.initial import generate_initial_state, generate_positions, hard_core_positions


def _nn_distances(positions: np.ndarray, box: float | None) -> np.ndarray:
    d, _ = cKDTree(positions, boxsize=box).query(positions, k=2)
    return d[:, 1]


@pytest.mark.parametrize("boundary", ["periodic", "open"])
def test_hard_core_positions_keep_all_pair_distances_above_the_minimum(boundary):
    cfg = SimulationConfig(
        n_particles=1500, boundary=boundary, min_separation_nn=0.6, r_cut_nn=3.0, n_bins=4
    )
    positions, spins, mags = generate_initial_state(cfg, seed=11)
    box = cfg.box if boundary == "periodic" else None
    r_min = cfg.min_separation
    assert positions.shape == (1500, 3) and np.all((positions >= 0.0) & (positions < cfg.box))
    # no pair (minimum image when periodic) closer than r_min, by brute force and by the tree
    i, j, _, r = brute_force_pairs(positions, box, r_max=r_min * (1.0 - 1e-12))
    assert len(i) == 0 and len(j) == 0 and r.size == 0
    nn = _nn_distances(positions, box)
    assert nn.min() >= r_min and nn.min() < 1.05 * r_min  # the hard core is reached, not exceeded
    # the hard core pushes the nearest neighbours outwards (Poisson mean: 1 r_nn)
    assert nn.mean() > 1.03 * cfg.r_nn
    # the spins are still drawn after the positions with the same conventions
    assert np.allclose(np.linalg.norm(spins, axis=1), mags)
    assert 0.0 <= mags.min() and mags.max() <= cfg.spin_scale


def test_poisson_positions_have_pairs_below_the_hard_core_distance():
    """Control for the test above: without a hard core about 8 % of the particles have a neighbour
    within 0.5 r_nn (``1 - exp(-(4 pi / 3) rho (0.5 r_nn)^3)``)."""
    cfg = SimulationConfig(n_particles=1500, r_cut_nn=3.0, n_bins=4)
    positions, _, _ = generate_initial_state(cfg, seed=11)
    close = np.mean(_nn_distances(positions, cfg.box) < 0.5 * cfg.r_nn)
    expected = 1.0 - np.exp(-4.0 / 3.0 * np.pi * cfg.density_effective * (0.5 * cfg.r_nn) ** 3)
    assert expected == pytest.approx(0.085, abs=0.002) and abs(close - expected) < 0.03


@pytest.mark.parametrize("boundary", ["periodic", "open"])
def test_hard_core_realizations_are_reproducible_and_seed_dependent(boundary):
    cfg = SimulationConfig(
        n_particles=400, boundary=boundary, min_separation_nn=0.5, r_cut_nn=2.0, n_bins=4
    )
    a = generate_initial_state(cfg, seed=3)
    b = generate_initial_state(cfg, seed=3)
    assert all(np.array_equal(x, y) for x, y in zip(a, b, strict=True))
    c = generate_initial_state(cfg, seed=4)
    assert not np.array_equal(a[0], c[0]) and not np.array_equal(a[1], c[1])
    # the generator is the only state: calling the position routine directly gives the same points
    rng = np.random.default_rng(3)
    assert np.array_equal(generate_positions(rng, cfg), a[0])
    rng = np.random.default_rng(3)
    direct = hard_core_positions(rng, 400, cfg.box, cfg.min_separation, boundary == "periodic")
    assert np.array_equal(direct, a[0])


@pytest.mark.parametrize("boundary", ["periodic", "open"])
@pytest.mark.parametrize("distribution", ["uniform", "lognormal"])
def test_zero_hard_core_reproduces_the_historic_random_stream_bit_for_bit(boundary, distribution):
    """``min_separation_nn = 0`` must leave positions, directions and lengths bit-identical to the
    code before the hard core existed (the fixed-seed tests and the prototype regression rely on it)."""
    n = 700
    cfg = SimulationConfig(
        n_particles=n,
        boundary=boundary,
        spin_distribution=distribution,
        spin_scale=2.5,
        r_cut_nn=2.0,
        n_bins=4,
    )
    assert cfg.min_separation_nn == 0.0
    positions, spins, mags = generate_initial_state(cfg, seed=21)
    rng = np.random.default_rng(21)  # the historic order of draws, operation by operation
    ref_pos = rng.uniform(0.0, cfg.box, size=(n, 3))
    if boundary == "periodic":
        np.mod(ref_pos, cfg.box, out=ref_pos)
    rng.uniform(0.0, 1.0, size=n)  # legacy particle masses
    vectors = rng.normal(size=(n, 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    directions = vectors / norms
    if distribution == "uniform":
        ref_mags = rng.uniform(0.0, 2.5, size=n)
    else:
        ref_mags = 2.5 * np.exp(rng.normal(0.0, 1.0, size=n))
    assert np.array_equal(positions, ref_pos)
    assert np.array_equal(mags, ref_mags)
    assert np.array_equal(spins, ref_mags[:, None] * directions)
    # the direct routine with r_min <= 0 is the same single vectorised draw
    rng = np.random.default_rng(21)
    assert np.array_equal(
        hard_core_positions(rng, n, cfg.box, 0.0, boundary == "periodic"), ref_pos
    )


def test_hard_core_distance_above_half_the_periodic_box_is_refused():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="half of the periodic box"):
        hard_core_positions(rng, 5, box=1.0, r_min=0.6, periodic=True)
    open_positions = hard_core_positions(rng, 5, box=1.0, r_min=0.6, periodic=False)
    assert open_positions.shape == (5, 3)  # open boundaries only need the points to fit


def test_hard_core_realization_has_empty_bins_below_the_minimum_separation():
    cfg = SimulationConfig(
        n_particles=500,
        min_separation_nn=0.5,
        r_cut_nn=3.0,
        n_sweeps=4,
        n_bins=20,
        x_max=4.0,
        nn_diagnostics=True,
    )
    record, state = run_realization(cfg, seed=8, return_state=True)
    edges = cfg.x_edges
    below = edges[1:] <= cfg.min_separation_nn * (1.0 + 1e-12)  # bins entirely inside the core
    assert below.sum() == 2 and np.all(record["init_cnt_C"][below] == 0)
    assert np.all(record["fin_cnt_C"][below] == 0) and record["fin_cnt_C"][~below].sum() > 0
    assert record["energy"][0] > record["energy"][-1] and record["energy_drift"] < 1e-12
    assert np.all(record["nn_frac_mutual"] >= 0.0)  # the nn diagnostics run; the Poisson theory
    # does not apply to these positions (docs/METHODS.md, section 6)
    assert _nn_distances(state["positions"], cfg.box).min() >= cfg.min_separation
    assert state["positions"].shape == (500, 3)


def test_impossible_hard_core_request_raises_instead_of_looping():
    """A direct call with an unreachable ``r_min`` (50 points at pairwise distance >= 0.6 in a unit
    cube) ends with an error after a bounded number of attempts; legitimate configurations
    (``min_separation_nn <= 0.7``) need about 1.2 attempts per point and never reach the cap."""
    rng = np.random.default_rng(1)
    with pytest.raises(RuntimeError, match="too many rejections"):
        hard_core_positions(rng, 50, box=1.0, r_min=0.6, periodic=False)
    cfg = SimulationConfig(n_particles=2000, min_separation_nn=0.7, r_cut_nn=3.0, n_bins=4)
    positions = generate_positions(np.random.default_rng(1), cfg)  # the bound is comfortably met
    assert _nn_distances(positions, cfg.box).min() >= cfg.min_separation
