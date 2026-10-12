"""Initial conditions: Poisson or hard-core positions and isotropic spins with random magnitudes."""

from __future__ import annotations

import numpy as np

from .config import SimulationConfig


def draw_magnitudes(rng: np.random.Generator, config: SimulationConfig) -> np.ndarray:
    """Draw the fixed spin magnitudes ``s_i`` according to ``config.spin_distribution``.

    ``uniform``       U(0, S)         (the distribution used in the manuscript)
    ``uniform_half``  U(S/2, 3S/2)
    ``equal``         S               (all spins have the same length)
    ``lognormal``     S exp(sigma Z)  (Z standard normal; median S)
    """
    n, s = config.n_particles, config.spin_scale
    kind = config.spin_distribution
    if kind == "uniform":
        return rng.uniform(0.0, s, size=n)
    if kind == "uniform_half":
        return rng.uniform(0.5 * s, 1.5 * s, size=n)
    if kind == "equal":
        return np.full(n, s, dtype=float)
    if kind == "lognormal":
        return s * np.exp(rng.normal(0.0, config.lognormal_sigma, size=n))
    raise ValueError(kind)  # pragma: no cover  (validated in SimulationConfig)


def hard_core_positions(
    rng: np.random.Generator, n: int, box: float, r_min: float, periodic: bool
) -> np.ndarray:
    """``n`` points in ``[0, box)^3`` with all pair distances ``>= r_min`` (sequential addition).

    Random sequential addition (RSA): candidates are drawn uniformly one at a time and rejected
    when they come closer than ``r_min`` to a point accepted earlier (minimum-image distance when
    ``periodic``). The result is deterministic for a given generator state. The excluded volume
    allowed by :data:`grspin.config.MAX_MIN_SEPARATION_NN` is at most 24 % of the box, so fewer
    than 1.2 attempts per point are needed on average and the loop always terminates; the cost
    is ``O(n^2)`` distance evaluations (about 0.4 s for ``n = 5000``). RSA positions are not an
    equilibrium hard-sphere fluid, but at these packing fractions (``<= 3 %``) the two coincide
    to a good approximation; the Poisson results of ``grspin.theory`` do not apply to them.

    ``r_min <= 0`` returns the ideal Poisson process in one vectorised draw: exactly the historic
    random stream of :func:`generate_initial_state`.
    """
    if r_min <= 0.0:
        positions = rng.uniform(0.0, box, size=(n, 3))
        if periodic:
            np.mod(positions, box, out=positions)  # guards against x == L from rounding
        return positions
    if periodic and 2.0 * r_min > box:
        raise ValueError("the hard-core distance must not exceed half of the periodic box")
    positions = np.empty((n, 3))
    r2_min = r_min * r_min
    # Legitimate configurations need ~1.2 attempts per point; the cap only turns an impossible
    # request (a direct call with a huge r_min) into an error instead of an endless loop.
    attempts_left = max(10_000, 100 * n)
    k = 0
    while k < n:
        if attempts_left <= 0:
            raise RuntimeError(
                f"hard-core positions: {k} of {n} points placed, too many rejections for "
                f"r_min = {r_min:g} in a box of {box:g}"
            )
        attempts_left -= 1
        candidate = rng.uniform(0.0, box, size=3)
        if periodic:
            np.mod(candidate, box, out=candidate)
        if k:
            d = np.abs(positions[:k] - candidate)
            if periodic:
                np.minimum(d, box - d, out=d)  # minimum image (all coordinates lie in [0, L))
            if np.min(np.einsum("ij,ij->i", d, d)) < r2_min:
                continue
        positions[k] = candidate
        k += 1
    return positions


def generate_positions(rng: np.random.Generator, config: SimulationConfig) -> np.ndarray:
    """Positions of one realization: Poisson, or hard-core RSA when ``config.min_separation_nn > 0``."""
    return hard_core_positions(
        rng,
        config.n_particles,
        config.box,
        config.min_separation,
        config.boundary == "periodic",
    )


def generate_initial_state(
    config: SimulationConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random positions in the cube ``[0, L)^3`` and isotropic spins of fixed length.

    Returns ``(positions, spins, magnitudes)``. For ``min_separation_nn = 0`` the random-number
    stream is identical to the prototype scripts used for the first exploratory runs (positions,
    one unused draw of particle masses, isotropic directions, magnitudes), so that ``uniform``
    magnitudes reproduce those historic realizations seed by seed. With a hard core the positions
    come from :func:`hard_core_positions` (a seed-dependent number of draws) and the remaining
    draws follow in the same order; realizations with a hard core therefore share nothing but the
    seed with the Poisson realizations of the same seed (comparisons are unpaired).
    """
    rng = np.random.default_rng(seed)
    n = config.n_particles

    positions = generate_positions(rng, config)
    rng.uniform(0.0, 1.0, size=n)  # legacy: particle masses (unused by the GR coupling)

    vectors = rng.normal(size=(n, 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    directions = vectors / norms

    magnitudes = draw_magnitudes(rng, config)
    return positions, magnitudes[:, None] * directions, magnitudes
