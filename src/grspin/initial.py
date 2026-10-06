"""Initial conditions: Poisson positions and isotropic spins with random magnitudes."""

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


def generate_initial_state(
    config: SimulationConfig, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random positions in the cube ``[0, L)^3`` and isotropic spins of fixed length.

    Returns ``(positions, spins, magnitudes)``. The random-number stream is identical to the
    prototype scripts used for the first exploratory runs (positions, one unused draw of
    particle masses, isotropic directions, magnitudes), so that ``uniform`` magnitudes
    reproduce those historic realizations seed by seed.
    """
    rng = np.random.default_rng(seed)
    n, box = config.n_particles, config.box

    positions = rng.uniform(0.0, box, size=(n, 3))
    if config.boundary == "periodic":
        np.mod(positions, box, out=positions)  # guards against x == L from rounding
    rng.uniform(0.0, 1.0, size=n)  # legacy: particle masses (unused by the GR coupling)

    vectors = rng.normal(size=(n, 3))
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    directions = vectors / norms

    magnitudes = draw_magnitudes(rng, config)
    return positions, magnitudes[:, None] * directions, magnitudes
