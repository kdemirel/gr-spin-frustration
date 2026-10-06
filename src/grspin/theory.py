"""Exact results for the Poisson point process used as references in the analysis.

With ``rho`` the number density, ``r_nn`` the mean nearest-neighbour distance and ``x = r / r_nn``,
the dimensionless combination is ``rho r^3 = (RNN_UNIT_DENSITY * x)^3``.
"""

from __future__ import annotations

import numpy as np

from .config import RNN_UNIT_DENSITY

#: Fraction of particles that are in a mutual nearest-neighbour pair (3D Poisson process): 16/27.
MUTUAL_PARTICLE_FRACTION: float = 16.0 / 27.0


def rho_r3(x: np.ndarray | float) -> np.ndarray | float:
    """``rho * r^3`` for ``r = x * r_nn``."""
    return (RNN_UNIT_DENSITY * np.asarray(x, dtype=float)) ** 3


def isolated_sphere_probability(x: np.ndarray | float) -> np.ndarray | float:
    """``exp(-4 pi rho r^3 / 3)``: probability that a ball of radius ``r`` around a particle is empty."""
    return np.exp(-(4.0 * np.pi / 3.0) * rho_r3(x))


def mutual_nn_probability(x: np.ndarray | float) -> np.ndarray | float:
    """``exp(-9 pi rho r^3 / 4)``: probability that a pair at separation ``r`` is a mutual-NN pair.

    Both members must have no other particle within ``r``; the union of the two balls of radius
    ``r`` whose centres are ``r`` apart has volume ``2 (4 pi/3) r^3 - (5 pi/12) r^3 = (9 pi/4) r^3``.
    """
    return np.exp(-(9.0 * np.pi / 4.0) * rho_r3(x))


def nn_distance_pdf(x: np.ndarray | float) -> np.ndarray | float:
    """Probability density (per unit ``x``) of the nearest-neighbour distance ``r / r_nn``."""
    c = RNN_UNIT_DENSITY**3
    x = np.asarray(x, dtype=float)
    return 4.0 * np.pi * c * x**2 * np.exp(-(4.0 * np.pi / 3.0) * c * x**3)
