"""Exact results used as references in the analysis.

Poisson point process: with ``rho`` the number density, ``r_nn`` the mean nearest-neighbour
distance and ``x = r / r_nn``, the dimensionless combination is ``rho r^3 = (RNN_UNIT_DENSITY * x)^3``.
The Poisson results (:func:`mutual_nn_probability`, :func:`nn_distance_pdf`, ...) hold for the
*ideal* process; a minimum-separation (hard-core) ensemble does not obey them.

Pair interaction: :func:`two_body_minimum_factor` gives the ground-state energy of an isolated pair
for a general coupling ``(sigma, gamma)``.
"""

from __future__ import annotations

import numpy as np

from .config import RNN_UNIT_DENSITY, pair_minimum_factor


def two_body_minimum_factor(sigma: float, gamma: float) -> float:
    """``f(sigma, gamma)`` in the two-body minimum ``U_min = -f G w s_i s_j`` of an isolated pair.

    The pair energy of the coupling family is ``U = -sigma G w s_i s_j u_i^T M u_j`` with
    ``w = 1 / r^3`` and ``M = 1 - gamma rhat rhat^T``, a symmetric matrix with the eigenvalues
    ``1`` (twice, transverse to the bond) and ``1 - gamma`` (along the bond). For unit vectors
    the bilinear form ``u_i^T M u_j`` ranges over ``[-m, m]`` with ``m = max |eigenvalue of M|``
    (its largest singular value; the sign is free because ``u_j -> -u_j`` flips it), so for
    either sign of ``sigma``

        min_{u_i, u_j} U = -G w s_i s_j max(1, |gamma - 1|).

    The minimising configuration is along the bond with ``u_i . u_j = sign(sigma (1 - gamma))``
    when ``|gamma - 1| > 1`` and transverse to it with ``u_i . u_j = sign(sigma)`` when
    ``|gamma - 1| < 1`` (degenerate at ``|gamma - 1| = 1``, i.e. ``gamma = 0`` or ``2``). Hence
    ``f = 2`` for the GR tensor (``sigma = 1, gamma = 3``: antiparallel along the bond) and for the
    dipolar control (``sigma = -1, gamma = 3``: parallel head-to-tail), ``f = 1`` for the isotropic
    antiferromagnet (``gamma = 0``) and for the transverse-only coupling (``gamma = 1``: parallel
    and transverse), ``f = gamma - 1`` for ``gamma >= 2``. ``sigma`` only selects the
    configuration, not the value; it is accepted so that callers can pass ``config.sigma_gamma``.
    The same number is :func:`grspin.config.pair_minimum_factor` (by coupling name).
    """
    return pair_minimum_factor((float(sigma), float(gamma)))


#: Fraction of particles that are in a mutual nearest-neighbour pair (3D **ideal Poisson** process,
#: ``min_separation_nn = 0``; a hard core changes it): 16/27.
#: The nearest neighbour of a particle lies at distance ``r`` with probability density
#: ``4 pi rho r^2 exp(-4 pi rho r^3 / 3)``; the pair is mutual when the part of the ball of radius
#: ``r`` around the neighbour that lies outside the ball around the particle, of volume
#: ``(11/12) pi r^3``, is empty as well (the union of the two balls is ``(4/3 + 11/12) pi r^3``).
#: Integrating ``exp(-(11/12) pi rho r^3)`` against the density gives
#: ``(4/3) / (4/3 + 11/12) = 16/27 = 0.5926``.
MUTUAL_PARTICLE_FRACTION: float = 16.0 / 27.0

#: Mean separation of the two members of a mutual nearest-neighbour pair in units of ``r_nn``.
#: The pair distance has the density ``r^2 exp(-9 pi rho r^3 / 4)`` (up to normalisation), whose
#: mean is ``Gamma(4/3) (4 / (9 pi rho))^(1/3) = (16/27)^(1/3) r_nn = 0.8400 r_nn``: dimers are
#: closer than the mean nearest-neighbour distance because mutuality favours short pairs.
MUTUAL_PAIR_MEAN_DISTANCE_X: float = (16.0 / 27.0) ** (1.0 / 3.0)


def rho_r3(x: np.ndarray | float) -> np.ndarray | float:
    """``rho * r^3`` for ``r = x * r_nn``."""
    return (RNN_UNIT_DENSITY * np.asarray(x, dtype=float)) ** 3


def isolated_sphere_probability(x: np.ndarray | float) -> np.ndarray | float:
    """``exp(-4 pi rho r^3 / 3)``: probability that a ball of radius ``r`` around a particle is empty."""
    return np.exp(-(4.0 * np.pi / 3.0) * rho_r3(x))


def mutual_nn_probability(x: np.ndarray | float) -> np.ndarray | float:
    """``exp(-9 pi rho r^3 / 4)``: probability that a pair at separation ``r`` is a mutual-NN pair.

    Both members must have no other particle within ``r``: the union of the two balls of radius
    ``r`` whose centres are ``r`` apart must be empty. Its volume is ``(4/3 + 11/12) pi r^3 =
    (9/4) pi r^3`` (two balls of ``(4/3) pi r^3`` minus their lens of ``(5/12) pi r^3``), so the
    probability is ``exp(-rho V_union)``. This is the dimer fraction ``f_m(x)``; ``C(r) -> -f_m``
    as ``r -> 0``. The isolated-sphere probability ``exp(-4 pi rho r^3 / 3)`` of
    :func:`isolated_sphere_probability` is *not* the dimer fraction (0.94 versus 0.89 at
    ``x = 0.45``). Valid for the ideal Poisson process only (``min_separation_nn = 0``): a hard
    core removes the short pairs, so the curve is not a reference for those runs.
    """
    return np.exp(-(9.0 * np.pi / 4.0) * rho_r3(x))


def nn_distance_pdf(x: np.ndarray | float) -> np.ndarray | float:
    """Probability density (per unit ``x``) of the nearest-neighbour distance ``r / r_nn``.

    Ideal Poisson process only (``min_separation_nn = 0``); hard-core positions have no
    neighbours below ``r_min`` and a different law.
    """
    c = RNN_UNIT_DENSITY**3
    x = np.asarray(x, dtype=float)
    return 4.0 * np.pi * c * x**2 * np.exp(-(4.0 * np.pi / 3.0) * c * x**3)
