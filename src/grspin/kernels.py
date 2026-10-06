"""Numba kernels: sequential alignment sweeps, energy, and Metropolis moves.

Conventions (see ``docs/METHODS.md``). For a coupling ``(sigma, gamma)`` the Hamiltonian is

    H = -sigma * G * sum_{i<j} w_ij [S_i . S_j - gamma (S_i . rhat_ij)(S_j . rhat_ij)],   w_ij = 1/r_ij^3

and the raw local field of spin ``i`` is ``h_i = sum_j w_ij [S_j - gamma (S_j . rhat_ij) rhat_ij]``
(without ``G``), so that ``H = -(sigma G / 2) sum_i S_i . h_i``. ``G`` never influences the
dynamics; it only rescales the energies and is applied outside the kernels. The general-relativistic
spin-spin interaction is ``sigma = +1``, ``gamma = 3``.
"""

from __future__ import annotations

import numpy as np
from numba import njit


@njit(cache=True)
def sweep(indptr, indices, weight, rhat, spins, smag, order, sigma, gamma):
    """One sequential (Gauss-Seidel) alignment sweep, in place.

    Each spin is set to ``S_i = s_i * sigma * h_i / |h_i|`` using the *current* neighbour
    spins, i.e. the exact minimizer of ``H`` over the direction of ``S_i``. Consequently the
    energy never increases, and the energy change of the sweep is accumulated exactly:
    ``dE_i = -sigma * (S_i_new - S_i_old) . h_i``.

    Returns ``(dmax, dE)``: the largest change of a unit direction and the total energy change
    of the sweep (both without ``G``).
    """
    dmax = 0.0
    d_energy = 0.0
    for t in range(order.shape[0]):
        i = order[t]
        hx = 0.0
        hy = 0.0
        hz = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            rx = rhat[e, 0]
            ry = rhat[e, 1]
            rz = rhat[e, 2]
            sx = spins[j, 0]
            sy = spins[j, 1]
            sz = spins[j, 2]
            dot = sx * rx + sy * ry + sz * rz
            w = weight[e]
            hx += w * (sx - gamma * dot * rx)
            hy += w * (sy - gamma * dot * ry)
            hz += w * (sz - gamma * dot * rz)
        hn = np.sqrt(hx * hx + hy * hy + hz * hz)
        s = smag[i]
        if hn > 0.0 and s > 0.0:
            f = sigma * s / hn
            nx = hx * f
            ny = hy * f
            nz = hz * f
            ox = spins[i, 0]
            oy = spins[i, 1]
            oz = spins[i, 2]
            d_energy -= sigma * ((nx - ox) * hx + (ny - oy) * hy + (nz - oz) * hz)
            ddx = (nx - ox) / s
            ddy = (ny - oy) / s
            ddz = (nz - oz) / s
            dl = np.sqrt(ddx * ddx + ddy * ddy + ddz * ddz)
            if dl > dmax:
                dmax = dl
            spins[i, 0] = nx
            spins[i, 1] = ny
            spins[i, 2] = nz
    return dmax, d_energy


@njit(cache=True)
def energy(indptr, indices, weight, rhat, spins, sigma, gamma):
    """Total energy ``H`` without the factor ``G`` (one pass over the directed neighbour list)."""
    total = 0.0
    for i in range(indptr.shape[0] - 1):
        hx = 0.0
        hy = 0.0
        hz = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            rx = rhat[e, 0]
            ry = rhat[e, 1]
            rz = rhat[e, 2]
            sx = spins[j, 0]
            sy = spins[j, 1]
            sz = spins[j, 2]
            dot = sx * rx + sy * ry + sz * rz
            w = weight[e]
            hx += w * (sx - gamma * dot * rx)
            hy += w * (sy - gamma * dot * ry)
            hz += w * (sz - gamma * dot * rz)
        total += spins[i, 0] * hx + spins[i, 1] * hy + spins[i, 2] * hz
    return -0.5 * sigma * total


@njit(cache=True)
def local_fields(indptr, indices, weight, rhat, spins, gamma):
    """Raw local fields ``h_i`` (without ``sigma`` and ``G``) for all spins."""
    n = indptr.shape[0] - 1
    fields = np.zeros((n, 3))
    for i in range(n):
        hx = 0.0
        hy = 0.0
        hz = 0.0
        for e in range(indptr[i], indptr[i + 1]):
            j = indices[e]
            rx = rhat[e, 0]
            ry = rhat[e, 1]
            rz = rhat[e, 2]
            sx = spins[j, 0]
            sy = spins[j, 1]
            sz = spins[j, 2]
            dot = sx * rx + sy * ry + sz * rz
            w = weight[e]
            hx += w * (sx - gamma * dot * rx)
            hy += w * (sy - gamma * dot * ry)
            hz += w * (sz - gamma * dot * rz)
        fields[i, 0] = hx
        fields[i, 1] = hy
        fields[i, 2] = hz
    return fields


@njit(cache=True)
def metropolis(
    indptr, indices, weight, rhat, spins, smag, temps, sweeps_per_temp, step, sigma, gamma, seed
):
    """Single-spin Metropolis annealing at fixed spin magnitudes, in place.

    ``temps`` are absolute temperatures in the (G-free) energy units of :func:`energy`.
    One sweep is ``N`` proposals at random sites; a proposal perturbs the unit direction by
    a Gaussian of width ``step`` and renormalizes. Returns the acceptance rate at the last
    temperature.
    """
    np.random.seed(seed)
    n = indptr.shape[0] - 1
    acc_last = 0.0
    for it in range(temps.shape[0]):
        temperature = temps[it]
        accepted = 0
        proposed = 0
        for _ in range(sweeps_per_temp):
            for _ in range(n):
                i = np.random.randint(0, n)
                s = smag[i]
                if s == 0.0:
                    continue
                hx = 0.0
                hy = 0.0
                hz = 0.0
                for e in range(indptr[i], indptr[i + 1]):
                    j = indices[e]
                    rx = rhat[e, 0]
                    ry = rhat[e, 1]
                    rz = rhat[e, 2]
                    sx = spins[j, 0]
                    sy = spins[j, 1]
                    sz = spins[j, 2]
                    dot = sx * rx + sy * ry + sz * rz
                    w = weight[e]
                    hx += w * (sx - gamma * dot * rx)
                    hy += w * (sy - gamma * dot * ry)
                    hz += w * (sz - gamma * dot * rz)
                ux = spins[i, 0] / s + step * np.random.randn()
                uy = spins[i, 1] / s + step * np.random.randn()
                uz = spins[i, 2] / s + step * np.random.randn()
                un = np.sqrt(ux * ux + uy * uy + uz * uz)
                nx = s * ux / un
                ny = s * uy / un
                nz = s * uz / un
                d_e = -sigma * (
                    (nx - spins[i, 0]) * hx + (ny - spins[i, 1]) * hy + (nz - spins[i, 2]) * hz
                )
                proposed += 1
                if d_e <= 0.0 or np.random.random() < np.exp(-d_e / temperature):
                    spins[i, 0] = nx
                    spins[i, 1] = ny
                    spins[i, 2] = nz
                    accepted += 1
        if proposed > 0:
            acc_last = accepted / proposed
    return acc_last
