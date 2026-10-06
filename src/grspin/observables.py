"""Pair observables from a spin configuration (all pairs closer than ``x_max * r_nn``).

For a pair ``(i, j)`` at separation ``r`` with unit spins ``u_i, u_j`` and bond direction ``rhat``:

* ``C``  = ``u_i . u_j``                                  spin-spin correlation
* ``Q``  = ``cos^2(phi)``, ``phi`` = angle between ``S_i + S_j`` and ``rhat``  (pair total spin)
* ``A``  = ``[(u_i . rhat)^2 + (u_j . rhat)^2] / 2``       single-spin alignment with the bond axis
* ``P``  = ``(u_i . rhat)(u_j . rhat)``                     bond-projected (longitudinal) correlation
* ``B``  = ``C - 3 P``                                      tensor correlation; the GR pair energy is
                                                            ``U_ij = -B G s_i s_j / r^3``, so ``B = 2`` for a
                                                            perfect dimer and ``B = 0`` for random spins

The functions return *sums and counts per radial bin* (not means), so that ensembles can be
re-binned exactly and pooled estimates can be formed afterwards.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .neighbors import minimum_image


def _bin_sums(cat: np.ndarray, mask: np.ndarray, values: np.ndarray | None, n_bins: int):
    sel = cat[mask]
    counts = np.bincount(sel, minlength=n_bins).astype(np.int64)
    if values is None:
        return counts
    sums = np.bincount(sel, weights=values[mask], minlength=n_bins)
    return counts, sums


def unit_vectors(spins: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit directions (zero for zero-length spins) and magnitudes."""
    smag = np.linalg.norm(spins, axis=1)
    unit = np.zeros_like(spins)
    nonzero = smag > 0.0
    unit[nonzero] = spins[nonzero] / smag[nonzero, None]
    return unit, smag


def pair_statistics(
    positions: np.ndarray,
    spins: np.ndarray,
    tree: cKDTree,
    box: float | None,
    r_nn: float,
    x_edges: np.ndarray,
    extras: bool = True,
) -> dict[str, np.ndarray]:
    """Binned sums/counts of the pair observables.

    Keys: ``cnt_C, sum_C, cnt_Q, sum_Q, sum_x`` and, if ``extras``, ``sum_A, sum_P, sum_B``
    (which share the counts ``cnt_C``). ``sum_x`` is the sum of ``r / r_nn`` over the pairs
    counted in ``cnt_C``; it gives the pair-weighted mean distance of each bin.
    """
    n_bins = len(x_edges) - 1
    r_edges = np.asarray(x_edges) * r_nn
    pairs = tree.query_pairs(float(r_edges[-1]), output_type="ndarray")
    i = pairs[:, 0]
    j = pairs[:, 1]
    d = positions[j] - positions[i]
    minimum_image(d, box)
    r = np.linalg.norm(d, axis=1)
    rhat = d / r[:, None]

    cat = np.digitize(r, r_edges) - 1
    in_range = (cat >= 0) & (cat < n_bins)

    unit, smag = unit_vectors(spins)
    nonzero = smag > 0.0
    valid_c = in_range & nonzero[i] & nonzero[j]
    ui = unit[i]
    uj = unit[j]
    cos_theta = np.clip(np.einsum("ij,ij->i", ui, uj), -1.0, 1.0)

    total = spins[i] + spins[j]
    total_norm = np.linalg.norm(total, axis=1)
    valid_q = in_range & (total_norm > 0.0)
    cos_phi = np.zeros_like(r)
    np.divide(np.einsum("ij,ij->i", total, rhat), total_norm, out=cos_phi, where=total_norm > 0.0)
    cos_phi_sq = np.clip(cos_phi, -1.0, 1.0) ** 2

    out: dict[str, np.ndarray] = {}
    out["cnt_C"], out["sum_C"] = _bin_sums(cat, valid_c, cos_theta, n_bins)
    out["cnt_Q"], out["sum_Q"] = _bin_sums(cat, valid_q, cos_phi_sq, n_bins)
    _, out["sum_x"] = _bin_sums(cat, valid_c, r / r_nn, n_bins)
    if extras:
        ai = np.einsum("ij,ij->i", ui, rhat)
        aj = np.einsum("ij,ij->i", uj, rhat)
        _, out["sum_A"] = _bin_sums(cat, valid_c, 0.5 * (ai * ai + aj * aj), n_bins)
        proj = ai * aj
        _, out["sum_P"] = _bin_sums(cat, valid_c, proj, n_bins)
        _, out["sum_B"] = _bin_sums(cat, valid_c, cos_theta - 3.0 * proj, n_bins)
    return out


def spin_summary(spins: np.ndarray) -> dict[str, float]:
    """Global spin diagnostics.

    ``total_spin``  = ``|sum_i S_i|`` (magnitude-weighted);
    ``s0_unit``     = ``|sum_i u_i|^2 / N``, the ``k = 0`` spin structure factor of unit spins
                      (``1`` for uncorrelated directions; ``~N`` for a ferromagnet).
    """
    unit, _ = unit_vectors(spins)
    n = spins.shape[0]
    return {
        "total_spin": float(np.linalg.norm(spins.sum(axis=0))),
        "s0_unit": float(np.sum(unit.sum(axis=0) ** 2) / n),
    }
