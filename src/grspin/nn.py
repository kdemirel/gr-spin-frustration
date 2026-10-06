"""Nearest-neighbour diagnostics of a relaxed configuration.

These quantities test the *dimer picture* behind the short-distance behaviour of ``C(r)``: a pair
that is mutually nearest neighbours (a "dimer") is expected to anti-align along its axis, whereas
the remaining pairs are governed by many-body effects. All results are sums/counts (or per-configuration
means) so that they can be aggregated over an ensemble.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .neighbors import minimum_image
from .observables import unit_vectors

#: Number of neighbour ranks resolved by the rank-resolved correlation ``C_k``.
NN_RANK_MAX = 8
#: Log-spaced bin edges (in units of ``r_nn``) for the small-distance law of ``1 + C``.
SMALL_R_EDGES_X = np.geomspace(0.05, 0.8, 13)
#: Bin edges (units of ``r_nn``) of the nearest-neighbour-distance histogram.
NN_HIST_EDGES_X = np.linspace(0.0, 3.0, 61)


def nn_statistics(
    positions: np.ndarray,
    spins: np.ndarray,
    tree: cKDTree,
    box: float | None,
    r_nn: float,
    x_edges: np.ndarray,
    k_max: int = NN_RANK_MAX,
) -> dict[str, np.ndarray]:
    """Nearest-neighbour statistics for one configuration.

    Returns arrays keyed by
      ``nn_hist``            counts of first-neighbour distances (``NN_HIST_EDGES_X``);
      ``nn_frac_mutual``     fraction of particles that belong to a mutual-NN pair (scalar array);
      ``nn_Ck, nn_rk``       mean ``u_i . u_(k-th neighbour)`` and mean ``r_k / r_nn`` for ``k = 1..k_max``;
      ``nn_{cnt,sum}_{mutual,oneway,far}``  per radial bin (edges ``x_edges``): pair counts and sums of
                             ``cos(theta)`` for mutual-NN pairs, pairs that are the NN of exactly one
                             member, and pairs that are nobody's NN;
      ``sr_cnt, sr_sum1pC, sr_sumx``  small-distance log bins (``SMALL_R_EDGES_X``): counts, sum of ``1 + cos``
                             and sum of ``r / r_nn``.
    """
    n = positions.shape[0]
    if n <= k_max + 1:
        raise ValueError(f"nn diagnostics need more than {k_max + 1} particles")
    unit, smag = unit_vectors(spins)
    nonzero = smag > 0.0

    dist, ind = tree.query(positions, k=k_max + 1)
    nn1 = ind[:, 1]
    frac_mutual = float(np.mean(nn1[nn1] == np.arange(n)))
    hist = np.histogram(dist[:, 1] / r_nn, bins=NN_HIST_EDGES_X)[0].astype(np.int64)

    ck = np.zeros(k_max)
    rk = np.zeros(k_max)
    for k in range(1, k_max + 1):
        neighbor = ind[:, k]
        ok = nonzero & nonzero[neighbor]
        ck[k - 1] = (
            np.mean(np.einsum("ij,ij->i", unit[ok], unit[neighbor[ok]])) if ok.any() else np.nan
        )
        rk[k - 1] = np.mean(dist[:, k]) / r_nn

    n_bins = len(x_edges) - 1
    r_edges = np.asarray(x_edges) * r_nn
    small_edges = SMALL_R_EDGES_X * r_nn
    r_max = float(max(r_edges[-1], small_edges[-1]))
    pairs = tree.query_pairs(r_max, output_type="ndarray")
    i = pairs[:, 0]
    j = pairs[:, 1]
    d = positions[j] - positions[i]
    minimum_image(d, box)
    r = np.linalg.norm(d, axis=1)
    valid = nonzero[i] & nonzero[j]
    cos_theta = np.clip(np.einsum("ij,ij->i", unit[i], unit[j]), -1.0, 1.0)

    nn_i = nn1[i] == j
    nn_j = nn1[j] == i
    categories = {"mutual": nn_i & nn_j, "oneway": nn_i ^ nn_j, "far": ~(nn_i | nn_j)}
    cat = np.digitize(r, r_edges) - 1
    in_range = (cat >= 0) & (cat < n_bins) & valid

    out: dict[str, np.ndarray] = {
        "nn_hist": hist,
        "nn_frac_mutual": np.array(frac_mutual),
        "nn_Ck": ck,
        "nn_rk": rk,
    }
    for name, mask in categories.items():
        sel = in_range & mask
        out[f"nn_cnt_{name}"] = np.bincount(cat[sel], minlength=n_bins).astype(np.int64)
        out[f"nn_sum_{name}"] = np.bincount(cat[sel], weights=cos_theta[sel], minlength=n_bins)

    small = np.digitize(r, small_edges) - 1
    n_small = len(small_edges) - 1
    sel = (small >= 0) & (small < n_small) & valid
    out["sr_cnt"] = np.bincount(small[sel], minlength=n_small).astype(np.int64)
    out["sr_sum1pC"] = np.bincount(small[sel], weights=1.0 + cos_theta[sel], minlength=n_small)
    out["sr_sumx"] = np.bincount(small[sel], weights=r[sel] / r_nn, minlength=n_small)
    return out
