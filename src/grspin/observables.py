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

Further diagnostics of a configuration live here: the **dimer-frame correlation map**
(:func:`dimer_frame_map`: ``C`` and ``P`` of third spins around a mutual-nearest-neighbour pair, in
cylindrical coordinates about the pair axis) with the **dimer-field test** (the direction of every
third spin predicted from the field of the two dimer members alone, :func:`dimer_field_prediction`),
the **spin structure factor** (:func:`spin_structure_factor`) of the unit spins on the reciprocal
lattice of the periodic box, and the **energy budget** (:func:`energy_budget`: the energy of all
bonds and of the mutual-nearest-neighbour bonds against their two-body minima).
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

from .neighbors import minimum_image
from .theory import two_body_minimum_factor

#: Radius (units of ``r_nn``) of the neighbourhood mapped around every spin of a dimer.
DIMER_MAP_R_NN: float = 2.5
#: Cells of the cylindrical grid of the dimer-frame map: ``z`` (along the dimer axis, ``[-R, R]``)
#: times ``rho`` (transverse distance, ``[0, R]``), i.e. cells of ``0.1 r_nn``.
DIMER_MAP_SHAPE: tuple[int, int] = (50, 25)
#: Angular table of the dimer frame: edges in the distance ``x_ik = r_ik / r_nn`` of the third spin ...
DIMER_ANG_X_EDGES: np.ndarray = np.array([0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5])
#: ... and in the signed ``cos(theta) = a . rhat_ik`` with the axis ``a`` pointing from ``i`` to its
#: partner (``cos > 0.8``: collinear beyond the partner; ``|cos| < 0.25``: transverse; ``cos < -0.8``:
#: collinear on the far side of ``i``).
DIMER_ANG_COS_EDGES: np.ndarray = np.array([-1.0, -0.8, -0.5, -0.25, 0.25, 0.5, 0.8, 1.0])


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


# --------------------------------------------------------------------------------------
# Dimer-frame correlation map
# --------------------------------------------------------------------------------------
def _grid_sums(
    prefix: str, flat: np.ndarray, shape: tuple[int, int], c: np.ndarray, p: np.ndarray
) -> dict[str, np.ndarray]:
    """Counts and sums of ``c`` and ``p`` on a 2D grid addressed by the flat cell index."""
    n_cells = shape[0] * shape[1]
    return {
        f"{prefix}_cnt": np.bincount(flat, minlength=n_cells).reshape(shape).astype(np.int64),
        f"{prefix}_sum_C": np.bincount(flat, weights=c, minlength=n_cells).reshape(shape),
        f"{prefix}_sum_P": np.bincount(flat, weights=p, minlength=n_cells).reshape(shape),
    }


def dimer_field_prediction(
    spins: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    partner: np.ndarray,
    d_ik: np.ndarray,
    d_ij: np.ndarray,
    box: float | None,
    sigma: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Direction of the third spins ``k`` predicted by the field of the two dimer members alone.

    For the pairs ``(i, k) = (src, dst)`` with ``j = partner[i]``, ``d_ik = r_k - r_i`` and
    ``d_ij = r_j - r_i`` (minimum images), the raw field of the dimer at ``k`` is

        h = T_ki S_i + T_kj S_j,   T_ab = w_ab [1 - gamma rhat_ab rhat_ab^T],   w = 1 / r^3

    (no ``G``; the actual spin vectors ``S = s u`` enter, so the longer member dominates), and the
    zero-temperature rule predicts ``u_k = sigma h / |h|``. Returns ``(pred, defined)``: the
    predicted unit vectors (``(n, 3)``, zero where ``h = 0``) and the mask of pairs with ``h != 0``
    (``h = 0`` needs both members to have zero length, or ``gamma = 1`` with both spins along
    their bonds to ``k``). The agreement of a third spin with the prediction is
    ``u_k . pred > 0``.
    """
    partner = np.asarray(partner)
    d_jk = minimum_image(d_ik - d_ij, box)  # r_k - r_j through the nearest image

    def tensor_field(d: np.ndarray, s: np.ndarray) -> np.ndarray:
        r2 = np.einsum("ij,ij->i", d, d)
        rhat = d / np.sqrt(r2)[:, None]
        w = 1.0 / (r2 * np.sqrt(r2))
        return w[:, None] * (s - gamma * np.einsum("ij,ij->i", s, rhat)[:, None] * rhat)

    h = tensor_field(d_ik, spins[src]) + tensor_field(d_jk, spins[partner[src]])
    norm = np.linalg.norm(h, axis=1)
    defined = norm > 0.0
    pred = np.zeros_like(h)
    pred[defined] = sigma * h[defined] / norm[defined, None]
    return pred, defined


def dimer_frame_map(
    positions: np.ndarray,
    spins: np.ndarray,
    tree: cKDTree,
    box: float | None,
    r_nn: float,
    partner: np.ndarray,
    r_max_nn: float = DIMER_MAP_R_NN,
    shape: tuple[int, int] = DIMER_MAP_SHAPE,
    sigma_gamma: tuple[float, float] | None = None,
) -> dict[str, np.ndarray]:
    """Correlations of third spins in the frame of the mutual-nearest-neighbour dimers.

    For every spin ``i`` with a partner ``j = partner[i] >= 0`` (``-1``: none; both members of a
    dimer are mapped, each in its own frame) the axis is ``a = rhat_ij``, pointing from ``i`` to
    ``j``. Every other spin ``k != j`` within ``R = r_max_nn * r_nn`` of ``i`` (minimum image for
    periodic boundaries) contributes ``C = u_i . u_k`` and ``P = (u_i . rhat_ik)(u_k . rhat_ik)``

    * to the cylindrical map at ``z = a . r_ik`` in ``[-R, R]`` and ``rho = sqrt(r_ik^2 - z^2)`` in
      ``[0, R]`` (``shape[0]`` cells in ``z`` times ``shape[1]`` cells in ``rho``, units of
      ``r_nn``): keys ``dimer_map_cnt``, ``dimer_map_sum_C``, ``dimer_map_sum_P``;
    * to the angular table in ``x_ik = r_ik / r_nn`` (:data:`DIMER_ANG_X_EDGES`) times the signed
      ``cos(theta) = a . rhat_ik`` (:data:`DIMER_ANG_COS_EDGES`): keys ``dimer_ang_cnt``,
      ``dimer_ang_sum_C``, ``dimer_ang_sum_P``.

    The scalars ``dimer_map_r_max`` (``R / r_nn``), ``dimer_map_n_spins`` (number of dimer spins)
    and ``dimer_map_sum_x_partner`` (sum of ``r_ij / r_nn`` over them) complete the record. Pairs
    with a zero-length spin are skipped. Fully vectorised (one ``query_pairs`` of the tree).

    **Dimer-field test.** With ``sigma_gamma = (sigma, gamma)`` of the coupling the record also
    holds the causal test of :func:`dimer_field_prediction`: for the same pairs the direction of
    ``k`` predicted by the field of ``i`` and ``j`` alone is compared with the actual ``u_k``
    (``agree = u_k . pred > 0``; random directions agree in half of the cases). Keys, over the
    pairs with a defined prediction: ``dimer_field_cnt`` and ``dimer_field_agree`` (counts and
    agreements per bin of ``x_ik``, the bins of the angular table), ``dimer_field_sum_cos``
    (sum of ``u_k . pred`` per bin), ``dimer_ang_sum_Cpred`` (sum of ``pred . u_i`` per angular
    cell: the dimer-only prediction of the angular table of ``C``), ``dimer_field_n`` and
    ``dimer_field_n_agree`` (all pairs within ``R``, including ``x_ik`` below the first edge), and
    ``dimer_field_sigma``, ``dimer_field_gamma``.
    """
    n_z, n_rho = shape
    r_max = r_max_nn * r_nn
    partner = np.asarray(partner)
    dimer = np.flatnonzero(partner >= 0)
    unit, smag = unit_vectors(spins)
    nonzero = smag > 0.0

    d_ij = positions[partner[dimer]] - positions[dimer]
    minimum_image(d_ij, box)
    r_ij = np.linalg.norm(d_ij, axis=1)
    axis = np.zeros_like(positions)
    axis[dimer] = d_ij / r_ij[:, None]
    to_partner = np.zeros_like(positions)  # r_j - r_i for every dimer spin i
    to_partner[dimer] = d_ij

    pairs = tree.query_pairs(float(r_max), output_type="ndarray")
    src = np.concatenate((pairs[:, 0], pairs[:, 1]))  # the dimer spin i
    dst = np.concatenate((pairs[:, 1], pairs[:, 0]))  # the third spin k
    keep = (partner[src] >= 0) & (partner[src] != dst) & nonzero[src] & nonzero[dst]
    src, dst = src[keep], dst[keep]
    d = positions[dst] - positions[src]
    minimum_image(d, box)
    r = np.linalg.norm(d, axis=1)
    z = np.einsum("ij,ij->i", axis[src], d)
    rho = np.sqrt(np.maximum(r * r - z * z, 0.0))
    rhat = d / r[:, None]
    c = np.einsum("ij,ij->i", unit[src], unit[dst])
    p = np.einsum("ij,ij->i", unit[src], rhat) * np.einsum("ij,ij->i", unit[dst], rhat)

    out: dict[str, np.ndarray] = {
        "dimer_map_r_max": np.array(float(r_max_nn)),
        "dimer_map_n_spins": np.array(dimer.size),
        "dimer_map_sum_x_partner": np.array(float(r_ij.sum() / r_nn)),
    }
    iz = np.clip(np.floor((z + r_max) / (2.0 * r_max) * n_z), 0, n_z - 1).astype(np.int64)
    ir = np.clip(np.floor(rho / r_max * n_rho), 0, n_rho - 1).astype(np.int64)
    out.update(_grid_sums("dimer_map", iz * n_rho + ir, shape, c, p))

    x = r / r_nn
    cos_theta = np.clip(z / r, -1.0, 1.0)
    n_x, n_c = len(DIMER_ANG_X_EDGES) - 1, len(DIMER_ANG_COS_EDGES) - 1
    ix = np.minimum(np.digitize(x, DIMER_ANG_X_EDGES) - 1, n_x - 1)  # x == last edge: last bin
    ic = np.minimum(np.digitize(cos_theta, DIMER_ANG_COS_EDGES) - 1, n_c - 1)  # cos == 1 likewise
    inside = (ix >= 0) & (x <= DIMER_ANG_X_EDGES[-1])
    flat = ix * n_c + ic
    out.update(_grid_sums("dimer_ang", flat[inside], (n_x, n_c), c[inside], p[inside]))

    if sigma_gamma is not None:
        sigma, gamma = (float(v) for v in sigma_gamma)
        pred, defined = dimer_field_prediction(
            spins, src, dst, partner, d, to_partner[src], box, sigma, gamma
        )
        cos_pred = np.einsum("ij,ij->i", unit[dst], pred)  # u_k . prediction (0 if undefined)
        agree = (cos_pred > 0.0) & defined
        sel = inside & defined
        out["dimer_field_cnt"] = np.bincount(ix[sel], minlength=n_x).astype(np.int64)
        out["dimer_field_agree"] = np.bincount(ix[sel & agree], minlength=n_x).astype(np.int64)
        out["dimer_field_sum_cos"] = np.bincount(ix[sel], weights=cos_pred[sel], minlength=n_x)
        c_pred = np.einsum("ij,ij->i", pred, unit[src])
        out["dimer_ang_sum_Cpred"] = np.bincount(
            flat[inside], weights=c_pred[inside], minlength=n_x * n_c
        ).reshape(n_x, n_c)
        out["dimer_field_n"] = np.array(int(np.count_nonzero(defined)))
        out["dimer_field_n_agree"] = np.array(int(np.count_nonzero(agree)))
        out["dimer_field_sigma"] = np.array(sigma)
        out["dimer_field_gamma"] = np.array(gamma)
    return out


# --------------------------------------------------------------------------------------
# Spin structure factor
# --------------------------------------------------------------------------------------
def reciprocal_lattice_vectors(box: float, n_max: int) -> tuple[np.ndarray, np.ndarray]:
    """Integer vectors ``n`` with ``0 < |n| <= n_max`` (one per ``±n`` pair) and ``k = 2 pi n / L``.

    ``S(k) = S(-k)`` for real spins, so only the half space ``n_x > 0``, or ``n_x = 0 < n_y``, or
    ``n_x = n_y = 0 < n_z`` is listed (about ``2 pi n_max^3 / 3`` vectors).
    """
    r = np.arange(-n_max, n_max + 1)
    n = np.stack(np.meshgrid(r, r, r, indexing="ij"), axis=-1).reshape(-1, 3)
    inside = np.einsum("ij,ij->i", n, n) <= n_max * n_max
    half = (n[:, 0] > 0) | ((n[:, 0] == 0) & ((n[:, 1] > 0) | ((n[:, 1] == 0) & (n[:, 2] > 0))))
    n = n[inside & half]
    return n, (2.0 * np.pi / box) * n.astype(float)


def spin_structure_factor(
    positions: np.ndarray, spins: np.ndarray, k_vectors: np.ndarray, chunk: int = 512
) -> np.ndarray:
    """``S(k) = |sum_i u_i exp(i k . r_i)|^2 / N`` of the unit spins at the vectors ``k_vectors``.

    Uncorrelated directions give ``S = 1`` on average; ``k = 0`` reproduces ``s0_unit`` of
    :func:`spin_summary` (``N`` for a ferromagnet). ``spins`` may also hold several spin
    configurations on the same positions (shape ``(S, N, 3)``, e.g. initial and final); the result
    then has shape ``(S, len(k_vectors))`` and the phases are computed once. Direct summation over
    ``chunk`` wave vectors at a time (``N x chunk`` complex phases), so the cost is ``N`` times the
    number of vectors.
    """
    spins = np.asarray(spins, dtype=float)
    k_vectors = np.asarray(k_vectors, dtype=float).reshape(-1, 3)
    sets = [spins] if spins.ndim == 2 else list(spins)
    units = [unit_vectors(s)[0] for s in sets]
    n = positions.shape[0]
    out = np.empty((len(units), k_vectors.shape[0]))
    for start in range(0, k_vectors.shape[0], chunk):
        phase = np.exp(1j * (positions @ k_vectors[start : start + chunk].T))  # (N, m)
        for row, unit in enumerate(units):
            amplitude = unit.T @ phase  # (3, m)
            out[row, start : start + chunk] = np.sum(np.abs(amplitude) ** 2, axis=0) / n
    return out[0] if spins.ndim == 2 else out


# --------------------------------------------------------------------------------------
# Pair energies and the energy budget
# --------------------------------------------------------------------------------------
def pair_energies(
    positions: np.ndarray,
    spins: np.ndarray,
    i: np.ndarray,
    j: np.ndarray,
    box: float | None,
    sigma: float,
    gamma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Raw pair energies, their two-body minima and the distances of the pairs ``(i, j)``.

    ``U_ij = -sigma w [S_i . S_j - gamma (S_i . rhat)(S_j . rhat)]`` with ``w = 1 / r^3`` (minimum
    image for periodic boundaries, no factor ``G``) and the isolated-pair minimum
    ``U_min = -f w s_i s_j`` with ``f = `` :func:`grspin.theory.two_body_minimum_factor`.
    Returns ``(U, U_min, r)``.
    """
    d = positions[j] - positions[i]
    minimum_image(d, box)
    r2 = np.einsum("ij,ij->i", d, d)
    r = np.sqrt(r2)
    w = 1.0 / (r2 * r)
    rhat = d / r[:, None]
    si, sj = spins[i], spins[j]
    dots = np.einsum("ij,ij->i", si, sj)
    proj = np.einsum("ij,ij->i", si, rhat) * np.einsum("ij,ij->i", sj, rhat)
    energy = -sigma * w * (dots - gamma * proj)
    factor = two_body_minimum_factor(sigma, gamma)
    minimum = -factor * w * np.linalg.norm(si, axis=1) * np.linalg.norm(sj, axis=1)
    return energy, minimum, r


def mutual_pairs(partner: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Index arrays ``(i, j)`` with ``i < j`` of the mutual-nearest-neighbour pairs (each once)."""
    partner = np.asarray(partner)
    i = np.flatnonzero(partner > np.arange(partner.shape[0]))
    return i, partner[i]


def energy_budget(
    positions: np.ndarray,
    spins: np.ndarray,
    tree: cKDTree,
    box: float | None,
    r_cut: float,
    sigma: float,
    gamma: float,
    partner: np.ndarray,
) -> dict[str, float]:
    """Energy of all bonds within ``r_cut`` and of the mutual-NN bonds against their two-body minima.

    Raw energies (no ``G``) of one configuration: ``energy_total`` is the sum of the pair energies
    over all pairs with ``r <= r_cut`` (the energy ``H`` of the relaxation), ``minimum_total`` the
    sum of their isolated-pair minima ``-f w s_i s_j``; ``energy_dimer`` and ``minimum_dimer`` are
    the same sums over the mutual-nearest-neighbour pairs (``partner`` of
    :func:`grspin.nn.mutual_nn_partners`; pairs beyond ``r_cut``, which cannot occur in practice,
    are left out). ``n_pairs`` and ``n_dimers`` count the pairs. The ratios formed from these
    sums (``energy_dimer / energy_total``: dimer share; ``energy_dimer / minimum_dimer``: realized
    fraction of the dimers; ``energy_total / minimum_total``: the frustration index ``E / sum of
    two-body minima``, ``1`` when every bond sits at its own minimum) do not depend on ``G``.
    """
    pairs = tree.query_pairs(float(r_cut), output_type="ndarray")
    energy, minimum, _ = pair_energies(
        positions, spins, pairs[:, 0], pairs[:, 1], box, sigma, gamma
    )
    i, j = mutual_pairs(partner)
    e_dimer, m_dimer, r_dimer = pair_energies(positions, spins, i, j, box, sigma, gamma)
    within = r_dimer <= float(r_cut)
    return {
        "n_pairs": int(pairs.shape[0]),
        "n_dimers": int(np.count_nonzero(within)),
        "energy_total": float(energy.sum()),
        "minimum_total": float(minimum.sum()),
        "energy_dimer": float(e_dimer[within].sum()),
        "minimum_dimer": float(m_dimer[within].sum()),
    }
