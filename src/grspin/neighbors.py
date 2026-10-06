"""Neighbour lists: CSR adjacency with periodic (minimum-image) or open boundaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


@dataclass
class NeighborList:
    """Directed CSR neighbour list within the cutoff radius.

    For particle ``i`` the neighbours are ``indices[indptr[i]:indptr[i+1]]`` with weights
    ``weight = 1/r^3`` and unit separation vectors ``rhat`` (the sign of ``rhat`` is
    irrelevant: it enters the interaction quadratically).
    """

    indptr: np.ndarray  # (N + 1,) int64
    indices: np.ndarray  # (E,) int32
    weight: np.ndarray  # (E,) float64
    rhat: np.ndarray  # (E, 3) float64
    tree: cKDTree
    box: float | None  # edge length for periodic boundaries, ``None`` for open

    @property
    def n_particles(self) -> int:
        return int(self.indptr.shape[0] - 1)

    @property
    def n_edges(self) -> int:
        """Number of directed edges (twice the number of pairs)."""
        return int(self.indices.shape[0])

    def neighbors_of(self, i: int) -> np.ndarray:
        return self.indices[self.indptr[i] : self.indptr[i + 1]]


def build_tree(positions: np.ndarray, box: float | None) -> cKDTree:
    """KD-tree of the positions (periodic if ``box`` is given)."""
    if box is not None and (np.any(positions < 0.0) or np.any(positions >= box)):
        raise ValueError("periodic boundaries require all coordinates in [0, L)")
    return cKDTree(positions, boxsize=box)


def minimum_image(displacement: np.ndarray, box: float | None) -> np.ndarray:
    """Apply the minimum-image convention in place (no-op for open boundaries)."""
    if box is not None:
        displacement -= box * np.round(displacement / box)
    return displacement


def build_neighbor_list(
    positions: np.ndarray,
    r_cut: float,
    box: float | None = None,
    tree: cKDTree | None = None,
) -> NeighborList:
    """All pairs closer than ``r_cut`` as a directed CSR list.

    For periodic boundaries ``r_cut`` must not exceed ``box / 2`` so that each pair has a
    unique minimum image inside the cutoff sphere.
    """
    if box is not None and r_cut > 0.5 * box * (1.0 + 1e-12):
        raise ValueError("periodic neighbour lists require r_cut <= box / 2")
    n = positions.shape[0]
    if tree is None:
        tree = build_tree(positions, box)

    pairs = tree.query_pairs(r_cut, output_type="ndarray")
    i = pairs[:, 0]
    j = pairs[:, 1]
    del pairs
    d = positions[j] - positions[i]
    minimum_image(d, box)
    r2 = np.einsum("ij,ij->i", d, d)
    if np.any(r2 == 0.0):
        raise ValueError("coincident particles are not allowed")
    r = np.sqrt(r2)
    d /= r[:, None]  # d now holds the unit vectors
    w = 1.0 / (r2 * r)
    del r2, r

    src = np.concatenate((i, j))
    order = np.argsort(src, kind="stable")
    indices = np.concatenate((j, i))[order].astype(np.int32)
    weight = np.concatenate((w, w))[order]
    rhat = np.concatenate((d, d))[order]
    counts = np.bincount(src, minlength=n)
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.cumsum(counts, out=indptr[1:])
    return NeighborList(
        indptr=indptr,
        indices=indices,
        weight=weight,
        rhat=np.ascontiguousarray(rhat),
        tree=tree,
        box=box,
    )
