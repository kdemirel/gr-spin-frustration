"""Relaxation driver: greedy alignment quench (optionally preceded by Metropolis annealing)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import kernels
from .config import SimulationConfig
from .neighbors import NeighborList

_NUMBA_SEED_MOD = 2**31 - 1


@dataclass
class RelaxResult:
    """Outcome of one relaxation run.

    ``energy`` (with the factor ``G`` applied) has ``n_sweeps + 1`` entries: index ``t`` is the
    energy after ``t`` sweeps of the greedy stage (``t = 0`` is the state handed to the greedy
    stage, i.e. the annealed state for ``protocol='anneal'``). It is accumulated exactly from
    the per-spin energy changes; ``energy_drift`` is its relative deviation from a direct
    recomputation at the end (numerical round-off only).

    For ``protocol='anneal'`` ``anneal_acceptance_per_T`` holds the Metropolis acceptance rate
    at every temperature of the schedule (:func:`anneal_schedule`, hot to cold), and
    ``anneal_acceptance_final`` is its last entry: the acceptance at ``t_end``, i.e. the
    fraction of soft spins at the end of the annealing stage, not an overall rate.
    """

    spins: np.ndarray
    energy: np.ndarray
    dmax: np.ndarray
    total_spin: np.ndarray
    energy_drift: float
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)
    anneal_acceptance_per_T: np.ndarray | None = None

    @property
    def anneal_acceptance_final(self) -> float | None:
        """Acceptance rate at the final (coldest) temperature; ``None`` without annealing."""
        if self.anneal_acceptance_per_T is None:
            return None
        return float(self.anneal_acceptance_per_T[-1])


def anneal_schedule(config: SimulationConfig) -> np.ndarray:
    """Absolute temperatures of the annealing schedule, hot to cold.

    ``config.anneal`` gives the schedule in units of ``G S^2 rho``; the kernels work without
    ``G``, so the temperatures are multiplied by ``spin_scale**2 * density_effective``.
    """
    a = config.anneal
    energy_scale = config.spin_scale**2 * config.density_effective
    return np.geomspace(a.t_start, a.t_end, a.n_temperatures) * energy_scale


def relax(
    spins0: np.ndarray,
    magnitudes: np.ndarray,
    nlist: NeighborList,
    config: SimulationConfig,
    seed: int,
) -> RelaxResult:
    """Relax ``spins0`` on the neighbour list according to ``config`` (input is not modified).

    ``protocol='quench'`` runs ``config.n_sweeps`` greedy alignment sweeps; ``'anneal'`` first
    runs Metropolis annealing along :func:`anneal_schedule` (recording the acceptance rate at
    every temperature) and then the same greedy quench.
    """
    sigma, gamma = config.sigma_gamma
    n = nlist.n_particles
    spins = np.ascontiguousarray(spins0, dtype=float).copy()
    smag = np.ascontiguousarray(magnitudes, dtype=float)
    args = (nlist.indptr, nlist.indices, nlist.weight, nlist.rhat)

    acceptance = None
    if config.protocol == "anneal":
        a = config.anneal
        acceptance = np.asarray(
            kernels.metropolis(
                *args,
                spins,
                smag,
                anneal_schedule(config),
                a.sweeps_per_temperature,
                a.step,
                sigma,
                gamma,
                int(seed) % _NUMBA_SEED_MOD,
            ),
            dtype=float,
        )

    e_now = float(kernels.energy(*args, spins, sigma, gamma))
    n_sweeps = config.n_sweeps
    energy = np.empty(n_sweeps + 1)
    dmax = np.full(n_sweeps + 1, np.nan)
    total_spin = np.empty(n_sweeps + 1)
    energy[0] = e_now
    total_spin[0] = np.linalg.norm(spins.sum(axis=0))

    order = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng((int(seed), 0x5EED))
    wanted = set(config.snapshots)
    snapshots: dict[int, np.ndarray] = {}
    for t in range(1, n_sweeps + 1):
        if config.update_order == "random":
            order = rng.permutation(n).astype(np.int64)
        step_dmax, d_energy = kernels.sweep(*args, spins, smag, order, sigma, gamma)
        e_now += d_energy
        energy[t] = e_now
        dmax[t] = step_dmax
        total_spin[t] = np.linalg.norm(spins.sum(axis=0))
        if t in wanted:
            snapshots[t] = spins.copy()

    e_exact = float(kernels.energy(*args, spins, sigma, gamma))
    scale = max(abs(e_exact), 1e-300)
    drift = abs(e_now - e_exact) / scale
    return RelaxResult(
        spins=spins,
        energy=config.G * energy,
        dmax=dmax,
        total_spin=total_spin,
        energy_drift=float(drift),
        snapshots=snapshots,
        anneal_acceptance_per_T=acceptance,
    )


def total_energy(spins: np.ndarray, nlist: NeighborList, config: SimulationConfig) -> float:
    """Direct evaluation of the energy (with ``G``) of a spin configuration."""
    sigma, gamma = config.sigma_gamma
    e = kernels.energy(
        nlist.indptr,
        nlist.indices,
        nlist.weight,
        nlist.rhat,
        np.ascontiguousarray(spins, dtype=float),
        sigma,
        gamma,
    )
    return float(config.G * e)
