"""Simulation configuration and derived quantities.

All lengths are measured in the units of the point process (number density ``density``),
spins in units of ``spin_scale`` and energies in units of ``G``. The relaxation dynamics is
invariant under a rescaling of ``box_length``, ``spin_scale`` and ``G`` (only the direction of
the local field matters), so the only physically independent quantities are the dimensionless
ones: ``n_particles``, ``r_cut_nn``, ``n_sweeps``, the *shape* of the spin-magnitude
distribution, the boundary condition and the coupling. Distances are reported in units of the
mean nearest-neighbour distance ``r_nn = Gamma(4/3) (3 / (4 pi rho))^(1/3) ~ 0.554 rho^(-1/3)``.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

#: Mean nearest-neighbour distance of a 3D Poisson process at unit density.
RNN_UNIT_DENSITY: float = math.gamma(4.0 / 3.0) * (3.0 / (4.0 * math.pi)) ** (1.0 / 3.0)

#: Coupling variants as ``name -> (sigma, gamma)`` in
#: ``H = -sigma * sum_{i<j} G / r^3 * [S_i.S_j - gamma (S_i.r)(S_j.r)]``.
#: ``gr`` is the general-relativistic spin-spin interaction; the other two are controls.
COUPLINGS: dict[str, tuple[float, float]] = {
    "gr": (1.0, 3.0),
    "isotropic_afm": (-1.0, 0.0),
    "dipolar": (-1.0, 3.0),
}
SPIN_DISTRIBUTIONS: tuple[str, ...] = ("uniform", "uniform_half", "equal", "lognormal")
BOUNDARIES: tuple[str, ...] = ("periodic", "open")
PROTOCOLS: tuple[str, ...] = ("quench", "anneal")
UPDATE_ORDERS: tuple[str, ...] = ("sequential", "random")

#: Default cutoff (in units of ``r_nn``) when ``r_cut_nn`` is not given.
DEFAULT_R_CUT_NN: float = 10.0

# Cost model of one realization on one core of an Apple-silicon laptop. The constants were fitted
# to ``grspin benchmark`` measurements (periodic boundaries, N = 500 ... 20000, r_cut = 3 ... 15 r_nn);
# the model reproduces them within ~25 %. Run ``grspin benchmark`` on your machine to check.
_SECONDS_PER_EDGE_BUILD: float = 2.5e-7  # neighbour-list assembly, per directed edge
_SECONDS_PER_EDGE_SWEEP: float = 2.3e-9  # one greedy sweep (or energy pass), per directed edge
_METROPOLIS_SWEEP_FACTOR: float = 1.3  # a Metropolis sweep costs this many greedy sweeps
_SECONDS_PER_PAIR_STATISTIC: float = 3.0e-7  # pair statistics, per pair within x_max and per call
_SECONDS_PER_PARTICLE: float = 5.0e-7  # initial state etc.
_BYTES_PER_EDGE_PEAK: float = 220.0  # peak memory while the neighbour list is assembled
_BASE_MEMORY_MB: float = 250.0  # interpreter, numba, scipy and the observables


@dataclass(frozen=True)
class AnnealConfig:
    """Metropolis annealing schedule (temperatures in units of ``G * spin_scale**2 * density``)."""

    t_start: float = 2.0
    t_end: float = 0.01
    n_temperatures: int = 40
    sweeps_per_temperature: int = 10
    step: float = 0.3  # width of the Gaussian direction proposal

    def __post_init__(self) -> None:
        if not (self.t_start > self.t_end > 0.0):
            raise ValueError("annealing requires t_start > t_end > 0")
        if self.n_temperatures < 1 or self.sweeps_per_temperature < 1:
            raise ValueError("n_temperatures and sweeps_per_temperature must be >= 1")
        if self.step <= 0.0:
            raise ValueError("annealing step must be positive")


@dataclass(frozen=True)
class SimulationConfig:
    """Parameters of one ensemble of relaxation runs (one set of physical parameters)."""

    # --- system -----------------------------------------------------------------------
    n_particles: int = 5000
    density: float = 1.0
    box_length: float | None = None  # if given, overrides ``density`` (rho = N / L^3)
    boundary: str = "periodic"
    coupling: str = "gr"
    spin_distribution: str = "uniform"
    spin_scale: float = 1.0
    lognormal_sigma: float = 1.0
    G: float = 1.0
    # --- numerics ---------------------------------------------------------------------
    r_cut_nn: float | None = None  # None: min(10, L / (2 r_nn)) for periodic, 10 for open
    n_sweeps: int = 100
    update_order: str = "sequential"
    protocol: str = "quench"
    anneal: AnnealConfig = field(default_factory=AnnealConfig)
    # --- observables ------------------------------------------------------------------
    n_bins: int = 40
    x_max: float = 4.0
    extra_observables: bool = True
    nn_diagnostics: bool = False
    snapshots: tuple[int, ...] = ()
    save_states: int = 0  # store positions/spins of the first k realizations
    # --- ensemble ---------------------------------------------------------------------
    n_realizations: int = 100
    base_seed: int = 1

    def __post_init__(self) -> None:
        def _bad(msg: str) -> None:
            raise ValueError(msg)

        if self.n_particles < 2:
            _bad("n_particles must be >= 2")
        if self.box_length is not None and not self.box_length > 0.0:
            _bad("box_length must be positive")
        if not self.density > 0.0:
            _bad("density must be positive")
        if self.boundary not in BOUNDARIES:
            _bad(f"boundary must be one of {BOUNDARIES}")
        if self.coupling not in COUPLINGS:
            _bad(f"coupling must be one of {tuple(COUPLINGS)}")
        if self.spin_distribution not in SPIN_DISTRIBUTIONS:
            _bad(f"spin_distribution must be one of {SPIN_DISTRIBUTIONS}")
        if not self.spin_scale > 0.0 or not self.G > 0.0 or not self.lognormal_sigma > 0.0:
            _bad("spin_scale, G and lognormal_sigma must be positive")
        if self.update_order not in UPDATE_ORDERS:
            _bad(f"update_order must be one of {UPDATE_ORDERS}")
        if self.protocol not in PROTOCOLS:
            _bad(f"protocol must be one of {PROTOCOLS}")
        if self.n_sweeps < 0:
            _bad("n_sweeps must be >= 0")
        if self.n_bins < 1 or not self.x_max > 0.0:
            _bad("n_bins must be >= 1 and x_max > 0")
        if self.n_realizations < 1 or self.base_seed < 0:
            _bad("n_realizations must be >= 1 and base_seed >= 0")
        if self.save_states < 0:
            _bad("save_states must be >= 0")
        snaps = tuple(int(s) for s in self.snapshots)
        if any(s < 1 or s > self.n_sweeps for s in snaps) or len(set(snaps)) != len(snaps):
            _bad("snapshots must be distinct sweep numbers in [1, n_sweeps]")
        object.__setattr__(self, "snapshots", tuple(sorted(snaps)))
        if self.r_cut_nn is not None and not self.r_cut_nn > 0.0:
            _bad("r_cut_nn must be positive")
        if self.boundary == "periodic":
            tol = 1.0 + 1e-12
            if self.r_cut > 0.5 * self.box * tol:
                n_min = int(math.ceil(self.density_effective * (2.0 * self.r_cut) ** 3))
                _bad(
                    "periodic boundaries require r_cut <= L/2: "
                    f"r_cut = {self.r_cut_nn_resolved:.3g} r_nn but L/2 = "
                    f"{0.5 * self.box / self.r_nn:.3g} r_nn. Use n_particles >= {n_min}, "
                    "a smaller r_cut_nn (or None for 'auto'), or boundary='open'."
                )
            if self.x_max * self.r_nn > 0.5 * self.box * tol:
                _bad("periodic boundaries require x_max * r_nn <= L/2")

    # --- derived geometry ------------------------------------------------------------
    @property
    def box(self) -> float:
        """Edge length ``L`` of the cubic box."""
        if self.box_length is not None:
            return float(self.box_length)
        return float((self.n_particles / self.density) ** (1.0 / 3.0))

    @property
    def density_effective(self) -> float:
        """Number density ``rho = N / L^3``."""
        return float(self.n_particles / self.box**3)

    @property
    def r_nn(self) -> float:
        """Mean nearest-neighbour distance of the Poisson process."""
        return float(RNN_UNIT_DENSITY * self.density_effective ** (-1.0 / 3.0))

    @property
    def r_cut_nn_resolved(self) -> float:
        """Cutoff in units of ``r_nn`` (resolving ``None`` to the automatic value)."""
        if self.r_cut_nn is not None:
            return float(self.r_cut_nn)
        if self.boundary == "periodic":
            return float(min(DEFAULT_R_CUT_NN, 0.5 * self.box / self.r_nn))
        return DEFAULT_R_CUT_NN

    @property
    def r_cut(self) -> float:
        """Interaction cutoff radius (absolute units)."""
        return float(self.r_cut_nn_resolved * self.r_nn)

    @property
    def x_edges(self) -> np.ndarray:
        """Bin edges in units of ``r_nn``."""
        return np.linspace(0.0, self.x_max, self.n_bins + 1)

    @property
    def seeds(self) -> list[int]:
        return [self.base_seed + k for k in range(self.n_realizations)]

    @property
    def sigma_gamma(self) -> tuple[float, float]:
        return COUPLINGS[self.coupling]

    @property
    def neighbors_per_particle(self) -> float:
        """Expected number of neighbours within the cutoff (bulk value)."""
        return float(4.0 / 3.0 * math.pi * self.density_effective * self.r_cut**3)

    def estimate_cost(self) -> dict[str, float]:
        """Rough single-realization cost: ``seconds`` (one core), ``memory_mb`` (peak) and ``edges``.

        The time is the sum of the neighbour-list assembly, the sweeps (plus the two energy
        passes), the pair statistics of the initial, final and snapshot states (plus the
        nearest-neighbour diagnostics if requested) and a small per-particle term.
        """
        edges = self.n_particles * self.neighbors_per_particle  # directed edges
        sweeps = self.n_sweeps + 2
        if self.protocol == "anneal":
            a = self.anneal
            sweeps += _METROPOLIS_SWEEP_FACTOR * a.n_temperatures * a.sweeps_per_temperature
        pairs_observed = (
            0.5
            * self.n_particles
            * 4.0
            / 3.0
            * math.pi
            * self.density_effective
            * (self.x_max * self.r_nn) ** 3
        )
        calls = 2 + len(self.snapshots) + (1 if self.nn_diagnostics else 0)
        seconds = (
            edges * (_SECONDS_PER_EDGE_BUILD + sweeps * _SECONDS_PER_EDGE_SWEEP)
            + pairs_observed * calls * _SECONDS_PER_PAIR_STATISTIC
            + self.n_particles * _SECONDS_PER_PARTICLE
            + 0.01
        )
        memory_mb = _BASE_MEMORY_MB + edges * _BYTES_PER_EDGE_PEAK / 1e6
        return {"seconds": float(seconds), "memory_mb": float(memory_mb), "edges": float(edges)}

    # --- (de)serialization -----------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["snapshots"] = list(self.snapshots)
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SimulationConfig:
        d = dict(d)
        if "anneal" in d and isinstance(d["anneal"], dict):
            d["anneal"] = AnnealConfig(**d["anneal"])
        if "snapshots" in d:
            d["snapshots"] = tuple(d["snapshots"])
        known = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})

    def compat_key(self) -> dict[str, Any]:
        """Everything that determines the *content* of a realization file.

        ``n_realizations`` (extend/shrink an ensemble) and ``save_states`` (which realizations
        also store their particle states) do not change the physics, so they are excluded.
        """
        d = self.to_dict()
        d.pop("n_realizations")
        d.pop("save_states")
        return d

    def replace(self, **overrides: Any) -> SimulationConfig:
        """Return a modified copy (re-validated)."""
        return dataclasses.replace(self, **overrides)
