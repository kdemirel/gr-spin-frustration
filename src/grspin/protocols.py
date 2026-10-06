"""Protocol check: how much do the relaxed states depend on how they were prepared?

The zero-temperature alignment rule only finds *a* local minimum of the energy. To quantify how
much the reported correlations depend on that choice, four preparations are compared on
**identical positions and spin magnitudes** (so that all differences are paired):

``sequential``    the production protocol: in-place sweeps in index order from the random state
``random_order``  the same, but every sweep visits the spins in a fresh random order
``other_init``    sequential sweeps from an independent isotropic initial direction field
``anneal``        Metropolis annealing (``config.anneal``) followed by the greedy quench

For every realization the final energy, the gauge-insensitive overlap of the final direction fields
with the production state and the binned pair sums are stored; :class:`ProtocolData` turns them
into paired differences with standard errors over realizations.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import landmark_values, mean_sem
from .config import SimulationConfig
from .ensemble import (
    CONFIG_FILENAME,
    RunInfo,
    _atomic_savez,
    _fmt_duration,
    execute_tasks,
    load_record,
    prepare_run_dir,
    read_config,
    resolve_workers,
    warmup,
    write_run_config,
)
from .initial import generate_initial_state
from .neighbors import build_neighbor_list
from .observables import pair_statistics, unit_vectors
from .relax import relax

PROTOCOL_DIRNAME = "protocol"
VARIANTS: tuple[str, ...] = ("sequential", "random_order", "other_init", "anneal")
_REFERENCE = "sequential"


def protocol_seed_path(out_dir: Path, seed: int) -> Path:
    return Path(out_dir) / PROTOCOL_DIRNAME / f"seed_{seed:08d}.npz"


def _overlaps(u_a: np.ndarray, u_b: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """``<u_a . u_b>`` and ``<(u_a . u_b)^2>`` over the spins selected by ``mask``."""
    dots = np.einsum("ij,ij->i", u_a[mask], u_b[mask])
    return float(np.mean(dots)), float(np.mean(dots**2))


def protocol_realization(config: SimulationConfig, seed: int) -> dict[str, np.ndarray]:
    """Relax one realization with all four preparations and collect the comparison record."""
    t0 = time.perf_counter()
    positions, spins0, magnitudes = generate_initial_state(config, seed)
    box = config.box if config.boundary == "periodic" else None
    nlist = build_neighbor_list(positions, config.r_cut, box)

    base = config.replace(
        protocol="quench", update_order="sequential", snapshots=(), nn_diagnostics=False
    )
    rng = np.random.default_rng((int(seed), 0xA17))
    other = rng.normal(size=spins0.shape)
    other = magnitudes[:, None] * other / np.linalg.norm(other, axis=1, keepdims=True)

    results = {
        "sequential": relax(spins0, magnitudes, nlist, base, seed),
        "random_order": relax(spins0, magnitudes, nlist, base.replace(update_order="random"), seed),
        "other_init": relax(other, magnitudes, nlist, base, seed),
        "anneal": relax(spins0, magnitudes, nlist, base.replace(protocol="anneal"), seed),
    }

    unit_ref, smag = unit_vectors(results[_REFERENCE].spins)
    nonzero = smag > 0.0
    record: dict[str, np.ndarray] = {"seed": np.array(seed)}
    for name, res in results.items():
        stats = pair_statistics(
            positions, res.spins, nlist.tree, box, config.r_nn, config.x_edges, extras=True
        )
        for key, value in stats.items():
            record[f"{name}_{key}"] = value
        record[f"{name}_energy"] = np.array(res.energy[-1])
        record[f"{name}_dmax"] = np.array(res.dmax[-1] if len(res.dmax) > 1 else np.nan)
        unit, _ = unit_vectors(res.spins)
        q1, q2 = _overlaps(unit, unit_ref, nonzero)
        record[f"{name}_q1"] = np.array(q1)
        record[f"{name}_q2"] = np.array(q2)
    record["anneal_acceptance"] = np.array(results["anneal"].anneal_acceptance)
    record["elapsed_s"] = np.array(time.perf_counter() - t0)
    return record


def _worker(task: tuple[SimulationConfig, int, str]) -> tuple[int, float]:
    config, seed, out_dir = task
    record = protocol_realization(config, seed)
    _atomic_savez(protocol_seed_path(Path(out_dir), seed), record)
    return seed, float(record["elapsed_s"])


def pending_protocol_seeds(config: SimulationConfig, out_dir: Path) -> list[int]:
    return [s for s in config.seeds if not protocol_seed_path(out_dir, s).exists()]


def protocol_cost(config: SimulationConfig) -> dict[str, float]:
    """Rough cost of one protocol-check realization (three quenches + annealing + quench)."""
    quench = config.replace(protocol="quench").estimate_cost()
    anneal = config.replace(protocol="anneal").estimate_cost()
    return {
        "seconds": 3.0 * quench["seconds"] + anneal["seconds"],
        "memory_mb": anneal["memory_mb"],
        "edges": anneal["edges"],
    }


def run_protocol_check(
    config: SimulationConfig,
    out_dir: str | Path,
    n_workers: int = 1,
    progress: bool = True,
    overwrite: bool = False,
    memory_limit_mb: float | None = None,
) -> RunInfo:
    """Run (or resume/extend) the protocol check in ``out_dir``; same semantics as ``run_ensemble``."""
    out_dir = Path(out_dir)
    history = prepare_run_dir(out_dir, config, "protocol", (PROTOCOL_DIRNAME,), overwrite)
    todo = pending_protocol_seeds(config, out_dir)
    cost = protocol_cost(config)
    workers = resolve_workers(n_workers, len(todo), cost["memory_mb"], memory_limit_mb)
    write_run_config(out_dir, config, "protocol", history, len(todo), workers)

    def say(message: str) -> None:
        if progress:
            print(message, file=sys.stderr, flush=True)

    total = len(config.seeds)
    if not todo:
        say(f"{out_dir}: all {total} realizations already present.")
        return RunInfo(out_dir, total, 0, 0.0, workers)
    say(
        f"{out_dir}: protocol check, {len(todo)} of {total} realizations "
        f"(N={config.n_particles}, ~{cost['seconds']:.0f} s each, {workers} worker(s))"
    )
    warmup(anneal=True)
    todo_set = set(todo)
    tasks = [(config, s, str(out_dir)) for s in config.seeds if s in todo_set]
    elapsed = execute_tasks(_worker, tasks, workers, say)
    say(f"{out_dir}: finished {len(tasks)} realizations in {_fmt_duration(elapsed)}.")
    return RunInfo(out_dir, total, len(tasks), elapsed, workers)


class ProtocolData:
    """Stacked per-realization records of a protocol check."""

    def __init__(self, config: SimulationConfig, arrays: dict[str, np.ndarray]):
        self.config = config
        self.arrays = arrays

    @classmethod
    def load(cls, run_dir: str | Path) -> ProtocolData:
        run_dir = Path(run_dir)
        config = read_config(run_dir)
        files = [protocol_seed_path(run_dir, s) for s in config.seeds]
        files = [f for f in files if f.exists()]
        if not files:
            raise FileNotFoundError(f"no protocol-check realizations in {run_dir}")
        records = [load_record(f) for f in files]
        return cls(config, {k: np.stack([r[k] for r in records]) for k in records[0]})

    @property
    def n_real(self) -> int:
        return int(self.arrays["seed"].shape[0])

    def x_center(self, rebin: int = 1) -> np.ndarray:
        edges = self.config.x_edges
        if self.config.n_bins % rebin:
            raise ValueError(f"rebin factor {rebin} must divide n_bins = {self.config.n_bins}")
        edges = edges[::rebin]
        return 0.5 * (edges[:-1] + edges[1:])

    def realization_means(self, name: str, obs: str = "C", rebin: int = 1) -> np.ndarray:
        """Per-realization bin means of ``obs`` for the variant ``name`` (NaN in empty bins)."""
        sums = self.arrays[f"{name}_sum_{obs}"]
        counts = self.arrays[f"{name}_cnt_{'Q' if obs == 'Q' else 'C'}"]
        if rebin > 1:
            shape = (sums.shape[0], sums.shape[1] // rebin, rebin)
            sums, counts = sums.reshape(shape).sum(-1), counts.reshape(shape).sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(counts > 0, sums / counts, np.nan)

    def paired_curve_difference(
        self, name: str, obs: str = "C", rebin: int = 1
    ) -> tuple[np.ndarray, np.ndarray]:
        """Mean and standard error of ``obs_name - obs_sequential`` per bin (paired by seed)."""
        diff = self.realization_means(name, obs, rebin) - self.realization_means(
            _REFERENCE, obs, rebin
        )
        mean, sem, _ = mean_sem(diff)
        return mean, sem

    def curves(self, obs: str = "C", rebin: int = 1) -> pd.DataFrame:
        """Ensemble-mean curves of every variant (``mean_*``, ``sem_*``) and paired differences."""
        data: dict[str, Any] = {"x_center": self.x_center(rebin)}
        for name in VARIANTS:
            mean, sem, _ = mean_sem(self.realization_means(name, obs, rebin))
            data[f"mean_{name}"], data[f"sem_{name}"] = mean, sem
            if name != _REFERENCE:
                dm, ds = self.paired_curve_difference(name, obs, rebin)
                data[f"diff_{name}"], data[f"sem_diff_{name}"] = dm, ds
        return pd.DataFrame(data)

    def summary(self, rebin: int = 1) -> pd.DataFrame:
        """One row per variant: energies, overlaps with the production state, curve differences."""
        n = self.config.n_particles
        energy_ref = self.arrays[f"{_REFERENCE}_energy"]
        x = self.x_center(rebin)
        rows = []
        for name in VARIANTS:
            energy = self.arrays[f"{name}_energy"]
            e_mean, e_sem, _ = mean_sem((energy / n)[:, None])
            row: dict[str, Any] = {
                "variant": name,
                "n_real": self.n_real,
                "energy_per_spin": float(e_mean[0]),
                "sem_energy_per_spin": float(e_sem[0]),
            }
            rel = 100.0 * (energy - energy_ref) / np.abs(energy_ref)  # negative = lower energy
            r_mean, r_sem, _ = mean_sem(rel[:, None])
            row["delta_energy_percent"] = float(r_mean[0])
            row["sem_delta_energy_percent"] = float(r_sem[0])
            for key in ("q1", "q2"):
                m, s, _ = mean_sem(self.arrays[f"{name}_{key}"][:, None])
                row[f"overlap_{key}"], row[f"sem_overlap_{key}"] = float(m[0]), float(s[0])
            c_mean = mean_sem(self.realization_means(name, "C", rebin))[0]
            q_mean = mean_sem(self.realization_means(name, "Q", rebin))[0]
            marks = landmark_values(x, c_mean, q_mean)
            row["C_peak"], row["x_peak_C"] = marks["C_peak"], marks["x_peak_C"]
            row["Q_min"], row["x_min_Q"] = marks.get("Q_min", np.nan), marks.get("x_min_Q", np.nan)
            if name != _REFERENCE:
                for obs in ("C", "Q"):
                    dm, ds = self.paired_curve_difference(name, obs, rebin)
                    row[f"max_abs_diff_{obs}"] = float(np.nanmax(np.abs(dm)))
                    with np.errstate(invalid="ignore", divide="ignore"):
                        row[f"max_abs_diff_{obs}_over_sem"] = float(np.nanmax(np.abs(dm) / ds))
            rows.append(row)
        return pd.DataFrame(rows)


def protocol_config_summary(run_dir: str | Path) -> dict[str, Any]:
    """Small dict describing the stored protocol check (for reports)."""
    run_dir = Path(run_dir)
    cfg = read_config(run_dir)
    return {
        "n_particles": cfg.n_particles,
        "r_cut_nn": cfg.r_cut_nn_resolved,
        "n_sweeps": cfg.n_sweeps,
        "anneal": cfg.to_dict()["anneal"],
        "config_file": str(run_dir / CONFIG_FILENAME),
    }
