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
into paired differences with standard errors over realizations. The annealing stage also records
its Metropolis acceptance rate at every temperature (``anneal_acceptance_per_T``) and at the final
temperature (``anneal_acceptance_final``; files of version 0.1 hold only the latter, under the
name ``anneal_acceptance``).

**Dimer / interstitial split.** The overlaps ``q1 = <u . u'>`` and ``q2 = <(u . u')^2>`` are also
stored separately for the spins that belong to a mutual-nearest-neighbour pair
(``<variant>_q1_dimer``, ``_q2_dimer``) and for the remaining, interstitial spins
(``_q1_interstitial``, ``_q2_interstitial``), and every variant records the energy of its
mutual-NN bonds ``<variant>_energy_dimer`` (sum of the pair energies of the two members with the
run's ``(sigma, gamma)``, with ``G``) and ``<variant>_energy_nondimer = energy - energy_dimer``.
The summary quotes the energy differences relative to the total energy *and* relative to the
energy of the non-dimer bonds, which is all a protocol can change. Files written before these
keys existed load without them (the corresponding summary columns are NaN).
"""

from __future__ import annotations

import sys
import time
import warnings
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
    record_stamp,
    resolve_workers,
    warmup,
    write_run_config,
)
from .initial import generate_initial_state
from .neighbors import build_neighbor_list
from .nn import mutual_nn_partners
from .observables import mutual_pairs, pair_energies, pair_statistics, unit_vectors
from .relax import relax

PROTOCOL_DIRNAME = "protocol"
VARIANTS: tuple[str, ...] = ("sequential", "random_order", "other_init", "anneal")
#: Spin classes of the overlap split: members of a mutual-NN pair and the rest.
SPIN_CLASSES: tuple[str, ...] = ("dimer", "interstitial")
_REFERENCE = "sequential"
_ACCEPTANCE_KEY_V01 = "anneal_acceptance"  # version 0.1 name of the final-temperature acceptance


def protocol_seed_path(out_dir: Path, seed: int) -> Path:
    return Path(out_dir) / PROTOCOL_DIRNAME / f"seed_{seed:08d}.npz"


def _overlaps(u_a: np.ndarray, u_b: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    """``<u_a . u_b>`` and ``<(u_a . u_b)^2>`` over the spins selected by ``mask`` (NaN if none)."""
    if not mask.any():
        return float("nan"), float("nan")
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
    partner = mutual_nn_partners(positions, nlist.tree)
    in_dimer = partner >= 0
    classes = {"dimer": nonzero & in_dimer, "interstitial": nonzero & ~in_dimer}
    pair_i, pair_j = mutual_pairs(partner)
    sigma, gamma = config.sigma_gamma
    record: dict[str, np.ndarray] = {
        "seed": np.array(seed),
        "n_dimer_spins": np.array(int(in_dimer.sum())),
        "n_nonzero_spins": np.array(int(nonzero.sum())),
    }
    for name, res in results.items():
        stats = pair_statistics(
            positions, res.spins, nlist.tree, box, config.r_nn, config.x_edges, extras=True
        )
        for key, value in stats.items():
            record[f"{name}_{key}"] = value
        energy = float(res.energy[-1])
        record[f"{name}_energy"] = np.array(energy)
        record[f"{name}_dmax"] = np.array(res.dmax[-1] if len(res.dmax) > 1 else np.nan)
        unit, _ = unit_vectors(res.spins)
        q1, q2 = _overlaps(unit, unit_ref, nonzero)
        record[f"{name}_q1"] = np.array(q1)
        record[f"{name}_q2"] = np.array(q2)
        for cls, mask in classes.items():
            q1_c, q2_c = _overlaps(unit, unit_ref, mask)
            record[f"{name}_q1_{cls}"] = np.array(q1_c)
            record[f"{name}_q2_{cls}"] = np.array(q2_c)
        e_pairs, _, r_pairs = pair_energies(positions, res.spins, pair_i, pair_j, box, sigma, gamma)
        # only bonds inside the cutoff contribute to the relaxation energy (as in energy_budget)
        e_dimer = float(config.G * e_pairs[r_pairs <= config.r_cut].sum())
        record[f"{name}_energy_dimer"] = np.array(e_dimer)
        record[f"{name}_energy_nondimer"] = np.array(energy - e_dimer)
    annealed = results["anneal"]
    record["anneal_acceptance_per_T"] = annealed.anneal_acceptance_per_T
    record["anneal_acceptance_final"] = np.array(annealed.anneal_acceptance_final)
    record["elapsed_s"] = np.array(time.perf_counter() - t0)
    return record


def _worker(task: tuple[SimulationConfig, int, str, dict[str, np.ndarray]]) -> tuple[int, float]:
    config, seed, out_dir, stamp = task
    record = protocol_realization(config, seed)
    record.update(stamp)
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
    stamp = record_stamp(write_run_config(out_dir, config, "protocol", history, len(todo), workers))

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
    tasks = [(config, s, str(out_dir), stamp) for s in config.seeds if s in todo_set]
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
        """Load all finished realizations; keys missing from some files are dropped with a warning
        (a check extended after the code gained new record keys)."""
        run_dir = Path(run_dir)
        config = read_config(run_dir)
        files = [protocol_seed_path(run_dir, s) for s in config.seeds]
        files = [f for f in files if f.exists()]
        if not files:
            raise FileNotFoundError(f"no protocol-check realizations in {run_dir}")
        records = [load_record(f) for f in files]
        keys = list(dict.fromkeys(key for record in records for key in record))
        common = [key for key in keys if all(key in record for record in records)]
        if len(common) < len(keys):
            dropped = ", ".join(key for key in keys if key not in common)
            warnings.warn(
                f"{run_dir}: keys missing from some files are ignored: {dropped}", stacklevel=2
            )
        return cls(config, {k: np.stack([r[k] for r in records]) for k in common})

    @property
    def n_real(self) -> int:
        return int(self.arrays["seed"].shape[0])

    # -- annealing acceptance -------------------------------------------------------------
    @property
    def anneal_acceptance_final(self) -> np.ndarray | None:
        """Per-realization acceptance rate at the final temperature (``(n_real,)``).

        Files of version 0.1 stored this value under ``anneal_acceptance``; both names are read.
        ``None`` if neither is available (a check whose files mix both versions).
        """
        for key in ("anneal_acceptance_final", _ACCEPTANCE_KEY_V01):
            if key in self.arrays:
                return self.arrays[key]
        return None

    @property
    def anneal_acceptance_per_T(self) -> np.ndarray | None:
        """Per-realization acceptance at every temperature (``(n_real, n_temperatures)``), or
        ``None`` for files of version 0.1."""
        return self.arrays.get("anneal_acceptance_per_T")

    def acceptance_table(self) -> pd.DataFrame:
        """Mean acceptance rate (and SEM over realizations) at every temperature, hot to cold.

        ``temperature`` is in the units of :class:`grspin.config.AnnealConfig` (``G S^2 rho``).
        For files of version 0.1 only the final temperature is available (one row).
        """
        a = self.config.anneal
        temps = np.geomspace(a.t_start, a.t_end, a.n_temperatures)
        per_t = self.anneal_acceptance_per_T
        if per_t is None:
            final = self.anneal_acceptance_final
            temps = temps[-1:] if final is not None else temps[:0]
            per_t = final[:, None] if final is not None else np.empty((self.n_real, 0))
        mean, sem, n = mean_sem(per_t)
        return pd.DataFrame(
            {"temperature": temps, "acceptance": mean, "sem_acceptance": sem, "n_real": n}
        )

    def _acceptance_summary(self) -> dict[str, float]:
        """Mean acceptance at the first and the final temperature for the summary table."""
        per_t, final = self.anneal_acceptance_per_T, self.anneal_acceptance_final
        return {
            "anneal_acceptance_first": float(np.nanmean(per_t[:, 0]))
            if per_t is not None
            else np.nan,
            "anneal_acceptance_final": float(np.nanmean(final)) if final is not None else np.nan,
        }

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

    @property
    def has_dimer_split(self) -> bool:
        """Whether the files hold the dimer / interstitial overlaps and the dimer-bond energies."""
        return all(
            f"{name}_{key}" in self.arrays
            for name in VARIANTS
            for key in ("q1_dimer", "q2_interstitial", "energy_dimer", "energy_nondimer")
        )

    def _scalar(self, key: str) -> tuple[float, float]:
        """Mean and standard error over realizations of a scalar record key (NaN if absent)."""
        if key not in self.arrays:
            return float("nan"), float("nan")
        m, s, _ = mean_sem(np.asarray(self.arrays[key], dtype=float)[:, None])
        return float(m[0]), float(s[0])

    def summary(self, rebin: int = 1) -> pd.DataFrame:
        """One row per variant: energies, overlaps with the production state, curve differences.

        The ``anneal`` row also carries the mean Metropolis acceptance at the first and the
        final temperature of the schedule (NaN for the other variants). With the dimer split
        (files of version 0.3 and later) every row also has the overlaps of the dimer and the
        interstitial spins (``overlap_q1_dimer``, ...), the energy of the mutual-NN bonds per spin
        and its share of the total (``energy_dimer_per_spin``, ``dimer_energy_share``), the
        energy difference to the production state relative to the energy of the non-dimer bonds
        of the production state (``delta_energy_nondimer_percent``: the same ``E - E_seq`` as
        ``delta_energy_percent``, divided by ``|E_nondimer,seq|`` instead of ``|E_seq|``) and the
        relative change of the dimer-bond energy itself (``delta_energy_dimer_percent``); the
        fraction of spins in a dimer is ``dimer_spin_fraction``. Older files give NaN there.
        """
        n = self.config.n_particles
        energy_ref = self.arrays[f"{_REFERENCE}_energy"]
        x = self.x_center(rebin)
        split = self.has_dimer_split
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
            if name == "anneal":
                row.update(self._acceptance_summary())
            rel = 100.0 * (energy - energy_ref) / np.abs(energy_ref)  # negative = lower energy
            r_mean, r_sem, _ = mean_sem(rel[:, None])
            row["delta_energy_percent"] = float(r_mean[0])
            row["sem_delta_energy_percent"] = float(r_sem[0])
            if split:
                e_dimer = self.arrays[f"{name}_energy_dimer"]
                e_nondimer_ref = self.arrays[f"{_REFERENCE}_energy_nondimer"]
                e_dimer_ref = self.arrays[f"{_REFERENCE}_energy_dimer"]
                with np.errstate(invalid="ignore", divide="ignore"):
                    rel_nd = 100.0 * (energy - energy_ref) / np.abs(e_nondimer_ref)
                    rel_d = 100.0 * (e_dimer - e_dimer_ref) / np.abs(e_dimer_ref)
                    share = e_dimer / energy
                for key, values in (
                    ("delta_energy_nondimer_percent", rel_nd),
                    ("delta_energy_dimer_percent", rel_d),
                    ("energy_dimer_per_spin", e_dimer / n),
                    ("dimer_energy_share", share),
                ):
                    m, s, _ = mean_sem(np.asarray(values, dtype=float)[:, None])
                    row[key], row[f"sem_{key}"] = float(m[0]), float(s[0])
                row["dimer_spin_fraction"] = float(np.mean(self.arrays["n_dimer_spins"]) / n)
            else:
                for key in (
                    "delta_energy_nondimer_percent",
                    "delta_energy_dimer_percent",
                    "energy_dimer_per_spin",
                    "dimer_energy_share",
                ):
                    row[key] = row[f"sem_{key}"] = float("nan")
                row["dimer_spin_fraction"] = float("nan")
            for key in ("q1", "q2"):
                m, s, _ = mean_sem(self.arrays[f"{name}_{key}"][:, None])
                row[f"overlap_{key}"], row[f"sem_overlap_{key}"] = float(m[0]), float(s[0])
                for cls in SPIN_CLASSES:
                    m_c, s_c = self._scalar(f"{name}_{key}_{cls}")
                    row[f"overlap_{key}_{cls}"], row[f"sem_overlap_{key}_{cls}"] = m_c, s_c
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
