"""Ensemble runner: independent realizations, one ``.npz`` file each, resumable and extendable.

Layout of an output directory::

    <out_dir>/
        config.json          configuration, provenance and run history
        seeds/seed_XXXXXXXX.npz   per-realization binned sums/counts, histories, diagnostics
        states/seed_XXXXXXXX.npz  (optional) positions and spins of the first ``save_states`` seeds

Re-running the same command skips finished realizations; increasing ``n_realizations`` extends the
ensemble. Results never depend on the number of workers (each realization is seeded independently).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__
from .config import SimulationConfig
from .initial import generate_initial_state
from .neighbors import build_neighbor_list
from .nn import nn_statistics
from .observables import pair_statistics, spin_summary
from .relax import relax

CONFIG_FILENAME = "config.json"
SEEDS_DIRNAME = "seeds"
STATES_DIRNAME = "states"


class ConfigMismatchError(RuntimeError):
    """The output directory holds results computed with a different configuration."""


# --------------------------------------------------------------------------------------
# One realization
# --------------------------------------------------------------------------------------
def run_realization(
    config: SimulationConfig, seed: int, return_state: bool = False
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray] | None]:
    """Generate, relax and analyse one realization.

    Returns ``(record, state)``; ``state`` (positions and spins) is only built if
    ``return_state`` is true.
    """
    t0 = time.perf_counter()
    positions, spins0, magnitudes = generate_initial_state(config, seed)
    box = config.box if config.boundary == "periodic" else None
    nlist = build_neighbor_list(positions, config.r_cut, box)
    t1 = time.perf_counter()
    result = relax(spins0, magnitudes, nlist, config, seed)
    t2 = time.perf_counter()

    r_nn, x_edges, extras = config.r_nn, config.x_edges, config.extra_observables

    def stats(spins: np.ndarray) -> dict[str, np.ndarray]:
        return pair_statistics(positions, spins, nlist.tree, box, r_nn, x_edges, extras)

    record: dict[str, np.ndarray] = {"seed": np.array(seed)}
    for key, value in stats(spins0).items():
        record[f"init_{key}"] = value
    for key, value in stats(result.spins).items():
        record[f"fin_{key}"] = value

    if result.snapshots:
        sweeps = sorted(result.snapshots)
        per_snapshot = [stats(result.snapshots[t]) for t in sweeps]
        record["snap_sweeps"] = np.array(sweeps)
        for key in per_snapshot[0]:
            record[f"snap_{key}"] = np.stack([s[key] for s in per_snapshot])

    record["energy"] = result.energy
    record["dmax"] = result.dmax
    record["total_spin_hist"] = result.total_spin
    record["energy_drift"] = np.array(result.energy_drift)
    for prefix, spins in (("init", spins0), ("fin", result.spins)):
        for key, value in spin_summary(spins).items():
            record[f"{prefix}_{key}"] = np.array(value)
    if result.anneal_acceptance is not None:
        record["anneal_acceptance"] = np.array(result.anneal_acceptance)

    if config.nn_diagnostics:
        record.update(nn_statistics(positions, result.spins, nlist.tree, box, r_nn, x_edges))

    t3 = time.perf_counter()
    record["timing_s"] = np.array([t1 - t0, t2 - t1, t3 - t2])
    record["elapsed_s"] = np.array(t3 - t0)

    state = None
    if return_state:
        state = {
            "positions": positions,
            "magnitudes": magnitudes,
            "spins_initial": spins0,
            "spins_final": result.spins,
        }
    return record, state


# --------------------------------------------------------------------------------------
# Files
# --------------------------------------------------------------------------------------
def seed_path(out_dir: Path, seed: int) -> Path:
    return Path(out_dir) / SEEDS_DIRNAME / f"seed_{seed:08d}.npz"


def state_path(out_dir: Path, seed: int) -> Path:
    return Path(out_dir) / STATES_DIRNAME / f"seed_{seed:08d}.npz"


def _atomic_savez(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(tmp, path)


def load_record(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: data[key] for key in data.files}


def read_config(out_dir: Path) -> SimulationConfig:
    """Configuration stored in ``<out_dir>/config.json``."""
    with open(Path(out_dir) / CONFIG_FILENAME) as fh:
        return SimulationConfig.from_dict(json.load(fh)["config"])


def provenance() -> dict[str, Any]:
    """Software versions, platform and (if available) the git commit of the code."""
    import numba
    import scipy

    info: dict[str, Any] = {
        "grspin": __version__,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "numba": numba.__version__,
        "platform": platform.platform(),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": None,
        "git_dirty": None,
    }
    here = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=5
        )
        if commit.returncode == 0:
            info["git_commit"] = commit.stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain", "--", str(here)],
                cwd=here,
                capture_output=True,
                text=True,
                timeout=5,
            )
            info["git_dirty"] = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return info


def pending_seeds(config: SimulationConfig, out_dir: Path) -> list[int]:
    """Seeds whose realization (or requested state) file is still missing."""
    pending = []
    for index, seed in enumerate(config.seeds):
        missing = not seed_path(out_dir, seed).exists()
        if index < config.save_states and not state_path(out_dir, seed).exists():
            missing = True
        if missing:
            pending.append(seed)
    return pending


# --------------------------------------------------------------------------------------
# Runner (shared with the protocol check in ``grspin.protocols``)
# --------------------------------------------------------------------------------------
@dataclass
class RunInfo:
    out_dir: Path
    n_total: int
    n_computed: int
    elapsed_s: float
    n_workers: int = 1


def available_memory_mb() -> float | None:
    """Physical memory in MB, or ``None`` if it cannot be determined on this platform."""
    try:
        return float(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e6)
    except (AttributeError, ValueError, OSError):
        return None


def resolve_workers(
    n_workers: int,
    n_tasks: int,
    memory_per_task_mb: float | None = None,
    memory_limit_mb: float | None = None,
) -> int:
    """Number of worker processes: requested (``-1``: all cores), capped by tasks and memory."""
    n = (os.cpu_count() or 1) if n_workers < 0 else max(1, int(n_workers))
    n = min(n, max(1, n_tasks))
    if memory_limit_mb is not None and memory_per_task_mb:
        n = min(n, max(1, int(memory_limit_mb // memory_per_task_mb)))
    return n


def prepare_run_dir(
    out_dir: Path,
    config: SimulationConfig,
    kind: str,
    subdirs: tuple[str, ...],
    overwrite: bool,
) -> list[dict[str, Any]]:
    """Create ``out_dir``, validate a pre-existing ``config.json`` and return the run history.

    Raises :class:`ConfigMismatchError` if the directory holds results of a different kind or
    configuration (unless ``overwrite``, which deletes the old realization files).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = out_dir / CONFIG_FILENAME
    history: list[dict[str, Any]] = []
    if cfg_path.exists():
        with open(cfg_path) as fh:
            saved = json.load(fh)
        if overwrite:
            for sub in (SEEDS_DIRNAME, STATES_DIRNAME, *subdirs):
                for f in (out_dir / sub).glob("seed_*.npz"):
                    f.unlink()
        else:
            if saved.get("kind", "ensemble") != kind:
                raise ConfigMismatchError(
                    f"{out_dir} holds a '{saved.get('kind', 'ensemble')}' run, not a '{kind}' run."
                )
            old = SimulationConfig.from_dict(saved["config"])
            if old.compat_key() != config.compat_key():
                raise ConfigMismatchError(
                    f"{out_dir} contains results for a different configuration; use another "
                    "output directory or pass overwrite=True (--overwrite)."
                )
            history = list(saved.get("history", []))
    for sub in {SEEDS_DIRNAME, STATES_DIRNAME, *subdirs}:
        for stale in (out_dir / sub).glob("*.part"):  # left over by an interrupted run
            stale.unlink()
    return history


def write_run_config(
    out_dir: Path,
    config: SimulationConfig,
    kind: str,
    history: list[dict[str, Any]],
    n_pending: int,
    n_workers: int,
) -> None:
    prov = provenance()
    history.append(
        {
            "started_utc": prov["created_utc"],
            "git_commit": prov["git_commit"],
            "n_realizations": config.n_realizations,
            "n_pending": n_pending,
            "workers": n_workers,
        }
    )
    with open(out_dir / CONFIG_FILENAME, "w") as fh:
        json.dump(
            {"kind": kind, "config": config.to_dict(), "provenance": prov, "history": history},
            fh,
            indent=2,
        )


def _fmt_duration(seconds: float) -> str:
    seconds = max(0.0, seconds)
    if seconds < 90:
        return f"{seconds:.0f}s"
    if seconds < 5400:
        return f"{seconds / 60:.1f}min"
    return f"{seconds / 3600:.1f}h"


def execute_tasks(
    worker: Callable[[Any], tuple[int, float]],
    tasks: list[Any],
    n_workers: int,
    say: Callable[[str], None],
) -> float:
    """Run ``worker(task) -> (seed, seconds)`` over ``tasks`` (in processes if ``n_workers > 1``).

    Returns the wall-clock time. On an exception or Ctrl-C pending tasks are cancelled; finished
    realizations stay on disk, so the run can be resumed.
    """
    t_start = time.perf_counter()
    done = 0

    def report(seed: int, secs: float) -> None:
        nonlocal done
        done += 1
        elapsed = time.perf_counter() - t_start
        eta = elapsed / done * (len(tasks) - done)
        say(
            f"  [{done}/{len(tasks)}] seed {seed} finished in {secs:.1f}s "
            f"(elapsed {_fmt_duration(elapsed)}, ETA {_fmt_duration(eta)})"
        )

    if n_workers <= 1:
        for task in tasks:
            report(*worker(task))
    else:
        pool = ProcessPoolExecutor(max_workers=n_workers)
        try:
            futures = [pool.submit(worker, task) for task in tasks]
            for future in as_completed(futures):
                report(*future.result())
        except BaseException:
            pool.shutdown(wait=False, cancel_futures=True)
            raise
        pool.shutdown()
    return time.perf_counter() - t_start


def _worker(task: tuple[SimulationConfig, int, str, bool]) -> tuple[int, float]:
    config, seed, out_dir, with_state = task
    record, state = run_realization(config, seed, return_state=with_state)
    _atomic_savez(seed_path(Path(out_dir), seed), record)
    if state is not None:
        _atomic_savez(state_path(Path(out_dir), seed), state)
    return seed, float(record["elapsed_s"])


def warmup(anneal: bool = False) -> None:
    """Compile (or load from the on-disk cache) the numba kernels once, before starting workers."""
    tiny = SimulationConfig(
        n_particles=40,
        boundary="open",
        n_sweeps=1,
        n_bins=4,
        x_max=1.0,
        n_realizations=1,
        protocol="anneal" if anneal else "quench",
    )
    run_realization(tiny, seed=0)


def run_ensemble(
    config: SimulationConfig,
    out_dir: str | Path,
    n_workers: int = 1,
    progress: bool = True,
    overwrite: bool = False,
    memory_limit_mb: float | None = None,
) -> RunInfo:
    """Run (or resume/extend) the ensemble described by ``config`` inside ``out_dir``.

    ``n_workers`` processes are used (``-1``: all cores), reduced if the estimated memory of that
    many concurrent realizations would exceed ``memory_limit_mb``. Every realization is written to
    disk as soon as it finishes, so an interrupted run can simply be started again.
    """
    out_dir = Path(out_dir)
    history = prepare_run_dir(out_dir, config, "ensemble", (), overwrite)

    todo = pending_seeds(config, out_dir)
    cost = config.estimate_cost()
    workers = resolve_workers(n_workers, len(todo), cost["memory_mb"], memory_limit_mb)
    write_run_config(out_dir, config, "ensemble", history, len(todo), workers)

    def say(message: str) -> None:
        if progress:
            print(message, file=sys.stderr, flush=True)

    total = len(config.seeds)
    if not todo:
        say(f"{out_dir}: all {total} realizations already present.")
        return RunInfo(out_dir, total, 0, 0.0, workers)

    capped = (
        " (reduced by the memory limit)"
        if workers < min((os.cpu_count() or 1) if n_workers < 0 else n_workers, len(todo))
        else ""
    )
    say(
        f"{out_dir}: {len(todo)} of {total} realizations to run "
        f"(N={config.n_particles}, boundary={config.boundary}, r_cut={config.r_cut_nn_resolved:.3g} r_nn, "
        f"{config.n_sweeps} sweeps, ~{cost['seconds']:.1f} s and ~{cost['memory_mb']:.0f} MB each, "
        f"{workers} worker{'s' if workers != 1 else ''}{capped})"
    )
    warmup(anneal=config.protocol == "anneal")

    todo_set = set(todo)
    tasks = [
        (config, seed, str(out_dir), index < config.save_states)
        for index, seed in enumerate(config.seeds)
        if seed in todo_set
    ]
    elapsed = execute_tasks(_worker, tasks, workers, say)
    say(f"{out_dir}: finished {len(tasks)} realizations in {_fmt_duration(elapsed)}.")
    return RunInfo(out_dir, total, len(tasks), elapsed, workers)
