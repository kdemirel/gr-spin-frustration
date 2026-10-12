"""Ensemble runner: independent realizations, one ``.npz`` file each, resumable and extendable.

Layout of an output directory::

    <out_dir>/
        config.json          configuration, provenance and run history
        seeds/seed_XXXXXXXX.npz   per-realization binned sums/counts, histories, diagnostics
        states/seed_XXXXXXXX.npz  (optional) positions and spins of the first ``save_states`` seeds

Re-running the same command skips finished realizations; increasing ``n_realizations`` extends the
ensemble. Results never depend on the number of workers (each realization is seeded independently).

Provenance: ``config.json`` records the software versions, the git commit, whether the tree was
dirty (then also the diff of the package sources) and a SHA-256 hash of the package sources
(:func:`provenance`); every realization file written by the runner carries the commit and the
source hash as well (:func:`record_stamp`), so that an ensemble extended with other code is
recognisable file by file.
"""

from __future__ import annotations

import hashlib
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
from .nn import mutual_nn_partners, nn_statistics
from .observables import dimer_frame_map, pair_statistics, spin_summary
from .relax import relax

CONFIG_FILENAME = "config.json"
SEEDS_DIRNAME = "seeds"
STATES_DIRNAME = "states"
#: ``git diff`` stored in ``config.json`` is truncated to this many characters.
GIT_DIFF_MAX_CHARS = 20 * 1024
#: Keys of a realization file that identify the code which wrote it (:func:`record_stamp`).
STAMP_KEYS: tuple[str, ...] = ("git_commit", "source_hash")


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
    ``return_state`` is true. With ``config.nn_diagnostics`` the record also holds the
    nearest-neighbour statistics and the dimer-frame correlation map with the dimer-field test
    (``dimer_map_*``, ``dimer_ang_*``, ``dimer_field_*`` keys, computed with the run's
    ``(sigma, gamma)``; see :func:`grspin.observables.dimer_frame_map`).
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
    if result.anneal_acceptance_per_T is not None:
        record["anneal_acceptance_per_T"] = result.anneal_acceptance_per_T
        record["anneal_acceptance_final"] = np.array(result.anneal_acceptance_final)

    if config.nn_diagnostics:
        record.update(nn_statistics(positions, result.spins, nlist.tree, box, r_nn, x_edges))
        partner = mutual_nn_partners(positions, nlist.tree)
        record.update(
            dimer_frame_map(
                positions,
                result.spins,
                nlist.tree,
                box,
                r_nn,
                partner,
                sigma_gamma=config.sigma_gamma,
            )
        )

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


def source_hash() -> str:
    """SHA-256 of the package sources: ``src/grspin/*.py`` concatenated in name order.

    Each file is preceded by a line with its name, so renamed files change the hash too. Unlike
    the git commit, the hash identifies the code even when the working tree is dirty.
    """
    here = Path(__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(here.glob("*.py")):
        digest.update(f"# {path.name}\n".encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _git(args: list[str], cwd: Path) -> str | None:
    """Standard output of ``git <args>`` in ``cwd``, or ``None`` if git fails or is absent."""
    try:
        done = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n... [truncated: {len(text) - limit} more characters]"


def provenance() -> dict[str, Any]:
    """Software versions, platform, the hash of the package sources and the git state of the code.

    ``git_commit`` and ``git_dirty`` are ``None`` outside a git checkout. When the package
    directory is dirty, ``git_status`` holds its porcelain status (modified *and* untracked
    files) and ``git_diff`` the output of ``git diff HEAD -- src/grspin`` truncated to
    :data:`GIT_DIFF_MAX_CHARS`; both are ``None`` for a clean tree. ``source_hash``
    (:func:`source_hash`) identifies the code in either case.
    """
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
        "git_status": None,
        "git_diff": None,
        "source_hash": source_hash(),
    }
    here = Path(__file__).resolve().parent
    commit = _git(["rev-parse", "HEAD"], here)
    if commit is None:
        return info
    info["git_commit"] = commit.strip()
    status = _git(["status", "--porcelain", "--", str(here)], here)
    if status is None:
        return info
    info["git_dirty"] = bool(status.strip())
    if info["git_dirty"]:
        info["git_status"] = status.rstrip()
        info["git_diff"] = _truncate(
            _git(["diff", "HEAD", "--", str(here)], here) or "", GIT_DIFF_MAX_CHARS
        )
    return info


def record_stamp(prov: dict[str, Any]) -> dict[str, np.ndarray]:
    """The provenance fields stored in every realization file (:data:`STAMP_KEYS`)."""
    return {
        "git_commit": np.array(prov.get("git_commit") or ""),
        "source_hash": np.array(prov.get("source_hash") or ""),
    }


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
            if old.compat_key(kind) != config.compat_key(kind):
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
) -> dict[str, Any]:
    """Write ``config.json`` (configuration, provenance, history) and return the provenance.

    Every invocation appends one history entry (start time, commit, dirty flag, source hash,
    pending realizations), so an ensemble extended later with other code is traceable.
    """
    prov = provenance()
    history.append(
        {
            "started_utc": prov["created_utc"],
            "git_commit": prov["git_commit"],
            "git_dirty": prov["git_dirty"],
            "source_hash": prov["source_hash"],
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
    return prov


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


def _worker(
    task: tuple[SimulationConfig, int, str, bool, dict[str, np.ndarray]],
) -> tuple[int, float]:
    config, seed, out_dir, with_state, stamp = task
    record, state = run_realization(config, seed, return_state=with_state)
    record.update(stamp)
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
    stamp = record_stamp(write_run_config(out_dir, config, "ensemble", history, len(todo), workers))

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
        (config, seed, str(out_dir), index < config.save_states, stamp)
        for index, seed in enumerate(config.seeds)
        if seed in todo_set
    ]
    elapsed = execute_tasks(_worker, tasks, workers, say)
    say(f"{out_dir}: finished {len(tasks)} realizations in {_fmt_duration(elapsed)}.")
    return RunInfo(out_dir, total, len(tasks), elapsed, workers)
