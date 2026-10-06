"""Ensemble runner: determinism, resume/extend, safety checks and file layout."""

from __future__ import annotations

import json

import numpy as np
import pytest

from grspin.analysis import EnsembleData
from grspin.config import SimulationConfig
from grspin.ensemble import (
    ConfigMismatchError,
    load_record,
    pending_seeds,
    resolve_workers,
    run_ensemble,
    run_realization,
    seed_path,
    state_path,
)

_VOLATILE = ("elapsed_s", "timing_s")


def _same(a, b):
    assert set(a) == set(b)
    for key in a:
        if key in _VOLATILE:
            continue
        assert np.array_equal(a[key], b[key], equal_nan=True), key


def test_record_is_internally_consistent(small_config):
    cfg = small_config.replace(snapshots=(2, 8))
    rec, state = run_realization(cfg, seed=5, return_state=True)
    n = cfg.n_particles
    assert rec["fin_cnt_C"].sum() <= n * (n - 1) // 2
    assert np.array_equal(rec["init_cnt_C"], rec["fin_cnt_C"])  # same positions, nonzero spins
    assert rec["energy"].shape == (cfg.n_sweeps + 1,) and rec["energy"][0] > rec["energy"][-1]
    assert np.all(np.diff(rec["energy"]) <= 1e-9 * np.abs(rec["energy"][:-1]))
    assert rec["energy_drift"] < 1e-12
    assert np.allclose(
        rec["snap_sum_C"][1], rec["fin_sum_C"]
    )  # snapshot at the last sweep == final
    assert rec["snap_sweeps"].tolist() == [2, 8]
    assert state is not None and state["positions"].shape == (n, 3)
    assert np.allclose(np.linalg.norm(state["spins_final"], axis=1), state["magnitudes"])
    assert np.all((state["positions"] >= 0) & (state["positions"] < cfg.box))


def test_realizations_are_deterministic(small_config):
    a, _ = run_realization(small_config, seed=9)
    b, _ = run_realization(small_config, seed=9)
    _same(a, b)
    c, _ = run_realization(small_config, seed=10)
    assert not np.array_equal(a["fin_sum_C"], c["fin_sum_C"])


def test_results_do_not_depend_on_the_number_of_workers(tmp_path, small_config):
    run_ensemble(small_config, tmp_path / "w1", n_workers=1, progress=False)
    run_ensemble(small_config, tmp_path / "w2", n_workers=2, progress=False)
    a, b = EnsembleData.load(tmp_path / "w1"), EnsembleData.load(tmp_path / "w2")
    assert np.array_equal(a.seeds, b.seeds)
    _same(a.arrays, b.arrays)


def test_resume_recomputes_only_the_missing_realizations(tmp_path, small_config):
    out = tmp_path / "run"
    assert run_ensemble(small_config, out, progress=False).n_computed == 3
    victim = seed_path(out, small_config.seeds[1])
    before = load_record(victim)
    victim.unlink()
    assert pending_seeds(small_config, out) == [small_config.seeds[1]]
    assert run_ensemble(small_config, out, progress=False).n_computed == 1
    _same(before, load_record(victim))
    assert run_ensemble(small_config, out, progress=False).n_computed == 0


def test_extending_an_ensemble_adds_only_new_seeds(tmp_path, small_config):
    out = tmp_path / "run"
    run_ensemble(small_config, out, progress=False)
    info = run_ensemble(small_config.replace(n_realizations=5), out, progress=False)
    assert info.n_computed == 2 and info.n_total == 5
    data = EnsembleData.load(out)
    assert data.n_real == 5 and data.seeds.tolist() == small_config.replace(n_realizations=5).seeds


def test_a_different_configuration_is_refused_unless_overwritten(tmp_path, small_config):
    out = tmp_path / "run"
    run_ensemble(small_config, out, progress=False)
    changed = small_config.replace(n_sweeps=small_config.n_sweeps + 1)
    with pytest.raises(ConfigMismatchError):
        run_ensemble(changed, out, progress=False)
    run_ensemble(changed, out, progress=False, overwrite=True)
    assert EnsembleData.load(out).arrays["energy"].shape[1] == changed.n_sweeps + 1


def test_states_provenance_and_run_history_are_written(tmp_path, small_config):
    cfg = small_config.replace(save_states=2)
    out = tmp_path / "run"
    run_ensemble(cfg, out, progress=False)
    run_ensemble(cfg, out, progress=False)  # second invocation appends to the history
    assert state_path(out, cfg.seeds[0]).exists() and state_path(out, cfg.seeds[1]).exists()
    assert not state_path(out, cfg.seeds[2]).exists()
    state = load_record(state_path(out, cfg.seeds[0]))
    assert set(state) == {"positions", "magnitudes", "spins_initial", "spins_final"}
    meta = json.loads((out / "config.json").read_text())
    assert meta["kind"] == "ensemble" and len(meta["history"]) == 2
    assert {"grspin", "python", "numpy", "numba", "git_commit"} <= set(meta["provenance"])
    assert SimulationConfig.from_dict(meta["config"]) == cfg


def test_stale_partial_files_of_an_interrupted_run_are_removed(tmp_path, small_config):
    out = tmp_path / "run"
    (out / "seeds").mkdir(parents=True)
    stale = out / "seeds" / "seed_00000005.npz.part"
    stale.write_bytes(b"junk")
    run_ensemble(small_config, out, progress=False)
    assert not list(out.rglob("*.part"))


def test_worker_count_respects_tasks_and_memory():
    assert resolve_workers(8, 3) == 3
    assert resolve_workers(8, 100, memory_per_task_mb=1000, memory_limit_mb=2500) == 2
    assert resolve_workers(8, 100, memory_per_task_mb=1000, memory_limit_mb=100) == 1
    assert resolve_workers(-1, 1000) >= 1


def test_memory_limit_caps_the_workers_of_a_run(tmp_path, small_config):
    info = run_ensemble(
        small_config, tmp_path / "run", n_workers=4, progress=False, memory_limit_mb=1.0
    )
    assert info.n_workers == 1
