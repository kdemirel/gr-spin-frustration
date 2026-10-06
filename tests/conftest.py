"""Shared helpers for the test-suite."""

from __future__ import annotations

import numpy as np
import pytest

from grspin.config import SimulationConfig


@pytest.fixture
def small_config() -> SimulationConfig:
    """A tiny, fast configuration (periodic, N=300)."""
    return SimulationConfig(
        n_particles=300,
        n_sweeps=8,
        n_bins=8,
        x_max=4.0,
        n_realizations=3,
        base_seed=5,
        r_cut_nn=None,
    )


def brute_force_pairs(positions: np.ndarray, box: float | None, r_max: float):
    """All pairs ``i < j`` with ``r <= r_max`` by O(N^2) enumeration (minimum image if ``box``)."""
    n = len(positions)
    i, j = np.triu_indices(n, k=1)
    d = positions[j] - positions[i]
    if box is not None:
        d -= box * np.round(d / box)
    r = np.linalg.norm(d, axis=1)
    keep = r <= r_max
    return i[keep], j[keep], d[keep], r[keep]


# --------------------------------------------------------------------------------------
# Shared (session-scoped) runs: computed once, reused by several test modules
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="session")
def tiny_run(tmp_path_factory):
    """A finished small ensemble with snapshots, nn diagnostics and one stored state."""
    from grspin.ensemble import run_ensemble

    cfg = SimulationConfig(
        n_particles=600,
        n_sweeps=12,
        n_bins=20,
        x_max=4.0,
        n_realizations=6,
        base_seed=21,
        nn_diagnostics=True,
        snapshots=(1, 3, 6, 12),
        save_states=1,
    )
    out = tmp_path_factory.mktemp("tiny_run")
    run_ensemble(cfg, out, n_workers=1, progress=False)
    return cfg, out


@pytest.fixture(scope="session")
def nn_run(tmp_path_factory):
    """Positions-only statistics need no relaxation: a larger system with a single sweep."""
    from grspin.ensemble import run_ensemble

    cfg = SimulationConfig(
        n_particles=3000,
        n_sweeps=1,
        n_bins=40,
        x_max=2.0,
        n_realizations=6,
        base_seed=7,
        nn_diagnostics=True,
    )
    out = tmp_path_factory.mktemp("nn_run")
    run_ensemble(cfg, out, n_workers=1, progress=False)
    return cfg, out


def synthetic_ensemble(n_real=30, n_bins=40, x_max=4.0, seed=0, noise=0.01, params=(2.7, 2.9, 0.1)):
    """An :class:`EnsembleData` built from analytic curves plus noise (no simulation)."""
    from grspin.analysis import EnsembleData
    from grspin.fit import damped_oscillatory, u_of_x

    cfg = SimulationConfig(
        n_particles=5000, n_bins=n_bins, x_max=x_max, n_realizations=n_real, base_seed=1
    )
    rng = np.random.default_rng(seed)
    edges = cfg.x_edges
    x = 0.5 * (edges[:-1] + edges[1:])
    curves = {
        "C": damped_oscillatory(u_of_x(x), *params),
        "Q": 1 / 3 + 0.65 * np.exp(-(x**2) / 0.3) - 0.14 * np.exp(-((x - 1.1) ** 2) / 0.4),
        "A": 1 / 3 + 0.66 * np.exp(-x / 0.5),
        "P": -0.66 * np.exp(-x / 0.6),
    }
    curves["B"] = curves["C"] - 3 * curves["P"]
    counts = np.full(n_bins, 500, dtype=np.int64)
    records = []
    for r in range(n_real):
        rec = {"seed": np.array(1 + r), "fin_sum_x": counts * x, "init_sum_x": counts * x}
        for state, scale in (("fin", 1.0), ("init", 0.0)):
            for obs, truth in curves.items():
                value = scale * truth + noise * rng.normal(size=n_bins)
                rec[f"{state}_sum_{obs}"] = counts * value
            rec[f"{state}_cnt_C"] = counts
            rec[f"{state}_cnt_Q"] = counts
            rec[f"{state}_total_spin"] = np.array(abs(rng.normal(30, 5)))
            rec[f"{state}_s0_unit"] = np.array(abs(rng.normal(0.4 if state == "fin" else 1.0, 0.2)))
        energy = -100.0 * (1.0 - np.exp(-np.arange(11) / 3.0)) + rng.normal(0, 0.5, size=11)
        energy = np.minimum.accumulate(energy)
        rec["energy"] = energy
        rec["dmax"] = np.concatenate([[np.nan], 1.5 * np.exp(-np.arange(1, 11) / 2.0)])
        rec["total_spin_hist"] = np.abs(rng.normal(30, 5, size=11))
        rec["energy_drift"] = np.array(1e-15)
        rec["elapsed_s"] = np.array(1.0)
        records.append(rec)
    return EnsembleData.from_records(cfg, records), curves, x
