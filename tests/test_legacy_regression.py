"""Regression against the prototype implementation (open boundaries, no fit).

``tests/data/legacy_reference.json`` holds ensemble statistics produced by the original exploratory
script for two tiny configurations. The package must reproduce them, which also anchors it to the
first published-quality runs (same seeds -> same realizations).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from grspin.analysis import EnsembleData
from grspin.config import SimulationConfig
from grspin.ensemble import run_realization

REFERENCE = json.loads((Path(__file__).parent / "data" / "legacy_reference.json").read_text())


def _nan(values):
    return np.array([np.nan if v is None else v for v in values], dtype=float)


@pytest.mark.parametrize("case", sorted(REFERENCE["cases"]))
def test_matches_prototype_ensemble_statistics(case):
    ref = REFERENCE["cases"][case]
    p = ref["params"]
    cfg = SimulationConfig(
        n_particles=p["N"],
        box_length=p["L"],
        spin_scale=p["S_max"],
        boundary="open",
        n_sweeps=p["iterations"],
        n_bins=p["nbins"],
        x_max=p["x_max"],
        n_realizations=p["n_seeds"],
        base_seed=p["base_seed"],
        extra_observables=False,
    )
    assert cfg.r_cut_nn_resolved == 10.0
    records = [run_realization(cfg, s)[0] for s in cfg.seeds]
    table = EnsembleData.from_records(cfg, records).table()

    cols = ref["columns"]
    assert np.allclose(table["x_center"], _nan(cols["x_center"]), rtol=1e-12)
    for name in ("count_C_total_final", "count_Q_total_final"):
        assert np.array_equal(table[name], _nan(cols[name]).astype(int)), name
    assert np.array_equal(table["n_real_C_final"], _nan(cols["n_seed_C_final"]).astype(int))
    assert np.array_equal(table["n_real_Q_final"], _nan(cols["n_seed_Q_final"]).astype(int))
    for name in (
        "mean_C_initial",
        "sem_C_initial",
        "mean_C_final",
        "sem_C_final",
        "mean_Q_initial",
        "sem_Q_initial",
        "mean_Q_final",
        "sem_Q_final",
    ):
        expected = _nan(cols[name])
        got = table[name].to_numpy()
        assert np.array_equal(np.isnan(got), np.isnan(expected)), f"NaN pattern of {name}"
        assert np.allclose(got, expected, rtol=1e-9, atol=1e-12, equal_nan=True), name

    for record, energy, spin in zip(
        records, ref["energy_history"], ref["total_spin_history"], strict=True
    ):
        assert np.allclose(record["energy"], energy, rtol=1e-9, atol=1e-9)
        assert np.allclose(record["total_spin_hist"], spin, rtol=1e-9)
