"""Protocol check: paired comparison of preparation methods."""

from __future__ import annotations

import numpy as np
import pytest

from grspin.config import AnnealConfig, SimulationConfig
from grspin.ensemble import ConfigMismatchError, run_ensemble
from grspin.protocols import VARIANTS, ProtocolData, protocol_realization, run_protocol_check
from grspin.report import analyze_protocol


@pytest.fixture(scope="module")
def protocol_dir(tmp_path_factory):
    cfg = SimulationConfig(
        n_particles=300,
        n_sweeps=15,
        n_bins=10,
        x_max=4.0,
        n_realizations=4,
        base_seed=11,
        anneal=AnnealConfig(n_temperatures=6, sweeps_per_temperature=3),
    )
    out = tmp_path_factory.mktemp("protocol")
    run_protocol_check(cfg, out, progress=False)
    return cfg, out


def test_every_variant_is_recorded_for_every_realization(protocol_dir):
    cfg, out = protocol_dir
    data = ProtocolData.load(out)
    assert data.n_real == 4
    for name in VARIANTS:
        assert data.arrays[f"{name}_energy"].shape == (4,)
        assert data.arrays[f"{name}_sum_C"].shape == (4, 10)
    assert np.allclose(data.arrays["sequential_q1"], 1.0) and np.allclose(
        data.arrays["sequential_q2"], 1.0
    )
    for name in VARIANTS[1:]:
        q2 = data.arrays[f"{name}_q2"]
        assert np.all((q2 > 0) & (q2 <= 1))


def test_summary_pairs_the_variants_with_the_production_protocol(protocol_dir):
    _, out = protocol_dir
    data = ProtocolData.load(out)
    summary = data.summary()
    assert list(summary["variant"]) == list(VARIANTS)
    ref = summary.iloc[0]
    assert ref["delta_energy_percent"] == 0.0 and ref["overlap_q2"] == pytest.approx(1.0)
    assert (summary["energy_per_spin"] < 0).all()
    assert np.all(
        np.abs(summary["delta_energy_percent"]) < 5.0
    )  # all protocols reach similar energies
    assert "max_abs_diff_C_over_sem" in summary.columns and np.isnan(
        ref.get("max_abs_diff_C", np.nan)
    )
    curves = data.curves("C")
    assert {"mean_anneal", "sem_anneal", "diff_anneal", "sem_diff_anneal"} <= set(curves.columns)
    assert np.allclose(data.curves("Q", rebin=2)["x_center"], data.x_center(2))


def test_realization_is_deterministic_and_uses_identical_geometry(protocol_dir):
    cfg, _ = protocol_dir
    a, b = protocol_realization(cfg, 11), protocol_realization(cfg, 11)
    for key in a:
        if key != "elapsed_s":
            assert np.array_equal(a[key], b[key], equal_nan=True), key
    # same positions for all variants -> identical pair counts per bin
    for name in VARIANTS[1:]:
        assert np.array_equal(a[f"{name}_cnt_C"], a["sequential_cnt_C"])


def test_resume_extend_and_kind_checks(protocol_dir, tmp_path):
    cfg, out = protocol_dir
    assert run_protocol_check(cfg, out, progress=False).n_computed == 0
    info = run_protocol_check(cfg.replace(n_realizations=5), out, progress=False)
    assert info.n_computed == 1 and ProtocolData.load(out).n_real == 5
    with pytest.raises(ConfigMismatchError):
        run_ensemble(cfg, out, progress=False)  # a protocol directory is not an ensemble directory


def test_analysis_writes_tables_and_figure(protocol_dir, tmp_path):
    _, out = protocol_dir
    written = analyze_protocol(out, tmp_path / "an")
    for key in ("protocol_summary", "protocol_curves_C", "protocol_curves_Q", "protocol.pdf"):
        assert written[key].exists() and written[key].stat().st_size > 0
