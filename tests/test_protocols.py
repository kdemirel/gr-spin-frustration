"""Protocol check: paired comparison of preparation methods."""

from __future__ import annotations

import shutil

import numpy as np
import pandas as pd
import pytest

from grspin.config import DIMER_BINDING_ENERGY, AnnealConfig, SimulationConfig
from grspin.ensemble import ConfigMismatchError, load_record, run_ensemble, run_realization
from grspin.initial import generate_initial_state
from grspin.neighbors import build_neighbor_list
from grspin.nn import mutual_nn_partners
from grspin.observables import mutual_pairs, pair_energies, unit_vectors
from grspin.protocols import (
    SPIN_CLASSES,
    VARIANTS,
    ProtocolData,
    protocol_realization,
    protocol_seed_path,
    run_protocol_check,
)
from grspin.relax import anneal_schedule, relax
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
    summary = pd.read_csv(written["protocol_summary"]).set_index("variant")
    assert 0 < summary.loc["anneal", "anneal_acceptance_final"] < 1
    assert np.isnan(summary.loc["sequential", "anneal_acceptance_first"])
    text = written["protocol_summary_md"].read_text()
    for needle in (
        "# Protocol check",
        "dimer spins (members of a mutual nearest-neighbour pair)",
        "dE / |E_nondimer,seq| (%)",
        "## Overlaps with the production state",
        "q1 dimer | q2 dimer | q1 interstitial | q2 interstitial",
        "| anneal |",
        "acceptance first / final",
    ):
        assert needle in text, needle
    assert "n/a" not in text.split("## Overlaps")[1].split("## Correlation")[0]


# --------------------------------------------------------------------------------------
# Dimer / interstitial split and the energy of the non-dimer bonds
# --------------------------------------------------------------------------------------
def test_overlaps_and_energies_are_split_into_dimer_and_interstitial_spins(protocol_dir):
    cfg, out = protocol_dir
    data = ProtocolData.load(out)
    assert data.has_dimer_split and SPIN_CLASSES == ("dimer", "interstitial")
    n = cfg.n_particles
    for cls in SPIN_CLASSES:  # the reference state is identical to itself in both classes
        assert np.allclose(data.arrays[f"sequential_q1_{cls}"], 1.0)
        assert np.allclose(data.arrays[f"sequential_q2_{cls}"], 1.0)
    fraction = data.arrays["n_dimer_spins"] / n
    assert np.all((fraction > 0.4) & (fraction < 0.8))  # Poisson: 16/27
    energy_ref = data.arrays["sequential_energy"]
    nondimer_ref = data.arrays["sequential_energy_nondimer"]
    for name in VARIANTS:
        e, e_d, e_nd = (
            data.arrays[f"{name}_{k}"] for k in ("energy", "energy_dimer", "energy_nondimer")
        )
        assert np.allclose(e_d + e_nd, e) and np.all(e_d < 0.0) and np.all(e_d / e > 0.3)
        for cls in SPIN_CLASSES:
            q2 = data.arrays[f"{name}_q2_{cls}"]
            assert np.all((q2 > 0.0) & (q2 <= 1.0 + 1e-12))
        # the same energy difference is larger relative to the non-dimer energy
        rel_total = (e - energy_ref) / np.abs(energy_ref)
        rel_nondimer = (e - energy_ref) / np.abs(nondimer_ref)
        assert np.all(np.abs(rel_nondimer) >= np.abs(rel_total))
        assert np.all(np.sign(rel_nondimer) == np.sign(rel_total))
    # the stored dimer energy is the sum of the pair energies of the mutual pairs of each state,
    # and the class overlaps are the overlaps restricted to the dimer / interstitial spins
    seed = cfg.seeds[0]
    rec = protocol_realization(cfg, seed)
    positions, spins0, mags = generate_initial_state(cfg, seed)
    nl = build_neighbor_list(positions, cfg.r_cut, cfg.box)
    base = cfg.replace(
        protocol="quench", update_order="sequential", snapshots=(), nn_diagnostics=False
    )
    reference = relax(spins0, mags, nl, base, seed).spins
    other = relax(spins0, mags, nl, base.replace(update_order="random"), seed).spins
    partner = mutual_nn_partners(positions, nl.tree)
    i, j = mutual_pairs(partner)
    assert rec["n_dimer_spins"] == 2 * len(i) == (partner >= 0).sum()
    for name, spins in (("sequential", reference), ("random_order", other)):
        e_pairs = pair_energies(positions, spins, i, j, cfg.box, *cfg.sigma_gamma)[0]
        assert rec[f"{name}_energy_dimer"] == pytest.approx(cfg.G * e_pairs.sum(), rel=1e-10)
        assert rec[f"{name}_energy_nondimer"] == pytest.approx(
            rec[f"{name}_energy"] - cfg.G * e_pairs.sum(), rel=1e-10
        )
    u_ref, smag = unit_vectors(reference)
    u_oth, _ = unit_vectors(other)
    dots = np.einsum("ij,ij->i", u_oth, u_ref)
    in_dimer = partner >= 0
    for cls, mask in (("dimer", in_dimer), ("interstitial", ~in_dimer)):
        sel = mask & (smag > 0.0)
        assert rec[f"random_order_q1_{cls}"] == pytest.approx(dots[sel].mean())
        assert rec[f"random_order_q2_{cls}"] == pytest.approx((dots[sel] ** 2).mean())
    nonzero = smag > 0.0
    assert rec["random_order_q1"] == pytest.approx(dots[nonzero].mean())
    assert rec["n_nonzero_spins"] == nonzero.sum()
    # summary columns
    summary = data.summary().set_index("variant")
    for column in (
        "overlap_q1_dimer",
        "sem_overlap_q1_dimer",
        "overlap_q2_interstitial",
        "sem_overlap_q2_interstitial",
        "delta_energy_nondimer_percent",
        "sem_delta_energy_nondimer_percent",
        "delta_energy_dimer_percent",
        "energy_dimer_per_spin",
        "dimer_energy_share",
        "dimer_spin_fraction",
    ):
        assert column in summary.columns, column
    ref = summary.loc["sequential"]
    assert ref["delta_energy_nondimer_percent"] == 0.0 and ref["delta_energy_dimer_percent"] == 0.0
    assert ref["overlap_q2_dimer"] == pytest.approx(1.0) and ref["overlap_q1_interstitial"] == 1.0
    assert summary["dimer_spin_fraction"].nunique() == 1
    assert summary.loc["sequential", "dimer_spin_fraction"] == pytest.approx(fraction.mean())
    assert (summary["dimer_energy_share"] > 0.3).all() and (summary["dimer_energy_share"] < 1).all()
    assert np.allclose(
        summary["energy_dimer_per_spin"] / summary["energy_per_spin"],
        [
            np.mean(data.arrays[f"{v}_energy_dimer"]) / np.mean(data.arrays[f"{v}_energy"])
            for v in VARIANTS
        ],
        rtol=0.2,
    )
    assert np.isfinite(summary.loc["anneal", "delta_energy_nondimer_percent"])


def test_files_without_the_dimer_split_still_load_and_summarize(protocol_dir, tmp_path):
    cfg, out = protocol_dir
    old = tmp_path / "old"
    shutil.copytree(out, old)
    for path in sorted((old / "protocol").glob("seed_*.npz")):
        rec = load_record(path)
        rec = {
            k: v
            for k, v in rec.items()
            if not k.endswith(("_dimer", "_interstitial", "_nondimer"))
            and k not in ("n_dimer_spins", "n_nonzero_spins")
        }
        np.savez(path, **rec)
    data = ProtocolData.load(old)
    assert not data.has_dimer_split and data.n_real >= 4
    summary = data.summary()
    assert list(summary["variant"]) == list(VARIANTS)
    for column in (
        "overlap_q1_dimer",
        "overlap_q2_interstitial",
        "delta_energy_nondimer_percent",
        "delta_energy_dimer_percent",
        "dimer_energy_share",
        "dimer_spin_fraction",
    ):
        assert summary[column].isna().all(), column
    assert np.isfinite(summary["delta_energy_percent"]).all()
    assert np.isfinite(summary["overlap_q2"]).all()
    written = analyze_protocol(old, tmp_path / "an", plots=False)
    text = written["protocol_summary_md"].read_text()
    assert "predate the dimer / interstitial split" in text and "| anneal |" in text
    assert "## Overlaps with the production state" in text


# --------------------------------------------------------------------------------------
# Annealing schedule and acceptance
# --------------------------------------------------------------------------------------
def test_annealing_starts_above_the_dimer_binding_energy_and_is_geometric():
    default = AnnealConfig()
    assert default.t_start == 10.0 > DIMER_BINDING_ENERGY == pytest.approx(2.94, abs=0.01)
    assert default.t_end == 0.01 and default.n_temperatures == 40
    cfg = SimulationConfig(
        n_particles=1000,
        spin_scale=2.0,
        density=0.5,
        anneal=AnnealConfig(t_start=10.0, t_end=0.01, n_temperatures=4),
    )
    scale = 2.0**2 * cfg.density_effective
    assert np.allclose(anneal_schedule(cfg), np.array([10.0, 1.0, 0.1, 0.01]) * scale)


def test_acceptance_is_recorded_at_every_temperature(protocol_dir):
    cfg, out = protocol_dir
    data = ProtocolData.load(out)  # (an earlier test may have extended the check)
    per_t = data.anneal_acceptance_per_T
    assert per_t.shape == (data.n_real, 6)  # realizations x temperatures, hot to cold
    assert np.all((per_t > 0) & (per_t <= 1))
    assert np.allclose(per_t[:, -1], data.anneal_acceptance_final)
    assert np.all(per_t[:, 0] > per_t[:, -1]) and per_t[:, 0].mean() > 0.8  # hot: nearly all
    assert "anneal_acceptance" not in data.arrays  # the 0.1 name is gone from new files
    table = data.acceptance_table()
    assert list(table.columns) == ["temperature", "acceptance", "sem_acceptance", "n_real"]
    assert len(table) == 6 and table["temperature"].iloc[0] == cfg.anneal.t_start
    assert table["temperature"].iloc[-1] == pytest.approx(cfg.anneal.t_end)
    assert np.allclose(table["acceptance"], per_t.mean(axis=0))
    assert (table["n_real"] == data.n_real).all()
    row = data.summary().set_index("variant").loc["anneal"]
    assert row["anneal_acceptance_final"] == pytest.approx(data.anneal_acceptance_final.mean())
    assert row["anneal_acceptance_first"] == pytest.approx(per_t[:, 0].mean())


def test_annealed_ensemble_records_both_acceptance_keys():
    cfg = SimulationConfig(
        n_particles=150,
        n_sweeps=3,
        n_bins=4,
        x_max=1.0,
        n_realizations=1,
        protocol="anneal",
        anneal=AnnealConfig(n_temperatures=3, sweeps_per_temperature=1),
    )
    rec, _ = run_realization(cfg, 1)
    assert rec["anneal_acceptance_per_T"].shape == (3,)
    assert rec["anneal_acceptance_final"] == rec["anneal_acceptance_per_T"][-1]
    assert "anneal_acceptance" not in rec


def test_files_of_version_0_1_with_the_old_acceptance_key_still_load(protocol_dir, tmp_path):
    cfg, out = protocol_dir
    old = tmp_path / "old"
    shutil.copytree(out, old)
    files = sorted((old / "protocol").glob("seed_*.npz"))  # all realizations present on disk
    for path in files:
        rec = load_record(path)
        rec["anneal_acceptance"] = rec.pop("anneal_acceptance_final")
        del rec["anneal_acceptance_per_T"]
        np.savez(path, **rec)
    data = ProtocolData.load(old)
    assert data.anneal_acceptance_per_T is None and data.n_real == len(files)
    assert np.allclose(data.anneal_acceptance_final, ProtocolData.load(out).anneal_acceptance_final)
    table = data.acceptance_table()
    assert len(table) == 1 and table["temperature"].iloc[0] == pytest.approx(cfg.anneal.t_end)
    row = data.summary().set_index("variant").loc["anneal"]
    assert np.isnan(row["anneal_acceptance_first"]) and row["anneal_acceptance_final"] > 0
    # a check extended after the rename mixes both kinds of files: it loads, with a warning,
    # and the keys that not all files share are dropped
    shutil.copy(protocol_seed_path(out, cfg.seeds[0]), protocol_seed_path(old, cfg.seeds[0]))
    with pytest.warns(UserWarning, match="ignored"):
        mixed = ProtocolData.load(old)
    assert mixed.n_real == len(files) and mixed.anneal_acceptance_final is None
    assert mixed.acceptance_table().empty and len(mixed.summary()) == len(VARIANTS)
