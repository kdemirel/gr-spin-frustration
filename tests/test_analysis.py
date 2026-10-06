"""Statistics over realizations: re-binning, bootstrap, landmarks, convergence, NN tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_ensemble

from grspin import theory
from grspin.analysis import (
    EnsembleData,
    bootstrap_means,
    landmark_table,
    landmark_values,
    mean_sem,
    paired_difference,
)
from grspin.config import RNN_UNIT_DENSITY, SimulationConfig
from grspin.ensemble import run_ensemble, run_realization


def test_mean_sem_matches_numpy_and_handles_nans():
    rng = np.random.default_rng(0)
    values = rng.normal(size=(12, 5))
    values[:, 1] = np.nan  # empty bin
    values[3:, 2] = np.nan  # a single valid realization
    values[:7, 3] = np.nan
    mean, sem, n = mean_sem(values)
    assert np.isnan(mean[1]) and np.isnan(sem[1]) and n[1] == 0
    assert n[2] == 3 and mean[2] == pytest.approx(np.mean(values[:3, 2]))
    assert sem[2] == pytest.approx(np.std(values[:3, 2], ddof=1) / np.sqrt(3))
    assert n[3] == 5 and sem[3] == pytest.approx(np.std(values[7:, 3], ddof=1) / np.sqrt(5))
    assert mean[0] == pytest.approx(values[:, 0].mean())
    assert sem[0] == pytest.approx(values[:, 0].std(ddof=1) / np.sqrt(12))
    single = np.array([[1.0, np.nan], [np.nan, np.nan]])
    assert np.isnan(mean_sem(single)[1]).all()  # one value: no error estimate


def test_bootstrap_reproduces_the_standard_error():
    rng = np.random.default_rng(1)
    values = rng.normal(2.0, 3.0, size=(60, 4))
    boot = bootstrap_means(values, 4000, np.random.default_rng(2))
    sem = mean_sem(values)[1]
    assert boot.shape == (4000, 4)
    assert np.allclose(boot.mean(axis=0), values.mean(axis=0), atol=0.05)
    assert np.allclose(boot.std(axis=0, ddof=1), sem, rtol=0.1)


def test_rebinning_is_exact():
    fine = SimulationConfig(n_particles=400, n_sweeps=5, n_bins=20, x_max=4.0, n_realizations=2)
    coarse = fine.replace(n_bins=10)
    rec_f, _ = run_realization(fine, 3)
    rec_c, _ = run_realization(coarse, 3)
    for obs in ("C", "Q", "A", "P", "B"):
        merged = rec_f[f"fin_sum_{obs}"].reshape(10, 2).sum(axis=1)
        assert np.allclose(merged, rec_c[f"fin_sum_{obs}"], atol=1e-10), obs
    assert np.array_equal(rec_f["fin_cnt_C"].reshape(10, 2).sum(1), rec_c["fin_cnt_C"])
    data_f = EnsembleData.from_records(fine, [rec_f, run_realization(fine, 4)[0]])
    data_c = EnsembleData.from_records(coarse, [rec_c, run_realization(coarse, 4)[0]])
    t_f, t_c = data_f.table(rebin=2), data_c.table()
    assert np.allclose(t_f["x_center"], t_c["x_center"])
    for column in ("mean_C_final", "mean_Q_final", "mean_A_final", "count_C_total_final"):
        assert np.allclose(t_f[column], t_c[column], atol=1e-10, equal_nan=True), column
    with pytest.raises(ValueError, match="divide"):
        data_f.table(rebin=3)


def test_table_columns_and_reference_curves():
    data, curves, x = synthetic_ensemble()
    table = data.table()
    for column in (
        "x_center",
        "x_mean",
        "mean_C_final",
        "sem_C_final",
        "mean_Q_initial",
        "mean_B_final",
        "count_C_total_final",
        "n_real_C_final",
        "C_NN_isolated",
        "C_NN_mutual",
        "Q_isotropic",
    ):
        assert column in table.columns, column
    assert np.allclose(table["x_center"], x)
    assert np.allclose(table["x_mean"], x)
    assert np.allclose(table["C_NN_mutual"], -theory.mutual_nn_probability(x))
    assert np.allclose(table["mean_C_final"], curves["C"], atol=4 * table["sem_C_final"].max())
    assert np.allclose(table["mean_C_initial"], 0.0, atol=4 * table["sem_C_initial"].max())
    assert (table["n_real_C_final"] == data.n_real).all()


def test_landmarks_of_analytic_curves():
    x = np.linspace(0.05, 3.95, 40)
    c = -np.cos(np.pi * x / 2.0)  # first zero at x = 1, maximum at x = 2
    q = 0.2 + 0.3 * (x - 1.05) ** 2  # minimum 0.2 at 1.05; crosses 1/3 downward at 1.05 - sqrt(4/9)
    a = 1.0 - 0.5 * x  # crosses 1/2 at x = 1
    marks = landmark_values(x, c, q, a)
    assert marks["x_zero_C"] == pytest.approx(1.0, abs=3e-3)
    assert marks["x_peak_C"] == pytest.approx(2.0, abs=1e-2) and marks["C_peak"] == pytest.approx(
        1.0, abs=1e-3
    )
    assert marks["x_min_Q"] == pytest.approx(1.05, abs=1e-3) and marks["Q_min"] == pytest.approx(
        0.2, abs=1e-4
    )
    assert marks["x_cross_Q_third"] == pytest.approx(1.05 - np.sqrt(0.1333333 / 0.3), abs=2e-3)
    assert marks["x_cross_A_half"] == pytest.approx(1.0, abs=1e-6)
    assert marks["C_first_bin"] == pytest.approx(c[0])
    dipolar = landmark_values(x, np.abs(c) + 0.1, q)  # never negative: no zero crossing, no peak
    assert np.isnan(dipolar["x_zero_C"]) and np.isnan(dipolar["C_peak"])


def test_landmarks_with_bootstrap_errors_on_a_synthetic_ensemble():
    data, curves, x = synthetic_ensemble(n_real=40, noise=0.01)
    marks = data.landmarks(n_boot=100)
    truth = landmark_values(x, curves["C"], curves["Q"], curves["A"])
    for key in ("x_zero_C", "C_peak", "Q_min", "x_cross_Q_third"):
        assert abs(marks[key]["value"] - truth[key]) < 4 * marks[key]["err"] + 0.02, key
        assert marks[key]["err"] > 0 and marks[key]["p16"] < marks[key]["p84"]
    assert marks["_meta"]["n_real"] == 40


def test_histories_and_scalar_summary():
    data, _, _ = synthetic_ensemble()
    h = data.histories()
    assert list(h["sweep"]) == list(range(11))
    assert h["mean_excess_energy"].iloc[-1] == 0.0 and h["mean_excess_energy"].iloc[0] > 0.5
    assert np.isnan(h["mean_max_direction_change"].iloc[0])
    scalars = data.scalar_summary()
    assert scalars["S0_unit_initial"]["mean"] > scalars["S0_unit_final"]["mean"]
    assert scalars["energy_drift"]["mean"] < 1e-12


def test_sweep_convergence_table(tiny_run):
    _, out = tiny_run
    conv = EnsembleData.load(out).sweep_convergence()
    assert conv["sweep"].tolist() == [1, 3, 6, 12]
    assert conv["max_abs_dC"].iloc[-1] == 0.0  # the last snapshot *is* the final state
    assert conv["max_abs_dC"].iloc[0] > conv["max_abs_dC"].iloc[-2] >= 0.0
    assert conv["mean_excess_energy"].iloc[0] > conv["mean_excess_energy"].iloc[-2] >= 0.0


def test_paired_difference_detects_a_changed_coupling_and_a_null_difference(tmp_path):
    base = SimulationConfig(
        n_particles=400, n_sweeps=10, n_bins=10, x_max=3.0, n_realizations=6, base_seed=3
    )
    run_ensemble(base, tmp_path / "gr", progress=False)
    run_ensemble(base.replace(coupling="dipolar"), tmp_path / "dip", progress=False)
    gr, dip = EnsembleData.load(tmp_path / "gr"), EnsembleData.load(tmp_path / "dip")
    same = paired_difference(gr, gr, "C")
    assert np.allclose(same["diff"].dropna(), 0.0) and (same["n_common"].iloc[1:] == 6).all()
    diff = paired_difference(gr, dip, "C")
    assert (
        diff["diff"].iloc[1] > 1.2
    )  # nearest pairs: C = -1 for GR dimers, +1 for the dipolar magnet
    assert diff["abs_diff_over_sem"].iloc[1] > 5
    table = landmark_table({"gr": gr, "dipolar": dip}, n_boot=20)
    assert list(table["run"]) == ["gr", "dipolar"] and "C_peak" in table.columns

    coarse_cfg = base.replace(n_bins=5)
    coarse = EnsembleData.from_records(
        coarse_cfg, [run_realization(coarse_cfg, s)[0] for s in (3, 4)]
    )
    with pytest.raises(ValueError, match="binning"):
        paired_difference(gr, coarse, "C")


def test_nn_tables_match_the_poisson_theory(nn_run):
    cfg, out = nn_run
    data = EnsembleData.load(out)
    fraction = data.scalar_summary()["mutual_nn_particle_fraction"]
    assert fraction["mean"] == pytest.approx(theory.MUTUAL_PARTICLE_FRACTION, abs=0.02)  # 16/27

    nn = data.nn_table(rebin=2, n_boot=20)
    pairs = data.table(rebin=2)["count_C_total_final"].to_numpy(float)
    p = nn["f_mutual_theory"].to_numpy()
    z = (nn["f_mutual"].to_numpy() - p) / np.sqrt(p * (1 - p) / pairs)
    window = ((nn["x_center"] > 0.2) & (nn["x_center"] < 1.4)).to_numpy()
    assert np.mean(z[window] ** 2) < 3.0  # exp(-9 pi rho r^3 / 4), up to correlations between pairs
    assert np.allclose((nn["f_mutual"] + nn["f_oneway"] + nn["f_far"]).dropna(), 1.0)

    ranks = data.nn_rank_table()
    assert ranks["mean_r_over_rnn"].iloc[0] == pytest.approx(
        1.0, abs=0.02
    )  # r_nn is the mean NN distance
    assert (np.diff(ranks["mean_r_over_rnn"]) > 0).all() and len(ranks) == 8

    counts = data.arrays["nn_hist"].sum(axis=0)
    edges = np.linspace(0, 3, 61)[1:]
    cdf_theory = 1 - np.exp(-(4 * np.pi / 3) * RNN_UNIT_DENSITY**3 * edges**3)
    assert np.abs(np.cumsum(counts) / counts.sum() - cdf_theory).max() < 0.02  # Kolmogorov distance
    dist = data.nn_distance_table()
    assert (dist["pdf_measured"] * 0.05).sum() == pytest.approx(1.0, abs=1e-9)

    small = data.small_r_table()
    assert (small["n_pairs"] > 0).sum() >= 6 and small["ratio_to_isolated_sphere"].notna().any()


def test_nn_tables_require_the_diagnostics_flag(tmp_path, small_config):
    run_ensemble(small_config, tmp_path / "run", progress=False)
    data = EnsembleData.load(tmp_path / "run")
    for method in (data.nn_table, data.nn_rank_table, data.nn_distance_table, data.small_r_table):
        with pytest.raises(ValueError, match="nn_diagnostics"):
            method()
    with pytest.raises(ValueError, match="snapshots"):
        data.sweep_convergence()


def test_loading_an_empty_or_partial_run(tmp_path, small_config):
    (tmp_path / "x").mkdir()
    out = tmp_path / "run"
    run_ensemble(small_config.replace(n_realizations=2), out, progress=False)
    # pretend that 3 realizations were planned but only 2 finished
    import json

    meta = json.loads((out / "config.json").read_text())
    meta["config"]["n_realizations"] = 3
    (out / "config.json").write_text(json.dumps(meta))
    with pytest.warns(UserWarning, match="only 2 of 3"):
        assert EnsembleData.load(out).n_real == 2
    assert EnsembleData.load(out, max_realizations=1).n_real == 1
    with pytest.raises(FileNotFoundError):
        EnsembleData.load(tmp_path / "x")


def test_x_axis_uses_the_poisson_mean_nearest_neighbour_distance():
    cfg = SimulationConfig(n_particles=1000, density=2.5)
    assert cfg.r_nn == pytest.approx(RNN_UNIT_DENSITY * 2.5 ** (-1 / 3))
    assert RNN_UNIT_DENSITY == pytest.approx(0.55396, abs=1e-5)
    assert isinstance(synthetic_ensemble()[0].table(), pd.DataFrame)
