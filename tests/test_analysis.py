"""Statistics over realizations: re-binning, bootstrap, landmarks, convergence, NN tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_ensemble

from grspin import theory
from grspin.analysis import (
    ENERGY_BUDGET_RATIOS,
    EnsembleData,
    bootstrap_means,
    energy_budget_from_states,
    is_paired,
    landmark_difference,
    landmark_error_budget,
    landmark_table,
    landmark_values,
    mean_sem,
    paired_difference,
    round_to_error,
    same_system,
    summarize_energy_budget,
)
from grspin.config import RNN_UNIT_DENSITY, SimulationConfig
from grspin.ensemble import load_record, run_ensemble, run_realization, seed_path


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


# --------------------------------------------------------------------------------------
# Channels C = P + Cperp
# --------------------------------------------------------------------------------------
def test_cperp_channel_is_derived_from_the_stored_sums(tiny_run):
    data, curves, x = synthetic_ensemble()
    assert "Cperp" in data.available_observables() and data.has_observable("Cperp", "initial")
    cperp = data.realization_means("Cperp")
    assert np.allclose(cperp, data.realization_means("C") - data.realization_means("P"))
    table = data.table()
    for column in (
        "mean_P_initial",
        "sem_P_initial",
        "mean_P_final",
        "sem_P_final",
        "mean_Cperp_initial",
        "sem_Cperp_initial",
        "mean_Cperp_final",
        "sem_Cperp_final",
    ):
        assert column in table.columns, column
    assert np.allclose(table["mean_Cperp_final"], table["mean_C_final"] - table["mean_P_final"])
    assert np.allclose(
        table["mean_Cperp_final"],
        curves["C"] - curves["P"],
        atol=4 * table["sem_Cperp_final"].max(),
    )
    assert np.allclose(table["mean_Cperp_initial"], 0.0, atol=4 * table["sem_Cperp_initial"].max())
    # on real records the identity B = C - 3P = Cperp - 2P holds pair by pair
    _, out = tiny_run
    real = EnsembleData.load(out)
    t = real.table()
    assert np.allclose(
        t["mean_B_final"], t["mean_Cperp_final"] - 2 * t["mean_P_final"], atol=1e-9, equal_nan=True
    )
    without_p = EnsembleData(
        real.config, {k: v for k, v in real.arrays.items() if "sum_P" not in k}
    )
    assert (
        not without_p.has_observable("Cperp") and "Cperp" not in without_p.available_observables()
    )
    assert "mean_Cperp_final" not in without_p.table().columns  # older runs still analyse


def test_landmarks_include_the_transverse_channel():
    x = np.linspace(0.05, 3.95, 40)
    c = -np.cos(np.pi * x / 2.0)
    q = 0.2 + 0.3 * (x - 1.05) ** 2
    p = -0.5 * np.exp(-x)  # longitudinal channel, negative everywhere
    cperp = 0.4 * np.exp(-((x - 1.0) ** 2) / 0.5)  # maximum 0.4 at x = 1
    marks = landmark_values(x, c, q, None, p, cperp)
    assert marks["x_peak_Cperp"] == pytest.approx(1.0, abs=2e-3)
    assert marks["Cperp_peak"] == pytest.approx(0.4, abs=1e-3)
    assert marks["P_at_Cperp_peak"] == pytest.approx(-0.5 * np.exp(-1.0), abs=2e-3)
    assert "x_peak_Cperp" not in landmark_values(x, c, q)
    assert "P_at_Cperp_peak" not in landmark_values(x, c, q, cperp=cperp)
    # the parabola through three bins is exact for a quadratic, also on an unequally spaced abscissa
    xm = x + 0.01 / np.maximum(x, 0.2)
    exact = 0.4 - 0.3 * (xm - 1.0) ** 2
    marks_m = landmark_values(xm, c, q, None, p, exact)
    assert marks_m["x_peak_Cperp"] == pytest.approx(1.0, abs=1e-9)
    assert marks_m["Cperp_peak"] == pytest.approx(0.4, abs=1e-9)


# --------------------------------------------------------------------------------------
# Landmarks on the pair-weighted mean distance, undefined landmarks, convergence table
# --------------------------------------------------------------------------------------
def test_landmarks_use_the_pair_weighted_mean_distance():
    data, curves, x = synthetic_ensemble(n_real=20, noise=0.0)
    shift = 0.01
    data.arrays["fin_sum_x"] = data.arrays["fin_cnt_C"] * (x + shift)
    assert np.allclose(data.x_mean(), x + shift) and np.allclose(data.x_center(), x)
    assert np.allclose(data.table()["x_mean"], x + shift)
    on_mean = data.landmarks(n_boot=10)
    on_center = data.landmarks(n_boot=10, abscissa="x_center")
    assert on_mean["_meta"]["abscissa"] == "x_mean" and on_center["_meta"]["abscissa"] == "x_center"
    assert on_mean["_meta"]["bin_width"] == pytest.approx(0.1) and on_mean["_meta"]["rebin"] == 1
    for key in (
        "x_zero_C",
        "x_peak_C",
        "x_min_Q",
        "x_cross_Q_third",
        "x_cross_A_half",
        "x_peak_Cperp",
    ):
        assert on_mean[key]["value"] - on_center[key]["value"] == pytest.approx(shift, abs=1e-9), (
            key
        )
    for key in ("C_peak", "Q_min", "Cperp_peak", "C_first_bin"):
        assert on_mean[key]["value"] == pytest.approx(on_center[key]["value"], abs=1e-12), key
    assert data.landmarks(rebin=2, n_boot=5)["_meta"]["bin_width"] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="abscissa"):
        data.abscissa(1, "bogus")
    # runs without stored sum_x fall back to the bin centres (and say so)
    del data.arrays["fin_sum_x"]
    fallback = data.landmarks(n_boot=10)
    assert fallback["_meta"]["abscissa"] == "x_center"
    assert fallback["x_zero_C"]["value"] == on_center["x_zero_C"]["value"]


def test_landmark_errors_are_nan_when_the_value_is_undefined():
    data, _, _ = synthetic_ensemble(n_real=12, noise=0.01)
    # a dipolar-like run: C never negative, hence no zero crossing and no positive peak
    data.arrays["fin_sum_C"] = np.abs(data.arrays["fin_sum_C"]) + 0.1 * data.arrays["fin_cnt_C"]
    marks = data.landmarks(n_boot=20)
    for key in ("x_zero_C", "x_peak_C", "C_peak"):
        assert np.isnan(marks[key]["value"]), key
        assert np.isnan(marks[key]["err"]) and np.isnan(marks[key]["p16"]), key
    assert marks["Cperp_peak"]["err"] > 0 and marks["Q_min"]["err"] > 0


def test_sweep_convergence_uses_the_parabolic_landmarks(tiny_run):
    _, out = tiny_run
    data = EnsembleData.load(out)
    conv = data.sweep_convergence()
    final = data.landmarks(n_boot=5)
    for key in ("C_peak", "x_zero_C", "x_peak_C", "Q_min"):
        assert conv[key].iloc[-1] == pytest.approx(final[key]["value"], nan_ok=True), key
    assert conv["median_excess_energy"].iloc[-1] == 0.0
    assert conv["median_excess_energy"].iloc[0] > 0.0


def test_nn_table_errors_are_nan_for_single_pair_categories(tiny_run):
    _, out = tiny_run
    data = EnsembleData.load(out)
    cnt, sums = data.arrays["nn_cnt_mutual"], data.arrays["nn_sum_mutual"]
    # bin 0: a single mutual pair in a single realization and nothing else
    cnt[:, 0], sums[:, 0] = 0, 0.0
    cnt[0, 0], sums[0, 0] = 1, -0.9
    for name in ("oneway", "far"):
        data.arrays[f"nn_cnt_{name}"][:, 0] = 0
        data.arrays[f"nn_sum_{name}"][:, 0] = 0.0
    data.arrays["fin_cnt_C"][:, 0] = cnt[:, 0]
    data.arrays["fin_sum_C"][:, 0] = sums[:, 0]
    nn = data.nn_table(n_boot=30)
    assert nn["C_mutual"].iloc[0] == pytest.approx(-0.9) and nn["C_all"].iloc[0] == pytest.approx(
        -0.9
    )
    assert np.isnan(nn["sem_C_mutual"].iloc[0]) and np.isnan(nn["sem_C_all"].iloc[0])
    assert np.isnan(nn["sem_C_oneway"].iloc[0])  # no pair at all
    fed = (cnt.sum(axis=0) >= 2) & ((cnt > 0).sum(axis=0) >= 2)
    k = int(np.argmax(cnt.sum(axis=0) * fed))
    assert np.isfinite(nn["sem_C_mutual"].iloc[k]) and nn["sem_C_mutual"].iloc[k] > 0


# --------------------------------------------------------------------------------------
# Energies, paired versus unpaired comparisons, error budget
# --------------------------------------------------------------------------------------
def test_histories_and_energy_summary_report_median_and_iqr():
    data, _, _ = synthetic_ensemble()
    h = data.histories()
    for column in (
        "mean_energy_per_spin",
        "median_energy_per_spin",
        "iqr_energy_per_spin",
        "median_excess_energy",
        "iqr_excess_energy",
        "median_max_direction_change",
    ):
        assert column in h.columns, column
    e = data.arrays["energy"] / data.config.n_particles
    assert np.allclose(h["median_energy_per_spin"], np.median(e, axis=0))
    assert np.allclose(h["mean_energy_per_spin"], e.mean(axis=0))
    assert h["median_excess_energy"].iloc[-1] == 0.0 and h["iqr_excess_energy"].iloc[-1] == 0.0
    summary = data.energy_summary()
    assert summary["per_spin_median"] == pytest.approx(np.median(e[:, -1]))
    assert summary["per_spin_min"] <= summary["per_spin_q25"] <= summary["per_spin_median"]
    assert summary["per_spin_median"] <= summary["per_spin_q75"] <= summary["per_spin_max"]
    assert summary["excess_sweep"] == 5 and summary["excess_median"] > 0.0
    assert summary["dmax_last_max"] >= summary["dmax_last_median"] > 0.0


def test_round_to_error():
    assert round_to_error(0.84464, 0.00496) == "0.845 ± 0.005"
    assert round_to_error(0.18613, 0.00138) == "0.1861 ± 0.0014"
    assert round_to_error(1.2297, 0.033) == "1.23 ± 0.03"
    assert round_to_error(float("nan"), 0.1) == "n/a"
    assert round_to_error(0.5, float("nan")) == "0.5" and round_to_error(0.5, 0.0) == "0.5"


def test_paired_difference_falls_back_to_unpaired_for_different_systems(tmp_path):
    base = SimulationConfig(
        n_particles=300, n_sweeps=6, n_bins=8, x_max=3.0, n_realizations=4, base_seed=11
    )
    run_ensemble(base, tmp_path / "small", progress=False)
    run_ensemble(
        base.replace(n_particles=400, n_realizations=6), tmp_path / "large", progress=False
    )
    run_ensemble(base.replace(boundary="open"), tmp_path / "open", progress=False)
    small, large, open_ = (EnsembleData.load(tmp_path / n) for n in ("small", "large", "open"))
    # the boundary condition does not change positions or initial directions; N does
    assert same_system(small.config, open_.config) and not same_system(small.config, large.config)
    assert is_paired(small, open_) and not is_paired(small, large)
    unpaired = paired_difference(small, large, "C")
    assert not unpaired["paired"].any() and (unpaired["n_common"] == 0).all()
    ma, sa, _ = mean_sem(small.realization_means("C"))
    mb, sb, nb = mean_sem(large.realization_means("C"))
    assert np.allclose(unpaired["diff"], mb - ma, equal_nan=True)
    assert np.allclose(unpaired["sem"], np.sqrt(sa**2 + sb**2), equal_nan=True)
    assert (unpaired["n_other"] == nb).all() and unpaired["n_reference"].max() == 4
    with pytest.raises(ValueError, match="not paired"):
        paired_difference(small, large, "C", paired=True)
    paired = paired_difference(small, open_, "C")
    assert paired["paired"].all() and (paired["n_common"].iloc[2:] == 4).all()
    assert "x_mean" in paired.columns
    for obs in ("P", "Cperp"):
        assert np.isfinite(paired_difference(small, open_, obs)["diff"].iloc[2]), obs


def test_landmark_difference_and_error_budget(tmp_path):
    base = SimulationConfig(
        n_particles=500,
        n_sweeps=8,
        n_bins=16,
        x_max=4.0,
        n_realizations=6,
        base_seed=31,
        r_cut_nn=5.0,
    )
    run_ensemble(base, tmp_path / "main", progress=False)
    run_ensemble(base.replace(n_sweeps=16, n_realizations=4), tmp_path / "long", progress=False)
    run_ensemble(base.replace(r_cut_nn=6.0, n_realizations=4), tmp_path / "wide", progress=False)
    main, long_, wide = (EnsembleData.load(tmp_path / n) for n in ("main", "long", "wide"))

    diff = landmark_difference(main, long_, n_boot=30)
    assert diff["_meta"]["paired"] and diff["_meta"]["n_common"] == 4
    for key in ("x_zero_C", "C_peak", "Q_min", "Cperp_peak"):
        info = diff[key]
        assert info["value"] == pytest.approx(info["other"] - info["reference"]), key
        assert np.isfinite(info["err"]), key
    free = landmark_difference(main, wide, n_boot=10, abscissa="x_center")
    assert free["_meta"]["abscissa"] == "x_center"

    budget = landmark_error_budget(main, long_, wide, rebins=(1, 2), n_boot=30)
    rows = budget.set_index("landmark")
    assert list(budget["landmark"]).count("x_zero_C") == 1 and "Cperp_peak" in rows.index
    row = rows.loc["Q_min"]
    assert row["stat"] > 0 and row["binning"] >= 0
    assert np.isfinite(row["sweeps"]) and np.isfinite(row["cutoff"]) and row["cutoff_err"] > 0
    assert row["total"] == pytest.approx(
        np.sqrt(row["stat"] ** 2 + row["binning"] ** 2 + row["sweeps"] ** 2 + row["cutoff"] ** 2)
    )
    assert "±" in row["rounded"]
    assert row["sweeps"] == pytest.approx(diff["Q_min"]["value"]) and row[
        "sweeps_err"
    ] == pytest.approx(diff["Q_min"]["err"])
    assert budget.attrs["meta"]["sweeps"] == {"main": 8.0, "run": 16.0, "n_common": 4}
    assert (
        budget.attrs["meta"]["cutoff"]["run"] == 6.0
        and budget.attrs["meta"]["abscissa"] == "x_mean"
    )
    m1, m2 = main.landmarks(1, 30), main.landmarks(2, 30)
    assert row["binning"] == pytest.approx(abs(m1["Q_min"]["value"] - m2["Q_min"]["value"]))
    assert row["value"] == pytest.approx(m1["Q_min"]["value"])

    partial = landmark_error_budget(main, None, None, rebins=(1,), n_boot=30).set_index("landmark")
    assert np.isnan(partial.loc["Q_min", "binning"]) and np.isnan(partial.loc["Q_min", "sweeps"])
    assert partial.loc["Q_min", "total"] == pytest.approx(partial.loc["Q_min", "stat"])
    assert "sweeps" not in landmark_error_budget(main, rebins=(1,), n_boot=5).attrs["meta"]

    foreign = EnsembleData.load(tmp_path / "long")
    foreign.seeds = foreign.seeds + 1000  # same system, disjoint seeds: not paired
    with pytest.raises(ValueError, match="not paired"):
        landmark_error_budget(main, foreign, None, rebins=(1,), n_boot=5)
    with pytest.raises(ValueError, match="larger cutoff"):  # not a cutoff variant of main
        landmark_error_budget(main, None, long_, rebins=(1,), n_boot=5)


# --------------------------------------------------------------------------------------
# Energy budget of the stored states
# --------------------------------------------------------------------------------------
def test_energy_budget_from_states(tiny_run, nn_run, tmp_path):
    cfg, out = tiny_run
    frame = energy_budget_from_states(out)
    assert len(frame) == 1 and frame["seed"].iloc[0] == cfg.seeds[0]
    rec = load_record(seed_path(out, cfg.seeds[0]))
    row = frame.iloc[0]
    # the budget reproduces the energies of the relaxation (final and initial state, with G)
    assert row["energy_total"] == pytest.approx(rec["energy"][-1], rel=1e-9)
    assert row["energy_total_initial"] == pytest.approx(rec["energy"][0], rel=1e-9)
    assert row["energy_dimer"] + row["energy_nondimer"] == pytest.approx(row["energy_total"])
    assert (
        row["energy_total"] >= row["minimum_total"] and row["energy_dimer"] >= row["minimum_dimer"]
    )
    assert 0.0 < row["dimer_share"] < 1.0
    assert row["nondimer_share"] == pytest.approx(1.0 - row["dimer_share"])
    assert 0.8 < row["realized_fraction_dimer"] <= 1.0
    assert 0.0 < row["realized_fraction_nondimer"] < row["realized_fraction_dimer"]
    assert 0.0 < row["frustration_index"] <= 1.0 and abs(row["frustration_index_initial"]) < 0.3
    assert row["frustration_index"] == pytest.approx(row["energy_total"] / row["minimum_total"])
    assert row["n_dimer_spins"] == 2 * row["n_dimers"]
    assert row["n_dimer_spins"] == pytest.approx(rec["nn_frac_mutual"] * cfg.n_particles)
    assert frame.attrs["meta"] == {
        "n_states": 1,
        "seeds": [cfg.seeds[0]],
        "coupling": "gr",
        "sigma": 1.0,
        "gamma": 3.0,
        "two_body_factor": 2.0,
        "r_cut_nn": cfg.r_cut_nn_resolved,
        "n_particles": cfg.n_particles,
        "G": 1.0,
    }
    stats = summarize_energy_budget(frame)
    assert stats["n_states"] == 1 and stats["meta"] == frame.attrs["meta"]
    assert set(ENERGY_BUDGET_RATIOS) <= set(stats)
    assert stats["dimer_share"]["mean"] == pytest.approx(row["dimer_share"])
    assert np.isnan(stats["dimer_share"]["sem"])  # a single state has no spread
    assert (
        stats["dimer_share"]["min"] == stats["dimer_share"]["max"] == stats["dimer_share"]["mean"]
    )
    assert stats["energy_total_per_spin"]["mean"] == pytest.approx(
        row["energy_total"] / cfg.n_particles
    )
    assert stats["minimum_total_per_spin"]["mean"] < stats["energy_total_per_spin"]["mean"] < 0
    with pytest.raises(FileNotFoundError, match="no stored states"):
        energy_budget_from_states(nn_run[1])
    # controls: the two-body factor follows the coupling and the budget stays within its bounds
    for coupling, factor in (("dipolar", 2.0), ("isotropic_afm", 1.0)):
        c = SimulationConfig(
            n_particles=200,
            n_sweeps=5,
            n_bins=4,
            x_max=2.0,
            n_realizations=2,
            save_states=2,
            coupling=coupling,
            base_seed=3,
        )
        run_ensemble(c, tmp_path / coupling, progress=False)
        f = energy_budget_from_states(tmp_path / coupling)
        meta = f.attrs["meta"]
        assert len(f) == 2 and meta["two_body_factor"] == factor and meta["coupling"] == coupling
        assert (meta["sigma"], meta["gamma"]) == c.sigma_gamma
        assert (f["energy_total"] >= f["minimum_total"]).all()
        assert (f["energy_dimer"] >= f["minimum_dimer"]).all()
        assert (f["frustration_index"] > 0.0).all() and (f["realized_fraction_dimer"] <= 1.0).all()
        records = [load_record(seed_path(tmp_path / coupling, s)) for s in c.seeds]
        assert np.allclose(f["energy_total"], [r["energy"][-1] for r in records], rtol=1e-9)
        assert energy_budget_from_states(tmp_path / coupling, max_states=1)["seed"].tolist() == [3]
        s = summarize_energy_budget(f)
        assert s["n_states"] == 2 and np.isfinite(s["frustration_index"]["sem"])
        assert s["frustration_index"]["min"] <= s["frustration_index"]["mean"]
        assert s["frustration_index"]["mean"] <= s["frustration_index"]["max"]


# --------------------------------------------------------------------------------------
# Interpolated curve values and the channel (gamma) family
# --------------------------------------------------------------------------------------
def test_curve_value_at_interpolates_mean_and_sem_on_x_mean():
    from grspin.analysis import curve_value_at

    data, curves, x = synthetic_ensemble(n_real=12, noise=0.0)
    # x_mean == x_center for the synthetic ensemble (sum_x = counts * x), so the interpolation is
    # exact at a bin centre and linear between two centres
    k = 10
    value, sem = curve_value_at(data, "C", float(x[k]))
    assert value == pytest.approx(curves["C"][k], abs=1e-12) and sem == pytest.approx(0.0)
    mid = 0.5 * (x[k] + x[k + 1])
    value, _ = curve_value_at(data, "C", float(mid))
    assert value == pytest.approx(0.5 * (curves["C"][k] + curves["C"][k + 1]), abs=1e-12)
    value, _ = curve_value_at(data, "Cperp", float(x[k]))
    assert value == pytest.approx(curves["C"][k] - curves["P"][k], abs=1e-12)
    assert np.isnan(curve_value_at(data, "C", 100.0)[0])  # outside the tabulated range
    noisy, _, _ = synthetic_ensemble(n_real=12, noise=0.05, seed=3)
    _, sem = noisy_value = curve_value_at(noisy, "C", float(x[k]))
    assert np.isfinite(noisy_value[0]) and 0.0 < sem < 0.05


def test_gamma_family_members_orders_by_gamma_and_rejects_other_families():
    from grspin.analysis import gamma_family_members

    def fake(coupling: str) -> EnsembleData:
        data, _, _ = synthetic_ensemble(n_real=3)
        return EnsembleData(data.config.replace(coupling=coupling), data.arrays)

    runs = {
        "gr": fake("gr"),
        "long": fake("longitudinal_only"),
        "trans": fake("transverse_only"),
        "g12": fake("gamma12"),
    }
    assert gamma_family_members(runs) == ["trans", "gr", "g12", "long"]
    assert gamma_family_members({"a": fake("gr"), "b": fake("transverse_only")}) == ["b", "a"]
    # the controls change the sign: not the family; one gamma at several cutoffs: not the family
    assert gamma_family_members({"gr": fake("gr"), "iso": fake("isotropic_afm")}) is None
    assert gamma_family_members({"a": fake("gr"), "b": fake("gr")}) is None
    assert gamma_family_members({"d": fake("dipolar"), "g": fake("gamma12")}) is None
