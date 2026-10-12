"""Reports and figures: channels, x_mean landmarks, error budget, paired/unpaired comparisons."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from conftest import synthetic_ensemble
from matplotlib.figure import Figure

from grspin.analysis import EnsembleData
from grspin.cli import main
from grspin.config import SimulationConfig
from grspin.ensemble import run_ensemble
from grspin.fit import FitResult, fit_ensemble
from grspin.plotting import (
    draw_A,
    draw_B,
    figure_C,
    figure_compare,
    figure_decomposition,
    figure_overview,
)
from grspin.report import analyze_run, compare_runs, find_budget_runs, summary_markdown


# --------------------------------------------------------------------------------------
# analyze_run: sibling search, error budget, summary sections
# --------------------------------------------------------------------------------------
def test_analyze_run_writes_channels_landmarks_and_error_budget(tmp_path):
    root = tmp_path / "results"
    base = SimulationConfig(
        n_particles=500,
        n_sweeps=8,
        n_bins=16,
        x_max=4.0,
        n_realizations=4,
        base_seed=100,
        r_cut_nn=5.0,
    )
    run_ensemble(base, root / "M", progress=False)
    run_ensemble(
        base.replace(n_sweeps=16, n_realizations=3, snapshots=(8, 16)), root / "S1", progress=False
    )
    run_ensemble(base.replace(r_cut_nn=5.5, n_realizations=3), root / "S2_a", progress=False)
    run_ensemble(base.replace(r_cut_nn=6.0, n_realizations=3), root / "S2_b", progress=False)
    run_ensemble(base.replace(coupling="dipolar", n_realizations=2), root / "S6", progress=False)
    run_ensemble(base.replace(r_cut_nn=6.5, n_realizations=1), root / "one", progress=False)
    (root / "_comparisons").mkdir()
    # the longest relaxation and the largest cutoff with >= 2 realizations; other couplings never
    assert find_budget_runs(root / "M") == {"sweeps_run": root / "S1", "cutoff_run": root / "S2_b"}
    assert find_budget_runs(root / "S6") == {}

    written = analyze_run(root / "M", rebin=2, n_boot=10)
    out = root / "M" / "analysis"
    budget = pd.read_csv(out / "error_budget.csv").set_index("landmark")
    for column in (
        "value",
        "stat",
        "binning",
        "sweeps",
        "sweeps_err",
        "cutoff",
        "cutoff_err",
        "total",
        "rounded",
    ):
        assert column in budget.columns, column
    assert (
        budget.loc["Q_min", ["value", "stat", "binning", "sweeps", "cutoff", "total"]].notna().all()
    )
    assert "Cperp_peak" in budget.index and "±" in budget.loc["Q_min", "rounded"]
    summary = json.loads((out / "summary.json").read_text())
    assert summary["rebin"] == 2 and summary["landmark_rebin"] == 1
    assert summary["landmarks"]["_meta"] == {
        "n_real": 4,
        "rebin": 1,
        "n_bins": 16,
        "n_boot": 10,
        "abscissa": "x_mean",
        "bin_width": 0.25,
    }
    assert summary["error_budget"]["meta"]["sweeps"] == {"main": 8.0, "run": 16.0, "n_common": 3}
    assert summary["error_budget"]["meta"]["cutoff"] == {"main": 5.0, "run": 6.0, "n_common": 3}
    assert summary["error_budget"]["meta"]["rebins"] == [1, 2]
    assert {"Cperp_peak", "x_peak_Cperp", "P_at_Cperp_peak"} <= set(summary["landmarks"])
    assert summary["energy"]["per_spin_q25"] <= summary["energy"]["per_spin_median"]
    assert summary["fit"]["abscissa"] == "x_mean"
    text = (out / "summary.md").read_text()
    for needle in (
        "pair-weighted mean distance",
        "bin width of 0.25 r_nn (rebin factor 1, 16 bins)",
        "## Error budget of the landmarks",
        "sweeps = 16 instead of 8 on 3 common seeds",
        "r_cut = 6 r_nn instead of 5 r_nn",
        "C_perp = C - P",
        "## Energies",
        "dominated by the closest pairs",
        "phenomenological guide (not used for quantitative statements)",
    ):
        assert needle in text, needle
    table = pd.read_csv(out / "ensemble_table.csv")
    assert {"mean_P_final", "sem_P_initial", "mean_Cperp_final", "sem_Cperp_final"} <= set(
        table.columns
    )
    histories = pd.read_csv(out / "histories.csv")
    assert {"median_energy_per_spin", "iqr_excess_energy"} <= set(histories.columns)
    assert "decomposition.pdf" in written and (out / "decomposition.pdf").exists()
    assert (out / "overview.pdf").exists() and (out / "error_budget.csv") == written["error_budget"]

    # explicit runs override the search; a missing explicit run is skipped with a warning
    with pytest.warns(UserWarning, match="not a run directory"):
        analyze_run(
            root / "M",
            tmp_path / "an2",
            n_boot=5,
            plots=False,
            fit=False,
            sweeps_run=tmp_path / "nowhere",
            cutoff_run=root / "S2_a",
        )
    second = pd.read_csv(tmp_path / "an2" / "error_budget.csv").set_index("landmark")
    assert np.isnan(second.loc["Q_min", "sweeps"]) and np.isfinite(second.loc["Q_min", "cutoff"])
    meta = json.loads((tmp_path / "an2" / "summary.json").read_text())["error_budget"]["meta"]
    assert meta["cutoff"]["run"] == 5.5 and "sweeps" not in meta
    assert (
        "not available (no paired run with a longer relaxation)"
        in (tmp_path / "an2" / "summary.md").read_text()
    )
    with pytest.raises(ValueError, match="longer"):  # a run with the same number of sweeps
        analyze_run(root / "M", tmp_path / "an4", n_boot=5, plots=False, sweeps_run=root / "S2_a")
    # the budget can be switched off; the summary then has no such section
    analyze_run(root / "M", tmp_path / "an3", n_boot=5, plots=False, fit=False, error_budget=False)
    assert not (tmp_path / "an3" / "error_budget.csv").exists()
    assert "Error budget" not in (tmp_path / "an3" / "summary.md").read_text()
    # a run without any usable sibling still gets a budget (statistical and binning terms only)
    analyze_run(root / "S6", n_boot=5, plots=False, fit=False)
    alone = pd.read_csv(root / "S6" / "analysis" / "error_budget.csv").set_index("landmark")
    assert np.isnan(alone.loc["Q_min", "sweeps"]) and np.isfinite(alone.loc["Q_min", "total"])
    assert alone.loc["Q_min", "total"] == pytest.approx(
        np.hypot(alone.loc["Q_min", "stat"], alone.loc["Q_min", "binning"])
    )


def test_summary_markdown_handles_missing_sections():
    data, _, _ = synthetic_ensemble(n_real=6)
    cfg = data.config
    summary = {
        "run_dir": "x",
        "n_realizations": 6,
        "rebin": 1,
        "rebin_n_bins": 40,
        "landmark_rebin": 1,
        "config": cfg.to_dict(),
        "derived": {
            "box_length": cfg.box,
            "density": 1.0,
            "r_nn": cfg.r_nn,
            "r_cut": cfg.r_cut,
            "r_cut_nn": cfg.r_cut_nn_resolved,
            "neighbors_per_particle": 1.0,
        },
        "scalars": {"S0_unit_final": {"mean": 0.4, "sem": 0.1}},
        "energy": None,
        "energy_budget": None,
        "landmarks": {"x_zero_C": {"value": 0.84, "err": 0.01}},
        "error_budget": None,
        "dimer_map": None,
        "dimer_field": None,
        "fit": None,
    }
    text = summary_markdown(summary)
    assert "the bin centres" in text and "0.84 ± 0.01" in text and "n/a" in text
    assert "Error budget" not in text and "Energies" not in text
    assert "Damped-oscillatory" not in text
    assert "Energy budget" not in text and "Dimer-field" not in text and "Dimer-frame" not in text
    # the energy-budget and dimer-field sections render from their summary dictionaries
    stat = {"mean": 0.85, "sem": 0.02, "min": 0.8, "max": 0.9}
    summary["energy_budget"] = {
        "meta": {"coupling": "dipolar", "sigma": -1.0, "gamma": 3.0, "two_body_factor": 2.0},
        "n_states": 3,
        **{
            key: dict(stat)
            for key in (
                "dimer_share",
                "nondimer_share",
                "realized_fraction_dimer",
                "realized_fraction_nondimer",
                "frustration_index",
                "frustration_index_initial",
                "energy_total_per_spin",
                "energy_dimer_per_spin",
                "energy_nondimer_per_spin",
                "minimum_total_per_spin",
            )
        },
    }
    bins = [
        {
            "x_lo": lo,
            "x_hi": hi,
            "n": 100,
            "n_agree": 90,
            "agree": 0.9,
            "sem_agree": 0.01,
            "agree_pooled": 0.9,
            "cos": 0.7,
            "sem_cos": 0.02,
            "n_real": 3,
        }
        for lo, hi in ((0.5, 0.75), (0.75, 1.0), (1.0, 1.25), (1.25, 1.5), (1.5, 2.0), (2.0, 2.5))
    ] + [
        {
            "x_lo": 0.0,
            "x_hi": 2.5,
            "n": 610,
            "n_agree": 545,
            "agree": 0.893,
            "sem_agree": 0.01,
            "agree_pooled": 0.893,
            "cos": None,
            "sem_cos": None,
            "n_real": 3,
        }
    ]
    summary["dimer_field"] = {
        "sigma": -1.0,
        "gamma": 3.0,
        "r_max_nn": 2.5,
        "n_pairs": 610,
        "n_agree": 545,
        "agree_pooled": 0.893,
        "agree_mean": 0.893,
        "agree_sem": 0.01,
        "n_real": 3,
        "source": "states",
        "n_states": 3,
        "bins": bins,
    }
    text = summary_markdown(summary)
    for needle in (
        "## Energy budget of the stored states",
        "3 stored relaxed states with the run's coupling `dipolar` (sigma = -1, gamma = 3)",
        "f = max(1, |gamma - 1|) = 2",
        "0.85 ± 0.02 (range 0.8 to 0.9)",
        "## Dimer-field test",
        "sigma = -1, gamma = 3",
        "| [0.5, 0.75) | 100 | 0.9 ± 0.01 | 0.9 | 0.7 ± 0.02 |",
        "| all within 2.5 r_nn | 610 | 0.893 ± 0.01 | 0.893 | n/a |",
        "Computed from the 3 stored states",
    ):
        assert needle in text, needle
    assert "Dimer-only prediction" not in text  # no angular table with C_pred was given


# --------------------------------------------------------------------------------------
# compare_runs: paired versus unpaired members, channels as observables
# --------------------------------------------------------------------------------------
def test_compare_runs_names_paired_and_unpaired_differences(tmp_path):
    base = SimulationConfig(
        n_particles=300, n_sweeps=6, n_bins=8, x_max=3.0, n_realizations=3, base_seed=7
    )
    run_ensemble(base, tmp_path / "ref", progress=False)
    run_ensemble(base.replace(boundary="open"), tmp_path / "open", progress=False)
    run_ensemble(
        base.replace(n_particles=400, n_realizations=4), tmp_path / "bigger", progress=False
    )
    runs = {label: EnsembleData.load(tmp_path / label) for label in ("ref", "open", "bigger")}
    out = tmp_path / "cmp"
    written = compare_runs(runs, out, ("C", "P", "Cperp"), rebin=1, n_boot=5)
    assert (out / "paired_Cperp_open_minus_ref.csv").exists()
    assert (out / "paired_P_open_minus_ref.csv").exists()
    assert (out / "unpaired_C_bigger_minus_ref.csv").exists()
    assert not (out / "paired_C_bigger_minus_ref.csv").exists()
    unpaired = pd.read_csv(out / "unpaired_P_bigger_minus_ref.csv")
    assert not unpaired["paired"].any() and unpaired["n_other"].max() == 4
    paired = pd.read_csv(out / "paired_C_open_minus_ref.csv")
    assert paired["paired"].all() and paired["n_common"].max() == 3
    diffs = pd.read_csv(out / "comparison_landmark_differences.csv").set_index("run")
    assert bool(diffs.loc["open", "paired"]) and not bool(diffs.loc["bigger", "paired"])
    assert diffs.loc["open", "n_common"] == 3 and diffs.loc["bigger", "n_common"] == 0
    assert "d_Cperp_peak_err" in diffs.columns and "d_x_zero_C" in diffs.columns
    text = (out / "comparison_summary.md").read_text()
    assert "paired (3 common seeds)" in text and "unpaired (all realizations)" in text
    assert "Landmark shifts" in text
    marks = pd.read_csv(out / "comparison_landmarks.csv")
    assert {"Cperp_peak", "x_peak_Cperp", "P_at_Cperp_peak", "coupling", "abscissa"} <= set(
        marks.columns
    )
    assert (marks["abscissa"] == "x_mean").all() and (marks["rebin"] == 1).all()
    assert "comparison_summary" in written and (out / "comparison.pdf").exists()
    assert "paired_Cperp_open" in written and "unpaired_C_bigger" in written


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------
def test_decomposition_and_overview_figures_with_data_driven_limits():
    data, _, _ = synthetic_ensemble(n_real=8)
    fig = figure_decomposition(data)
    assert len(fig.axes) == 2
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert any("C_\\perp" in lab for lab in labels) and any("longitudinal" in lab for lab in labels)
    assert fig.axes[1].get_xlim()[0] == pytest.approx(2.0)  # the tail panel starts at x_max / 2
    assert len(figure_decomposition(data, tail=False).axes) == 1
    overview = figure_overview(data)
    assert len(overview.axes) == 6
    assert all(ax.axison for ax in overview.axes)
    assert not any("fit" in lab for lab in overview.axes[0].get_legend_handles_labels()[1])
    # B limits follow the data: a dipolar-like run with B < 0 stays inside the axis
    data.arrays["fin_sum_B"] = -np.abs(data.arrays["fin_sum_B"])
    t = data.table()
    ax = Figure().subplots()
    draw_B(ax, data)
    lo, hi = ax.get_ylim()
    assert lo < t["mean_B_final"].min() and hi > max(0.0, float(t["mean_B_initial"].max()))
    ax = Figure().subplots()
    draw_A(ax, data)
    lo, hi = ax.get_ylim()
    assert lo < t["mean_A_final"].min() and lo < 1.0 / 3.0 < hi
    # without the extra observables the dependent panels are blank, the figure still renders
    reduced = EnsembleData(
        data.config, {k: v for k, v in data.arrays.items() if not k.endswith(("_A", "_P", "_B"))}
    )
    blank = figure_overview(reduced)
    assert [ax.axison for ax in blank.axes] == [True, False, True, False, False, False]


def test_draw_C_offers_the_dimer_fraction_reference_and_no_fit_by_default():
    data, _, _ = synthetic_ensemble(n_real=8)
    labels = figure_C(data, show_nn=True).axes[0].get_legend_handles_labels()[1]
    assert any("dimer fraction" in lab for lab in labels)
    assert not any("4\\pi" in lab for lab in labels)  # the isolated-sphere curve is not offered
    assert figure_C(data).axes[0].get_legend_handles_labels()[1] == ["initial (random)", "relaxed"]
    fit = fit_ensemble(data, n_boot=0)["fit"]
    with_fit = figure_C(data, fit=FitResult(**fit)).axes[0].get_legend_handles_labels()[1]
    assert "damped-oscillatory fit" in with_fit
    comp = figure_compare({"a": data}, ("C", "Cperp"), show_nn=True)
    assert len(comp.axes) == 2
    assert any("dimer fraction" in lab for lab in comp.axes[0].get_legend_handles_labels()[1])


# --------------------------------------------------------------------------------------
# Command line
# --------------------------------------------------------------------------------------
def test_cli_analyze_options_and_compare_with_channels(tmp_path, capsys):
    common = ["-N", "300", "-R", "3", "--n-sweeps", "4", "--n-bins", "8", "--x-max", "3", "--quiet"]
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["run", *common, "--out", str(a)]) == 0
    assert main(["run", *common, "--n-sweeps", "8", "--out", str(b)]) == 0  # sibling: longer run
    capsys.readouterr()
    assert main(["analyze", str(a), "--landmark-rebin", "2", "--show-fit", "--n-boot", "5"]) == 0
    printed = capsys.readouterr().out
    assert "Error budget" in printed and "sweeps = 8 instead of 4 on 3 common seeds" in printed
    summary = json.loads((a / "analysis" / "summary.json").read_text())
    assert summary["landmark_rebin"] == 2 and summary["landmarks"]["_meta"]["bin_width"] == 0.75
    assert summary["error_budget"]["meta"]["rebins"] == [2, 4]
    budget = pd.read_csv(a / "analysis" / "error_budget.csv").set_index("landmark")
    assert np.isfinite(budget.loc["Q_min", "sweeps"]) and np.isnan(budget.loc["Q_min", "cutoff"])
    explicit = ["analyze", str(a), "--out", str(tmp_path / "an"), "--no-plots", "--no-fit"]
    # an explicit cutoff run must have a larger cutoff (b only relaxes longer): refused with status 2
    with pytest.warns(UserWarning, match="not a run directory"):
        status = main(
            [*explicit, "--sweeps-run", str(tmp_path / "missing"), "--cutoff-run", str(b)]
        )
    assert status == 2 and "larger cutoff" in capsys.readouterr().err
    assert main([*explicit, "--n-boot", "5", "--sweeps-run", str(b)]) == 0
    capsys.readouterr()
    meta = json.loads((tmp_path / "an" / "summary.json").read_text())["error_budget"]["meta"]
    assert meta["sweeps"] == {"main": 4.0, "run": 8.0, "n_common": 3} and "cutoff" not in meta


def test_cli_compare_accepts_channels(tmp_path, capsys):
    common = ["-N", "300", "-R", "3", "--n-sweeps", "4", "--n-bins", "8", "--x-max", "3", "--quiet"]
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["run", *common, "--out", str(a)]) == 0
    assert main(["run", *common, "--coupling", "isotropic_afm", "--out", str(b)]) == 0
    cmp_dir = tmp_path / "cmp"
    args = [
        "compare",
        str(a),
        str(b),
        "--labels",
        "gr",
        "afm",
        "--out",
        str(cmp_dir),
        "--n-boot",
        "5",
    ]
    assert main([*args, "--obs", "C", "P", "Cperp", "--landmark-rebin", "2"]) == 0
    capsys.readouterr()
    assert (cmp_dir / "paired_Cperp_afm_minus_gr.csv").exists()
    marks = pd.read_csv(cmp_dir / "comparison_landmarks.csv")
    assert (marks["rebin"] == 2).all() and list(marks["coupling"]) == ["gr", "isotropic_afm"]
    assert (cmp_dir / "comparison_summary.md").exists()


# --------------------------------------------------------------------------------------
# Channel (gamma) family: landmark table against gamma and the family figure
# --------------------------------------------------------------------------------------
def test_compare_runs_writes_the_gamma_family_table_and_figure_for_the_channel_family(tmp_path):
    from grspin.plotting import figure_gamma_family
    from grspin.report import gamma_family_table

    base = SimulationConfig(
        n_particles=300, n_sweeps=6, n_bins=8, x_max=3.0, n_realizations=3, base_seed=11
    )
    for coupling in ("gr", "transverse_only", "longitudinal_only"):
        run_ensemble(base.replace(coupling=coupling), tmp_path / coupling, progress=False)
    runs = {
        "gamma = 3 (GR)": EnsembleData.load(tmp_path / "gr"),
        "gamma = 1": EnsembleData.load(tmp_path / "transverse_only"),
        "gamma = 100": EnsembleData.load(tmp_path / "longitudinal_only"),
    }
    out = tmp_path / "cmp"
    written = compare_runs(runs, out, ("C", "P", "Cperp"), rebin=1, n_boot=5)
    assert "gamma_family" in written and "gamma_family.pdf" in written
    family = pd.read_csv(out / "gamma_family.csv")
    assert family["gamma"].tolist() == [1.0, 3.0, 100.0]  # ordered by gamma, not by label
    assert family["coupling"].tolist() == ["transverse_only", "gr", "longitudinal_only"]
    assert (family["sigma"] == 1.0).all() and (family["x_sign"] == 1.2).all()
    assert {"x_zero_C", "x_zero_C_err", "C_peak", "Cperp_peak", "Q_min", "C_at_x_sign"} <= set(
        family.columns
    )
    assert set(family["sign_C_at_x_sign"].dropna()) <= {-1.0, 0.0, 1.0}
    marks = pd.read_csv(out / "comparison_landmarks.csv").set_index("run")
    for _, row in family.iterrows():
        assert row["C_peak"] == pytest.approx(marks.loc[row["run"], "C_peak"], nan_ok=True)
    text = (out / "comparison_summary.md").read_text()
    assert "## Channel family: landmarks against gamma" in text and "C(1.2 r_nn)" in text
    assert "| positions |" in text and text.count("| Poisson |") == 3
    # the table alone, without the files
    assert gamma_family_table(runs, marks.reset_index(), 1) is not None
    fig = figure_gamma_family(runs, rebin=1)
    assert isinstance(fig, Figure) and len(fig.axes) == 3
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert labels[0].startswith(r"$\gamma = 1$") and labels[-1].startswith(r"$\gamma = 100$")
    # not the family: GR versus a sign-reversed control -> no gamma table, no figure
    run_ensemble(base.replace(coupling="dipolar"), tmp_path / "dipolar", progress=False)
    others = {"gr": runs["gamma = 3 (GR)"], "dip": EnsembleData.load(tmp_path / "dipolar")}
    written = compare_runs(others, tmp_path / "cmp2", ("C",), rebin=1, n_boot=5)
    assert "gamma_family" not in written and not (tmp_path / "cmp2" / "gamma_family.csv").exists()
    assert "Channel family" not in (tmp_path / "cmp2" / "comparison_summary.md").read_text()
    assert gamma_family_table(others, marks.reset_index(), 1) is None
    with pytest.raises(ValueError, match="channel family"):
        figure_gamma_family(others)


def test_summary_header_names_the_position_process(tmp_path):
    data, _, _ = synthetic_ensemble(n_real=4)
    cfg = data.config
    base = {
        "run_dir": "x",
        "n_realizations": 4,
        "rebin": 1,
        "rebin_n_bins": 40,
        "landmark_rebin": 1,
        "derived": {
            "box_length": cfg.box,
            "density": 1.0,
            "r_nn": cfg.r_nn,
            "r_cut": cfg.r_cut,
            "r_cut_nn": cfg.r_cut_nn_resolved,
            "neighbors_per_particle": 1.0,
        },
        "scalars": {},
        "landmarks": {},
    }
    assert "positions = Poisson" in summary_markdown({**base, "config": cfg.to_dict()})
    hard = cfg.replace(min_separation_nn=0.5).to_dict()
    text = summary_markdown({**base, "config": hard})
    assert "hard core 0.5 r_nn" in text and "Poisson references do not apply" in text
    old = {k: v for k, v in cfg.to_dict().items() if k != "min_separation_nn"}  # a 0.2.0 summary
    assert "positions = Poisson" in summary_markdown({**base, "config": old})


def test_cli_analyze_max_states_and_no_energy_budget(tmp_path, capsys):
    out = tmp_path / "run"
    common = ["-N", "300", "-R", "3", "--n-sweeps", "4", "--n-bins", "8", "--x-max", "3"]
    assert main(["run", *common, "--save-states", "3", "--quiet", "--out", str(out)]) == 0
    capsys.readouterr()
    assert main(["analyze", str(out), "--n-boot", "5", "--no-plots", "--max-states", "2"]) == 0
    printed = capsys.readouterr().out
    assert "energy_budget" in printed and "Energy budget of the stored states" in printed
    budget = pd.read_csv(out / "analysis" / "energy_budget.csv")
    assert len(budget) == 2  # --max-states
    assert (
        json.loads((out / "analysis" / "summary.json").read_text())["structure_factor"]["n_states"]
        == 2
    )
    again = tmp_path / "again"
    assert (
        main(
            [
                "analyze",
                str(out),
                "--out",
                str(again),
                "--n-boot",
                "5",
                "--no-plots",
                "--no-energy-budget",
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "energy_budget" not in printed and not (again / "energy_budget.csv").exists()
    assert (again / "structure_factor.csv").exists()
