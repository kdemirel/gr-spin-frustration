"""Dimer-frame correlation map and spin structure factor: aggregation, stored states, outputs."""

from __future__ import annotations

import json
import shutil
import warnings

import numpy as np
import pandas as pd
import pytest

from grspin import theory
from grspin.analysis import (
    DIMER_ANG_REGIONS,
    EnsembleData,
    aggregate_dimer_map,
    dimer_map_from_states,
    structure_factor_from_states,
    tabulate_dimer_angular,
    tabulate_dimer_field,
    tabulate_dimer_map,
)
from grspin.config import SimulationConfig
from grspin.ensemble import load_record, run_ensemble, seed_path
from grspin.plotting import (
    figure_dimer_map,
    figure_structure_factor,
    figure_structure_factor_comparison,
)
from grspin.report import analyze_run, structure_factor_comparison


def _synthetic_map_records(n_real: int = 5, seed: int = 0) -> tuple[list[dict], np.ndarray]:
    """Records with a dense cell (10, 5), a cell fed by part of the realizations and empty cells."""
    rng = np.random.default_rng(seed)
    values = rng.uniform(-0.8, 0.8, size=(n_real, 50, 25))
    records = []
    for r in range(n_real):
        cnt = np.zeros((50, 25), dtype=np.int64)
        cnt[10, 5] = 30  # dense cell
        cnt[20, 7] = 4 if r % 2 == 0 else 0  # fed by every second realization, pooled 12 or 8
        cnt[30, 9] = 7  # pooled 35 with 5 realizations
        ang = rng.integers(0, 5, size=(6, 7))
        records.append(
            {
                "seed": np.array(r),
                "dimer_map_cnt": cnt,
                "dimer_map_sum_C": cnt * values[r],
                "dimer_map_sum_P": cnt * (-0.5 * values[r]),
                "dimer_map_r_max": np.array(2.5),
                "dimer_map_n_spins": np.array(100 + r),
                "dimer_map_sum_x_partner": np.array(0.84 * (100 + r)),
                "dimer_ang_cnt": ang,
                "dimer_ang_sum_C": ang * 0.25,
                "dimer_ang_sum_P": ang * -0.1,
            }
        )
    return records, values


# --------------------------------------------------------------------------------------
# Aggregation
# --------------------------------------------------------------------------------------
def test_dimer_map_aggregates_cell_means_over_realizations_and_masks_sparse_cells():
    records, values = _synthetic_map_records(n_real=5)
    data = EnsembleData.from_records(SimulationConfig(n_particles=300), records)
    m = data.dimer_map(min_count=20)
    assert m["C"].shape == (50, 25) and m["count"].sum() == 5 * 30 + 12 + 35
    assert m["r_max"] == 2.5 and m["z_edges"][0] == -2.5 and m["rho_edges"][-1] == 2.5
    assert len(m["z_edges"]) == 51 and len(m["rho_edges"]) == 26
    assert m["C"][10, 5] == pytest.approx(values[:, 10, 5].mean())
    assert m["sem_C"][10, 5] == pytest.approx(values[:, 10, 5].std(ddof=1) / np.sqrt(5))
    assert m["P"][10, 5] == pytest.approx(-0.5 * values[:, 10, 5].mean())
    assert m["C"][30, 9] == pytest.approx(values[:, 30, 9].mean())  # 35 pooled pairs: kept
    assert np.isnan(m["C"][20, 7]) and np.isnan(m["sem_C"][20, 7])  # 12 pooled pairs: masked
    assert np.isnan(m["C"][0, 0]) and np.isnan(m["P"][0, 0])  # empty
    assert np.isfinite(m["C"]).sum() == 2
    low = data.dimer_map(min_count=10)
    assert low["C"][20, 7] == pytest.approx(values[::2, 20, 7].mean())  # only fed realizations
    assert m["n_real"] == 5 and m["n_pairs"] == 197 and m["min_count"] == 20
    assert m["x_partner_mean"] == pytest.approx(0.84)
    table = data.dimer_map_table(min_count=20)
    assert len(table) == 1250 and table["n"].sum() == 197
    row = table[np.isclose(table["z_center"], -1.45) & np.isclose(table["rho_center"], 0.55)]
    assert len(row) == 1 and row["C"].iloc[0] == pytest.approx(m["C"][10, 5])
    assert set(table.columns) >= {"z_lo", "z_hi", "rho_lo", "rho_hi", "sem_C", "P", "sem_P"}
    assert np.allclose(tabulate_dimer_map(data.arrays)["C"], table["C"], equal_nan=True)


def test_dimer_angular_table_layout_and_means():
    records, _ = _synthetic_map_records(n_real=3)
    data = EnsembleData.from_records(SimulationConfig(n_particles=300), records)
    t = data.dimer_angular_table()
    assert len(t) == 42 and list(t.columns) == [
        "x_lo",
        "x_hi",
        "cos_lo",
        "cos_hi",
        "region",
        "n",
        "C",
        "sem_C",
        "P",
        "sem_P",
    ]
    assert t["x_lo"].iloc[0] == 0.5 and t["x_hi"].iloc[-1] == 2.5
    assert t["cos_lo"].iloc[0] == -1.0 and t["cos_hi"].iloc[6] == 1.0
    assert list(t["region"].iloc[:7]) == list(DIMER_ANG_REGIONS)
    assert t["region"].iloc[3] == "transverse" and t["region"].iloc[6].startswith("collinear")
    pooled = sum(r["dimer_ang_cnt"] for r in records).ravel()
    assert np.array_equal(t["n"].to_numpy(), pooled)
    fed = t["n"] > 0
    assert np.allclose(t["C"][fed], 0.25) and np.allclose(t["P"][fed], -0.1)
    assert t["C"][~fed].isna().all() and (t["sem_C"][fed].fillna(0.0) < 1e-12).all()
    assert tabulate_dimer_angular(data.arrays).equals(t)


def test_dimer_map_requires_the_diagnostics(tmp_path, small_config):
    run_ensemble(small_config, tmp_path / "run", progress=False)
    data = EnsembleData.load(tmp_path / "run")
    for method in (data.dimer_map, data.dimer_map_table, data.dimer_angular_table):
        with pytest.raises(ValueError, match="nn_diagnostics"):
            method()
    with pytest.raises(ValueError, match="nn_diagnostics"):
        aggregate_dimer_map({"seed": np.arange(3)})


def test_mixed_ensembles_treat_missing_keys_as_empty(tiny_run):
    cfg, out = tiny_run
    records = [load_record(seed_path(out, s)) for s in cfg.seeds[:3]]
    old = {k: v for k, v in records[0].items() if not k.startswith("dimer_")}
    with pytest.warns(UserWarning, match="dimer_map_cnt \\(2 of 3\\)"):
        data = EnsembleData.from_records(cfg, [old, records[1], records[2]])
    assert data.n_real == 3 and data.arrays["dimer_map_cnt"].shape == (3, 50, 25)
    assert (data.arrays["dimer_map_cnt"][0] == 0).all()
    assert np.isnan(data.arrays["dimer_map_sum_C"][0]).all()
    assert np.isnan(data.arrays["dimer_map_sum_x_partner"][0])
    assert data.arrays["dimer_map_n_spins"][0] == 0
    m = data.dimer_map(min_count=1)
    assert m["n_real"] == 2
    assert np.array_equal(m["count"], records[1]["dimer_map_cnt"] + records[2]["dimer_map_cnt"])
    both = EnsembleData.from_records(cfg, records[1:]).dimer_map(min_count=1)
    assert np.allclose(m["C"], both["C"], equal_nan=True)
    assert m["x_partner_mean"] == pytest.approx(both["x_partner_mean"])
    assert data.table()["mean_C_final"].notna().any()  # the ordinary analysis is unaffected


# --------------------------------------------------------------------------------------
# Real runs and stored states
# --------------------------------------------------------------------------------------
def test_dimer_map_of_a_run_and_from_its_stored_state(tiny_run, nn_run):
    cfg, out = tiny_run
    data = EnsembleData.load(out)
    m = data.dimer_map(min_count=1)
    assert m["n_real"] == cfg.n_realizations
    assert m["count"].sum() == data.arrays["dimer_map_cnt"].sum()
    assert np.nanmax(np.abs(m["C"])) <= 1.0 and np.nanmax(np.abs(m["P"])) <= 1.0
    # the map only reaches rho^2 + z^2 <= R^2: the corners of the square are empty
    assert m["count"][0, -1] == 0 and m["count"][-1, -1] == 0
    from_states = dimer_map_from_states(out)
    assert from_states["seed"].tolist() == [cfg.seeds[0]]
    first = load_record(seed_path(out, cfg.seeds[0]))
    for key in ("dimer_map_cnt", "dimer_ang_cnt"):
        assert np.array_equal(from_states[key][0], first[key]), key
    assert np.allclose(from_states["dimer_map_sum_C"][0], first["dimer_map_sum_C"])
    single = aggregate_dimer_map(from_states, min_count=1)
    assert np.isnan(single["sem_C"][np.isfinite(single["C"])]).all()  # one state: no error
    # dimers are closer than r_nn: the mean partner distance of Poisson points is (16/27)^(1/3) r_nn
    assert theory.MUTUAL_PAIR_MEAN_DISTANCE_X == pytest.approx(0.8400, abs=1e-4)
    big = EnsembleData.load(nn_run[1]).dimer_map()
    assert big["x_partner_mean"] == pytest.approx(theory.MUTUAL_PAIR_MEAN_DISTANCE_X, abs=0.03)
    with pytest.raises(FileNotFoundError, match="no stored states"):
        dimer_map_from_states(nn_run[1])


def test_structure_factor_from_states(tiny_run, nn_run):
    cfg, out = tiny_run
    table = structure_factor_from_states(out, n_shells=10, n_max=6)
    assert list(table.columns) == [
        "k_rnn_lo",
        "k_rnn_hi",
        "k_rnn_mean",
        "n_vectors",
        "wavelength_rnn",
        "S_final",
        "sem_S_final",
        "S_initial",
        "sem_S_initial",
    ]
    meta = table.attrs["meta"]
    assert meta["n_states"] == 1 and meta["seeds"] == [cfg.seeds[0]] and meta["n_max"] == 6
    assert meta["k_max_rnn"] == pytest.approx(2 * np.pi * 6 / cfg.box * cfg.r_nn)
    zero = table.iloc[0]
    assert zero["k_rnn_mean"] == 0.0 and zero["n_vectors"] == 1 and np.isnan(zero["wavelength_rnn"])
    first = load_record(seed_path(out, cfg.seeds[0]))
    assert zero["S_final"] == pytest.approx(float(first["fin_s0_unit"]))
    assert zero["S_initial"] == pytest.approx(float(first["init_s0_unit"]))
    shells = table.iloc[1:]
    assert (shells["n_vectors"] > 0).all() and shells["n_vectors"].sum() == meta["n_vectors"]
    assert (shells["k_rnn_lo"] < shells["k_rnn_mean"]).all()
    assert (shells["k_rnn_mean"] <= shells["k_rnn_hi"]).all()
    assert np.allclose(shells["wavelength_rnn"], 2 * np.pi / shells["k_rnn_mean"])
    assert table["sem_S_final"].isna().all()  # a single state has no spread
    weights = shells["n_vectors"] / shells["n_vectors"].sum()
    assert abs((shells["S_initial"] * weights).sum() - 1.0) < 0.15  # random spins: S = 1
    assert (shells["S_final"] >= 0).all()
    # a random subset per shell, reproducible through the seed
    kw = {"n_shells": 10, "n_max": 6, "max_per_shell": 12, "seed": 1}
    sub = structure_factor_from_states(out, **kw)
    assert (sub.iloc[1:]["n_vectors"] <= 12).all()
    assert sub.attrs["meta"]["n_vectors"] < meta["n_vectors"]
    assert sub.equals(structure_factor_from_states(out, **kw))
    with pytest.raises(FileNotFoundError, match="no stored states"):
        structure_factor_from_states(nn_run[1])


# --------------------------------------------------------------------------------------
# Outputs and figures
# --------------------------------------------------------------------------------------
def test_analyze_run_writes_the_dimer_map_and_structure_factor(tiny_run, tmp_path, small_config):
    cfg, out = tiny_run
    an = tmp_path / "an"
    written = analyze_run(out, an, n_boot=5, fit=False, error_budget=False)
    for name in ("dimer_map.csv", "dimer_angular.csv", "dimer_map.pdf", "structure_factor.csv"):
        assert (an / name).exists(), name
    assert (an / "structure_factor.pdf").exists()
    assert written["dimer_map"] == an / "dimer_map.csv" and "structure_factor" in written
    assert len(pd.read_csv(an / "dimer_map.csv")) == 1250
    angular = pd.read_csv(an / "dimer_angular.csv")
    assert (
        len(angular) == 42
        and angular["n"].sum() == EnsembleData.load(out).arrays["dimer_ang_cnt"].sum()
    )
    summary = json.loads((an / "summary.json").read_text())
    assert summary["derived"]["dimer_map"] == {
        "r_max_nn": 2.5,
        "shape": [50, 25],
        "x_edges": [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5],
        "cos_edges": [-1.0, -0.8, -0.5, -0.25, 0.25, 0.5, 0.8, 1.0],
        "min_count": 20,
    }
    dimer = summary["dimer_map"]
    assert dimer["n_real"] == cfg.n_realizations and len(dimer["angular"]) == 42
    assert dimer["angular"][3]["region"] == "transverse"
    sk = summary["structure_factor"]
    assert sk["n_states"] == 1 and sk["n_max"] == 12 and sk["S0_final"] > 0
    assert sk["k_first_rnn"] > 0 and sk["S_max_final"] >= sk["S_first_final"]
    text = (an / "summary.md").read_text()
    for needle in (
        "## Dimer-frame correlation map",
        "| x_ik \\ cos(theta) |",
        "[-0.25, 0.25) transverse",
        "## Spin structure factor of the stored states",
        "S(0): relaxed",
    ):
        assert needle in text, needle
    # without the diagnostics and without states nothing is written and nothing breaks
    run_ensemble(small_config, tmp_path / "plain", progress=False)
    plain = tmp_path / "plain_an"
    written = analyze_run(tmp_path / "plain", plain, n_boot=5, fit=False, error_budget=False)
    assert not (plain / "dimer_map.csv").exists() and not (plain / "structure_factor.csv").exists()
    assert "dimer_map" not in written and "structure_factor" not in written
    summary = json.loads((plain / "summary.json").read_text())
    assert summary["dimer_map"] is None and summary["structure_factor"] is None
    assert summary["derived"]["dimer_map"] is None
    text = (plain / "summary.md").read_text()
    assert "Dimer-frame" not in text and "structure factor" not in text
    # the structure factor can be switched off for runs with states
    off = tmp_path / "off"
    analyze_run(
        out, off, n_boot=5, fit=False, error_budget=False, plots=False, structure_factor=False
    )
    assert not (off / "structure_factor.csv").exists() and (off / "dimer_angular.csv").exists()


def test_dimer_map_and_structure_factor_figures():
    records, _ = _synthetic_map_records(n_real=4)
    data = EnsembleData.from_records(SimulationConfig(n_particles=300), records)
    dmap = data.dimer_map(min_count=1)
    fig = figure_dimer_map(dmap)
    assert len(fig.axes) == 4  # two maps and their colour bars
    images = [ax.images[0] for ax in fig.axes[:2]]
    assert all(im.get_clim() == (-0.6, 0.6) for im in images)
    assert images[0].get_array().shape == (50, 50)  # mirrored in rho
    assert tuple(images[0].get_extent()) == (-2.5, 2.5, -2.5, 2.5)
    texts = [t.get_text() for t in fig.axes[0].texts]
    assert "i" in texts and "j" in texts and any("realizations" in t for t in texts)
    assert fig.axes[0].get_title(loc="left").startswith("(a)")
    assert figure_dimer_map(dmap, vlim=0.3).axes[0].images[0].get_clim() == (-0.3, 0.3)
    # a map without a known partner distance still draws spin i
    dmap["x_partner_mean"] = float("nan")
    texts = [t.get_text() for t in figure_dimer_map(dmap).axes[0].texts]
    assert "i" in texts and "j" not in texts
    table = pd.DataFrame(
        {
            "k_rnn_lo": [0.0, 0.0, 0.1],
            "k_rnn_hi": [0.0, 0.1, 0.2],
            "k_rnn_mean": [0.0, 0.08, 0.15],
            "n_vectors": [1, 3, 9],
            "wavelength_rnn": [np.nan, 78.5, 41.9],
            "S_final": [0.4, 0.5, 0.7],
            "sem_S_final": [0.05, 0.04, np.nan],
            "S_initial": [1.1, 0.9, 1.0],
            "sem_S_initial": [0.1, 0.1, 0.1],
        }
    )
    fig = figure_structure_factor(table)
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert "relaxed" in labels and "initial (random)" in labels
    assert fig.axes[0].get_ylim()[0] == 0.0


# --------------------------------------------------------------------------------------
# Dimer-field test: aggregation, outputs, states fallback
# --------------------------------------------------------------------------------------
def _synthetic_field_records(n_real: int = 4, seed: int = 1) -> list[dict]:
    rng = np.random.default_rng(seed)
    records, _ = _synthetic_map_records(n_real, seed)
    for r in records:
        cnt = rng.integers(5, 50, size=6)
        agree = (cnt * rng.uniform(0.5, 1.0, size=6)).astype(np.int64)
        r["dimer_field_cnt"] = cnt
        r["dimer_field_agree"] = agree
        r["dimer_field_sum_cos"] = cnt * rng.uniform(0.0, 0.9, size=6)
        r["dimer_ang_sum_Cpred"] = r["dimer_ang_cnt"] * rng.uniform(-0.5, 0.5, size=(6, 7))
        r["dimer_field_n"] = np.array(int(cnt.sum() + 3))  # three pairs below the first edge
        r["dimer_field_n_agree"] = np.array(int(agree.sum() + 2))
        r["dimer_field_sigma"] = np.array(1.0)
        r["dimer_field_gamma"] = np.array(3.0)
    return records


def test_dimer_field_table_aggregates_per_realization_fractions():
    records = _synthetic_field_records(4)
    cfg = SimulationConfig(n_particles=300)
    data = EnsembleData.from_records(cfg, records)
    t = data.dimer_field_table()
    assert list(t.columns) == [
        "x_lo",
        "x_hi",
        "n",
        "n_agree",
        "agree",
        "sem_agree",
        "agree_pooled",
        "cos",
        "sem_cos",
        "n_real",
    ]
    assert len(t) == 7 and t["x_lo"].tolist()[:6] == [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
    assert t["x_hi"].tolist()[:6] == [0.75, 1.0, 1.25, 1.5, 2.0, 2.5]
    assert t["x_lo"].iloc[-1] == 0.0 and t["x_hi"].iloc[-1] == 2.5  # all pairs within R
    cnt = np.array([r["dimer_field_cnt"] for r in records])
    agree = np.array([r["dimer_field_agree"] for r in records])
    frac = agree / cnt
    assert np.allclose(t["agree"].iloc[:6], frac.mean(axis=0))
    assert np.allclose(t["sem_agree"].iloc[:6], frac.std(axis=0, ddof=1) / 2.0)
    assert np.allclose(t["agree_pooled"].iloc[:6], agree.sum(axis=0) / cnt.sum(axis=0))
    assert t["n"].iloc[:6].tolist() == cnt.sum(axis=0).tolist()
    assert t["n_agree"].iloc[:6].tolist() == agree.sum(axis=0).tolist()
    cos = np.array([r["dimer_field_sum_cos"] for r in records]) / cnt
    assert np.allclose(t["cos"].iloc[:6], cos.mean(axis=0))
    n_all = np.array([int(r["dimer_field_n"]) for r in records])
    a_all = np.array([int(r["dimer_field_n_agree"]) for r in records])
    assert t["n"].iloc[-1] == n_all.sum() and t["n_agree"].iloc[-1] == a_all.sum()
    assert t["agree"].iloc[-1] == pytest.approx((a_all / n_all).mean())
    assert t["agree_pooled"].iloc[-1] == pytest.approx(a_all.sum() / n_all.sum())
    assert np.isnan(t["cos"].iloc[-1]) and np.isnan(t["sem_cos"].iloc[-1])
    assert (t["n_real"] == 4).all()
    meta = t.attrs["meta"]
    assert meta["sigma"] == 1.0 and meta["gamma"] == 3.0 and meta["r_max_nn"] == 2.5
    assert meta["n_pairs"] == n_all.sum() and meta["n_agree"] == a_all.sum()
    assert meta["agree_pooled"] == pytest.approx(a_all.sum() / n_all.sum())
    assert meta["agree_mean"] == pytest.approx((a_all / n_all).mean()) and meta["n_real"] == 4
    assert tabulate_dimer_field(data.arrays).equals(t)
    # the angular table gains the dimer-only prediction C_pred
    ang = data.dimer_angular_table()
    assert list(ang.columns)[-2:] == ["C_pred", "sem_C_pred"]
    acnt = np.array([r["dimer_ang_cnt"] for r in records], dtype=float)
    cpred = np.array([r["dimer_ang_sum_Cpred"] for r in records])
    with np.errstate(invalid="ignore", divide="ignore"):
        per_real = np.where(acnt > 0, cpred / acnt, np.nan)
    with warnings.catch_warnings():  # cells empty in every realization are all-NaN
        warnings.simplefilter("ignore", RuntimeWarning)
        expected = np.nanmean(per_real, axis=0).ravel()
    assert np.allclose(ang["C_pred"].to_numpy(), expected, equal_nan=True)
    assert np.isnan(ang["C_pred"][ang["n"] == 0]).all()
    assert np.isfinite(ang["C_pred"][ang["n"] > 0]).all()
    # records without the test raise; mixed ensembles treat the missing keys as empty
    with pytest.raises(ValueError, match="dimer-field"):
        tabulate_dimer_field({"seed": np.arange(2), "dimer_map_cnt": np.zeros((2, 50, 25))})
    old = {
        k: v
        for k, v in records[0].items()
        if not (k.startswith("dimer_field") or k == "dimer_ang_sum_Cpred")
    }
    with pytest.warns(UserWarning, match="dimer_field_cnt \\(3 of 4\\)"):
        mixed = EnsembleData.from_records(cfg, [old, *records[1:]])
    tm = mixed.dimer_field_table()
    assert (tm["n_real"] == 3).all() and tm["n"].iloc[0] == cnt[1:, 0].sum()
    assert np.allclose(tm["agree"].iloc[:6], frac[1:].mean(axis=0))
    assert tm.attrs["meta"]["sigma"] == 1.0 and tm.attrs["meta"]["n_pairs"] == n_all[1:].sum()
    assert "C_pred" in mixed.dimer_angular_table().columns


def test_analyze_run_writes_dimer_field_and_energy_budget(tiny_run, tmp_path):
    cfg, out = tiny_run
    an = tmp_path / "an"
    written = analyze_run(out, an, n_boot=5, fit=False, error_budget=False, plots=False)
    for key in ("dimer_field", "energy_budget", "energy_budget_json"):
        assert written[key].exists(), key
    field = pd.read_csv(an / "dimer_field.csv")
    assert len(field) == 7 and (field["n_real"] == cfg.n_realizations).all()
    assert (field["agree"] > 0.55).all() and (field["cos"].iloc[:6] > 0.0).all()  # relaxed states
    assert field["n"].iloc[-1] >= field["n"].iloc[:6].sum()
    angular = pd.read_csv(an / "dimer_angular.csv")
    assert {"C_pred", "sem_C_pred"} <= set(angular.columns)
    data = EnsembleData.load(out)
    assert field["n"].iloc[-1] == data.arrays["dimer_field_n"].sum()
    budget = pd.read_csv(an / "energy_budget.csv")
    assert len(budget) == 1 and 0.0 < budget["dimer_share"].iloc[0] < 1.0
    assert budget["seed"].iloc[0] == cfg.seeds[0]
    stats = json.loads((an / "energy_budget.json").read_text())
    assert stats["n_states"] == 1 and len(stats["rows"]) == 1
    assert stats["meta"]["two_body_factor"] == 2.0 and stats["meta"]["coupling"] == "gr"
    summary = json.loads((an / "summary.json").read_text())
    assert summary["dimer_map"]["source"] == "records" and summary["dimer_map"]["n_states"] is None
    df = summary["dimer_field"]
    assert df["source"] == "records" and df["n_real"] == cfg.n_realizations and len(df["bins"]) == 7
    assert df["sigma"] == 1.0 and df["gamma"] == 3.0 and 0.5 < df["agree_pooled"] < 1.0
    assert summary["energy_budget"]["frustration_index"]["mean"] == pytest.approx(
        budget["frustration_index"].iloc[0]
    )
    text = (an / "summary.md").read_text()
    for needle in (
        "## Dimer-field test",
        "| all within 2.5 r_nn |",
        "Dimer-only prediction of the angular table",
        "## Energy budget of the stored states",
        "frustration index",
        "f = max(1, |gamma - 1|) = 2",
        "(n=",
        "Computed from the per-realization records.",
    ):
        assert needle in text, needle
    # a run whose records predate the map gets the map and the field test from its stored state
    old = tmp_path / "old"
    shutil.copytree(out, old)
    for path in (old / "seeds").glob("seed_*.npz"):
        rec = {k: v for k, v in load_record(path).items() if not k.startswith("dimer_")}
        np.savez(path, **rec)
    an2 = tmp_path / "an2"
    written = analyze_run(old, an2, n_boot=5, fit=False, error_budget=False, plots=False)
    assert written["dimer_map"].exists() and written["dimer_field"].exists()
    s2 = json.loads((an2 / "summary.json").read_text())
    assert s2["dimer_map"]["source"] == "states" and s2["dimer_map"]["n_states"] == 1
    assert s2["dimer_field"]["source"] == "states" and s2["dimer_field"]["n_real"] == 1
    assert s2["dimer_map"]["n_real"] == 1 and len(pd.read_csv(an2 / "dimer_angular.csv")) == 42
    assert "stored states of the run" in (an2 / "summary.md").read_text()
    first = load_record(seed_path(out, cfg.seeds[0]))
    expected = first["dimer_field_n_agree"] / first["dimer_field_n"]
    assert s2["dimer_field"]["agree_pooled"] == pytest.approx(expected)
    # records with the map but without the field test: C_pred and the field come from the state
    partial = tmp_path / "partial"
    shutil.copytree(out, partial)
    for path in (partial / "seeds").glob("seed_*.npz"):
        rec = {
            k: v
            for k, v in load_record(path).items()
            if not (k.startswith("dimer_field") or k == "dimer_ang_sum_Cpred")
        }
        np.savez(path, **rec)
    an3 = tmp_path / "an3"
    analyze_run(partial, an3, n_boot=5, fit=False, error_budget=False, plots=False)
    s3 = json.loads((an3 / "summary.json").read_text())
    assert s3["dimer_map"]["source"] == "records" and s3["dimer_map"]["n_real"] == 6
    assert s3["dimer_field"]["source"] == "states" and s3["dimer_field"]["n_real"] == 1
    mixed = pd.read_csv(an3 / "dimer_angular.csv")
    assert "C_pred" in mixed.columns and mixed["n"].sum() == angular["n"].sum()
    # the state-based diagnostics can be switched off
    an4 = tmp_path / "an4"
    analyze_run(
        old,
        an4,
        n_boot=5,
        fit=False,
        error_budget=False,
        plots=False,
        energy_budget=False,
        dimer_map_states=False,
        structure_factor=False,
    )
    for name in ("dimer_map.csv", "dimer_field.csv", "energy_budget.csv", "structure_factor.csv"):
        assert not (an4 / name).exists(), name
    s4 = json.loads((an4 / "summary.json").read_text())
    assert s4["dimer_map"] is None and s4["dimer_field"] is None and s4["energy_budget"] is None
    text = (an4 / "summary.md").read_text()
    assert "Dimer-field" not in text and "Energy budget" not in text


def test_structure_factor_comparison_across_runs(tiny_run, small_config, tmp_path):
    cfg, out = tiny_run
    other = tmp_path / "other"
    run_ensemble(small_config.replace(save_states=2, n_realizations=2), other, progress=False)
    none = tmp_path / "none"
    run_ensemble(small_config.replace(n_realizations=1), none, progress=False)
    cmp_dir = tmp_path / "cmp"
    with pytest.warns(UserWarning, match="no stored states"):
        written = structure_factor_comparison(
            [out, other, none], cmp_dir, labels=["tiny", "other", "none"], n_max=6, n_shells=10
        )
    table = pd.read_csv(written["structure_factor_vs_cutoff"])
    assert table["label"].tolist() == ["tiny", "other"]
    assert table["n_states"].tolist() == [1, 2]
    assert table["n_particles"].tolist() == [cfg.n_particles, small_config.n_particles]
    for column in (
        "r_cut_nn",
        "box_rnn",
        "k_min_rnn",
        "S0",
        "sem_S0",
        "S0_initial",
        "k1_rnn",
        "S1",
        "sem_S1",
        "k2_rnn",
        "S2",
        "sem_S2",
        "k3_rnn",
        "S3",
        "sem_S3",
        "k_max_rnn",
        "S_max",
        "sem_S_max",
        "n_max",
        "n_shells",
    ):
        assert column in table.columns, column
    first = load_record(seed_path(out, cfg.seeds[0]))
    assert table.loc[0, "S0"] == pytest.approx(float(first["fin_s0_unit"]))
    assert np.isnan(table.loc[0, "sem_S0"]) and np.isfinite(table.loc[1, "sem_S0"])
    assert (table["k1_rnn"] < table["k2_rnn"]).all() and (table["k2_rnn"] < table["k3_rnn"]).all()
    assert (table["S_max"] >= table[["S1", "S2", "S3"]].max(axis=1) - 1e-12).all()
    assert table.loc[0, "k_min_rnn"] == pytest.approx(2 * np.pi * cfg.r_nn / cfg.box)
    assert table.loc[0, "box_rnn"] == pytest.approx(cfg.box / cfg.r_nn)
    assert (table["n_max"] == 6).all() and (table["n_shells"] == 10).all()
    reference = structure_factor_from_states(out, n_shells=10, n_max=6)
    shells = reference[reference["k_rnn_mean"] > 0]
    assert table.loc[0, "S1"] == pytest.approx(shells["S_final"].iloc[0])
    assert table.loc[0, "k1_rnn"] == pytest.approx(shells["k_rnn_mean"].iloc[0])
    assert table.loc[0, "S_max"] == pytest.approx(shells["S_final"].max())
    assert (cmp_dir / "structure_factor_comparison.pdf").exists()
    assert written["structure_factor_comparison.pdf"].stat().st_size > 0
    text = (cmp_dir / "structure_factor_comparison.md").read_text()
    assert "| tiny |" in text and "| other |" in text and "| none |" not in text
    assert "S(k_1) at k_1 r_nn" in text
    # labels default to the directory names; mismatching labels are refused; no states at all fails
    with pytest.warns(UserWarning):
        again = structure_factor_comparison([other, none], tmp_path / "cmp2", n_max=6, n_shells=10)
    assert pd.read_csv(again["structure_factor_vs_cutoff"])["label"].tolist() == ["other"]
    with pytest.raises(ValueError, match="labels"):
        structure_factor_comparison([out], tmp_path / "x", labels=["a", "b"])
    with pytest.warns(UserWarning), pytest.raises(FileNotFoundError, match="stored states"):
        structure_factor_comparison([none], tmp_path / "y")
    fig = figure_structure_factor_comparison({"a": reference, "b": reference})
    labels = fig.axes[0].get_legend_handles_labels()[1]
    assert sum("S(0)" in lab for lab in labels) == 2 and fig.axes[0].get_xscale() == "log"
    assert fig.axes[0].get_ylim()[0] == 0.0
