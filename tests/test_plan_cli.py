"""The simulation plan (structure, status, documentation) and the command line interface."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from grspin import __version__
from grspin.cli import main
from grspin.ensemble import run_ensemble, seed_path
from grspin.plan import (
    BASE_SEED,
    PRIORITIES,
    build_plan,
    entry_status,
    plan_cost,
    plan_markdown,
    plan_status,
)

PLAN_DOC = Path(__file__).resolve().parent.parent / "docs" / "SIMULATION_PLAN.md"


# --------------------------------------------------------------------------------------
# Plan structure
# --------------------------------------------------------------------------------------
def test_plan_entries_are_unique_valid_and_share_the_seed_convention():
    plan = build_plan()
    assert len(plan.ids) == len(set(plan.ids)) >= 20
    assert {e.group for e in plan.entries} == {"M", "S1", "S2", "S3", "S4", "S5", "S6"}
    assert all(e.priority in PRIORITIES and e.purpose and e.feeds for e in plan.entries)
    assert all(e.config.base_seed == BASE_SEED for e in plan.entries)
    assert all(
        e.config.n_bins % e.rebin == 0 for e in plan.entries
    )  # analysis re-binning is possible
    main_entry = plan.get("M")
    assert main_entry.config.nn_diagnostics and main_entry.config.n_particles == 5000
    assert main_entry.config.boundary == "periodic" and main_entry.config.r_cut_nn_resolved == 10.0
    with pytest.raises(KeyError):
        plan.get("nope")


def test_comparisons_refer_to_existing_paired_entries():
    plan = build_plan()
    assert len(plan.comparisons) >= 6
    for comp in plan.comparisons:
        assert len(comp.members) >= 2 and len({m.label for m in comp.members}) == len(comp.members)
        configs = []
        for m in comp.members:
            entry = plan.get(m.entry)
            configs.append(entry.config)
            assert m.max_realizations is None or m.max_realizations <= entry.n_realizations
        assert len({(c.n_bins, c.x_max) for c in configs}) == 1, (
            comp.id
        )  # required for paired differences
    spin = next(c for c in plan.comparisons if c.id == "cmp_spin_distribution")
    assert spin.members[0].entry == "M" and spin.members[0].max_realizations == 100


def test_selection_by_id_group_and_priority_level():
    plan = build_plan()
    assert [e.id for e in plan.select(["M"])] == ["M"]
    assert {e.id for e in plan.select(["S2"])} == {e.id for e in plan.entries if e.group == "S2"}
    required = {e.id for e in plan.select(["required"])}
    recommended = {e.id for e in plan.select(["recommended"])}
    everything = {e.id for e in plan.select(["all"])}
    assert required < recommended < everything == set(plan.ids)
    assert {e.id for e in plan.select(["optional"])} == everything
    assert [e.id for e in plan.select(["S1", "M"])] == ["M", "S1"]  # plan order, no duplicates
    with pytest.raises(ValueError, match="unknown plan target"):
        plan.select(["nope"])


def test_costs_are_positive_and_additive():
    plan = build_plan()
    for e in plan.entries:
        c = e.cost()
        assert c["seconds"] > 0 and c["memory_mb"] > 0
        assert c["total_core_seconds"] == pytest.approx(c["seconds"] * e.n_realizations)
    req, everything = plan_cost(plan.select(["required"])), plan_cost(plan.select(["all"]))
    assert 0 < req["core_hours"] < everything["core_hours"]
    assert everything["n_realizations"] == sum(e.n_realizations for e in plan.entries)


def test_smoke_plan_is_tiny_but_complete():
    smoke = build_plan(smoke=True)
    assert smoke.smoke and len(smoke.comparisons) >= 5
    assert {e.group for e in smoke.entries} == {"M", "S1", "S2", "S3", "S4", "S5", "S6"}
    for e in smoke.entries:
        assert e.config.n_particles <= 300 and e.config.n_sweeps <= 10 and e.n_realizations <= 3


def test_committed_plan_document_is_up_to_date():
    """`docs/SIMULATION_PLAN.md` is generated; only the ticked boxes may differ."""

    def normalize(text: str) -> str:
        return re.sub(r"- \[[xX]\]", "- [ ]", text)

    assert PLAN_DOC.exists(), (
        "generate it with: grspin plan markdown --write docs/SIMULATION_PLAN.md"
    )
    assert normalize(PLAN_DOC.read_text()) == normalize(plan_markdown(build_plan(False))), (
        "docs/SIMULATION_PLAN.md is stale; regenerate: grspin plan markdown --write docs/SIMULATION_PLAN.md"
    )


# --------------------------------------------------------------------------------------
# Status follows the files on disk
# --------------------------------------------------------------------------------------
def test_plan_status_follows_the_disk(tmp_path):
    root = tmp_path / "res"
    assert main(["plan", "run", "M", "--smoke", "--root", str(root), "--quiet"]) == 0
    plan = build_plan(smoke=True)
    status = {s.entry.id: s for s in plan_status(plan, root)}
    assert status["M"].state == "done" and status["M"].finished == status["M"].total == 3
    assert status["S1"].state == "todo" and status["S1"].finished == 0

    seed_path(root / "M", plan.get("M").config.seeds[1]).unlink()
    assert entry_status(plan.get("M"), root).state == "partial"
    assert entry_status(plan.get("M"), root).finished == 2

    s1 = plan.get("S1")
    run_ensemble(s1.config.replace(n_sweeps=5, snapshots=(1, 2, 3, 5)), root / "S1", progress=False)
    assert entry_status(s1, root).state == "stale-config"  # the plan no longer matches what was run


def test_cli_plan_list_show_status_and_dry_run(tmp_path, capsys):
    root = str(tmp_path / "res")
    assert main(["plan", "list"]) == 0
    listing = capsys.readouterr().out
    assert "S3_N20000_open" in listing and "core-hours" in listing and "cmp_controls" in listing
    assert main(["plan", "show", "S6"]) == 0
    assert "isotropic_afm" in capsys.readouterr().out
    assert main(["plan", "status", "--smoke", "--root", root]) == 0
    assert "0/15 entries done" in capsys.readouterr().out
    assert main(["plan", "run", "S2", "--smoke", "--root", root, "--dry-run"]) == 0
    assert "S2_rc3" in capsys.readouterr().out and not Path(root).exists()
    assert main(["plan", "run", "bogus", "--smoke", "--root", root]) == 2
    assert "unknown plan target" in capsys.readouterr().err


def test_cli_plan_analyze_runs_entries_and_comparisons_and_skips_unfinished(tmp_path, capsys):
    root = str(tmp_path / "res")
    base = ["--smoke", "--root", root]
    assert main(["plan", "run", "M", "S6", *base, "--quiet"]) == 0
    capsys.readouterr()
    assert (
        main(["plan", "analyze", "M", "cmp_controls", "cmp_cutoff", "--n-boot", "10", *base]) == 0
    )
    out = capsys.readouterr().out
    assert "analysed M" in out and "compared cmp_controls" in out and "skip cmp_cutoff" in out
    assert (Path(root) / "M" / "analysis" / "summary.md").exists()
    assert (Path(root) / "_comparisons" / "cmp_controls" / "comparison_landmarks.csv").exists()
    assert main(["plan", "analyze", "S1", "--n-boot", "10", *base]) == 0
    assert "skip S1: 0/3" in capsys.readouterr().out


def test_cli_plan_run_includes_the_protocol_check_and_its_analysis(tmp_path, capsys):
    root = str(tmp_path / "res")
    base = ["--smoke", "--root", root]
    assert main(["plan", "run", "S5_protocol_short", *base, "--quiet"]) == 0
    assert main(["plan", "analyze", "S5_protocol_short", *base]) == 0
    assert (Path(root) / "S5_protocol_short" / "analysis" / "protocol_summary.csv").exists()
    assert "S5_protocol_short" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# Command line: run / analyze / compare / protocol-check / benchmark
# --------------------------------------------------------------------------------------
def test_cli_version(capsys):
    with pytest.raises(SystemExit) as info:
        main(["--version"])
    assert info.value.code == 0 and f"grspin {__version__}" in capsys.readouterr().out
    assert main([]) == 0  # no command: prints help


def test_cli_run_analyze_and_compare(tmp_path, capsys):
    common = [
        "-N",
        "300",
        "-R",
        "3",
        "--n-sweeps",
        "6",
        "--n-bins",
        "10",
        "--x-max",
        "3",
        "--quiet",
    ]
    a, b = tmp_path / "a", tmp_path / "b"
    assert main(["run", *common, "--out", str(a), "--snapshots", "2,6", "--nn-diagnostics"]) == 0
    assert main(["run", *common, "--coupling", "dipolar", "--out", str(b)]) == 0
    capsys.readouterr()
    assert (
        main(["analyze", str(a), "--rebin", "2", "--n-boot", "10", "--formats", "pdf", "png"]) == 0
    )
    out = capsys.readouterr().out
    assert "Model-independent landmarks" in out
    analysis = a / "analysis"
    for name in (
        "summary.md",
        "summary.json",
        "ensemble_table.csv",
        "fit.json",
        "C_of_r.pdf",
        "C_of_r.png",
        "overview.pdf",
        "sweep_convergence.csv",
        "nn_pairs.csv",
        "nn_diagnostics.pdf",
    ):
        assert (analysis / name).exists(), name
    assert (
        main(
            [
                "analyze",
                str(b),
                "--no-fit",
                "--no-plots",
                "--n-boot",
                "5",
                "--out",
                str(tmp_path / "b_an"),
            ]
        )
        == 0
    )
    assert not (tmp_path / "b_an" / "fit.json").exists() and not list(
        (tmp_path / "b_an").glob("*.pdf")
    )

    cmp_dir = tmp_path / "cmp"
    assert (
        main(
            [
                "compare",
                f"{a}:2",
                str(b),
                "--labels",
                "gr",
                "dipolar",
                "--out",
                str(cmp_dir),
                "--n-boot",
                "5",
            ]
        )
        == 0
    )
    assert (cmp_dir / "comparison_landmarks.csv").exists() and (cmp_dir / "comparison.pdf").exists()
    assert list(cmp_dir.glob("paired_C_dipolar_minus_gr.csv"))
    assert main(["compare", str(a), str(b), "--labels", "only-one", "--out", str(cmp_dir)]) == 2


def test_cli_reports_errors_with_exit_status_2(tmp_path, capsys):
    assert main(["analyze", str(tmp_path / "missing")]) == 2
    assert "error:" in capsys.readouterr().err
    out = tmp_path / "run"
    base = ["-N", "300", "-R", "2", "--n-bins", "10", "--x-max", "3", "--quiet", "--out", str(out)]
    assert main(["run", *base, "--n-sweeps", "4"]) == 0
    assert main(["run", *base, "--n-sweeps", "5"]) == 2
    assert "different configuration" in capsys.readouterr().err
    assert (
        main(["run", "-N", "300", "--r-cut-nn", "50", "--out", str(tmp_path / "x"), "--quiet"]) == 2
    )
    assert "r_cut <= L/2" in capsys.readouterr().err


def test_cli_run_with_analysis_flag_and_extension(tmp_path):
    out = tmp_path / "run"
    args = [
        "run",
        "-N",
        "300",
        "--n-sweeps",
        "4",
        "--n-bins",
        "10",
        "--x-max",
        "3",
        "--quiet",
        "--out",
        str(out),
    ]
    assert main([*args, "-R", "2", "--analyze", "--n-boot", "5"]) == 0
    assert (out / "analysis" / "summary.md").exists()
    assert main([*args, "-R", "4"]) == 0  # extension: two more realizations
    assert len(list((out / "seeds").glob("seed_*.npz"))) == 4


def test_cli_protocol_check_and_benchmark(tmp_path, capsys):
    out = tmp_path / "proto"
    assert (
        main(
            [
                "protocol-check",
                "-N",
                "200",
                "-R",
                "2",
                "--n-sweeps",
                "5",
                "--n-bins",
                "10",
                "--x-max",
                "3",
                "--anneal-n-temperatures",
                "3",
                "--anneal-sweeps-per-temperature",
                "2",
                "--quiet",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    printed = capsys.readouterr().out
    assert "sequential" in printed and "anneal" in printed
    assert main(["benchmark", "-N", "300", "--n-sweeps", "4", "--repeats", "2"]) == 0
    assert "measured/estimate" in capsys.readouterr().out
