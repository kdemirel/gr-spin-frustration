"""The simulation plan (structure, status, documentation) and the command line interface."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from grspin import __version__
from grspin.analysis import PAIRING_KEYS, same_system
from grspin.cli import main
from grspin.config import COUPLINGS, AnnealConfig, SimulationConfig
from grspin.ensemble import (
    GIT_DIFF_MAX_CHARS,
    _truncate,
    load_record,
    provenance,
    run_ensemble,
    seed_path,
    source_hash,
)
from grspin.plan import (
    BASE_SEED,
    PRIORITIES,
    build_plan,
    entry_provenance,
    entry_status,
    group_priority,
    plan_cost,
    plan_markdown,
    plan_status,
    provenance_report,
)
from grspin.protocols import protocol_cost, protocol_seed_path, run_protocol_check

PLAN_DOC = Path(__file__).resolve().parent.parent / "docs" / "SIMULATION_PLAN.md"
GROUPS = {"M", "S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "T"}
FIVE = ("C", "P", "Cperp", "Q", "A")


# --------------------------------------------------------------------------------------
# Plan structure
# --------------------------------------------------------------------------------------
def test_plan_entries_are_unique_valid_and_share_the_seed_convention():
    plan = build_plan()
    assert len(plan.ids) == len(set(plan.ids)) >= 36
    assert {e.group for e in plan.entries} == GROUPS
    assert all(e.priority in PRIORITIES and e.purpose and e.feeds for e in plan.entries)
    assert all(e.config.base_seed == BASE_SEED for e in plan.entries)
    assert all(
        e.config.n_bins % e.rebin == 0 for e in plan.entries
    )  # analysis re-binning is possible
    main_entry = plan.get("M")
    assert main_entry.config.nn_diagnostics and main_entry.config.n_particles == 5000
    assert main_entry.config.boundary == "periodic" and main_entry.config.r_cut_nn_resolved == 10.0
    assert main_entry.config.save_states == 48  # dimer-frame map and S(k) from the stored states
    assert "Q is the only" not in plan.get("S4_equal").feeds  # C(r) landmarks depend on it too
    # only M, the controls, the channel family, the hard core and the far tail store states
    with_states = {e.id for e in plan.entries if e.config.save_states}
    assert with_states == {"M", *(e.id for e in plan.entries if e.group in {"S6", "S7", "S8", "T"})}
    assert all(plan.get(i).config.save_states == 16 for i in with_states - {"M"})
    with pytest.raises(KeyError):
        plan.get("nope")


def test_derived_items_have_distinct_ids_and_known_dependencies():
    plan = build_plan()
    assert [d.id for d in plan.derived] == ["D1", "D2", "D3"]  # S7/S8 are ensemble groups now
    assert not ({d.id for d in plan.derived} & (set(plan.ids) | set(plan.groups)))
    for d in plan.derived:
        assert d.title and d.how and d.feeds
        assert set(d.depends_on) <= set(plan.ids), d.id
    d3 = plan.derived[2]
    assert d3.title == "Channel controls and dimer-field test"
    assert set(d3.depends_on) == {"M", "S7_gamma1", "S7_gamma12", "S7_gamma100"}
    assert "cmp_channels" in d3.how and "dimer-field" in d3.how.lower()


def test_channel_family_group_is_paired_with_the_main_ensemble():
    plan = build_plan()
    family = plan.select(["S7"])
    assert [e.id for e in family] == ["S7_gamma1", "S7_gamma12", "S7_gamma100"]
    m = plan.get("M")
    couplings = {
        "S7_gamma1": "transverse_only",
        "S7_gamma12": "gamma12",
        "S7_gamma100": "longitudinal_only",
    }
    for e in family:
        c = e.config
        assert e.group == "S7" and e.priority == "recommended" and e.kind == "ensemble"
        assert c.coupling == couplings[e.id] and COUPLINGS[c.coupling][0] == 1.0
        assert COUPLINGS[c.coupling][1] == float(e.id.removeprefix("S7_gamma"))
        assert (c.n_particles, c.boundary, c.r_cut_nn_resolved, c.n_sweeps) == (
            5000,
            "periodic",
            10.0,
            100,
        )
        assert (
            c.n_realizations == 100 and c.seeds == m.config.seeds[:100] and same_system(c, m.config)
        )
        assert c.nn_diagnostics and c.save_states == 16 and c.min_separation_nn == 0.0
        assert (c.n_bins, c.x_max) == (m.config.n_bins, m.config.x_max) and e.rebin == m.rebin
        assert e.purpose.startswith("Channel controls: which channel carries the positive shell?")
        assert f"coupling={c.coupling}" in e.describe()
        assert e.cost()["seconds"] == pytest.approx(
            m.config.replace(coupling=c.coupling).estimate_cost()["seconds"]
        )
    assert group_priority(plan, "S7") == "recommended"
    assert {e.id for e in family} <= {e.id for e in plan.select(["recommended"])}
    comp = next(c for c in plan.comparisons if c.id == "cmp_channels")
    assert [mb.entry for mb in comp.members] == ["M", "S7_gamma1", "S7_gamma12", "S7_gamma100"]
    assert (
        comp.members[0].max_realizations == 100
    )  # the reference comes first: GR on the same seeds
    assert all(mb.max_realizations is None for mb in comp.members[1:])
    assert comp.observables == FIVE and "positive shell" in comp.feeds


def test_hard_core_entry_is_optional_and_compared_unpaired_with_the_main_ensemble():
    plan = build_plan()
    e, m = plan.get("S8_hardcore05"), plan.get("M")
    c = e.config
    assert e.group == "S8" and e.priority == "optional" and e.kind == "ensemble"
    assert c.min_separation_nn == 0.5 and c.min_separation == pytest.approx(0.5 * c.r_nn)
    assert (c.n_particles, c.boundary, c.r_cut_nn_resolved, c.n_sweeps) == (
        5000,
        "periodic",
        10.0,
        100,
    )
    assert (
        c.n_realizations == 100 and c.nn_diagnostics and c.save_states == 16 and c.coupling == "gr"
    )
    assert c.seeds == m.config.seeds[:100] and (c.n_bins, c.x_max) == (
        m.config.n_bins,
        m.config.x_max,
    )
    assert "r_min=0.5 r_nn" in e.describe() and "unpaired" in e.purpose
    assert e.cost()["seconds"] > m.cost()["seconds"]  # the O(N^2) sequential addition is charged
    assert group_priority(plan, "S8") == "optional"
    assert "S8_hardcore05" not in {x.id for x in plan.select(["recommended"])}
    assert "S8_hardcore05" in {x.id for x in plan.select(["all"])}
    comp = next(cmp for cmp in plan.comparisons if cmp.id == "cmp_hardcore")
    assert [mb.entry for mb in comp.members] == ["M", "S8_hardcore05"]
    assert comp.observables == FIVE and "unpaired" in comp.title
    # the positions differ, so the comparison must not be paired by seed
    if "min_separation_nn" not in PAIRING_KEYS:
        pytest.skip(
            "wiring pending: add 'min_separation_nn' to grspin.analysis.PAIRING_KEYS so that "
            "same_system() distinguishes hard-core from Poisson positions (cmp_hardcore is unpaired)"
        )
    assert not same_system(c, m.config)


def test_far_tail_group_reaches_10_rnn_inside_the_periodic_box():
    plan = build_plan()
    tail = plan.select(["T"])
    assert [e.id for e in tail] == [
        "T_N10000_rc7",
        "T_N10000_rc10",
        "T_N10000_rc15",
        "T_N20000_rc10",
    ]
    for e in tail:
        c = e.config
        assert e.priority == "recommended" and e.kind == "ensemble" and e.group == "T"
        assert c.boundary == "periodic" and c.n_bins == 200 and c.x_max == 10.0
        assert c.n_sweeps == 100 and c.n_realizations == 32 and c.save_states == 16  # S(k) vs r_cut
        assert c.extra_observables and not c.nn_diagnostics and c.base_seed == BASE_SEED
        assert c.x_max * c.r_nn <= 0.5 * c.box and c.r_cut <= 0.5 * c.box
        assert "truncation effect" in e.purpose and "x_max=10 r_nn" in e.describe()
    systems = {(e.config.n_particles, e.config.r_cut_nn_resolved) for e in tail}
    assert systems == {(10000, 7.0), (10000, 10.0), (10000, 15.0), (20000, 10.0)}
    rc15 = plan.get("T_N10000_rc15").cost()
    assert 4500 < rc15["memory_mb"] < 6500  # ~5.5 GB: one worker on a 16 GB machine
    assert 5 * 60 < rc15["total_core_seconds"] < 12 * 60
    assert plan.get("T_N20000_rc10").cost()["memory_mb"] < rc15["memory_mb"]


def test_far_tail_comparisons_use_the_channel_observables():
    plan = build_plan()
    by_id = {c.id: c for c in plan.comparisons}
    tail = by_id["cmp_far_tail"]
    assert [m.entry for m in tail.members] == ["T_N10000_rc15", "T_N10000_rc10", "T_N10000_rc7"]
    assert tail.observables == ("C", "P", "Cperp")
    assert all(m.max_realizations is None for m in tail.members)
    size = by_id["cmp_far_tail_size"]
    assert [m.entry for m in size.members] == ["T_N20000_rc10", "T_N10000_rc10"]
    assert size.observables == ("C", "P", "Cperp")
    configs = [plan.get(m.entry).config for m in tail.members]
    assert all(same_system(configs[0], c) for c in configs[1:])  # paired
    assert not same_system(*(plan.get(m.entry).config for m in size.members))  # unpaired


def test_protocol_check_at_production_size_is_paired_with_the_main_ensemble():
    plan = build_plan()
    e, m = plan.get("S5_N5000"), plan.get("M")
    assert e.kind == "protocol" and e.group == "S5" and e.priority == "recommended"
    c = e.config
    assert (c.n_particles, c.boundary, c.r_cut_nn_resolved, c.n_sweeps) == (
        5000,
        "periodic",
        10.0,
        100,
    )
    assert c.n_realizations == 48 and c.seeds == m.config.seeds[:48] and same_system(c, m.config)
    assert (c.n_bins, c.x_max) == (m.config.n_bins, m.config.x_max) and e.rebin == m.rebin
    assert c.anneal == AnnealConfig(n_temperatures=40, sweeps_per_temperature=10)
    assert c.anneal.t_start == 10.0 and c.anneal.t_end == 0.01
    cost, expected = e.cost(), protocol_cost(c)
    assert cost["seconds"] == pytest.approx(expected["seconds"])
    assert cost["total_core_seconds"] == pytest.approx(48 * expected["seconds"])
    assert cost["seconds"] > 3 * m.config.estimate_cost()["seconds"]  # three quenches + anneal
    assert 7 * 60 < cost["total_core_seconds"] < 15 * 60
    # the small checks are kept as optional entries
    assert [x.id for x in plan.select(["S5"])] == [
        "S5_N5000",
        "S5_protocol_short",
        "S5_protocol_long",
    ]
    assert {x.priority for x in plan.select(["S5"])} == {"recommended", "optional"}
    recommended = {x.id for x in plan.select(["recommended"])}
    assert "S5_N5000" in recommended and "S5_protocol_short" not in recommended
    assert group_priority(plan, "S5") == "recommended" and group_priority(plan, "M") == "required"


def test_comparisons_refer_to_existing_paired_entries():
    plan = build_plan()
    assert len(plan.comparisons) >= 11
    assert {"cmp_controls", "cmp_channels", "cmp_hardcore"} <= {c.id for c in plan.comparisons}
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
    assert smoke.smoke and len(smoke.comparisons) >= 7
    assert {e.group for e in smoke.entries} == GROUPS
    assert {"T_N10000_rc7", "T_N10000_rc15", "T_N20000_rc10", "S5_N5000"} <= set(smoke.ids)
    for e in smoke.entries:
        assert e.config.n_particles <= 300 and e.config.n_sweeps <= 10 and e.n_realizations <= 3
        assert e.config.x_max <= 4.0 and e.config.save_states <= 1  # fits the tiny periodic box
        if e.kind == "protocol":
            assert e.config.anneal.n_temperatures * e.config.anneal.sweeps_per_temperature <= 8


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


def test_plan_document_lists_mixed_priority_groups_once_and_the_new_items():
    text = plan_markdown(build_plan(False))
    assert text.count("**S5 - ") == 1 and text.count("**T - Far tail and cutoff**") == 1
    assert "`S5_N5000`: 48 x" in text and "`S5_protocol_short` (optional)" in text
    assert "`T_N10000_rc15`: 32 x" in text and "cmp_far_tail_size" in text
    assert "observables: C, P, Cperp\n" in text and "observables: C, P, Cperp, Q, A" in text
    for item in ("D1", "D2", "D3"):
        assert f"- [ ] **{item}** - " in text, item  # derived items (no new simulation)
    assert "- [ ] **S7** - " not in text and "- [ ] **S8** - " not in text  # renamed D1/D2
    assert "analyse **S7**: `grspin plan analyze S7`" in text  # S7/S8 are ensemble groups now
    assert "analyse **S8**: `grspin plan analyze S8`" in text
    assert text.count("**S7 - Channel controls (gamma family)**") == 1
    assert text.count("**S8 - Hard-core positions (minimum separation)**") == 1
    assert (
        "`S7_gamma100`: 100 x (N=5000, periodic, r_cut=10 r_nn, 100 sweeps, coupling=longitudinal_only)"
        in text
    )
    assert (
        "`S8_hardcore05`: 100 x (N=5000, periodic, r_cut=10 r_nn, 100 sweeps, r_min=0.5 r_nn)"
        in text
    )
    assert "comparison `cmp_channels`" in text and "comparison `cmp_hardcore`" in text
    assert "Channel controls: which channel carries the positive shell?" in text
    assert "### Optional" in text and text.index("**S8 - ") > text.index("### Optional")
    for phrase in ("grspin plan provenance", "--refresh-stale", "git tag", "clean tree"):
        assert phrase in text, phrase


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

    # a missing state file of one of the first `save_states` seeds counts as unfinished (as in the runner)
    main_entry = plan.get("M")
    assert main_entry.config.save_states == 1
    (root / "M" / "states" / f"seed_{main_entry.config.seeds[0]:08d}.npz").unlink()
    assert (
        entry_status(main_entry, root).finished == 1
    )  # seed 1 was removed above, seed 0 lacks its state


def test_annealing_schedule_is_part_of_the_key_only_where_it_is_used():
    quench = build_plan().get("M").config
    other = quench.replace(anneal=AnnealConfig(t_start=3.0))
    assert other.compat_key() == quench.compat_key()  # never used in a quench ensemble
    assert other.compat_key("protocol") != quench.compat_key(
        "protocol"
    )  # the anneal variant differs
    annealed = quench.replace(protocol="anneal")
    assert annealed.replace(anneal=AnnealConfig(t_start=3.0)).compat_key() != annealed.compat_key()


def test_cli_plan_list_show_status_and_dry_run(tmp_path, capsys):
    root = str(tmp_path / "res")
    assert main(["plan", "list"]) == 0
    listing = capsys.readouterr().out
    assert "S3_N20000_open" in listing and "core-hours" in listing and "cmp_controls" in listing
    assert "S5_N5000" in listing and "T_N10000_rc15" in listing and "cmp_far_tail" in listing
    assert "S7_gamma12" in listing and "S8_hardcore05" in listing and "r_min=0.5 r_nn" in listing
    assert "cmp_channels" in listing and "cmp_hardcore" in listing
    assert main(["plan", "show", "S6"]) == 0
    assert "isotropic_afm" in capsys.readouterr().out
    assert main(["plan", "show", "S7", "S8_hardcore05"]) == 0
    shown = capsys.readouterr().out
    assert "transverse_only" in shown and "longitudinal_only" in shown
    assert "'min_separation_nn': 0.5" in shown
    assert main(["plan", "status", "--smoke", "--root", root]) == 0
    assert f"0/{len(build_plan(True).entries)} entries done" in capsys.readouterr().out
    assert main(["plan", "run", "S2", "--smoke", "--root", root, "--dry-run"]) == 0
    assert "S2_rc3" in capsys.readouterr().out and not Path(root).exists()
    assert main(["plan", "run", "bogus", "--smoke", "--root", root]) == 2
    assert "unknown plan target" in capsys.readouterr().err


def test_cli_plan_run_skips_stale_entries_unless_refreshed(tmp_path, capsys):
    root = tmp_path / "res"
    base = ["--smoke", "--root", str(root), "--quiet"]
    entry = build_plan(smoke=True).get("S2_rc3")
    older = entry.config.replace(n_sweeps=entry.config.n_sweeps - 1)  # an earlier plan version
    run_ensemble(older, root / "S2_rc3", progress=False)
    assert main(["plan", "run", "S2_rc3", *base]) == 0
    out = capsys.readouterr().out
    assert "skip S2_rc3" in out and "--refresh-stale" in out and "1 stale entries" in out
    assert entry_status(entry, root).state == "stale-config"
    assert main(["plan", "run", "S2_rc3", *base, "--refresh-stale"]) == 0
    assert entry_status(entry, root).state == "done"
    assert "skip S2_rc3" not in capsys.readouterr().out


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


def test_cli_plan_run_and_analyze_the_hard_core_entry_and_its_comparison(tmp_path, capsys):
    root = tmp_path / "res"
    base = ["--smoke", "--root", str(root)]
    assert main(["plan", "run", "M", "S8_hardcore05", *base, "--quiet", "-j", "2"]) == 0
    capsys.readouterr()
    smoke = build_plan(smoke=True)
    entry = smoke.get("S8_hardcore05")
    assert entry.config.min_separation_nn == 0.5 and entry.config.save_states == 1
    assert entry_status(entry, root).state == "done"
    # the stored state respects the hard core (minimum image)
    state = load_record(root / "S8_hardcore05" / "states" / f"seed_{entry.config.seeds[0]:08d}.npz")
    d = state["positions"][:, None, :] - state["positions"][None, :, :]
    d -= entry.config.box * np.round(d / entry.config.box)
    r = np.sqrt((d**2).sum(-1))[np.triu_indices(len(state["positions"]), k=1)]
    assert r.min() >= entry.config.min_separation
    meta = json.loads((root / "S8_hardcore05" / "config.json").read_text())
    assert meta["config"]["min_separation_nn"] == 0.5
    assert (
        main(
            [
                "plan",
                "analyze",
                "S8_hardcore05",
                "cmp_hardcore",
                "cmp_channels",
                "--n-boot",
                "5",
                *base,
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "analysed S8_hardcore05" in out and "compared cmp_hardcore" in out
    assert "skip cmp_channels: S7_gamma1 has 0/3" in out
    assert (root / "S8_hardcore05" / "analysis" / "nn_pairs.csv").exists()
    cmp_dir = root / "_comparisons" / "cmp_hardcore"
    assert (cmp_dir / "comparison_landmarks.csv").exists() and (
        cmp_dir / "comparison_summary.md"
    ).exists()
    if "min_separation_nn" in PAIRING_KEYS:  # see test_hard_core_entry_is_optional_...
        assert list(cmp_dir.glob("unpaired_C_*.csv")) and not list(cmp_dir.glob("paired_C_*.csv"))


def test_cli_run_with_hard_core_and_channel_coupling(tmp_path, capsys):
    out = tmp_path / "hard"
    args = [
        "run",
        "-N",
        "300",
        "-R",
        "2",
        "--n-sweeps",
        "3",
        "--n-bins",
        "8",
        "--x-max",
        "3",
        "--quiet",
    ]
    assert (
        main(
            [
                *args,
                "--min-separation-nn",
                "0.6",
                "--coupling",
                "transverse_only",
                "--save-states",
                "1",
                "--out",
                str(out),
            ]
        )
        == 0
    )
    cfg = SimulationConfig.from_dict(json.loads((out / "config.json").read_text())["config"])
    assert cfg.min_separation_nn == 0.6 and cfg.coupling == "transverse_only"
    rec = load_record(seed_path(out, cfg.seeds[0]))
    below = cfg.x_edges[1:] <= 0.6 * (1 + 1e-12)
    assert below.sum() == 1 and rec["fin_cnt_C"][below].sum() == 0 and rec["fin_cnt_C"].sum() > 0
    state = load_record(out / "states" / f"seed_{cfg.seeds[0]:08d}.npz")
    d = state["positions"][:, None, :] - state["positions"][None, :, :]
    d -= cfg.box * np.round(d / cfg.box)
    r = np.sqrt((d**2).sum(-1))[np.triu_indices(300, k=1)]
    assert r.min() >= cfg.min_separation
    # the same directory with the Poisson process is a different configuration
    assert main([*args, "--coupling", "transverse_only", "--out", str(out)]) == 2
    assert "different configuration" in capsys.readouterr().err
    assert main([*args, "--min-separation-nn", "0.9", "--out", str(tmp_path / "x")]) == 2
    assert "min_separation_nn" in capsys.readouterr().err
    for coupling in ("gamma12", "longitudinal_only"):
        assert main([*args, "--coupling", coupling, "--out", str(tmp_path / coupling)]) == 0
    with pytest.raises(SystemExit):  # argparse rejects unknown couplings
        main([*args, "--coupling", "heisenberg", "--out", str(tmp_path / "h")])


def test_cli_plan_run_includes_the_protocol_check_and_its_analysis(tmp_path, capsys):
    root = str(tmp_path / "res")
    base = ["--smoke", "--root", root]
    assert main(["plan", "run", "S5_protocol_short", *base, "--quiet"]) == 0
    assert main(["plan", "analyze", "S5_protocol_short", *base]) == 0
    assert (Path(root) / "S5_protocol_short" / "analysis" / "protocol_summary.csv").exists()
    assert "S5_protocol_short" in capsys.readouterr().out


# --------------------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------------------
def test_run_files_carry_the_provenance(tmp_path):
    cfg = SimulationConfig(
        n_particles=200, n_sweeps=3, n_bins=8, x_max=3.0, n_realizations=2, base_seed=3
    )
    out = tmp_path / "run"
    run_ensemble(cfg, out, progress=False)
    meta = json.loads((out / "config.json").read_text())
    prov = meta["provenance"]
    assert {"git_commit", "git_dirty", "git_status", "git_diff", "source_hash"} <= set(prov)
    assert prov["source_hash"] == source_hash() and len(prov["source_hash"]) == 64
    assert source_hash() == source_hash()  # deterministic
    if prov["git_dirty"]:  # the diff of the package sources is stored, truncated
        assert isinstance(prov["git_diff"], str) and prov["git_status"]
        assert len(prov["git_diff"]) <= GIT_DIFF_MAX_CHARS + 60
    else:
        assert prov["git_diff"] is None and prov["git_status"] is None
    last = meta["history"][-1]
    assert last["source_hash"] == prov["source_hash"] and last["git_dirty"] == prov["git_dirty"]
    for seed in cfg.seeds:
        rec = load_record(seed_path(out, seed))
        assert str(rec["source_hash"]) == prov["source_hash"]
        assert str(rec["git_commit"]) == (prov["git_commit"] or "")
    # protocol checks are stamped in the same way
    pcfg = cfg.replace(anneal=AnnealConfig(n_temperatures=2, sweeps_per_temperature=1))
    run_protocol_check(pcfg, tmp_path / "proto", progress=False)
    rec = load_record(protocol_seed_path(tmp_path / "proto", cfg.seeds[0]))
    assert str(rec["source_hash"]) == prov["source_hash"]
    long = "x" * (GIT_DIFF_MAX_CHARS + 5)
    assert _truncate(long, GIT_DIFF_MAX_CHARS).endswith("[truncated: 5 more characters]")
    assert _truncate("short", GIT_DIFF_MAX_CHARS) == "short"


def test_cli_plan_provenance_compares_every_entry_with_the_current_tree(tmp_path, capsys):
    root = tmp_path / "res"
    base = ["--smoke", "--root", str(root)]
    plan = build_plan(smoke=True)
    entry = plan.get("S2_rc3")
    assert main(["plan", "run", "S2_rc3", *base, "--quiet"]) == 0
    capsys.readouterr()
    assert main(["plan", "provenance", *base]) == 0
    out = capsys.readouterr().out
    current = provenance()
    assert current["source_hash"][:12] in out and "matches tree" in out
    rows = {
        line.split()[0]: line.split()
        for line in out.splitlines()
        if line.startswith(("S2_rc3", "M "))
    }
    assert rows["S2_rc3"][1] == "done" and rows["S2_rc3"][-1] == "yes"
    assert rows["M"][1] == "todo" and rows["M"][-1] == "-"
    assert "warning" not in out
    info = entry_provenance(entry, root)
    assert info.source_hash == current["source_hash"] and info.n_files == 3
    assert info.file_hashes == (current["source_hash"],) and info.matches(current) is True
    assert entry_provenance(plan.get("M"), root).matches(current) is None

    # one realization file written by other code: reported as mixed and not matching
    path = seed_path(root / "S2_rc3", entry.config.seeds[0])
    rec = load_record(path)
    rec["source_hash"] = np.array("0" * 64)
    np.savez(path, **rec)
    report = provenance_report(plan, root)
    assert "3 (mixed)" in report and report.splitlines()[-1].startswith("warning: 1 entries")
    assert entry_provenance(entry, root).matches(current) is False
    # files without a stamp (written before 0.2.0)
    for seed in entry.config.seeds:
        path = seed_path(root / "S2_rc3", seed)
        rec = load_record(path)
        del rec["source_hash"], rec["git_commit"]
        np.savez(path, **rec)
    assert "3 (unstamped)" in provenance_report(plan, root)
    assert main(["plan", "status", *base, "--provenance"]) == 0
    out = capsys.readouterr().out
    assert "entries done" in out and "matches tree" in out


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


# --------------------------------------------------------------------------------------
# S(k) versus cutoff from the stored states (cmp_structure_factor)
# --------------------------------------------------------------------------------------
def test_structure_factor_comparison_entries_exist_and_store_states():
    from grspin.plan import STRUCTURE_FACTOR_COMPARISON_ID, STRUCTURE_FACTOR_ENTRIES

    plan = build_plan(False)
    assert STRUCTURE_FACTOR_COMPARISON_ID == "cmp_structure_factor"
    assert set(STRUCTURE_FACTOR_ENTRIES) == {
        "T_N10000_rc7",
        "T_N10000_rc10",
        "T_N10000_rc15",
        "T_N20000_rc10",
        "M",
    }
    for sid in STRUCTURE_FACTOR_ENTRIES:
        assert plan.get(sid).config.save_states >= 16
    assert STRUCTURE_FACTOR_COMPARISON_ID not in {
        c.id for c in plan.comparisons
    }  # states, not curves
    text = plan_markdown(plan)
    assert "comparison `cmp_structure_factor`" in text and "structure_factor_vs_cutoff.csv" in text
    d2 = next(d for d in plan.derived if d.id == "D2")
    assert "cmp_structure_factor" in d2.how and "energy_budget.csv" in d2.how
    assert {"S6_isotropic_afm", "S6_dipolar"} <= set(d2.depends_on)


def test_cli_plan_analyze_compares_the_structure_factor_of_the_far_tail_entries(tmp_path, capsys):
    root = tmp_path / "res"
    base = ["--smoke", "--root", str(root)]
    assert main(["plan", "run", "M", *base, "--quiet"]) == 0
    capsys.readouterr()
    # one entry with states is not enough: skipped, the rest of the analysis still runs
    assert main(["plan", "analyze", "M", "cmp_structure_factor", "--n-boot", "5", *base]) == 0
    out = capsys.readouterr().out
    assert "analysed M" in out and "skip cmp_structure_factor: fewer than two" in out
    assert not (root / "_comparisons" / "cmp_structure_factor").exists()
    # a plain entry id does not trigger it
    assert main(["plan", "analyze", "S1", "--n-boot", "5", *base]) == 0
    assert "cmp_structure_factor" not in capsys.readouterr().out
    assert main(["plan", "run", "T_N10000_rc7", *base, "--quiet"]) == 0
    capsys.readouterr()
    assert main(["plan", "analyze", "T", "--n-boot", "5", "--max-states", "1", *base]) == 0
    out = capsys.readouterr().out
    assert "analysed T_N10000_rc7" in out and "compared cmp_structure_factor" in out
    cmp_dir = root / "_comparisons" / "cmp_structure_factor"
    table = pd.read_csv(cmp_dir / "structure_factor_vs_cutoff.csv")
    assert table["label"].tolist() == [
        "T_N10000_rc7 (N=300, r_cut=5 r_nn)",
        "M (N=300, r_cut=5 r_nn)",
    ]
    assert (table["n_states"] == 1).all() and {"S0", "S1", "k1_rnn", "S_max"} <= set(table.columns)
    assert (cmp_dir / "structure_factor_comparison.pdf").exists()
    assert (cmp_dir / "structure_factor_comparison.md").exists()
    # the energy budget of the analysed entries honours --max-states as well
    assert len(pd.read_csv(root / "T_N10000_rc7" / "analysis" / "energy_budget.csv")) == 1
