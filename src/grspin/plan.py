"""The simulation plan as code: every run needed for the manuscript, its purpose and its cost.

The plan is data (:func:`build_plan`); the documentation (``docs/SIMULATION_PLAN.md``), the status
report and the runner (``grspin plan ...``) are all generated from it, so they cannot drift apart.

Conventions shared by all sets
------------------------------
* spins ``U(0, S)`` distributed magnitudes, GR coupling, sequential greedy relaxation of 100 sweeps,
  periodic boundaries (unless the set studies the boundary), density ``rho = 1`` (irrelevant: all
  distances are in units of ``r_nn``);
* 80 radial bins on ``[0, 4] r_nn``; the analysis re-bins by 2 to the 40 bins quoted in the paper;
* every set starts at seed 1000. Sets with the same system therefore share realizations (identical
  positions and initial directions), which turns comparisons into *paired* differences.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from .config import RNN_UNIT_DENSITY, AnnealConfig, SimulationConfig
from .ensemble import CONFIG_FILENAME, seed_path
from .protocols import protocol_cost, protocol_seed_path

BASE_SEED = 1000
N_BINS = 80
REBIN = 2
SWEEPS = 100
PRIORITIES = ("required", "recommended", "optional")
SNAPSHOT_SWEEPS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500)
SMOKE_ROOT = Path("results") / "_smoke"
DEFAULT_ROOT = Path("results")


@dataclass(frozen=True)
class PlanEntry:
    """One ensemble (or protocol-check) run."""

    id: str
    group: str
    title: str
    purpose: str
    priority: str
    config: SimulationConfig
    kind: str = "ensemble"  # "ensemble" | "protocol"
    feeds: str = ""
    rebin: int = REBIN

    @property
    def n_realizations(self) -> int:
        return self.config.n_realizations

    def cost(self) -> dict[str, float]:
        """``seconds`` per realization, ``total_core_seconds`` and ``memory_mb`` per worker."""
        c = protocol_cost(self.config) if self.kind == "protocol" else self.config.estimate_cost()
        return {
            "seconds": c["seconds"],
            "total_core_seconds": c["seconds"] * self.n_realizations,
            "memory_mb": c["memory_mb"],
        }

    def describe(self) -> str:
        c = self.config
        parts = [
            f"N={c.n_particles}",
            c.boundary,
            f"r_cut={c.r_cut_nn_resolved:g} r_nn",
            f"{c.n_sweeps} sweeps",
        ]
        if c.coupling != "gr":
            parts.append(f"coupling={c.coupling}")
        if c.spin_distribution != "uniform":
            parts.append(f"s~{c.spin_distribution}")
        if self.kind == "protocol":
            parts.append(f"anneal {c.anneal.sweeps_per_temperature} sweeps/T")
        return ", ".join(parts)


@dataclass(frozen=True)
class DerivedItem:
    """A deliverable that needs no new simulation."""

    id: str
    title: str
    depends_on: tuple[str, ...]
    how: str
    feeds: str


@dataclass(frozen=True)
class Member:
    """A run (optionally truncated to its first ``max_realizations``) inside a comparison."""

    entry: str
    label: str
    max_realizations: int | None = None


@dataclass(frozen=True)
class Comparison:
    """Several runs analysed side by side; the first member is the reference of paired differences."""

    id: str
    title: str
    members: tuple[Member, ...]
    observables: tuple[str, ...] = ("C", "Q", "A")
    feeds: str = ""


@dataclass(frozen=True)
class Plan:
    entries: tuple[PlanEntry, ...]
    derived: tuple[DerivedItem, ...]
    comparisons: tuple[Comparison, ...]
    smoke: bool = False

    @property
    def ids(self) -> list[str]:
        return [e.id for e in self.entries]

    @property
    def groups(self) -> list[str]:
        seen: list[str] = []
        for e in self.entries:
            if e.group not in seen:
                seen.append(e.group)
        return seen

    def get(self, entry_id: str) -> PlanEntry:
        for e in self.entries:
            if e.id == entry_id:
                return e
        raise KeyError(f"unknown plan entry {entry_id!r}; known: {', '.join(self.ids)}")

    def select(self, names: Iterable[str]) -> list[PlanEntry]:
        """Entries named by id, group (``S2``), priority level or ``all``.

        Priority levels are cumulative: ``required`` < ``recommended`` < ``all`` (``optional``).
        Order and uniqueness follow the plan.
        """
        wanted: set[str] = set()
        for name in names:
            if name in self.ids:
                wanted.add(name)
            elif name in self.groups:
                wanted.update(e.id for e in self.entries if e.group == name)
            elif name in PRIORITIES or name == "all":
                rank = (
                    len(PRIORITIES) if name in ("all", "optional") else PRIORITIES.index(name) + 1
                )
                wanted.update(e.id for e in self.entries if PRIORITIES.index(e.priority) < rank)
            else:
                valid = [*self.ids, *self.groups, *PRIORITIES, "all"]
                raise ValueError(f"unknown plan target {name!r}; valid targets: {', '.join(valid)}")
        return [e for e in self.entries if e.id in wanted]


# --------------------------------------------------------------------------------------
# Construction
# --------------------------------------------------------------------------------------
def _tag(value: float) -> str:
    """Filesystem-friendly number tag: 7.5 -> ``7p5``, 12.0 -> ``12``."""
    return f"{value:g}".replace(".", "p")


def _config(smoke: bool, **kwargs) -> SimulationConfig:
    """Production configuration, or its scaled-down smoke twin (seconds instead of hours)."""
    base = dict(
        n_particles=5000,
        boundary="periodic",
        n_sweeps=SWEEPS,
        n_bins=N_BINS,
        x_max=4.0,
        extra_observables=True,
        base_seed=BASE_SEED,
    )
    base.update(kwargs)
    if smoke:
        n = min(base["n_particles"], 300)
        sweeps = min(base["n_sweeps"], 10)
        rc = base.get("r_cut_nn")
        if rc is not None:
            rc = min(rc, 5.0)
            if base["boundary"] == "periodic":  # keep r_cut <= L/2 in the tiny box
                rc = min(
                    rc,
                    0.999 * 0.5 * (n / base.get("density", 1.0)) ** (1.0 / 3.0) / RNN_UNIT_DENSITY,
                )
        base.update(
            n_particles=n,
            n_sweeps=sweeps,
            n_realizations=min(base["n_realizations"], 3),
            n_bins=20,
            snapshots=tuple(s for s in base.get("snapshots", ()) if s <= sweeps),
            save_states=min(base.get("save_states", 0), 1),
            r_cut_nn=rc,
        )
    return SimulationConfig(**base)


def build_plan(smoke: bool = False) -> Plan:
    """The full plan, or a tiny version of it (a few seconds per entry) for testing the pipeline."""
    entries: list[PlanEntry] = []

    def add(**kw) -> None:
        entries.append(PlanEntry(**kw))

    # ---- M: main ensemble ----------------------------------------------------------
    add(
        id="M",
        group="M",
        title="Main ensemble",
        priority="required",
        purpose="C(r), Q(r), A(r), B(r) with error bars; landmarks, bootstrap fit, nearest-neighbour "
        "diagnostics, S(0); the headline result.",
        feeds="Main-text figures of C(r) and Q(r); numbers quoted in the text; SM tables.",
        config=_config(
            smoke, r_cut_nn=10.0, n_realizations=200, nn_diagnostics=True, save_states=3
        ),
    )
    # ---- S1: sweep convergence -----------------------------------------------------
    add(
        id="S1",
        group="S1",
        title="Relaxation sweeps",
        priority="required",
        purpose="How many sweeps are needed? C(r) and the energy at sweeps 1 ... 500 (same seeds as M).",
        feeds="SM: relaxation sweeps (convergence of energy and C(r)).",
        config=_config(
            smoke, r_cut_nn=10.0, n_realizations=48, n_sweeps=500, snapshots=SNAPSHOT_SWEEPS
        ),
    )
    # ---- S2: cutoff -------------------------------------------------------------------
    s2_cutoffs = (3.0, 4.0, 5.0, 7.5, 12.0, 15.0) if not smoke else (3.0, 5.0)
    for rc in s2_cutoffs:
        add(
            id=f"S2_rc{_tag(rc)}",
            group="S2",
            title=f"Cutoff r_cut = {rc:g} r_nn",
            priority="required",
            purpose="Dependence on the interaction cutoff; r_cut = 10 r_nn is entry M (first 48 seeds).",
            feeds="SM: cutoff dependence.",
            config=_config(smoke, r_cut_nn=rc, n_realizations=48),
        )
    # ---- S3: finite size and boundary -----------------------------------------------
    s3_sizes = ((500, 100), (1000, 100), (2000, 100), (5000, 100), (10000, 48), (20000, 24))
    if smoke:
        s3_sizes = ((150, 3), (300, 3))
    for n, reps in s3_sizes:
        for boundary in ("periodic", "open"):
            add(
                id=f"S3_N{n}_{boundary}",
                group="S3",
                title=f"N = {n}, {boundary} boundaries",
                priority="required",
                purpose="Finite-size and boundary effects at a common cutoff of 7 r_nn "
                "(the largest value allowed by periodic boundaries at N = 500).",
                feeds="SM: finite-size scaling and boundary effects.",
                config=_config(
                    smoke, n_particles=n, boundary=boundary, r_cut_nn=7.0, n_realizations=reps
                ),
            )
    # ---- S4: spin magnitudes -----------------------------------------------------------
    for dist in ("equal", "uniform_half", "lognormal"):
        add(
            id=f"S4_{dist}",
            group="S4",
            title=f"Spin magnitudes: {dist}",
            priority="recommended",
            purpose="Is the correlation structure independent of the distribution of spin lengths? "
            "Baseline: the first 100 seeds of M (U(0, S)).",
            feeds="SM: spin-magnitude distribution (Q is the only observable that depends on it).",
            config=_config(smoke, r_cut_nn=10.0, n_realizations=100, spin_distribution=dist),
        )
    # ---- S6: controls -------------------------------------------------------------------
    for coupling, why in (
        ("isotropic_afm", "isotropic 1/r^3 antiferromagnet (no tensor structure)"),
        ("dipolar", "sign-reversed tensor, i.e. a dipolar ferromagnet"),
    ):
        add(
            id=f"S6_{coupling}",
            group="S6",
            title=f"Control: {coupling.replace('_', ' ')}",
            priority="recommended",
            purpose=f"Same geometry and seeds as M with the {why}: shows which features are specific "
            "to the GR tensor.",
            feeds="Main text or SM: control experiments.",
            config=_config(smoke, r_cut_nn=10.0, n_realizations=100, coupling=coupling),
        )
    # ---- S5: protocol check ----------------------------------------------------------
    for tag, sweeps_per_t, reps in (("short", 10, 32), ("long", 100, 16)):
        add(
            id=f"S5_protocol_{tag}",
            group="S5",
            title=f"Protocol check ({tag} annealing)",
            priority="optional",
            kind="protocol",
            purpose="Do random update order, other initial conditions or Metropolis annealing change the "
            "relaxed state? Paired energies, overlaps and correlation curves.",
            feeds="SM / one sentence: low-energy metastable state, protocol independence.",
            config=_config(
                smoke,
                n_particles=1000,
                r_cut_nn=8.0,
                n_bins=40,
                n_realizations=reps,
                anneal=AnnealConfig(
                    n_temperatures=4 if smoke else 40,
                    sweeps_per_temperature=2 if smoke else sweeps_per_t,
                ),
            ),
            rebin=1,
        )

    by_id = {e.id: e for e in entries}

    def member(entry: str, label: str, n: int | None = None) -> Member | None:
        if entry not in by_id:
            return None
        return Member(entry, label, None if n is None else min(n, by_id[entry].n_realizations))

    def compare(
        cid: str, title: str, members: Sequence[Member | None], feeds: str, obs=("C", "Q", "A")
    ):
        kept = tuple(m for m in members if m is not None)
        return Comparison(cid, title, kept, tuple(obs), feeds) if len(kept) >= 2 else None

    cutoffs = sorted({*s2_cutoffs, 10.0}, reverse=True)  # reference first: the largest cutoff
    comparisons = [
        compare(
            "cmp_cutoff",
            "Cutoff dependence (reference: the largest cutoff)",
            [
                member("M", "r_cut = 10", 48)
                if rc == 10.0
                else member(f"S2_rc{_tag(rc)}", f"r_cut = {rc:g}")
                for rc in cutoffs
            ],
            "SM: cutoff dependence.",
        )
    ]
    sizes = sorted((n for n, _ in s3_sizes), reverse=True)
    for boundary in ("periodic", "open"):
        comparisons.append(
            compare(
                f"cmp_size_{boundary}",
                f"Finite-size dependence, {boundary} boundaries (reference: largest N)",
                [member(f"S3_N{n}_{boundary}", f"N = {n}") for n in sizes],
                "SM: finite-size scaling.",
            )
        )
    for n in (1000, 5000) if not smoke else (300,):
        comparisons.append(
            compare(
                f"cmp_boundary_N{n}",
                f"Open versus periodic boundaries at N = {n} (paired)",
                [member(f"S3_N{n}_periodic", "periodic"), member(f"S3_N{n}_open", "open")],
                "SM: boundary effects.",
            )
        )
    comparisons.append(
        compare(
            "cmp_spin_distribution",
            "Spin-magnitude distribution (reference: U(0, S), first 100 seeds of M)",
            [member("M", "U(0,S) [M]", 100)]
            + [
                member(f"S4_{d}", d.replace("_", " "))
                for d in ("equal", "uniform_half", "lognormal")
            ],
            "SM: spin-magnitude distribution.",
        )
    )
    comparisons.append(
        compare(
            "cmp_controls",
            "Control couplings (reference: GR tensor, first 100 seeds of M)",
            [
                member("M", "GR [M]", 100),
                member("S6_isotropic_afm", "isotropic AFM"),
                member("S6_dipolar", "dipolar (sign-reversed tensor)"),
            ],
            "Controls: which features need the GR tensor?",
        )
    )

    derived = (
        DerivedItem(
            id="S7",
            title="Nearest-neighbour diagnostics, S(0), tensor correlations, binning check",
            depends_on=("M",),
            how="`grspin plan analyze M` writes nn_*.csv, small_r.csv and summary.json; for the binning check "
            "run `grspin analyze results/M --rebin R --out results/M/analysis_rebin_R` with R = 1, 2, 4, 8.",
            feeds="SM: nearest-neighbour diagnostics and binning; main text: dimer picture.",
        ),
    )
    return Plan(tuple(entries), derived, tuple(c for c in comparisons if c is not None), smoke)


# --------------------------------------------------------------------------------------
# Status and cost
# --------------------------------------------------------------------------------------
@dataclass
class EntryStatus:
    entry: PlanEntry
    finished: int
    total: int
    state: str  # todo | partial | done | stale-config


def entry_dir(entry: PlanEntry, root: Path) -> Path:
    return Path(root) / entry.id


def entry_status(entry: PlanEntry, root: Path) -> EntryStatus:
    """Progress of ``entry`` on disk (finished realization files versus planned)."""
    run_dir = entry_dir(entry, root)
    path_of = protocol_seed_path if entry.kind == "protocol" else seed_path
    finished = sum(1 for s in entry.config.seeds if path_of(run_dir, s).exists())
    total = len(entry.config.seeds)
    state = "done" if finished == total else "partial" if finished else "todo"
    cfg_file = run_dir / CONFIG_FILENAME
    if cfg_file.exists():
        try:
            saved = SimulationConfig.from_dict(json.loads(cfg_file.read_text())["config"])
            if saved.compat_key() != entry.config.compat_key():
                state = "stale-config"
        except (ValueError, KeyError, TypeError):
            state = "stale-config"
    return EntryStatus(entry, finished, total, state)


def plan_status(plan: Plan, root: Path) -> list[EntryStatus]:
    return [entry_status(e, root) for e in plan.entries]


def plan_cost(entries: Sequence[PlanEntry]) -> dict[str, float]:
    """Total core-seconds, core-hours and the largest per-worker memory of ``entries``."""
    total = sum(e.cost()["total_core_seconds"] for e in entries)
    return {
        "core_seconds": total,
        "core_hours": total / 3600.0,
        "n_realizations": float(sum(e.n_realizations for e in entries)),
        "max_memory_mb": max((e.cost()["memory_mb"] for e in entries), default=0.0),
    }


# --------------------------------------------------------------------------------------
# Documentation
# --------------------------------------------------------------------------------------
def _fmt_core_time(seconds: float) -> str:
    if seconds < 90:
        return f"{seconds:.0f} core-s"
    if seconds < 5400:
        return f"{seconds / 60:.0f} core-min"
    return f"{seconds / 3600:.1f} core-h"


GROUP_TITLES = {
    "M": "Main ensemble",
    "S1": "Relaxation sweeps",
    "S2": "Cutoff dependence",
    "S3": "Finite size and boundary conditions",
    "S4": "Spin-magnitude distribution",
    "S5": "Protocol / metastability check",
    "S6": "Control couplings",
}


def plan_markdown(plan: Plan | None = None) -> str:
    """``docs/SIMULATION_PLAN.md``: an overview table and a checklist, generated from the plan."""
    plan = plan or build_plan(False)
    lines = [
        "# Simulation plan",
        "",
        "<!-- Generated by `grspin plan markdown --write docs/SIMULATION_PLAN.md` from `src/grspin/plan.py`. -->",
        "<!-- Tick the boxes freely while working through the list; to change the plan edit `plan.py`. -->",
        "",
        "Everything the manuscript needs, as a checklist. `grspin plan status` shows the live progress "
        "from the files in `results/`; `grspin plan run <ids>` runs entries (finished realizations are "
        "skipped, so interrupted runs simply resume). Targets can be entry ids, groups (`S2`), "
        "`required`, `recommended` or `all`.",
        "",
        "## Conventions",
        "",
        "- Periodic boundaries (minimum image), Poisson positions at unit density, GR coupling, spin "
        "lengths `U(0, S)`, sequential greedy relaxation of 100 sweeps - unless an entry says otherwise.",
        "- Distances are in units of the mean nearest-neighbour distance "
        "`r_nn = Gamma(4/3) (3/(4 pi rho))^(1/3)`.",
        f"- {N_BINS} radial bins on `[0, 4] r_nn` are stored; the analysis re-bins by {REBIN} to the "
        f"{N_BINS // REBIN} bins quoted in the paper (and allows 1, 2, 4, 8 for the binning check).",
        f"- Every set starts at seed {BASE_SEED}. Sets with the same system share realizations (identical "
        "positions and initial directions), so their differences are *paired* and much less noisy than "
        "the individual curves.",
        "- Results go to `results/<id>/` (git-ignored); smoke tests to `results/_smoke/`.",
        "- Run times are single-core estimates from the cost model in `config.py` (fitted to "
        "`grspin benchmark` measurements; check it on your machine); divide by the number of cores. "
        "Memory is per worker; `grspin plan run` limits the workers to half of the RAM by default "
        "(`--max-memory-gb`).",
        "",
        "## Overview",
        "",
        "| id | priority | system | realizations | cost | memory / worker |",
        "|---|---|---|---:|---:|---:|",
    ]
    for e in plan.entries:
        c = e.cost()
        lines.append(
            f"| `{e.id}` | {e.priority} | {e.describe()} | {e.n_realizations} | "
            f"{_fmt_core_time(c['total_core_seconds'])} | {c['memory_mb']:.0f} MB |"
        )
    lines.append("")
    for level in ("required", "recommended", "all"):
        chosen = plan.select([level])
        cost = plan_cost(chosen)
        lines.append(
            f"- **{level}**: {len(chosen)} runs, {int(cost['n_realizations'])} realizations, "
            f"{cost['core_hours']:.2f} core-hours (about {cost['core_hours'] * 60 / 8:.0f} min on "
            f"8 cores), largest worker ~{cost['max_memory_mb']:.0f} MB"
        )

    lines += ["", "## Checklist: simulations", ""]
    for priority in PRIORITIES:
        groups = [g for g in plan.groups if plan.select([g])[0].priority == priority]
        if not groups:
            continue
        lines += [f"### {priority.capitalize()}", ""]
        for group in groups:
            members = [e for e in plan.entries if e.group == group]
            purposes = list(dict.fromkeys(e.purpose for e in members))
            feeds = list(dict.fromkeys(e.feeds for e in members))
            lines.append(
                f"**{group} - {GROUP_TITLES.get(group, members[0].title)}** - run all: `grspin plan run {group}`"
            )
            lines.append("")
            lines += [f"- purpose: {text}" for text in purposes]
            lines += [f"- feeds: {text}" for text in feeds]
            for e in members:
                c = e.cost()
                lines.append(
                    f"- [ ] `{e.id}`: {e.n_realizations} x ({e.describe()}) - "
                    f"{_fmt_core_time(c['total_core_seconds'])}"
                )
            lines.append("")

    lines += ["## Checklist: analysis", ""]
    lines.append(
        "Per-run analysis (tables, landmarks with bootstrap errors, fit, figures) is written to "
        "`results/<id>/analysis/`; comparisons to `results/_comparisons/<id>/`. `grspin plan analyze` "
        "without arguments does everything that is finished."
    )
    lines.append("")
    for group in plan.groups:
        lines.append(f"- [ ] analyse **{group}**: `grspin plan analyze {group}`")
    lines.append("")
    for comp in plan.comparisons:
        members = ", ".join(
            f"`{m.entry}`" + (f" (first {m.max_realizations})" if m.max_realizations else "")
            for m in comp.members
        )
        lines += [
            f"- [ ] comparison `{comp.id}` - {comp.title}: `grspin plan analyze {comp.id}`",
            f"  - members: {members}",
            f"  - feeds: {comp.feeds}",
        ]
    lines.append("")
    for d in plan.derived:
        needs = ", ".join(f"`{x}`" for x in d.depends_on)
        lines += [
            f"- [ ] **{d.id}** - {d.title} (needs {needs}; no new simulation)",
            f"  - how: {d.how}",
            f"  - feeds: {d.feeds}",
        ]
    lines += [
        "",
        "## Checklist: before submitting",
        "",
        "- [ ] all required runs finished with the final code version (`git tag` the commit; the commit "
        "hash is stored in every `config.json`)",
        "- [ ] `grspin plan status` shows `done` for every entry used in the paper",
        "- [ ] figures and numbers in the manuscript regenerated from `results/`",
        "- [ ] data-availability statement points to the tagged repository release (e.g. a Zenodo DOI)",
        "",
    ]
    return "\n".join(lines)
