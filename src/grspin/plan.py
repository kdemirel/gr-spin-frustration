"""The simulation plan as code: every run needed for the manuscript, its purpose and its cost.

The plan is data (:func:`build_plan`); the documentation (``docs/SIMULATION_PLAN.md``), the status
report and the runner (``grspin plan ...``) are all generated from it, so they cannot drift apart.

Conventions shared by all sets
------------------------------
* spins ``U(0, S)`` distributed magnitudes, GR coupling, sequential greedy relaxation of 100 sweeps,
  periodic boundaries (unless the set studies the boundary), density ``rho = 1`` (irrelevant: all
  distances are in units of ``r_nn``);
* 80 radial bins on ``[0, 4] r_nn``; the analysis re-bins by 2 to the 40 bins quoted in the paper
  (the far-tail set ``T`` stores 200 bins on ``[0, 10] r_nn``);
* every set starts at seed 1000. Sets with the same system therefore share realizations (identical
  positions and initial directions), which turns comparisons into *paired* differences; the
  hard-core set ``S8`` draws other positions, so its comparison with ``M`` is unpaired;
* ``M`` stores 48 particle states, the controls ``S6``, the channel family ``S7``, ``S8`` and the
  far-tail set ``T`` 16 each (energy budget, dimer-field test and ``S(k)`` from stored states).

Ensemble groups are ``M``, ``S1`` ... ``S8`` and ``T``; the derived items (analysis only, no new
simulation) are ``D1``, ``D2``, ``D3``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import RNN_UNIT_DENSITY, AnnealConfig, SimulationConfig
from .ensemble import CONFIG_FILENAME, pending_seeds, provenance, seed_path
from .protocols import protocol_cost, protocol_seed_path

BASE_SEED = 1000
N_BINS = 80
REBIN = 2
SWEEPS = 100
X_MAX = 4.0
TAIL_N_BINS = 200
TAIL_X_MAX = 10.0
PRIORITIES = ("required", "recommended", "optional")
SNAPSHOT_SWEEPS = (1, 2, 3, 5, 10, 20, 50, 100, 200, 500)
SMOKE_ROOT = Path("results") / "_smoke"
DEFAULT_ROOT = Path("results")
#: ``S(k)`` at the smallest wavevectors versus cutoff and system size, from the stored states of
#: the far-tail entries and ``M`` (:func:`grspin.report.structure_factor_comparison`; written by
#: ``grspin plan analyze`` to ``results/_comparisons/<id>/`` once at least two of the entries have
#: states). It is not a :class:`Comparison` because it compares states, not binned curves.
STRUCTURE_FACTOR_COMPARISON_ID = "cmp_structure_factor"
STRUCTURE_FACTOR_ENTRIES: tuple[str, ...] = (
    "T_N10000_rc7",
    "T_N10000_rc10",
    "T_N10000_rc15",
    "T_N20000_rc10",
    "M",
)


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
        if c.min_separation_nn:
            parts.append(f"r_min={c.min_separation_nn:g} r_nn")
        if c.spin_distribution != "uniform":
            parts.append(f"s~{c.spin_distribution}")
        if c.x_max != X_MAX:
            parts.append(f"x_max={c.x_max:g} r_nn")
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
        x_max=X_MAX,
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
            x_max=min(base["x_max"], X_MAX),  # the far-tail window does not fit the tiny box
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
        purpose="C(r), P(r), Cperp(r), Q(r), A(r), B(r) with error bars; landmarks with their error "
        "budget, nearest-neighbour diagnostics, the dimer-frame correlation map, S(0) and S(k) "
        "from 48 stored states; the headline result.",
        feeds="Main-text figures of C(r), its channel decomposition and the dimer-frame map; "
        "numbers quoted in the text; SM tables.",
        config=_config(
            smoke, r_cut_nn=10.0, n_realizations=200, nn_diagnostics=True, save_states=48
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
            feeds="SM: spin-magnitude distribution. The sign pattern (P < 0, Cperp > 0) and the "
            "transverse maximum (Cperp 0.44-0.45) are robust, but the landmarks of C(r) shift: "
            "x_0 = 0.747 (lognormal), 0.844 (U(0, S)), 0.880 (U(S/2, 3S/2)), 0.890 (equal); Q "
            "depends on the lengths through |S_i + S_j| and is undefined for equal lengths.",
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
            "to the GR tensor. 16 stored states feed the energy budget (frustration index with the "
            "two-body minimum of the control coupling) and S(k).",
            feeds="Main text or SM: control experiments.",
            config=_config(
                smoke, r_cut_nn=10.0, n_realizations=100, coupling=coupling, save_states=16
            ),
        )
    # ---- S7: channel controls (gamma family) -------------------------------------------
    channel_purpose = (
        "Channel controls: which channel carries the positive shell? sigma = +1 with gamma = 1 "
        "(transverse channel only), 12 (longitudinal : transverse = 11 : 1) and 100 (longitudinal "
        "channel only up to a 1 % transverse admixture; the overall scale of H is irrelevant at "
        "T = 0) on the geometry and seeds of M, whose GR coupling gamma = 3 (weights 2 : 1) "
        "completes the family."
    )
    for coupling, gamma, what in (
        ("transverse_only", 1, "transverse channel only"),
        ("gamma12", 12, "longitudinal : transverse = 11 : 1"),
        ("longitudinal_only", 100, "longitudinal channel only, 1 % transverse"),
    ):
        add(
            id=f"S7_gamma{gamma}",
            group="S7",
            title=f"Channel control: gamma = {gamma} ({what})",
            priority="recommended",
            purpose=channel_purpose,
            feeds="Main text or SM: channel controls - the landmarks x_0, C_peak, x_peak, Cperp_max, "
            "Q_min and the sign of C at 1.2 r_nn against gamma, and C, P, Cperp of the four "
            "couplings in one figure; the nn diagnostics record the dimer map, the 16 stored states "
            "feed the energy budget and the dimer-field test.",
            config=_config(
                smoke,
                r_cut_nn=10.0,
                n_realizations=100,
                coupling=coupling,
                nn_diagnostics=True,
                save_states=16,
            ),
        )
    # ---- S8: hard-core (minimum-separation) positions ----------------------------------
    add(
        id="S8_hardcore05",
        group="S8",
        title="Hard core: minimum separation 0.5 r_nn",
        priority="optional",
        purpose="Hard-core positions: sequential random addition with all pair distances >= 0.5 r_nn "
        "removes the closest pairs (which dominate the mean energy) and changes the dimer "
        "statistics (the Poisson formulas f_m(r) and 16/27 do not apply). Which features survive: "
        "the channel signs P < 0 and Cperp > 0, the positive shell, x_0? The positions differ from "
        "M, so the comparison is unpaired.",
        feeds="SM: hard-core ensemble; tests the claim that the channel structure is universal while "
        "the Poisson numbers (dimer fraction, landmarks) are not.",
        config=_config(
            smoke,
            r_cut_nn=10.0,
            n_realizations=100,
            min_separation_nn=0.5,
            nn_diagnostics=True,
            save_states=16,
        ),
    )
    # ---- T: far tail and cutoff ------------------------------------------------------
    tail_purpose = (
        "far tail of C, P, Cperp out to 10 r_nn and its cutoff dependence (the negative far-tail "
        "dip is a truncation effect whose position scales with r_cut); 16 stored states per entry "
        "for S(k) at the smallest wavevectors versus r_cut and for the energy budget"
    )
    for n, rc, reps in ((10000, 7.0, 32), (10000, 10.0, 32), (10000, 15.0, 32), (20000, 10.0, 32)):
        add(
            id=f"T_N{n}_rc{_tag(rc)}",
            group="T",
            title=f"Far tail: N = {n}, r_cut = {rc:g} r_nn",
            priority="recommended",
            purpose=tail_purpose,
            feeds="SM: far tail and truncation; justifies quoting C(r) for x <= 4 at r_cut = 10 r_nn.",
            config=_config(
                smoke,
                n_particles=n,
                r_cut_nn=rc,
                n_bins=TAIL_N_BINS,
                x_max=TAIL_X_MAX,
                n_realizations=reps,
                save_states=16,
            ),
        )
    # ---- S5: protocol check ----------------------------------------------------------
    protocol_purpose = (
        "Do random update order, other initial conditions or Metropolis annealing change the "
        "relaxed state? Paired energies, overlaps and correlation curves."
    )

    def anneal(sweeps_per_t: int) -> AnnealConfig:
        return AnnealConfig(
            n_temperatures=4 if smoke else 40,
            sweeps_per_temperature=2 if smoke else sweeps_per_t,
        )

    add(
        id="S5_N5000",
        group="S5",
        title="Protocol check at production size",
        priority="recommended",
        kind="protocol",
        purpose=protocol_purpose + " Same system and seeds as M, so the comparison is paired at "
        "the precision of the main result.",
        feeds="SM table: protocol independence of the landmarks (energies, overlaps, Delta C) at "
        "N = 5000; the per-temperature acceptance of the annealing stage.",
        config=_config(smoke, r_cut_nn=10.0, n_realizations=48, anneal=anneal(10)),
    )
    for tag, sweeps_per_t, reps in (("short", 10, 32), ("long", 100, 16)):
        add(
            id=f"S5_protocol_{tag}",
            group="S5",
            title=f"Protocol check ({tag} annealing)",
            priority="optional",
            kind="protocol",
            purpose=protocol_purpose + " Small system (N = 1000) with short and long annealing.",
            feeds="SM / one sentence: low-energy metastable state, protocol independence.",
            config=_config(
                smoke,
                n_particles=1000,
                r_cut_nn=8.0,
                n_bins=40,
                n_realizations=reps,
                anneal=anneal(sweeps_per_t),
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
    five = ("C", "P", "Cperp", "Q", "A")
    comparisons.append(
        compare(
            "cmp_channels",
            "Channel controls gamma = 1, 3, 12, 100 (reference: GR, gamma = 3, first 100 seeds "
            "of M; paired)",
            [
                member("M", "gamma = 3 (GR) [M]", 100),
                member("S7_gamma1", "gamma = 1 (transverse only)"),
                member("S7_gamma12", "gamma = 12"),
                member("S7_gamma100", "gamma = 100 (longitudinal only)"),
            ],
            "Channel controls: which channel carries the positive shell?",
            five,
        )
    )
    comparisons.append(
        compare(
            "cmp_hardcore",
            "Hard-core positions versus the Poisson process (reference: M; unpaired, the "
            "positions differ)",
            [member("M", "Poisson [M]"), member("S8_hardcore05", "hard core 0.5 r_nn")],
            "SM: hard-core ensemble - channel signs, positive shell and x_0 without the closest "
            "pairs.",
            five,
        )
    )
    channels = ("C", "P", "Cperp")
    comparisons.append(
        compare(
            "cmp_far_tail",
            "Far tail versus cutoff at N = 10000 (reference: r_cut = 15 r_nn, paired)",
            [member(f"T_N10000_rc{_tag(rc)}", f"r_cut = {rc:g}") for rc in (15.0, 10.0, 7.0)],
            "SM: the far-tail dip moves with r_cut (truncation effect); C, P, Cperp on [0, 10].",
            channels,
        )
    )
    comparisons.append(
        compare(
            "cmp_far_tail_size",
            "Far tail versus system size at r_cut = 10 r_nn (reference: N = 20000, unpaired)",
            [member("T_N20000_rc10", "N = 20000"), member("T_N10000_rc10", "N = 10000")],
            "SM: the far tail is N-independent at fixed r_cut.",
            channels,
        )
    )

    derived = (
        DerivedItem(
            id="D1",
            title="Nearest-neighbour diagnostics, S(0), tensor correlations, binning check",
            depends_on=("M",),
            how="`grspin plan analyze M` writes nn_*.csv, small_r.csv and summary.json; for the binning check "
            "run `grspin analyze results/M --rebin R --out results/M/analysis_rebin_R` with R = 1, 2, 4, 8.",
            feeds="SM: nearest-neighbour diagnostics and binning; main text: dimer picture.",
        ),
        DerivedItem(
            id="D2",
            title="Error budget of the landmarks, dimer-frame correlation map, spin structure "
            "factor, energy budget",
            depends_on=("M", "S1", "S2_rc15", "S6_isotropic_afm", "S6_dipolar", "T_N10000_rc15"),
            how="`grspin plan analyze M` writes error_budget.csv (statistical, binning, sweeps and cutoff "
            "terms from the paired runs S1 and S2_rc15), dimer_map.csv / dimer_angular.csv / dimer_map.pdf "
            "(from the nn diagnostics of M), structure_factor.csv / .pdf and energy_budget.csv / .json "
            "(dimer share, realized fraction and the frustration index E / sum of two-body minima, "
            "from the stored states; the same files for every run with states, so S6, S7, S8 and T "
            "give the index with their own coupling). `grspin plan analyze T M` (or `comparisons`) "
            f"writes `{STRUCTURE_FACTOR_COMPARISON_ID}`: structure_factor_vs_cutoff.csv, "
            "structure_factor_comparison.pdf / .md with S(0), the first three shells and the maximum "
            f"of S(k) for {', '.join(STRUCTURE_FACTOR_ENTRIES)} (needs stored states of at least two).",
            feeds="Main text: landmark errors and the dimer-frame map (mechanism); SM: S(k) versus "
            "r_cut and N, the energy budget and frustration index of GR and of the controls.",
        ),
        DerivedItem(
            id="D3",
            title="Channel controls and dimer-field test",
            depends_on=("M", "S7_gamma1", "S7_gamma12", "S7_gamma100"),
            how="`grspin plan analyze S7 cmp_channels` writes the per-run analyses of the gamma family "
            "and the paired comparison with M: the landmarks x_0, C_peak, x_peak, Cperp_max, Q_min and "
            "the sign of C at 1.2 r_nn against gamma = 1, 3, 12, 100, and C, P, Cperp of the four "
            "couplings in one figure. The dimer-field test (does the field of the two dimer spins alone "
            "predict the direction of a third spin? agreement fraction per distance bin, random = 1/2, "
            "and the dimer-only prediction next to the measured angular table) comes from the nn "
            "diagnostics of M and of the S7 runs (dimer_field.csv, a column of dimer_angular.csv and a "
            "summary.md section).",
            feeds="Main text: the positive shell is carried by the transverse channel if it survives "
            "gamma = 1 and vanishes for gamma = 100, and the causal dimer-field test; SM: the gamma "
            "family (landmarks against gamma).",
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
    """Progress of ``entry`` on disk (finished realization files versus planned).

    Like the runner, an ensemble realization also needs its state file when it is one of the
    first ``save_states`` seeds.
    """
    run_dir = entry_dir(entry, root)
    if entry.kind == "protocol":
        finished = sum(1 for s in entry.config.seeds if protocol_seed_path(run_dir, s).exists())
    else:
        finished = len(entry.config.seeds) - len(pending_seeds(entry.config, run_dir))
    total = len(entry.config.seeds)
    state = "done" if finished == total else "partial" if finished else "todo"
    cfg_file = run_dir / CONFIG_FILENAME
    if cfg_file.exists():
        try:
            saved = SimulationConfig.from_dict(json.loads(cfg_file.read_text())["config"])
            if saved.compat_key(entry.kind) != entry.config.compat_key(entry.kind):
                state = "stale-config"
        except (ValueError, KeyError, TypeError):
            state = "stale-config"
    return EntryStatus(entry, finished, total, state)


def plan_status(plan: Plan, root: Path) -> list[EntryStatus]:
    return [entry_status(e, root) for e in plan.entries]


@dataclass
class EntryProvenance:
    """Which code produced the files of one entry (from ``config.json`` and the realization files)."""

    entry: PlanEntry
    state: str  # as in :class:`EntryStatus`
    git_commit: str | None
    git_dirty: bool | None
    source_hash: str | None
    created_utc: str | None
    n_files: int  # realization files on disk
    file_hashes: tuple[str, ...]  # distinct source hashes found in them ("" = unstamped file)

    def matches(self, current: dict[str, Any]) -> bool | None:
        """``True`` if ``config.json`` and every realization file carry the hash of the current
        sources; ``None`` when nothing is on disk."""
        if self.source_hash is None and not self.n_files:
            return None
        wanted = current.get("source_hash")
        return self.source_hash == wanted and set(self.file_hashes) <= {wanted}


def entry_provenance(entry: PlanEntry, root: Path) -> EntryProvenance:
    """Provenance of ``entry`` on disk; realization files without a stamp count as ``""``."""
    run_dir = entry_dir(entry, root)
    status = entry_status(entry, root)
    prov: dict[str, Any] = {}
    cfg_file = run_dir / CONFIG_FILENAME
    if cfg_file.exists():
        try:
            prov = json.loads(cfg_file.read_text()).get("provenance", {})
        except (ValueError, AttributeError):
            prov = {}
    path_of = protocol_seed_path if entry.kind == "protocol" else seed_path
    hashes: set[str] = set()
    for seed in entry.config.seeds:
        path = path_of(run_dir, seed)
        if not path.exists():
            continue
        with np.load(path) as data:
            hashes.add(str(data["source_hash"]) if "source_hash" in data.files else "")
    return EntryProvenance(
        entry,
        status.state,
        prov.get("git_commit"),
        prov.get("git_dirty"),
        prov.get("source_hash"),
        prov.get("created_utc"),
        status.finished,
        tuple(sorted(hashes)),
    )


def plan_provenance(plan: Plan, root: Path) -> list[EntryProvenance]:
    return [entry_provenance(e, root) for e in plan.entries]


def provenance_report(plan: Plan, root: Path, current: dict[str, Any] | None = None) -> str:
    """Text table: commit, dirty flag and source hash of every entry versus the current tree."""
    current = provenance() if current is None else current
    dirty = {True: "dirty", False: "clean", None: "no git"}

    def short(text: str | None, n: int = 12) -> str:
        return (text or "-")[:n]

    rows = []
    for p in plan_provenance(plan, root):
        match = p.matches(current)
        if len(p.file_hashes) > 1:
            files = f"{p.n_files} (mixed)"
        elif p.file_hashes == ("",):
            files = f"{p.n_files} (unstamped)"
        else:
            files = str(p.n_files)
        rows.append(
            [
                p.entry.id,
                p.state,
                short(p.git_commit, 7),
                dirty[p.git_dirty] if p.git_commit else "-",
                short(p.source_hash),
                files,
                "-" if match is None else "yes" if match else "NO",
            ]
        )
    header = ["id", "state", "commit", "tree", "source hash", "files", "matches tree"]
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    lines = [
        f"current tree: commit {short(current.get('git_commit'), 7)} "
        f"({dirty[current.get('git_dirty')]}), source hash {short(current.get('source_hash'))}",
        f"results under {root}/",
        "",
        fmt.format(*header),
        fmt.format(*("-" * w for w in widths)),
        *(fmt.format(*r) for r in rows),
        "",
        "'matches tree': config.json and every realization file carry the hash of the current "
        "src/grspin sources; 'files' counts realization files on disk ('unstamped': written before "
        "0.2.0, 'mixed': several code versions).",
    ]
    mismatch = [r[0] for r in rows if r[-1] == "NO"]
    if mismatch:
        lines.append(
            f"warning: {len(mismatch)} entries were produced by other sources than the current "
            f"tree: {', '.join(mismatch)}"
        )
    return "\n".join(lines)


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
    "S7": "Channel controls (gamma family)",
    "S8": "Hard-core positions (minimum separation)",
    "T": "Far tail and cutoff",
}


def group_priority(plan: Plan, group: str) -> str:
    """The highest priority among the entries of ``group`` (a group may mix priorities)."""
    return min((e.priority for e in plan.entries if e.group == group), key=PRIORITIES.index)


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
        f"- {N_BINS} radial bins on `[0, {X_MAX:g}] r_nn` are stored; the analysis re-bins by {REBIN} to the "
        f"{N_BINS // REBIN} bins quoted in the paper (and allows 1, 2, 4, 8 for the binning check). "
        f"The far-tail set `T` stores {TAIL_N_BINS} bins on `[0, {TAIL_X_MAX:g}] r_nn`.",
        f"- Every set starts at seed {BASE_SEED}. Sets with the same system share realizations (identical "
        "positions and initial directions), so their differences are *paired* and much less noisy than "
        "the individual curves; comparisons across different `N` or different position processes "
        "(the hard-core set `S8`, whose positions come from sequential random addition with all pair "
        "distances >= 0.5 r_nn) are unpaired.",
        "- Couplings: `gr` (sigma = +1, gamma = 3) unless stated; the controls `S6` reverse the sign or "
        "drop the tensor, the channel family `S7` keeps sigma = +1 and varies gamma (1: transverse "
        "channel only, 12: longitudinal : transverse = 11 : 1, 100: longitudinal only to 1 %; the "
        "overall scale of `H` is irrelevant at T = 0).",
        "- Stored particle states: 48 for `M`, 16 for `S6`, `S7`, `S8` and `T` (energy budget, "
        "dimer-field test, `S(k)`), none elsewhere.",
        "- Results go to `results/<id>/` (git-ignored); smoke tests to `results/_smoke/`.",
        "- Run times are single-core estimates from the cost model in `config.py` (fitted to "
        "`grspin benchmark` measurements; check it on your machine); divide by the number of cores. "
        "Memory is per worker; `grspin plan run` limits the workers to half of the RAM by default "
        "(`--max-memory-gb`): `T_N10000_rc15` (~5.5 GB) runs with a single worker on a 16 GB machine.",
        "- Protocol checks (`S5`) anneal from `t_start = 10 G S^2 rho` (above the typical dimer binding "
        "energy 2.9) to 0.01 over 40 temperatures and record the acceptance rate at every temperature.",
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
        groups = [g for g in plan.groups if group_priority(plan, g) == priority]
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
                tag = f" ({e.priority})" if e.priority != priority else ""
                lines.append(
                    f"- [ ] `{e.id}`{tag}: {e.n_realizations} x ({e.describe()}) - "
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
            f"  - members: {members}; observables: {', '.join(comp.observables)}",
            f"  - feeds: {comp.feeds}",
        ]
    lines += [
        f"- [ ] comparison `{STRUCTURE_FACTOR_COMPARISON_ID}` - S(k) at the smallest wavevectors "
        f"versus cutoff and system size, from the stored states: `grspin plan analyze "
        f"{STRUCTURE_FACTOR_COMPARISON_ID}`",
        "  - members: " + ", ".join(f"`{e}`" for e in STRUCTURE_FACTOR_ENTRIES) + " (those with "
        "stored states; at least two); outputs structure_factor_vs_cutoff.csv, "
        "structure_factor_comparison.pdf / .md",
        "  - feeds: SM: is the intermediate-scale feature of S(k) a cutoff artefact, a box effect or "
        "real? (compare r_cut = 7, 10, 15 r_nn at N = 10000 and N = 10000 versus 20000)",
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
        "- [ ] commit the final code (`git status` clean under `src/grspin`) and tag it (`git tag vX.Y.Z` "
        "with the version in `pyproject.toml`)",
        "- [ ] run the plan from that clean tree: `grspin plan run all` (finished realizations are "
        "kept; entries whose stored configuration no longer matches the plan are reported as "
        "`stale-config` and skipped, `--refresh-stale` recomputes them; `--overwrite` recomputes "
        "everything, about 10 min on 8 cores, so that every file is stamped with the tagged commit), "
        "then `grspin plan analyze`",
        "- [ ] verify the provenance: `grspin plan provenance` shows the tagged commit, a clean tree and "
        "`matches tree = yes` for every entry used in the paper (`config.json` and every realization "
        "file carry the commit and the SHA-256 of the sources; realizations are deterministic, so "
        "recomputing changes only the stamps)",
        "- [ ] `grspin plan status` shows `done` for every entry used in the paper",
        "- [ ] figures and numbers in the manuscript regenerated from `results/`",
        "- [ ] data-availability statement points to the tagged repository release (e.g. a Zenodo DOI)",
        "",
    ]
    return "\n".join(lines)
