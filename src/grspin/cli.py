"""Command line interface: ``grspin <command>`` (or ``python -m grspin <command>``)."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from . import __version__
from . import plan as planmod
from .analysis import EnsembleData
from .config import (
    BOUNDARIES,
    COUPLINGS,
    PROTOCOLS,
    SPIN_DISTRIBUTIONS,
    UPDATE_ORDERS,
    AnnealConfig,
    SimulationConfig,
)
from .ensemble import (
    ConfigMismatchError,
    available_memory_mb,
    run_ensemble,
    run_realization,
    warmup,
)
from .protocols import run_protocol_check
from .report import analyze_protocol, analyze_run, compare_runs

_D = SimulationConfig()  # defaults shown in --help
_A = AnnealConfig()


# --------------------------------------------------------------------------------------
# Shared argument groups
# --------------------------------------------------------------------------------------
def _int_list(text: str | None) -> tuple[int, ...]:
    if not text:
        return ()
    return tuple(int(part) for part in text.replace(" ", "").split(",") if part)


def add_config_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("system")
    g.add_argument(
        "-N",
        "--n-particles",
        type=int,
        default=_D.n_particles,
        help="number of spins (default: %(default)s)",
    )
    g.add_argument(
        "--density",
        type=float,
        default=_D.density,
        help="number density (irrelevant for the physics; default: %(default)s)",
    )
    g.add_argument("--box-length", type=float, default=None, help="cube edge; overrides --density")
    g.add_argument(
        "--boundary", choices=BOUNDARIES, default=_D.boundary, help="(default: %(default)s)"
    )
    g.add_argument(
        "--coupling",
        choices=tuple(COUPLINGS),
        default=_D.coupling,
        help="'gr' is the physical interaction; the others are controls (default: %(default)s)",
    )
    g.add_argument(
        "--spin-distribution",
        choices=SPIN_DISTRIBUTIONS,
        default=_D.spin_distribution,
        help="(default: %(default)s)",
    )
    g.add_argument("--spin-scale", type=float, default=_D.spin_scale)
    g.add_argument("--lognormal-sigma", type=float, default=_D.lognormal_sigma)
    g.add_argument("--G", type=float, default=_D.G, help="coupling constant (scales energies only)")
    g = parser.add_argument_group("numerics")
    g.add_argument(
        "--r-cut-nn",
        type=float,
        default=None,
        help="cutoff in units of r_nn (default: 10, or L/2 if smaller for periodic boundaries)",
    )
    g.add_argument("--n-sweeps", type=int, default=_D.n_sweeps, help="(default: %(default)s)")
    g.add_argument("--update-order", choices=UPDATE_ORDERS, default=_D.update_order)
    g.add_argument(
        "--protocol",
        choices=PROTOCOLS,
        default=_D.protocol,
        help="'anneal' = Metropolis annealing before the greedy quench",
    )
    g.add_argument("--anneal-t-start", type=float, default=_A.t_start)
    g.add_argument("--anneal-t-end", type=float, default=_A.t_end)
    g.add_argument("--anneal-n-temperatures", type=int, default=_A.n_temperatures)
    g.add_argument("--anneal-sweeps-per-temperature", type=int, default=_A.sweeps_per_temperature)
    g.add_argument("--anneal-step", type=float, default=_A.step)
    g = parser.add_argument_group("observables")
    g.add_argument(
        "--n-bins",
        type=int,
        default=_D.n_bins,
        help="radial bins on [0, x-max] r_nn (default: %(default)s)",
    )
    g.add_argument("--x-max", type=float, default=_D.x_max, help="(default: %(default)s)")
    g.add_argument(
        "--no-extra-observables", action="store_true", help="skip A, P, B (keep only C and Q)"
    )
    g.add_argument(
        "--nn-diagnostics", action="store_true", help="record nearest-neighbour diagnostics"
    )
    g.add_argument(
        "--snapshots",
        default="",
        help="comma-separated sweeps at which to record C(r), e.g. 1,2,5,10,20",
    )
    g.add_argument(
        "--save-states",
        type=int,
        default=0,
        help="store positions/spins of the first K realizations",
    )
    g = parser.add_argument_group("ensemble")
    g.add_argument(
        "-R", "--n-realizations", type=int, default=_D.n_realizations, help="(default: %(default)s)"
    )
    g.add_argument(
        "--base-seed",
        type=int,
        default=_D.base_seed,
        help="realizations use seeds base, base+1, ... (default: %(default)s)",
    )


def add_execution_arguments(parser: argparse.ArgumentParser) -> None:
    g = parser.add_argument_group("execution")
    g.add_argument(
        "-j",
        "--workers",
        type=int,
        default=1,
        help="worker processes (-1: all cores; default: %(default)s)",
    )
    g.add_argument(
        "--max-memory-gb",
        type=float,
        default=None,
        help="reduce the workers so that their estimated memory stays below this (default: half of the RAM; 0 disables)",
    )
    g.add_argument("--quiet", action="store_true", help="no progress output")


def config_from_args(args: argparse.Namespace) -> SimulationConfig:
    return SimulationConfig(
        n_particles=args.n_particles,
        density=args.density,
        box_length=args.box_length,
        boundary=args.boundary,
        coupling=args.coupling,
        spin_distribution=args.spin_distribution,
        spin_scale=args.spin_scale,
        lognormal_sigma=args.lognormal_sigma,
        G=args.G,
        r_cut_nn=args.r_cut_nn,
        n_sweeps=args.n_sweeps,
        update_order=args.update_order,
        protocol=args.protocol,
        anneal=AnnealConfig(
            t_start=args.anneal_t_start,
            t_end=args.anneal_t_end,
            n_temperatures=args.anneal_n_temperatures,
            sweeps_per_temperature=args.anneal_sweeps_per_temperature,
            step=args.anneal_step,
        ),
        n_bins=args.n_bins,
        x_max=args.x_max,
        extra_observables=not args.no_extra_observables,
        nn_diagnostics=args.nn_diagnostics,
        snapshots=_int_list(args.snapshots),
        save_states=args.save_states,
        n_realizations=args.n_realizations,
        base_seed=args.base_seed,
    )


def memory_limit_mb(args: argparse.Namespace) -> float | None:
    if args.max_memory_gb is None:
        available = available_memory_mb()
        return None if available is None else 0.5 * available
    return None if args.max_memory_gb <= 0 else args.max_memory_gb * 1024.0


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    return "\n".join(
        [
            fmt.format(*header),
            fmt.format(*("-" * w for w in widths)),
            *(fmt.format(*r) for r in rows),
        ]
    )


# --------------------------------------------------------------------------------------
# Commands: run / analyze / compare / protocol-check / benchmark
# --------------------------------------------------------------------------------------
def cmd_run(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    info = run_ensemble(
        config,
        args.out,
        n_workers=args.workers,
        progress=not args.quiet,
        overwrite=args.overwrite,
        memory_limit_mb=memory_limit_mb(args),
    )
    print(
        f"{info.n_computed} new realization(s), {info.n_total} planned, {info.elapsed_s:.1f} s -> {info.out_dir}"
    )
    if args.analyze:
        written = analyze_run(args.out, rebin=args.rebin, n_boot=args.n_boot)
        print(f"analysis written to {written['summary_md'].parent}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    fit_range = tuple(args.fit_range) if args.fit_range else None
    written = analyze_run(
        args.run_dir,
        args.out,
        rebin=args.rebin,
        n_boot=args.n_boot,
        fit=not args.no_fit,
        fit_range=fit_range,
        plots=not args.no_plots,
        formats=tuple(args.formats),
    )
    for key, path in written.items():
        print(f"{key:24s} {path}")
    print()
    print(written["summary_md"].read_text())
    return 0


def _parse_run_spec(spec: str) -> tuple[Path, int | None]:
    """``path`` or ``path:K`` (use only the first K realizations)."""
    path, sep, count = spec.rpartition(":")
    if sep and count.isdigit() and path:
        return Path(path), int(count)
    return Path(spec), None


def cmd_compare(args: argparse.Namespace) -> int:
    runs: dict[str, EnsembleData] = {}
    specs = [_parse_run_spec(s) for s in args.runs]
    labels = args.labels or [path.name for path, _ in specs]
    if len(labels) != len(specs):
        raise ValueError("--labels must have one entry per run")
    for label, (path, count) in zip(labels, specs, strict=True):
        runs[label] = EnsembleData.load(path, max_realizations=count)
    written = compare_runs(
        runs,
        args.out,
        tuple(args.obs),
        rebin=args.rebin,
        n_boot=args.n_boot,
        formats=tuple(args.formats),
    )
    for key, path in written.items():
        print(f"{key:40s} {path}")
    return 0


def cmd_protocol_check(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    info = run_protocol_check(
        config,
        args.out,
        n_workers=args.workers,
        progress=not args.quiet,
        overwrite=args.overwrite,
        memory_limit_mb=memory_limit_mb(args),
    )
    print(
        f"{info.n_computed} new realization(s), {info.n_total} planned, {info.elapsed_s:.1f} s -> {info.out_dir}"
    )
    if not args.no_analyze:
        written = analyze_protocol(args.out, rebin=args.rebin)
        import pandas as pd

        with pd.option_context("display.width", 200, "display.max_columns", 30):
            print(pd.read_csv(written["protocol_summary"]).round(4).to_string(index=False))
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    config = config_from_args(args)
    warmup(anneal=config.protocol == "anneal")
    est = config.estimate_cost()
    rows = []
    for k in range(args.repeats):
        record, _ = run_realization(config, config.base_seed + k)
        rows.append(record["timing_s"])
    mean = [sum(col) / len(rows) for col in zip(*rows, strict=True)]
    total = sum(mean)
    try:  # the resource module exists on Unix only
        import resource

        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mb = rss / 1e6 if sys.platform == "darwin" else rss / 1e3
    except ImportError:  # pragma: no cover
        rss_mb = float("nan")
    print(
        f"N={config.n_particles}, {config.boundary}, r_cut={config.r_cut_nn_resolved:g} r_nn, {config.n_sweeps} sweeps, "
        f"{config.neighbors_per_particle:.0f} neighbours/particle, {est['edges'] / 1e6:.2f} M directed edges"
    )
    print(
        f"measured per realization ({args.repeats} repeats): build {mean[0]:.2f} s, relax {mean[1]:.2f} s, "
        f"observables {mean[2]:.2f} s, total {total:.2f} s"
    )
    print(
        f"cost-model estimate:                      {est['seconds']:.2f} s   (measured/estimate = {total / est['seconds']:.2f})"
    )
    print(
        f"peak resident memory of this process:     {rss_mb:.0f} MB   (model: {est['memory_mb']:.0f} MB per worker)"
    )
    return 0


# --------------------------------------------------------------------------------------
# plan ...
# --------------------------------------------------------------------------------------
def _plan_root(args: argparse.Namespace) -> Path:
    if args.root:
        return Path(args.root)
    return planmod.SMOKE_ROOT if args.smoke else planmod.DEFAULT_ROOT


def cmd_plan_list(args: argparse.Namespace) -> int:
    plan = planmod.build_plan(args.smoke)
    rows = []
    for e in plan.entries:
        c = e.cost()
        rows.append(
            [
                e.id,
                e.priority,
                e.kind,
                str(e.n_realizations),
                f"{c['total_core_seconds'] / 60:.1f}",
                f"{c['memory_mb']:.0f}",
                e.describe(),
            ]
        )
    print(
        _table(rows, ["id", "priority", "kind", "realizations", "core-min", "MB/worker", "system"])
    )
    print()
    for level in ("required", "recommended", "all"):
        cost = planmod.plan_cost(plan.select([level]))
        print(
            f"{level:12s} {len(plan.select([level])):3d} runs  {int(cost['n_realizations']):5d} realizations  "
            f"{cost['core_hours']:.2f} core-hours (~{cost['core_hours'] * 60 / 8:.0f} min on 8 cores)"
        )
    print("\ncomparisons:", ", ".join(c.id for c in plan.comparisons))
    return 0


def cmd_plan_show(args: argparse.Namespace) -> int:
    plan = planmod.build_plan(args.smoke)
    for name in args.ids:
        entries = plan.select([name])
        for e in entries:
            print(f"== {e.id}: {e.title} ({e.priority}, {e.kind})")
            print(f"   purpose : {e.purpose}")
            print(f"   feeds   : {e.feeds}")
            print(f"   system  : {e.describe()}")
            print(f"   config  : {e.config.to_dict()}")
            print(f"   run     : grspin plan run {e.id}{' --smoke' if args.smoke else ''}")
    return 0


def cmd_plan_status(args: argparse.Namespace) -> int:
    plan = planmod.build_plan(args.smoke)
    root = _plan_root(args)
    statuses = planmod.plan_status(plan, root)
    rows = []
    for s in statuses:
        c = s.entry.cost()
        rows.append(
            [
                s.entry.id,
                s.entry.priority,
                f"{s.finished}/{s.total}",
                s.state,
                f"{c['total_core_seconds'] / 60:.1f}",
            ]
        )
    print(f"plan status under {root}/")
    print(_table(rows, ["id", "priority", "done", "state", "core-min"]))
    done = sum(1 for s in statuses if s.state == "done")
    req = [s for s in statuses if s.entry.priority == "required"]
    print(
        f"\n{done}/{len(statuses)} entries done; required: {sum(1 for s in req if s.state == 'done')}/{len(req)}"
    )
    stale = [s.entry.id for s in statuses if s.state == "stale-config"]
    if stale:
        print(
            f"warning: stored configuration differs from the plan for: {', '.join(stale)} "
            "(the plan changed after these runs; re-run with --overwrite or use another --root)"
        )
    return 0


def cmd_plan_run(args: argparse.Namespace) -> int:
    plan = planmod.build_plan(args.smoke)
    root = _plan_root(args)
    entries = plan.select(args.targets or ["required"])
    cost = planmod.plan_cost(entries)
    print(
        f"{len(entries)} entries, {int(cost['n_realizations'])} realizations, ~{cost['core_hours']:.2f} core-hours "
        f"-> {root}/"
    )
    if args.dry_run:
        for s in (planmod.entry_status(e, root) for e in entries):
            print(f"  {s.entry.id:22s} {s.finished}/{s.total} {s.state}")
        return 0
    limit = memory_limit_mb(args)
    for entry in entries:
        out = planmod.entry_dir(entry, root)
        print(f"\n### {entry.id}: {entry.title}")
        runner = run_protocol_check if entry.kind == "protocol" else run_ensemble
        runner(
            entry.config,
            out,
            n_workers=args.workers,
            progress=not args.quiet,
            overwrite=args.overwrite,
            memory_limit_mb=limit,
        )
    return 0


def _comparison_ready(
    comp: planmod.Comparison, plan: planmod.Plan, root: Path, allow_partial: bool
) -> str | None:
    """``None`` if all members are finished, else a message naming what is missing."""
    for m in comp.members:
        status = planmod.entry_status(plan.get(m.entry), root)
        needed = m.max_realizations or status.total
        if status.finished < (min(2, needed) if allow_partial else needed):
            return f"{m.entry} has {status.finished}/{needed} realizations"
    return None


def cmd_plan_analyze(args: argparse.Namespace) -> int:
    plan = planmod.build_plan(args.smoke)
    root = _plan_root(args)
    comp_ids = {c.id for c in plan.comparisons}
    names = args.targets or ["all"]
    entry_names = [n for n in names if n not in comp_ids and n != "comparisons"]
    entries = plan.select(entry_names) if entry_names else []
    comps = [
        c
        for c in plan.comparisons
        if c.id in names or "comparisons" in names or "all" in names or not args.targets
    ]
    for entry in entries:
        status = planmod.entry_status(entry, root)
        if status.finished == 0 or (status.finished < status.total and not args.allow_partial):
            print(f"skip {entry.id}: {status.finished}/{status.total} realizations")
            continue
        out = planmod.entry_dir(entry, root) / "analysis"
        if entry.kind == "protocol":
            analyze_protocol(planmod.entry_dir(entry, root), out, rebin=entry.rebin)
        else:
            analyze_run(planmod.entry_dir(entry, root), out, rebin=entry.rebin, n_boot=args.n_boot)
        print(f"analysed {entry.id} -> {out}")
    for comp in comps:
        missing = _comparison_ready(comp, plan, root, args.allow_partial)
        if missing:
            print(f"skip {comp.id}: {missing}")
            continue
        runs = {
            m.label: EnsembleData.load(
                planmod.entry_dir(plan.get(m.entry), root), max_realizations=m.max_realizations
            )
            for m in comp.members
        }
        out = root / "_comparisons" / comp.id
        compare_runs(
            runs,
            out,
            comp.observables,
            rebin=planmod.REBIN if not args.smoke else 2,
            n_boot=args.n_boot,
        )
        print(f"compared {comp.id} -> {out}")
    return 0


def cmd_plan_markdown(args: argparse.Namespace) -> int:
    text = planmod.plan_markdown(planmod.build_plan(False))
    if args.write:
        Path(args.write).write_text(text)
        print(f"wrote {args.write}")
    else:
        print(text, end="")
    return 0


# --------------------------------------------------------------------------------------
# Parser
# --------------------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="grspin",
        description="Relaxation of classical spins with the general-relativistic spin-spin interaction "
        "on random point sets, and the analysis of their pair correlations.",
    )
    parser.add_argument("--version", action="version", version=f"grspin {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    p = sub.add_parser("run", help="run (or resume/extend) an ensemble of realizations")
    add_config_arguments(p)
    add_execution_arguments(p)
    p.add_argument("--out", required=True, help="output directory")
    p.add_argument(
        "--overwrite", action="store_true", help="discard existing realizations in --out"
    )
    p.add_argument("--analyze", action="store_true", help="analyse the run afterwards")
    p.add_argument("--rebin", type=int, default=1)
    p.add_argument("--n-boot", type=int, default=200)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("analyze", help="tables, landmarks, fit and figures of a finished run")
    p.add_argument("run_dir")
    p.add_argument("--out", help="default: <run_dir>/analysis")
    p.add_argument(
        "--rebin",
        type=int,
        default=1,
        help="merge this many neighbouring bins (must divide n_bins)",
    )
    p.add_argument("--n-boot", type=int, default=200, help="bootstrap resamples over realizations")
    p.add_argument("--no-fit", action="store_true")
    p.add_argument("--fit-range", type=float, nargs=2, metavar=("XMIN", "XMAX"))
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--formats", nargs="+", default=["pdf"], help="figure formats (default: pdf)")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser(
        "compare", help="compare several runs (landmark table, overlay figure, paired differences)"
    )
    p.add_argument(
        "runs",
        nargs="+",
        help="run directories; use DIR:K for the first K realizations; first = reference",
    )
    p.add_argument("--labels", nargs="+")
    p.add_argument("--obs", nargs="+", default=["C", "Q", "A"], choices=["C", "Q", "A", "P", "B"])
    p.add_argument("--rebin", type=int, default=1)
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--formats", nargs="+", default=["pdf"])
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_compare)

    p = sub.add_parser(
        "protocol-check", help="compare preparations (update order, initial state, annealing)"
    )
    add_config_arguments(p)
    add_execution_arguments(p)
    p.add_argument("--out", required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-analyze", action="store_true")
    p.add_argument("--rebin", type=int, default=1)
    p.set_defaults(func=cmd_protocol_check)

    p = sub.add_parser("benchmark", help="time a few realizations and compare with the cost model")
    add_config_arguments(p)
    p.add_argument("--repeats", type=int, default=3)
    p.set_defaults(func=cmd_benchmark)

    plan = sub.add_parser("plan", help="the simulation plan: list, status, run, analyze")
    psub = plan.add_subparsers(dest="plan_command", metavar="<plan-command>", required=True)

    def plan_parser(name: str, help_: str, func) -> argparse.ArgumentParser:
        q = psub.add_parser(name, help=help_)
        q.add_argument(
            "--smoke",
            action="store_true",
            help="tiny version of the plan (default root results/_smoke)",
        )
        q.add_argument(
            "--root", help="results directory (default: results, or results/_smoke with --smoke)"
        )
        q.set_defaults(func=func)
        return q

    plan_parser("list", "all entries with their estimated cost", cmd_plan_list)
    q = plan_parser("show", "details of entries/groups", cmd_plan_show)
    q.add_argument("ids", nargs="+")
    plan_parser("status", "progress of every entry, from the files on disk", cmd_plan_status)
    q = plan_parser(
        "run",
        "run entries (default: the required ones); finished realizations are skipped",
        cmd_plan_run,
    )
    q.add_argument("targets", nargs="*", help="ids, groups (S2), required | recommended | all")
    add_execution_arguments(q)
    q.add_argument("--dry-run", action="store_true")
    q.add_argument("--overwrite", action="store_true")
    q = plan_parser("analyze", "analyse finished entries and comparisons", cmd_plan_analyze)
    q.add_argument(
        "targets",
        nargs="*",
        help="ids, groups, comparison ids (cmp_*), 'comparisons' or nothing for everything",
    )
    q.add_argument("--n-boot", type=int, default=200)
    q.add_argument("--allow-partial", action="store_true", help="also analyse unfinished entries")
    q = psub.add_parser("markdown", help="print (or write) docs/SIMULATION_PLAN.md")
    q.add_argument("--write", metavar="PATH")
    q.set_defaults(func=cmd_plan_markdown)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "func", None) is None:
        parser.print_help()
        return 0
    started = time.perf_counter()
    try:
        status = int(args.func(args) or 0)
    except (ConfigMismatchError, ValueError, FileNotFoundError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    if getattr(args, "command", None) in ("analyze", "compare"):
        print(f"({time.perf_counter() - started:.1f} s)", file=sys.stderr)
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
