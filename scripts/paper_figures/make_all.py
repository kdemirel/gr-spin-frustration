#!/usr/bin/env python3
"""Run every figure script of the PRL manuscript.

    python3 scripts/paper_figures/make_all.py [--results-root R] [--out DIR] [--entry-map KEY=ID ...]
                                              [--only fig2 figS3 ...] [--skip ...]

The common options are passed to every data-figure script; fig1_schematic.py (no data) gets only
the output directory.  Each script runs in its own process; failures are reported at the end and
make the exit status 1, but do not stop the other figures.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS = [
    "fig1_schematic.py",
    "fig2_correlations.py",
    "fig3_dimer_map.py",
    "figS1_sweeps.py",
    "figS2_cutoff.py",
    "figS3_size_boundary.py",
    "figS4_magnitudes.py",
    "figS5_controls.py",
    "figS6_nn.py",
    "figS7_energy.py",
    "figS8_structure_factor.py",
]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results-root", type=Path, default=None)
    p.add_argument("--out", type=Path, default=None)
    p.add_argument("--entry-map", nargs="*", default=[], metavar="KEY=ID")
    p.add_argument("--dpi", type=int, default=None)
    p.add_argument("--only", nargs="*", default=None, help="script stems to run (e.g. fig2 figS3)")
    p.add_argument("--skip", nargs="*", default=[], help="script stems to skip")
    args = p.parse_args()

    here = Path(__file__).resolve().parent
    common: list[str] = []
    if args.results_root is not None:
        common += ["--results-root", str(args.results_root)]
    if args.out is not None:
        common += ["--out", str(args.out)]
    if args.entry_map:
        common += ["--entry-map", *args.entry_map]
    if args.dpi is not None:
        common += ["--dpi", str(args.dpi)]

    def selected(name: str) -> bool:
        stem = name[: -len(".py")]
        if args.only and not any(stem == s or stem.startswith(s + "_") or stem == s + ".py" for s in args.only):
            return False
        return not any(stem == s or stem.startswith(s + "_") for s in args.skip)

    failures: list[tuple[str, int]] = []
    for name in SCRIPTS:
        if not selected(name):
            continue
        script = here / name
        if not script.is_file():
            print(f"=== {name}: not found, skipped")
            continue
        if name == "fig1_schematic.py":
            cmd = [sys.executable, str(script)] + ([str(args.out)] if args.out is not None else [])
        else:
            cmd = [sys.executable, str(script), *common]
        print(f"=== {name}")
        sys.stdout.flush()
        rc = subprocess.call(cmd, cwd=str(here))
        if rc != 0:
            failures.append((name, rc))
            print(f"=== {name}: FAILED (exit {rc})")
    print("=== summary")
    if failures:
        for name, rc in failures:
            print(f"  FAILED  {name} (exit {rc})")
        return 1
    print("  all figures written")
    return 0


if __name__ == "__main__":
    sys.exit(main())
