"""Regenerate the example figure shown in the README (a modest demo run, not a production result).

Usage:  python scripts/make_readme_figure.py [WORKDIR]
"""

from __future__ import annotations

import sys
from pathlib import Path

from grspin import SimulationConfig
from grspin.analysis import EnsembleData
from grspin.ensemble import run_ensemble
from grspin.plotting import figure_overview, save_figure

REBIN = 2


def main() -> None:
    work = Path(sys.argv[1] if len(sys.argv) > 1 else "results/_demo")
    config = SimulationConfig(
        n_particles=2000, n_sweeps=100, n_bins=80, n_realizations=24, base_seed=1000
    )
    run_ensemble(config, work, n_workers=-1)
    data = EnsembleData.load(work)
    target = Path(__file__).resolve().parent.parent / "docs" / "images" / "example_overview.png"
    save_figure(figure_overview(data, rebin=REBIN), target)
    print(f"wrote {target}")


if __name__ == "__main__":
    main()
