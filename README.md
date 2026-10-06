# gr-spin-frustration

[![CI](https://github.com/kdemirel/gr-spin-frustration/actions/workflows/ci.yml/badge.svg)](https://github.com/kdemirel/gr-spin-frustration/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)

Zero-temperature relaxation of classical spins fixed at random (Poisson) positions and coupled by the
**general-relativistic spin-spin interaction**, and the ensemble analysis of their pair correlations.
Two spins interacting through this tensor force prefer to be antiparallel along the line joining them;
in a random many-body geometry these pairwise preferences cannot all be satisfied, and the relaxed state is a
geometrically frustrated spin texture. This package simulates that state and measures it.

The code accompanies a manuscript in preparation.

<p align="center">
  <img src="docs/images/example_overview.png" width="760" alt="Example output: C(r), Q(r), A(r) and B(r)">
</p>

*Example output of `grspin` for a modest demo run (N = 2000, 24 realizations; regenerate with
`scripts/make_readme_figure.py`). It is not the production ensemble of the paper.*
Distances are in units of the mean nearest-neighbour distance `r_nn`; gray: random initial spins, blue: relaxed;
dashed/dotted red: nearest-neighbour references.

## Physics in brief

For classical spins `S_i` (fixed length, free direction) the pair energy is

```
U_ij = -(G / r^3) [ S_i . S_j - 3 (S_i . rhat)(S_j . rhat) ],     H = sum_{i<j} U_ij
```

The ground state of an isolated pair is `u_i . u_j = -1` with both spins along the bond. The relaxation
sets every spin, sweep after sweep, to the direction of its local field `h_i = sum_j T_ij S_j` (an exact
minimization over that spin's direction, so the energy never increases) until a low-energy metastable
configuration is reached. The measured observables are the radial spin correlation `C(r) = <u_i . u_j>`, the
axis locking of the pair total spin `Q(r) = <cos^2 phi>` and, in addition, the single-spin axis alignment
`A(r)` and the tensor (pair-energy) correlation `B(r)`. Details, definitions and validation: [docs/METHODS.md](docs/METHODS.md).

## Installation

```bash
git clone https://github.com/kdemirel/gr-spin-frustration.git
cd gr-spin-frustration
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -e ".[dev]"                                # numpy, scipy, pandas, matplotlib, numba (+ pytest, ruff)
pytest -q                                              # ~15 s; numba compiles the kernels on first use
```

## Quick start

### Command line

```bash
# 20 realizations of 5000 spins, 100 sweeps, 4 worker processes; analyse afterwards
grspin run -N 5000 -R 20 --n-sweeps 100 --out results/demo -j 4 --analyze

# the same run, extended to 100 realizations (finished ones are skipped)
grspin run -N 5000 -R 100 --n-sweeps 100 --out results/demo -j 4

# control experiment on identical geometries: sign-reversed tensor (dipolar magnet)
grspin run -N 5000 -R 20 --n-sweeps 100 --coupling dipolar --out results/demo_dipolar -j 4
grspin compare results/demo results/demo_dipolar --labels GR dipolar --out results/compare
```

| command | what it does |
|---|---|
| `grspin run` | run, resume or extend an ensemble of independent realizations |
| `grspin analyze` | tables, model-independent landmarks with bootstrap errors, damped-oscillatory fit, figures |
| `grspin compare` | overlay figure, landmark table and paired differences of several runs |
| `grspin protocol-check` | compare update orders, initial states and Metropolis annealing on identical systems |
| `grspin benchmark` | time one realization and compare with the cost model |
| `grspin plan ...` | the simulation plan: `list`, `show`, `status`, `run`, `analyze`, `markdown` |

`grspin <command> --help` lists all options.

### Python

```python
from grspin import SimulationConfig
from grspin.analysis import EnsembleData
from grspin.ensemble import run_ensemble
from grspin.plotting import figure_overview, save_figure


def main():
    config = SimulationConfig(n_particles=2000, n_sweeps=100, n_bins=80, n_realizations=24)
    run_ensemble(config, "results/example", n_workers=-1)

    data = EnsembleData.load("results/example")
    print(data.table(rebin=2).head())  # pandas DataFrame, one row per radial bin
    print(data.landmarks(rebin=2)["C_peak"])  # value and bootstrap error of the positive peak
    save_figure(figure_overview(data, rebin=2), "overview.pdf")


# Required when n_workers > 1: macOS and Windows start worker processes by spawning.
if __name__ == "__main__":
    main()
```

A notebook version is in [notebooks/quickstart.ipynb](notebooks/quickstart.ipynb).

## The simulation plan

Every run needed for the manuscript is defined in code ([src/grspin/plan.py](src/grspin/plan.py)) and listed
as a checklist in **[docs/SIMULATION_PLAN.md](docs/SIMULATION_PLAN.md)**:

| set | purpose |
|---|---|
| `M` | main ensemble: `C, Q, A, B`, landmarks, fit, nearest-neighbour diagnostics |
| `S1` | convergence with the number of relaxation sweeps |
| `S2` | dependence on the interaction cutoff |
| `S3` | finite size and open versus periodic boundaries |
| `S4` | distribution of spin lengths |
| `S5` | protocol / metastability check (update order, initial state, annealing) |
| `S6` | control couplings (isotropic antiferromagnet, sign-reversed tensor) |

```bash
grspin plan list                      # entries with estimated cost
grspin plan run all --smoke -j 4      # the whole plan in miniature (~30 s): test the pipeline first
grspin plan run required -j 8         # production runs (finished realizations are skipped)
grspin plan status                    # progress from the files on disk
grspin plan analyze                   # analysis of everything that is finished, plus comparisons
```

## Output layout

```
results/<id>/config.json               configuration, provenance (versions, platform, git commit), run history
results/<id>/seeds/seed_XXXXXXXX.npz   one file per realization: binned sums/counts, histories, diagnostics
results/<id>/states/...                optional positions and spins of the first realizations
results/<id>/analysis/                 summary.md/json, ensemble_table.csv, landmarks, fit.json, figures (pdf/png)
results/_comparisons/<id>/             comparison tables, overlays and paired differences
```

Results are not committed (`results/` is git-ignored); `config.json` plus the code version reproduce them.

## Performance

Cost per realization is dominated by the neighbour list (`E = N (4 pi / 3) rho r_cut^3` directed edges):
about `2.5e-7 s` per edge to build and `2.3e-9 s` per edge and sweep, with `~250 MB + 220 B` per edge of
peak memory. On a laptop core, `N = 5000`, `r_cut = 10 r_nn`, 100 sweeps take ~2 s; the complete plan is
roughly 0.7 core-hours. `grspin run` limits the workers so that their estimated memory stays below half of the
RAM (`--max-memory-gb`).

## Reproducibility

* realization `k` uses seed `base_seed + k`; results are independent of the number of workers;
* the random stream matches the earlier prototype scripts, and a regression test ties the package to them;
* every `config.json` records package/library versions, the platform and the git commit of the code;
* the kernels use strict IEEE arithmetic (no `fastmath`).

## Development

```bash
pytest -q                 # test suite
ruff check . && ruff format --check .
grspin plan markdown --write docs/SIMULATION_PLAN.md   # after changing src/grspin/plan.py
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [CHANGELOG.md](CHANGELOG.md).

## Project layout

```
src/grspin/
    config.py        configuration, units, cost model
    initial.py       Poisson positions, isotropic spins, spin-length distributions
    neighbors.py     CSR neighbour lists, periodic (minimum image) or open
    kernels.py       numba: sweeps, energy, Metropolis
    relax.py         relaxation driver (quench, annealing), incremental energy
    observables.py   C, Q, A, P, B and global spin diagnostics
    nn.py, theory.py nearest-neighbour diagnostics and exact Poisson results
    ensemble.py      parallel, resumable ensemble runner
    analysis.py      statistics over realizations, bootstrap, landmarks, paired differences
    fit.py           damped-oscillatory description of C(r)
    protocols.py     protocol / metastability check
    plan.py          the simulation plan (data), status, checklist generator
    plotting.py, report.py, cli.py
tests/               pytest suite (~15 s)
docs/                METHODS.md, SIMULATION_PLAN.md
notebooks/           quickstart notebook
scripts/             regeneration of the README figure
```

## Citation

If you use this code, please cite it via [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository");
the article will be added once published.

## License

MIT, see [LICENSE](LICENSE).
