# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-10-06

First public-ready release. It consolidates the earlier prototype scripts into a
tested Python package.

### Added
- `grspin` package: numba-accelerated sequential relaxation of fixed-length spins in the
  local GR spin-spin field, on Poisson point sets with **periodic** (minimum image) or
  open boundaries, spherical cutoff in units of the mean nearest-neighbour distance.
- Exact incremental energy bookkeeping (no extra pass over the neighbour list per sweep).
- Pair observables `C(r)`, `Q(r)` (as in the manuscript) plus the axis alignment `A(r)`,
  the bond-projected correlation `P(r)` and the tensor (pair-energy) correlation `B(r)`;
  total-spin diagnostics `|sum S|` and `S(0)`.
- Nearest-neighbour diagnostics (mutual-NN fraction, dimer / non-dimer decomposition,
  rank-resolved correlations, small-distance scaling, NN-distance histogram).
- Control couplings (isotropic 1/r^3 antiferromagnet, sign-reversed dipolar tensor),
  alternative spin-magnitude distributions, random update order and simulated annealing.
- Resumable / extendable ensemble runner with one `.npz` file per realization and full
  provenance (`config.json`).
- Analysis tools: exact re-binning, mean-of-means ensemble statistics, bootstrap over
  realizations, model-independent landmarks, paired differences between runs that share seeds,
  multi-start damped-oscillatory fit.
- Publication-style figures (APS column widths, 8 pt text) and per-run reports (`summary.md`).
- Memory-aware parallel execution: the number of worker processes is limited by the estimated memory
  of a realization (default: half of the RAM).
- Command line interface `grspin` (`run`, `analyze`, `compare`, `plan`, `protocol-check`, `benchmark`).
- The simulation plan as code and as a checklist (`docs/SIMULATION_PLAN.md`).
- Regression tests against the earlier prototype implementation (agreement of the ensemble
  statistics and energy histories to round-off for open boundaries).

### Changed with respect to the prototype scripts
- Periodic boundary conditions are the default; the neighbour search and all pair
  statistics use a KD-tree, so memory is `O(N)` instead of `O(N^2)`.
- The mutual-nearest-neighbour probability `exp(-9 pi rho r^3 / 4)` is reported next to the
  isolated-sphere probability `exp(-4 pi rho r^3 / 3)`.
