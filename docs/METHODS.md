# Methods

This document defines the model, the algorithm, the observables and the statistics implemented in
`grspin`, and records how the numerical results are validated. Section numbers are referenced from the
source code.

## 1. Model

`N` classical spins `S_i = s_i u_i` (fixed length `s_i`, unit direction `u_i`) sit at fixed positions
`r_i`. The general-relativistic spin-spin interaction energy of a pair is

```
U_ij = -(G / r^3) [ S_i . S_j - 3 (S_i . rhat)(S_j . rhat) ]  = -S_i^T T_ij S_j,
T_ij = (G / r^3) (1 - 3 rhat rhat^T),            r = |r_i - r_j|,  rhat = (r_i - r_j) / r,
```

and the many-body Hamiltonian is `H = sum_{i<j} U_ij`. The tensor `T_ij` is traceless; its eigenvalue is
`-2G/r^3` along the bond and `+G/r^3` perpendicular to it. An isolated pair is minimized by antiparallel
spins along the bond, `U_min = -2 G s_i s_j / r^3`, i.e. `u_i . u_j = -1`.

The code implements the family

```
H = -sigma G sum_{i<j} r_ij^-3 [ S_i . S_j - gamma (S_i . rhat_ij)(S_j . rhat_ij) ]
```

with `(sigma, gamma)`:

| `coupling` | `(sigma, gamma)` | meaning |
|---|---|---|
| `gr` | `(+1, 3)` | the general-relativistic interaction (the physical case) |
| `isotropic_afm` | `(-1, 0)` | isotropic `1/r^3` antiferromagnet - control without tensor structure |
| `dipolar` | `(-1, 3)` | sign-reversed tensor, i.e. the magnetic dipole-dipole energy - control |

### Units and scale invariance

The relaxation rule (section 2) only uses the *direction* of the local field. Consequently the final
directions do not depend on `G`, on the spin scale `S`, on the box size or on the density (these only
rescale energies and lengths); this is verified numerically (`tests/test_kernels.py`). The physically
independent parameters are `N`, the cutoff in units of `r_nn`, the number of sweeps, the *shape* of the
distribution of spin lengths, the boundary condition and the coupling. All distances are therefore
reported in units of the mean nearest-neighbour distance of the Poisson process,

```
r_nn = Gamma(4/3) (3 / (4 pi rho))^(1/3) = 0.55396 rho^(-1/3),      x = r / r_nn.
```

## 2. Zero-temperature relaxation

The part of `H` that depends on `S_i` is `-sigma S_i . h_i` with the local field

```
h_i = sum_{j != i, r_ij < r_cut}  G r_ij^-3 [ S_j - gamma (S_j . rhat_ij) rhat_ij ].
```

For fixed length `s_i` the minimum over the direction of `S_i` is attained by `S_i = s_i sigma h_i / |h_i|`
(Lagrange multiplier argument; `|h_i| = 0` or `s_i = 0` leaves the spin unchanged). One *sweep* applies
this update to all spins **sequentially and in place** (Gauss-Seidel): later spins in a sweep already see
the updated earlier spins. Optionally (`update_order="random"`) the visiting order is a fresh random
permutation in every sweep.

Properties used and tested:

* every single update is an exact minimization over one direction, so the energy never increases;
* the energy is accumulated *incrementally* from the per-spin changes, `dE_i = -sigma (S_i^new - S_i^old) . h_i`,
  and agrees with a direct evaluation to round-off (`energy_drift` in every record, ~1e-15);
* the iteration is **not** a dynamics. It finds a low-energy *metastable* configuration; different
  update orders or initial conditions lead to different but statistically equivalent states
  (quantified by the protocol check, section 8);
* convergence is monitored through `(E_t - E_final)/|E_final|` and the largest change of any unit
  direction per sweep (`histories.csv`, `histories.pdf`). The production plan uses 100 sweeps; set `S1`
  documents the convergence of `C(r)` up to 500 sweeps.

`protocol="anneal"` precedes the greedy quench by Metropolis single-spin annealing at fixed spin lengths
(Gaussian proposal of width `anneal.step` on the unit direction, geometric temperature schedule in units
of `G S^2 rho`).

## 3. Geometry, boundaries and cutoff

Positions are uniform in the cube `[0, L)^3`, `L = (N / rho)^(1/3)`. Two boundary conditions are available:

* `periodic` (default): minimum-image convention; each pair interacts once, through its nearest image.
  This requires `r_cut <= L/2` (and `x_max r_nn <= L/2` for the observables); the configuration validates this
  and tells you the minimum `N` if it is violated. If `r_cut_nn` is not given it is `min(10, L/(2 r_nn))`.
* `open`: no images; spins near the faces have fewer neighbours, which biases the correlations near the
  boundary (set `S3` quantifies this).

The interaction is truncated sharply at `r_cut` (spherical cutoff; no Ewald summation). The tensor
`1 - 3 rhat rhat^T` averages to zero over the sphere, so the truncated tail has no mean-field part;
the cutoff dependence is measured in set `S2`.

Neighbour search uses a `scipy.spatial.cKDTree` (with `boxsize` for periodic boundaries); the neighbour
lists are stored in CSR form with the weights `r^-3` and unit vectors, so a sweep is a single pass over
contiguous arrays inside a numba kernel. Memory is `O(N)` apart from the neighbour list itself
(`E = N * (4 pi / 3) rho r_cut^3` directed edges).

## 4. Observables

For a pair `(i, j)` at separation `r` (`x = r / r_nn`) with unit directions `u_i`, `u_j` and bond direction
`rhat`, all pairs with `r < x_max r_nn` are binned in `x`:

| symbol | definition | random spins | perfect dimer |
|---|---|---:|---:|
| `C` | `u_i . u_j` | 0 | -1 |
| `Q` | `cos^2 phi`, `phi` = angle between `S_i + S_j` and `rhat` | 1/3 | 1 (unequal lengths) |
| `A` | `[(u_i . rhat)^2 + (u_j . rhat)^2] / 2` | 1/3 | 1 |
| `P` | `(u_i . rhat)(u_j . rhat)` | 0 | -1 |
| `B` | `C - 3 P`; the GR pair energy is `U_ij = -B G s_i s_j / r^3` | 0 | 2 |

* `C` and `Q` are the observables of the manuscript. `Q` is a property of the *pair total spin* and
  therefore depends on the distribution of the spin lengths: for equal lengths an antiparallel pair has
  `S_i + S_j = 0` and drops out of the average, so `Q -> 0` (not 1) at short distance (set `S4`).
  `A` measures the locking of a single spin to the bond axis and does not have this problem.
* `P` and `B` split the anti-alignment into its longitudinal and transverse parts and give the
  *pair-energy correlation*, a quantitative measure of frustration (`B` falls from the dimer value 2 to 0).
* Pairs containing a zero-length spin are excluded from `C, A, P, B`; pairs with `S_i + S_j = 0` are
  excluded from `Q`.
* Global diagnostics: `|sum_i S_i|` (reported as `|sum S|/sqrt(N)`) and `S(0) = |sum_i u_i|^2 / N` of the
  unit spins (`~1` for uncorrelated directions, `~N` for a ferromagnet).

The runner stores **sums and counts per bin** for the initial and the final state (and for requested
intermediate sweeps), not means. This allows exact re-binning by any integer divisor of `n_bins`
(`rebin=` in the analysis) and pooled estimates.

## 5. Ensemble statistics

* **Mean over realizations.** As in the manuscript, the binned mean of an observable is first formed over
  all pairs in the bin *within each realization*, then averaged over realizations (`analysis.mean_sem`);
  realizations without pairs in a bin are skipped.
* **Error bars** are standard errors over realizations, `std(ddof=1) / sqrt(n)`. They do not include the
  spread of pair values inside a bin.
* **Bootstrap.** Landmarks, fit parameters and the nearest-neighbour tables get uncertainties by resampling
  *realizations* with replacement (`n_boot` times, default 200) and repeating the whole computation.
* **Landmarks** (model-independent features of the mean curves): the first upward zero `x_0` of `C`
  (linear interpolation), the position and height of the positive peak (parabola through the highest bin
  after `x_0` and its neighbours), the minimum of `Q`, the `x` at which `Q` falls below `1/3`, and the `x` at
  which `A` falls below `1/2`.
* **Paired differences.** Runs that share the system and `base_seed` have identical positions and initial
  directions; `analysis.paired_difference` forms `obs_other - obs_reference` per seed before averaging, which
  removes most of the sampling noise. `grspin compare` writes these for every run against the first one.

## 6. Nearest-neighbour (dimer) picture

For a Poisson process the probability that a ball of radius `r` around a particle is empty is
`exp(-4 pi rho r^3 / 3)`. The probability that a pair at separation `r` is a **mutual** nearest-neighbour
pair is `exp(-9 pi rho r^3 / 4)`: neither member may have another particle within `r`, and the union of the two
balls (centres `r` apart) has volume `2 (4 pi/3) r^3 - 5 pi r^3 / 12 = 9 pi r^3 / 4`. The fraction of particles that belong
to a mutual-NN pair is `16/27`. Both curves are drawn next to `C(r)`; the diagnostics
(`nn_diagnostics=True`) measure, per radial bin and pooled over all pairs,

* the fraction of mutual-NN pairs (`f_mutual`), versus the theory above,
* `C` separately for mutual-NN pairs, pairs that are the NN of exactly one member (`oneway`) and pairs that are
  nobody's NN (`far`),
* the rank-resolved correlation `C_k` between a spin and its `k`-th nearest neighbour (`k = 1..8`),
* the histogram of first-neighbour distances against the Poisson law, and
* the small-distance law of `1 + C(r)` in log-spaced bins.

## 7. Damped-oscillatory description

`C_fit(x) = -exp(-alpha u^3) cos(k u + phi)` with `u = rho^(1/3) r = 0.55396 x`. It is **phenomenological**:
it describes the negative core, the sign change and a positive lobe, but it leaves systematic residuals
(`chi^2/dof >> 1` for production-size ensembles) and its three parameters are only weakly determined (several
nearby `chi^2` minima, dependence on the fit range). The fit therefore starts from a grid of 75 initial
guesses, keeps the lowest `chi^2` (weights `1/SEM^2`), restricts `alpha, k in [0, 50]`, `phi in [-2 pi, 2 pi]`, and
reports bootstrap percentiles. Quote the landmarks of section 5 for quantitative statements.

## 8. Controls, variants and the protocol check

* **Control couplings** (`coupling="isotropic_afm"`, `"dipolar"`) use exactly the same code and seeds
  (set `S6`); they show which features need the GR tensor.
* **Spin-length distributions:** `uniform` `U(0, S)` (manuscript), `uniform_half` `U(S/2, 3S/2)`, `equal`,
  `lognormal` (`S exp(sigma Z)`).
* **Protocol check** (`grspin protocol-check`, set `S5`): on identical positions and lengths it compares the
  production protocol with random update order, with an independent isotropic initial direction field and
  with Metropolis annealing followed by the quench. Reported per realization and paired: the final energy, the
  overlaps `<u . u'>` and `<(u . u')^2>` with the production state, and the binned pair sums, summarized as
  paired differences with standard errors.

## 9. Output files

```
<run>/config.json              kind, configuration, provenance (versions, platform, git commit), run history
<run>/seeds/seed_XXXXXXXX.npz  one file per realization
<run>/states/seed_XXXXXXXX.npz positions, magnitudes, spins_initial, spins_final (first `save_states` seeds)
<run>/analysis/...             written by `grspin analyze`
```

Keys of a realization file (`n` = number of bins, `T` = number of sweeps, `K` = number of snapshots):

| key | shape | content |
|---|---|---|
| `init_*`, `fin_*` | | `cnt_C, cnt_Q` and `sum_C, sum_Q, sum_A, sum_P, sum_B, sum_x` (`(n,)`); `total_spin`, `s0_unit` (scalars) |
| `snap_*`, `snap_sweeps` | `(K, n)`, `(K,)` | the same sums at the requested intermediate sweeps |
| `energy`, `dmax`, `total_spin_hist` | `(T+1,)` | energy (with `G`), largest direction change, `|sum S|` after each sweep |
| `energy_drift` | scalar | relative difference between incremental and direct final energy |
| `nn_*`, `sr_*` | | nearest-neighbour diagnostics (section 6), if requested |
| `seed`, `timing_s`, `elapsed_s` | | provenance and timing (build / relax / observables) |

Re-running the same command skips finished realizations (resume); raising `n_realizations` extends the
ensemble; a different configuration in an existing directory is refused. Files are written atomically.

## 10. Reproducibility

* Realization `k` of an ensemble uses seed `base_seed + k` and nothing else; results do not depend on the
  number of worker processes (tested).
* The random stream (positions, one discarded draw for particle masses, isotropic directions, spin lengths) is
  identical to that of the exploratory prototype scripts, so historic realizations are reproduced seed by
  seed: `tests/test_legacy_regression.py` checks that the ensemble statistics and energy histories agree
  with reference output of the prototype (open boundaries) to round-off, i.e. a relative `1e-9`
  (`tests/data/README.md`).
* The numba kernels use strict IEEE arithmetic (no `fastmath`); a fast-math variant is ~28 % faster per sweep
  but was not adopted. Bitwise equality across different CPU architectures is not guaranteed, statistical
  equality is.
* `config.json` records the package version, the versions of Python, numpy, scipy and numba, the platform and
  the git commit (with a dirty flag) of the code that produced the data.

## 11. Cost

Per realization: neighbour-list assembly `~2.5e-7 s` per directed edge, `~2.3e-9 s` per edge and sweep, and a
small term for the pair statistics; peak memory `~250 MB + 220 B` per edge. Example (Apple-silicon laptop, one
core, `grspin benchmark`):

| system | edges | measured | memory (peak RSS) |
|---|---:|---:|---:|
| `N=5000`, `r_cut=10 r_nn`, 100 sweeps | 3.6 M | 1.7 s | 1.0 GB |
| `N=5000`, `r_cut=15 r_nn`, 100 sweeps | 12.0 M | 6.1 s | 2.1 GB |
| `N=20000`, `r_cut=7 r_nn`, 100 sweeps | 4.9 M | 2.8 s | 1.4 GB |

Time scales as `N * r_cut^3 * sweeps`. `grspin` limits the number of workers so that their estimated memory
stays below half of the RAM (`--max-memory-gb` to change).

## 12. Validation

`pytest` (about 15 s) covers, among others:

* the two-body ground states of all three couplings (energy `-2 G s s' / r^3` etc.);
* monotone energy, exactness of the incremental energy, fixed-point property of the relaxed state,
  conservation of spin lengths, invariance under rescaling of `G`, `S`, `L`;
* the periodic neighbour list against `O(N^2)` minimum-image enumeration; open versus periodic;
* all pair observables against a brute-force reference (and the prototype's logic), dimer values, random-spin
  baselines, exactness of re-binning;
* determinism, worker-count independence, resume/extend/overwrite and the refusal of mismatching configurations;
* the Poisson results of section 6 (mean NN distance, `16/27`, `exp(-9 pi rho r^3 / 4)`, Kolmogorov distance of the
  NN-distance law, union volume of two balls);
* bootstrap, landmarks, fit recovery on synthetic data, paired differences;
* the plan (consistency, pairing, selection), its status tracking on disk, the generated plan document and the
  command line interface;
* regression against the prototype implementation.

## 13. Limitations

* Spins are classical, of fixed length and at fixed positions; there is no dynamics, temperature or backreaction
  of the spin texture on the positions.
* The relaxation finds *a* local minimum; protocol dependence is small for `C` and `Q` (set `S5`) but the
  states are not ground states.
* The interaction is truncated at `r_cut <= L/2` (no Ewald sum for the conditionally convergent `1/r^3`
  tensor); the dependence is measured, not eliminated.
* Positions are an ideal Poisson process; real galaxy samples have clustering, which is outside the scope of
  this code.
