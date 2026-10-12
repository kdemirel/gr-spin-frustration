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
| `transverse_only` | `(+1, 1)` | `U = -(G/r^3) S_perp,i . S_perp,j`: the ferromagnetic transverse channel alone - channel control |
| `gamma12` | `(+1, 12)` | longitudinal : transverse weights 11 : 1 - channel control |
| `longitudinal_only` | `(+1, 100)` | weights 99 : 1, the longitudinal channel alone up to a 1 % admixture - channel control |

**Channel family.** In the components along the bond and transverse to it (section 4) the pair
energy of the family is `U_ij = -(sigma G / r^3) [S_perp,i . S_perp,j - (gamma - 1) S_r,i S_r,j]`,
so `gamma - 1` is the weight of the longitudinal channel (antiferromagnetic for `sigma = +1`)
relative to the transverse one (ferromagnetic): 0 for `transverse_only`, 2 for GR, 11 and 99 for the
two other channel controls (`config.CHANNEL_FAMILY`; set `S7`, section 8.1). The relaxation of
section 2 uses only the direction of the local field, so the overall scale of `H` is irrelevant at
`T = 0`: `gamma = 100` is the longitudinal-only limit up to a 1 % transverse admixture (a purely
longitudinal coupling is `gamma -> infinity`). The **two-body minimum** of the whole family is

```
U_min = -G s_i s_j r^-3 max(1, |gamma - 1|)        for either sign of sigma,
```

because the pair form is `u_i^T M u_j` with `M = 1 - gamma rhat rhat^T` (eigenvalues 1, 1 and
`1 - gamma`), whose minimum over both directions is minus the spectral norm of `M`, the sign of
`sigma` being absorbed by flipping one spin: `-2 w s_i s_j` antiparallel along the bond for GR (and
parallel along the bond for the dipolar control), `-w s_i s_j` with parallel spins transverse to the
bond for `gamma = 1`, `-w s_i s_j` for the isotropic antiferromagnet, `-11 w s_i s_j` and
`-99 w s_i s_j` antiparallel along the bond for `gamma = 12` and 100 (`config.pair_minimum_factor`,
verified numerically in `tests/test_config.py`; `theory.two_body_minimum_factor(sigma, gamma)` is
the same number). These are the reference energies of the frustration index (energy divided by
the sum of the two-body minima, section 8.8) for every coupling.

### Units and scale invariance

The relaxation rule (section 2) only uses the *direction* of the local field. Consequently the final
directions do not depend on `G`, on the spin scale `S`, on the box size or on the density (these only
rescale energies and lengths); this is verified numerically (`tests/test_kernels.py`). The physically
independent parameters are `N`, the cutoff in units of `r_nn`, the number of sweeps, the *shape* of the
distribution of spin lengths, the boundary condition, the coupling and the hard-core distance
`min_separation_nn` (section 3). All distances are therefore reported in units of the mean
nearest-neighbour distance of the ideal Poisson process at the same density,

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

Positions are uniform in the cube `[0, L)^3`, `L = (N / rho)^(1/3)`: an ideal Poisson process. With
`min_separation_nn = x_min > 0` (set `S8`: `x_min = 0.5`) they are instead drawn by **sequential
random addition** (`initial.hard_core_positions`): candidates are drawn uniformly one at a time and
rejected when they lie closer than `r_min = x_min r_nn` to an already accepted particle (minimum
image when periodic), so all pair distances are `>= r_min`. The construction is deterministic for a
given seed, its cost is `O(N^2)` (0.2 s at `N = 5000`, charged by the cost model), and the bound
`x_min <= 0.7` (`config.MAX_MIN_SEPARATION_NN`) keeps the excluded volume `(4 pi / 3) rho r_min^3`
below 24 % of the box (packing fraction `(pi / 6) rho r_min^3 <= 3 %`), so fewer than 1.2 attempts
per particle are needed; such RSA configurations approximate a dilute hard-sphere fluid. `r_nn`
keeps its Poisson value as the length unit (the actual mean nearest-neighbour distance is `1.05 r_nn`
at `x_min = 0.5`), and the Poisson results of section 6 do not apply to these positions. For
`x_min = 0` the random stream is bit-identical to the code before the hard core existed
(`tests/test_initial.py`), so all fixed-seed results are unchanged.

Two boundary conditions are available:

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
| `P` | `(u_i . rhat)(u_j . rhat)`: the **longitudinal channel** | 0 | -1 |
| `Cperp` | `C - P = u_perp,i . u_perp,j`: the **transverse channel** (derived from the stored sums of `C` and `P`) | 0 | 0 |
| `B` | `C - 3 P = Cperp - 2 P`; the GR pair energy is `U_ij = -B G s_i s_j / r^3` | 0 | 2 |
| `Q` | `cos^2 phi`, `phi` = angle between `S_i + S_j` and `rhat` | 1/3 | 1 (unequal lengths) |
| `A` | `[(u_i . rhat)^2 + (u_j . rhat)^2] / 2` | 1/3 | 1 |

**Two channels.** Writing every spin as `S = S_r rhat + S_perp` (component along the bond plus the
transverse part), the GR pair energy is

```
U_ij = -(G / r^3) [ S_i . S_j - 3 S_r,i S_r,j ] = -(G / r^3) [ S_perp,i . S_perp,j - 2 S_r,i S_r,j ].
```

The longitudinal components couple *antiferromagnetically* with weight 2, the transverse components
*ferromagnetically* with weight 1. `P` measures the longitudinal channel, `Cperp` the transverse one and
`C = P + Cperp`. In the relaxed Poisson ensemble `P < 0` and `Cperp > 0` at every distance: the sign change
of `C` occurs where `|P| = Cperp`, and the positive shell of `C` is carried by the transverse channel. An
isotropic correlation has `P = C/3` (the isotropic-AFM control obeys this exactly); the dipolar control is
the mirror image (`P > 0`, `Cperp < 0`).

* `C` and `Q` are the observables of the manuscript, `P` and `Cperp` the channel decomposition behind
  them (`ensemble_table.csv` columns `mean_P_*`, `sem_P_*`, `mean_Cperp_*`, `sem_Cperp_*`; `grspin compare
  --obs P Cperp`). `Q` is a property of the *pair total spin* and depends on the spin lengths through
  `|S_i + S_j|`: for equal lengths an antiparallel pair has `S_i + S_j = 0` and drops out of the average,
  so `Q -> 0` (not 1) at short distance (set `S4`). `A` measures the locking of a single spin to the bond
  axis and does not have this problem. The landmarks of `C(r)` also depend on the *shape* of the
  length distribution (set `S4`: `x_0` from 0.75 r_nn for a log-normal to 0.89 r_nn for equal lengths),
  whereas the sign structure `P < 0`, `Cperp > 0` does not.
* `B` is the *pair-energy correlation*, a quantitative measure of frustration (`B` falls from the dimer
  value 2 to 0; it is negative for the dipolar control).
* Pairs containing a zero-length spin are excluded from `C, A, P, Cperp, B`; pairs with `S_i + S_j = 0`
  are excluded from `Q`.
* Global diagnostics: `|sum_i S_i|` (reported as `|sum S|/sqrt(N)`) and `S(0) = |sum_i u_i|^2 / N` of the
  unit spins (`~1` for uncorrelated directions, `~N` for a ferromagnet).

The runner stores **sums and counts per bin** for the initial and the final state (and for requested
intermediate sweeps), not means. This allows exact re-binning by any integer divisor of `n_bins`
(`rebin=` in the analysis) and pooled estimates. It also stores the sum of `x` over the pairs of every
bin, which gives the **pair-weighted mean distance** `x_mean = sum_x / cnt_C` of a bin; for Poisson
positions it exceeds the bin centre `c` by about `h^2 / (6 c)` (bin width `h`), i.e. by 0.0016 r_nn at
`x = 1` for `h = 0.1`.

## 5. Ensemble statistics

* **Mean over realizations.** As in the manuscript, the binned mean of an observable is first formed over
  all pairs in the bin *within each realization*, then averaged over realizations (`analysis.mean_sem`);
  realizations without pairs in a bin are skipped.
* **Error bars** are standard errors over realizations, `std(ddof=1) / sqrt(n)`. They do not include the
  spread of pair values inside a bin.
* **Bootstrap.** Landmarks, fit parameters and the nearest-neighbour tables get uncertainties by resampling
  *realizations* with replacement (`n_boot` times, default 200) and repeating the whole computation; all
  observables are resampled with the same realization indices. Nearest-neighbour errors are reported as
  NaN where a category holds fewer than two pairs (or is fed by fewer than two realizations).
* **Abscissa.** Curves are tabulated at the bin centres `x_center` and at the pair-weighted mean distance
  `x_mean` (section 4). Landmarks are evaluated **on `x_mean` at the stored bin width** (`landmark_rebin =
  1`, i.e. 0.05 r_nn in the plan): bin centres bias positions by `h^2 / (6x)` and coarse bins lower a curved
  peak by `|C''| h^2 / 24` (0.001 for the peak of `C` at `h = 0.1`). `summary.md` states the abscissa and bin
  width that the quoted landmarks refer to; the tables and figures keep the plan's rebin factor 2.
* **Landmarks** (model-independent features of the mean curves): the first upward zero `x_0` of `C`
  (linear interpolation), the position and height of the positive peak (parabola through the highest bin
  after `x_0` and its neighbours; the parabola allows for the unequal spacing of `x_mean`), the maximum of
  the transverse channel `Cperp` and its position (parabola) together with `P` interpolated at that
  position, the minimum of `Q`, the `x` at which `Q` falls below `1/3`, and the `x` at which `A` falls below
  `1/2`. The bootstrap error is NaN when the landmark itself is undefined (e.g. no zero crossing of `C`
  for the dipolar control).
* **Error budget** (`analysis.landmark_error_budget`, written as `error_budget.csv` and a section of
  `summary.md`). For every landmark: `stat` (bootstrap), `binning` = |value at rebin 1 - value at rebin 2|,
  `sweeps` = paired shift between the main run and a run that relaxes *the same realizations* longer (set
  `S1`: 500 instead of 100 sweeps on 48 common seeds), `cutoff` = paired shift against a larger cutoff
  (`S2_rc15`: 15 instead of 10 r_nn, 48 common seeds), `total = sqrt(stat^2 + binning^2 + sweeps^2 +
  cutoff^2)` (missing components count as zero) and the value rounded to the precision of the total. The
  shifts are signed (run minus main) and carry their own paired bootstrap error. `grspin analyze` finds the
  paired sibling runs automatically (`--sweeps-run`, `--cutoff-run` to override); `grspin plan analyze`
  passes `S1` and `S2_rc15` for entry `M`. For `M` this gives `x_0 = 0.845 ± 0.005` (stat 0.002, cutoff
  shift -0.004), `C_peak = 0.186 ± 0.005`, `Cperp_max = 0.436 ± 0.003` at `x = 0.97 ± 0.02`,
  `Q_min = 0.198 ± 0.002`: the statistical errors alone understate the uncertainty by a factor 2-3.
* **Paired and unpaired differences.** Two runs are *paired* when they agree in the fields that generate
  the initial state (`N`, density or box, spin-length distribution, scale and `lognormal_sigma`) and share
  seeds; coupling, cutoff, boundary condition, number of sweeps and protocol may differ.
  `analysis.paired_difference` then forms `obs_other - obs_reference` per seed before averaging, which
  removes most of the sampling noise. For runs that are not paired (e.g. different `N` in set `S3`) it
  returns the *unpaired* difference of the two ensemble means over all realizations of both runs with
  the error `sqrt(sem_a^2 + sem_b^2)`. `grspin compare` names the files `paired_*` or `unpaired_*`
  accordingly, `comparison_summary.md` records which was used, and `comparison_landmark_differences.csv`
  holds the landmark shifts with a joint (paired where possible) bootstrap.
* **Energies.** The ensemble mean of the energy is dominated by the closest pair of each realization
  (its energy `2 G s_i s_j / r^3` has a heavy tail for Poisson positions: `M` gives `E/N = -13.2 ± 0.8`
  mean, `-9.7` median, one realization at `-117`). `histories.csv` and `summary.md` therefore report the
  median and interquartile range of the energy per spin and of the relative excess energy next to the
  means; the largest direction change of a unit spin per sweep is the convergence metric, and only paired
  energy differences on common seeds (set `S5`) are precise.

## 6. Nearest-neighbour (dimer) picture

For a Poisson process the probability that a ball of radius `r` around a particle is empty is
`exp(-4 pi rho r^3 / 3)`. The probability that a pair at separation `r` is a **mutual** nearest-neighbour
pair is `exp(-9 pi rho r^3 / 4)`: neither member may have another particle within `r`, and the union of the two
balls (centres `r` apart) has volume `2 (4 pi/3) r^3 - 5 pi r^3 / 12 = 9 pi r^3 / 4`. The fraction of particles that belong
to a mutual-NN pair is `16/27`. The dimer fraction `-f_m(x) = -exp(-9 pi rho r^3 / 4)` is the exact
`r -> 0` limit of `C(r)` and can be drawn next to it (`draw_C(show_nn=True)`, label "dimer fraction");
both probabilities are tabulated (`C_NN_mutual`, `C_NN_isolated`). All of these results - `f_m(r)`,
`16/27`, the nearest-neighbour distance law and `theory.mutual_nn_probability` - describe the
**ideal Poisson process** and apply only to `min_separation_nn = 0`; for hard-core positions
(section 3, set `S8`) the diagnostics below are still recorded, but the Poisson curves drawn next to
them are not references (a hard core of `0.5 r_nn` removes exactly the short pairs that make `f_m`
approach 1). The diagnostics (`nn_diagnostics=True`) measure, per radial bin and pooled over all
pairs,

* the fraction of mutual-NN pairs (`f_mutual`), versus the theory above,
* `C` separately for mutual-NN pairs, pairs that are the NN of exactly one member (`oneway`) and pairs that are
  nobody's NN (`far`),
* the rank-resolved correlation `C_k` between a spin and its `k`-th nearest neighbour (`k = 1..8`),
* the histogram of first-neighbour distances against the Poisson law, and
* the small-distance law of `1 + C(r)` in log-spaced bins.

## 7. Damped-oscillatory form (not used)

`C_fit(x) = -exp(-alpha u^3) cos(k u + phi)` with `u = rho^(1/3) r = 0.55396 x` is kept in `grspin.fit` as a
**phenomenological guide only**. It reproduces the truncated-interaction curves (set `S2` with
`r_cut <= 4 r_nn`, which do show a negative dip at `x ~ 2.3`) but not the full interaction: the relaxed
`C(r)` of `M` has no oscillation (`chi^2/dof ~ 700`, systematic residuals up to 0.09), and the three
parameters are only weakly determined. `grspin analyze` still writes `fit.json` (multi-start fit on
`x_mean`, bootstrap percentiles) and labels it as a guide in `summary.md`; the curve is not drawn unless
`--show-fit` is given, and no quantitative statement uses it. Quote the landmarks and the error budget of
section 5 instead.

## 8. Controls, variants, the far tail and the protocol check

### 8.1 Control couplings

`coupling="isotropic_afm"` and `"dipolar"` use exactly the same code and seeds as `M` (set `S6`); they
show which features need the GR tensor. The isotropic `1/r^3` antiferromagnet has `P = C/3` and
`Q = A = 1/3` at every distance (no channel structure): `C` rises monotonically from -1, crosses zero at
1.84 r_nn and has only a weak hump (0.049 at 2.25 r_nn, the parity of chains). The dipolar coupling is
the mirror image of the GR case: `P > 0`, `Cperp < 0` (minimum -0.16 at 1.05 r_nn), `C` decreases
monotonically from +1 without a sign change in `[0, 4]`, and `Q` approaches 1/3 from above. The sign
change of `C`, the positive shell and `Q < 1/3` are therefore specific to the GR tensor.

**Channel controls (set `S7`).** The two controls above change the sign *and* the channel weights
at once, so they cannot attribute the positive shell to one channel (the isotropic antiferromagnet
even shows that a weak shell can arise without any transverse channel). The channel family of
section 1 keeps `sigma = +1` and varies `gamma`: `S7_gamma1` (`transverse_only`), `S7_gamma12` and
`S7_gamma100` (`longitudinal_only`) run with the geometry, cutoff, sweeps and seeds of `M`, whose
`gamma = 3` completes the family, with the nearest-neighbour diagnostics (dimer map) and 16 stored
states each. The comparison `cmp_channels` (reference `M`, paired, observables `C, P, Cperp, Q,
A`) and the derived item `D3` tabulate the landmarks `x_0`, `C_peak`, `x_peak`, `Cperp_max`,
`Q_min` and the sign of `C` at `1.2 r_nn` against `gamma` and draw `C`, `P`, `Cperp` of the four
couplings on one axis each: `compare_runs` recognises the family (all members with `sigma = +1`
and at least two values of `gamma`, `analysis.gamma_family_members`) and writes
`gamma_family.csv` (`report.gamma_family_table`: the landmarks with bootstrap errors, `C(1.2 r_nn)
± SEM` interpolated on `x_mean` and its sign, one row per run in order of `gamma`),
`gamma_family.pdf` (`plotting.figure_gamma_family`) and the section "Channel family: landmarks
against gamma" of `comparison_summary.md`. The attribution is a prediction to be tested, not assumed: if the
positive shell and `Q < 1/3` survive `gamma = 1` and disappear (or move far out) for `gamma = 100`,
the shell is carried by the transverse channel. The two-body minima of section 1 are the reference
energies of the frustration index for each member of the family.

### 8.2 Spin-length distributions

`uniform` `U(0, S)` (manuscript), `uniform_half` `U(S/2, 3S/2)`, `equal` and `lognormal` (`S exp(sigma Z)`,
`sigma = 1`), set `S4` on the seeds of `M`. The sign pattern `P < 0`, `Cperp > 0` and the transverse
maximum (`Cperp_max` = 0.44-0.45) are robust, but the landmarks of `C(r)` shift with the *shape* of the
distribution: `x_0` = 0.747 (lognormal), 0.844 (`U(0, S)`), 0.880 (`U(S/2, 3S/2)`), 0.890 (equal);
`C_peak` = 0.184-0.187 and 0.222 (lognormal); the paired differences of `C` reach 0.18 (40 standard
errors) at `x = 0.55`. Heavy-tailed distributions let rare strong spins dominate the local fields of
their neighbours and move the zero crossing inward. `Q` depends on the lengths through `|S_i + S_j|`
and is undefined for equal lengths (antiparallel pairs have `S_i + S_j = 0`); `A` is the
length-independent alternative. Quoted landmarks must therefore name the distribution.

### 8.3 Far tail and cutoff (set `T`)

The main sets store `C(r)` on `[0, 4] r_nn`. The entries `T_N10000_rc7`, `T_N10000_rc10`,
`T_N10000_rc15` (`N = 10000`, `r_cut = 7, 10, 15 r_nn`) and `T_N20000_rc10` extend the window to
`x_max = 10 r_nn` (200 bins of 0.05 r_nn, 32 realizations, two stored states each; `x_max r_nn = 5.5
< L/2 = 10.8` at `N = 10000`). The comparisons `cmp_far_tail` (paired, reference `r_cut = 15`) and
`cmp_far_tail_size` (`N = 20000` versus 10000, unpaired) analyse `C`, `P` and `Cperp`.

Exploratory runs with these parameters (other seeds) show a shallow negative dip of `C` in the far
tail whose position scales with the cutoff: minimum -0.018 at `x = 5.5` for `r_cut = 7`, -0.008 at
8.1 for `r_cut = 10`, -0.002 at `x >= 9.9` for `r_cut = 15`; the paired difference `C(r_cut = 15) -
C(10)` reaches +0.007 at `x = 8` (SEM 0.0002), while for `x <= 3.5` the two cutoffs agree to 0.002,
and `N = 10000` and `20000` give identical tails at `r_cut = 10`. The dip is a **truncation effect**
of the sharp cutoff, not a property of the interaction: the tensor averages to zero over a sphere, so
the missing tail has no mean-field part, but the spins inside `r_cut` arrange themselves against an
interaction that ends abruptly at `r_cut`, and the resulting anti-correlation sits at a fixed fraction
of `r_cut`. For `r_cut <= 4 r_nn` it enters the quoted window (`C = -0.1` at `x ~ 2.3` for
`r_cut = 3`), which is the origin of the "damped-oscillatory" shape of section 7. `C(r)` is quoted for
`x <= 4` at `r_cut = 10 r_nn`, where it is converged to about 0.002 (section 5, error budget); within
the accessible window the untruncated tail is positive and decays monotonically.

### 8.4 Protocol check and annealing (set `S5`)

`grspin protocol-check` compares, on identical positions and lengths, the production protocol
(`sequential`) with `random_order` (a fresh permutation in every sweep), `other_init` (an independent
isotropic initial direction field) and `anneal` (Metropolis annealing followed by the same quench).
Per realization and paired it stores the final energy, the overlaps `q1 = <u . u'>` and
`q2 = <(u . u')^2>` with the production state over the spins of nonzero length (random axes give
`q2 = 1/3`, identical states up to a global flip give 1), the binned pair sums of all observables and,
for the annealed variant, the Metropolis acceptance rate. `S5_protocol_short` / `_long` (`N = 1000`,
`r_cut = 8 r_nn`, 32 and 16 realizations, 10 and 100 sweeps per temperature) are the small checks;
`S5_N5000` repeats the short check at production size on the first 48 seeds of `M`, so that the
landmark differences are paired at the precision of the main result (the small checks bound them only
at about +-0.01 in `x_0` and `C_peak`).

**Dimer / interstitial split and the non-dimer energy.** Because the closest pairs are nearly
isolated dimers whose antiparallel state every protocol finds, the overlaps and the energy
differences are also reported for the spins that belong to a mutual nearest-neighbour pair
(`nn.mutual_nn_partners`, "dimer", 59 % of the spins for Poisson positions) and for the rest
("interstitial") separately: record keys `<variant>_q1_dimer`, `_q2_dimer`, `_q1_interstitial`,
`_q2_interstitial` (`protocols.SPIN_CLASSES`), `n_dimer_spins`, `n_nonzero_spins`. The dimer-bond
energy of each variant, `E_dimer = sum over mutual pairs of U_ij` with the run's `(sigma, gamma)`
and `G` (`<variant>_energy_dimer`), and `E_nondimer = E - E_dimer` (`<variant>_energy_nondimer`)
let the energy change of a variant be measured relative to the bonds that are actually frustrated:
`protocol_summary.csv` carries `overlap_q{1,2}_{dimer,interstitial}` (with `sem_`),
`delta_energy_nondimer_percent` (the same `Delta E` divided by `|E_nondimer|` of the production
state), `delta_energy_dimer_percent`, `energy_dimer_per_spin`, `dimer_energy_share`
(`E_dimer / E`) and `dimer_spin_fraction`; `protocol_summary.md` is the Markdown digest. Files of
version 0.2 without the split still load and give NaN in these columns
(`ProtocolData.has_dimer_split`). Since `|E_nondimer| < |E|`, `|Delta E / E_nondimer| >=
|Delta E / E|` with the same sign for every realization (tested).

Results of the small checks with the version-0.1 schedule (`t_start = 2`): relative energy changes
-0.03 +- 0.03 % (random order), -0.05 +- 0.04 % (other initial directions), -0.13 +- 0.04 % and
-0.35 +- 0.05 % (annealing with 10 and 100 sweeps per temperature; lower in 72 % and 100 % of the
seeds); overlaps `q1` = 0.43, -0.04, 0.23 and `q2` = 0.68, 0.62, 0.64; `max |Delta C| <= 0.026`
(`<= 3.3` SEM) per bin and landmark shifts within one standard error. Many near-degenerate minima
(`q1 ~ 0`) share the same axis field (`q2 ~ 0.65`) and the same correlation functions: `C(r)` is a
property of the ensemble of low-energy states, not of one state. Annealing finds lower energies and
the gain grows with the annealing effort, so the quench states are metastable and no protocol has
reached the ground state; annealed energies must be read as "lower-energy states exist", not as
ground-state estimates.

**Annealing schedule** (`AnnealConfig`): `n_temperatures = 40` temperatures, geometric from
`t_start = 10` to `t_end = 0.01` (units `G S^2 rho`), `sweeps_per_temperature` Metropolis sweeps of
`N` single-spin proposals each (Gaussian step 0.3 on the unit direction, acceptance
`min(1, exp(-dE/T))`). `t_start` lies above the typical dimer binding energy
`2 G <s_i s_j> / r_nn^3 = 2.9 G S^2 rho` (`config.DIMER_BINDING_ENERGY`; version 0.1 started at 2.0,
below it), so the typical dimers are molten at the start of the schedule, whereas the closest pairs
(binding energies of order 100) stay frozen at any practical temperature. The acceptance rate is
recorded **at every temperature** (`anneal_acceptance_per_T`, hot to cold) and at the final temperature
(`anneal_acceptance_final`; version 0.1 stored only the latter as `anneal_acceptance`, which is still
read). The final value is not an overall rate: it is the fraction of accepted 0.3-rad proposals at
`T = 0.01`, i.e. the fraction of soft spins whose `|h_i| s_i` is of order `T_end` (about 11 % in the
version-0.1 runs). `ProtocolData.acceptance_table()` tabulates the mean acceptance per temperature
(the hot stage is equilibrated when its acceptance is close to 1), and `protocol_summary.csv` carries
the first and the final value in the `anneal` row (`anneal_acceptance_first`, `anneal_acceptance_final`).

### 8.5 Dimer-frame correlation map

`observables.dimer_frame_map` (stored with `nn_diagnostics=True`; keys `dimer_map_*`, `dimer_ang_*`)
measures how third spins arrange themselves around a dimer. For every spin `i` of a mutual
nearest-neighbour pair (`nn.mutual_nn_partners`; both members are mapped, each in its own frame) the
axis `a = rhat_ij` points from `i` to its partner `j`. Every other spin `k != j` within `2.5 r_nn` of
`i` (minimum image for periodic boundaries) contributes `u_i . u_k` (`C`) and
`(u_i . rhat_ik)(u_k . rhat_ik)` (`P`) with a count

* to a cylindrical map in `z = a . r_ik` in `[-2.5, 2.5]` and `rho = sqrt(r_ik^2 - z^2)` in `[0, 2.5]`
  (50 x 25 cells of 0.1 r_nn; `i` sits at the origin, `j` on the positive axis), and
* to an angular table in `x_ik = r_ik / r_nn` (edges 0.5, 0.75, 1, 1.25, 1.5, 2, 2.5) times the signed
  `cos(theta) = a . rhat_ik` (edges -1, -0.8, -0.5, -0.25, 0.25, 0.5, 0.8, 1; upper edges inclusive).

Pairs with a zero-length spin are skipped; the record also holds the number of dimer spins and the sum
of their partner distances. The analysis (`EnsembleData.dimer_map()`, `dimer_map_table()`,
`dimer_angular_table()`) averages the per-realization cell means over realizations with standard
errors and masks cells with fewer than 20 pooled pairs; `grspin analyze` writes `dimer_map.csv` (long
format), `dimer_angular.csv` and `dimer_map.pdf` (`C` and `P` panels, the dimer drawn with `j` at the
mean partner distance; the Poisson mean separation of a mutual pair is `(16/27)^(1/3) r_nn = 0.84 r_nn`,
`theory.MUTUAL_PAIR_MEAN_DISTANCE_X`). `analysis.dimer_map_from_states(run_dir)` builds the map from
the stored states of runs recorded before it existed. Measured on `M`: a parallel lobe transverse to and
behind `i` (`<u_i . u_k>` = +0.23 to +0.25 for `|cos theta| < 0.25`, `x_ik < 1.2`), an antiparallel lobe
around the partner (`-0.51` to `-0.30` at 30-60 degrees from the axis on the partner's side, where `k`
is transverse to `j`) and a parallel region beyond `j` on the axis: every spin aligns parallel to the
dimer members to which it is transverse and antiparallel to those along whose axis it lies. This is
the transverse ferromagnetic channel at work, and it is what the isotropic antiferromagnet lacks.
The angular table of `dimer_angular.csv` and of `summary.md` carries the pooled number of pairs
of every cell next to `C ± SEM`.

**Dimer-field test** (`observables.dimer_field_prediction`, computed inside `dimer_frame_map` when
it is given `sigma_gamma = config.sigma_gamma`; `ensemble.run_realization` stores it with the nn
diagnostics). The map above is correlational; the causal question is whether the two dimer spins
alone explain the direction of a third spin. For every pair `(i, k)` of the map the raw field of
the two members at `k` is formed with the actual spin vectors `S = s u` (so the longer member
dominates),

```
h = T_ki S_i + T_kj S_j,      T_ab = w_ab [1 - gamma rhat_ab rhat_ab^T],   w = 1 / r^3,
```

with `r_k - r_j` through the nearest image, and the zero-temperature rule predicts
`u_k = sigma h / |h|`. Recorded per bin of `x_ik` (the six bins of the angular table):
`dimer_field_cnt` (pairs with `h != 0`), `dimer_field_agree` (pairs with `u_k . sigma h > 0`),
`dimer_field_sum_cos` (sum of `u_k . sigma h / |h|`); per angular cell `dimer_ang_sum_Cpred` (sum
of `(sigma h / |h|) . u_i`, the **dimer-only prediction** of the angular table of `C`); and
`dimer_field_n`, `dimer_field_n_agree` (all pairs within `2.5 r_nn`), `dimer_field_sigma`,
`dimer_field_gamma`. Random directions give an agreement fraction of `0.5` and a mean cosine of
`0` (tested; `N = 3000`: `0.50 ± 0.02`, `0.00`). The analysis (`tabulate_dimer_field`,
`EnsembleData.dimer_field_table()`) reports the agreement fraction and the mean cosine per bin as
the mean over realizations ± SEM together with the pooled fraction and an "all within `2.5 r_nn`"
row (`dimer_field.csv`), and adds the column `C_pred` (with `sem_C_pred`) to `dimer_angular.csv`;
`summary.md` has the section "Dimer-field test" with both tables, `summary.json` the block
`dimer_field`. **Caveat:** `C_pred` is a function of `S_i` and `S_j` only, so its baseline for
*uncorrelated* third spins is not 0 (`+0.09` for `gamma = 3`); compare `C_pred` with the measured
`C` cell by cell, as the referee did, and read the agreement fraction and the cosine (which involve
the independent `u_k`) as the unbiased statistics. For runs whose realization files predate the
map or the field test (`results/M` as recorded with 0.2.0) `analyze_run` computes both from the
stored states and says so (`source: states` in `summary.md` and `summary.json`); the rerun of `M`
with the current code records them per realization.

Measured on two stored states of `M` (seeds 1000 and 1001, 59789 pairs within `2.5 r_nn`;
to be replaced by the 48-state values of the rerun): agreement `0.972` (`x_ik` in `[0.5, 0.75)`,
278 pairs), `0.961` (`[0.75, 1)`, 1137), `0.910` (`[1, 1.25)`, 2860), `0.853` (`[1.25, 1.5)`,
5347), `0.752` (`[1.5, 2)`, 18606), `0.668` (`[2, 2.5)`, 31530), `0.730` overall; mean cosine
`0.89, 0.82, 0.69, 0.57, 0.38, 0.24`. The dimer-only prediction of the angular table follows the
measured `C` cell by cell (collinear far side of `i` at `[0.5, 0.75)`: `C_pred = -0.48` versus
`C = -0.41 ± 0.02`; beside the partner at `[0.5, 0.75)`: `-0.76` versus `-0.66 ± 0.09`; transverse
at `[0.75, 1)`: `+0.18` versus `+0.25 ± 0.03`), i.e. the two dimer spins alone predict the sign of
the third spin's correlation in every cell and its direction in 90 % of the cases out to
`1.25 r_nn`, falling towards the random value `0.5` at `2.5 r_nn` where the rest of the
neighbourhood takes over.

### 8.6 Spin structure factor

`observables.spin_structure_factor` and `analysis.structure_factor_from_states` evaluate
`S(k) = |sum_i u_i exp(i k . r_i)|^2 / N` of the unit spins at the reciprocal vectors `k = 2 pi n / L`
of the periodic box (`0 < |n| <= 12`, one of each `+-k` pair), averaged over 24 equal shells of `|k|`
and over the stored states (`save_states`; standard errors over the states) for the initial and the
final spins; the first row is `k = 0`, i.e. `s0_unit`. Uncorrelated directions give `S = 1`; `M`
gives `S(0) = 0.38 +- 0.02` after relaxation (`|sum S| / sqrt(N)` drops from 0.55 to 0.30):
long-wavelength fluctuations are suppressed without ferro- or antiferromagnetic order. A single state
gives a noisy `S(k)`, which is why `M` stores 48 states; the far-tail entries (set `T`) and the
controls `S6`, `S7`, `S8` store 16 each, so that `S(k)` at the smallest wavevectors can be compared
across `r_cut = 7, 10, 15 r_nn` and `N = 10000, 20000` and the energy budget can be evaluated for
every coupling. Output: `structure_factor.csv`, `.pdf` and a `summary.json` block.

**Across runs** (`report.structure_factor_comparison(run_dirs, out_dir, labels=...)`): one table
`structure_factor_vs_cutoff.csv` with a row per run (`label`, `N`, `r_cut / r_nn`, `L / r_nn`,
`k_min r_nn = 2 pi r_nn / L`, number of states, `S(0) ± SEM`, `S(0)` of the initial spins, the first
three shells `k{m}_rnn, S{m}, sem_S{m}` and the maximum of `S(k > 0)` with its `k`), the figure
`structure_factor_comparison.pdf` (all runs on one logarithmic `k r_nn` axis with the line
`S = 1`) and the digest `structure_factor_comparison.md`; runs without states are skipped with a
warning. `grspin plan analyze` writes it as `cmp_structure_factor` under
`results/_comparisons/cmp_structure_factor/` for `T_N10000_rc7`, `T_N10000_rc10`, `T_N10000_rc15`,
`T_N20000_rc10` and `M` (`plan.STRUCTURE_FACTOR_ENTRIES`; whenever at least two of them have
stored states, and whenever the targets include `all`, `comparisons`, `T`, one of these ids or
`cmp_structure_factor`; `--max-states` limits the states used). The smallest wavevector depends on
`L`, so the shells of different box sizes sit at different `k r_nn` (`0.206` at `N = 10000`,
`0.164` at `N = 20000` for the first shell). This is the comparison that decides whether the
intermediate-scale feature of `S(k)` seen in the first `M` states (`S ~ 2-4` at `k r_nn ~
0.5-0.75`, wavelength `8-13 r_nn`, close to `r_cut = 10 r_nn` and `L / 3`) is a cutoff artefact, a
box effect or real; it is not interpreted until the 16-state `T` entries exist.

### 8.7 Hard-core positions (set `S8`)

`S8_hardcore05` (optional) repeats the main system with `min_separation_nn = 0.5` (section 3): all
pair distances are at least `0.5 r_nn`, which removes the closest pairs that dominate the mean
energy (section 5) and the trivial dimerization of the Poisson process (section 6; `f_m(r)` and
`16/27` do not hold there). The comparison `cmp_hardcore` with `M` (observables `C, P, Cperp, Q, A`)
is **unpaired**: the positions differ although the seeds coincide, so `min_separation_nn` must be
one of the `analysis.PAIRING_KEYS` that make `same_system` false. It tests which features are
universal - the channel signs `P < 0`, `Cperp > 0` - and which are Poisson numbers - the dimer
fraction, `x_0`, the shell height. The nearest-neighbour diagnostics and 16 stored states are
recorded as for `M`. `summary.md` and `comparison_summary.md` state the position process
(`positions = hard core 0.5 r_nn`), and the Poisson reference curves drawn or tabulated next to
the nn diagnostics are not references for this run (section 6).

### 8.8 Energy budget and frustration index

`observables.energy_budget` and `analysis.energy_budget_from_states(run_dir)` evaluate, for every
stored relaxed state (and the stored initial state), the energy of all bonds within `r_cut` against
the sum of their isolated-pair minima. With `w = 1 / r^3` and the run's `(sigma, gamma)` and `G`,

```
U_ij   = -sigma G w [ S_i . S_j - gamma (S_i . rhat)(S_j . rhat) ],
U_min  = -f G w s_i s_j,     f = max(1, |gamma - 1|)     (section 1; theory.two_body_minimum_factor)
```

for either sign of `sigma`: along the bond with `u_i . u_j = sign(sigma (1 - gamma))` when
`|gamma - 1| > 1`, transverse with `u_i . u_j = sign(sigma)` when `|gamma - 1| < 1`; `f = 2` for
`gr` and `dipolar`, `1` for `isotropic_afm` and `transverse_only`, `11` for `gamma12`, `99` for
`longitudinal_only`. (The brief's first guess "`-(gamma - 1)` if `gamma >= 2` else `-1` for
`sigma = +1`" is the same number; the general statement is the spectral norm of
`1 - gamma rhat rhat^T`.) Per state, `energy_budget.csv` holds `energy_total = sum U_ij` (it
reproduces the final energy of the relaxation to round-off, `1e-15` relative), `energy_dimer`
(the sum over the mutual-nearest-neighbour pairs, `observables.mutual_pairs`), `energy_nondimer =
energy_total - energy_dimer`, `minimum_total` and `minimum_dimer` (the sums of `U_min` over the
same sets), `energy_total_initial`, `n_pairs`, `n_dimers`, `n_dimer_spins`, and the ratios
(`analysis.ENERGY_BUDGET_RATIOS`, all independent of `G`)

* **dimer share** `energy_dimer / energy_total` and `nondimer_share = 1 - dimer_share`: how much
  of the energy sits in the mutual-NN bonds (the closest pairs, section 5);
* **realized fraction** `energy_dimer / minimum_dimer` (1: every dimer at its isolated-pair
  minimum) and the same for the non-dimer bonds;
* **frustration index** `energy_total / minimum_total`: the energy in units of the sum of the
  two-body minima of all bonds within `r_cut`; `1` if every bond sat at its own minimum (an
  unfrustrated system), about `0` for uncorrelated spins (`frustration_index_initial` gives the
  value of the stored initial state). It depends on the cutoff, because `sum_{r < r_cut} w` grows
  like `log r_cut` while the realized energy converges, so it is compared between runs with the
  same `r_cut` only.

`analysis.summarize_energy_budget` gives mean ± SEM, minimum and maximum over the states, also
for the energies per spin; `analyze_run` writes `energy_budget.csv`, `energy_budget.json` and the
section "Energy budget of the stored states" of `summary.md` for **every run with stored states**
(`M`, `S6`, `S7`, `S8`, `T`), so the same index with the coupling's own `f` is quoted for GR, the
controls and the channel family (`grspin analyze --no-energy-budget` skips it, `--max-states`
limits the states). Two isolated dimers give `E_nondimer = 0` and a realized fraction of exactly 1
(tested). On two stored states of `M` (`gr`, `f = 2`, `r_cut = 10 r_nn`): dimer share `0.891` and
`0.849` (the 59 % of spins in mutual pairs hold 87 % of the energy), realized fraction of the dimers
`0.982` and `0.975` (the dimers are at their isolated-pair minimum), realized fraction of the
non-dimer bonds `0.197` and `0.205`, frustration index `0.684` and `0.623` (initial states `-0.013`
and `-0.127`), `E / N = -12.3 ± 1.9 G S^2 rho`: the frustration lives in the non-dimer bonds, which
realize only a fifth of their two-body energy. The cost is one pair enumeration per state
(`1.8 M` pairs and `0.45 s` for `M`; `12 M` pairs, `~1.5 GB` and `3 s` for `T_N10000_rc15`).

## 9. Output files

```
<run>/config.json              kind, configuration, provenance (versions, platform, git commit, dirty flag,
                               git status/diff of the sources if dirty, SHA-256 of the sources), run history
<run>/seeds/seed_XXXXXXXX.npz  one file per realization (ensembles)
<run>/protocol/seed_XXXXXXXX.npz  one file per realization (protocol checks)
<run>/states/seed_XXXXXXXX.npz positions, magnitudes, spins_initial, spins_final (first `save_states` seeds)
<run>/analysis/...             written by `grspin analyze` / `grspin plan analyze`
results/_comparisons/<id>/     written by `grspin compare` / `grspin plan analyze`
```

Keys of a realization file (`n` = number of bins, `T` = number of sweeps, `K` = number of snapshots,
`M` = number of annealing temperatures):

| key | shape | content |
|---|---|---|
| `init_*`, `fin_*` | | `cnt_C, cnt_Q` and `sum_C, sum_Q, sum_A, sum_P, sum_B, sum_x` (`(n,)`); `total_spin`, `s0_unit` (scalars) |
| `snap_*`, `snap_sweeps` | `(K, n)`, `(K,)` | the same sums at the requested intermediate sweeps |
| `energy`, `dmax`, `total_spin_hist` | `(T+1,)` | energy (with `G`), largest direction change, `|sum S|` after each sweep |
| `energy_drift` | scalar | relative difference between incremental and direct final energy |
| `anneal_acceptance_per_T`, `anneal_acceptance_final` | `(M,)`, scalar | Metropolis acceptance at every temperature and at the last one (`protocol="anneal"`; version 0.1: `anneal_acceptance` = final) |
| `nn_*`, `sr_*` | | nearest-neighbour diagnostics (section 6), if requested |
| `dimer_map_cnt`, `dimer_map_sum_C`, `dimer_map_sum_P` | `(50, 25)` | dimer-frame map (section 8.5), with the nn diagnostics |
| `dimer_ang_cnt`, `dimer_ang_sum_C`, `dimer_ang_sum_P` | `(6, 7)` | angular table of the dimer frame |
| `dimer_map_r_max`, `dimer_map_n_spins`, `dimer_map_sum_x_partner` | scalars | radius of the map (`r_nn`), number of dimer spins, sum of their partner distances |
| `dimer_field_cnt`, `dimer_field_agree`, `dimer_field_sum_cos` | `(6,)` | dimer-field test per `x_ik` bin (section 8.5): pairs with a defined prediction, agreements, sum of `u_k . sigma h/\|h\|` |
| `dimer_ang_sum_Cpred` | `(6, 7)` | dimer-only prediction of the angular table (sum of `(sigma h/\|h\|) . u_i`) |
| `dimer_field_n`, `dimer_field_n_agree`, `dimer_field_sigma`, `dimer_field_gamma` | scalars | all pairs within the map radius and their agreements; the `(sigma, gamma)` used |
| `git_commit`, `source_hash` | text | the code that wrote the file (section 10) |
| `seed`, `timing_s`, `elapsed_s` | | seed and timing (build / relax / observables) |

A protocol-check file holds `<variant>_<key>` for the pair sums, `<variant>_energy`, `_dmax`, `_q1`,
`_q2` of the four variants, the dimer / interstitial split of section 8.4 (`<variant>_q1_dimer`,
`_q2_dimer`, `_q1_interstitial`, `_q2_interstitial`, `_energy_dimer`, `_energy_nondimer`,
`n_dimer_spins`, `n_nonzero_spins`) plus the acceptance and provenance keys. Files of an ensemble
extended after the code gained new keys are loaded with the missing keys treated as empty (a
warning names them); the dimer-field and energy-budget outputs of runs recorded before these keys
existed come from the stored states (sections 8.5 and 8.8).

`analysis/` of an ensemble: `ensemble_table.csv` (all observables, initial and final, `x_center` and
`x_mean`), `histories.csv`, `sweep_convergence.csv` (with snapshots), `nn_pairs.csv`, `nn_ranks.csv`,
`nn_distance.csv`, `small_r.csv` (with nn diagnostics), `dimer_map.csv`, `dimer_angular.csv` (with
`C_pred`) and `dimer_field.csv` (with the dimer map), `structure_factor.csv`, `energy_budget.csv`
and `energy_budget.json` (with stored states), `error_budget.csv`, `fit.json`, `summary.json`,
`summary.md`, and the figures `C_of_r`, `decomposition`, `Q_A_of_r`, `overview`, `histories`,
`sweep_convergence`, `nn_diagnostics`, `dimer_map`, `structure_factor` (`.pdf`, or `--formats`).
A protocol check gets `protocol_summary.csv`, `protocol_summary.md`, `protocol_curves_C.csv`,
`protocol_curves_Q.csv` and `protocol.pdf`; a comparison `comparison_landmarks.csv`,
`comparison_landmark_differences.csv`, `comparison.pdf`,
`paired_<obs>_<label>_minus_<reference>.csv` (or `unpaired_*`), `comparison_summary.md` and, for
the channel family, `gamma_family.csv` and `gamma_family.pdf`; `results/_comparisons/cmp_structure_factor/`
holds `structure_factor_vs_cutoff.csv`, `structure_factor_comparison.pdf` and `.md` (section 8.6).

Re-running the same command skips finished realizations (resume); raising `n_realizations` extends the
ensemble; a different configuration in an existing directory is refused (`compat_key`: everything
but `n_realizations`, `save_states` and, for quench ensembles, the annealing schedule). A
`config.json` written before `min_separation_nn` existed loads with its default 0 (the ideal
Poisson process), so the keys of the production runs still match the plan. Files are written
atomically.

## 10. Reproducibility and provenance

* Realization `k` of an ensemble uses seed `base_seed + k` and nothing else; results do not depend on the
  number of worker processes (tested). Re-running a realization reproduces its file bit for bit (apart
  from the timings), so recomputing an entry changes only its provenance stamps.
* The random stream (positions, one discarded draw for particle masses, isotropic directions, spin lengths) is
  identical to that of the exploratory prototype scripts, so historic realizations are reproduced seed by
  seed: `tests/test_legacy_regression.py` checks that the ensemble statistics and energy histories agree
  with reference output of the prototype (open boundaries) to round-off, i.e. a relative `1e-9`
  (`tests/data/README.md`). With a hard core the positions consume a seed-dependent number of draws
  (rejections) before the same sequence continues; they are reproducible and worker-independent as
  well, but share nothing except the seed with the Poisson realization of that seed.
* The numba kernels use strict IEEE arithmetic (no `fastmath`); a fast-math variant is ~28 % faster per sweep
  but was not adopted. Bitwise equality across different CPU architectures is not guaranteed, statistical
  equality is.
* **Provenance.** `config.json` records the package version, the versions of Python, numpy, scipy and
  numba, the platform, the git commit and dirty flag of the code and `source_hash`, the SHA-256 of the
  package sources (`src/grspin/*.py` concatenated in name order, each preceded by its name), which
  identifies the code even when the tree is dirty; for a dirty tree it also stores the porcelain status
  of the package directory and `git diff HEAD -- src/grspin` truncated to 20 kB. The run history lists
  every invocation with its start time, commit, dirty flag and source hash. Every realization file
  written by the runners carries `git_commit` and `source_hash`, so an ensemble extended with other
  code is recognisable file by file.
* `grspin plan provenance` (or `grspin plan status --provenance`) prints, for every entry, the commit,
  the dirty flag and the source hash of `config.json`, the number of realization files (and whether
  they are unstamped or stamped by several code versions) and whether everything matches the current
  tree. `grspin plan status` reports `stale-config` when the stored configuration no longer matches the
  plan; `grspin plan run` skips such entries with a warning unless `--refresh-stale` (recompute them)
  or `--overwrite` is given. The annealing schedule is not part of the configuration key of a plain
  quench ensemble (it is never used there) but it is for protocol checks.
* **Before submission:** commit the final code and tag it, run the plan from the clean tree
  (`grspin plan run all`, with `--overwrite` to restamp everything, then `grspin plan analyze`), and
  verify with `grspin plan provenance` that every entry used in the paper carries the tagged commit,
  a clean tree and a matching source hash (`docs/SIMULATION_PLAN.md`, last section).

## 11. Cost

Per realization: neighbour-list assembly `~2.5e-7 s` per directed edge, `~2.3e-9 s` per edge and sweep, and a
small term for the pair statistics; peak memory `~250 MB + 220 B` per edge. Example (Apple-silicon laptop, one
core, `grspin benchmark`):

| system | edges | measured | memory (peak RSS) |
|---|---:|---:|---:|
| `N=5000`, `r_cut=10 r_nn`, 100 sweeps | 3.6 M | 1.7 s | 1.0 GB |
| `N=5000`, `r_cut=15 r_nn`, 100 sweeps | 12.0 M | 6.1 s | 2.1 GB |
| `N=20000`, `r_cut=7 r_nn`, 100 sweeps | 4.9 M | 2.8 s | 1.4 GB |

Time scales as `N * r_cut^3 * sweeps`. The far-tail set `T` (`x_max = 10 r_nn`, so the pair statistics
cover `(10/4)^3 = 16` times more pairs) and the protocol check at production size are the largest entries
of the plan (cost-model estimates; the exploratory `T` runs took about 13 min on 8 cores, 8 of them for
`T_N10000_rc15` with a single worker):

| entry | edges | estimate per realization | memory per worker |
|---|---:|---:|---:|
| `T_N10000_rc7` | 2.4 M | 3 s | 0.8 GB |
| `T_N10000_rc10` | 7.1 M | 6 s | 1.8 GB |
| `T_N10000_rc15` | 24.0 M | 14 s | 5.5 GB (one worker on a 16 GB machine) |
| `T_N20000_rc10` | 14.2 M | 11 s | 3.4 GB |
| `S5_N5000` (3 quenches + 400 Metropolis sweeps + quench) | 3.6 M | 12 s | 1.0 GB |

The channel controls `S7_gamma1/12/100` cost the same as 100 realizations of `M` (3.1 core-min
each, with the nearest-neighbour diagnostics); `S8_hardcore05` adds the `O(N^2)` sequential
addition of the positions (0.2 s per realization, 3.4 core-min in total). `grspin` limits the
number of workers so that their estimated memory stays below half of the RAM (`--max-memory-gb` to
change). The whole plan is about 1.4 core-hours (`grspin plan list`).

## 12. Validation

`pytest` (about 40 s) covers, among others:

* the two-body ground states of all six couplings: the spectral-norm formula `max(1, |gamma - 1|)`
  of section 1 against a numerical minimization over both directions and against the relaxation
  kernel (antiparallel along the bond for `gamma >= 2`, parallel and transverse for `gamma = 1`,
  parallel along the bond for the dipolar control), and the 1 % transverse admixture of `gamma = 100`;
* monotone energy, exactness of the incremental energy, fixed-point property of the relaxed state,
  conservation of spin lengths, invariance under rescaling of `G`, `S`, `L`;
* hard-core positions: all pair distances `>= r_min` (periodic minimum image and open boundaries),
  reproducibility, the bit-identical Poisson stream for `min_separation_nn = 0`, empty radial bins
  below `r_min`, the configuration round trip and compatibility key (files without the field load
  as 0), the validation bound and the cost term;
* the periodic neighbour list against `O(N^2)` minimum-image enumeration; open versus periodic;
* all pair observables against a brute-force reference (and the prototype's logic), dimer values, random-spin
  baselines, exactness of re-binning;
* the dimer-frame map against an `O(N^2)` reference (open and periodic), synthetic dimers with transverse
  and collinear third spins, the angular table, aggregation and masking, mixed ensembles, reproduction
  from a stored state; reciprocal lattice vectors and `S(k)` of random spins, of a spin wave and at `k = 0`;
* the dimer-field test: a synthetic dimer with the third spin set along / against its dimer field
  gives agreement 1 / 0 and cosine +-1, `sigma` flips the prediction, the frame of `j` gives the
  same field, a brute-force reference for open/GR, periodic/dipolar and periodic/GR, random
  directions give 0.5 and 0, the per-realization aggregation with SEM, the pooled row, the `C_pred`
  column, the stored keys reproduced bit for bit from the stored state, and the fallback to the
  states for records without the map or the field;
* the energy budget: `energy_budget` against `kernels.energy` and an `O(N^2)` sum for open/GR,
  periodic/dipolar and periodic/isotropic with `U_ij >= U_min` for every pair, two isolated dimers
  give `E_nondimer = 0` and realized fraction 1, `energy_budget_from_states` reproduces the stored
  relaxation energies for `gr`, `dipolar` and `isotropic_afm`, the two-body factor against
  Nelder-Mead minimisation over the four angles for nine `(sigma, gamma)` and `theory` against
  `config`; the structure-factor comparison across runs (columns and values against
  `structure_factor_from_states`, default labels, runs without states skipped, the CLI wiring
  `cmp_structure_factor` on the smoke plan with `--max-states`);
* the protocol split: `q = 1` for both classes on the reference state, `E_dimer + E_nondimer = E`,
  `|Delta E / E_nondimer| >= |Delta E / E|` with the same sign, the dimer energy against the pair
  energies of the mutual pairs, the summary columns, version-0.2 files without the split (NaN,
  `protocol_summary.md` says so); the channel-family table and figure of `compare_runs` (order by
  `gamma`, no table for the sign-reversed controls);
* determinism, worker-count independence, resume/extend/overwrite and the refusal of mismatching configurations;
* the Poisson results of section 6 (mean NN distance, `16/27`, `exp(-9 pi rho r^3 / 4)`, Kolmogorov distance of the
  NN-distance law, union volume of two balls);
* bootstrap, landmarks on `x_mean` (including the transverse channel), the error budget, the channel
  decomposition `C = P + Cperp`, fit recovery on synthetic data, paired and unpaired differences;
* the annealing schedule (geometric, `t_start` above the dimer binding energy), the acceptance recorded
  at every temperature, reading of version-0.1 files with the old acceptance key and of mixed checks;
* the provenance (source hash, truncated diff, history, the stamps in every realization file, the
  `plan provenance` report with matching, mixed and unstamped entries);
* the plan (consistency, pairing of `S5_N5000` and of the channel family `S7` with `M`, the far-tail
  set and its comparisons, the hard-core entry `S8` and its unpaired comparison, the derived items
  `D1`-`D3`, the stored-state counts, costs, priorities of mixed groups), its status tracking on disk
  (including stale configurations and state files), the skip/refresh of stale entries, the generated
  plan document and the command line interface (including `--min-separation-nn` and the new couplings);
* regression against the prototype implementation.

## 13. Limitations

* Spins are classical, of fixed length and at fixed positions; there is no dynamics, temperature or backreaction
  of the spin texture on the positions.
* The relaxation finds *a* local minimum; protocol dependence is small for `C` and `Q` (set `S5`) but the
  states are not ground states, and the annealed energies are not converged either (section 8.4).
* The interaction is truncated at `r_cut <= L/2` (no Ewald sum for the conditionally convergent `1/r^3`
  tensor); the dependence is measured, not eliminated. The far tail of `C(r)` beyond `x ~ 4` carries a
  truncation dip whose position scales with `r_cut` (section 8.3), so only `x <= 4` at `r_cut = 10 r_nn`
  is quoted.
* The dimer-frame map is defined for mutual nearest-neighbour pairs only (59 % of the spins for
  Poisson positions).
* Positions are an ideal Poisson process or, as a control, a hard-core process with a minimum
  separation of up to `0.7 r_nn` (section 3, set `S8`); real galaxy samples have clustering, which is
  outside the scope of this code.
