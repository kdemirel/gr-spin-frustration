# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [1.0.0] - 2026-10-12

First public release; the code version of the manuscript "Emergent Spin Textures from Frustrated
Gravitational Spin-Spin Interactions". It collects the changes below.

Follow-up to the referee report on the frustrated-magnetism aspects of the manuscript: channel
controls that vary the weight of the longitudinal channel at fixed sign, a causal dimer-field test,
the energy budget and frustration index for every coupling, the structure factor versus cutoff, the
dimer / interstitial split of the protocol check, a hard-core (minimum-separation) ensemble, and the
plan bookkeeping that feeds the energy budget and `S(k)` from stored states. All result files of
0.2.0 still load and analyse (missing keys are filled from the stored states or give NaN).

### Added
- **Dimer-field test** (referee M4). `observables.dimer_field_prediction` and
  `dimer_frame_map(..., sigma_gamma=(sigma, gamma))`: for every (dimer spin `i`, third spin `k`)
  pair of the map the field of the two dimer members alone, `h = T_ki S_i + T_kj S_j` with the
  actual spin vectors, predicts `u_k = sigma h/|h|`; stored per `x_ik` bin (`dimer_field_cnt`,
  `dimer_field_agree`, `dimer_field_sum_cos`), per angular cell (`dimer_ang_sum_Cpred`, the
  dimer-only prediction of the angular table) and overall (`dimer_field_n`, `dimer_field_n_agree`,
  `dimer_field_sigma`, `dimer_field_gamma`); `run_realization` records it with the nn diagnostics.
  Analysis: `tabulate_dimer_field`, `EnsembleData.dimer_field_table()`, `C_pred`/`sem_C_pred` in
  `tabulate_dimer_angular`, `dimer_map_from_states` computes it for older runs; `analyze_run`
  writes `dimer_field.csv`, the `C_pred` column of `dimer_angular.csv`, the `summary.md` section
  "Dimer-field test" and `summary.json["dimer_field"]`, falls back to the stored states for
  records without the map or the field (`source: states`), and prints pooled counts in the angular
  table. Random directions give agreement 0.5 and cosine 0 (tested); on two stored `M` states the
  agreement is 0.97 at `x_ik < 0.75 r_nn` and 0.73 overall.
- **Energy budget / frustration index** (referee M1, m4). `theory.two_body_minimum_factor(sigma,
  gamma)` (= `config.pair_minimum_factor`), `observables.pair_energies`, `observables.mutual_pairs`,
  `observables.energy_budget`, `analysis.energy_budget_from_states` (one row per stored state:
  `energy_total` reproduces the stored relaxation energy to `1e-15`, `energy_dimer`,
  `energy_nondimer`, `minimum_total`, `minimum_dimer`, `dimer_share`, `nondimer_share`,
  `realized_fraction_dimer/nondimer`, `frustration_index = E / sum of two-body minima`,
  `frustration_index_initial`; `analysis.ENERGY_BUDGET_RATIOS`), `analysis.summarize_energy_budget`;
  `analyze_run` writes `energy_budget.csv`, `energy_budget.json` and a `summary.md` section for
  every run with stored states, with the coupling's own two-body minimum; `grspin analyze
  --no-energy-budget`, `--max-states`.
- **Protocol check: dimer / interstitial split** (referee M6). `protocol_realization` records
  `<variant>_q1/q2_dimer`, `_q1/q2_interstitial` (`protocols.SPIN_CLASSES`), `_energy_dimer`,
  `_energy_nondimer`, `n_dimer_spins`, `n_nonzero_spins`; `ProtocolData.summary()` adds
  `overlap_q{1,2}_{dimer,interstitial}`, `delta_energy_nondimer_percent`,
  `delta_energy_dimer_percent`, `energy_dimer_per_spin`, `dimer_energy_share`,
  `dimer_spin_fraction` (NaN for files without the split, `ProtocolData.has_dimer_split`);
  `analyze_protocol` writes `protocol_summary.md` (`report.protocol_markdown`).
- **Structure factor versus cutoff** (referee M5). `report.structure_factor_comparison(run_dirs,
  out_dir, labels=...)` writes `structure_factor_vs_cutoff.csv` (S(0), the first three shells, the
  maximum of S(k) per run), `structure_factor_comparison.pdf` (`plotting.figure_structure_factor_comparison`)
  and `.md`; `grspin plan analyze` writes it as `cmp_structure_factor`
  (`plan.STRUCTURE_FACTOR_COMPARISON_ID`, `plan.STRUCTURE_FACTOR_ENTRIES` = the four `T` entries
  and `M`) whenever at least two of them have stored states; `grspin plan analyze --max-states`.
- **Channel-family figure and table.** `plotting.figure_gamma_family` (`C`, `P`, `Cperp` of the
  members in order of `gamma`), `report.gamma_family_table` (landmarks with bootstrap errors,
  `C(1.2 r_nn) ± SEM` and its sign against `gamma`; `report.GAMMA_FAMILY_X_SIGN`),
  `analysis.gamma_family_members`, `analysis.curve_value_at`; `compare_runs` writes
  `gamma_family.csv`, `gamma_family.pdf` and a `comparison_summary.md` section whenever its members
  form the family (`cmp_channels`).
- **Channel ("gamma") family of couplings.** `transverse_only` `(sigma, gamma) = (1, 1)`,
  `gamma12` `(1, 12)` and `longitudinal_only` `(1, 100)` in `config.COUPLINGS` (hence in
  `--coupling`, the plan and the docs); `config.CHANNEL_FAMILY` orders the family
  (`gamma = 1, 3, 12, 100`). The relaxation uses only the direction of the local field, so the
  overall scale of `H` is irrelevant at `T = 0` and `gamma = 100` is the longitudinal-only limit up
  to a 1 % transverse admixture. `config.pair_minimum_factor(coupling)` returns the two-body
  minimum `U_min = -max(1, |gamma - 1|) G s_i s_j / r^3` of any coupling (either sign of `sigma`;
  2 for GR and dipolar, 1 for `gamma = 1` and the isotropic antiferromagnet, 11 and 99 for
  `gamma = 12, 100`), verified against a numerical minimization over both directions and against
  the relaxation kernel.
- **Hard core.** `SimulationConfig.min_separation_nn` (`0 ... 0.7`, `config.MAX_MIN_SEPARATION_NN`)
  with the property `min_separation`; `initial.hard_core_positions` / `initial.generate_positions`
  draw the positions by sequential random addition with all pair distances
  `>= min_separation_nn r_nn` (minimum image when periodic; deterministic for the seed; `O(N^2)`,
  0.2 s at `N = 5000`); the CLI flag `--min-separation-nn`; the field is part of `to_dict`,
  `from_dict` and `compat_key` (a `config.json` written before it existed loads with 0 and keeps
  matching) and the cost model charges the sequential addition. For `min_separation_nn = 0`
  positions, directions and lengths are bit-identical to 0.2.0.
- **Plan.** Group `S7` "Channel controls (gamma family)": `S7_gamma1`, `S7_gamma12`, `S7_gamma100`
  (`N = 5000`, periodic, `r_cut = 10 r_nn`, 100 sweeps, 100 realizations on the seeds of `M`,
  nearest-neighbour diagnostics, 16 stored states, recommended; purpose "Channel controls: which
  channel carries the positive shell?") with the paired comparison `cmp_channels` (reference `M`,
  observables `C, P, Cperp, Q, A`). Group `S8`: `S8_hardcore05` (`min_separation_nn = 0.5`, 100
  realizations, nearest-neighbour diagnostics, 16 stored states, optional) with the unpaired
  comparison `cmp_hardcore`. Derived item `D3` "channel controls and dimer-field test".
  `PlanEntry.describe()` shows `r_min`; `docs/SIMULATION_PLAN.md` regenerated (36 runs, 1.4
  core-hours).
- Tests: `tests/test_config.py` (coupling table, two-body minima of all couplings, the 1 %
  admixture, hard-core validation, round trip, compatibility key, cost), `tests/test_initial.py`
  (hard-core positions periodic and open, reproducibility, the historic stream bit for bit, empty
  bins below `r_min`), the dimer-field test (synthetic dimers, brute force, random directions,
  aggregation, the states fallback), the energy budget (against the kernel energy and an `O(N^2)`
  sum, isolated dimers, stored energies, the two-body factor by numerical minimisation), the
  protocol split, the structure-factor comparison, the channel-family table and figure, and
  plan/CLI tests of the new entries, comparisons, derived items and flags (about 190 tests, 45 s).

### Changed
- Version `1.0.0` in `pyproject.toml`, the fallback `__version__` and `CITATION.cff`.
- `analysis.PAIRING_KEYS` includes `min_separation_nn`: hard-core runs are never paired with
  Poisson runs of the same seeds (`cmp_hardcore` is unpaired). `comparison_summary.md` gains a
  `positions` column (`Poisson` or `hard core <x> r_nn`) and the `summary.md` header names the
  position process; the docstrings of the Poisson results in `theory` say that they hold for the
  ideal process only.
- The derived items `S7` and `S8` of the plan are renamed `D1` and `D2`; `S7` and `S8` now denote
  the ensemble groups above. `D2` also covers the energy budget and `cmp_structure_factor`.
- `save_states = 16` for the far-tail entries `T_N10000_rc7/rc10/rc15` and `T_N20000_rc10` (was 2)
  and for `S6_isotropic_afm` and `S6_dipolar` (was 0): their stored states feed the structure
  factor versus cutoff and the energy budget. Existing results of these entries are reported as
  `partial` until `grspin plan run T S6` writes the missing state files (only the realizations
  whose states are now requested are recomputed; realizations are deterministic, so their records
  do not change).
- `docs/METHODS.md`: channel family and two-body minima (section 1), hard-core positions
  (section 3), the Poisson-only validity of the dimer formulas (section 6), channel controls and
  the gamma-family table (8.1), the protocol split (8.4), the dimer-field test (8.5), stored states
  for `S(k)` and the comparison across runs (8.6), the hard-core set (8.7), the energy budget and
  frustration index (new 8.8), output files and record keys (9), cost (11), validation (12) and
  limitations (13); README plan table, quick start, command table and layout.

## [0.2.0] - 2026-10-10

The release behind the production results of the manuscript: the channel decomposition of the
pair correlation, the dimer-frame correlation map, a full error budget for the quoted landmarks,
the far-tail runs and a provenance trail that identifies the code behind every file.

### Added
- **Channel decomposition.** `P(r) = <(u_i.rhat)(u_j.rhat)>` (longitudinal channel) and
  `Cperp(r) = C - P` (transverse channel) are first-class observables of the analysis
  (`OBSERVABLES = C, Q, A, P, Cperp, B`; `mean/sem_P_*` and `mean/sem_Cperp_*` columns of
  `ensemble_table.csv`; `realization_means`, `pooled_means`, `grspin compare --obs P Cperp`).
  New landmarks `x_peak_Cperp`, `Cperp_peak` (parabolic) and `P_at_Cperp_peak` (interpolated).
- **Landmarks on `x_mean`.** Landmarks are evaluated on the pair-weighted mean distance of each
  bin (`x_mean()`, `abscissa()`, `bin_width()`; falls back to the bin centre for runs without
  `sum_x`) at the stored bin width (`landmark_rebin`, default 1); `_parabolic_extremum` handles
  unequal spacing; the landmark `_meta` records abscissa, rebin, number and width of the bins;
  all observables are resampled with one bootstrap index set.
- **Error budget.** `analysis.landmark_error_budget(main, sweeps_run, cutoff_run, rebins=(1, 2))`
  combines the bootstrap error with the binning difference and the paired shifts against a longer
  relaxation (`S1`) and a larger cutoff (`S2_rc15`) into `error_budget.csv` and a `summary.md`
  section; `analysis.landmark_difference` gives paired landmark shifts with a joint bootstrap;
  `report.find_budget_runs` finds the paired sibling runs; `grspin analyze --sweeps-run/--cutoff-run`.
- **Paired versus unpaired comparisons.** `same_system()` / `is_paired()` (`PAIRING_KEYS`:
  what generates positions, lengths and directions); `paired_difference` falls back to the
  unpaired difference over all realizations (error `sqrt(sem_a^2 + sem_b^2)`, `paired` column);
  `compare_runs` writes `paired_*` or `unpaired_*` tables, `comparison_landmark_differences.csv`
  and `comparison_summary.md`; the landmark tables gain coupling, spin distribution, rebin and
  abscissa columns.
- **Energies.** `histories.csv` adds mean, median and interquartile range of the energy per
  spin, the median and IQR of the excess energy and the median maximal direction change;
  `EnsembleData.energy_summary()` and a `summary.md` section (ensemble-mean energies are
  dominated by the closest pairs; medians and paired differences are the robust quantities).
- **Dimer-frame correlation map** (`observables.dimer_frame_map`, stored with
  `nn_diagnostics=True`, keys `dimer_map_*` and `dimer_ang_*`): for every spin of a mutual
  nearest-neighbour pair, `C` and `P` of all third spins within 2.5 r_nn on a cylindrical
  `(z, rho)` grid of 50 x 25 cells and in a 6 x 7 angular table; `nn.mutual_partners()` and
  `nn.mutual_nn_partners()`; `EnsembleData.dimer_map()`, `dimer_map_table()`,
  `dimer_angular_table()`, `analysis.dimer_map_from_states()` for older runs with stored states;
  `plotting.figure_dimer_map`; `dimer_map.csv`, `dimer_angular.csv`, `dimer_map.pdf` and new
  `summary.json` / `summary.md` sections. `theory.MUTUAL_PAIR_MEAN_DISTANCE_X = (16/27)^(1/3)`
  (mean dimer separation 0.84 r_nn) with the derivations of the 16/27 and `9 pi r^3 / 4` results.
- **Spin structure factor** of the stored states (`observables.reciprocal_lattice_vectors`,
  `observables.spin_structure_factor`, `analysis.structure_factor_from_states`,
  `plotting.figure_structure_factor`; `structure_factor.csv` and `.pdf`, shell-averaged for the
  initial and the final spins with errors over the states).
- **Plan.** Group `T` "Far tail and cutoff" (`T_N10000_rc7/rc10/rc15`, `T_N20000_rc10`:
  `x_max = 10 r_nn`, 200 bins, 32 realizations, two stored states each) with the comparisons
  `cmp_far_tail` (paired, reference `r_cut = 15`) and `cmp_far_tail_size` on `C`, `P`, `Cperp`;
  the protocol check at production size `S5_N5000` (N = 5000, r_cut = 10 r_nn, 48 realizations
  on the seeds of `M`); `M` stores 48 states (dimer map and `S(k)`); derived item `S8` (renamed
  `D2` in 0.3.0); groups
  with mixed priorities are listed once in `docs/SIMULATION_PLAN.md`; the pre-submission
  checklist asks to commit, tag, run from the clean tree and verify the provenance.
- **Provenance.** `config.json` records a SHA-256 hash of the package sources (`source_hash`)
  and, for a dirty tree, `git diff HEAD -- src/grspin` (truncated to 20 kB) and the porcelain
  status; the run history records the hash and the dirty flag; every realization file written
  by the runners carries `git_commit` and `source_hash`. `grspin plan provenance` (and
  `grspin plan status --provenance`) prints commit, dirty flag and source hash of every entry
  and whether they match the current tree. `grspin plan run --refresh-stale` recomputes entries
  whose stored configuration differs from the plan; without it they are skipped with a warning
  instead of aborting the run.
- **Annealing acceptance at every temperature.** `kernels.metropolis` returns the acceptance
  rate of every temperature; records and protocol checks store `anneal_acceptance_per_T` and
  `anneal_acceptance_final` (`ProtocolData.acceptance_table()`, columns
  `anneal_acceptance_first/final` of `protocol_summary.csv`); `relax.anneal_schedule()`.
- **Figures.** `draw_decomposition` / `figure_decomposition` (C, P, Cperp with the tail zoom);
  `figure_overview` is a 2 x 3 grid with the decomposition; `draw_C(show_nn=True)` draws the
  exact mutual-nearest-neighbour curve `-exp(-9 pi rho r^3 / 4)` ("dimer fraction");
  `figure_compare` handles derived observables; `grspin analyze --show-fit`,
  `--landmark-rebin`, `grspin compare --landmark-rebin`.
- Tests for all of the above (`tests/test_report.py`, `tests/test_dimer_map.py` and additions to
  the existing modules; 123 tests, about 35 s).

### Changed
- `AnnealConfig.t_start` default raised from 2.0 to 10.0 `G S^2 rho`, above the typical dimer
  binding energy 2.9 (`config.DIMER_BINDING_ENERGY`); `t_end = 0.01` and 40 temperatures are
  unchanged. Protocol checks recorded with the old schedule are reported as `stale-config`.
- `SimulationConfig.compat_key(kind)` ignores the annealing schedule for a plain quench
  ensemble (it is never used there), so the production ensembles of 0.1 remain compatible;
  for protocol checks (`kind="protocol"`) the schedule is part of the key.
- `RelaxResult.anneal_acceptance` (a scalar) is replaced by `anneal_acceptance_per_T` (array)
  and the property `anneal_acceptance_final`; the record key `anneal_acceptance` of 0.1 files is
  still read by `ProtocolData`.
- `plan.entry_status` counts a realization as finished only when its state file (first
  `save_states` seeds) exists as well, consistently with the runner.
- The damped-oscillatory fit is a phenomenological guide only: not drawn by default, relabelled
  in `summary.md`, not used for any quantitative statement (`docs/METHODS.md`, section 7); its
  curve is evaluated on `x_mean`.
- `draw_A` / `draw_B` derive their y-limits from the data (the dipolar `B` curve is visible).
- `EnsembleData.from_records` tolerates records that lack some keys (ensembles extended after
  the code gained new keys); `ProtocolData.load` drops keys missing from some files.
- Nearest-neighbour tables report NaN errors where a category has fewer than two pairs or
  contributing realizations; `sweep_convergence` uses the landmark values (parabolic peak etc.).
- `docs/METHODS.md`: channel form of the interaction, `P`/`Cperp`, `x_mean`, error budget,
  paired/unpaired differences, energies, far tail and truncation effect, protocol check and
  annealing schedule, dimer-frame map, `S(k)`, output keys, provenance and cost of the `T` set;
  the plan text no longer states that only `Q` depends on the spin-length distribution.

### Fixed
- `draw_B` clipped the dipolar control; landmark positions were biased by the bin-centre
  abscissa; comparisons across different `N` were labelled "paired".

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
