# Test data

## `legacy_reference.json`

Ensemble statistics produced by the exploratory prototype script `gr_ensemble_simulation_with_fit.py`
(SHA-256 `8d6d45f60ab162ddcd18e62277293e193224582b9bc90f1b69980d79e05eeef5`) for two tiny configurations. `tests/test_legacy_regression.py` uses them to check
that the package reproduces the earlier results seed by seed (open boundaries, cutoff 10 r_nn, sequential
updates, `U(0, S_max)` spin lengths).

The two cases were generated with

```bash
python gr_ensemble_simulation_with_fit.py --N 60  --L 1000 --M-max 1 --S-max 1000 --iterations 5 \
    --n-seeds 3 --base-seed 1  --nbins 10 --x-max 4 --no-save-particles --output-dir A --output-prefix ref
python gr_ensemble_simulation_with_fit.py --N 150 --L 1000 --M-max 1 --S-max 500  --iterations 8 \
    --n-seeds 4 --base-seed 10 --nbins 16 --x-max 3 --no-save-particles --output-dir B --output-prefix ref
```

and the columns `x_center`, `count_*_total_final`, `n_seed_*_final`, `mean_*`, `sem_*` of
`ref_ensemble_binned_data.csv` plus the per-seed energy and total-spin histories of `ref_histories.csv`
were stored (`null` denotes NaN). The prototype is not part of this repository; the file is a frozen
reference and must not be regenerated from the package.
