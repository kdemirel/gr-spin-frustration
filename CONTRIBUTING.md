# Contributing

Thank you for your interest in `gr-spin-frustration`.

## Development setup

```bash
python -m venv .venv && source .venv/bin/activate     # Python >= 3.10
pip install -e ".[dev]"
pytest -q                                              # ~15 s (numba compiles on first run)
ruff check . && ruff format --check .
```

## Guidelines

- Keep the physics conventions of `docs/METHODS.md` (units, sign of the coupling, definitions
  of the observables). Changes that alter numerical results must update
  `tests/data/` reference values and the changelog, and explain why.
- New functionality needs tests. Statistical tests must be deterministic (fixed seeds).
- After changing `src/grspin/plan.py` regenerate the checklist with
  `grspin plan markdown --write docs/SIMULATION_PLAN.md` (a test checks that it is up to date).
- Commit messages: short imperative summary line, optional body explaining the *why*.
- Do not commit simulation output (`results/` is ignored); commit the configuration that
  produced it (`config.json` is written automatically) or the plan entry in
  `src/grspin/plan.py`.
