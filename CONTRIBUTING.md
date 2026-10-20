# Contributing

Thanks for your interest! Issues and pull requests are welcome.

## Set up

```bash
git clone https://github.com/Codes-Dennis-Jung/saa-pm.git
cd saa-pm
pip install -e ".[dev]"
pre-commit install          # runs ruff on every commit
```

## Before opening a pull request

```bash
ruff check src tests examples
ruff format src tests examples
mypy
pytest
```

All four must pass; CI runs the same checks on Linux and Windows.

## Guidelines

- **Never commit real market data or client data.** Examples and tests use the synthetic generator
  (`saa-pm sample-data`).
- Objects describing the universe and settings are immutable. Derive new ones (`Universe.updated`,
  `dataclasses.replace`) instead of mutating them.
- Anything a decision at date *t* uses must come from `StateBuilder.build(t)`, so there is no
  look-ahead.
- All portfolio constraints live in `optimizer/constraints.py:ConstraintSpec`. Add new constraints
  there, so the solver, the diagnostics and the audit stay consistent.
- Add a test for every bug fix, and add an entry to `CHANGELOG.md` under *Unreleased*.
