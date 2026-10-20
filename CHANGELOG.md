# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Added
- **Semi-liquid (evergreen / ELTIF) sleeves.** A new optional `assets.csv` column, `redemption_limit`, is the maximum share of a holding that can be redeemed per rebalance (the fund's gate). A gated sleeve (`liquid=True` + `redemption_limit`):
  - can be bought at NAV, with no commitments, pacing or funding ladder;
  - can only be sold through the gate (a constraint plus a `redemption gates` audit check);
  - gets the illiquidity uplift;
  - is never used for forced sales or as the sweep asset;
  - is reported as its own `semi_liquid` liquidity tier.

  New `Asset.semi_liquid`, `Universe.semi_liquid_names` and `Universe.fully_liquid_mask`.
- `examples/data_check.py` has two input checks. `check_inputs` covers the file inventory, a value-by-value reconciliation of every input file against the loaded model (returns recomputed independently from the levels) and plausibility checks: outliers, typo spikes, stale feeds, gaps, the NAV calendar, several rows in one month, truncated series, unknown keys and unknown factors. `stress_test_inputs` runs 31 fault-injection cases: format variants must load identically, and faults must be caught or flagged. Command line: `python examples/data_check.py <folder> [--stress]`.
- Two executed case-study notebooks for the same USD 100m mandate:
  - `examples/Cases_study_SAA_ClosedEnd_PrivateMarkets.ipynb`: closed-end funds, commitment pacing plan, funding ladder;
  - `examples/Cases_study_SAA_SemiLiquid_PrivateMarkets.ipynb`: ELTIF funds, subscription plan, gate-based liquidity management, comparison of the two vehicles.
- `docs/TECHNICAL_DESCRIPTION.md`: every computation, with formulas, from the input layer to the analytics.
- Optional extra `notebooks` (`pip install -e ".[notebooks]"`): JupyterLab plus the report dependencies, to run the case studies.
- Tests for the data check, fault injection and semi-liquid sleeves (`tests/test_data_check.py`, plus new cases in `tests/test_core.py`).

### Fixed
- Python 3.13: the `from_frames` test compared settings that contain NaN ("not set") with dataclass `==`, which since Python 3.13 compares fields with `==` (`NaN != NaN`). The test now compares field by field.
- `Universe.pin_privates` widened a group cap only when the pinned private NAV *alone* exceeded it. In mixed groups (e.g. gold + infrastructure), pinned NAV plus the liquid members' lower bounds could exceed the cap and make the transition solve infeasible. The liquid lower bounds now count towards the forced group weight.

### Changed
- The synthetic example mandate (`saa-pm sample-data`, `examples/data`) now has 24 asset classes (government bonds by region, IG/HY/EM credit, nine equity sleeves, gold, three private sleeves) and the stress factors `Global_Equity`, `EUR_USD`, `Oil_Brent`, `Global_Govt`, `Global_Credit`. Returns are drawn jointly from rounded, plausible volatilities and correlations; private sleeves report quarterly, appraisal-smoothed NAVs. The repository ships no market data.
- README examples, `docs/EXAMPLES.md` and the figures in `docs/images` were regenerated from the new example data.
- Project URLs point to `github.com/Codes-Dennis-Jung/saa-pm`.

## [1.1.0] - 2026-10-19

### Added
- `MandateData.from_frames(...)`: build a mandate from in-memory pandas DataFrames instead of a folder.
- Package logging through the `saa_pm` logger. The package is silent by default; `SAAStudy(verbose=True)` and the CLI print to the console.
- Optional extras: `pip install saa-pm[report]` adds charts and the Excel workbook. The core needs only numpy, pandas, scipy and cvxpy.
- `py.typed` marker, so type checkers see the package's annotations.
- CI on GitHub Actions: ruff, mypy, and tests on Linux and Windows with Python 3.10–3.13, an end-to-end example run, and a build check.
- Release workflow that publishes to PyPI on version tags, using trusted publishing.
- pre-commit configuration, `CONTRIBUTING.md`, `CITATION.cff`, this changelog.

### Changed
- Importing `saa_pm` no longer loads matplotlib. Charts are drawn on detached figures inside a temporary style context, so the user's backend and global style are never changed (important in Jupyter).
- Code formatted with ruff. Lint and type checks are clean.
- Sensitivity cases are built from the named helpers `shift_cma`, `scale_covariance` and `change_config`.

### Fixed
- `TodayProblem.config` is now typed as `OptimizerConfig`.

## [1.0.0] - 2026-10-19

### Added
- First release of the `saa_pm` package: goal-ordered
  optimiser, funding ladder, private-markets pacing book, point-in-time allocation, transition plan,
  walk-forward backtest, side analyses, CLI, synthetic example data, tests and documentation.
