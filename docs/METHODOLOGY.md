# Methodology

## 1. Goals and the optimiser

The mandate has three goals, and their order is binding:

1. Expected return `μᵀw ≥ CPI + spread` (arithmetic expected return).
2. Expected income `yᵀw ≥ income target`. The target can be phased in linearly up to a target year.
3. Volatility `σ(w) ∈ [vol_lo, vol_hi]`.

`σ(w) ≤ vol_hi` is a convex (second-order cone) constraint. `σ(w) ≥ vol_lo` is reverse-convex,
so the band cannot be imposed in a single solve. Instead, the optimiser solves one convex problem
for each target volatility on a grid that spans the band. At each grid point, three stages run in
strict order:

| Stage | Solve |
|---|---|
| A | maximise `μᵀw` |
| B | minimise the income shortfall subject to `μᵀw ≥ min(target, A* − tol)` |
| C | re-maximise `μᵀw` subject to the stage-B shortfall |

The selection then works through four rules, in order:

1. If points in the band meet both return and income, take the lowest risk among them.
2. Otherwise, if points in the band meet return, take the smallest income gap.
3. Otherwise, if allowed (`vol_relax_max`), take the lowest volatility above the band that meets return.
4. Otherwise, take the highest return inside the band.

Within each rule, points that actually reach `vol_lo` are preferred. If no point is feasible at all,
the fallback does two things, and every relaxation is recorded in `selection_rule`:

- it lets the band yield, choosing the lowest volatility that meets the goals;
- only if the constraints themselves are infeasible, it relaxes the turnover cap step by step.

**Constraints.** All constraints are built in one place (`ConstraintSpec`):

- budget;
- single-asset bounds;
- overlapping group limits;
- funding-ladder floors;
- in rebalance mode, a one-way turnover cap and no-trade bands. Private assets have no-trade bands
  by default, because they cannot be traded.

The same constraint set drives:

- the solver;
- the feasibility diagnostics;
- the "maximum attainable weight" problem used to clamp floors that policy cannot accommodate;
- the post-solve audit.

**Performance.** Each stage is a cvxpy problem with Parameters (target volatility, return floor,
shortfall cap). It compiles once per decision date and is then re-solved across the grid.

## 2. Risk model

The covariance is estimated in six steps (`RiskModel.estimate(universe, returns, t)`), using only
`returns[:t]`:

1. Aggregate monthly returns to the coarsest reporting frequency in the universe. Quarterly
   private NAVs are carried flat between reports, never interpolated.
2. Remove appraisal smoothing (Geltner):
   `r* = (r − φ r₋₁)/(1 − φ)`, where φ is the lag-1 autocorrelation, capped at 0.6.
3. Compute an EWMA covariance with a half-life in months (converted to the estimation frequency).
   It is bias-corrected and annualised.
4. Shrink toward constant correlation, with a fixed intensity.
5. Apply a volatility uplift for illiquid sleeves (correlations are kept).
6. Repair to the nearest positive semi-definite matrix.

## 3. Private markets

**Takahashi–Alexander model, per cohort.** Each commitment, capital call or slice of existing NAV
ages on its own clock. At each anniversary, two things are fixed for the coming year:

- the annual call, `RC(age) × uncalled`, which is paid in 12 instalments;
- the payout fraction, `min(1, (age/L)^B + yield)`.

The payout fraction is converted to the monthly rate `m` that satisfies `1 − (1 − m)¹² =` annual
fraction. NAV is marked with:

- realised (reported) returns in the backtest;
- the expected return in projections.

Growth is set equal to the sleeve's expected return, so the vintage IRR equals the optimiser's
expected return. The yield only splits that return between income and appreciation.

**Funding ladder.** Known future calls are bucketed by time to call, using calendar years. The
default edges are 2 and 3 years, so:

- bucket 1 (cash) pre-funds calls due within 2 years;
- bucket 2 (T-bills) pre-funds calls due in 2–3 years;
- bucket 3 (IG bonds) pre-funds calls due later.

Each bucket's floor is `cover × Σ calls / portfolio value`, and buckets do not substitute for each
other. If policy caps cannot hold a floor, the floor is clamped, earliest money first, and the
shortfall is reported.

## 4. Decisions without look-ahead

`StateBuilder.build(t)` is the only place decision inputs are assembled. A decision at month `t`
sees:

- the policy in force (`policy_history.csv`);
- the income target in force;
- the expected-return vintage in force (`cma_history.csv`; if there is none, an optional blend
  with trailing realised returns; if not, the static `assets.csv` values);
- private pacing re-synchronised to those expected returns;
- a covariance estimated on data before `t`.

All objects are immutable, so a state is a pure function of the input files and `t`. The
point-in-time decision is simply `build(T)`.

## 5. Point-in-time allocation, transition plan, backtest

**Strategic target.** Private assets are free within policy, and there are no funding floors.

**Today's portfolio.** Private assets are pinned at their current NAV. Any group limits that NAV
already breaches are widened and reported. Ladder floors come from `commitment_future.csv`. The
problem is solved in rebalance mode, with the fallback.

**Transition plan.** Market inputs are frozen at today's values. The book is rolled forward month
by month from one call date to the next:

1. Liquid assets grow at their expected return.
2. Private cohorts grow and distribute; distributions go to ladder bucket 1.
3. Calls are paid down the ladder, then pro rata from the rest of the liquid book. Anything left
   is reported as an overdraft.
4. Private weights are re-pinned, the ladder is re-sized from the remaining calls, and the liquid
   book is re-optimised.

**Walk-forward backtest.** Each month:

1. Mark the portfolio: liquid income accrues as cash, and private NAV moves with realised returns.
2. Pay calls through the waterfall: cash, then the ladder, then a costed forced sale, then an
   overdraft.
3. Apply external flows.
4. At each annual review, solve the strategic target. Commitments are simulated only if no
   historical call file exists.
5. At each quarterly rebalance, solve with private assets pinned and ladder floors set from the
   remaining calls.

Benchmarks use identical mechanics: no ladder, hold current weights, and 60/40.

## 6. Side analyses

| Module | Method |
|---|---|
| `analytics.risk` | Euler risk contributions on the covariance of record; concentration measured as effective N of weights and of risk |
| `analytics.stress` | Conditional factor stress `m = Σ_aK Σ_KK⁻¹ s` on a joint asset/factor panel treated like the risk model; historical episodes measured as cumulative return and anchored drawdown |
| `analytics.frontier` | Policy, no-privates and unconstrained frontiers; the vertical gaps show what private markets add and what the policy costs |
| `analytics.private_markets` | Monthly cash flows per cohort, IRR / DPI / RVPI / TVPI with legacy NAV counted as paid-in, J-curve, effective number of vintages |
| `analytics.liquidity` | Ladder, other liquid, semi-liquid (gated ELTIF) and illiquid tiers; coverage of the calls due within 1, 2 and 3 years, measured from each portfolio's own date |
| `analytics.sensitivity` | One-at-a-time changes to expected returns per asset class, the covariance level, the targets and the band; binding bounds |
| `analytics.wealth` | Monthly joint-lognormal Monte Carlo with log-drift `log(1+μ) − σ²/2`, following the projected allocation path |
| `analytics.performance` | Annualised return, volatility, Sharpe (excess return), anchored drawdown, VaR/CVaR, tracking error, information ratio |
