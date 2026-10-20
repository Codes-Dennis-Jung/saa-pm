# Technical description

This document specifies every computation in `saa-pm` precisely enough to re-implement it. It
starts with the global architecture and data flow (section 1). It then follows the processing
chain from the input files to the reported analytics, one subsection per stage, each more granular
than the last. Formulas use the notation of section 0. Defaults are the values in
`saa_pm/config.py` unless stated otherwise. The source location of each step is named in
*italics*.

For the investment rationale, see [`METHODOLOGY.md`](METHODOLOGY.md). For worked, executed
examples, see the case-study notebooks in [`examples/`](../examples).

---

## Contents

0. [Notation](#0-notation)
1. [System overview](#1-system-overview)
2. [Input layer](#2-input-layer)
3. [Universe and the constraint set](#3-universe-and-the-constraint-set)
4. [Risk model](#4-risk-model)
5. [Point-in-time decision state](#5-point-in-time-decision-state)
6. [Goal-ordered optimiser](#6-goal-ordered-optimiser)
7. [Private markets: pacing model, private book, funding ladder](#7-private-markets-pacing-model-private-book-funding-ladder)
8. [Point-in-time allocation](#8-point-in-time-allocation)
9. [Transition projection at capital-call dates](#9-transition-projection-at-capital-call-dates)
10. [Walk-forward backtest](#10-walk-forward-backtest)
11. [Analytics](#11-analytics)
12. [Semi-liquid (evergreen / ELTIF) sleeves](#12-semi-liquid-evergreen--eltif-sleeves)
13. [Case-study computations (notebooks)](#13-case-study-computations-notebooks)
14. [Input data check and fault injection](#14-input-data-check-and-fault-injection)
15. [Numerical settings and tolerances](#15-numerical-settings-and-tolerances)
16. [Verification](#16-verification)

---

## 0. Notation

| Symbol | Meaning |
|---|---|
| $n$ | number of assets (sleeves) in the universe, indexed $i = 1..n$ |
| $w \in \mathbb{R}^n$ | portfolio weights, $\mathbf{1}^\top w = 1$ |
| $w^{prev}$ | weights before a rebalance |
| $\mu, y \in \mathbb{R}^n$ | expected **arithmetic** annual total return and expected annual cash income (`exp_return`, `income`) |
| $\Sigma \in \mathbb{R}^{n\times n}$ | annualised covariance of record, $\Sigma = L L^\top$ (Cholesky) |
| $\sigma(w) = \lVert L^\top w \rVert_2 = \sqrt{w^\top \Sigma w}$ | portfolio volatility |
| $\ell_i, u_i$ | single-asset bounds (`lb`, `ub`) |
| $G = (M_G, lo_G, hi_G)$ | group constraint with member set $M_G$ |
| $R^*$, $Y^*$ | return target $=\text{CPI}+\text{spread}$ and income target in force |
| $[\sigma_{lo}, \sigma_{hi}]$ | risk band (`vol_lo`, `vol_hi`) |
| $k$ | estimation period in months (1, 3, 6 or 12) |
| $r_t$ | simple return of period $t$ |
| $\varphi$ | Geltner smoothing parameter |
| $PV$ | portfolio value in currency |
| $\mathcal{L}$ | set of liquid assets; $\mathcal{I}$ illiquid (closed-end) assets; $\mathcal{S}\subset\mathcal{L}$ semi-liquid (gated) assets |

All rates are annual decimals. "Month" means calendar month-end.

---

## 1. System overview

### 1.1 Processing chain

```text
input folder (CSV) ──► io.loaders.load_mandate ──► MandateData ──► io.validation.validate
                                                     │
                                                     ▼
                                engine.state.StateBuilder.build(t)   (point in time)
                                  ├─ policy in force          (policy_history)
                                  ├─ CMAs in force            (cma_history / blend / static)
                                  ├─ goals & income glide     (config)
                                  ├─ pacing synced to CMAs    (pacing)
                                  └─ RiskModel.estimate       (returns before t)
                                                     │  DecisionState
                 ┌───────────────────────┬───────────┴───────────┬────────────────────────┐
                 ▼                       ▼                       ▼                        ▼
   PointInTimeAllocator     CallDateProjector        WalkForwardBacktest /       analytics.*
   strategic + today        transition per call      BacktestSuite               risk, stress, frontier,
   (GoalOptimizer)          date (GoalOptimizer)     (GoalOptimizer each month)  liquidity, income, pm,
                                                                                 sensitivity, wealth
                 └───────────────────────┴───────────────────────┴────────────────────────┘
                                                     ▼
                                reporting.writer / charts ──► results/ (CSV, xlsx, PNG, SUMMARY.md)
```

`SAAStudy.run(steps)` (*study.py*) runs the steps `allocation`, `projection`, `backtest` and
`analyses` in that order and writes the results. The CLI (`saa-pm check | run | sample-data`,
*cli.py*) is a thin wrapper around it.

### 1.2 Design invariants

1. **One constraint set.** `ConstraintSpec` (*optimizer/constraints.py*) is the only definition of
   feasibility. It is used by the solver, the diagnostics, floor clamping, the frontiers and the
   audit.
2. **Point in time.** A decision at month $t$ uses only `returns[: t]`, the policy and CMA
   vintages effective on or before $t$. The live decision is `build(T)`, built with the same code
   as every backtest date.
3. **Immutability.** `Asset`, `GroupConstraint`, `Universe` and the config objects are frozen. A
   state is a pure function of (files, $t$).
4. **No silent repair.** Every relaxation (a clamped floor, a widened cap, a relaxed band or
   turnover cap, an overdraft) is recorded in `selection_rule`, `notes` or `events`.

---

## 2. Input layer

### 2.1 File parsing (*io/csvio.py*)

| Step | Rule |
|---|---|
| Separator | Vote over the first ≤ 6 non-comment lines among `, ; \t \|`; ties are broken by equal counts per line. |
| Encoding | UTF-8 with BOM stripping (`utf-8-sig`). |
| Decimal style | Decided **once per file**. German style (decimal comma) if numeric tokens match `[-+]?(\d{1,3}(\.\d{3})+|\d+)(,\d+)?%?`; every numeric token is then rewritten to US form (`'1.114,07' → '1114.07'`, `'3,5%' → '3.5%'`). Dates such as `31.03.2016` do not match and pass unchanged. |
| Scalars | `to_float` accepts `0.034`, `3.4%`, `1,500,000`, `1'500`; blank or NA tokens → NaN. |
| Rates | `as_rate(v)`: if $\lvert v\rvert > 1$ then $v \leftarrow v/100$, with a note. Applied only to rate fields (`exp_return`, `income`, `lb`, `ub`, `max_trade`, `redemption_limit`, rate keys in `config`/`policy_history`, `drawdown_ratio`, `yield_rate`). |
| Dates | `parse_dates` tries day-first and month-first. It prefers fewer failures, then a monotonic result, then day-first. Unparseable dates raise an error. |

### 2.2 Prices to returns (*io/loaders.load_prices*)

Given the levels $P_{i,d}$ on raw dates $d$:

1. Sort; duplicate dates raise; non-numeric cells raise.
2. Resample to month-end with the **last** observation in each month: $P_{i,m} = P_{i,\max\{d \in m\}}$.
3. For each asset, restrict to $[\text{first valid}, \text{last valid}]$. Non-positive levels raise
   (this catches returns supplied in place of levels).
4. Internal gaps are **carried flat** (forward fill). For `nav_frequency = M` this is reported as a
   note. For `Q/SA/A` it is the intended treatment of low-frequency NAVs: the period return is
   booked in the reporting month. There is no interpolation, which would leak look-ahead and add
   autocorrelation.
5. Monthly returns: $r_{i,m} = P_{i,m} / P_{i,m-1} - 1$.

`common_returns` is the set of months in which every asset has a return. The as-of date is its
last month, so a series ending early pulls the as-of date back (the data check flags this).

### 2.3 Other files

| File | Parsed into | Key rules |
|---|---|---|
| `assets.csv` | `Asset` objects | required: `name, group, exp_return, income, liquid`. `ladder_bucket` must be an integer; `nav_frequency ∈ {M,Q,SA,A}`; optional `redemption_limit` (section 12) |
| `groups.csv` | `GroupConstraint(name, members, lo, hi)` | members separated by `\|` |
| `holdings.csv` | `Holdings(market_value, undrawn)` | unknown or duplicate assets raise; negative values raise; undrawn amounts only on illiquid sleeves |
| `config.csv` | `Settings` | unknown keys are kept and reported; obsolete keys are ignored |
| `pacing.csv` | `PacingParams` per illiquid sleeve | `growth_rate` is **forced** to the CMA (section 7.1) |
| `commitment_future.csv` | `schedule` (date, asset, amount) | only illiquid sleeves; amounts ≥ 0 |
| `cma_history.csv`, `policy_history.csv`, `flows.csv`, `commitments_actual.csv` | `Panel` with as-of lookups | policy keys are validated against a whitelist |
| `stress_factors.csv`, `stress_scenarios.csv` | factor levels (month-end) and scenarios | shocks with $\lvert s\rvert > 1.5$ raise (probably a percent unit error) |

`MandateData.from_frames` writes in-memory frames to a temporary folder and runs the identical
loader.

### 2.4 Settings mapping (*config.Settings.from_mapping*)

`config.csv` keys map onto five frozen objects:

* `Goals`: cpi, spread, income target, glide, band;
* `OptimizerConfig`: grid, turnover, tolerances, solver;
* `RiskModelConfig`: half-life, shrinkage, uplift, lookback, φ-cap;
* `LadderConfig`: edges, cover;
* `BacktestConfig`.

The ladder edges are `ladder_edge_1..N-1` for $N$ ladder buckets. A missing edge raises an error.
The number of edges must equal $N-1$.

**Income glide.** With $Y_{fin}$ = `target_income`, $T_{fin}$ = `income_target_year`, $T_0$ =
`income_glide_start_year` and $Y_0$ = `income_glide_start_income`, the target at decimal year
$\tau = \text{year} + (\text{month}-1)/12$ is

$$
Y^*(\tau) = \begin{cases} Y_0 & \tau \le T_0 \\ Y_0 + (Y_{fin}-Y_0)\dfrac{\tau - T_0}{T_{fin}-T_0} & T_0 < \tau < T_{fin} \\ Y_{fin} & \tau \ge T_{fin} \end{cases}
$$

The return target is $R^* = \text{cpi} + \text{return\_spread}$, unless `target_return` overrides
it.

### 2.5 Validation (*io/validation.validate*)

**Errors** block the study:

* structural conflicts (`Universe.structural_problems`):
  * $\ell_i > u_i$;
  * $\sum \ell > 1$ or $\sum u < 1$;
  * unknown group members;
  * $lo_G > hi_G$;
  * member lower bounds above $hi_G$;
  * a group floor unreachable given non-member floors;
  * non-members forcing the group above its cap;
* ladder numbering not equal to $1..N$;
* an illiquid or gated asset in the ladder;
* pacing defined for a liquid asset;
* pacing calibration problems (section 7.1);
* common history shorter than $\max(24, 3k)$ months;
* scheduled sleeves without pacing;
* `redemption_limit` outside $(0, 1]$.

**Warnings** do not block:

* current weights outside the bounds;
* an illiquid sleeve without pacing, or not appraisal-based;
* realised vs expected return gap $> 6\%$ (compared with the geometric approximation
  $(1+\mu)e^{-\sigma^2/2}-1$);
* a series that looks like a *price* index;
* lag-1 autocorrelation $> 0.30$ in a sleeve that is not appraisal-based;
* $\varphi$ at its cap;
* undrawn amounts that differ from the scheduled calls;
* a schedule without a ladder;
* a `redemption_limit` on a closed-end sleeve.

The **coverage table** reports, per asset:

* first and last month, and the number of months;
* realised annualised return;
* reported and desmoothed volatility at the estimation frequency;
* lag-1 autocorrelation;
* $\varphi$.

---

## 3. Universe and the constraint set

### 3.1 Asset classes of liquidity

| Class | Flag(s) | Treatment |
|---|---|---|
| liquid | `liquid=True`, no `redemption_limit` | freely traded |
| semi-liquid | `liquid=True`, `redemption_limit = ρ` | bought freely; sells limited by the gate $\rho$ (section 12) |
| illiquid (closed-end) | `liquid=False` | pinned at NAV in every rebalance; changes only through calls and distributions |

### 3.2 `ConstraintSpec.build` (*optimizer/constraints.py*)

For decision variable $w$:

$$
\begin{aligned}
&\mathbf{1}^\top w = 1, \qquad \ell \le w \le u \\
&lo_G \le \textstyle\sum_{i\in M_G} w_i \le hi_G \qquad \forall G \\
&\textstyle\sum_{i \in B_k} w_i \ge f_k \qquad \text{(funding-ladder floor of bucket } k\text{, section 7.4)}
\end{aligned}
$$

and, **only in rebalance mode** with $w^{prev}$ given:

$$
\begin{aligned}
&\tfrac12 \lVert w - w^{prev}\rVert_1 \le \tau &&\text{(one-way turnover cap, if set)}\\
&\lvert w_i - w^{prev}_i\rvert \le c_i &&\text{(no-trade band; } c_i = \texttt{max\_trade}_i\text{, default } \texttt{illiquid\_max\_trade}=1\%\text{ for } i\in\mathcal{I})\\
&w^{prev}_i - w_i \le \rho_i\, w^{prev}_i &&\forall i \in \mathcal{S}\ \text{(redemption gate)}
\end{aligned}
$$

The volatility constraint $\lVert L^\top w\rVert_2 \le \bar\sigma$ is added by the optimiser
(section 6).

### 3.3 Audit (*ConstraintSpec.audit*)

The audit re-checks a solution numerically and independently of the solver. Each check has a
tolerance:

| Check | Tolerance |
|---|---|
| budget | $10^{-7}$ |
| asset bounds | $10^{-6}$ |
| group bounds | $10^{-6}$ |
| $\sigma \le$ ceiling | $10^{-5}$ |
| $\sigma \ge \sigma_{lo}$ | $10^{-4}$ |
| income $\ge Y^* -$ `return_tol` | $10^{-6}$ |
| return $\ge R^* -$ `return_tol` | $10^{-6}$ |
| turnover, no-trade bands, gates, ladder floors | as specified in section 3.2 |

The result is a table with `check`, `pass` and `value`. Failures are surfaced as
`audit_failures`.

### 3.4 Pinning closed-end sleeves (*Universe.pin_privates*)

Given current weights $w^0$, every $i\in\mathcal{I}$ gets the bounds
$[\max(0, w^0_i-\epsilon), w^0_i+\epsilon]$ with $\epsilon = 10^{-6}$. For each group, the
**forced** weight is

$$
F_G = \sum_{i\in M_G \cap \mathcal{I}} w^0_i + \sum_{i \in M_G \cap \mathcal{L}} \ell_i
$$

(zero for purely liquid groups). If $F_G > hi_G$, the cap is widened to $F_G+\epsilon$ and a
`GroupBreach(G, F_G, hi_G)` is recorded.

### 3.5 Clamping group floors (*goal_optimizer.clamp_group_floors*)

For each group $G$, an LP over the remaining constraints computes
$\bar x_G = \max \sum_{i\in M_G} w_i$. If $lo_G > \bar x_G$, the floor is set to
$\max(0, \bar x_G - 10^{-6})$ and reported. This is needed, for example, when pinned privates
make the "Private Markets ≥ 6%" floor unreachable.

---

## 4. Risk model

*risk/covariance.RiskModel.estimate(universe, returns, t)*, using `returns.iloc[max(0, t-L):t]`
with $L$ = `max_lookback` months.

### 4.1 Aggregation to the estimation frequency

$k = $ `agg_months` or the coarsest `nav_frequency` in the universe. Monthly returns are compounded
into $k$-month buckets aligned to the **end** of the window (the ragged head is dropped):

$$
R_{i,b} = \prod_{m \in b} (1 + r_{i,m}) - 1 .
$$

### 4.2 Geltner desmoothing (appraisal-based sleeves)

$\varphi_i = \operatorname{clip}\big(\operatorname{corr}(R_{i,b}, R_{i,b-1}), 0, \varphi_{cap}\big)$,
with $\varphi_{cap}=0.6$. If the correlation is not finite, $\varphi_i = 0$. Then

$$
R^{*}_{i,b} = \frac{R_{i,b} - \varphi_i R_{i,b-1}}{1-\varphi_i}.
$$

The first bucket becomes NaN, and rows with NaN are dropped. The smoothing model behind the filter
is $R_b = (1-\varphi)R^*_b + \varphi R_{b-1}$. The filter scales volatility by roughly
$\sqrt{(1+\varphi)/(1-\varphi)}$ and leaves the mean unchanged.

### 4.3 EWMA covariance

The half-life in periods is $h = \max(2, \texttt{halflife}/k)$ and $\lambda = 0.5^{1/h}$. For $N$
observations $x_1..x_N$ (oldest first):

$$
\omega_j = \frac{\lambda^{N-j}}{\sum_{l}\lambda^{N-l}},\quad
\bar x = \sum_j \omega_j x_j,\quad
S = \frac{\sum_j \omega_j (x_j-\bar x)(x_j-\bar x)^\top}{1-\sum_j \omega_j^2}\cdot \frac{12}{k}.
$$

The denominator is the unbiased-weights correction. $N_{eff} = 1/\sum_j\omega_j^2$ is reported.

### 4.4 Constant-correlation shrinkage

With $v_i=\sqrt{S_{ii}}$, $C = S/(vv^\top)$ and $\bar r$ = the mean off-diagonal correlation:

$$
C^{s} = (1-\delta)C + \delta T,\quad T_{ij} = \bar r\ (i\ne j),\ T_{ii}=1,\qquad S^s = C^s \odot vv^\top ,\quad \delta = 0.25 .
$$

Variances are unchanged.

### 4.5 Illiquidity uplift

With $f_i = 1+\upsilon$ for $i \in \mathcal{I}\cup\mathcal{S}$ and $f_i=1$ otherwise ($\upsilon$ =
`uplift` = 0.15): $\Sigma \leftarrow \Sigma \odot ff^\top$. Volatilities are multiplied by $f_i$
and correlations are unchanged.

### 4.6 PSD repair

$A = (S+S^\top)/2 = Q\Lambda Q^\top$, and $\Lambda \leftarrow \max(\Lambda, 10^{-10})$. The
result is symmetrised again. The final $\Sigma$ is aligned to the universe order (`align_to`
raises an error on a label mismatch).

---

## 5. Point-in-time decision state

*engine/state.StateBuilder.build(t)* returns a `DecisionState` with fields `asof`, `universe`,
`risk`, `goals`, `optimizer`, `pacing`, `cma_source` and `policy`:

1. **Policy in force.** The last value on or before `asof` of each `policy_history` key is applied
   (asset bounds, group limits, goal keys, turnover).
2. **Goals → optimiser targets.** $R^*$, $Y^*(\text{asof})$ and the band.
3. **CMAs.** The vintage from `cma_history` effective at `asof` (`cma_source = vintage`).
   Otherwise, if `cma_blend` $=\beta>0$, a blend with the trailing realised return over at most
   `cma_trailing_years` (`blend`):

   $$
   \mu_i \leftarrow (1-\beta)\mu_i + \beta\Big[\prod (1+r_{i,m})\Big]^{12/n}-1 .
   $$

   Otherwise the values in `assets.csv` are used (`static`).
4. **Pacing sync.** `growth_rate` $= \mu_i$; `yield_rate` $= \min(y_i, \mu_i)$, unless the yield
   was stated explicitly.
5. **Risk.** `RiskModel.estimate(universe, common_returns, t)`.

---

## 6. Goal-ordered optimiser

*optimizer/goal_optimizer.GoalOptimizer(universe, Σ, config, floors, w_prev, mode)*

### 6.1 Why a grid

$\sigma(w) \le \sigma_{hi}$ is a second-order-cone constraint, but $\sigma(w)\ge\sigma_{lo}$ is
reverse-convex. The band is therefore enforced by solving at fixed targets
$\bar\sigma_g \in \operatorname{linspace}(\sigma_{lo}, \sigma_{hi}, \texttt{vol\_steps})$ and then
**selecting** among the solutions.

### 6.2 The three-stage lexicographic solve at one grid point

The common constraints are $\mathcal{C}(w)$ from section 3.2, plus
$\lVert L^\top w\rVert_2 \le \bar\sigma_g$ and $y^\top w + s \ge Y^*$, $s \ge 0$ (plus
$\mu^\top w \ge R^*$ if `hard_return_floor`).

| Stage | Problem |
|---|---|
| A | $\max_{w} \mu^\top w$ → $r^{A}$ |
| B | $\min_{w,s} s$ subject to $\mu^\top w \ge r_{floor}$, where $r_{floor} = R^*$ if $r^A \ge R^* - \epsilon_r$, else $r^A - \epsilon_r$ |
| C | $\max_{w} \mu^\top w$ subject to $\mu^\top w \ge r_{floor}$ and $s \le s^B + \epsilon_r$ |

Here $\epsilon_r$ = `return_tol` = $10^{-5}$. Stage C recovers return left on the table by stage
B. If C fails, B's solution is used; if B fails, A's. Each stage is compiled once as a
parametrised cvxpy problem (parameters $\bar\sigma$, $r_{floor}$, $s_{cap}$) and re-solved across
the grid. With `lexicographic = False`, a single problem
$\max \mu^\top w - \lambda_{inc}\, s$ is solved instead.

Each solution records:

* $\mu^\top w$, $y^\top w$ and $\sigma(w)$;
* the shortfall $\max(0, Y^* - y^\top w)$;
* turnover;
* `band_relaxed` ($\bar\sigma > \sigma_{hi}$);
* `risk_binding` ($\sigma \ge \bar\sigma - 10^{-4}$).

Weights below $10^{-8}$ in absolute value are set to 0.

### 6.3 Scan extension

If no point in the band reaches $R^*$ and `vol_relax_max` $>\sigma_{hi}$, the grid is extended
above $\sigma_{hi}$ in band-sized steps up to `vol_relax_max`. It stops at the first point that
meets the return goal.

### 6.4 Selection (*GoalOptimizer.select*)

With $gap(s) = \max(0, \sigma_{lo} - \sigma - 10^{-4})$ and tolerance $10^{-5}$:

1. in-band points that meet return **and** income → $\arg\min (gap, \sigma)$: *"return+income met
   in band → min risk"*;
2. otherwise in-band points that meet return → $\arg\min(\text{shortfall}, gap, \sigma)$;
3. otherwise relaxed points that meet return → $\arg\min(\mathbb{1}[\text{shortfall} > tol], \sigma)$;
4. otherwise $\arg\max(\mu^\top w, -\text{shortfall})$ in the band: *"targets unreachable → max
   return, then min income gap"*.

A selected point with $gap>0$ is labelled *"frontier flat"*.

### 6.5 Fallbacks (`run(allow_fallback=True)`)

If the scan is empty:

1. **The band yields.** $\sigma_{min} = \min \lVert L^\top w\rVert$ subject to $\mathcal{C}$. Scan
   $\operatorname{linspace}(\sigma_{min}+10^{-6}, \max(0.4, 2\sigma_{hi}, \sigma_{min}+0.01),
   \max(2\cdot\texttt{vol\_steps}, 21))$ and take the lowest $\sigma$ that meets
   return + income, else return, else the minimum-volatility point.
2. **Turnover relaxes.** Only if the constraint set itself is infeasible: rerun with
   $\tau \times 1.5, 2, 3$, then no cap. The relaxation is stamped
   `[turnover cap relaxed to …]`.

### 6.6 Clamping ladder floors (*clamp_ladder_floors*)

Buckets are processed in order $k=1..N$ (the earliest money first). Given the floors already
granted, $\bar f_k = \max\sum_{i\in B_k} w_i$ subject to the policy constraints. If
$f_k > \bar f_k$, the floor becomes $\bar f_k - 10^{-6}$ and a `LadderShortfall(k, f_k, granted)`
is recorded.

### 6.7 Diagnostics (*diagnose*)

Under the same constraint set, these values are computed:

* $\max\mu^\top w$ and $\max y^\top w$;
* $\min\sigma$;
* $\max\mu^\top w$ and $\max y^\top w$ subject to $\sigma\le\sigma_{hi}$;
* $\sigma_{min}(R^*)$: the minimum volatility needed to reach $R^*$, found by 25 bisection steps
  on $\bar\sigma\in[\sigma_{min},\sigma_{hi}]$ using $\max \mu^\top w$ subject to $\sigma \le
  \bar\sigma$;
* feasibility flags for the return and income goals.

---

## 7. Private markets: pacing model, private book, funding ladder

### 7.1 Takahashi–Alexander vintage model (annual clock, *privates/pacing.py*)

A vintage has NAV $N$, uncalled amount $U$ and age $a$. Parameters: life $L$, commitment period
$CP$, contribution rates $RC(a)$ (applied to the *remaining* uncalled amount; the last rate must
be 1), bow factor $B$, growth $g$ ($=\mu$), yield $Y$, drawdown ratio $DR$. A commitment $c$
enters as $U_0 = c\cdot DR$. Each year:

$$
\begin{aligned}
a &\leftarrow a+1,\qquad C = RC(a)\,U,\qquad U \leftarrow U - C\ \ (U\leftarrow 0 \text{ if } a\ge CP)\\
RD(a) &= \min\!\big(1, (a/L)^{B}\big),\qquad \tilde N = N(1+g)\\
D &= \min\!\big(RD(a)\,\tilde N + Y N,\ \tilde N + C\big),\qquad N \leftarrow \tilde N + C - D .
\end{aligned}
$$

* `single_vintage_profile` is the path of a commitment of 1.
* `implied_irr` is the IRR of $(D - C)$ plus the terminal NAV, computed from the polynomial
  roots. It equals $g$ by construction (the consistency invariant).
* `simulate_pacing` aggregates over vintages (new commitments plus seeded legacy vintages).
* Legacy seeding scales the steady-state NAV and uncalled profiles separately to $(N_0, U_0)$.

`PacingParams.problems` reports the following calibration problems:

* rates outside $(0,1]$;
* $\prod (1-RC) > 10^{-6}$ (part of the commitment is never called);
* more rates than $CP$;
* $CP \ge L$;
* $Y > g$;
* $DR \notin [0.5, 1]$.

### 7.2 Monthly cohort book (*privates/book.py*)

The engine (backtest, projection, cash-flow analytics) uses a **monthly** clock. A cohort is a
commitment, a paid call (NAV, $U=0$) or a slice of legacy NAV. At each anniversary
($a_m \bmod 12 = 0$, contract year $j = \lfloor a_m/12\rfloor + 1$):

$$
C_j = RC(j)\,U,\ \ U\leftarrow U - C_j\ (\,U\leftarrow 0 \text{ if } j\ge CP),\quad
c_m = C_j/12,\quad
\pi_j = \min(1, RD(j)+Y),\quad
m_j = 1 - (1-\pi_j)^{1/12}.
$$

Each month, with a market return $r$ (realised in the backtest; $g_m = (1+g)^{1/12}-1$ in
projections):

$$
N \leftarrow N(1+r) + c_m,\qquad
D = \begin{cases} N\,m_j & j < L\\ N/(12 - a_m \bmod 12) & j \ge L \text{ (linear run-off)}\end{cases},\qquad
N \leftarrow N - D,\quad a_m \leftarrow a_m+1 .
$$

The monthly rate $m_j$ distributes exactly the annual fraction $\pi_j$ over 12 months
($1-(1-m_j)^{12} = \pi_j$).

### 7.3 Portfolio state and the cash waterfall (*engine/portfolio_state.py*)

The state holds values $V_i$ in currency, cash $K$ and a book per closed-end sleeve (with
$V_i \equiv N_i$).

* **`fund(need)`**, a call or withdrawal, is paid in this order:
  1. cash $K$;
  2. the ladder assets in bucket order;
  3. a **forced pro-rata sale** of the other *fully liquid* assets (gated sleeves are excluded),
     with cost $\sum_i x_i(\text{tx}_i + \text{extra})/10^4$;
  4. anything left is an **overdraft**, $K<0$, recorded as a shortfall. Value is never created.
* **`trade_to(w^*)`** trades only liquid assets to $w^*\cdot PV$. Costs
  $\sum \lvert\Delta V_i\rvert\,\text{tx}_i/10^4$ are charged pro rata to the liquid book.
  One-way turnover is $\tfrac12\sum_{i\in\mathcal{L}}\lvert\Delta V_i\rvert/PV$.
* **Sweep asset.** The first ladder asset, or else the first *fully liquid* asset. It receives
  income, distributions and inflows.

### 7.4 Funding ladder (*privates/ladder.py*)

The time to call is measured in calendar years:
$t = (Y_d - Y_a) + (M_d - M_a)/12 + (D_d - D_a)/365.25$.

With the edges $e_1<\dots<e_{N-1}$ (default 2, 3 years), the bucket of a call is
$k = 1 + \#\{e_j \le t\}$. For calls strictly after the as-of date:

$$
f_k = \text{cover}\cdot\frac{\sum_{\text{calls in } k}\text{amount}}{PV},\qquad
\text{floor:}\ \sum_{i\in B_k} w_i \ge f_k .
$$

Buckets do not substitute for each other. Floors are clamped as in section 6.6. The ladder table
reports, per bucket: need, held weight and cover = held / need.

---

## 8. Point-in-time allocation

*engine/allocation.PointInTimeAllocator.run(t)*

| Portfolio | Problem |
|---|---|
| **current** | $w^0$ from `holdings.csv` |
| **strategic** | `GoalOptimizer(universe, Σ, cfg, floors=∅, w_prev=w⁰, STRATEGIC).run()`: privates free within policy, no trading rules |
| **today** | universe with privates pinned (section 3.4) and group floors clamped (section 3.5); ladder floors from `commitment_future`; `REBALANCE` mode from $w^0$; `run(allow_fallback=True)` |

**Commitment gap** per closed-end sleeve:

$$
\text{gap}_i = w^{SAA}_i - w^0_i - \frac{\sum \text{scheduled calls}_i}{PV}
$$

A positive gap means "commit more".

`summary()` reports $\mu^\top w$, $y^\top w$, $\sigma(w)$, the illiquid share,
$\tfrac12\lVert w-w^0\rVert_1$, the targets and the selection rules.

---

## 9. Transition projection at capital-call dates

*engine/projection.CallDateProjector.run(schedule, planning_schedule, ladder_cover, distributions)*

The market state is frozen at `asof` (CMAs, Σ, goals); only the book evolves. The inputs are:

* `schedule`: the calls actually **paid**;
* `planning_schedule`: what the ladder is **sized on**. It defaults to `schedule`; a different one
  models a surprise.

1. **$t_0$.** `PortfolioState.from_weights(w⁰, PV₀)`. The ladder requirement is computed from the
   planning schedule. The solve runs in `REBALANCE` mode with privates pinned, then `trade_to`.
2. **For each call date $d$, in order:**
   1. For each month since the previous date: liquid values grow as $V_i \leftarrow V_i(1+g_{m,i})$
      with $g_{m,i}=(1+\mu_i)^{1/12}-1$; books step with $g_m$ of the sleeve; distributions
      accumulate (or are suppressed when `distributions=False`).
   2. Distributions are swept into the sweep asset.
   3. The calls due are paid with `fund`; any shortfall is recorded as an overdraft. Each call
      becomes a new cohort (`call_now`).
   4. The ladder is re-sized from the planning calls after $d$. The solve runs in `STRATEGIC`
      mode with privates pinned and those floors.
   5. If that solve is infeasible because pinned privates plus the liquid lower bounds exceed
      100%, the liquid lower bounds are scaled by $k = (1 - \text{priv} - 10^{-4})/\sum
      \ell_{liq}$ (reported) and the solve is retried.
   6. `trade_to`, then record the weights, $\mu^\top w$, $y^\top w$, $\sigma$, the book (NAV per
      sleeve, calls, distributions, remaining schedule) and the ladder table.

`funding_table()` pivots the fund plan (sources −, uses +) by asset × date.

---

## 10. Walk-forward backtest

*engine/backtest.WalkForwardBacktest.run()*, over months $t = $ burn-in $..T-1$, with
$m = t -$ burn-in:

1. **State.** At reviews, rebalances and every 12 months: `build(t)`. Book parameters are
   re-synced.
2. **Mark.** For liquid $i$: income $V_i y_i/12$ accrues to cash and
   $V_i \leftarrow V_i(1 + r_{i,t} - y_i/12)$. Books step with the **realised** reported return.
   Historical calls due in $(t-1, t]$ are replayed (`call_now`).
3. **Calls.** `fund(calls, ladder order, forced_sale_extra_bps)`. Forced sales and unfunded calls
   are recorded as events. Cash is swept.
4. **External flows.** Inflows go to the sweep asset; outflows go through `fund`.
5. **Review** (every `review_months`, $m>0$, process run only). Strategic solve. If no historical
   calls exist, the top-up commitment is

   $$
   \text{commit}_i = \frac{\max\big(0, w^*_i - (N_i + U_i)/PV\big)\,PV}{DR_i} .
   $$

6. **Ladder floors.** From the remaining replayed calls, or else from the book's own projected
   calls (`_projected_floors`: pending instalments plus a TA projection over the ladder horizon).
7. **Rebalance** (every `rebalance_months`):
   * **process:** a pinned, clamped `REBALANCE` solve with the floors;
   * **fixed-weight benchmarks:** private weights are kept and liquid weights rescaled to
     $1-\sum_{\mathcal{I}} w$;
   * then `trade_to`.
8. **Record.** $PV$; return $(PV_t - \text{flow}_t)/PV_{t-1}-1$; income; calls; distributions;
   illiquid and ladder shares; turnover; costs; forced sales; overdraft; the targets in force.

`BacktestSuite` runs `process`, `no_ladder`, `hold_current` and `60_40` (60% first equity sleeve,
40% first rates or credit sleeve, no privates).

**Metrics** (*analytics/performance.backtest_metrics*):

* annualised geometric return, volatility, return/vol and maximum drawdown;
* realised income per year;
* the share of months with the rolling 36-month volatility inside the band in force;
* the share of rolling `goal_return_window_months` windows whose annualised return is at least the
  target in force;
* turnover and costs per year, and the illiquid and ladder extremes;
* counts of forced sales, unfunded calls, group breaches, infeasible solves and underfunded
  ladders.

---

## 11. Analytics

### 11.1 Risk (*analytics/risk.py, performance.py*)

$$
MCR = \frac{\Sigma w}{\sigma(w)},\quad CCR_i = w_i MCR_i\ \ (\textstyle\sum CCR = \sigma),\quad PCR_i = \frac{CCR_i}{\sigma}\ \ (\textstyle\sum PCR = 1)
$$

* Diversification ratio $DR = \sum_i w_i\sqrt{\Sigma_{ii}}/\sigma(w)$.
* Effective $N$ $=1/\sum_i s_i^2$ for normalised shares $s$ (of weights, or of $\lvert PCR\rvert$).
* Eigen diagnostics on the correlation matrix: the largest eigen share, the condition number and
  $(\sum\lambda)^2/\sum\lambda^2$.
* Aggregation by asset class (sums to 100%) or by policy group (groups may overlap).

### 11.2 Stress (*analytics/stress.py*)

**Factor stress.** Asset returns and factor returns $\Delta F/F$ are joined, aggregated to $k$
months and desmoothed. The sample covariance of this panel is computed (at least 20 observations).
For a shock vector $s$ on factors $K$:

$$
m = \Sigma_{aK}\,\Sigma_{KK}^{+}\, s,\qquad \text{P\&L}_p = w_p^\top m .
$$

**Scenario shocks.** Explicit values are used as given. A window scenario uses
$s_c = F_c(e)/F_c(s)-1$ from the factor levels, and explicit values override it.
$\lvert s\rvert > 1.5$ raises an error.

**Historical episodes.** For portfolio returns $r_p = R\,w_p$ and a window $(s, e]$: cumulative
return $\prod(1+r_p)-1$ and the anchored maximum drawdown. The worst non-overlapping 12-month
windows of a reference portfolio are added.

### 11.3 Frontiers (*analytics/frontier.py*)

$\max \mu^\top w$ subject to $\mathcal{C}(w)$ and $\lVert L^\top w\rVert\le \bar\sigma$, for
$\bar\sigma \in \operatorname{linspace}(0.03, 0.15, 25)$. Points where the realised $\sigma$ falls
(unused risk budget) are dropped. There are three universes:

* **policy**;
* **no privates:** $u_i = 0$ for $i\in\mathcal{I}$, with purely-private group floors set to 0;
* **unconstrained:** bounds $[0,1]$, no groups.

`frontier_gaps` interpolates each frontier at given volatilities:

* *privates add* = policy − no privates;
* *policy cost* = unconstrained − policy.

### 11.4 Private-markets cash flows and performance (*analytics/private_markets.py*)

Cohorts are:

* the legacy NAV per sleeve, starting at `asof`;
* one cohort per scheduled call, starting at its call date.

Each steps monthly (section 7.2) with $g_m$ until its NAV is about 0 or the horizon ends (the
largest fund life + 2 years after the last call).

**Performance per group.** Paid-in $PI$ = calls + the legacy NAV at its start (legacy counts as
contributed at NAV). Flows are $D - PI$ plus the terminal NAV.

$$
DPI = \frac{\sum D}{PI},\quad RVPI = \frac{NAV_{end}}{PI},\quad TVPI = DPI + RVPI,\quad
\text{XIRR}: \sum_j \frac{CF_j}{(1+r)^{(t_j - t_0)/365}} = 0 .
$$

The XIRR is found by bisection on $[-0.99, 10]$, and is NaN if all flows have the same sign.

* **J-curve:** cumulative net cash flow by years since the call.
* **Effective vintages:** $1/\text{HHI}$ of year-end NAV by cohort.

### 11.5 Liquidity and income (*analytics/liquidity.py*)

Per portfolio $p$ (with its own date $d_p$ and value $PV_p$), the tiers are:

* ladder: $\sum_{B} w$;
* other liquid: $\sum_{\mathcal{L}} w$ − ladder − semi-liquid;
* semi-liquid: $\sum_{\mathcal{S}} w$;
* illiquid: $\sum_{\mathcal{I}} w$.

For $h=1,2,3$ years: calls$_h$ = cover · (calls in $(d_p, d_p+h]$)/$PV_p$, ladder cover =
ladder / calls$_h$, and liquid cover = (ladder + other liquid) / calls$_h$.

Income decomposition: $\sum_{i\in\text{class}} w_i y_i$, compared with $Y^*$.

### 11.6 Sensitivity (*analytics/sensitivity.py*)

One-at-a-time cases, each re-solved with the same optimiser:

* $\mu_{class} \pm 1\%$;
* $\Sigma\times 1.2$ and $\Sigma\times 0.8$;
* $R^*\pm0.5\%$ and $Y^*\pm0.5\%$;
* the band $\pm1\%$.

For each case: $\mu^\top w$, $y^\top w$, $\sigma$, turnover vs the base $\tfrac12\lVert
w-w^{base}\rVert_1$, the largest move and the rule. `binding_constraints` lists the bounds and
groups within $10^{-4}$ of a limit.

### 11.7 Wealth Monte Carlo (*analytics/wealth.py*)

The model is monthly joint lognormal: $\ln(1+r_m) \sim \mathcal N(\delta, \Sigma/12)$ with

$$
\delta_i = \frac{\ln(1+\mu_i) - \Sigma_{ii}/2}{12},\qquad r_m = \exp(\delta + L_{12} z) - 1,\quad L_{12}L_{12}^\top = \Sigma/12 .
$$

The expected annual simple return is therefore $\mu_i$. Wealth evolves as
$W_{m+1} = W_m(1 + w_m^\top r_m)$, with $w_m$ piecewise constant along a weight path (rebalanced
monthly). The output has:

* the percentiles 5, 25, 50, 75 and 95 and the mean, in nominal and real terms (deflated by
  $(1+\text{cpi})^t$);
* $P(W_t \ge W_0(1+R^*)^t)$, $P(W_t<W_0)$ and $P(W_t/(1+\text{cpi})^t < W_0)$.

### 11.8 Performance measures (*analytics/performance.py*)

| Measure | Formula |
|---|---|
| annualised return | $\big[\prod(1+r)\big]^{12/n} - 1$ |
| annualised volatility | $\mathrm{sd}(r)\sqrt{12}$ |
| Sharpe | $12\,\overline{(r-r_f)}\,/\,(\mathrm{sd}(r-r_f)\sqrt{12})$ |
| drawdown | $\text{cum}/\max(\text{cummax}, 1) - 1$, anchored at the starting capital |
| VaR / CVaR at 95% | $-q_{5\%}(r)$ and $-\mathbb E[r \mid r\le q_{5\%}]$ (at least 12 observations) |
| tracking error, information ratio, beta | the usual definitions on monthly differences |

---

## 12. Semi-liquid (evergreen / ELTIF) sleeves

An asset with `liquid = True` and `redemption_limit = ρ ∈ (0,1]` is **semi-liquid**
(`Asset.semi_liquid`). Its semantics, each implemented once:

| Aspect | Rule | Location |
|---|---|---|
| buying | unrestricted (subscription at NAV); `max_trade` still applies if set | `ConstraintSpec.build` |
| selling | $w^{prev}_i - w_i \le \rho\, w^{prev}_i$ per rebalance, audited as `redemption gates` | `ConstraintSpec.build/audit` |
| risk | illiquidity uplift as for closed-end sleeves (`fully_liquid_mask`) | `RiskModel.estimate` |
| smoothing | desmoothed if `appraisal_based` | section 4.2 |
| call funding | never part of a forced sale | `PortfolioState.fund` |
| sweep | never the sweep asset | `Universe.sweep_asset` |
| liquidity tier | `semi_liquid` column | `liquidity_profile` |
| validation | $\rho\notin(0,1]$ is an error; a gated ladder asset is an error; $\rho$ on a closed-end sleeve is a warning | `validate` |
| pacing, ladder | not needed: there are no commitments or calls | n/a |

**The gate over time.** With the gate binding every period, the holding left after $q$ periods
is $w_0(1-\rho)^q$. The cumulative cash raised is $w_0[1-(1-\rho)^q]$.

---

## 13. Case-study computations (notebooks)

The two notebooks `examples/Cases_study_SAA_ClosedEnd_PrivateMarkets.ipynb` and
`examples/Cases_study_SAA_SemiLiquid_PrivateMarkets.ipynb` add the following computations on top
of the library.

### 13.1 Commitment pacing plan (closed-end case)

**Target path.** For sleeve $s$ with strategic weight $w^{SAA}_s$, portfolio growth
$g = \mu^\top w^{SAA}$ and ramp-up $R = 5$ years:

$$
N^{tgt}_{s,t} = w^{SAA}_s\,PV_0\,(1+g)^t\min(1, t/R),\qquad t = 1..H_{fit}\ (H_{fit} = 15) .
$$

**Step 1, calls.** $A^{N}_{tj}$ and $A^{D}_{tj}$ are the NAV and the distributions in year $t$
from a call of 1 paid at the end of year $j$, simulated with the monthly book (section 7.2) at
$g_m$. Solve the non-negative least-squares problem

$$
c^* = \arg\min_{c\ge 0} \lVert A^N c - N^{tgt}\rVert^2 + \lVert \lambda \lVert A^N\rVert_{\max} \Delta c\rVert^2 ,\qquad \lambda = 0.3 ,
$$

where $\Delta$ is the first-difference operator. This is solved as stacked NNLS.

**Step 2, commitments.** $B_{tv}$ is the call in year $t$ from a commitment of 1 signed in year $v$
(vintage model, section 7.1). Solve

$$
k^* = \arg\min_{k\ge 0}\lVert Bk - c^*\rVert^2 + \lVert \lambda\lVert B\rVert_{\max}\Delta k\rVert^2 ,
$$

and round $k$ to USD 0.1m. The calls used downstream are $c = \operatorname{round}(Bk, 10^4)$.
The distributions are $A^D c$ and the NAV is $A^N c$. The first $H = 10$ years are reported.

**Ladder schedule on net calls.**
$\text{net}_{s,t} = \max(c_{s,t} - d_{s,t-1}, 0)$, rounded to USD 0.01m. The projector is run
with `schedule` = gross calls and `planning_schedule` = net calls.

**Stress runs.** Over the first 5 years of calls:

* `distributions=False`;
* calls shifted 1 year earlier, with the ladder still sized on the original plan.

### 13.2 ELTIF construction (semi-liquid case)

For a liquidity sleeve share $c$ (0.15), an extra fee $f$ (0.6% p.a.) and a gate $\rho$ (5% per
quarter):

$$
\mu^E = (1-c)\mu^{PM} + c\,\mu^{cash} - f,\qquad
y^E = (1-c)y^{PM} + c\,y^{cash} - f,\qquad
r^E_q = (1-c)\,r^{PM}_q + c\,r^{cash}_q - f/4 .
$$

The returns $r^E_q$ are computed on the private index's reporting dates and compounded into a
level series (`nav_frequency = Q`, appraisal-based). The ELTIF sleeve gets the closed-end sleeve's
bounds and group memberships. Ladder buckets and ladder settings are removed from the case.

**Quarterly transition.** Starting from $w^{(0)} = w^0$, solve repeatedly

$$
w^{(q+1)} = \text{today\_problem}(\text{state}, w_{prev} = w^{(q)}).\text{solve}()
$$

until $\tfrac12\lVert w^{(q+1)} - w^{(q)}\rVert_1 < 10^{-3}$. Each step respects the turnover cap
and the gates.

**Denominator effect.** A severe shock $3\times$ `Equities −10%` gives the true moves $m$
(section 11.2). The reported NAV moves are $m_i(1-\varphi_i)$ for appraisal-based sleeves. Then

$$
w^1 = \frac{w\odot(1+m^{rep})}{\sum w\odot(1+m^{rep})},\qquad
p_q = \frac{p_0(1-\rho)^q}{1 - p_0 + p_0(1-\rho)^q},
$$

where $p_0$ is the share of the private group after the shock and $p_q$ its share after $q$ gated
redemptions (the proceeds stay in the liquid book).

**Withdrawal.** A withdrawal $x$ paid from liquid assets raises the gated share to
$p/(1-x)$.

### 13.3 Common steps (both notebooks)

* The case inputs are written to a folder and loaded with `load_mandate`, so the data check sees
  exactly the files the model reads.
* The PE floor is a policy decision ($\ell_{PE} = 3\%$). Its cost is the difference in strategic
  $\mu^\top w$ at equal $\sigma$.

---

## 14. Input data check and fault injection

`examples/data_check.py`; the command line is `python examples/data_check.py <folder> [--stress]`.

### 14.1 `check_inputs(folder)`: four stages

1. **Inventory.** Required and optional files are listed with rows, columns and separator.
   Unknown `*.csv` names get a WARN with a close-match hint (`difflib`).
2. **Load and validate.** `load_mandate` + `validate`. Errors become FAIL rows, warnings WARN,
   notes INFO.
3. **Reconciliation.** Each raw value is compared with the loaded model:
   * every asset field (rates follow the `as_rate` rule);
   * **returns recomputed independently** from the raw levels (month-end last, flat carry,
     `pct_change`), requiring a maximum absolute difference of at most $10^{-10}$;
   * holdings totals and each position; undrawn totals;
   * group members and limits;
   * config values after the rate rule; pacing parameters (and growth = CMA);
   * schedules (row counts and totals);
   * the CMA, policy and flow histories.
4. **Plausibility.**
   * $\mu \in [-5\%, 30\%]$ and $y\in[0,20\%]$;
   * dates not at month-end, and several rows in one month;
   * per series:
     * blank months;
     * $\ge 3$ consecutive zero returns (stale feed);
     * reports off the NAV calendar;
     * outliers beyond $25\%\cdot\sqrt{k}$ per period or 8 robust $z$ ($z = (r -
       \text{med})/(1.4826\,\text{MAD})$);
     * **typo spikes**: an outlier followed by a reversal of comparable size;
   * series ending before the file;
   * unknown config keys;
   * factor overlap of at least 20 estimation periods;
   * scenario factors that exist; windows inside the history;
   * calls outside the history or before the as-of date;
   * incomplete CMA vintages.

The output is a `DataCheckReport`: `table`, `summary()`, `issues()`, `ok`, `raise_if_failed()`
and `text()`. The process exit code is 1 on any FAIL.

### 14.2 `stress_test_inputs(folder)`: fault injection

Each case copies the folder, applies one mutation and runs `check_inputs`.

* **Format variants** must produce an identical model, compared on the returns, every asset field
  (float tolerance $10^{-12}$), groups, holdings, settings, pacing, schedule and factors. The
  variants are:
  * German locale (`;`, decimal comma, dd.mm.yyyy);
  * percent strings;
  * shuffled rows and columns;
  * a BOM with CRLF line endings;
  * padded names;
  * tab separators.
* **Faults** must be *caught* (FAIL) or *flagged* (WARN or FAIL matching a keyword). There are 25
  of them, covering:
  * names: a typo in an asset name;
  * prices: a missing column, text in a number, a duplicate date, a negative level, returns
    instead of levels, a ×10 spike, a stale feed, gaps, a truncated series, off-calendar NAVs,
    two rows in one month;
  * limits and goals: lb > ub, an unknown group member, $\sigma_{lo}>\sigma_{hi}$;
  * pacing and calls: rates that do not reach 1, a call on a liquid asset, a negative call;
  * holdings: a negative value;
  * histories and scenarios: an unknown asset in the CMA history, an unknown policy key, an
    unknown stress factor;
  * units and names: a percent unit error, a config-key typo, a wrong file name.

---

## 15. Numerical settings and tolerances

| Item | Value |
|---|---|
| Solver | Clarabel via cvxpy (SCS as fallback) |
| Return give-back between stages | `return_tol` = $10^{-5}$ |
| Band membership | $\sigma \ge \sigma_{lo} - 10^{-4}$ |
| Weight clean-up | $\lvert w_i\rvert < 10^{-8} \to 0$ |
| Cholesky jitter | up to 6 attempts, starting from $10^{-12}\operatorname{tr}\Sigma$, ×10 each |
| PSD eigenvalue floor | $10^{-10}$ |
| Geltner cap | $\varphi \le 0.6$ |
| Pinning band | $\epsilon = 10^{-6}$; private no-trade band 1% |
| Grid | `vol_steps` points (example: 13); backtest `bt_vol_steps` |
| Ladder edges / cover | 2y, 3y / 1.0 |
| Monte Carlo | 10,000 paths by default (5,000 in the notebooks), seed 7 |

---

## 16. Verification

* `tests/test_core.py` covers the numerical building blocks:
  * parsing and the risk model;
  * the pacing invariants (IRR equals growth; full drawdown; monthly payout = the annual rate);
  * ladder buckets on anniversaries;
  * goal order and audit;
  * the waterfall (an overdraft creates no value);
  * Euler sums;
  * pinning with mixed groups;
  * semi-liquid gates, uplift, the forced-sale exclusion and the sweep.
* `tests/test_end_to_end.py` and `tests/test_package.py` cover the full study and the public API.
* `tests/test_data_check.py` covers:
  * reconciliation of the example data with zero FAIL;
  * every fault-injection case behaving as expected;
  * `redemption_limit` loading, validation and the gate in today's portfolio;
  * the liquidity tiers.
* Both case-study notebooks are executed end to end. Their narrative numbers are checked against
  the executed outputs.
