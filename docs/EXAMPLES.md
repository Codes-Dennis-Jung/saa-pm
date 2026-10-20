# Examples: every building block, with real output

All output below was produced on the synthetic example mandate in `examples/data`. Run any snippet
from the repository root after `pip install -e .`.

**Contents**

1. [Load and validate inputs](#1-load-and-validate-inputs)
2. [Settings](#2-settings)
3. [Universe](#3-universe)
4. [Point-in-time state](#4-point-in-time-state)
5. [Risk model](#5-risk-model)
6. [Goal optimiser](#6-goal-optimiser)
7. [Feasibility diagnostics](#7-feasibility-diagnostics)
8. [What-if: a different risk band](#8-what-if-a-different-risk-band)
9. [Strategic target and today's portfolio](#9-strategic-target-and-todays-portfolio)
10. [Funding ladder](#10-funding-ladder)
11. [Transition plan at each call date](#11-transition-plan-at-each-call-date)
12. [Stress variants of the transition plan](#12-stress-variants-of-the-transition-plan)
13. [Private-markets pacing model](#13-private-markets-pacing-model)
14. [Walk-forward backtest](#14-walk-forward-backtest)
15. [Backtest suite with benchmarks](#15-backtest-suite-with-benchmarks)
16. [Side analyses on their own](#16-side-analyses-on-their-own)
17. [The whole study in one call](#17-the-whole-study-in-one-call)
18. [Inputs from DataFrames instead of files](#18-inputs-from-dataframes-instead-of-files)
19. [Logging](#19-logging)

---

## 1. Load and validate inputs

```python
from saa_pm import load_mandate, validate

data = load_mandate("examples/data")
print(data.universe, "|", data.returns.shape, "| as of", data.asof.date(),
      "| PV", f"{data.holdings.total:,.0f}")

rep = validate(data)
print("ok:", rep.ok, "| warnings:", len(rep.warnings))
print(rep.coverage[["months", "realised_return", "cma", "vol_reported",
                    "vol_desmoothed", "phi"]].tail(4).round(4))
```

```text
Universe(24 assets, 13 groups) | (189, 24) | as of 2026-09-30 | PV 100,000,000
ok: True | warnings: 14
            months  realised_return     cma  vol_reported  vol_desmoothed     phi
asset                                                                            
Gold           189           0.0438  0.0560        0.1350          0.1350  0.0000
Priv Infra     189           0.1036  0.1040        0.0631          0.0982  0.4142
Priv Eq        189           0.1047  0.1135        0.1629          0.2543  0.4126
Priv Debt      189           0.0785  0.0790        0.0413          0.0661  0.4309
```

The 14 warnings all say the same thing: the current (legacy) holdings sit outside some single-asset
policy bounds, for example `current weight of 'Cash' 40.00% is outside [1.00%, 28.00%]`. They do
not block anything; the optimiser moves the portfolio back inside the bounds.

`rep.errors`, `rep.warnings` and `rep.notes` are plain lists. `rep.raise_if_errors()` stops a script
on blocking problems, and `rep.text()` gives a printable report.

## 2. Settings

```python
s = data.settings
print(s.goals.target_return, s.goals.target_income, s.goals.vol_lo, s.goals.vol_hi,
      s.ladder.edges, s.risk.halflife)
print(s.describe().head())            # every setting in force, as a table
```

```text
0.0625 0.035 0.06 0.08 (2.0, 3.0) 36.0
  section            key   value
0   goals            cpi  0.0325
1   goals  return_spread    0.03
2   goals  target_income   0.035
3   goals         vol_lo    0.06
4   goals         vol_hi    0.08
```

Settings can also be built in code, with no file needed:

```python
from saa_pm import Settings
s = Settings.from_mapping({"cpi": 0.02, "return_spread": 0.035, "vol_hi": 0.10})
print(s.goals.target_return)          # 0.055
```

## 3. Universe

```python
u = data.universe
print(u.names[:4], u.ladder_map, u.illiquid_names)
print(u.mu[:3])
print(u["Priv Eq"])

u2 = u.updated({"US Nasdaq Eq": {"ub": 0.10}})    # new object; the original is unchanged
```

```text
['Cash', 'US STR', 'US Govt', 'UK Govt'] {1: ['Cash'], 2: ['US STR'], 3: ['US IG']} ['Priv Infra', 'Priv Eq', 'Priv Debt']
[0.037 0.038 0.047]
Asset(name='Priv Eq', group='Private Markets', exp_return=0.1135, income=0.022, liquid=False, lb=0.0, ub=0.15, appraisal_based=True, max_trade=None, ladder_bucket=None, tx_cost_bps=0.0, nav_frequency='Q', price_column='Priv Eq')
```

## 4. Point-in-time state

This is everything a decision at a given month is allowed to know: the policy in force, the
return-assumption vintage, and a covariance estimated from earlier data only.

```python
from saa_pm import StateBuilder

builder = StateBuilder(data)
state = builder.build()               # last closed month; builder.build(t) for any earlier month
print(state.asof.date(), state.cma_source,
      round(state.optimizer.target_return, 4), round(state.optimizer.target_income, 4))
print(state.risk.vols.round(4).tail(4))
print({k: round(v, 3) for k, v in state.risk.phi.items()})   # Geltner smoothing removed per private sleeve
```

```text
2026-09-30 vintage 0.0625 0.035
Gold          0.1002
Priv Infra    0.1161
Priv Eq       0.2525
Priv Debt     0.0615
dtype: float64
{'Priv Infra': 0.403, 'Priv Eq': 0.337, 'Priv Debt': 0.347}
```

## 5. Risk model

```python
from saa_pm import RiskModel

est = RiskModel(data.settings.risk).estimate(data.universe, data.common_returns)
print(est.agg_months, est.n_obs, round(est.n_eff, 1))
print(est.correlation.loc[["US Nasdaq Eq", "Priv Eq"],
                          ["US Nasdaq Eq", "Priv Eq", "US Govt"]].round(2))
```

```text
3 39 28.0
              US Nasdaq Eq  Priv Eq  US Govt
US Nasdaq Eq          1.00     0.74     0.20
Priv Eq               0.74     1.00     0.17
```

The model is estimated on **quarterly** returns (`agg_months = 3`) because the private sleeves
report quarterly.

## 6. Goal optimiser

```python
import pandas as pd
from saa_pm import GoalOptimizer

res = GoalOptimizer(state.universe, state.sigma, state.optimizer).run()
sol = res.solution
print(f"return {sol.exp_return:.2%}  income {sol.income:.2%}  vol {sol.vol:.2%}  "
      f"rule: {sol.selection_rule}")
print(pd.Series(sol.weights, index=state.universe.names)[lambda x: x > 1e-4].round(3))
print(res.audit.to_string(index=False))
```

```text
return 7.22%  income 4.56%  vol 6.00%  rule: return+income met in band -> min risk
Cash            0.020
US Govt         0.250
UK Govt         0.044
EM HC           0.030
US IG           0.050
EU IG           0.020
US HY           0.020
US Dvd Eq       0.100
EU ex UK Eq     0.020
EU Dvd Eq       0.020
UK Eq           0.020
JP Eq           0.050
Pac ex JP Eq    0.010
EM Eq           0.030
Gold            0.030
Priv Infra      0.146
Priv Debt       0.140
dtype: float64
              check  pass      value
   budget sums to 1  True 0.99999997
single-asset bounds  True         ok
       group bounds  True         ok
     vol <= ceiling  True    6.0000%
  vol >= band floor  True    6.0000%
      income target  True    4.5556%
      return target  True    7.2250%
```

Other useful parts of the result:

- `res.scan`: every grid point.
- `res.fallback`: set if the band had to yield.
- `res.ladder_shortfalls`: ladder floors that policy could not hold.
- `res.audit_failures`: failed audit checks.

## 7. Feasibility diagnostics

These show what is attainable *before* any target is imposed.

```python
GoalOptimizer(state.universe, state.sigma, state.optimizer).diagnose()
```

Output (floats rounded to 4 decimals here):

```text
{'max_return': 0.0845, 'max_income': 0.0499, 'min_vol': 0.0443,
 'max_return_at_vol_hi': 0.0793, 'max_income_at_vol_hi': 0.0499,
 'min_vol_for_return_target': 0.0495,
 'return_target_feasible': True, 'income_target_feasible': True}
```

## 8. What-if: a different risk band

```python
from dataclasses import replace

cfg = replace(state.optimizer, vol_lo=0.07, vol_hi=0.10)
r2 = GoalOptimizer(state.universe, state.sigma, cfg).run().solution
print(f"return {r2.exp_return:.2%} vol {r2.vol:.2%} rule {r2.selection_rule}")
```

```text
return 7.65% vol 7.00% rule return+income met in band -> min risk
```

Any target, bound or group can be changed the same way, for example with
`state.universe.updated({...})` or `replace(cfg, target_income=0.035)`.

## 9. Strategic target and today's portfolio

```python
from saa_pm import PointInTimeAllocator

alloc = PointInTimeAllocator(data).run()
print(alloc.summary()[["exp_return", "income", "vol", "illiquid", "turnover_vs_current"]].round(4))
print(alloc.weights().round(3).head(6))
print(alloc.commitment_gap().round(4))
```

```text
           exp_return  income     vol  illiquid  turnover_vs_current
current        0.0576  0.0349  0.0444    0.0700               0.0000
strategic      0.0722  0.0456  0.0600    0.2865               0.6400
today          0.0626  0.0350  0.0667    0.0700               0.3192
         current  strategic  today
Cash        0.40      0.020  0.240
US STR      0.00      0.000  0.045
US Govt     0.36      0.250  0.210
UK Govt     0.00      0.044  0.000
JP Govt     0.00      0.000  0.000
EU Govt     0.00      0.000  0.000
            strategic_target  nav_today  scheduled_calls     gap
asset                                                           
Priv Infra            0.1465       0.00             0.08  0.0665
Priv Eq               0.0000       0.07             0.10 -0.1700
Priv Debt             0.1400       0.00             0.12  0.0200
```

`alloc.notes` lists everything that was relaxed or clamped. Here the 30% turnover cap could not be
held for today's portfolio, so it was relaxed (one-way turnover 31.9%):

```text
['today: return+income met in band -> min risk [turnover cap relaxed to 45%]',
 'today audit: turnover limit: 31.9224%']
```

`alloc.diagnostics` holds the feasibility envelope.

## 10. Funding ladder

```python
print(alloc.ladder_table.round(4).to_string(index=False))

# or directly
from saa_pm import FundingLadder
lad = FundingLadder(data.universe, data.settings.ladder)
req = lad.requirement(data.schedule, data.asof, data.holdings.total)
print(req.need.round(4))
print(req.detail.head(4).round(3).to_string(index=False))
```

```text
 bucket assets       window  need_pct_pv   need_ccy  held_pct_pv  cover
      1   Cash 0y <= t < 2y        0.240 24000000.0        0.240 1.0000
      2 US STR 2y <= t < 3y        0.045  4500000.0        0.045 1.0000
      3  US IG      t >= 3y        0.015  1500000.0        0.040 2.6667
[0.24  0.045 0.015]
      date      asset    amount  years_to_call  bucket  pct_of_pv
2027-03-31  Priv Debt 6000000.0          0.503       1      0.060
2027-03-31    Priv Eq 2500000.0          0.503       1      0.025
2027-03-31 Priv Infra 3200000.0          0.503       1      0.032
2028-03-31  Priv Debt 6000000.0          1.503       1      0.060
```

## 11. Transition plan at each call date

```python
from saa_pm import CallDateProjector

proj = CallDateProjector(data).run()
print(proj.allocations[["exp_return", "income", "vol"]].round(4))
print(proj.private_book[["portfolio_value", "calls_paid", "distributions"]].round(0))
print(proj.funding_table().round(2).head(6))          # USD m: sources (-) and uses (+)
```

```text
            exp_return  income     vol
date                                  
2026-09-30      0.0626  0.0350  0.0667
2027-03-31      0.0645  0.0378  0.0650
2028-03-31      0.0689  0.0387  0.0717
2029-03-31      0.0689  0.0379  0.0733
2030-03-31      0.0665  0.0367  0.0717
            portfolio_value  calls_paid  distributions
date                                                  
2027-03-31      103008104.0  11700000.0      1071439.0
2028-03-31      109505176.0  12300000.0      3218314.0
2029-03-31      116830343.0   4500000.0      5166292.0
2030-03-31      124601575.0   1500000.0      6343487.0
date     2026-09-30  2027-03-31  2028-03-31  2029-03-31  2030-03-31
asset                                                              
Cash         -16.01      -12.10       -5.82        3.55        7.70
US STR         4.50       -0.08       -3.16       -1.55        0.00
US Govt      -14.97       -7.38       -3.99       -1.59       -3.11
UK Govt        0.00        7.84        0.53        0.15        0.16
JP Govt        0.00        4.94        3.63        0.24        0.25
EM HC          3.00        0.00        0.01        0.03        0.03
```

## 12. Stress variants of the transition plan

```python
# distributions delayed: private funds pay nothing out
p2 = CallDateProjector(data).run(distributions=False)
print(p2.allocations[["exp_return", "vol"]].round(4).tail(2))
print([n for n in p2.notes if not n.startswith("today")][:2])

# calls arrive a year EARLIER than the ladder was sized for (a surprise)
early = data.schedule.assign(date=data.schedule["date"] - pd.DateOffset(years=1))
p3 = CallDateProjector(data).run(schedule=early, planning_schedule=data.schedule)

# pre-fund 120% of every call
p4 = CallDateProjector(data).run(ladder_cover=1.2)
```

```text
            exp_return     vol
date                          
2029-03-31      0.0717  0.0851
2030-03-31      0.0732  0.0893
["2029-03-31: private NAV breaches 'Private Markets' (34.92% > 32.00%)", '2029-03-31: band infeasible -> lowest vol with return+income met (8.51%)']
```

Without distributions, the private book grows past its 32% group cap by 2029 (34.9%, and 37.0% by
2030). The cap is widened for those dates and the breach is reported. With that much illiquid NAV
the 6-8% risk band can no longer be met, so the solve falls back to the lowest volatility that
still meets the return and income goals (8.5% in 2029, 8.9% in 2030), and that fallback is reported
too.

## 13. Private-markets pacing model

```python
from saa_pm.privates.pacing import single_vintage_profile, implied_irr

p = state.pacing["Priv Eq"]
print(single_vintage_profile(p).head(6).round(3))     # a commitment of 1.0, by fund age
print("IRR", round(implied_irr(p), 4), "growth", p.growth_rate)
```

```text
      call  distribution    nav  uncalled
age
1    0.225         0.000  0.225     0.675
2    0.338         0.008  0.579     0.338
3    0.253         0.035  0.859     0.084
4    0.084         0.092  0.942     0.000
5    0.000         0.170  0.872     0.000
6    0.000         0.250  0.715     0.000
IRR 0.1062 growth 0.1062
```

The vintage IRR equals the return assumption, so the pacing model and the optimiser always agree.

## 14. Walk-forward backtest

```python
from saa_pm import WalkForwardBacktest

bt = WalkForwardBacktest(data).run()
print(bt.monthly[["pv", "ret", "income", "illiquid", "ladder_held", "turnover"]].tail(3).round(4))
print(bt.metrics(0.06, 0.08).round(4).head(8))
```

```text
                      pv     ret  income  illiquid  ladder_held  turnover
date                                                                     
2026-07-31  1.737187e+08 -0.0022  0.0028    0.0946       0.1954    0.0358
2026-08-31  1.734600e+08 -0.0015  0.0028    0.0927       0.1992    0.0000
2026-09-30  1.715861e+08 -0.0108  0.0028    0.0866       0.2054    0.0000
ann_return                          0.0448
excess_vs_target                   -0.0177
ann_vol                             0.0586
return_per_vol                      0.7643
max_drawdown                       -0.0871
realised_income                     0.0325
share_36m_vol_in_band               0.1132
share_60m_windows_hitting_target    0.3293
Name: process, dtype: float64
```

Other parts of the result:

- `bt.weights`: monthly weights.
- `bt.reviews`: each annual review, with the return-assumption source and the targets in force.
- `bt.events`: forced sales, unfunded calls, relaxed solves.

Variants:

```python
WalkForwardBacktest(data, label="no_ladder", use_ladder=False).run()
WalkForwardBacktest(data, label="fixed", fixed_weights=alloc.strategic.weights).run()
```

## 15. Backtest suite with benchmarks

```python
from saa_pm import BacktestSuite

suite = BacktestSuite(data)
runs = suite.run()                    # process, no_ladder, hold_current, 60_40
print(suite.metrics(runs).round(4).loc[["ann_return", "ann_vol", "max_drawdown",
                                        "realised_income", "forced_sales"]])
```

```text
                 process  no_ladder  hold_current   60_40
ann_return        0.0448     0.0445        0.0505  0.0459
ann_vol           0.0586     0.0603        0.0336  0.0900
max_drawdown     -0.0871    -0.1070       -0.0391 -0.1487
realised_income   0.0325     0.0316        0.0296  0.0294
forced_sales      0.0000    22.0000        0.0000  1.0000
```

## 16. Side analyses on their own

Every analysis is a plain function that takes weights, a universe and a covariance:

```python
from saa_pm.analytics import risk, stress, frontier, liquidity, sensitivity, private_markets
from saa_pm.analytics.wealth import simulate_wealth

ports = {"current": alloc.current, "strategic": alloc.strategic.weights,
         "today": alloc.today.weights}
u, S = state.universe, state.sigma

risk.concentration(ports, S, u)                                   # effective N, diversification
risk.attribution_by(risk.attribution(ports, S, u), u)             # risk share by asset class
stress.FactorStress(u, data.common_returns, data.stress_factors).run(
    stress.scenario_shocks(data.stress_scenarios, data.stress_factors), ports)[0]
stress.historical_episodes(data.common_returns[u.names], ports)
frontier.frontier_gaps(frontier.reference_frontiers(u, S), [0.06, 0.08])
sensitivity.sensitivity(u, S, state.optimizer)[0]
sensitivity.binding_constraints(alloc.strategic.weights, u)
cf = private_markets.project_cashflows(state.pacing, data.schedule, state.asof,
                                       {"Priv Eq": 5e6})
private_markets.performance(cf, ("asset",))
simulate_wealth(u.mu, S.values, alloc.today.weights, years=10, n_paths=5000,
                start_value=100e6, cpi=0.0325, target_return=0.0625).probabilities
```

The outputs below are rounded to 4 decimals.

**Example: `frontier.frontier_gaps(...)`.** The value of private markets at equal risk:

```text
      policy  no_privates  unconstrained  privates_add  policy_cost
vol
0.06  0.0722       0.0641         0.0876        0.0082       0.0153
0.08  0.0793       0.0717         0.0937        0.0076       0.0145
```

**Example: `stress.FactorStress(...).run(...)[0]`.**

```text
                                  current  strategic   today
scenario
Equities -10%                     -0.0257    -0.0331 -0.0409
Equities +10%                      0.0257     0.0331  0.0409
EUR -10% vs USD                    0.0257     0.0349  0.0397
EUR +10% vs USD                   -0.0257    -0.0349 -0.0397
Oil price shock +50%               0.0016     0.0131  0.0109
Rates shock (govt bonds -8%)      -0.0660    -0.0655 -0.0743
Credit spread widening            -0.0409    -0.0517 -0.0585
Stagflation                       -0.0644    -0.0655 -0.0803
Historical episode 1 (simulated)  -0.0473    -0.0579 -0.0662
Historical episode 2 (simulated)  -0.0361    -0.0402 -0.0500
Historical episode 3 (simulated)  -0.0094    -0.0237 -0.0289
```

**Example: `private_markets.performance(cf, ("asset",))`.**

```text
        asset     paid_in  distributions  final_nav     dpi  rvpi    tvpi     irr
0   Priv Debt  12000000.0     14715652.0        0.0  1.2263   0.0  1.2263  0.0854
1     Priv Eq  15000000.0     24725442.0        0.0  1.6484   0.0  1.6484  0.1061
2  Priv Infra   8000000.0     12274716.0        0.0  1.5343   0.0  1.5343  0.0984
```

## 17. The whole study in one call

```python
from saa_pm import SAAStudy

res = SAAStudy("examples/data", "results").run()      # or .run(steps=("allocation", "analyses"))
sorted(res.tables)[:6]
```

```text
['frontier/gaps', 'frontier/no_privates', 'frontier/policy', 'frontier/unconstrained',
 'income/decomposition', 'liquidity/profile']
```

`res` holds the following, and the same content is written to `results/` as CSV, Excel and charts:

- `res.allocation`
- `res.projection`
- `res.backtests`
- `res.backtest_metrics`
- `res.tables` (every side analysis)

## 18. Inputs from DataFrames instead of files

Every frame has the columns of the CSV file of the same name. `prices` is indexed by date, and
`config` is a plain dict. The frames go through exactly the same parser and checks as files do.

```python
import pandas as pd
from saa_pm import MandateData

def rd(name, **kw):
    return pd.read_csv(f"examples/data/{name}.csv", **kw)

data = MandateData.from_frames(
    assets=rd("assets"),
    prices=rd("prices", index_col=0, parse_dates=True),
    holdings=rd("holdings"),
    config={"cpi": 0.0325, "return_spread": 0.03, "target_income": 0.035,
            "vol_lo": 0.06, "vol_hi": 0.08},
    groups=rd("groups"), pacing=rd("pacing"), commitment_future=rd("commitment_future"),
)
print(data.folder, data.universe, f"{data.holdings.total:,.0f}")
```

```text
<in-memory> Universe(24 assets, 13 groups) 100,000,000
```

## 19. Logging

`saa_pm` logs to the standard `saa_pm` logger and is silent unless you ask:

```python
import logging
logging.basicConfig(level=logging.INFO)
SAAStudy("examples/data", None, verbose=False).run(steps=("allocation",), write=False)
```

```text
INFO:saa_pm:Loaded 24 assets, 189 months (2011-01 to 2026-09), portfolio 100,000,000
WARNING:saa_pm:  ! current weight of 'Cash' 40.00% is outside [1.00%, 28.00%]
WARNING:saa_pm:  ! current weight of 'US Govt' 36.00% is outside [3.00%, 25.00%]
...                                   (12 more validation warnings, as in section 1)
INFO:saa_pm:  allocation                      2.0s
```

`SAAStudy(..., verbose=True)` (the default) prints the same messages without the `INFO:` prefix.
