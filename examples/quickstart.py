"""Quick start: run the full study on the synthetic example mandate.

    python examples/quickstart.py

or from the command line:

    saa-pm run examples/data -o results
"""

from dataclasses import replace

from saa_pm import GoalOptimizer, PointInTimeAllocator, SAAStudy, StateBuilder, load_mandate

# 1. Everything in one call: allocation, forward plan, backtest, side analyses,
#    charts, Excel workbook and SUMMARY.md in ./results
results = SAAStudy("examples/data", "results").run()
print(results.allocation.summary().round(4))

# 2. Or use the building blocks directly
data = load_mandate("examples/data")
state = StateBuilder(data).build()  # point-in-time inputs at the last month
print(f"as of {state.asof:%Y-%m-%d}: return target {state.optimizer.target_return:.2%}")

alloc = PointInTimeAllocator(data).run()
print(alloc.weights().round(3))

# 3. What if the risk band were 1pp higher?
cfg = replace(state.optimizer, vol_lo=state.optimizer.vol_lo + 0.01, vol_hi=state.optimizer.vol_hi + 0.01)
res = GoalOptimizer(state.universe, state.sigma, cfg).run()
print(f"higher band: return {res.solution.exp_return:.2%}, vol {res.solution.vol:.2%}")
