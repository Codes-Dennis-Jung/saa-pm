"""saa_pm - goal-based Strategic Asset Allocation with Private Markets.

Quick start::

    from saa_pm import SAAStudy
    results = SAAStudy("examples/data", "results").run()

Main building blocks (usable on their own):

    load_mandate / validate      read and check an input folder
    StateBuilder                 point-in-time inputs (CMAs, policy, covariance)
    GoalOptimizer                return > income > risk-band optimiser
    PointInTimeAllocator         strategic target + today's implementable portfolio
    CallDateProjector            allocation and funding at each capital-call date
    BacktestSuite                walk-forward backtest with benchmarks
"""

from .config import BacktestConfig, Goals, LadderConfig, OptimizerConfig, RiskModelConfig, Settings
from .engine.allocation import AllocationResult, PointInTimeAllocator
from .engine.backtest import BacktestResult, BacktestSuite, WalkForwardBacktest
from .engine.projection import CallDateProjector, ProjectionResult
from .engine.state import DecisionState, StateBuilder
from .io.loaders import MandateData, load_mandate
from .io.validation import ValidationReport, validate
from .optimizer.goal_optimizer import GoalOptimizer, InfeasibleError, OptimizationResult
from .privates.ladder import FundingLadder
from .privates.pacing import PacingParams
from .risk.covariance import RiskModel
from .study import SAAStudy, StudyResults
from .universe import Asset, GroupConstraint, Universe

__version__ = "1.1.0"

__all__ = [
    "SAAStudy",
    "StudyResults",
    "load_mandate",
    "validate",
    "MandateData",
    "ValidationReport",
    "Settings",
    "Goals",
    "OptimizerConfig",
    "RiskModelConfig",
    "LadderConfig",
    "BacktestConfig",
    "Universe",
    "Asset",
    "GroupConstraint",
    "PacingParams",
    "FundingLadder",
    "RiskModel",
    "StateBuilder",
    "DecisionState",
    "GoalOptimizer",
    "OptimizationResult",
    "InfeasibleError",
    "PointInTimeAllocator",
    "AllocationResult",
    "CallDateProjector",
    "ProjectionResult",
    "BacktestSuite",
    "WalkForwardBacktest",
    "BacktestResult",
    "__version__",
]
