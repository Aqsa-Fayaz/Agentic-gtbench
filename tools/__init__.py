"""Agent tool registry — LangChain-compatible tools bound to player agents."""

from tools.move_validator import MoveValidatorTool, validate_move
from tools.state_tracker import StateTrackerTool, describe_state
from tools.strategy_analyzer import StrategyAnalyzerTool
from tools.history_manager import HistoryManagerTool


def default_toolset() -> list:
    """Return the canonical set of tools bound to a PlayerAgent."""
    return [
        MoveValidatorTool(),
        StateTrackerTool(),
        StrategyAnalyzerTool(),
        HistoryManagerTool(),
    ]


__all__ = [
    "MoveValidatorTool",
    "StateTrackerTool",
    "StrategyAnalyzerTool",
    "HistoryManagerTool",
    "validate_move",
    "describe_state",
    "default_toolset",
]
