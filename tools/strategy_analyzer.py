"""
Strategy Analyzer Tool
Analyzes opponent's move history to infer their behavioral strategy.
Enables opponent modeling — a key component of strategic reasoning.
"""

import json
from collections import Counter
from langchain.tools import BaseTool
from pydantic import BaseModel, Field


class StrategyAnalyzerInput(BaseModel):
    move_history: list = Field(description="List of past moves/actions by opponent")
    game_name: str = Field(description="Name of the game being played")
    player_perspective: str = Field(description="'player_a' or 'player_b' (your perspective)")


class StrategyAnalyzerTool(BaseTool):
    """
    Analyzes opponent move history to infer their strategy pattern.
    Particularly useful in Prisoner's Dilemma and repeated games.
    """
    name: str = "analyze_opponent_strategy"
    description: str = (
        "Analyze the opponent's past moves to infer their strategy. "
        "Helps you predict their next action and counter-strategize. "
        "Input: move_history (list), game_name, player_perspective. "
        "Output: {inferred_strategy, confidence, recommendation}"
    )
    args_schema: type = StrategyAnalyzerInput

    def _run(self, move_history: list, game_name: str, player_perspective: str) -> str:
        if game_name == "prisoners_dilemma":
            return json.dumps(self._analyze_pd(move_history, player_perspective))
        return json.dumps(self._analyze_generic(move_history, game_name))

    def _analyze_pd(self, history: list, player: str) -> dict:
        """Detect well-known Prisoner's Dilemma strategies."""
        opponent = "player_b" if player == "player_a" else "player_a"
        opponent_actions = [
            r.get(f"{opponent}_action", r.get("action", "")) 
            for r in history if isinstance(r, dict)
        ]

        if not opponent_actions:
            return {
                "inferred_strategy": "unknown",
                "confidence": 0.0,
                "recommendation": "Insufficient history. Start with cooperation.",
            }

        cooperate_count = opponent_actions.count("cooperate")
        defect_count = opponent_actions.count("defect")
        total = len(opponent_actions)
        coop_rate = cooperate_count / total

        # Pattern detection
        if coop_rate >= 0.9:
            strategy = "Always Cooperate"
            confidence = coop_rate
            recommendation = "Cooperate to maximize joint payoff (Pareto optimal)."
        elif coop_rate <= 0.1:
            strategy = "Always Defect"
            confidence = 1 - coop_rate
            recommendation = "Defect. Cooperating yields 0 payoff against always-defect."
        elif self._is_tit_for_tat(opponent_actions, history, player):
            strategy = "Tit-for-Tat"
            confidence = 0.8
            recommendation = "Cooperate consistently — TfT matches your last action."
        elif self._is_grim_trigger(opponent_actions):
            strategy = "Grim Trigger"
            confidence = 0.7
            recommendation = "Never defect again — Grim Trigger defects forever after first defection."
        else:
            strategy = "Mixed / Unknown"
            confidence = 0.4
            recommendation = f"Coop rate: {coop_rate:.0%}. Consider mirroring opponent."

        return {
            "inferred_strategy": strategy,
            "confidence": round(confidence, 2),
            "coop_rate": round(coop_rate, 2),
            "recommendation": recommendation,
            "history_length": total,
        }

    def _is_tit_for_tat(self, opponent_actions: list, history: list, player: str) -> bool:
        """Check if opponent mirrors our last action."""
        if len(history) < 2:
            return False
        my_key = f"{player}_action"
        matches = 0
        for i in range(1, len(history)):
            my_last = history[i - 1].get(my_key, "")
            opp_current = opponent_actions[i] if i < len(opponent_actions) else ""
            if my_last == opp_current:
                matches += 1
        return matches / max(len(history) - 1, 1) >= 0.8

    def _is_grim_trigger(self, opponent_actions: list) -> bool:
        """Grim Trigger: cooperates until first defection, then always defects."""
        if "defect" not in opponent_actions:
            return False
        first_defect = opponent_actions.index("defect")
        return all(a == "defect" for a in opponent_actions[first_defect:])

    def _analyze_generic(self, history: list, game_name: str) -> dict:
        """Generic pattern analysis for other game types."""
        if not history:
            return {
                "inferred_strategy": "unknown",
                "confidence": 0.0,
                "recommendation": "No history yet. Play optimally.",
            }
        return {
            "inferred_strategy": "data-driven",
            "confidence": 0.5,
            "moves_observed": len(history),
            "recommendation": "Analyze the last 3 moves to detect positional preference.",
        }

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError
