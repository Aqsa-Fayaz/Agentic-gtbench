"""
Direct (zero-shot) reasoning strategy.
One LLM call. The model receives state + legal moves and responds with JSON.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.strategies._common import build_state_prompt, parse_move

NAME = "direct"
SYSTEM_PROMPT = (
    "You are a strategic game-playing AI. Your goal is to win. "
    "Analyze the board state and choose the best legal move. "
    "Always respond with a valid JSON move object and nothing else."
)


def run(
    llm,
    game_state: dict,
    game_name: str,
    legal_moves: list,
    error: Optional[str] = None,
    tools: Optional[list] = None,
) -> dict:
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=build_state_prompt(game_state, game_name, legal_moves, error)),
    ]
    response = llm.invoke(messages)
    return parse_move(response.content)
