"""
Chain-of-Thought (CoT) reasoning strategy.
One LLM call. Prompt asks the model to think step-by-step before outputting JSON.
"""

from __future__ import annotations

from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.strategies._common import build_state_prompt, parse_move

NAME = "cot"
SYSTEM_PROMPT = (
    "You are a strategic game-playing AI. Your goal is to win.\n"
    "Think step by step before deciding:\n"
    "  1. Describe the current game state in your own words.\n"
    "  2. Identify immediate threats (opponent's winning moves).\n"
    "  3. Identify your opportunities (your winning moves).\n"
    "  4. Evaluate each of the legal moves you are given.\n"
    "  5. Pick the move that leads to the best outcome.\n"
    'Finish your response with JSON: {"move": {...}, "reasoning": "short explanation"}'
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
