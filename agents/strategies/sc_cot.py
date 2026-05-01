"""
Self-Consistent Chain-of-Thought (SC-CoT) reasoning strategy.

Per GTBench paper §3.2: sample N reasoning trajectories at higher temperature,
parse each as a candidate move, then majority-vote across the candidates.

Trades latency (N× LLM calls) for robustness — when CoT randomness produces
inconsistent moves, the consensus pick is more reliable than any single sample.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.strategies._common import build_state_prompt, parse_move

NAME = "sc_cot"
DEFAULT_SAMPLES = 5
SAMPLE_TEMPERATURE = 0.7   # higher than CoT's default to encourage divergence

SYSTEM_PROMPT = (
    "You are a strategic game-playing AI. Think step by step before deciding:\n"
    "  1. Describe the current game state in your own words.\n"
    "  2. Identify immediate threats (opponent's winning moves).\n"
    "  3. Identify your opportunities (your winning moves).\n"
    "  4. Evaluate each of the legal moves you are given.\n"
    "  5. Pick the move that leads to the best outcome.\n"
    'Finish your response with JSON: {"move": {...}, "reasoning": "short explanation"}'
)

logger = logging.getLogger(__name__)


def _move_key(move: dict) -> tuple:
    """
    Hashable representation of a move dict, used to count votes.
    Sorted-items tuple handles arbitrary nested primitives.
    """
    def freeze(value):
        if isinstance(value, dict):
            return tuple(sorted((k, freeze(v)) for k, v in value.items()))
        if isinstance(value, list):
            return tuple(freeze(x) for x in value)
        return value

    return freeze(move)


def _sample_once(llm, system, user, sample_temperature: float) -> Optional[dict]:
    """
    One LLM call at higher temperature. Returns a parsed move or None on failure.
    LangChain ChatOpenAI lets us override temperature via .bind().
    """
    try:
        bound = llm.bind(temperature=sample_temperature) if hasattr(llm, "bind") else llm
        response = bound.invoke([system, user])
        return parse_move(response.content)
    except Exception as exc:
        logger.warning(f"SC-CoT sample failed: {exc}")
        return None


def run(
    llm,
    game_state: dict,
    game_name: str,
    legal_moves: list,
    error: Optional[str] = None,
    tools: Optional[list] = None,
    n_samples: int = DEFAULT_SAMPLES,
) -> dict:
    system = SystemMessage(content=SYSTEM_PROMPT)
    user = HumanMessage(content=build_state_prompt(game_state, game_name, legal_moves, error))

    candidates: list[dict] = []
    for _ in range(n_samples):
        move = _sample_once(llm, system, user, SAMPLE_TEMPERATURE)
        if move is not None:
            candidates.append(move)

    if not candidates:
        # All samples failed — fall back to a single best-effort call so the
        # caller still gets *something* and the retry loop in PlayerAgent can
        # report a clean error.
        response = llm.invoke([system, user])
        return parse_move(response.content)

    # Majority vote on the hashable move key.
    counts = Counter(_move_key(c) for c in candidates)
    best_key, _ = counts.most_common(1)[0]
    # Recover the original dict for the winning key.
    for c in candidates:
        if _move_key(c) == best_key:
            return c
    return candidates[0]   # unreachable, but defensive
