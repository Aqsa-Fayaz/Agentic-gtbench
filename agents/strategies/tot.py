"""
Tree-of-Thoughts (ToT) reasoning strategy.

Three sequential LLM calls:
    1. Generate  — propose 3 candidate moves with short justifications.
    2. Evaluate  — score each candidate 1..10.
    3. Select    — pick the top-scoring candidate and emit JSON.
"""

from __future__ import annotations

import json
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.strategies._common import build_state_prompt, parse_move, safe_parse_json

NAME = "tot"
SYSTEM_PROMPT = (
    "You are a strategic game-playing AI. Your goal is to win. "
    "You will be guided through a structured tree-of-thoughts process: "
    "first generating candidate moves, then scoring them, then selecting the best."
)


def run(
    llm,
    game_state: dict,
    game_name: str,
    legal_moves: list,
    error: Optional[str] = None,
    tools: Optional[list] = None,
) -> dict:
    system = SystemMessage(content=SYSTEM_PROMPT)
    base_state = build_state_prompt(game_state, game_name, legal_moves, error)

    generate_prompt = (
        f"{base_state}\n\n"
        "STEP 1 — GENERATE: Produce exactly 3 distinct candidate moves drawn "
        "from the legal moves above. For each candidate provide a 1-2 sentence "
        'justification. Respond as a JSON array: [{"move": {...}, "justification": "..."}]'
    )
    gen_response = llm.invoke([system, HumanMessage(content=generate_prompt)])
    try:
        candidates = safe_parse_json(gen_response.content)
    except Exception:
        candidates = []

    evaluate_prompt = (
        "STEP 2 — EVALUATE: Score each candidate from 1 (bad) to 10 (excellent). "
        'Respond as a JSON array: [{"move": {...}, "score": int, "reason": "..."}]\n\n'
        f"Candidates:\n{json.dumps(candidates, indent=2, default=str)}"
    )
    eval_response = llm.invoke([system, HumanMessage(content=evaluate_prompt)])
    try:
        scored = safe_parse_json(eval_response.content)
    except Exception:
        scored = candidates

    select_prompt = (
        "STEP 3 — SELECT: From the scored candidates below choose the single best move. "
        'Respond ONLY with JSON: {"move": {...}}\n\n'
        f"Scored candidates:\n{json.dumps(scored, indent=2, default=str)}"
    )
    select_response = llm.invoke([system, HumanMessage(content=select_prompt)])
    return parse_move(select_response.content)
