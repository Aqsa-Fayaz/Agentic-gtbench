"""
Shared helpers for reasoning strategies.
"""

from __future__ import annotations

import json
from typing import Optional


def build_state_prompt(
    game_state: dict,
    game_name: str,
    legal_moves: list,
    error: Optional[str] = None,
) -> str:
    """Assemble a canonical game-state prompt block."""
    prompt = f"=== {game_name.upper()} ===\n"
    prompt += f"Current State:\n{json.dumps(game_state, indent=2, default=str)}\n\n"
    prompt += f"Legal Moves ({len(legal_moves)}):\n{json.dumps(legal_moves, indent=2)}\n\n"
    if error:
        prompt += (
            "[WARNING] Your previous move was invalid: "
            f"{error}\nPlease choose a DIFFERENT legal move.\n\n"
        )
    prompt += "Choose your move. Respond ONLY with valid JSON."
    return prompt


def parse_move(text: str) -> dict:
    """
    Extract a JSON move object from the LLM's textual response.
    Accepts either `{"move": {...}}` wrappers or a bare move object.
    """
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in response: {text[:200]}")
    raw = json.loads(text[start:end])
    if isinstance(raw, dict) and "move" in raw and isinstance(raw["move"], dict):
        return raw["move"]
    return raw


def safe_parse_json(text: str):
    """Parse JSON, gracefully handling markdown code fences."""
    text = text.strip()
    if "```json" in text:
        text = text.split("```json", 1)[1].split("```", 1)[0].strip()
    elif "```" in text:
        text = text.split("```", 1)[1].split("```", 1)[0].strip()
    return json.loads(text)
