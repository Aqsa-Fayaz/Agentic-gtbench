"""
Reasoning-strategy registry.

Each strategy exposes a `run(llm, game_state, game_name, legal_moves, error, tools)`
function that performs the LLM call(s) and returns a move dict.

Strategies:
    direct  — single zero-shot LLM call
    cot     — single LLM call with chain-of-thought scaffolding
    tot     — three sequential calls: generate → evaluate → select
    react   — Thought / Action / Observation loop with tool use
"""

from agents.strategies import direct, cot, tot, react

STRATEGY_REGISTRY = {
    "direct": direct,
    "cot": cot,
    "tot": tot,
    "react": react,
}


def load_strategy(name: str):
    """Return the strategy module registered under `name` (case-insensitive)."""
    key = name.lower()
    if key not in STRATEGY_REGISTRY:
        raise ValueError(
            f"Unknown strategy '{name}'. Available: {list(STRATEGY_REGISTRY.keys())}"
        )
    return STRATEGY_REGISTRY[key]


__all__ = ["STRATEGY_REGISTRY", "load_strategy", "direct", "cot", "tot", "react"]
