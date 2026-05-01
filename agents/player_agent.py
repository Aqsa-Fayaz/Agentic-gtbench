"""
PlayerAgent — LLM-powered game-playing agent.

Delegates the actual reasoning to one of the pluggable strategy modules in
`agents.strategies`, while owning:
    - the LLM client (ChatOpenAI / Groq via OpenAI-compatible endpoint)
    - retry / invalid-move bookkeeping
    - per-session conversation + move history
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from agents.strategies import load_strategy
from config.settings import settings

logger = logging.getLogger(__name__)


class PlayerAgent(BaseAgent):
    """
    Autonomous game-playing agent.

    decide() flow per turn:
        1. Delegate to the chosen strategy (direct/cot/tot/react) to produce a move.
        2. On parse / format failure, retry up to settings.max_retries times,
           passing the prior error back to the strategy so it can correct itself.
        3. Return the final move dict (or raise RuntimeError if all retries fail;
           the LangGraph orchestrator converts this to a forfeit).

    Tracks:
        - invalid_move_count / total_moves (for metrics)
        - move_history (per-game audit trail with timestamps)
    """

    SUPPORTED_STRATEGIES = ("direct", "cot", "tot", "react")

    def __init__(
        self,
        agent_id: str,
        strategy: str = "cot",
        model: str = None,
        temperature: float = None,
        tools: list = None,
        provider: str = "openai",
    ):
        super().__init__(agent_id=agent_id, model=model, temperature=temperature)
        self.strategy_name = strategy.lower()
        if self.strategy_name not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. "
                f"Choose from: {self.SUPPORTED_STRATEGIES}"
            )
        self.strategy = load_strategy(self.strategy_name)
        self.tools = tools or []
        self.provider = provider.lower()

        self.invalid_move_count = 0
        self.total_moves = 0
        self.move_history: list[dict] = []
        self.conversation_history: list = []

        self.llm = self._build_llm()

    def _build_llm(self) -> ChatOpenAI:
        """Instantiate a ChatOpenAI-compatible client for the configured provider."""
        if self.provider == "groq":
            kwargs = settings.get_groq_client_kwargs(model=self.model, temperature=self.temperature)
            return ChatOpenAI(
                model=kwargs["model"],
                temperature=kwargs["temperature"],
                api_key=kwargs["api_key"],
                base_url=kwargs["base_url"],
            )
        if self.provider == "openrouter":
            kwargs = settings.get_openrouter_client_kwargs(
                model=self.model,
                temperature=self.temperature,
            )
            return ChatOpenAI(
                model=kwargs["model"],
                temperature=kwargs["temperature"],
                api_key=kwargs["api_key"],
                base_url=kwargs["base_url"],
            )
        if self.provider != "openai":
            raise ValueError(
                f"Unsupported provider '{self.provider}'. "
                "Choose from: openai, groq, openrouter."
            )
        return settings.build_chat_openai_client(
            model=self.model,
            temperature=self.temperature,
        )

    @property
    def strategy_key(self) -> str:
        """Alias for older call-sites that expected `agent.strategy` to be a string."""
        return self.strategy_name

    def decide(self, game_state: dict, game_name: str, legal_moves: list) -> dict:
        """
        Produce a move. Raises RuntimeError after exhausting retries, so the
        LangGraph orchestrator can route to the forfeit branch.
        """
        self.total_moves += 1
        last_error: Optional[str] = None

        for attempt in range(settings.max_retries + 1):
            try:
                move = self.strategy.run(
                    llm=self.llm,
                    game_state=game_state,
                    game_name=game_name,
                    legal_moves=legal_moves,
                    error=last_error,
                    tools=self.tools,
                )
                logger.info(
                    f"[{self.agent_id}][{self.strategy_name}] move#{self.total_moves} "
                    f"-> {json.dumps(move, default=str)}"
                )
                return move
            except (ValueError, json.JSONDecodeError, KeyError) as exc:
                last_error = str(exc)
                self.invalid_move_count += 1
                logger.warning(
                    f"[{self.agent_id}] parse/retry {attempt + 1}: {exc}"
                )

        raise RuntimeError(
            f"Agent {self.agent_id} failed to produce a valid move after "
            f"{settings.max_retries} retries."
        )

    def record_move(self, move: dict, valid: bool, game_name: str) -> None:
        self.move_history.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "game": game_name,
                "move": move,
                "valid": valid,
            }
        )

    def get_stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "strategy": self.strategy_name,
            "model": self.model,
            "total_moves": self.total_moves,
            "invalid_moves": self.invalid_move_count,
            "invalid_rate": self.invalid_move_count / max(self.total_moves, 1),
        }

    def reset_session(self) -> None:
        """Reset per-game conversation state (keeps cumulative stats)."""
        self.conversation_history = []
