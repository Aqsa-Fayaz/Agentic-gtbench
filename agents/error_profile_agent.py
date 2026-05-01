"""
ErrorProfileAgent — LLM-as-judge for game-move failure classification.

Reproduces the GTBench paper's §4.4 / Table 5 error taxonomy:
    1. misinterpretation     — misread the game state
    2. factual_error         — plan and action don't align
    3. overconfidence        — accepted unnecessary risk
    4. calculation_mistake   — arithmetic / counting error
    5. endgame_misdetection  — missed an immediate win/loss
    6. ok                    — no clear error (default bucket)

Workflow per session:
    1. Replay session.move_history through a fresh game instance to capture
       the state immediately *before* each move.
    2. Filter to losing-side moves (the moves we most want to diagnose).
    3. For each candidate move, ask the LLM to classify it.
    4. Return a list of structured findings the EvaluatorAgent / MetricsEngine
       can aggregate.

The classifier does NOT play games and does NOT mutate session data. It is a
pure post-pass over already-collected sessions, so it can be re-run without
re-spending API budget on game play.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from agents.strategies._common import safe_parse_json
from config.settings import settings
from games import load_game
from tools.state_tracker import describe_state

logger = logging.getLogger(__name__)


class ErrorProfileAgent(BaseAgent):
    """
    Post-hoc judge that classifies losing-side moves into the GTBench
    error taxonomy. Uses the same Groq backend as the player agents.
    """

    CATEGORIES = (
        "misinterpretation",
        "factual_error",
        "overconfidence",
        "calculation_mistake",
        "endgame_misdetection",
        "ok",
    )

    SYSTEM_PROMPT = (
        "You are an expert game-theoretic move evaluator. Given a game state "
        "and the move a player made, classify the move into EXACTLY ONE of:\n"
        "  1. misinterpretation     — player misread the state (e.g. wrong "
        "piece ownership, missed a vacant cell, ignored own card)\n"
        "  2. factual_error         — player's stated plan and chosen action "
        "do not match\n"
        "  3. overconfidence        — player took unnecessary risk for a "
        "marginal reward\n"
        "  4. calculation_mistake   — arithmetic or counting error (e.g. "
        "miscounted Nim pile XOR)\n"
        "  5. endgame_misdetection  — player failed to recognise an immediate "
        "winning or losing situation\n"
        "  6. ok                    — no clear error; the move was reasonable "
        "given the information available\n\n"
        "Be strict: only classify a non-'ok' category if there is concrete "
        "evidence in the state. Respond ONLY with JSON: "
        '{"category": "<one of the six>", "rationale": "<10-word reason>"}'
    )

    def __init__(
        self,
        agent_id: str = "error_profile_judge",
        model: str = None,
        temperature: float = 0.0,
        provider: str = None,
    ):
        self.provider = (provider or settings.default_provider).lower()
        resolved_model = settings.resolve_model(self.provider, model)
        super().__init__(agent_id=agent_id, model=resolved_model, temperature=temperature)
        self._llm: Optional[ChatOpenAI] = None
        self.findings: list[dict] = []

    # ─────────────────────────────────────────────
    # LLM lazy init
    # ─────────────────────────────────────────────

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            if self.provider == "groq":
                kwargs = settings.get_groq_client_kwargs(
                    model=self.model, temperature=self.temperature
                )
                self._llm = ChatOpenAI(
                    model=kwargs["model"],
                    temperature=kwargs["temperature"],
                    max_tokens=kwargs["max_tokens"],
                    api_key=kwargs["api_key"],
                    base_url=kwargs["base_url"],
                )
            else:
                kwargs = settings.get_openai_client_kwargs(
                    model=self.model, temperature=self.temperature
                )
                self._llm = ChatOpenAI(
                    model=kwargs["model"],
                    temperature=kwargs["temperature"],
                    max_tokens=kwargs["max_tokens"],
                    api_key=kwargs["api_key"],
                )
        return self._llm

    def decide(self, game_state: dict, game_name: str, legal_moves: list) -> dict:
        raise NotImplementedError("ErrorProfileAgent does not play games.")

    # ─────────────────────────────────────────────
    # Per-move classification
    # ─────────────────────────────────────────────

    def classify_move(
        self,
        game_name: str,
        state_before: dict,
        move: dict,
        player: str,
        legal_moves: list,
        outcome: str,
    ) -> dict:
        """Single LLM call. Returns {category, rationale} (always falls back to 'ok' on failure)."""
        try:
            tracker = describe_state(game_name, state_before, player)
            description = tracker.get("description", json.dumps(state_before, default=str)[:400])
        except Exception:
            description = json.dumps(state_before, default=str)[:400]

        prompt = (
            f"=== {game_name.upper()} — turn-{state_before.get('turn_number', '?')} review ===\n"
            f"Player evaluated: {player}\n"
            f"Final game outcome for this player: {outcome}\n\n"
            f"State BEFORE the move:\n{description}\n\n"
            f"Legal moves at this turn:\n{json.dumps(legal_moves, default=str)[:400]}\n\n"
            f"Move actually played:\n{json.dumps(move, default=str)}\n\n"
            "Classify the move."
        )

        try:
            response = self.llm.invoke(
                [SystemMessage(content=self.SYSTEM_PROMPT), HumanMessage(content=prompt)]
            )
            payload = safe_parse_json(response.content or "{}")
            category = str(payload.get("category", "ok")).strip().lower()
            if category not in self.CATEGORIES:
                category = "ok"
            return {
                "category": category,
                "rationale": str(payload.get("rationale", ""))[:200],
            }
        except Exception as exc:
            logger.warning(f"classify_move failed ({exc}); defaulting to 'ok'")
            return {"category": "ok", "rationale": f"[judge error: {exc}]"}

    # ─────────────────────────────────────────────
    # Per-session orchestration
    # ─────────────────────────────────────────────

    def classify_session(
        self,
        session: dict,
        only_losing_side: bool = True,
    ) -> list[dict]:
        """
        Replay the session's move_history, classify losing-side moves.
        Returns a list of finding dicts; also appends to self.findings.

        Skips draws unless only_losing_side=False.
        """
        game_name = session.get("game_name")
        if not game_name:
            return []

        # Replay needs the same construction kwargs as the original run.
        # They live in session_meta.game_kwargs in the LangGraph state, but
        # the persisted session dict doesn't carry meta — fall back to {}.
        game_kwargs = session.get("game_kwargs", {}) or {}
        try:
            game = load_game(game_name, **game_kwargs)
        except Exception as exc:
            logger.warning(f"classify_session: cannot load game '{game_name}': {exc}")
            return []
        game.reset()

        winner = session.get("winner")
        losing_side: Optional[str] = None
        if winner == "player_a":
            losing_side = "player_b"
        elif winner == "player_b":
            losing_side = "player_a"
        # winner == "draw" or None → no losing side; only classify if requested.

        if only_losing_side and losing_side is None:
            return []

        strategy_for: dict[str, str] = {
            "player_a": session.get("player_a_strategy", "unknown"),
            "player_b": session.get("player_b_strategy", "unknown"),
        }

        findings: list[dict] = []
        for entry in session.get("move_history", []):
            player = entry.get("player")
            move = entry.get("move")
            if player is None or move is None:
                continue

            # Snapshot state BEFORE applying the move.
            state_before = game.get_state_for_player(player)
            legal_before = game.get_legal_moves(player)

            try:
                game.make_move(player, move)
            except Exception as exc:
                logger.warning(
                    f"classify_session: invalid move during replay "
                    f"({game_name}, turn {entry.get('turn')}): {exc}"
                )
                # Bail out — replay diverged from the original run.
                break

            should_classify = (
                not only_losing_side
                or player == losing_side
            )
            if not should_classify:
                continue

            outcome = "loss" if player == losing_side else (
                "win" if player and winner == player else "draw_or_unknown"
            )

            verdict = self.classify_move(
                game_name=game_name,
                state_before=state_before,
                move=move,
                player=player,
                legal_moves=legal_before,
                outcome=outcome,
            )
            findings.append(
                {
                    "session_id": session.get("session_id"),
                    "game_name": game_name,
                    "turn": entry.get("turn"),
                    "player": player,
                    "strategy": strategy_for.get(player, "unknown"),
                    "move": move,
                    "category": verdict["category"],
                    "rationale": verdict["rationale"],
                }
            )

        self.findings.extend(findings)
        return findings

    # ─────────────────────────────────────────────
    # Aggregation helpers
    # ─────────────────────────────────────────────

    def aggregate_by_strategy(self) -> dict:
        """{strategy: {category: count, ..., total: N}}"""
        agg: dict[str, dict[str, int]] = {}
        for f in self.findings:
            strat = f.get("strategy", "unknown")
            cat = f.get("category", "ok")
            bucket = agg.setdefault(
                strat,
                {c: 0 for c in self.CATEGORIES} | {"total": 0},
            )
            bucket[cat] += 1
            bucket["total"] += 1
        # Add per-category percentages.
        for strat, counts in agg.items():
            total = counts["total"] or 1
            for c in self.CATEGORIES:
                counts[f"{c}_pct"] = round(counts[c] / total, 3)
        return agg

    def aggregate_by_strategy_game(self) -> dict:
        """{(strategy, game): {category: count, ...}}"""
        agg: dict[tuple, dict[str, int]] = {}
        for f in self.findings:
            key = (f.get("strategy", "unknown"), f.get("game_name", "unknown"))
            bucket = agg.setdefault(
                key,
                {c: 0 for c in self.CATEGORIES} | {"total": 0},
            )
            bucket[f.get("category", "ok")] += 1
            bucket["total"] += 1
        return agg
