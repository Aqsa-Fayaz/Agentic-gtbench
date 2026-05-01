"""
Negotiation Game Environment
Type: Incomplete-information, dynamic, non-zero-sum
GTBench Category: Incomplete-information dynamic gaming (paper Table 1)

Two players negotiate over a fixed pool of 3 item types (default 5 books, 3
hats, 2 balls). Each player has SECRET per-item valuations drawn at random
from [0, 10] such that each player's total potential payoff is bounded.

Each turn the active player picks ONE of three actions:
  1. {"action": "propose", "for_me": {"book": k1, "hat": k2, "ball": k3}}
       — proposes that the active player keeps k_i of each item, the
         opponent gets the rest. Counts must be non-negative and ≤ pool.
  2. {"action": "accept"}
       — accepts the OPPONENT'S most recent proposal (interpreted from the
         opponent's perspective: opponent keeps what they proposed for
         themselves; the accepting player gets the rest).
       — illegal if no opponent proposal exists yet.
  3. {"action": "walk_away"}
       — terminate immediately. Both players get 0 payoff.

Terminal conditions:
  * accept                        → both get their share of the accepted split
  * walk_away                     → both get 0
  * turn_number >= max_turns      → both get 0  (deadline expiry)

Winner = player with strictly higher payoff; ties → "draw".
"""

from __future__ import annotations

import random
from typing import Optional

from games.base_game import BaseGame


class Negotiation(BaseGame):

    PROPOSE = "propose"
    ACCEPT = "accept"
    WALK_AWAY = "walk_away"
    VALID_ACTIONS = {PROPOSE, ACCEPT, WALK_AWAY}

    DEFAULT_POOL: dict = {"book": 5, "hat": 3, "ball": 2}

    def __init__(
        self,
        pool: Optional[dict] = None,
        max_turns: int = 10,
        valuation_max: int = 10,
        seed: Optional[int] = None,
    ):
        self.pool: dict = dict(pool) if pool else dict(self.DEFAULT_POOL)
        self.max_turns = int(max_turns)
        self.valuation_max = int(valuation_max)
        self._rng = random.Random(seed)
        super().__init__()

    # ─────────────────────────────────────────────
    # State
    # ─────────────────────────────────────────────

    def reset(self) -> dict:
        # Sample valuations for each player, ensuring at least one non-zero.
        def sample_valuations() -> dict:
            while True:
                v = {item: self._rng.randint(0, self.valuation_max) for item in self.pool}
                if any(val > 0 for val in v.values()):
                    return v

        self._valuations = {
            self.PLAYER_A: sample_valuations(),
            self.PLAYER_B: sample_valuations(),
        }
        self.proposal_history: list[dict] = []     # [{turn, player, for_me, for_opp}]
        self.turn_number = 0
        self.payoffs: dict = {self.PLAYER_A: 0, self.PLAYER_B: 0}
        self._winner: Optional[str] = None
        self._terminal = False
        self._terminal_reason: Optional[str] = None
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "pool": dict(self.pool),
            "valuations": {p: dict(v) for p, v in self._valuations.items()},   # full info
            "proposal_history": [dict(p) for p in self.proposal_history],
            "turn_number": self.turn_number,
            "max_turns": self.max_turns,
            "current_player": self.get_current_player(),
            "payoffs": dict(self.payoffs),
            "winner": self._winner,
            "terminal": self._terminal,
            "terminal_reason": self._terminal_reason,
        }

    def get_state_for_player(self, player: str) -> dict:
        """Hide opponent's valuations."""
        state = self.get_state()
        state["my_valuations"] = dict(self._valuations[player])
        # Replace full valuations with only the player's own.
        state["valuations"] = {player: dict(self._valuations[player])}
        return state

    # ─────────────────────────────────────────────
    # Move validation + application
    # ─────────────────────────────────────────────

    def _last_proposal_by(self, player: str) -> Optional[dict]:
        for p in reversed(self.proposal_history):
            if p["player"] == player:
                return p
        return None

    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        if self._terminal:
            return False, "Negotiation already ended."
        if player != self.get_current_player():
            return False, f"It is not {player}'s turn."
        if "action" not in move:
            return False, "Move must have an 'action' key."
        action = move["action"]
        if action not in self.VALID_ACTIONS:
            return False, f"Action must be one of {sorted(self.VALID_ACTIONS)}. Got: {action}"

        if action == self.PROPOSE:
            for_me = move.get("for_me")
            if not isinstance(for_me, dict):
                return False, "Propose must include 'for_me' as a dict of item counts."
            for item, total in self.pool.items():
                count = for_me.get(item, 0)
                if not isinstance(count, int) or count < 0:
                    return False, f"'for_me[{item}]' must be a non-negative integer."
                if count > total:
                    return False, f"Cannot propose {count} {item}; only {total} in pool."
            extras = set(for_me.keys()) - set(self.pool.keys())
            if extras:
                return False, f"Unknown items in proposal: {sorted(extras)}"
            return True, "Valid."

        if action == self.ACCEPT:
            opponent = self.PLAYER_B if player == self.PLAYER_A else self.PLAYER_A
            if self._last_proposal_by(opponent) is None:
                return False, "Cannot accept — opponent has not made any proposal yet."
            return True, "Valid."

        # action == walk_away
        return True, "Valid."

    def make_move(self, player: str, move: dict) -> dict:
        valid, reason = self.is_valid_move(player, move)
        if not valid:
            raise ValueError(f"Invalid move: {reason}")

        action = move["action"]
        self.turn_number += 1

        if action == self.PROPOSE:
            for_me = {item: int(move["for_me"].get(item, 0)) for item in self.pool}
            for_opp = {item: self.pool[item] - for_me[item] for item in self.pool}
            self.proposal_history.append(
                {
                    "turn": self.turn_number,
                    "player": player,
                    "for_me": for_me,
                    "for_opp": for_opp,
                }
            )
            if self.turn_number >= self.max_turns:
                self._end_with_walk_away(reason="max_turns_reached")

        elif action == self.ACCEPT:
            opponent = self.PLAYER_B if player == self.PLAYER_A else self.PLAYER_A
            last = self._last_proposal_by(opponent)
            assert last is not None  # validated above
            # Opponent keeps `last['for_me']`; this player gets `last['for_opp']`.
            opp_share = last["for_me"]
            self_share = last["for_opp"]
            self._settle(
                shares={opponent: opp_share, player: self_share},
                reason="accepted",
            )

        elif action == self.WALK_AWAY:
            self._end_with_walk_away(reason="walked_away")

        return self.get_state()

    def _settle(self, shares: dict, reason: str) -> None:
        for p, share in shares.items():
            self.payoffs[p] = sum(
                share.get(item, 0) * self._valuations[p][item] for item in self.pool
            )
        pa = self.payoffs[self.PLAYER_A]
        pb = self.payoffs[self.PLAYER_B]
        if pa > pb:
            self._winner = self.PLAYER_A
        elif pb > pa:
            self._winner = self.PLAYER_B
        else:
            self._winner = "draw"
        self._terminal = True
        self._terminal_reason = reason

    def _end_with_walk_away(self, reason: str) -> None:
        self.payoffs = {self.PLAYER_A: 0, self.PLAYER_B: 0}
        self._winner = "draw"
        self._terminal = True
        self._terminal_reason = reason

    # ─────────────────────────────────────────────
    # Terminal / winner / legal moves
    # ─────────────────────────────────────────────

    def is_terminal(self) -> bool:
        return self._terminal

    def get_winner(self) -> Optional[str]:
        return self._winner if self._terminal else None

    def get_legal_moves(self, player: str) -> list:
        """
        Negotiation has a continuous proposal space. We surface a small set
        of canonical anchors so the LLM has discrete options:
          - accept (only if opponent has proposed)
          - walk_away
          - propose: a few "split" templates (all-mine, all-yours, equal-ish)
        """
        if self._terminal or player != self.get_current_player():
            return []
        moves: list[dict] = []
        opponent = self.PLAYER_B if player == self.PLAYER_A else self.PLAYER_A
        if self._last_proposal_by(opponent) is not None:
            moves.append({"action": self.ACCEPT})
        moves.append({"action": self.WALK_AWAY})

        # Three canonical proposal templates.
        all_mine = {item: total for item, total in self.pool.items()}
        none_mine = {item: 0 for item in self.pool}
        equal_split = {item: total // 2 for item, total in self.pool.items()}
        for tpl in (all_mine, equal_split, none_mine):
            moves.append({"action": self.PROPOSE, "for_me": tpl})
        return moves

    # ─────────────────────────────────────────────
    # Render + restore
    # ─────────────────────────────────────────────

    def render(self) -> str:
        lines = [
            f"=== Negotiation — turn {self.turn_number}/{self.max_turns} ===",
            f"  Pool: {self.pool}",
            f"  Active: {self.get_current_player()}",
        ]
        for p, v in self._valuations.items():
            lines.append(f"  Valuations[{p}] (private): {v}")
        if self.proposal_history:
            last = self.proposal_history[-1]
            lines.append(f"  Last proposal: {last}")
        if self._terminal:
            lines.append(
                f"  Outcome: {self._terminal_reason}  payoffs={self.payoffs}  winner={self._winner}"
            )
        return "\n".join(lines)

    @classmethod
    def from_state(cls, state: dict) -> "Negotiation":
        pool = state.get("pool") or dict(cls.DEFAULT_POOL)
        max_turns = state.get("max_turns", 10)
        game = cls(pool=pool, max_turns=max_turns)
        # Restore valuations (full state). For a per-player state, only one
        # side's valuations will be present; we leave the other side as the
        # randomly-sampled one — adequate for move validation.
        full = state.get("valuations") or {}
        if isinstance(full, dict):
            for p, v in full.items():
                if p in game._valuations:
                    game._valuations[p] = dict(v)
        game.proposal_history = [dict(p) for p in state.get("proposal_history", [])]
        game.turn_number = state.get("turn_number", len(game.proposal_history))
        game.payoffs = dict(state.get("payoffs", {cls.PLAYER_A: 0, cls.PLAYER_B: 0}))
        game._winner = state.get("winner")
        game._terminal = bool(state.get("terminal", False))
        game._terminal_reason = state.get("terminal_reason")
        return game
