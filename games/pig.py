"""
Pig Game Environment
Type: Probabilistic, perfect-information, dynamic
GTBench Category: Probabilistic gaming (paper Table 1)

Single-die push-your-luck:
- On a turn, the active player chooses to ROLL or HOLD.
- ROLL: roll a 6-sided die.
    - 2-6: add to current `turn_total`. Player may roll again or hold.
    - 1:   `turn_total` is wiped to 0 and the turn ends — opponent moves next.
- HOLD: bank `turn_total` into the player's `score`. Opponent moves next.
- First player whose `score` reaches `target_score` (default 100) wins.

Move format: {"action": "roll"} or {"action": "hold"}
"""

from __future__ import annotations

import random
from typing import Optional

from games.base_game import BaseGame


class Pig(BaseGame):

    ROLL = "roll"
    HOLD = "hold"
    VALID_ACTIONS = {ROLL, HOLD}

    def __init__(
        self,
        target_score: int = 100,
        seed: Optional[int] = None,
    ):
        self.target_score = int(target_score)
        self._rng = random.Random(seed)
        super().__init__()

    # ─────────────────────────────────────────────
    # State management
    # ─────────────────────────────────────────────

    def reset(self) -> dict:
        self.scores = {self.PLAYER_A: 0, self.PLAYER_B: 0}
        self.turn_total = 0
        self._active_player: str = self.PLAYER_A
        self.turn_number = 0
        self._winner: Optional[str] = None
        self._last_roll: Optional[int] = None
        self.roll_history: list[dict] = []
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "scores": dict(self.scores),
            "turn_total": self.turn_total,
            "current_player": self._active_player,
            "turn_number": self.turn_number,
            "target_score": self.target_score,
            "last_roll": self._last_roll,
            "roll_history": list(self.roll_history),
            "winner": self._winner,
        }

    def get_state_for_player(self, player: str) -> dict:
        # Pig is perfect-info — same state for both players.
        return self.get_state()

    def get_current_player(self) -> str:
        # Override BaseGame's turn-parity rule: Pig's active player only
        # changes on a 1-roll bust or a HOLD, not on every move.
        return self._active_player

    # ─────────────────────────────────────────────
    # Move validation + application
    # ─────────────────────────────────────────────

    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        if self._winner is not None:
            return False, "Game already over."
        if player != self._active_player:
            return False, f"It is not {player}'s turn."
        if "action" not in move:
            return False, "Move must have an 'action' key."
        action = move["action"]
        if action not in self.VALID_ACTIONS:
            return False, f"Action must be 'roll' or 'hold'. Got: {action}"
        return True, "Valid."

    def make_move(self, player: str, move: dict) -> dict:
        valid, reason = self.is_valid_move(player, move)
        if not valid:
            raise ValueError(f"Invalid move: {reason}")

        action = move["action"]
        self.turn_number += 1

        if action == self.ROLL:
            roll = self._rng.randint(1, 6)
            self._last_roll = roll
            self.roll_history.append({"player": player, "roll": roll})
            if roll == 1:
                # Bust — wipe turn_total and switch active player.
                self.turn_total = 0
                self._switch_player()
            else:
                self.turn_total += roll
        elif action == self.HOLD:
            self.scores[player] += self.turn_total
            self.turn_total = 0
            if self.scores[player] >= self.target_score:
                self._winner = player
            else:
                self._switch_player()

        return self.get_state()

    def _switch_player(self) -> None:
        self._active_player = (
            self.PLAYER_B if self._active_player == self.PLAYER_A else self.PLAYER_A
        )

    # ─────────────────────────────────────────────
    # Terminal / winner
    # ─────────────────────────────────────────────

    def is_terminal(self) -> bool:
        return self._winner is not None

    def get_winner(self) -> Optional[str]:
        return self._winner

    def get_legal_moves(self, player: str) -> list:
        if self._winner is not None or player != self._active_player:
            return []
        return [{"action": self.ROLL}, {"action": self.HOLD}]

    # ─────────────────────────────────────────────
    # Render
    # ─────────────────────────────────────────────

    def render(self) -> str:
        lines = [
            f"=== Pig — target {self.target_score} — turn {self.turn_number} ===",
            f"  Scores: A={self.scores[self.PLAYER_A]}  B={self.scores[self.PLAYER_B]}",
            f"  Active: {self._active_player}  Turn total: {self.turn_total}",
        ]
        if self._last_roll is not None:
            lines.append(f"  Last roll: {self._last_roll}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # State restoration
    # ─────────────────────────────────────────────

    @classmethod
    def from_state(cls, state: dict) -> "Pig":
        game = cls(target_score=state.get("target_score", 100))
        game.scores = dict(state.get("scores", {cls.PLAYER_A: 0, cls.PLAYER_B: 0}))
        game.turn_total = state.get("turn_total", 0)
        game._active_player = state.get("current_player", cls.PLAYER_A)
        game.turn_number = state.get("turn_number", 0)
        game._last_roll = state.get("last_roll")
        game.roll_history = list(state.get("roll_history", []))
        game._winner = state.get("winner")
        return game
