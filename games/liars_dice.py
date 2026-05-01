"""
Liar's Dice (2-player simplified)
Type: Incomplete-information, probabilistic, zero-sum
GTBench Category: Incomplete & probabilistic gaming (paper Table 1)

Each player rolls N dice privately (default N=5). Players then alternate making
bids, where a bid is `(face_value, count)` claiming that AT LEAST `count` dice
across BOTH hands show `face_value`. Each new bid must strictly raise the prior
either by face value (with count >= prior count) or by count (any face).
A player can instead `call` (challenge) the opponent's bid:
- count actual dice across BOTH hands matching the bid's face_value
  (face_value=1 is wild and ALSO matches everything — common variant; here
  we use the strict variant where 1s only count as 1s for simplicity).
- if actual_count >= bid_count, the bidder wins; else the caller wins.

Move format:
  {"action": "bid", "face": int, "count": int}
  {"action": "call"}
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Optional

from games.base_game import BaseGame


class LiarsDice(BaseGame):

    BID = "bid"
    CALL = "call"
    VALID_ACTIONS = {BID, CALL}

    def __init__(
        self,
        dice_per_player: int = 5,
        die_faces: int = 6,
        seed: Optional[int] = None,
    ):
        self.dice_per_player = int(dice_per_player)
        self.die_faces = int(die_faces)
        self._rng = random.Random(seed)
        super().__init__()

    # ─────────────────────────────────────────────
    # State
    # ─────────────────────────────────────────────

    def reset(self) -> dict:
        self._dice = {
            self.PLAYER_A: sorted(
                self._rng.randint(1, self.die_faces) for _ in range(self.dice_per_player)
            ),
            self.PLAYER_B: sorted(
                self._rng.randint(1, self.die_faces) for _ in range(self.dice_per_player)
            ),
        }
        self.bid_history: list[dict] = []     # [{player, face, count}]
        self.turn_number = 0
        self._winner: Optional[str] = None
        self._terminal = False
        self._called_bid: Optional[dict] = None
        self._actual_count: Optional[int] = None
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "dice": {p: list(d) for p, d in self._dice.items()},   # full info
            "dice_per_player": self.dice_per_player,
            "die_faces": self.die_faces,
            "bid_history": [dict(b) for b in self.bid_history],
            "turn_number": self.turn_number,
            "current_player": self.get_current_player(),
            "winner": self._winner,
            "terminal": self._terminal,
            "called_bid": self._called_bid,
            "actual_count_at_call": self._actual_count,
        }

    def get_state_for_player(self, player: str) -> dict:
        """Hide the opponent's dice until the round is called."""
        state = self.get_state()
        state["my_dice"] = list(self._dice[player])
        if self._terminal:
            opp = self.PLAYER_B if player == self.PLAYER_A else self.PLAYER_A
            state["opponent_dice"] = list(self._dice[opp])
        else:
            state["opponent_dice"] = None
        # Strip raw `dice` so info isn't leaked.
        state.pop("dice", None)
        return state

    # ─────────────────────────────────────────────
    # Move validation + application
    # ─────────────────────────────────────────────

    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        if self._terminal:
            return False, "Game already over."
        if player != self.get_current_player():
            return False, f"It is not {player}'s turn."
        if "action" not in move:
            return False, "Move must have an 'action' key."
        action = move["action"]
        if action not in self.VALID_ACTIONS:
            return False, f"Action must be 'bid' or 'call'. Got: {action}"

        if action == self.CALL:
            if not self.bid_history:
                return False, "Cannot call before any bid has been made."
            return True, "Valid."

        # action == bid
        face = move.get("face")
        count = move.get("count")
        if not (isinstance(face, int) and isinstance(count, int)):
            return False, "Bid must include integer 'face' and 'count'."
        if not (1 <= face <= self.die_faces):
            return False, f"Face must be 1..{self.die_faces}. Got: {face}"
        max_count = 2 * self.dice_per_player
        if not (1 <= count <= max_count):
            return False, f"Count must be 1..{max_count}. Got: {count}"

        if not self.bid_history:
            return True, "Valid."
        last = self.bid_history[-1]
        if face > last["face"]:
            if count >= 1:
                return True, "Valid (raised face)."
            return False, "Count must be positive."
        if face == last["face"]:
            if count > last["count"]:
                return True, "Valid (raised count at same face)."
            return False, f"Count must exceed previous count {last['count']}."
        # face < last["face"]
        if count > last["count"]:
            return True, "Valid (lower face but higher count)."
        return False, "New bid must strictly raise face or count."

    def make_move(self, player: str, move: dict) -> dict:
        valid, reason = self.is_valid_move(player, move)
        if not valid:
            raise ValueError(f"Invalid move: {reason}")

        action = move["action"]
        self.turn_number += 1

        if action == self.BID:
            self.bid_history.append(
                {"player": player, "face": int(move["face"]), "count": int(move["count"])}
            )
        else:  # CALL
            last = self.bid_history[-1]
            self._called_bid = dict(last)
            tally = Counter(self._dice[self.PLAYER_A]) + Counter(self._dice[self.PLAYER_B])
            actual = tally.get(last["face"], 0)
            self._actual_count = actual
            bidder = last["player"]
            caller = player
            if actual >= last["count"]:
                self._winner = bidder      # bid was honest
            else:
                self._winner = caller      # bidder lied
            self._terminal = True

        return self.get_state()

    # ─────────────────────────────────────────────
    # Turn order, terminal, legal moves
    # ─────────────────────────────────────────────

    def get_current_player(self) -> str:
        return self.PLAYER_A if self.turn_number % 2 == 0 else self.PLAYER_B

    def is_terminal(self) -> bool:
        return self._terminal

    def get_winner(self) -> Optional[str]:
        return self._winner if self._terminal else None

    def get_legal_moves(self, player: str) -> list:
        if self._terminal or player != self.get_current_player():
            return []
        moves: list[dict] = []
        max_count = 2 * self.dice_per_player
        if self.bid_history:
            moves.append({"action": self.CALL})
            last = self.bid_history[-1]
            # Suggest a few sensible raises rather than enumerating all bids.
            for face in range(last["face"], self.die_faces + 1):
                for count in (last["count"] + 1, last["count"] + 2):
                    if face == last["face"] and count <= last["count"]:
                        continue
                    if count > max_count:
                        continue
                    moves.append({"action": self.BID, "face": face, "count": count})
        else:
            # Opening bid — give a few starter options.
            for face in range(2, self.die_faces + 1):
                moves.append({"action": self.BID, "face": face, "count": 2})
            moves.append({"action": self.BID, "face": 1, "count": 2})
        return moves

    def render(self) -> str:
        lines = [
            f"=== Liar's Dice ({self.dice_per_player} dice each, faces 1-{self.die_faces}) ===",
            f"  Active: {self.get_current_player()}   Turn: {self.turn_number}",
            f"  Bid history: {self.bid_history}",
        ]
        if self._terminal:
            lines.append(
                f"  Called bid {self._called_bid} → actual count "
                f"{self._actual_count} → winner {self._winner}"
            )
        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # State restoration
    # ─────────────────────────────────────────────

    @classmethod
    def from_state(cls, state: dict) -> "LiarsDice":
        dice_per_player = state.get("dice_per_player", 5)
        die_faces = state.get("die_faces", 6)
        game = cls(dice_per_player=dice_per_player, die_faces=die_faces)
        # Restore dice if present (full-info state). Otherwise leave RNG dice in
        # place — for move-validation purposes we mostly need the bid history.
        dice = state.get("dice")
        if dice:
            for p, d in dice.items():
                game._dice[p] = list(d)
        game.bid_history = [dict(b) for b in state.get("bid_history", [])]
        game.turn_number = state.get("turn_number", len(game.bid_history))
        game._winner = state.get("winner")
        game._terminal = bool(state.get("terminal", False))
        game._called_bid = state.get("called_bid")
        game._actual_count = state.get("actual_count_at_call")
        return game
