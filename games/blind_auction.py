"""
Blind Auction (First-Price Sealed-Bid)
Type: Incomplete-information, simultaneous-move, non-zero-sum
GTBench Category: Incomplete & probabilistic gaming (paper Table 1)

Each player is assigned a private valuation v_i in [v_min, v_max].
Both players submit a sealed bid b_i ∈ [0, max_bid] simultaneously.
- The higher bid wins the item.
- Winner's payoff = v_winner − b_winner (if positive; otherwise win still
  resolves but payoff may be negative — risk of overbidding).
- Loser's payoff = 0.
- Ties → split the item: each gets (v_self − b_self) / 2.

Move format: {"bid": int}

The two bids are submitted sequentially in code, but Player B does not see
Player A's bid before committing — preserving simultaneity.
"""

from __future__ import annotations

import random
from typing import Optional

from games.base_game import BaseGame


class BlindAuction(BaseGame):

    def __init__(
        self,
        valuation_range: tuple[int, int] = (10, 100),
        max_bid: int = 100,
        seed: Optional[int] = None,
    ):
        self.valuation_range = (int(valuation_range[0]), int(valuation_range[1]))
        self.max_bid = int(max_bid)
        self._rng = random.Random(seed)
        super().__init__()

    # ─────────────────────────────────────────────
    # State
    # ─────────────────────────────────────────────

    def reset(self) -> dict:
        lo, hi = self.valuation_range
        self._valuations = {
            self.PLAYER_A: self._rng.randint(lo, hi),
            self.PLAYER_B: self._rng.randint(lo, hi),
        }
        self.turn_number = 0
        self._pending_a_bid: Optional[int] = None
        self._bids: dict = {}
        self.payoffs: dict = {self.PLAYER_A: 0, self.PLAYER_B: 0}
        self._winner: Optional[str] = None
        self._terminal = False
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "valuations": dict(self._valuations),       # full info — only used by tests
            "max_bid": self.max_bid,
            "valuation_range": list(self.valuation_range),
            "current_player": self.get_current_player(),
            "turn_number": self.turn_number,
            "bids": dict(self._bids),
            "payoffs": dict(self.payoffs),
            "winner": self._winner,
            "terminal": self._terminal,
        }

    def get_state_for_player(self, player: str) -> dict:
        """
        Hide opponent's valuation AND opponent's pending bid (simultaneity).
        """
        state = self.get_state()
        opp = self.PLAYER_B if player == self.PLAYER_A else self.PLAYER_A
        # Replace full valuations with only the player's own.
        state["my_valuation"] = self._valuations[player]
        state["valuations"] = {player: self._valuations[player]}
        # Hide opponent's bid until terminal.
        if not self._terminal:
            state["bids"] = {p: b for p, b in self._bids.items() if p == player}
        return state

    # ─────────────────────────────────────────────
    # Move validation + application
    # ─────────────────────────────────────────────

    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        if self._terminal:
            return False, "Auction already settled."
        if player != self.get_current_player():
            return False, f"It is not {player}'s turn."
        if "bid" not in move:
            return False, "Move must have a 'bid' key."
        bid = move.get("bid")
        if not isinstance(bid, int):
            return False, f"Bid must be an integer. Got: {bid!r}"
        if bid < 0 or bid > self.max_bid:
            return False, f"Bid must be in [0, {self.max_bid}]. Got: {bid}"
        return True, "Valid."

    def make_move(self, player: str, move: dict) -> dict:
        valid, reason = self.is_valid_move(player, move)
        if not valid:
            raise ValueError(f"Invalid move: {reason}")

        bid = int(move["bid"])
        self._bids[player] = bid
        self.turn_number += 1

        if player == self.PLAYER_A:
            self._pending_a_bid = bid
        else:
            # Player B has bid; resolve the auction.
            self._resolve()

        return self.get_state()

    def _resolve(self) -> None:
        ba = self._bids[self.PLAYER_A]
        bb = self._bids[self.PLAYER_B]
        va = self._valuations[self.PLAYER_A]
        vb = self._valuations[self.PLAYER_B]

        if ba > bb:
            self.payoffs[self.PLAYER_A] = va - ba
            self.payoffs[self.PLAYER_B] = 0
            self._winner = self.PLAYER_A
        elif bb > ba:
            self.payoffs[self.PLAYER_B] = vb - bb
            self.payoffs[self.PLAYER_A] = 0
            self._winner = self.PLAYER_B
        else:
            # Tie — split.
            self.payoffs[self.PLAYER_A] = (va - ba) // 2
            self.payoffs[self.PLAYER_B] = (vb - bb) // 2
            # For the win-rate metric we still need a label; treat ties as draw.
            self._winner = "draw"

        # Apply non-zero-sum winner-by-payoff override only when both bids
        # were below valuation (so payoff was actually positive).
        self._terminal = True

    # ─────────────────────────────────────────────
    # Terminal / winner
    # ─────────────────────────────────────────────

    def is_terminal(self) -> bool:
        return self._terminal

    def get_winner(self) -> Optional[str]:
        return self._winner if self._terminal else None

    def get_legal_moves(self, player: str) -> list:
        if self._terminal or player != self.get_current_player():
            return []
        # Bid space is large; expose 11 evenly-spaced anchor bids so the LLM
        # has discrete options without enumerating 0..100.
        step = max(1, self.max_bid // 10)
        return [{"bid": b} for b in range(0, self.max_bid + 1, step)]

    def render(self) -> str:
        lines = [
            f"=== Blind Auction (max_bid={self.max_bid}) ===",
            f"  Active: {self.get_current_player()}  Turn: {self.turn_number}",
        ]
        for p in (self.PLAYER_A, self.PLAYER_B):
            v = self._valuations.get(p)
            b = self._bids.get(p)
            lines.append(f"  {p}: valuation={v}  bid={b}")
        if self._terminal:
            lines.append(f"  Outcome: winner={self._winner} payoffs={self.payoffs}")
        return "\n".join(lines)

    # ─────────────────────────────────────────────
    # State restoration
    # ─────────────────────────────────────────────

    @classmethod
    def from_state(cls, state: dict) -> "BlindAuction":
        valuation_range = tuple(state.get("valuation_range", (10, 100)))
        game = cls(valuation_range=valuation_range, max_bid=state.get("max_bid", 100))
        # Restore valuations if present (full state); otherwise leave random.
        full_vals = state.get("valuations") or {}
        if isinstance(full_vals, dict) and len(full_vals) >= 1:
            for p, v in full_vals.items():
                game._valuations[p] = int(v)
        game.turn_number = state.get("turn_number", 0)
        game._bids = dict(state.get("bids", {}))
        game.payoffs = dict(state.get("payoffs", {cls.PLAYER_A: 0, cls.PLAYER_B: 0}))
        game._winner = state.get("winner")
        game._terminal = bool(state.get("terminal", False))
        return game
