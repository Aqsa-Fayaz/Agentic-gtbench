"""
Nim Game Environment (Misère variant)
Type: Perfect-information, deterministic combinatorial game
GTBench Category: Perfect-information combinatorial games

Misère rule: the player who takes the LAST object LOSES.
"""

from typing import Optional

from games.base_game import BaseGame


class Nim(BaseGame):
    """
    Classic Nim played over configurable piles.

    - Piles default to [3, 5, 7]
    - Move format: {"pile": int, "count": int}
        pile   -> 0-indexed pile selector
        count  -> number of objects to remove (>= 1 and <= pile size)
    - Misère variant: the player who takes the LAST object loses.
    """

    DEFAULT_PILES = [3, 5, 7]

    def __init__(self, piles: list = None):
        self._initial_piles = list(piles) if piles else list(self.DEFAULT_PILES)
        super().__init__()

    def reset(self) -> dict:
        self.piles = list(self._initial_piles)
        self.turn_number = 0
        self._winner = None
        self._last_move = None
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "piles": list(self.piles),
            "num_piles": len(self.piles),
            "total_remaining": sum(self.piles),
            "current_player": self.get_current_player(),
            "turn_number": self.turn_number,
            "winner": self._winner,
            "last_move": self._last_move,
            "variant": "misere",
        }

    def get_state_for_player(self, player: str) -> dict:
        return self.get_state()

    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        if player != self.get_current_player():
            return False, f"It is not {player}'s turn."
        if "pile" not in move or "count" not in move:
            return False, "Move must have both 'pile' and 'count' keys."
        pile = move.get("pile")
        count = move.get("count")
        if not (isinstance(pile, int) and isinstance(count, int)):
            return False, "'pile' and 'count' must be integers."
        if not (0 <= pile < len(self.piles)):
            return False, f"Pile index must be 0..{len(self.piles) - 1}. Got pile={pile}."
        if count < 1:
            return False, f"Must remove at least 1 object. Got count={count}."
        if count > self.piles[pile]:
            return False, (
                f"Pile {pile} has only {self.piles[pile]} object(s); "
                f"cannot remove {count}."
            )
        return True, "Valid move."

    def make_move(self, player: str, move: dict) -> dict:
        valid, reason = self.is_valid_move(player, move)
        if not valid:
            raise ValueError(f"Invalid move: {reason}")
        pile, count = move["pile"], move["count"]
        self.piles[pile] -= count
        self._last_move = {"pile": pile, "count": count, "player": player}
        self.turn_number += 1
        if sum(self.piles) == 0:
            # Misère: the player who emptied the last pile LOSES.
            self._winner = self.PLAYER_B if player == self.PLAYER_A else self.PLAYER_A
        return self.get_state()

    def is_terminal(self) -> bool:
        return sum(self.piles) == 0

    def get_winner(self) -> Optional[str]:
        if not self.is_terminal():
            return None
        return self._winner

    def get_legal_moves(self, player: str) -> list:
        if self.is_terminal() or player != self.get_current_player():
            return []
        moves = []
        for idx, size in enumerate(self.piles):
            for count in range(1, size + 1):
                moves.append({"pile": idx, "count": count})
        return moves

    @classmethod
    def from_state(cls, state: dict) -> "Nim":
        piles = state.get("piles") or list(cls.DEFAULT_PILES)
        game = cls(piles=piles)
        # Reset already used the supplied piles, but turn_number is what
        # determines current_player — restore it from the state dict.
        game.turn_number = state.get("turn_number", 0)
        game._winner = state.get("winner")
        game._last_move = state.get("last_move")
        return game

    def render(self) -> str:
        lines = [f"=== Nim (Misère) — Turn {self.turn_number} ==="]
        for idx, size in enumerate(self.piles):
            lines.append(f"  Pile {idx}: {'|' * size} ({size})")
        lines.append(f"Current player: {self.get_current_player()}")
        return "\n".join(lines)
