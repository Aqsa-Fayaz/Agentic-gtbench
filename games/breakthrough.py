"""
Breakthrough Game Environment
Type: Complete-information, deterministic, zero-sum
GTBench Category: Complete & deterministic gaming (paper Table 1)

6×6 board. Each side starts with 12 pieces filling the two rows nearest them.
- player_a (symbol 'A') starts on rows 0-1, advances DOWN (toward row 5)
- player_b (symbol 'B') starts on rows 4-5, advances UP   (toward row 0)

Each turn the active player moves ONE of their pieces by exactly one square:
  * Forward straight   — must land on empty square (no straight captures)
  * Forward diagonal   — may land on empty square OR capture an enemy piece
                         (diagonal captures only)
A player may not move onto their own piece.

Terminal conditions:
  * Any of player_a's pieces reaches row 5    →  player_a wins
  * Any of player_b's pieces reaches row 0    →  player_b wins
  * The active player has no legal moves      →  the OTHER player wins
"""

from __future__ import annotations

from typing import Optional

from games.base_game import BaseGame


class Breakthrough(BaseGame):

    ROWS = 6
    COLS = 6
    EMPTY = "."
    SYMBOL = {BaseGame.PLAYER_A: "A", BaseGame.PLAYER_B: "B"}
    DIRECTION = {BaseGame.PLAYER_A: 1, BaseGame.PLAYER_B: -1}     # row delta to advance

    def __init__(self):
        super().__init__()

    # ─────────────────────────────────────────────
    # State
    # ─────────────────────────────────────────────

    def reset(self) -> dict:
        self.board = [[self.EMPTY] * self.COLS for _ in range(self.ROWS)]
        for c in range(self.COLS):
            self.board[0][c] = self.SYMBOL[self.PLAYER_A]
            self.board[1][c] = self.SYMBOL[self.PLAYER_A]
            self.board[self.ROWS - 1][c] = self.SYMBOL[self.PLAYER_B]
            self.board[self.ROWS - 2][c] = self.SYMBOL[self.PLAYER_B]
        self.turn_number = 0
        self._winner: Optional[str] = None
        self._last_move: Optional[dict] = None
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "board": [row[:] for row in self.board],
            "rows": self.ROWS,
            "cols": self.COLS,
            "current_player": self.get_current_player(),
            "turn_number": self.turn_number,
            "winner": self._winner,
            "last_move": self._last_move,
        }

    def get_state_for_player(self, player: str) -> dict:
        return self.get_state()

    # ─────────────────────────────────────────────
    # Move validation + application
    # ─────────────────────────────────────────────

    def _parse_move(self, move: dict) -> Optional[tuple[int, int, int, int]]:
        try:
            f = move["from"]
            t = move["to"]
            return int(f["row"]), int(f["col"]), int(t["row"]), int(t["col"])
        except (KeyError, TypeError, ValueError):
            return None

    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        if self._winner is not None:
            return False, "Game already over."
        if player != self.get_current_player():
            return False, f"It is not {player}'s turn."
        parsed = self._parse_move(move)
        if parsed is None:
            return False, "Move must contain {'from': {row, col}, 'to': {row, col}}."
        fr, fc, tr, tc = parsed
        if not (0 <= fr < self.ROWS and 0 <= fc < self.COLS):
            return False, f"From ({fr},{fc}) is off-board."
        if not (0 <= tr < self.ROWS and 0 <= tc < self.COLS):
            return False, f"To ({tr},{tc}) is off-board."

        my_sym = self.SYMBOL[player]
        if self.board[fr][fc] != my_sym:
            return False, f"From ({fr},{fc}) is not your piece."

        opp_sym = self.SYMBOL[self.PLAYER_B if player == self.PLAYER_A else self.PLAYER_A]
        target = self.board[tr][tc]
        if target == my_sym:
            return False, f"To ({tr},{tc}) is occupied by your own piece."

        forward = self.DIRECTION[player]
        dr = tr - fr
        dc = tc - fc
        if dr != forward:
            return False, f"Pieces only move {abs(forward)} square forward."
        if abs(dc) > 1:
            return False, "Pieces only move forward straight or one diagonal."

        if dc == 0 and target == opp_sym:
            return False, "Straight moves cannot capture; only diagonals can."

        return True, "Valid."

    def make_move(self, player: str, move: dict) -> dict:
        valid, reason = self.is_valid_move(player, move)
        if not valid:
            raise ValueError(f"Invalid move: {reason}")

        fr, fc, tr, tc = self._parse_move(move)
        my_sym = self.SYMBOL[player]
        self.board[tr][tc] = my_sym
        self.board[fr][fc] = self.EMPTY
        self.turn_number += 1
        self._last_move = {"from": {"row": fr, "col": fc}, "to": {"row": tr, "col": tc}, "player": player}

        # Win by reaching opponent's back row.
        target_row = self.ROWS - 1 if player == self.PLAYER_A else 0
        if tr == target_row:
            self._winner = player
            return self.get_state()

        # Win by stalemate: opponent has no legal moves.
        opponent = self.PLAYER_B if player == self.PLAYER_A else self.PLAYER_A
        if not self.get_legal_moves(opponent):
            self._winner = player

        return self.get_state()

    # ─────────────────────────────────────────────
    # Terminal / winner / legal moves
    # ─────────────────────────────────────────────

    def is_terminal(self) -> bool:
        return self._winner is not None

    def get_winner(self) -> Optional[str]:
        return self._winner

    def get_legal_moves(self, player: str) -> list:
        if self._winner is not None:
            return []
        my_sym = self.SYMBOL[player]
        opp_sym = self.SYMBOL[self.PLAYER_B if player == self.PLAYER_A else self.PLAYER_A]
        forward = self.DIRECTION[player]
        moves: list[dict] = []
        for r in range(self.ROWS):
            for c in range(self.COLS):
                if self.board[r][c] != my_sym:
                    continue
                tr = r + forward
                if not (0 <= tr < self.ROWS):
                    continue
                # Straight: only to empty
                if self.board[tr][c] == self.EMPTY:
                    moves.append({"from": {"row": r, "col": c}, "to": {"row": tr, "col": c}})
                # Diagonals: empty or capture
                for dc in (-1, 1):
                    tc = c + dc
                    if not (0 <= tc < self.COLS):
                        continue
                    target = self.board[tr][tc]
                    if target == my_sym:
                        continue
                    moves.append({"from": {"row": r, "col": c}, "to": {"row": tr, "col": tc}})
        return moves

    # ─────────────────────────────────────────────
    # Render + restore
    # ─────────────────────────────────────────────

    def render(self) -> str:
        header = "  " + " ".join(str(c) for c in range(self.COLS))
        rows = [f"{r} {' '.join(self.board[r])}" for r in range(self.ROWS)]
        return "\n".join([header] + rows)

    @classmethod
    def from_state(cls, state: dict) -> "Breakthrough":
        game = cls()
        board = state.get("board")
        if board:
            game.board = [list(row) for row in board]
        game.turn_number = state.get("turn_number", 0)
        game._winner = state.get("winner")
        game._last_move = state.get("last_move")
        return game
