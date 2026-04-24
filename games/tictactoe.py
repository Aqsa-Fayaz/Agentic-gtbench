"""
Tic-Tac-Toe Game Environment
Type: Complete-information, deterministic, zero-sum
GTBench Category: Perfect-information games
"""

from typing import Optional
from games.base_game import BaseGame


class TicTacToe(BaseGame):
    """
    Standard 3×3 Tic-Tac-Toe.
    - player_a plays 'X', player_b plays 'O'
    - Move format: {"row": int, "col": int}  (both 0-indexed)
    - Winner: first to get 3 in a row/col/diagonal
    """

    SYMBOL = {BaseGame.PLAYER_A: "X", BaseGame.PLAYER_B: "O"}
    EMPTY = "."

    def reset(self) -> dict:
        self.board = [[self.EMPTY] * 3 for _ in range(3)]
        self.turn_number = 0
        self._winner = None
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "board": [row[:] for row in self.board],
            "current_player": self.get_current_player(),
            "turn_number": self.turn_number,
            "winner": self._winner,
        }

    def get_state_for_player(self, player: str) -> dict:
        # Perfect information — same as full state
        return self.get_state()

    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        if player != self.get_current_player():
            return False, f"It is not {player}'s turn."
        if "row" not in move or "col" not in move:
            return False, "Move must have 'row' and 'col' keys."
        row, col = move.get("row"), move.get("col")
        if not (isinstance(row, int) and isinstance(col, int)):
            return False, "Row and col must be integers."
        if not (0 <= row <= 2 and 0 <= col <= 2):
            return False, f"Row and col must be 0–2. Got row={row}, col={col}."
        if self.board[row][col] != self.EMPTY:
            return False, f"Cell ({row},{col}) is already occupied."
        return True, "Valid move."

    def make_move(self, player: str, move: dict) -> dict:
        valid, reason = self.is_valid_move(player, move)
        if not valid:
            raise ValueError(f"Invalid move: {reason}")
        row, col = move["row"], move["col"]
        self.board[row][col] = self.SYMBOL[player]
        self.turn_number += 1
        self._winner = self._check_winner()
        return self.get_state()

    def is_terminal(self) -> bool:
        return self._winner is not None or all(
            self.board[r][c] != self.EMPTY for r in range(3) for c in range(3)
        )

    def get_winner(self) -> Optional[str]:
        if not self.is_terminal():
            return None
        if self._winner is None:
            return "draw"
        return self._winner

    def get_legal_moves(self, player: str) -> list:
        if self.is_terminal() or player != self.get_current_player():
            return []
        return [
            {"row": r, "col": c}
            for r in range(3)
            for c in range(3)
            if self.board[r][c] == self.EMPTY
        ]

    def render(self) -> str:
        header = "  0 1 2"
        rows = [f"{i} {' '.join(self.board[i])}" for i in range(3)]
        return "\n".join([header] + rows)

    def _check_winner(self) -> Optional[str]:
        b = self.board
        lines = (
            # Rows
            [b[0], b[1], b[2]] +
            # Cols
            [[b[r][c] for r in range(3)] for c in range(3)] +
            # Diagonals
            [[b[i][i] for i in range(3)], [b[i][2 - i] for i in range(3)]]
        )
        for line in lines:
            if line[0] != self.EMPTY and all(v == line[0] for v in line):
                # Find which player has this symbol
                for player, symbol in self.SYMBOL.items():
                    if line[0] == symbol:
                        return player
        return None
