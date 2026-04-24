"""
Connect-4 Game Environment
Type: Complete-information, deterministic, zero-sum
GTBench Category: Perfect-information games
"""

from typing import Optional

from games.base_game import BaseGame


class Connect4(BaseGame):
    """
    Standard 6-row x 7-column Connect-4.

    - player_a plays 'X', player_b plays 'O'
    - Move format: {"col": int}  (0..6)
    - Pieces fall to the lowest empty row in the chosen column (gravity)
    - Winner: first to align 4 pieces (horizontal, vertical, or diagonal)
    """

    ROWS = 6
    COLS = 7
    WIN_LEN = 4
    EMPTY = "."
    SYMBOL = {BaseGame.PLAYER_A: "X", BaseGame.PLAYER_B: "O"}

    def reset(self) -> dict:
        self.board = [[self.EMPTY] * self.COLS for _ in range(self.ROWS)]
        self.turn_number = 0
        self._winner = None
        self._last_move = None
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

    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        if player != self.get_current_player():
            return False, f"It is not {player}'s turn."
        if "col" not in move:
            return False, "Move must have a 'col' key."
        col = move.get("col")
        if not isinstance(col, int):
            return False, "Column must be an integer."
        if not (0 <= col < self.COLS):
            return False, f"Column must be 0..{self.COLS - 1}. Got col={col}."
        if self.board[0][col] != self.EMPTY:
            return False, f"Column {col} is full."
        return True, "Valid move."

    def make_move(self, player: str, move: dict) -> dict:
        valid, reason = self.is_valid_move(player, move)
        if not valid:
            raise ValueError(f"Invalid move: {reason}")
        col = move["col"]
        symbol = self.SYMBOL[player]
        row = self._drop_row(col)
        self.board[row][col] = symbol
        self._last_move = {"row": row, "col": col, "player": player}
        self.turn_number += 1
        if self._check_winner_at(row, col, symbol):
            self._winner = player
        return self.get_state()

    def is_terminal(self) -> bool:
        if self._winner is not None:
            return True
        return all(self.board[0][c] != self.EMPTY for c in range(self.COLS))

    def get_winner(self) -> Optional[str]:
        if not self.is_terminal():
            return None
        if self._winner is None:
            return "draw"
        return self._winner

    def get_legal_moves(self, player: str) -> list:
        if self.is_terminal() or player != self.get_current_player():
            return []
        return [{"col": c} for c in range(self.COLS) if self.board[0][c] == self.EMPTY]

    def render(self) -> str:
        header = "  " + " ".join(str(c) for c in range(self.COLS))
        rows = [f"  {' '.join(self.board[r])}" for r in range(self.ROWS)]
        return "\n".join([header] + rows)

    def _drop_row(self, col: int) -> int:
        for r in range(self.ROWS - 1, -1, -1):
            if self.board[r][col] == self.EMPTY:
                return r
        raise ValueError(f"Column {col} is full (should have been caught in validation).")

    def _check_winner_at(self, row: int, col: int, symbol: str) -> bool:
        directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
        for dr, dc in directions:
            count = 1
            count += self._count_direction(row, col, dr, dc, symbol)
            count += self._count_direction(row, col, -dr, -dc, symbol)
            if count >= self.WIN_LEN:
                return True
        return False

    def _count_direction(self, row: int, col: int, dr: int, dc: int, symbol: str) -> int:
        count = 0
        r, c = row + dr, col + dc
        while 0 <= r < self.ROWS and 0 <= c < self.COLS and self.board[r][c] == symbol:
            count += 1
            r += dr
            c += dc
        return count
