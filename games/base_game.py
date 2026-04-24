"""
Abstract base class for all game environments.
Every game must implement this interface exactly.
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseGame(ABC):
    """
    Abstract game environment interface.
    All 5 game implementations must inherit from this class.
    """

    PLAYER_A = "player_a"
    PLAYER_B = "player_b"

    def __init__(self):
        self.turn_number = 0
        self._state = {}
        self.reset()

    @abstractmethod
    def reset(self) -> dict:
        """
        Reset game to initial state.
        Returns the initial state dict.
        """
        raise NotImplementedError

    @abstractmethod
    def get_state(self) -> dict:
        """
        Return a fully serializable snapshot of the current game state.
        Must include all information needed to reconstruct the game.
        Must NOT include opponent's hidden information (for imperfect info games).
        """
        raise NotImplementedError

    @abstractmethod
    def get_state_for_player(self, player: str) -> dict:
        """
        Return game state from a specific player's perspective.
        For perfect-information games: same as get_state().
        For imperfect-information games: hides opponent's private info.
        """
        raise NotImplementedError

    @abstractmethod
    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        """
        Check if a move is legal for the given player.
        Returns: (is_valid: bool, reason: str)
        """
        raise NotImplementedError

    @abstractmethod
    def make_move(self, player: str, move: dict) -> dict:
        """
        Apply a move to the game state.
        Assumes move has already been validated.
        Returns the updated state dict.
        Raises: ValueError if move is invalid (caller should validate first).
        """
        raise NotImplementedError

    @abstractmethod
    def is_terminal(self) -> bool:
        """Return True if the game has ended."""
        raise NotImplementedError

    @abstractmethod
    def get_winner(self) -> Optional[str]:
        """
        Return the winner identifier or special value.
        Returns: "player_a" | "player_b" | "draw" | None (if not terminal)
        """
        raise NotImplementedError

    @abstractmethod
    def get_legal_moves(self, player: str) -> list:
        """
        Return list of all legal moves for the given player.
        Each move is a dict matching the move format for make_move().
        """
        raise NotImplementedError

    @abstractmethod
    def render(self) -> str:
        """
        Return a human-readable string representation of the current game state.
        Used for debugging and including in LLM prompts.
        """
        raise NotImplementedError

    @property
    def name(self) -> str:
        """Game name — used for logging and tool dispatch."""
        return self.__class__.__name__.lower()

    def get_current_player(self) -> str:
        """Return which player should move next."""
        return self.PLAYER_A if self.turn_number % 2 == 0 else self.PLAYER_B

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(turn={self.turn_number}, terminal={self.is_terminal()})"
