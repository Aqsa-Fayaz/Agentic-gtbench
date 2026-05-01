"""Game registry — maps name strings to game classes."""

from games.tictactoe import TicTacToe
from games.prisoners_dilemma import PrisonersDilemma
from games.connect4 import Connect4
from games.nim import Nim
from games.kuhn_poker import KuhnPoker
from games.pig import Pig
from games.blind_auction import BlindAuction
from games.liars_dice import LiarsDice
from games.breakthrough import Breakthrough
from games.negotiation import Negotiation

GAME_REGISTRY = {
    "tictactoe": TicTacToe,
    "prisoners_dilemma": PrisonersDilemma,
    "connect4": Connect4,
    "nim": Nim,
    "kuhn_poker": KuhnPoker,
    "pig": Pig,
    "blind_auction": BlindAuction,
    "liars_dice": LiarsDice,
    "breakthrough": Breakthrough,
    "negotiation": Negotiation,
}


def load_game(name: str, **kwargs):
    """
    Instantiate a game by name.

    Usage:
        game = load_game("tictactoe")
        game = load_game("prisoners_dilemma", num_rounds=10)
        game = load_game("nim", piles=[3, 5, 7])
        game = load_game("kuhn_poker", seed=42)
    """
    name = name.lower()
    if name not in GAME_REGISTRY:
        raise ValueError(f"Unknown game '{name}'. Available: {list(GAME_REGISTRY.keys())}")
    return GAME_REGISTRY[name](**kwargs)


__all__ = [
    "BaseGame",
    "TicTacToe",
    "PrisonersDilemma",
    "Connect4",
    "Nim",
    "KuhnPoker",
    "Pig",
    "BlindAuction",
    "LiarsDice",
    "Breakthrough",
    "Negotiation",
    "GAME_REGISTRY",
    "load_game",
]

from games.base_game import BaseGame  # placed at bottom to avoid circular imports
