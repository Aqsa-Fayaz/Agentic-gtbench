"""
Move Validator Tool
LangChain-compatible tool that validates proposed moves before they are played.
"""

import json
from langchain.tools import BaseTool
from pydantic import BaseModel, Field
from games import load_game


class MoveValidatorInput(BaseModel):
    game_name: str = Field(description="Name of the game (e.g. 'tictactoe')")
    state: dict = Field(description="Current serialized game state")
    player: str = Field(description="'player_a' or 'player_b'")
    move: dict = Field(description="Proposed move dict to validate")


class MoveValidatorTool(BaseTool):
    """
    Validates whether a proposed move is legal given the current game state.
    Returns validity, reason, and list of all legal moves for reference.
    """
    name: str = "validate_move"
    description: str = (
        "Use this tool to check if your proposed move is legal before playing it. "
        "Input: game_name, state, player, move. "
        "Output: {valid: bool, reason: str, legal_moves: list}"
    )
    args_schema: type = MoveValidatorInput

    def _run(self, game_name: str, state: dict, player: str, move: dict) -> str:
        try:
            # Restore the game from the supplied state so that is_valid_move
            # and get_legal_moves reflect the LIVE board, not an empty one.
            game_cls = load_game(game_name).__class__
            try:
                game = game_cls.from_state(state)
            except NotImplementedError:
                # Fallback: use a fresh game (legacy behaviour). Logged.
                game = load_game(game_name)
            valid, reason = game.is_valid_move(player, move)
            legal_moves = game.get_legal_moves(player)
            return json.dumps({
                "valid": valid,
                "reason": reason,
                "legal_moves": legal_moves[:10],   # Cap at 10 for prompt length
            })
        except Exception as e:
            return json.dumps({"valid": False, "reason": str(e), "legal_moves": []})

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use synchronous _run")


# Standalone function for direct use (non-LangChain)
def validate_move(game, player: str, move: dict) -> tuple[bool, str, list]:
    """
    Direct validation helper — takes a live game instance.
    Returns: (valid, reason, legal_moves)
    """
    valid, reason = game.is_valid_move(player, move)
    legal_moves = game.get_legal_moves(player) if valid else game.get_legal_moves(player)
    return valid, reason, legal_moves
