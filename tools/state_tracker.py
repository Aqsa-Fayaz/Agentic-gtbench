"""
State Tracker Tool.

Converts a raw game state dict into a natural-language description for the LLM
and extracts simple strategic features (threats / opportunities). Usable as a
LangChain tool (inside ReAct loops) or as a plain helper class.
"""

from __future__ import annotations

import json
from typing import Any

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from games import load_game


class StateTrackerInput(BaseModel):
    game_name: str = Field(description="Name of the game (e.g. 'tictactoe', 'connect4')")
    state: dict = Field(description="Serialized game state from game.get_state_for_player()")
    player: str = Field(description="'player_a' or 'player_b' — whose perspective to describe")


class StateTrackerTool(BaseTool):
    """
    Describe the current game state in natural language and surface obvious
    threats / opportunities, so the agent can reason more effectively.
    """

    name: str = "describe_state"
    description: str = (
        "Translate the raw game state into a natural-language description and "
        "list any immediate threats and opportunities for the requested player. "
        "Input: game_name, state, player. "
        'Output: {"description": str, "threats": list, "opportunities": list}'
    )
    args_schema: type = StateTrackerInput

    def _run(self, game_name: str, state: dict, player: str) -> str:
        try:
            description = self._describe(game_name, state, player)
            threats = self._find_threats(game_name, state, player)
            opportunities = self._find_opportunities(game_name, state, player)
            return json.dumps(
                {
                    "description": description,
                    "threats": threats,
                    "opportunities": opportunities,
                },
                default=str,
            )
        except Exception as exc:  # pragma: no cover - tool must be robust
            return json.dumps(
                {"description": f"[tracker error: {exc}]", "threats": [], "opportunities": []}
            )

    async def _arun(self, *args, **kwargs):
        raise NotImplementedError("Use synchronous _run")

    @staticmethod
    def _describe(game_name: str, state: dict, player: str) -> str:
        game_name = game_name.lower()
        if game_name == "tictactoe":
            rows = state.get("board", [])
            grid = "\n".join(" ".join(r) for r in rows)
            return (
                f"Tic-Tac-Toe, turn {state.get('turn_number', 0)}. "
                f"You are {player}.\nBoard:\n{grid}"
            )
        if game_name == "connect4":
            rows = state.get("board", [])
            grid = "\n".join(" ".join(r) for r in rows)
            return (
                f"Connect-4, turn {state.get('turn_number', 0)}. You are {player}. "
                f"Drop pieces into columns 0..6.\nBoard (top row is row 0):\n{grid}"
            )
        if game_name == "nim":
            piles = state.get("piles", [])
            return (
                f"Nim (misère) — piles: {piles}. You are {player}. "
                "Remove any positive count from one pile. The player who takes "
                "the LAST object loses."
            )
        if game_name == "prisoners_dilemma":
            pay = state.get("cumulative_payoffs", {})
            hist = state.get("round_history", [])
            return (
                f"Iterated Prisoner's Dilemma, round "
                f"{state.get('current_round', 1)}/{state.get('num_rounds', '?')}. "
                f"You are {player}. Cumulative payoffs: {pay}. "
                f"Rounds played so far: {len(hist)}."
            )
        if game_name == "kuhn_poker":
            return (
                f"Kuhn Poker. Your card: {state.get('my_card')}. Pot: {state.get('pot')}. "
                f"Actions so far: {state.get('action_history', [])}."
            )
        if game_name == "pig":
            return (
                f"Pig (push-your-luck), target {state.get('target_score', 100)}. "
                f"You are {player}. Scores: {state.get('scores')}. "
                f"Current turn_total: {state.get('turn_total', 0)}. "
                f"Last roll: {state.get('last_roll')}. "
                "Choose 'roll' (gain die value, but a 1 wipes your turn) "
                "or 'hold' (bank turn_total, opponent moves)."
            )
        if game_name == "blind_auction":
            return (
                f"Blind Auction. You are {player}. Your private valuation: "
                f"{state.get('my_valuation')}. Max bid: {state.get('max_bid')}. "
                "Bids are simultaneous and sealed; opponent's bid is hidden. "
                "Higher bid wins and pays own bid; payoff = valuation − bid."
            )
        if game_name == "liars_dice":
            return (
                f"Liar's Dice. You are {player}. Your dice: {state.get('my_dice')}. "
                f"Dice per player: {state.get('dice_per_player')}. "
                f"Bid history: {state.get('bid_history')}. "
                "Each new bid must strictly raise the previous (higher face, "
                "or same face with higher count). 'call' challenges the last "
                "bid: count actual dice across both hands at that face; bidder "
                "wins if count met, else caller wins."
            )
        if game_name == "breakthrough":
            board = state.get("board", [])
            grid = "\n".join(" ".join(r) for r in board)
            forward = "DOWN (toward row 5)" if player == "player_a" else "UP (toward row 0)"
            return (
                f"Breakthrough 6x6, turn {state.get('turn_number', 0)}. You are "
                f"{player} (symbol {'A' if player == 'player_a' else 'B'}); "
                f"you advance {forward}. Move ONE piece one square forward "
                "straight (empty only) or one diagonal (empty or capture). "
                "Win by reaching the opponent's back row, or by leaving them "
                f"with no legal moves.\nBoard:\n{grid}"
            )
        if game_name == "negotiation":
            return (
                f"Negotiation, turn {state.get('turn_number', 0)}/"
                f"{state.get('max_turns', '?')}. You are {player}. "
                f"Pool: {state.get('pool')}. Your private valuations: "
                f"{state.get('my_valuations')}. Proposal history: "
                f"{state.get('proposal_history')}. Actions: 'propose' a split "
                "(for_me dict), 'accept' opponent's last proposal, or "
                "'walk_away' (both get 0)."
            )
        return f"{game_name} state: {json.dumps(state, default=str)[:300]}"

    @staticmethod
    def _find_threats(game_name: str, state: dict, player: str) -> list[str]:
        """Return a list of textual threat descriptions (opponent near-wins)."""
        game_name = game_name.lower()
        if game_name == "tictactoe":
            return StateTrackerTool._tictactoe_lines(state, player, threats=True)
        if game_name == "connect4":
            return StateTrackerTool._connect4_threats(state, player)
        if game_name == "prisoners_dilemma":
            hist = state.get("round_history", [])
            opp_key = "player_b_action" if player == "player_a" else "player_a_action"
            last = hist[-1][opp_key] if hist else None
            if last == "defect":
                return ["Opponent defected last round — retaliation may be warranted."]
            return []
        return []

    @staticmethod
    def _find_opportunities(game_name: str, state: dict, player: str) -> list[str]:
        game_name = game_name.lower()
        if game_name == "tictactoe":
            return StateTrackerTool._tictactoe_lines(state, player, threats=False)
        if game_name == "prisoners_dilemma":
            hist = state.get("round_history", [])
            if not hist:
                return ["No prior history — cooperating invites cooperation."]
            return []
        return []

    @staticmethod
    def _tictactoe_lines(state: dict, player: str, threats: bool) -> list[str]:
        """Detect 2-in-a-row threats or opportunities for TicTacToe."""
        board = state.get("board")
        if not board:
            return []
        me = "X" if player == "player_a" else "O"
        opp = "O" if me == "X" else "X"
        target = opp if threats else me
        lines: list[str] = []
        triples: list[tuple[tuple[int, int], ...]] = []
        for r in range(3):
            triples.append(tuple((r, c) for c in range(3)))
        for c in range(3):
            triples.append(tuple((r, c) for r in range(3)))
        triples.append(tuple((i, i) for i in range(3)))
        triples.append(tuple((i, 2 - i) for i in range(3)))
        for cells in triples:
            symbols = [board[r][c] for r, c in cells]
            if symbols.count(target) == 2 and symbols.count(".") == 1:
                empty = cells[symbols.index(".")]
                label = "threat" if threats else "opportunity"
                lines.append(f"{label}: line {cells} — play {{'row': {empty[0]}, 'col': {empty[1]}}}")
        return lines

    @staticmethod
    def _connect4_threats(state: dict, player: str) -> list[str]:
        board = state.get("board")
        if not board:
            return []
        opp_sym = "O" if player == "player_a" else "X"
        rows = len(board)
        cols = len(board[0])
        threats: list[str] = []
        for r in range(rows):
            for c in range(cols - 3):
                row_slice = [board[r][c + i] for i in range(4)]
                if row_slice.count(opp_sym) == 3 and row_slice.count(".") == 1:
                    threats.append(
                        f"threat: opponent has 3-in-a-row starting at ({r},{c})"
                    )
                    break
        return threats


def describe_state(game_or_name: Any, state: dict, player: str) -> dict:
    """
    Non-LangChain helper.
    Accepts either a game name string or a BaseGame-like object (for convenience).
    """
    if isinstance(game_or_name, str):
        name = game_or_name
    else:
        name = getattr(game_or_name, "name", None) or load_game(str(game_or_name)).name
    tool = StateTrackerTool()
    return json.loads(tool._run(game_name=name, state=state, player=player))
