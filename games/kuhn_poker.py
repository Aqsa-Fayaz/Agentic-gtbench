"""
Kuhn Poker Game Environment
Type: Imperfect-information, stochastic, zero-sum
GTBench Category: Imperfect-information games

3-card deck (J=1, Q=2, K=3). Each player antes 1 chip and is dealt one private
card. Players alternate, each choosing to `pass` or `bet`. The game terminates
at a showdown (compare cards) or when a player folds after the opponent bets.
"""

import random
from typing import Optional

from games.base_game import BaseGame


class KuhnPoker(BaseGame):
    """
    Simplified Kuhn Poker.

    Actions: {"action": "pass"}  or  {"action": "bet"}

    Terminal sequences (A acts first):
      pp             -> showdown for pot 2 (ante only)
      pbp            -> A folds,  B wins pot 2
      pbb            -> showdown for pot 4 (both bet)
      bp             -> B folds,  A wins pot 2
      bb             -> showdown for pot 4

    Each player antes 1 before play. A `bet` adds 1 chip to the pot.
    """

    PASS = "pass"
    BET = "bet"
    VALID_ACTIONS = {PASS, BET}
    DECK = ["J", "Q", "K"]
    CARD_RANK = {"J": 1, "Q": 2, "K": 3}

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)
        super().__init__()

    def reset(self) -> dict:
        deck = list(self.DECK)
        self._rng.shuffle(deck)
        self._cards = {self.PLAYER_A: deck[0], self.PLAYER_B: deck[1]}
        self._unseen_card = deck[2]
        self.action_history: list[dict] = []
        self.pot = 2
        self._player_chips = {self.PLAYER_A: -1, self.PLAYER_B: -1}
        self.turn_number = 0
        self._winner = None
        self._terminal = False
        self._showdown = False
        self._folded_by: Optional[str] = None
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "pot": self.pot,
            "action_history": [dict(a) for a in self.action_history],
            "turn_number": self.turn_number,
            "current_player": self.get_current_player(),
            "winner": self._winner,
            "terminal": self._terminal,
            "showdown": self._showdown,
            "folded_by": self._folded_by,
            "player_chips": dict(self._player_chips),
        }

    def get_state_for_player(self, player: str) -> dict:
        """
        Hide opponent's private card. Reveal opponent card only at showdown.
        """
        state = self.get_state()
        state["my_card"] = self._cards[player]
        if self._terminal and self._showdown:
            opponent = self.PLAYER_B if player == self.PLAYER_A else self.PLAYER_A
            state["opponent_card"] = self._cards[opponent]
        else:
            state["opponent_card"] = None
        return state

    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        if self._terminal:
            return False, "Game already over."
        if "action" not in move:
            return False, "Move must have an 'action' key."
        action = move["action"]
        if action not in self.VALID_ACTIONS:
            return False, f"Action must be 'pass' or 'bet'. Got: {action}"
        if player != self.get_current_player():
            return False, f"It is not {player}'s turn."
        return True, "Valid."

    def make_move(self, player: str, move: dict) -> dict:
        valid, reason = self.is_valid_move(player, move)
        if not valid:
            raise ValueError(f"Invalid move: {reason}")

        action = move["action"]
        self.action_history.append({"player": player, "action": action})
        self.turn_number += 1

        if action == self.BET:
            self.pot += 1
            self._player_chips[player] -= 1

        self._resolve_if_terminal(player, action)
        return self.get_state()

    def is_terminal(self) -> bool:
        return self._terminal

    def get_winner(self) -> Optional[str]:
        if not self._terminal:
            return None
        return self._winner

    def get_legal_moves(self, player: str) -> list:
        if self._terminal or player != self.get_current_player():
            return []
        return [{"action": self.PASS}, {"action": self.BET}]

    def render(self) -> str:
        lines = ["=== Kuhn Poker ==="]
        lines.append(f"Pot: {self.pot}")
        lines.append("Action history:")
        for a in self.action_history:
            lines.append(f"  {a['player']}: {a['action']}")
        lines.append(f"Current player: {self.get_current_player()}")
        if self._terminal:
            lines.append(f"Winner: {self._winner} (showdown={self._showdown})")
        return "\n".join(lines)

    @classmethod
    def from_state(cls, state: dict) -> "KuhnPoker":
        """
        Restore from a state dict. Note: a state taken from
        get_state_for_player() hides the opponent's card; the reconstructed
        game won't be able to do showdown logic correctly until a state
        from get_state() (with both cards) is provided. For move-validation
        purposes (which only needs current_player + action_history), this is
        sufficient.
        """
        game = cls()
        game.action_history = list(state.get("action_history", []))
        game.turn_number = state.get("turn_number", len(game.action_history))
        game.pot = state.get("pot", 2)
        game._terminal = state.get("terminal", False)
        game._winner = state.get("winner")
        game._showdown = state.get("showdown", False)
        game._folded_by = state.get("folded_by")
        # Cards: prefer my_card / opponent_card if present, else keep dealt cards.
        if "my_card" in state and "opponent_card" in state:
            # We don't know which player this state was taken from without
            # more info — leave the dealt cards in place and let validation
            # work from action_history.
            pass
        return game

    def _resolve_if_terminal(self, mover: str, action: str) -> None:
        """
        Inspect action history and decide if the hand has ended.
        Sequences (A first): pp, pbp, pbb, bp, bb.
        """
        actions = [a["action"] for a in self.action_history]
        players = [a["player"] for a in self.action_history]

        def finish(winner: str, showdown: bool, folded_by: Optional[str] = None) -> None:
            self._terminal = True
            self._winner = winner
            self._showdown = showdown
            self._folded_by = folded_by
            if winner == self.PLAYER_A:
                self._player_chips[self.PLAYER_A] += self.pot
            elif winner == self.PLAYER_B:
                self._player_chips[self.PLAYER_B] += self.pot
            else:
                half = self.pot / 2
                self._player_chips[self.PLAYER_A] += half
                self._player_chips[self.PLAYER_B] += half

        if actions == [self.PASS, self.PASS]:
            finish(self._higher_card_player(), showdown=True)
        elif actions == [self.PASS, self.BET, self.PASS]:
            finish(self.PLAYER_B, showdown=False, folded_by=self.PLAYER_A)
        elif actions == [self.PASS, self.BET, self.BET]:
            finish(self._higher_card_player(), showdown=True)
        elif actions == [self.BET, self.PASS]:
            finish(self.PLAYER_A, showdown=False, folded_by=self.PLAYER_B)
        elif actions == [self.BET, self.BET]:
            finish(self._higher_card_player(), showdown=True)

    def _higher_card_player(self) -> str:
        a_rank = self.CARD_RANK[self._cards[self.PLAYER_A]]
        b_rank = self.CARD_RANK[self._cards[self.PLAYER_B]]
        if a_rank > b_rank:
            return self.PLAYER_A
        if b_rank > a_rank:
            return self.PLAYER_B
        return "draw"
