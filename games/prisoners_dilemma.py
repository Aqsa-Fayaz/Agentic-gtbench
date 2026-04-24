"""
Iterated Prisoner's Dilemma Game Environment
Type: Repeated game, complete information (after each round)
GTBench Category: Repeated strategic interaction
Evaluates: Cooperation, equilibrium behavior, Pareto efficiency
"""

from typing import Optional
from games.base_game import BaseGame


class PrisonersDilemma(BaseGame):
    """
    Iterated Prisoner's Dilemma over N rounds.
    
    Payoff matrix (row=player_a, col=player_b):
        Cooperate   Defect
    C   (3, 3)     (0, 5)
    D   (5, 0)     (1, 1)
    
    Move format: {"action": "cooperate"} or {"action": "defect"}
    Both players choose simultaneously each round (simulated as sequential, 
    but player_b does NOT see player_a's choice before committing).
    """

    COOPERATE = "cooperate"
    DEFECT = "defect"
    PAYOFFS = {
        (COOPERATE, COOPERATE): (3, 3),
        (COOPERATE, DEFECT):    (0, 5),
        (DEFECT, COOPERATE):    (5, 0),
        (DEFECT, DEFECT):       (1, 1),
    }
    VALID_ACTIONS = {COOPERATE, DEFECT}

    def __init__(self, num_rounds: int = 10):
        self.num_rounds = num_rounds
        super().__init__()

    def reset(self) -> dict:
        self.turn_number = 0
        self.current_round = 1
        self.round_history = []          # [{round, a_action, b_action, a_payoff, b_payoff}]
        self._pending_a_action = None    # Player A acts first (hidden from B)
        self.cumulative_payoffs = {self.PLAYER_A: 0, self.PLAYER_B: 0}
        self._winner = None
        return self.get_state()

    def get_state(self) -> dict:
        return {
            "current_round": self.current_round,
            "num_rounds": self.num_rounds,
            "round_history": list(self.round_history),
            "cumulative_payoffs": dict(self.cumulative_payoffs),
            "turn_number": self.turn_number,
            "current_player": self.get_current_player(),
        }

    def get_state_for_player(self, player: str) -> dict:
        """
        Player B cannot see Player A's pending action for the current round —
        preserving simultaneity semantics.
        """
        state = self.get_state()
        if player == self.PLAYER_B:
            state["pending_a_action"] = None   # Hidden
        else:
            state["pending_a_action"] = self._pending_a_action
        return state

    def is_valid_move(self, player: str, move: dict) -> tuple[bool, str]:
        if "action" not in move:
            return False, "Move must have an 'action' key."
        if move["action"] not in self.VALID_ACTIONS:
            return False, f"Action must be 'cooperate' or 'defect'. Got: {move['action']}"
        if player != self.get_current_player():
            return False, f"Not {player}'s turn."
        return True, "Valid."

    def make_move(self, player: str, move: dict) -> dict:
        valid, reason = self.is_valid_move(player, move)
        if not valid:
            raise ValueError(f"Invalid move: {reason}")

        action = move["action"]

        if player == self.PLAYER_A:
            # Store A's action; wait for B
            self._pending_a_action = action
            self.turn_number += 1

        elif player == self.PLAYER_B:
            # Resolve round
            a_action = self._pending_a_action
            b_action = action
            a_payoff, b_payoff = self.PAYOFFS[(a_action, b_action)]

            self.round_history.append({
                "round": self.current_round,
                "player_a_action": a_action,
                "player_b_action": b_action,
                "player_a_payoff": a_payoff,
                "player_b_payoff": b_payoff,
            })
            self.cumulative_payoffs[self.PLAYER_A] += a_payoff
            self.cumulative_payoffs[self.PLAYER_B] += b_payoff
            self._pending_a_action = None
            self.current_round += 1
            self.turn_number += 1

            if self.current_round > self.num_rounds:
                self._winner = self._determine_winner()

        return self.get_state()

    def is_terminal(self) -> bool:
        return self.current_round > self.num_rounds

    def get_winner(self) -> Optional[str]:
        if not self.is_terminal():
            return None
        return self._winner

    def get_legal_moves(self, player: str) -> list:
        if self.is_terminal():
            return []
        return [{"action": self.COOPERATE}, {"action": self.DEFECT}]

    def render(self) -> str:
        lines = [f"=== Prisoner's Dilemma — Round {self.current_round}/{self.num_rounds} ==="]
        lines.append(f"Cumulative: A={self.cumulative_payoffs[self.PLAYER_A]}, B={self.cumulative_payoffs[self.PLAYER_B]}")
        if self.round_history:
            lines.append("\nRound History:")
            for r in self.round_history[-5:]:   # Last 5 rounds
                lines.append(
                    f"  R{r['round']}: A={r['player_a_action'][:1].upper()} ({r['player_a_payoff']}) "
                    f"| B={r['player_b_action'][:1].upper()} ({r['player_b_payoff']})"
                )
        return "\n".join(lines)

    def _determine_winner(self) -> str:
        pa = self.cumulative_payoffs[self.PLAYER_A]
        pb = self.cumulative_payoffs[self.PLAYER_B]
        if pa > pb:
            return self.PLAYER_A
        if pb > pa:
            return self.PLAYER_B
        return "draw"

    def get_pareto_efficiency(self) -> float:
        """
        Measures how close the joint outcome is to Pareto-optimal (all cooperate).
        Pareto-optimal payoff per player: 3 * num_rounds
        Returns: ratio in [0, 1]
        """
        max_joint = 3 * self.num_rounds * 2  # Both always cooperate
        actual_joint = sum(self.cumulative_payoffs.values())
        return actual_joint / max_joint if max_joint > 0 else 0.0
