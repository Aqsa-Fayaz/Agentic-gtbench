"""
Unit tests for all game environments.
Run: python -m pytest tests/test_games.py -v
"""

import pytest
from games.connect4 import Connect4
from games.kuhn_poker import KuhnPoker
from games.nim import Nim
from games.prisoners_dilemma import PrisonersDilemma
from games.tictactoe import TicTacToe


# ─────────────────────────────────────────────
# TicTacToe Tests
# ─────────────────────────────────────────────

class TestTicTacToe:
    def test_initial_state(self):
        game = TicTacToe()
        state = game.get_state()
        assert state["current_player"] == "player_a"
        assert state["turn_number"] == 0
        assert state["winner"] is None

    def test_valid_move_accepted(self):
        game = TicTacToe()
        valid, reason = game.is_valid_move("player_a", {"row": 0, "col": 0})
        assert valid is True

    def test_invalid_move_out_of_bounds(self):
        game = TicTacToe()
        valid, reason = game.is_valid_move("player_a", {"row": 5, "col": 0})
        assert valid is False
        assert "0–2" in reason

    def test_invalid_move_occupied_cell(self):
        game = TicTacToe()
        game.make_move("player_a", {"row": 0, "col": 0})
        valid, reason = game.is_valid_move("player_b", {"row": 0, "col": 0})
        assert valid is False
        assert "occupied" in reason

    def test_wrong_player_rejected(self):
        game = TicTacToe()
        valid, reason = game.is_valid_move("player_b", {"row": 0, "col": 0})
        assert valid is False

    def test_row_win(self):
        game = TicTacToe()
        moves = [
            ("player_a", {"row": 0, "col": 0}),
            ("player_b", {"row": 1, "col": 0}),
            ("player_a", {"row": 0, "col": 1}),
            ("player_b", {"row": 1, "col": 1}),
            ("player_a", {"row": 0, "col": 2}),   # Player A wins row 0
        ]
        for player, move in moves:
            game.make_move(player, move)
        assert game.is_terminal()
        assert game.get_winner() == "player_a"

    def test_diagonal_win(self):
        game = TicTacToe()
        moves = [
            ("player_a", {"row": 0, "col": 0}),
            ("player_b", {"row": 0, "col": 1}),
            ("player_a", {"row": 1, "col": 1}),
            ("player_b", {"row": 0, "col": 2}),
            ("player_a", {"row": 2, "col": 2}),   # Main diagonal
        ]
        for player, move in moves:
            game.make_move(player, move)
        assert game.get_winner() == "player_a"

    def test_draw(self):
        game = TicTacToe()
        draw_moves = [
            ("player_a", {"row": 0, "col": 0}),
            ("player_b", {"row": 0, "col": 1}),
            ("player_a", {"row": 0, "col": 2}),
            ("player_b", {"row": 1, "col": 0}),
            ("player_a", {"row": 1, "col": 2}),
            ("player_b", {"row": 1, "col": 1}),
            ("player_a", {"row": 2, "col": 0}),
            ("player_b", {"row": 2, "col": 2}),
            ("player_a", {"row": 2, "col": 1}),
        ]
        for player, move in draw_moves:
            game.make_move(player, move)
        assert game.is_terminal()
        assert game.get_winner() == "draw"

    def test_legal_moves_count(self):
        game = TicTacToe()
        assert len(game.get_legal_moves("player_a")) == 9

    def test_legal_moves_decrease_after_move(self):
        game = TicTacToe()
        game.make_move("player_a", {"row": 0, "col": 0})
        assert len(game.get_legal_moves("player_b")) == 8

    def test_no_legal_moves_after_terminal(self):
        game = TicTacToe()
        # Force a win
        for player, move in [
            ("player_a", {"row": 0, "col": 0}),
            ("player_b", {"row": 1, "col": 0}),
            ("player_a", {"row": 0, "col": 1}),
            ("player_b", {"row": 1, "col": 1}),
            ("player_a", {"row": 0, "col": 2}),
        ]:
            game.make_move(player, move)
        assert game.get_legal_moves("player_b") == []

    def test_render_returns_string(self):
        game = TicTacToe()
        rendered = game.render()
        assert isinstance(rendered, str)
        assert "0" in rendered


# ─────────────────────────────────────────────
# Prisoner's Dilemma Tests
# ─────────────────────────────────────────────

class TestPrisonersDilemma:
    def test_initial_state(self):
        game = PrisonersDilemma(num_rounds=5)
        state = game.get_state()
        assert state["current_round"] == 1
        assert state["num_rounds"] == 5

    def test_valid_cooperate(self):
        game = PrisonersDilemma()
        valid, _ = game.is_valid_move("player_a", {"action": "cooperate"})
        assert valid is True

    def test_valid_defect(self):
        game = PrisonersDilemma()
        valid, _ = game.is_valid_move("player_a", {"action": "defect"})
        assert valid is True

    def test_invalid_action(self):
        game = PrisonersDilemma()
        valid, reason = game.is_valid_move("player_a", {"action": "run_away"})
        assert valid is False

    def test_mutual_cooperation_payoff(self):
        game = PrisonersDilemma(num_rounds=1)
        game.make_move("player_a", {"action": "cooperate"})
        game.make_move("player_b", {"action": "cooperate"})
        assert game.cumulative_payoffs["player_a"] == 3
        assert game.cumulative_payoffs["player_b"] == 3

    def test_defect_vs_cooperate_payoff(self):
        game = PrisonersDilemma(num_rounds=1)
        game.make_move("player_a", {"action": "defect"})
        game.make_move("player_b", {"action": "cooperate"})
        assert game.cumulative_payoffs["player_a"] == 5
        assert game.cumulative_payoffs["player_b"] == 0

    def test_terminal_after_all_rounds(self):
        game = PrisonersDilemma(num_rounds=3)
        for _ in range(3):
            game.make_move("player_a", {"action": "cooperate"})
            game.make_move("player_b", {"action": "cooperate"})
        assert game.is_terminal()

    def test_pareto_efficiency_full_coop(self):
        game = PrisonersDilemma(num_rounds=5)
        for _ in range(5):
            game.make_move("player_a", {"action": "cooperate"})
            game.make_move("player_b", {"action": "cooperate"})
        assert game.get_pareto_efficiency() == 1.0

    def test_pareto_efficiency_full_defect(self):
        game = PrisonersDilemma(num_rounds=5)
        for _ in range(5):
            game.make_move("player_a", {"action": "defect"})
            game.make_move("player_b", {"action": "defect"})
        # Nash equilibrium (all defect) = joint payoff 10 / optimal 30
        assert round(game.get_pareto_efficiency(), 3) == round(10 / 30, 3)


# ─────────────────────────────────────────────
# Connect4 Tests
# ─────────────────────────────────────────────

class TestConnect4:
    def test_initial_state(self):
        game = Connect4()
        state = game.get_state()
        assert state["current_player"] == "player_a"
        assert state["rows"] == 6 and state["cols"] == 7

    def test_invalid_column(self):
        game = Connect4()
        valid, reason = game.is_valid_move("player_a", {"col": 99})
        assert valid is False
        assert "Column" in reason

    def test_gravity_applies(self):
        game = Connect4()
        game.make_move("player_a", {"col": 3})
        state = game.get_state()
        assert state["board"][5][3] == "X"  # Bottom row

    def test_vertical_win(self):
        game = Connect4()
        # A drops 4 in column 0; B alternates in column 1
        for _ in range(3):
            game.make_move("player_a", {"col": 0})
            game.make_move("player_b", {"col": 1})
        game.make_move("player_a", {"col": 0})  # 4-in-a-row vertical
        assert game.is_terminal()
        assert game.get_winner() == "player_a"

    def test_horizontal_win(self):
        game = Connect4()
        for col in range(4):
            game.make_move("player_a", {"col": col})
            if col < 3:
                game.make_move("player_b", {"col": col})  # On row above
                # Undo into a harmless col so A wins first
        # Easier: deterministic scripted sequence
        game = Connect4()
        seq = [
            ("player_a", 0), ("player_b", 0),
            ("player_a", 1), ("player_b", 1),
            ("player_a", 2), ("player_b", 2),
            ("player_a", 3),   # A wins bottom row 0-3
        ]
        for p, c in seq:
            game.make_move(p, {"col": c})
        assert game.get_winner() == "player_a"

    def test_legal_moves_shrinks(self):
        game = Connect4()
        # Fill column 0
        for _ in range(6):
            game.make_move(game.get_current_player(), {"col": 0})
        legal = game.get_legal_moves(game.get_current_player())
        assert all(m["col"] != 0 for m in legal)

    def test_render_has_columns(self):
        rendered = Connect4().render()
        assert "0" in rendered and "6" in rendered


# ─────────────────────────────────────────────
# Nim Tests
# ─────────────────────────────────────────────

class TestNim:
    def test_initial_piles(self):
        game = Nim(piles=[2, 3])
        assert game.get_state()["piles"] == [2, 3]

    def test_invalid_empty_pile(self):
        game = Nim(piles=[1])
        game.make_move("player_a", {"pile": 0, "count": 1})
        valid, _ = game.is_valid_move("player_b", {"pile": 0, "count": 1})
        assert valid is False

    def test_invalid_count_zero(self):
        game = Nim(piles=[3])
        valid, _ = game.is_valid_move("player_a", {"pile": 0, "count": 0})
        assert valid is False

    def test_misere_loser_takes_last(self):
        game = Nim(piles=[1])
        game.make_move("player_a", {"pile": 0, "count": 1})
        assert game.is_terminal()
        # Misère: player_a took the last → player_a LOSES
        assert game.get_winner() == "player_b"

    def test_legal_moves_full(self):
        game = Nim(piles=[2, 3])
        legal = game.get_legal_moves("player_a")
        # Pile 0 (size 2): counts 1..2, Pile 1 (size 3): counts 1..3 => 5 options
        assert len(legal) == 5

    def test_render_non_empty(self):
        assert "Pile" in Nim(piles=[2]).render()


# ─────────────────────────────────────────────
# Kuhn Poker Tests
# ─────────────────────────────────────────────

class TestKuhnPoker:
    def test_cards_dealt_distinct(self):
        game = KuhnPoker(seed=0)
        assert game._cards["player_a"] != game._cards["player_b"]

    def test_private_info_hidden(self):
        game = KuhnPoker(seed=1)
        state_a = game.get_state_for_player("player_a")
        state_b = game.get_state_for_player("player_b")
        assert state_a["my_card"] != state_b["my_card"]
        assert state_a["opponent_card"] is None

    def test_pass_pass_showdown(self):
        game = KuhnPoker(seed=42)
        game.make_move("player_a", {"action": "pass"})
        game.make_move("player_b", {"action": "pass"})
        assert game.is_terminal()
        assert game._showdown is True

    def test_bet_fold_wins_pot(self):
        game = KuhnPoker(seed=7)
        game.make_move("player_a", {"action": "bet"})
        game.make_move("player_b", {"action": "pass"})  # folds
        assert game.is_terminal()
        assert game.get_winner() == "player_a"
        assert game._folded_by == "player_b"

    def test_invalid_action(self):
        game = KuhnPoker(seed=0)
        valid, _ = game.is_valid_move("player_a", {"action": "raise"})
        assert valid is False

    def test_full_bet_call(self):
        game = KuhnPoker(seed=3)
        game.make_move("player_a", {"action": "bet"})
        game.make_move("player_b", {"action": "bet"})  # call
        assert game.is_terminal()
        assert game._showdown is True
