"""
Unit tests for all game environments.
Run: python -m pytest tests/test_games.py -v
"""

import pytest
from games.blind_auction import BlindAuction
from games.breakthrough import Breakthrough
from games.connect4 import Connect4
from games.kuhn_poker import KuhnPoker
from games.liars_dice import LiarsDice
from games.negotiation import Negotiation
from games.nim import Nim
from games.pig import Pig
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


# ─────────────────────────────────────────────
# Pig Tests
# ─────────────────────────────────────────────

class TestPig:
    def test_initial_state(self):
        game = Pig(target_score=20, seed=0)
        state = game.get_state()
        assert state["scores"] == {"player_a": 0, "player_b": 0}
        assert state["turn_total"] == 0
        assert state["current_player"] == "player_a"

    def test_hold_banks_score_and_switches(self):
        game = Pig(target_score=100, seed=42)
        # Force a non-1 roll
        game.turn_total = 7
        game.make_move("player_a", {"action": "hold"})
        assert game.scores["player_a"] == 7
        assert game.turn_total == 0
        assert game.get_current_player() == "player_b"

    def test_roll_of_one_busts(self):
        game = Pig(seed=2)   # seeded so first roll is 1 — actually depends on RNG
        # Force-set RNG to produce 1 next
        import random
        game._rng = random.Random()
        game._rng.seed(0)
        # We can't guarantee what seed=0 gives — just verify the bust mechanic
        # by injecting a fake RNG.
        class FakeRng:
            def randint(self, a, b): return 1
        game._rng = FakeRng()
        game.turn_total = 5
        game.make_move("player_a", {"action": "roll"})
        assert game.turn_total == 0
        assert game.get_current_player() == "player_b"

    def test_roll_non_one_accumulates(self):
        game = Pig(seed=0)
        class FakeRng:
            def randint(self, a, b): return 4
        game._rng = FakeRng()
        game.make_move("player_a", {"action": "roll"})
        assert game.turn_total == 4
        assert game.get_current_player() == "player_a"   # still A's turn

    def test_winner_at_target(self):
        game = Pig(target_score=5)
        game.scores["player_a"] = 4
        game.turn_total = 1
        game.make_move("player_a", {"action": "hold"})
        assert game.is_terminal()
        assert game.get_winner() == "player_a"

    def test_invalid_action_rejected(self):
        game = Pig()
        valid, _ = game.is_valid_move("player_a", {"action": "skip"})
        assert valid is False

    def test_legal_moves(self):
        game = Pig()
        legal = game.get_legal_moves("player_a")
        assert {"action": "roll"} in legal
        assert {"action": "hold"} in legal

    def test_from_state_restores(self):
        game = Pig(target_score=10)
        game.scores = {"player_a": 5, "player_b": 3}
        game.turn_total = 2
        game._active_player = "player_b"
        state = game.get_state()
        new_game = Pig.from_state(state)
        assert new_game.scores == game.scores
        assert new_game.turn_total == 2
        assert new_game.get_current_player() == "player_b"


# ─────────────────────────────────────────────
# Blind Auction Tests
# ─────────────────────────────────────────────

class TestBlindAuction:
    def test_initial_state_has_private_valuations(self):
        game = BlindAuction(seed=0)
        state_a = game.get_state_for_player("player_a")
        state_b = game.get_state_for_player("player_b")
        assert "my_valuation" in state_a
        assert "my_valuation" in state_b
        # Each player only sees their own valuation in `valuations`.
        assert "player_b" not in state_a.get("valuations", {})
        assert "player_a" not in state_b.get("valuations", {})

    def test_bid_outside_range_rejected(self):
        game = BlindAuction(max_bid=50)
        valid, reason = game.is_valid_move("player_a", {"bid": 999})
        assert valid is False

    def test_higher_bid_wins_and_pays_own_bid(self):
        game = BlindAuction(seed=0)
        game._valuations = {"player_a": 80, "player_b": 60}
        game.make_move("player_a", {"bid": 50})
        game.make_move("player_b", {"bid": 30})
        assert game.is_terminal()
        assert game.get_winner() == "player_a"
        assert game.payoffs["player_a"] == 80 - 50
        assert game.payoffs["player_b"] == 0

    def test_overbid_yields_negative_payoff(self):
        game = BlindAuction(seed=0)
        game._valuations = {"player_a": 30, "player_b": 70}
        game.make_move("player_a", {"bid": 90})
        game.make_move("player_b", {"bid": 10})
        assert game.get_winner() == "player_a"
        assert game.payoffs["player_a"] == 30 - 90   # -60

    def test_b_does_not_see_a_pending_bid(self):
        game = BlindAuction(seed=0)
        game.make_move("player_a", {"bid": 25})
        state_b = game.get_state_for_player("player_b")
        assert "player_a" not in state_b.get("bids", {})

    def test_tie_split(self):
        game = BlindAuction()
        game._valuations = {"player_a": 50, "player_b": 50}
        game.make_move("player_a", {"bid": 30})
        game.make_move("player_b", {"bid": 30})
        assert game.get_winner() == "draw"
        assert game.payoffs["player_a"] == game.payoffs["player_b"]

    def test_legal_moves_anchors(self):
        game = BlindAuction(max_bid=100)
        legal = game.get_legal_moves("player_a")
        assert {"bid": 0} in legal
        assert {"bid": 100} in legal


# ─────────────────────────────────────────────
# Liar's Dice Tests
# ─────────────────────────────────────────────

class TestLiarsDice:
    def test_initial_dice_dealt(self):
        game = LiarsDice(dice_per_player=5, seed=0)
        assert len(game._dice["player_a"]) == 5
        assert len(game._dice["player_b"]) == 5

    def test_opponent_dice_hidden(self):
        game = LiarsDice(seed=0)
        state = game.get_state_for_player("player_a")
        assert state["opponent_dice"] is None
        assert "my_dice" in state

    def test_first_action_must_be_bid(self):
        game = LiarsDice(seed=0)
        valid, _ = game.is_valid_move("player_a", {"action": "call"})
        assert valid is False

    def test_bid_then_call_resolves(self):
        game = LiarsDice(seed=0, dice_per_player=2)
        # Inject deterministic dice
        game._dice = {"player_a": [3, 3], "player_b": [3, 5]}
        game.make_move("player_a", {"action": "bid", "face": 3, "count": 4})
        # Player B calls. Actual 3-count = 3 < bid 4 → bidder loses.
        game.make_move("player_b", {"action": "call"})
        assert game.is_terminal()
        assert game.get_winner() == "player_b"

    def test_honest_bid_called_wins(self):
        game = LiarsDice(seed=0, dice_per_player=2)
        game._dice = {"player_a": [4, 4], "player_b": [4, 4]}
        game.make_move("player_a", {"action": "bid", "face": 4, "count": 3})
        game.make_move("player_b", {"action": "call"})
        # actual 4-count = 4 >= bid 3 → bidder wins.
        assert game.get_winner() == "player_a"

    def test_must_strictly_raise(self):
        game = LiarsDice(seed=0)
        game.make_move("player_a", {"action": "bid", "face": 3, "count": 2})
        valid, _ = game.is_valid_move("player_b", {"action": "bid", "face": 3, "count": 2})
        assert valid is False
        valid, _ = game.is_valid_move("player_b", {"action": "bid", "face": 3, "count": 3})
        assert valid is True

    def test_legal_moves_include_call_after_first_bid(self):
        game = LiarsDice(seed=0)
        game.make_move("player_a", {"action": "bid", "face": 3, "count": 2})
        legal = game.get_legal_moves("player_b")
        assert {"action": "call"} in legal


# ─────────────────────────────────────────────
# Breakthrough Tests
# ─────────────────────────────────────────────

class TestBreakthrough:
    def test_initial_layout(self):
        g = Breakthrough()
        # Player A occupies rows 0-1, Player B occupies rows 4-5
        for c in range(g.COLS):
            assert g.board[0][c] == "A"
            assert g.board[1][c] == "A"
            assert g.board[4][c] == "B"
            assert g.board[5][c] == "B"
        assert all(g.board[2][c] == "." and g.board[3][c] == "." for c in range(g.COLS))

    def test_straight_forward_legal(self):
        g = Breakthrough()
        valid, _ = g.is_valid_move(
            "player_a",
            {"from": {"row": 1, "col": 3}, "to": {"row": 2, "col": 3}},
        )
        assert valid is True

    def test_diagonal_capture_legal(self):
        g = Breakthrough()
        # Move A down to threaten B
        g.make_move("player_a", {"from": {"row": 1, "col": 3}, "to": {"row": 2, "col": 3}})
        # B advances on a diagonal — but actually not yet adjacent. Easier: hand-craft.
        g = Breakthrough()
        g.board[3][3] = "A"
        g.board[2][3] = "."
        g.board[1][3] = "."
        # B's turn — fast-forward turn_number so it's B's move
        g.turn_number = 1
        # B at (4,2) can capture A at (3,3) diagonally
        valid, _ = g.is_valid_move(
            "player_b",
            {"from": {"row": 4, "col": 2}, "to": {"row": 3, "col": 3}},
        )
        assert valid is True

    def test_straight_capture_rejected(self):
        g = Breakthrough()
        g.board[2][3] = "B"   # Place B in front of A's piece
        valid, reason = g.is_valid_move(
            "player_a",
            {"from": {"row": 1, "col": 3}, "to": {"row": 2, "col": 3}},
        )
        assert valid is False
        assert "diagonals" in reason or "Straight" in reason

    def test_backward_move_rejected(self):
        g = Breakthrough()
        g.board[2][3] = "A"
        valid, _ = g.is_valid_move(
            "player_a",
            {"from": {"row": 2, "col": 3}, "to": {"row": 1, "col": 3}},
        )
        assert valid is False

    def test_reaching_back_row_wins(self):
        g = Breakthrough()
        # Hand-craft: A at (4, 0), empty (5, 0), B has been depleted enough
        g.board = [["." for _ in range(g.COLS)] for _ in range(g.ROWS)]
        g.board[4][0] = "A"
        # Provide B at least one piece so the legal-moves stalemate check
        # isn't triggered prematurely.
        g.board[5][3] = "B"
        g.board[4][4] = "B"
        valid, _ = g.is_valid_move(
            "player_a",
            {"from": {"row": 4, "col": 0}, "to": {"row": 5, "col": 0}},
        )
        assert valid is True
        g.make_move("player_a", {"from": {"row": 4, "col": 0}, "to": {"row": 5, "col": 0}})
        assert g.is_terminal()
        assert g.get_winner() == "player_a"

    def test_legal_moves_nonempty_at_start(self):
        g = Breakthrough()
        legal = g.get_legal_moves("player_a")
        # Row-0 pieces are blocked entirely (own pieces in row 1).
        # Row-1 pieces all have row-2 free; edge cols (0 and 5) get
        # 2 moves (straight + 1 diagonal); middle cols (1-4) get 3 each.
        # Total: 2 + 3*4 + 2 = 16.
        assert len(legal) == 16


# ─────────────────────────────────────────────
# Negotiation Tests
# ─────────────────────────────────────────────

class TestNegotiation:
    def test_initial_state_hides_opponent_valuations(self):
        g = Negotiation(seed=0)
        sa = g.get_state_for_player("player_a")
        sb = g.get_state_for_player("player_b")
        assert "my_valuations" in sa
        assert "my_valuations" in sb
        assert "player_b" not in sa["valuations"]
        assert "player_a" not in sb["valuations"]

    def test_propose_then_accept_settles(self):
        g = Negotiation(pool={"book": 4}, seed=0)
        g._valuations = {"player_a": {"book": 3}, "player_b": {"book": 5}}
        g.make_move("player_a", {"action": "propose", "for_me": {"book": 1}})
        # B accepts → A keeps 1, B gets 3
        g.make_move("player_b", {"action": "accept"})
        assert g.is_terminal()
        assert g.payoffs["player_a"] == 3 * 1
        assert g.payoffs["player_b"] == 5 * 3
        assert g.get_winner() == "player_b"

    def test_walk_away_zero_payoff(self):
        g = Negotiation(seed=0)
        g.make_move("player_a", {"action": "walk_away"})
        assert g.is_terminal()
        assert g.payoffs == {"player_a": 0, "player_b": 0}
        assert g.get_winner() == "draw"

    def test_accept_without_proposal_rejected(self):
        g = Negotiation(seed=0)
        valid, _ = g.is_valid_move("player_a", {"action": "accept"})
        assert valid is False

    def test_propose_overpool_rejected(self):
        g = Negotiation(pool={"book": 2}, seed=0)
        valid, _ = g.is_valid_move(
            "player_a", {"action": "propose", "for_me": {"book": 5}}
        )
        assert valid is False

    def test_max_turns_walk_away(self):
        g = Negotiation(pool={"book": 1}, max_turns=2, seed=0)
        g.make_move("player_a", {"action": "propose", "for_me": {"book": 1}})
        g.make_move("player_b", {"action": "propose", "for_me": {"book": 1}})
        # Should have hit max_turns and ended with walk_away.
        assert g.is_terminal()
        assert g._terminal_reason == "max_turns_reached"
        assert g.payoffs == {"player_a": 0, "player_b": 0}

    def test_legal_moves_include_anchors(self):
        g = Negotiation(seed=0)
        legal = g.get_legal_moves("player_a")
        # No proposal yet, so no accept — but walk_away + 3 proposal anchors.
        assert {"action": "walk_away"} in legal
        # Three proposal anchors expected.
        proposals = [m for m in legal if m.get("action") == "propose"]
        assert len(proposals) == 3
