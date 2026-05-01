"""
Tests for agent classes and supporting tools (no network calls).
Run: python -m pytest tests/test_agents.py -v
"""

from __future__ import annotations

import json
import types

import pytest

from agents.orchestrator_agent import OrchestratorAgent
from agents.player_agent import PlayerAgent
from agents.strategies import load_strategy
from games import load_game
from tools.history_manager import HistoryManagerTool
from tools.move_validator import MoveValidatorTool, validate_move
from tools.state_tracker import StateTrackerTool
from tools.strategy_analyzer import StrategyAnalyzerTool


class DummyResponse:
    def __init__(self, content: str):
        self.content = content


class DummyLLM:
    """Stand-in ChatOpenAI that returns pre-canned responses."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        if not self.replies:
            raise RuntimeError("DummyLLM ran out of replies")
        return DummyResponse(self.replies.pop(0))


def _make_player(strategy: str = "direct") -> PlayerAgent:
    agent = PlayerAgent.__new__(PlayerAgent)
    agent.agent_id = f"test_{strategy}"
    agent.model = "dummy"
    agent.temperature = 0.0
    agent.strategy_name = strategy
    agent.strategy = load_strategy(strategy)
    agent.tools = []
    agent.provider = "openai"
    agent.invalid_move_count = 0
    agent.total_moves = 0
    agent.move_history = []
    agent.conversation_history = []
    agent.llm = DummyLLM([])
    return agent


class TestStrategyModules:
    def test_direct_parses_json(self):
        player = _make_player("direct")
        player.llm.replies = ['{"move": {"row": 1, "col": 1}}']
        move = player.decide({"turn": 0}, "tictactoe", [{"row": 1, "col": 1}])
        assert move == {"row": 1, "col": 1}

    def test_cot_unwraps_move_key(self):
        player = _make_player("cot")
        player.llm.replies = [
            'Reasoning... Final: {"move": {"row": 0, "col": 2}, "reasoning": "corner"}'
        ]
        move = player.decide({"turn": 0}, "tictactoe", [{"row": 0, "col": 2}])
        assert move == {"row": 0, "col": 2}

    def test_tot_three_calls(self):
        player = _make_player("tot")
        player.llm.replies = [
            '[{"move": {"row": 0, "col": 0}, "justification": "a"},'
            ' {"move": {"row": 1, "col": 1}, "justification": "b"},'
            ' {"move": {"row": 2, "col": 2}, "justification": "c"}]',
            '[{"move": {"row": 1, "col": 1}, "score": 9, "reason": "center"}]',
            '{"move": {"row": 1, "col": 1}}',
        ]
        move = player.decide({"turn": 0}, "tictactoe", [{"row": 1, "col": 1}])
        assert move == {"row": 1, "col": 1}
        assert len(player.llm.calls) == 3

    def test_react_final_move(self):
        player = _make_player("react")
        player.llm.replies = [
            "Thought: take center.\nAction: MOVE {\"row\": 1, \"col\": 1}"
        ]
        move = player.decide({"turn": 0}, "tictactoe", [{"row": 1, "col": 1}])
        assert move == {"row": 1, "col": 1}

    def test_retry_on_bad_json_then_success(self):
        player = _make_player("direct")
        player.llm.replies = ["not json", '{"row": 0, "col": 0}']
        move = player.decide({"turn": 0}, "tictactoe", [{"row": 0, "col": 0}])
        assert move == {"row": 0, "col": 0}
        assert player.invalid_move_count == 1

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            load_strategy("mcts")


class TestMoveValidatorTool:
    def test_valid_move(self):
        tool = MoveValidatorTool()
        game = load_game("tictactoe")
        out = json.loads(
            tool._run(
                game_name="tictactoe",
                state=game.get_state(),
                player="player_a",
                move={"row": 0, "col": 0},
            )
        )
        assert out["valid"] is True

    def test_invalid_move(self):
        tool = MoveValidatorTool()
        game = load_game("tictactoe")
        out = json.loads(
            tool._run(
                game_name="tictactoe",
                state=game.get_state(),
                player="player_a",
                move={"row": 9, "col": 9},
            )
        )
        assert out["valid"] is False

    def test_helper_validate_move(self):
        game = load_game("tictactoe")
        valid, reason, legal = validate_move(game, "player_a", {"row": 0, "col": 0})
        assert valid is True
        assert isinstance(legal, list)


class TestStateTrackerTool:
    def test_tictactoe_threat_detection(self):
        game = load_game("tictactoe")
        game.make_move("player_a", {"row": 0, "col": 0})
        game.make_move("player_b", {"row": 1, "col": 0})
        game.make_move("player_a", {"row": 0, "col": 1})  # A has 2-in-a-row
        game.make_move("player_b", {"row": 2, "col": 2})
        out = json.loads(
            StateTrackerTool()._run(
                game_name="tictactoe",
                state=game.get_state(),
                player="player_b",
            )
        )
        assert any("threat" in t for t in out["threats"])

    def test_prisoners_dilemma_description(self):
        game = load_game("prisoners_dilemma", num_rounds=5)
        out = json.loads(
            StateTrackerTool()._run(
                game_name="prisoners_dilemma",
                state=game.get_state(),
                player="player_a",
            )
        )
        assert "Prisoner" in out["description"]


class TestStrategyAnalyzerTool:
    def test_always_defect_detected(self):
        history = [
            {"player_a_action": "cooperate", "player_b_action": "defect"},
            {"player_a_action": "defect", "player_b_action": "defect"},
            {"player_a_action": "cooperate", "player_b_action": "defect"},
        ]
        out = json.loads(
            StrategyAnalyzerTool()._run(
                move_history=history,
                game_name="prisoners_dilemma",
                player_perspective="player_a",
            )
        )
        assert out["inferred_strategy"] == "Always Defect"

    def test_empty_history(self):
        out = json.loads(
            StrategyAnalyzerTool()._run(
                move_history=[],
                game_name="prisoners_dilemma",
                player_perspective="player_a",
            )
        )
        assert out["confidence"] == 0.0


class TestHistoryManagerTool:
    def test_save_load_list(self, tmp_path):
        tool = HistoryManagerTool(store_dir=tmp_path)
        save_out = json.loads(
            tool._run(operation="save", session_id="abc", game_log={"winner": "player_a"})
        )
        assert save_out["saved"] is True

        load_out = json.loads(tool._run(operation="load", session_id="abc"))
        assert load_out["winner"] == "player_a"

        list_out = json.loads(tool._run(operation="list"))
        assert "abc" in list_out["sessions"]

    def test_stats_aggregation(self, tmp_path):
        tool = HistoryManagerTool(store_dir=tmp_path)
        tool.save_game("s1", {
            "player_a_id": "A", "player_b_id": "B", "winner": "A",
        })
        tool.save_game("s2", {
            "player_a_id": "A", "player_b_id": "B", "winner": "B",
        })
        tool.save_game("s3", {
            "player_a_id": "A", "player_b_id": "B", "winner": "draw",
        })
        stats = tool.get_stats("A")
        assert stats["sessions"] == 3
        assert stats["wins"] == 1
        assert stats["losses"] == 1
        assert stats["draws"] == 1


class TestConventionalAgents:
    """Tests for the non-LLM playable agents (Random, TfT, MCTS)."""

    def test_random_picks_only_legal(self):
        from agents.random_agent import RandomAgent
        agent = RandomAgent("rnd", seed=0)
        legal = [{"row": 0, "col": 0}, {"row": 1, "col": 1}]
        for _ in range(5):
            move = agent.decide({}, "tictactoe", legal)
            assert move in legal

    def test_tft_first_move_is_cooperate(self):
        from agents.tit_for_tat_agent import TitForTatAgent
        agent = TitForTatAgent("tft")
        move = agent.decide(
            {"current_player": "player_a", "round_history": []},
            "prisoners_dilemma",
            [{"action": "cooperate"}, {"action": "defect"}],
        )
        assert move == {"action": "cooperate"}

    def test_tft_mirrors_opponent_last(self):
        from agents.tit_for_tat_agent import TitForTatAgent
        agent = TitForTatAgent("tft")
        # I'm player_a; opponent (player_b) defected last round → I should defect.
        move = agent.decide(
            {
                "current_player": "player_a",
                "round_history": [
                    {"player_a_action": "cooperate", "player_b_action": "defect"}
                ],
            },
            "prisoners_dilemma",
            [{"action": "cooperate"}, {"action": "defect"}],
        )
        assert move == {"action": "defect"}

    def test_mcts_beats_random_on_tictactoe(self):
        """MCTS with even modest sims should reliably beat Random on TTT."""
        from agents.mcts_agent import MCTSAgent
        from agents.random_agent import RandomAgent

        wins = 0
        for trial in range(3):
            mcts = MCTSAgent("mcts", n_simulations=100, seed=trial)
            rnd = RandomAgent("rnd", seed=trial + 100)
            orch = OrchestratorAgent()
            result = orch.run_session("tictactoe", mcts, rnd)
            if result["winner"] == "player_a":
                wins += 1
        assert wins >= 2, f"MCTS should win 2+/3 games against Random, got {wins}"


class TestOrchestrator:
    def test_run_session_with_deterministic_players(self, monkeypatch):
        """
        Replace PlayerAgent.decide with a scripted function so we can drive a
        full tic-tac-toe game through the orchestrator without any API calls.
        """
        orch = OrchestratorAgent()
        moves_a = iter([
            {"row": 0, "col": 0},
            {"row": 0, "col": 1},
            {"row": 0, "col": 2},  # winning row
        ])
        moves_b = iter([
            {"row": 1, "col": 0},
            {"row": 1, "col": 1},
        ])

        player_a = _make_player("direct")
        player_b = _make_player("direct")

        def decide_a(self, *args, **kwargs):
            self.total_moves += 1
            return next(moves_a)

        def decide_b(self, *args, **kwargs):
            self.total_moves += 1
            return next(moves_b)

        player_a.decide = types.MethodType(decide_a, player_a)
        player_b.decide = types.MethodType(decide_b, player_b)

        result = orch.run_session("tictactoe", player_a, player_b)
        assert result["winner"] == "player_a"
        assert result["turns"] == 5
