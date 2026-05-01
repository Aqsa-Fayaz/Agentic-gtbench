"""
Tests for the MetricsEngine and Reporter.
Run: python -m pytest tests/test_evaluation.py -v
"""

from __future__ import annotations

import json

import pytest

from evaluation.metrics import MetricsEngine
from evaluation.reporter import Reporter


def _mk_session(**overrides) -> dict:
    """
    Build a session dict that mirrors what game_graph.node_record_result emits:
    `winner` is the role label ("player_a" / "player_b" / "draw"), NOT the
    agent_id. The metrics engine joins to agent_ids via the per-role fields.
    """
    base = {
        "session_id": "s1",
        "game_name": "tictactoe",
        "player_a_id": "A",
        "player_b_id": "B",
        "player_a_strategy": "direct",
        "player_b_strategy": "cot",
        "winner": "player_a",
        "turns": 5,
        "invalid_moves_a": 0,
        "invalid_moves_b": 1,
        "total_moves_a": 3,
        "total_moves_b": 3,
        "forfeited_by": None,
        "extra": {},
    }
    base.update(overrides)
    return base


class TestMetricsEngine:
    def test_record_and_report(self):
        engine = MetricsEngine()
        engine.record_session(_mk_session())
        engine.record_session(_mk_session(session_id="s2", winner="player_b"))
        engine.record_session(_mk_session(session_id="s3", winner="draw"))

        report = engine.full_report()
        assert report["total_sessions"] == 3
        assert any("direct" in k for k in report["win_rates"].keys())
        assert set(report["elo_ratings"].keys()) == {"A", "B"}

    def test_invalid_rate(self):
        engine = MetricsEngine()
        engine.record_session(_mk_session())
        rates = engine.compute_invalid_rates()
        assert rates["A"]["invalid_rate"] == 0.0
        assert rates["B"]["invalid_rate"] > 0

    def test_elo_changes_with_wins(self):
        engine = MetricsEngine()
        engine.record_session(_mk_session())
        engine.record_session(_mk_session(session_id="s2"))
        elo = engine.get_elo_ratings()
        assert elo["A"] > 1000
        assert elo["B"] < 1000

    def test_cooperation_rate(self):
        engine = MetricsEngine()
        pd_session = _mk_session(
            session_id="pd1",
            game_name="prisoners_dilemma",
            player_a_strategy="tit_for_tat",
            player_b_strategy="always_defect",
            winner="player_b",
            extra={
                "player_a_actions": ["cooperate", "cooperate", "defect"],
                "player_b_actions": ["defect", "defect", "defect"],
            },
        )
        engine.record_session(pd_session)
        rates = engine.compute_cooperation_rates()
        assert rates["tit_for_tat"]["cooperation_rate"] == pytest.approx(2 / 3)
        assert rates["always_defect"]["cooperation_rate"] == 0.0

    def test_pareto_efficiency(self):
        engine = MetricsEngine()
        engine.record_session(_mk_session(
            game_name="prisoners_dilemma",
            extra={"pareto_efficiency": 0.75},
        ))
        pareto = engine.compute_pareto_efficiency()
        assert pareto
        matchup = next(iter(pareto.values()))
        assert matchup["avg_pareto"] == 0.75

    def test_avg_turns_to_win(self):
        engine = MetricsEngine()
        engine.record_session(_mk_session(turns=5))
        engine.record_session(_mk_session(session_id="s2", turns=7))
        engine.record_session(_mk_session(session_id="s3", winner="draw"))  # excluded
        avg = engine.compute_avg_turns()
        assert any(v["avg_turns"] == 6 for v in avg.values())


class TestReporter:
    def test_writes_all_three_formats(self, tmp_path):
        engine = MetricsEngine()
        engine.record_session(_mk_session())
        engine.record_session(_mk_session(session_id="s2", winner="B"))
        report = engine.full_report()

        reporter = Reporter(output_dir=tmp_path)
        json_path = reporter.write_json(report, engine.sessions)
        csv_path = reporter.write_csv(engine.sessions)
        html_path = reporter.write_html(report, engine.sessions)

        assert json_path.exists()
        assert csv_path.exists()
        assert html_path.exists()

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        assert payload["report"]["total_sessions"] == 2
        assert "<html" in html_path.read_text(encoding="utf-8").lower()
