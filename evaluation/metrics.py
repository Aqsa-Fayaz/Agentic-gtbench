"""
Evaluation Metrics Engine
Computes all metrics used in experiments.
"""

import math
from collections import defaultdict
from typing import Optional


class MetricsEngine:
    """
    Computes all evaluation metrics from game session logs.

    Metrics:
    - Win rate per agent/strategy/game
    - Invalid move rate per agent/strategy
    - Elo ratings (tournament-style)
    - Average turns to win
    - Cooperation rate (Prisoner's Dilemma)
    - Pareto efficiency (Prisoner's Dilemma)
    """

    ELO_K = 32          # Elo K-factor
    ELO_DEFAULT = 1000  # Starting Elo

    def __init__(self):
        self.sessions = []
        self.elo_ratings = defaultdict(lambda: self.ELO_DEFAULT)

    def record_session(self, session: dict):
        """
        Record a completed game session.
        Expected session keys:
          session_id, game_name, player_a_id, player_b_id,
          player_a_strategy, player_b_strategy, winner,
          turns, invalid_moves_a, invalid_moves_b,
          total_moves_a, total_moves_b, extra (game-specific data)
        """
        self.sessions.append(session)
        self._update_elo(session)

    def compute_win_rates(self) -> dict:
        """Returns win rate breakdown by strategy and game."""
        results = defaultdict(lambda: {"wins": 0, "losses": 0, "draws": 0, "games": 0})

        for s in self.sessions:
            winner = s["winner"]
            game = s["game_name"]

            for role in ["player_a", "player_b"]:
                key = (s[f"{role}_strategy"], game)
                results[key]["games"] += 1
                if winner == s[f"{role}_id"]:
                    results[key]["wins"] += 1
                elif winner == "draw":
                    results[key]["draws"] += 1
                else:
                    results[key]["losses"] += 1

        # Compute rates
        output = {}
        for (strategy, game), counts in results.items():
            n = counts["games"]
            output[(strategy, game)] = {
                "strategy": strategy,
                "game": game,
                "games": n,
                "win_rate": counts["wins"] / n if n > 0 else 0,
                "draw_rate": counts["draws"] / n if n > 0 else 0,
                "loss_rate": counts["losses"] / n if n > 0 else 0,
            }
        return output

    def compute_invalid_rates(self) -> dict:
        """Returns invalid move rate per agent."""
        agent_stats = defaultdict(lambda: {"invalid": 0, "total": 0})

        for s in self.sessions:
            agent_stats[s["player_a_id"]]["invalid"] += s.get("invalid_moves_a", 0)
            agent_stats[s["player_a_id"]]["total"] += s.get("total_moves_a", 0)
            agent_stats[s["player_b_id"]]["invalid"] += s.get("invalid_moves_b", 0)
            agent_stats[s["player_b_id"]]["total"] += s.get("total_moves_b", 0)

        return {
            agent_id: {
                "invalid_moves": v["invalid"],
                "total_moves": v["total"],
                "invalid_rate": v["invalid"] / max(v["total"], 1),
            }
            for agent_id, v in agent_stats.items()
        }

    def get_elo_ratings(self) -> dict:
        """Return current Elo ratings for all agents."""
        return dict(self.elo_ratings)

    def compute_avg_turns(self) -> dict:
        """Average turns to win, per strategy and game."""
        stats = defaultdict(list)
        for s in self.sessions:
            if s["winner"] in (s["player_a_id"], s["player_b_id"]):  # Skip draws
                winner_strategy = (
                    s["player_a_strategy"] if s["winner"] == s["player_a_id"]
                    else s["player_b_strategy"]
                )
                stats[(winner_strategy, s["game_name"])].append(s.get("turns", 0))

        return {
            str(k): {
                "strategy": k[0],
                "game": k[1],
                "avg_turns": sum(v) / len(v),
                "min_turns": min(v),
                "max_turns": max(v),
                "samples": len(v),
            }
            for k, v in stats.items()
        }

    def compute_pareto_efficiency(self) -> dict:
        """Pareto efficiency for Prisoner's Dilemma sessions."""
        pd_sessions = [s for s in self.sessions if s["game_name"] == "prisoners_dilemma"]
        if not pd_sessions:
            return {}

        matchup_stats = defaultdict(list)
        for s in pd_sessions:
            key = f"{s['player_a_strategy']} vs {s['player_b_strategy']}"
            if "pareto_efficiency" in s.get("extra", {}):
                matchup_stats[key].append(s["extra"]["pareto_efficiency"])

        return {
            matchup: {
                "avg_pareto": sum(v) / len(v),
                "samples": len(v),
            }
            for matchup, v in matchup_stats.items()
        }

    def compute_cooperation_rates(self) -> dict:
        """Cooperation rate per strategy in Prisoner's Dilemma."""
        pd_sessions = [s for s in self.sessions if s["game_name"] == "prisoners_dilemma"]
        coop_stats = defaultdict(lambda: {"cooperate": 0, "total": 0})

        for s in pd_sessions:
            for role, strategy in [
                ("player_a", s["player_a_strategy"]),
                ("player_b", s["player_b_strategy"]),
            ]:
                history = s.get("extra", {}).get(f"{role}_actions", [])
                coop_stats[strategy]["cooperate"] += history.count("cooperate")
                coop_stats[strategy]["total"] += len(history)

        return {
            strategy: {
                "cooperation_rate": v["cooperate"] / max(v["total"], 1),
                "total_actions": v["total"],
            }
            for strategy, v in coop_stats.items()
        }

    def full_report(self) -> dict:
        """Aggregate all metrics into one dict."""
        return {
            "total_sessions": len(self.sessions),
            "win_rates": {str(k): v for k, v in self.compute_win_rates().items()},
            "invalid_rates": self.compute_invalid_rates(),
            "elo_ratings": self.get_elo_ratings(),
            "avg_turns": self.compute_avg_turns(),
            "pareto_efficiency": self.compute_pareto_efficiency(),
            "cooperation_rates": self.compute_cooperation_rates(),
        }

    def _update_elo(self, session: dict):
        """Update Elo ratings after a game."""
        a_id = session["player_a_id"]
        b_id = session["player_b_id"]
        winner = session["winner"]

        ra = self.elo_ratings[a_id]
        rb = self.elo_ratings[b_id]

        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea

        if winner == a_id:
            sa, sb = 1, 0
        elif winner == b_id:
            sa, sb = 0, 1
        else:
            sa = sb = 0.5

        self.elo_ratings[a_id] = ra + self.ELO_K * (sa - ea)
        self.elo_ratings[b_id] = rb + self.ELO_K * (sb - eb)
