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
                if winner == role:
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
            if s["winner"] in ("player_a", "player_b"):  # Skip draws / forfeits
                winner_strategy = (
                    s["player_a_strategy"] if s["winner"] == "player_a"
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

    def compute_nra(self) -> dict:
        """
        Normalized Relative Advantage per GTBench paper §2.2:

            NRA(Mi, Mo, fs) = (Σ fs(Mi) − Σ fs(Mo)) / (Σ fs(Mi) + Σ fs(Mo))

        Range: [-1, 1].  > 0 means Mi outperforms Mo; < 0 the opposite.

        For zero-sum games, fs(M, m) = 1 win, 0.5 draw, 0 loss.
        For non-zero-sum games (PD), we use cumulative payoff fraction
        (player payoff / sum of both payoffs) so the metric stays bounded.

        Returns a nested dict:
            {
              "by_strategy": {strat: {wins, draws, losses, score, opp_score, nra}},
              "by_pair":     {(strat_i, strat_o): {... nra ...}},
            }
        """
        by_strategy: dict[str, dict] = defaultdict(
            lambda: {"score_self": 0.0, "score_opp": 0.0, "n": 0}
        )
        by_pair: dict[tuple, dict] = defaultdict(
            lambda: {"score_self": 0.0, "score_opp": 0.0, "n": 0}
        )

        for s in self.sessions:
            winner = s["winner"]
            game = s["game_name"]
            extra = s.get("extra", {}) or {}
            payoffs = extra.get("cumulative_payoffs") or {}

            for role, opp_role in (("player_a", "player_b"), ("player_b", "player_a")):
                strat_self = s.get(f"{role}_strategy", "unknown")
                strat_opp = s.get(f"{opp_role}_strategy", "unknown")

                if game == "prisoners_dilemma" and payoffs:
                    p_self = float(payoffs.get(role, 0))
                    p_opp = float(payoffs.get(opp_role, 0))
                    total = p_self + p_opp
                    if total <= 0:
                        f_self, f_opp = 0.5, 0.5
                    else:
                        f_self = p_self / total
                        f_opp = p_opp / total
                else:
                    if winner == role:
                        f_self, f_opp = 1.0, 0.0
                    elif winner == opp_role:
                        f_self, f_opp = 0.0, 1.0
                    else:
                        f_self, f_opp = 0.5, 0.5

                for bucket in (by_strategy[strat_self], by_pair[(strat_self, strat_opp)]):
                    bucket["score_self"] += f_self
                    bucket["score_opp"] += f_opp
                    bucket["n"] += 1

        def finalize(bucket: dict) -> dict:
            total = bucket["score_self"] + bucket["score_opp"]
            nra = (bucket["score_self"] - bucket["score_opp"]) / total if total > 0 else 0.0
            return {**bucket, "nra": round(nra, 4)}

        return {
            "by_strategy": {k: finalize(v) for k, v in by_strategy.items()},
            "by_pair": {f"{k[0]} vs {k[1]}": finalize(v) for k, v in by_pair.items()},
        }

    def compute_regret(self) -> dict:
        """
        Ex-post regret per GTBench paper §5 / Fig 5.

        Regret = best_fixed_strategy_payoff_against_opponent_actual − actual_payoff

        Implemented for two games where it is well-defined:

        * Iterated Prisoner's Dilemma — counterfactually play either always-
          cooperate or always-defect against the opponent's actual action
          sequence. The best of the two minus the actual cumulative payoff
          is the regret. Lower is better; 0 means the player was already
          ex-post optimal among constant strategies.

        * Blind Auction — best fixed bid in {0, 1, ..., max_bid} against the
          opponent's actual bid. If actual_bid > opp_bid: payoff = v − own_bid.
          Sweep all bids, pick the max payoff, regret = max − actual_payoff.

        Returns:
            {
              "by_strategy_per_game": { (strategy, game): {n, avg_regret, total_regret} },
              "by_strategy":          { strategy: {n, avg_regret} },
            }
        """
        per_pair: dict[tuple, dict] = defaultdict(lambda: {"total": 0.0, "n": 0})
        per_strategy: dict[str, dict] = defaultdict(lambda: {"total": 0.0, "n": 0})

        for s in self.sessions:
            game = s.get("game_name")
            extra = s.get("extra", {}) or {}

            if game == "prisoners_dilemma":
                actions_a = extra.get("player_a_actions") or []
                actions_b = extra.get("player_b_actions") or []
                payoffs = extra.get("cumulative_payoffs") or {}
                if not actions_a or not actions_b:
                    continue
                # Use PD's payoff matrix.
                pay = {
                    ("cooperate", "cooperate"): (3, 3),
                    ("cooperate", "defect"):    (0, 5),
                    ("defect",    "cooperate"): (5, 0),
                    ("defect",    "defect"):    (1, 1),
                }
                for role, opp_actions, role_idx in (
                    ("player_a", actions_b, 0),
                    ("player_b", actions_a, 1),
                ):
                    actual = float(payoffs.get(role, 0))
                    best = float("-inf")
                    for hypothetical in ("cooperate", "defect"):
                        cum = 0.0
                        for opp_act in opp_actions:
                            if role == "player_a":
                                cum += pay[(hypothetical, opp_act)][0]
                            else:
                                cum += pay[(opp_act, hypothetical)][1]
                        best = max(best, cum)
                    regret = max(0.0, best - actual)
                    strat = s.get(f"{role}_strategy", "unknown")
                    per_pair[(strat, game)]["total"] += regret
                    per_pair[(strat, game)]["n"] += 1
                    per_strategy[strat]["total"] += regret
                    per_strategy[strat]["n"] += 1

            elif game == "blind_auction":
                # The current Blind Auction state-record carries the final
                # bids and payoffs but not max_bid; we infer it from the
                # game_kwargs if present, else fall back to 100.
                bids = extra.get("bids") or {}
                payoffs = extra.get("payoffs") or {}
                valuations = extra.get("valuations") or {}
                if not bids:
                    continue
                max_bid = (s.get("game_kwargs") or {}).get("max_bid", 100)
                for role in ("player_a", "player_b"):
                    opp = "player_b" if role == "player_a" else "player_a"
                    own_v = float(valuations.get(role, 0))
                    opp_b = int(bids.get(opp, 0))
                    actual = float(payoffs.get(role, 0))
                    best = float("-inf")
                    for hypothetical in range(0, max_bid + 1):
                        if hypothetical > opp_b:
                            best = max(best, own_v - hypothetical)
                        elif hypothetical == opp_b:
                            best = max(best, (own_v - hypothetical) / 2)
                        else:
                            best = max(best, 0.0)
                    regret = max(0.0, best - actual)
                    strat = s.get(f"{role}_strategy", "unknown")
                    per_pair[(strat, game)]["total"] += regret
                    per_pair[(strat, game)]["n"] += 1
                    per_strategy[strat]["total"] += regret
                    per_strategy[strat]["n"] += 1

        def finalize(bucket: dict) -> dict:
            n = bucket["n"] or 1
            return {
                "n": bucket["n"],
                "total_regret": round(bucket["total"], 3),
                "avg_regret": round(bucket["total"] / n, 3),
            }

        return {
            "by_strategy_per_game": {f"{k[0]} on {k[1]}": finalize(v) for k, v in per_pair.items()},
            "by_strategy": {k: finalize(v) for k, v in per_strategy.items()},
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
        report = {
            "total_sessions": len(self.sessions),
            "win_rates": {str(k): v for k, v in self.compute_win_rates().items()},
            "invalid_rates": self.compute_invalid_rates(),
            "elo_ratings": self.get_elo_ratings(),
            "avg_turns": self.compute_avg_turns(),
            "nra": self.compute_nra(),
            "regret": self.compute_regret(),
            "pareto_efficiency": self.compute_pareto_efficiency(),
            "cooperation_rates": self.compute_cooperation_rates(),
        }
        if getattr(self, "error_profile", None):
            report["error_profile"] = self.error_profile
        return report

    def attach_error_profile(self, profile: dict) -> None:
        """Attach a pre-computed error-profile aggregate to the engine."""
        self.error_profile = profile

    def _update_elo(self, session: dict):
        """Update Elo ratings after a game."""
        a_id = session["player_a_id"]
        b_id = session["player_b_id"]
        winner = session["winner"]

        ra = self.elo_ratings[a_id]
        rb = self.elo_ratings[b_id]

        ea = 1 / (1 + 10 ** ((rb - ra) / 400))
        eb = 1 - ea

        if winner == "player_a":
            sa, sb = 1, 0
        elif winner == "player_b":
            sa, sb = 0, 1
        else:
            sa = sb = 0.5

        self.elo_ratings[a_id] = ra + self.ELO_K * (sa - ea)
        self.elo_ratings[b_id] = rb + self.ELO_K * (sb - eb)
