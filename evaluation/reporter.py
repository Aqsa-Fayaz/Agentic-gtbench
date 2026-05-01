"""
Reporter — exports tournament results in JSON, CSV, and HTML formats.

The JSON export is the canonical machine-readable artifact.
The CSV export is a flat per-session table for spreadsheet analysis.
The HTML export is a self-contained page with simple tables for humans.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Reporter:
    """Write tournament reports in multiple formats."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def write_json(self, report: dict, sessions: Iterable[dict]) -> Path:
        """Dump aggregated metrics + raw sessions into a JSON file."""
        path = self.output_dir / "report.json"
        payload = {
            "generated_at": _utcnow_iso(),
            "report": report,
            "sessions": list(sessions),
        }
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        return path

    def write_csv(self, sessions: Iterable[dict]) -> Path:
        """Flatten each session into one CSV row."""
        path = self.output_dir / "sessions.csv"
        rows = list(sessions)
        if not rows:
            path.write_text("", encoding="utf-8")
            return path

        fieldnames = [
            "session_id",
            "game_name",
            "player_a_id",
            "player_b_id",
            "player_a_strategy",
            "player_b_strategy",
            "winner",
            "turns",
            "forfeited_by",
            "invalid_moves_a",
            "invalid_moves_b",
            "total_moves_a",
            "total_moves_b",
            "pareto_efficiency",
        ]
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for s in rows:
                extra = s.get("extra", {}) or {}
                writer.writerow(
                    {
                        "session_id": s.get("session_id"),
                        "game_name": s.get("game_name"),
                        "player_a_id": s.get("player_a_id"),
                        "player_b_id": s.get("player_b_id"),
                        "player_a_strategy": s.get("player_a_strategy"),
                        "player_b_strategy": s.get("player_b_strategy"),
                        "winner": s.get("winner"),
                        "turns": s.get("turns"),
                        "forfeited_by": s.get("forfeited_by"),
                        "invalid_moves_a": s.get("invalid_moves_a"),
                        "invalid_moves_b": s.get("invalid_moves_b"),
                        "total_moves_a": s.get("total_moves_a"),
                        "total_moves_b": s.get("total_moves_b"),
                        "pareto_efficiency": extra.get("pareto_efficiency"),
                    }
                )
        return path

    def write_html(self, report: dict, sessions: Iterable[dict]) -> Path:
        """Render a simple self-contained HTML report."""
        path = self.output_dir / "report.html"
        sessions = list(sessions)
        html = _render_html(report, sessions)
        path.write_text(html, encoding="utf-8")
        return path


def _render_table(headers: list[str], rows: list[list]) -> str:
    head_html = "".join(f"<th>{h}</th>" for h in headers)
    body_html = "".join(
        "<tr>" + "".join(f"<td>{_fmt(v)}</td>" for v in r) + "</tr>" for r in rows
    )
    return (
        f"<table class='tbl'><thead><tr>{head_html}</tr></thead>"
        f"<tbody>{body_html}</tbody></table>"
    )


def _fmt(v) -> str:
    if isinstance(v, float):
        return f"{v:.3f}"
    if v is None:
        return ""
    return str(v)


def _render_html(report: dict, sessions: list[dict]) -> str:
    generated = _utcnow_iso()
    total = report.get("total_sessions", len(sessions))

    win_rows = [
        [v.get("strategy"), v.get("game"), v.get("games"),
         v.get("win_rate"), v.get("draw_rate"), v.get("loss_rate")]
        for v in (report.get("win_rates") or {}).values()
    ]
    invalid_rows = [
        [agent_id, v.get("invalid_moves"), v.get("total_moves"), v.get("invalid_rate")]
        for agent_id, v in (report.get("invalid_rates") or {}).items()
    ]
    elo_rows = [[a, r] for a, r in (report.get("elo_ratings") or {}).items()]
    turn_rows = [
        [v.get("strategy"), v.get("game"), v.get("avg_turns"),
         v.get("min_turns"), v.get("max_turns"), v.get("samples")]
        for v in (report.get("avg_turns") or {}).values()
    ]
    coop_rows = [
        [strat, v.get("cooperation_rate"), v.get("total_actions")]
        for strat, v in (report.get("cooperation_rates") or {}).items()
    ]
    pareto_rows = [
        [matchup, v.get("avg_pareto"), v.get("samples")]
        for matchup, v in (report.get("pareto_efficiency") or {}).items()
    ]
    nra = report.get("nra") or {}
    nra_strategy_rows = [
        [strat, v.get("n"), v.get("score_self"), v.get("score_opp"), v.get("nra")]
        for strat, v in (nra.get("by_strategy") or {}).items()
    ]
    nra_pair_rows = [
        [pair, v.get("n"), v.get("score_self"), v.get("score_opp"), v.get("nra")]
        for pair, v in (nra.get("by_pair") or {}).items()
    ]
    regret = report.get("regret") or {}
    regret_strategy_rows = [
        [strat, v.get("n"), v.get("avg_regret"), v.get("total_regret")]
        for strat, v in (regret.get("by_strategy") or {}).items()
    ]
    regret_pair_rows = [
        [pair, v.get("n"), v.get("avg_regret"), v.get("total_regret")]
        for pair, v in (regret.get("by_strategy_per_game") or {}).items()
    ]
    error_rows = []
    error_categories = (
        "misinterpretation",
        "factual_error",
        "overconfidence",
        "calculation_mistake",
        "endgame_misdetection",
        "ok",
    )
    for strat, counts in (report.get("error_profile") or {}).items():
        error_rows.append(
            [strat, counts.get("total", 0)]
            + [counts.get(c, 0) for c in error_categories]
        )
    session_rows = [
        [
            s.get("session_id"), s.get("game_name"),
            s.get("player_a_strategy"), s.get("player_b_strategy"),
            s.get("winner"), s.get("turns"), s.get("forfeited_by"),
        ]
        for s in sessions
    ]

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Agentic GTBench Report</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 2em; color: #222; }}
h1 {{ color: #1a365d; }}
h2 {{ margin-top: 1.8em; border-bottom: 1px solid #ccc; padding-bottom: .2em; }}
.tbl {{ border-collapse: collapse; margin: .5em 0 2em; width: 100%; font-size: 14px; }}
.tbl th, .tbl td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
.tbl th {{ background: #f0f4fa; }}
.meta {{ color: #666; font-size: 13px; }}
</style></head>
<body>
<h1>Agentic GTBench — Tournament Report</h1>
<p class="meta">Generated: {generated} &middot; Total sessions: {total}</p>

<h2>Win Rates (by strategy × game)</h2>
{_render_table(['Strategy', 'Game', 'Games', 'Win', 'Draw', 'Loss'], win_rows)}

<h2>Invalid Move Rates (per agent)</h2>
{_render_table(['Agent', 'Invalid', 'Total', 'Rate'], invalid_rows)}

<h2>Elo Ratings</h2>
{_render_table(['Agent', 'Elo'], elo_rows)}

<h2>Average Turns to Win</h2>
{_render_table(['Strategy', 'Game', 'Avg', 'Min', 'Max', 'Samples'], turn_rows)}

<h2>Normalized Relative Advantage — by strategy (paper §2.2)</h2>
<p class="meta">NRA &gt; 0: this strategy outperforms its opponents on average. Range [-1, 1].</p>
{_render_table(['Strategy', 'Matches', 'Score Self', 'Score Opp', 'NRA'], nra_strategy_rows)}

<h2>Normalized Relative Advantage — by matchup</h2>
{_render_table(['Pairing (self vs opp)', 'Matches', 'Score Self', 'Score Opp', 'NRA'], nra_pair_rows)}

<h2>Ex-Post Regret — by strategy (paper §5)</h2>
<p class="meta">Lower is better. 0 means the player's actual play matched the best fixed-strategy ex-post.</p>
{_render_table(['Strategy', 'Samples', 'Avg Regret', 'Total Regret'], regret_strategy_rows)}

<h2>Ex-Post Regret — per (strategy × game)</h2>
{_render_table(['Strategy on Game', 'Samples', 'Avg Regret', 'Total Regret'], regret_pair_rows)}

<h2>Cooperation Rates (Prisoner's Dilemma)</h2>
{_render_table(['Strategy', 'Cooperation Rate', 'Total Actions'], coop_rows)}

<h2>Pareto Efficiency (Prisoner's Dilemma)</h2>
{_render_table(['Matchup', 'Avg Pareto', 'Samples'], pareto_rows)}

<h2>Error Profile (LLM-as-judge over losing-side moves)</h2>
{_render_table(
    ['Strategy', 'Moves Judged'] + [c.replace('_', ' ').title() for c in error_categories],
    error_rows,
)}

<h2>Raw Sessions</h2>
{_render_table(['Session', 'Game', 'Strategy A', 'Strategy B', 'Winner', 'Turns', 'Forfeit'], session_rows)}
</body></html>
"""
