"""
Main experiment runner.

Usage:
    python experiments/run_experiments.py --config experiments/configs/exp1_reasoning.yaml
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.evaluator_agent import EvaluatorAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.player_agent import PlayerAgent
from config.settings import settings
from orchestration.game_graph import build_game_graph, create_initial_state

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_experiments")


def _mk_player(
    role: str,
    matchup: dict,
    config: dict,
) -> PlayerAgent:
    """Build a PlayerAgent for the given role from the matchup/config dicts."""
    strategy = matchup.get(
        f"player_{role}_strategy",
        config.get("strategy", "cot"),
    )
    model = matchup.get(
        f"player_{role}_model",
        config.get("model", settings.default_model),
    )
    provider = matchup.get(f"player_{role}_provider", "openai")
    return PlayerAgent(
        agent_id=f"{role.upper()}_{strategy}_{model}",
        strategy=strategy,
        model=model,
        temperature=config.get("temperature", settings.default_temperature),
        provider=provider,
    )


def run_experiment(config: dict) -> dict:
    experiment_name = config.get("experiment_name", "experiment")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = settings.results_dir / f"{experiment_name.replace(' ', '_')}_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Results directory: {out_dir}")

    graph = build_game_graph()
    orchestrator = OrchestratorAgent()
    evaluator = EvaluatorAgent()

    games = config.get("games", ["tictactoe"])
    rounds = config.get("rounds_per_matchup", 5)
    matchups = config.get("matchups", [])

    total = len(matchups) * len(games) * rounds
    completed = 0

    for matchup in matchups:
        for game_name in games:
            for round_n in range(rounds):
                player_a = _mk_player("a", matchup, config)
                player_b = _mk_player("b", matchup, config)

                assignment = orchestrator.assign_players(player_a, player_b, game_name)
                meta = {
                    "experiment": experiment_name,
                    "round": round_n + 1,
                    "matchup": f"{player_a.strategy_name} vs {player_b.strategy_name}",
                    "assignment": assignment,
                }

                initial_state = create_initial_state(
                    game_name=game_name,
                    player_a=player_a,
                    player_b=player_b,
                    evaluator=evaluator,
                    session_meta=meta,
                )
                initial_state["session_id"] = assignment["session_id"]

                try:
                    final_state = graph.invoke(
                        initial_state,
                        config={"recursion_limit": 200},
                    )
                    result = final_state["session_meta"].get("result", {})
                    completed += 1
                    logger.info(
                        f"[{completed}/{total}] {game_name} | "
                        f"{player_a.strategy_name} vs {player_b.strategy_name} | "
                        f"round {round_n + 1} | winner={result.get('winner', '?')}"
                    )
                except Exception as exc:
                    logger.error(f"Session failed: {exc}")

                time.sleep(0.5)

    formats = config.get("output_formats", ["json"])
    include_llm = bool(config.get("include_llm_summary", False))
    manifest = evaluator.generate_report(
        output_dir=out_dir,
        formats=formats,
        include_llm_summary=include_llm,
    )

    logger.info(f"Complete. {completed}/{total} sessions.")
    logger.info(f"Artifacts: {manifest}")
    return evaluator.full_report()


def main():
    parser = argparse.ArgumentParser(description="Run Agentic GTBench experiments.")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)

    run_experiment(config)


if __name__ == "__main__":
    main()
