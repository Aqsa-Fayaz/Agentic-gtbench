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
from agents.mcts_agent import MCTSAgent
from agents.orchestrator_agent import OrchestratorAgent
from agents.player_agent import PlayerAgent
from agents.random_agent import RandomAgent
from agents.tit_for_tat_agent import TitForTatAgent
from config.settings import settings
from orchestration.game_graph import build_game_graph, create_initial_state
from tools.history_manager import HistoryManagerTool
from tools.move_validator import MoveValidatorTool
from tools.state_tracker import StateTrackerTool
from tools.strategy_analyzer import StrategyAnalyzerTool

CONVENTIONAL_AGENTS = {
    "random": RandomAgent,
    "tit_for_tat": TitForTatAgent,
    "mcts": MCTSAgent,
}

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

    # Conventional (non-LLM) agents short-circuit the LLM provider plumbing.
    if strategy in CONVENTIONAL_AGENTS:
        cls = CONVENTIONAL_AGENTS[strategy]
        # MCTS accepts a per-matchup simulation budget; default to paper's 1000.
        if strategy == "mcts":
            n_sims = matchup.get("mcts_simulations", config.get("mcts_simulations", 1000))
            return cls(agent_id=f"{role.upper()}_{strategy}_{n_sims}", n_simulations=n_sims)
        return cls(agent_id=f"{role.upper()}_{strategy}")

    provider = matchup.get(
        f"player_{role}_provider",
        config.get("provider", settings.default_provider),
    )
    # Pick a sensible default model for the chosen provider when the YAML
    # doesn't pin one explicitly.
    fallback_model = settings.resolve_model(provider, config.get("model"))
    model = matchup.get(f"player_{role}_model", fallback_model)
    # ReAct is the only strategy that actually invokes tools inside its loop;
    # other strategies ignore the bound list, so we only pay the import cost
    # for ReAct matchups.
    tools = []
    if strategy.lower() == "react":
        tools = [
            MoveValidatorTool(),
            StateTrackerTool(),
            StrategyAnalyzerTool(),
            HistoryManagerTool(),
        ]
    return PlayerAgent(
        agent_id=f"{role.upper()}_{strategy}_{model}",
        strategy=strategy,
        model=model,
        temperature=config.get("temperature", settings.default_temperature),
        provider=provider,
        tools=tools,
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
    # Per-game constructor kwargs, e.g. {"prisoners_dilemma": {"num_rounds": 5}}
    game_kwargs_map: dict = config.get("game_kwargs", {}) or {}

    # `games` may be either a flat list of names OR a list of dicts with
    # inline kwargs: [{name: prisoners_dilemma, kwargs: {num_rounds: 5}}, ...]
    normalized_games: list[tuple[str, dict]] = []
    for entry in games:
        if isinstance(entry, dict):
            name = entry["name"]
            kwargs = entry.get("kwargs", {})
        else:
            name = entry
            kwargs = game_kwargs_map.get(name, {})
        normalized_games.append((name, kwargs))

    total = len(matchups) * len(normalized_games) * rounds
    completed = 0

    for matchup in matchups:
        for game_name, game_kwargs in normalized_games:
            for round_n in range(rounds):
                player_a = _mk_player("a", matchup, config)
                player_b = _mk_player("b", matchup, config)

                assignment = orchestrator.assign_players(player_a, player_b, game_name)
                meta = {
                    "experiment": experiment_name,
                    "round": round_n + 1,
                    "matchup": f"{player_a.strategy_name} vs {player_b.strategy_name}",
                    "assignment": assignment,
                    "game_kwargs": game_kwargs,
                }

                initial_state = create_initial_state(
                    game_name=game_name,
                    player_a=player_a,
                    player_b=player_b,
                    evaluator=evaluator,
                    session_meta=meta,
                    game_kwargs=game_kwargs,
                )
                initial_state["session_id"] = assignment["session_id"]

                try:
                    final_state = graph.invoke(initial_state)
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
