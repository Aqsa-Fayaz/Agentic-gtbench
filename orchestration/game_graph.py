"""
LangGraph Game Orchestration
Defines the state machine that coordinates all agents.

State flow:
init_session → player_a_turn → validate_move_a → apply_move_a
→ check_terminal → player_b_turn → validate_move_b → apply_move_b
→ check_terminal → record_result → run_evaluator → [next game or done]
"""

import uuid
import logging
from typing import TypedDict, Literal, Optional, Any

from langgraph.graph import StateGraph, END

from games import load_game
from tools.move_validator import MoveValidatorTool

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# State Schema
# ─────────────────────────────────────────────

class GameSessionState(TypedDict):
    session_id: str
    game_name: str
    game_kwargs: dict                  # Constructor args for load_game (e.g. num_rounds)
    game: Any                          # BaseGame instance (not serialized)
    player_a: Any                      # PlayerAgent instance
    player_b: Any                      # PlayerAgent instance
    evaluator: Any                     # EvaluatorAgent instance
    current_player: str
    turn_number: int
    proposed_move: Optional[dict]
    move_history: list
    retry_count: int
    terminal: bool
    winner: Optional[str]
    forfeited_by: Optional[str]
    error_log: list
    session_meta: dict                 # Experiment metadata


# ─────────────────────────────────────────────
# Graph Nodes
# ─────────────────────────────────────────────

def node_init_session(state: GameSessionState) -> GameSessionState:
    """Initialize a new game session."""
    game_kwargs = state.get("game_kwargs") or {}
    game = load_game(state["game_name"], **game_kwargs)
    game.reset()
    logger.info(
        f"[{state['session_id']}] Game '{state['game_name']}' initialized "
        f"(kwargs={game_kwargs})."
    )
    return {
        **state,
        "game": game,
        "current_player": "player_a",
        "turn_number": 0,
        "proposed_move": None,
        "move_history": [],
        "retry_count": 0,
        "terminal": False,
        "winner": None,
        "forfeited_by": None,
        "error_log": [],
    }


def node_player_a_turn(state: GameSessionState) -> GameSessionState:
    """Player A decides their move."""
    player = state["player_a"]
    game = state["game"]
    legal_moves = game.get_legal_moves("player_a")
    game_state = game.get_state_for_player("player_a")

    try:
        move = player.decide(game_state, state["game_name"], legal_moves)
    except Exception as e:
        logger.error(f"[player_a] Decision error: {e}")
        move = None

    return {**state, "proposed_move": move, "current_player": "player_a"}


def node_player_b_turn(state: GameSessionState) -> GameSessionState:
    """Player B decides their move."""
    player = state["player_b"]
    game = state["game"]
    legal_moves = game.get_legal_moves("player_b")
    game_state = game.get_state_for_player("player_b")

    try:
        move = player.decide(game_state, state["game_name"], legal_moves)
    except Exception as e:
        logger.error(f"[player_b] Decision error: {e}")
        move = None

    return {**state, "proposed_move": move, "current_player": "player_b"}


def node_validate_move(state: GameSessionState) -> GameSessionState:
    """Validate the current player's proposed move."""
    player_key = state["current_player"]
    move = state["proposed_move"]
    game = state["game"]

    if move is None:
        return {**state, "proposed_move": None, "retry_count": state["retry_count"] + 1,
                "error_log": state["error_log"] + [f"{player_key}: no move proposed"]}

    valid, reason = game.is_valid_move(player_key, move)
    if not valid:
        logger.warning(f"[{player_key}] Invalid move {move}: {reason}")
        return {
            **state,
            "proposed_move": None,
            "retry_count": state["retry_count"] + 1,
            "error_log": state["error_log"] + [f"{player_key}: {reason}"],
        }

    return {**state, "retry_count": 0}   # Reset retries on valid move


def node_apply_move(state: GameSessionState) -> GameSessionState:
    """Apply the validated move to the game."""
    player_key = state["current_player"]
    move = state["proposed_move"]
    game = state["game"]

    game.make_move(player_key, move)
    new_turn = state["turn_number"] + 1

    move_log = {
        "turn": new_turn,
        "player": player_key,
        "move": move,
        "valid": True,
    }

    return {
        **state,
        "turn_number": new_turn,
        "proposed_move": None,
        "move_history": state["move_history"] + [move_log],
        "retry_count": 0,
    }


def node_check_terminal(state: GameSessionState) -> GameSessionState:
    """Check if the game has ended."""
    game = state["game"]
    if game.is_terminal():
        winner = game.get_winner()
        logger.info(f"[{state['session_id']}] Game over. Winner: {winner}")
        return {**state, "terminal": True, "winner": winner}
    return {**state, "terminal": False}


def node_handle_invalid(state: GameSessionState) -> GameSessionState:
    """Handle invalid move — logs forfeit if max retries exceeded."""
    current = state["current_player"]
    logger.error(f"[{current}] Max retries exceeded — forfeit.")
    winner = "player_b" if current == "player_a" else "player_a"
    return {**state, "terminal": True, "winner": winner, "forfeited_by": current}


def node_record_result(state: GameSessionState) -> GameSessionState:
    """Package session result for evaluator."""
    meta = state.get("session_meta", {})
    result = {
        "session_id": state["session_id"],
        "game_name": state["game_name"],
        "game_kwargs": state.get("game_kwargs", {}),
        "player_a_id": state["player_a"].agent_id,
        "player_b_id": state["player_b"].agent_id,
        "player_a_strategy": getattr(state["player_a"], "strategy_name", "unknown"),
        "player_b_strategy": getattr(state["player_b"], "strategy_name", "unknown"),
        "winner": state["winner"],
        "turns": state["turn_number"],
        "forfeited_by": state["forfeited_by"],
        "invalid_moves_a": state["player_a"].invalid_move_count,
        "invalid_moves_b": state["player_b"].invalid_move_count,
        "total_moves_a": state["player_a"].total_moves,
        "total_moves_b": state["player_b"].total_moves,
        "move_history": state["move_history"],
        "error_log": state["error_log"],
        "extra": {},
    }

    # Game-specific extras
    game = state["game"]
    if hasattr(game, "get_pareto_efficiency"):
        result["extra"]["pareto_efficiency"] = game.get_pareto_efficiency()
    if hasattr(game, "round_history"):
        actions_a = [r["player_a_action"] for r in game.round_history]
        actions_b = [r["player_b_action"] for r in game.round_history]
        result["extra"]["player_a_actions"] = actions_a
        result["extra"]["player_b_actions"] = actions_b
    if hasattr(game, "cumulative_payoffs"):
        result["extra"]["cumulative_payoffs"] = dict(game.cumulative_payoffs)

    return {**state, "session_meta": {**meta, "result": result}}


def node_run_evaluator(state: GameSessionState) -> GameSessionState:
    """Send result to evaluator agent."""
    evaluator = state["evaluator"]
    result = state["session_meta"].get("result", {})
    if result:
        evaluator.record_session(result)
    return state


# ─────────────────────────────────────────────
# Routing Functions (Conditional Edges)
# ─────────────────────────────────────────────

def route_after_validate(state: GameSessionState) -> Literal["apply_move", "retry_a", "retry_b", "forfeit"]:
    """Route based on move validity and retry count."""
    move = state["proposed_move"]
    retries = state["retry_count"]
    current = state["current_player"]

    if move is not None and retries == 0:
        return "apply_move"
    elif retries >= 3:
        return "forfeit"
    elif current == "player_a":
        return "retry_a"
    else:
        return "retry_b"


def route_after_terminal_check(state: GameSessionState) -> Literal["player_b_turn", "player_a_turn", "record_result"]:
    if state["terminal"]:
        return "record_result"
    if state["current_player"] == "player_a":
        return "player_b_turn"
    return "player_a_turn"


# ─────────────────────────────────────────────
# Graph Builder
# ─────────────────────────────────────────────

def build_game_graph():
    """Compile and return the LangGraph game workflow."""

    workflow = StateGraph(GameSessionState)

    # Add nodes
    workflow.add_node("init_session", node_init_session)
    workflow.add_node("player_a_turn", node_player_a_turn)
    workflow.add_node("player_b_turn", node_player_b_turn)
    workflow.add_node("validate_move", node_validate_move)
    workflow.add_node("apply_move", node_apply_move)
    workflow.add_node("check_terminal", node_check_terminal)
    workflow.add_node("handle_invalid", node_handle_invalid)
    workflow.add_node("record_result", node_record_result)
    workflow.add_node("run_evaluator", node_run_evaluator)

    # Set entry point
    workflow.set_entry_point("init_session")

    # Static edges
    workflow.add_edge("init_session", "player_a_turn")
    workflow.add_edge("player_a_turn", "validate_move")
    workflow.add_edge("player_b_turn", "validate_move")
    workflow.add_edge("apply_move", "check_terminal")
    workflow.add_edge("handle_invalid", "record_result")
    workflow.add_edge("record_result", "run_evaluator")
    workflow.add_edge("run_evaluator", END)

    # Conditional: after validate
    workflow.add_conditional_edges(
        "validate_move",
        route_after_validate,
        {
            "apply_move": "apply_move",
            "retry_a": "player_a_turn",
            "retry_b": "player_b_turn",
            "forfeit": "handle_invalid",
        },
    )

    # Conditional: after terminal check
    workflow.add_conditional_edges(
        "check_terminal",
        route_after_terminal_check,
        {
            "player_a_turn": "player_a_turn",
            "player_b_turn": "player_b_turn",
            "record_result": "record_result",
        },
    )

    return workflow.compile()


def create_initial_state(
    game_name: str,
    player_a,
    player_b,
    evaluator,
    session_meta: dict = None,
    game_kwargs: dict = None,
) -> GameSessionState:
    """Helper to build initial state for a session."""
    return GameSessionState(
        session_id=str(uuid.uuid4())[:8],
        game_name=game_name,
        game_kwargs=game_kwargs or {},
        game=None,                    # Set by init_session node
        player_a=player_a,
        player_b=player_b,
        evaluator=evaluator,
        current_player="player_a",
        turn_number=0,
        proposed_move=None,
        move_history=[],
        retry_count=0,
        terminal=False,
        winner=None,
        forfeited_by=None,
        error_log=[],
        session_meta=session_meta or {},
    )
