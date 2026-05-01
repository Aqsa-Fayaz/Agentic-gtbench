"""
MCTSAgent — Monte Carlo Tree Search baseline (paper §3.2 / Appendix A3.1).

Generic search that works on any BaseGame whose state can be reconstructed
via `from_state(state)` and whose move space is enumerable via
`get_legal_moves(player)`.

Algorithm (vanilla MCTS with UCB1):
  1. SELECT      — descend the in-memory tree picking the child that
                   maximises UCB1 = mean_reward + c * sqrt(ln(parent_visits)
                   / child_visits) until reaching a leaf.
  2. EXPAND      — if the leaf is non-terminal, add one untried child.
  3. SIMULATE    — play uniformly-random moves from that child until terminal.
  4. BACKPROPAGATE — push the rollout's reward back up the visited path,
                   alternating sign by perspective (zero-sum convention).

Reward convention (from the perspective of the player whose turn it was at
the ROOT):  +1 win, 0 loss, 0.5 draw.  For non-zero-sum games (PD,
Negotiation, Blind Auction) we fall back to a payoff-based reward
normalised to [0, 1] using the rollout's per-player payoff fraction.

Performance note:
  Each move spends `n_simulations` rollouts. Rollouts call into the game's
  pure-Python `make_move` so latency is dominated by Python overhead, not
  network calls. On TicTacToe / Pig 1000 sims completes in well under a
  second; on Connect-4 / Breakthrough expect a few seconds per move.
"""

from __future__ import annotations

import math
import random
from typing import Optional

from agents.base_agent import BaseAgent
from games import load_game


def _move_key(move: dict) -> tuple:
    """Hashable representation of a move dict for tree-node bookkeeping."""
    def freeze(v):
        if isinstance(v, dict):
            return tuple(sorted((k, freeze(x)) for k, x in v.items()))
        if isinstance(v, list):
            return tuple(freeze(x) for x in v)
        return v
    return freeze(move)


class _Node:
    __slots__ = ("parent", "move", "player_to_move", "children",
                 "untried_moves", "visits", "total_reward")

    def __init__(self, parent, move, player_to_move, untried_moves):
        self.parent = parent
        self.move = move
        self.player_to_move = player_to_move
        self.children: dict = {}                    # _move_key(move) -> _Node
        self.untried_moves: list = list(untried_moves)
        self.visits = 0
        self.total_reward = 0.0

    def is_fully_expanded(self) -> bool:
        return not self.untried_moves

    def best_child_ucb(self, c: float) -> "_Node":
        """Pick the child maximising UCB1."""
        log_n = math.log(self.visits) if self.visits > 0 else 0.0
        best = None
        best_score = float("-inf")
        for child in self.children.values():
            if child.visits == 0:
                return child
            avg = child.total_reward / child.visits
            score = avg + c * math.sqrt(log_n / child.visits)
            if score > best_score:
                best_score = score
                best = child
        return best


class MCTSAgent(BaseAgent):
    """
    Conventional MCTS player. Conforms to the PlayerAgent interface so it can
    drop into the LangGraph runtime exactly like an LLM-driven PlayerAgent.

    Constructor mirrors PlayerAgent's so it can be built from the same factory.
    """

    SUPPORTED_STRATEGIES = ("mcts",)
    EXPLORATION_C = math.sqrt(2)
    DEFAULT_SIMULATIONS = 1000
    ROLLOUT_DEPTH_CAP = 200          # safety net on infinite rollouts

    def __init__(
        self,
        agent_id: str,
        n_simulations: int = DEFAULT_SIMULATIONS,
        seed: Optional[int] = None,
        # Accept-but-ignore the rest so this is a drop-in for PlayerAgent.
        strategy: str = "mcts",
        model: str = "mcts",
        temperature: float = 0.0,
        provider: str = "none",
        tools: Optional[list] = None,
        **kwargs,
    ):
        super().__init__(agent_id=agent_id, model=model, temperature=temperature)
        self.strategy_name = "mcts"
        self.provider = provider
        self.tools = tools or []
        self.n_simulations = int(n_simulations)
        self._rng = random.Random(seed)
        self.invalid_move_count = 0
        self.total_moves = 0
        self.move_history: list[dict] = []
        self._root_player: Optional[str] = None     # set per call to decide()
        self._game_name: Optional[str] = None

    # ─────────────────────────────────────────────
    # Public API (PlayerAgent contract)
    # ─────────────────────────────────────────────

    def decide(self, game_state: dict, game_name: str, legal_moves: list) -> dict:
        if not legal_moves:
            raise RuntimeError(f"{self.agent_id}: no legal moves available.")
        self.total_moves += 1
        self._root_player = game_state.get("current_player")
        if self._root_player is None:
            # Reconstruct the game just to ask whose turn it is.
            self._root_player = self._restore(game_name, game_state).get_current_player()
        self._game_name = game_name

        root = _Node(parent=None, move=None,
                     player_to_move=self._root_player,
                     untried_moves=legal_moves)

        for _ in range(self.n_simulations):
            game = self._restore(game_name, game_state)
            node = self._select(root, game)
            node = self._expand(node, game)
            reward = self._rollout(game)
            self._backpropagate(node, reward)

        # Pick the most-visited child of the root (robust child selection).
        if not root.children:
            return self._rng.choice(legal_moves)
        best_key = max(root.children, key=lambda k: root.children[k].visits)
        return root.children[best_key].move

    def reset_session(self) -> None:
        pass

    def get_stats(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "strategy": "mcts",
            "model": f"mcts_{self.n_simulations}",
            "total_moves": self.total_moves,
            "invalid_moves": 0,
            "invalid_rate": 0.0,
        }

    # ─────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────

    def _restore(self, game_name: str, state: dict):
        """Rebuild a game instance from the state dict."""
        game_cls = load_game(game_name).__class__
        try:
            return game_cls.from_state(state)
        except NotImplementedError:
            # No from_state — bail out by returning a fresh game and hoping
            # the rollout still tells us something useful.
            return load_game(game_name)

    def _select(self, root: _Node, game) -> _Node:
        """
        Descend until we hit a non-fully-expanded node or a terminal state.
        We re-derive the active player from the LIVE game on every step rather
        than trusting node.player_to_move — this handles stochastic games
        (Pig, Kuhn Poker, etc.) where the same move can lead to different
        active-player outcomes across rollouts.
        """
        node = root
        while not game.is_terminal() and node.is_fully_expanded() and node.children:
            child = node.best_child_ucb(self.EXPLORATION_C)
            mover = game.get_current_player()
            try:
                game.make_move(mover, child.move)
            except Exception:
                # Tree state diverged from this rollout's reality (legal in
                # one rollout, illegal in another due to stochasticity).
                # Stop descending and treat current node as the leaf.
                break
            node = child
        return node

    def _expand(self, node: _Node, game) -> _Node:
        if game.is_terminal() or not node.untried_moves:
            return node
        mover = game.get_current_player()
        # Re-filter untried moves against the live legal set; stochastic games
        # may have a different legal set than what was cached at node creation.
        live_legal = game.get_legal_moves(mover)
        live_keys = {_move_key(m) for m in live_legal}
        valid_untried = [m for m in node.untried_moves if _move_key(m) in live_keys]
        if not valid_untried:
            return node
        idx = self._rng.randrange(len(valid_untried))
        move = valid_untried[idx]
        # Remove from the original untried list (preserve key identity).
        node.untried_moves = [m for m in node.untried_moves if _move_key(m) != _move_key(move)]

        try:
            game.make_move(mover, move)
        except Exception:
            return node

        next_player = game.get_current_player() if not game.is_terminal() else mover
        legal_after = game.get_legal_moves(next_player) if not game.is_terminal() else []
        child = _Node(parent=node, move=move,
                      player_to_move=next_player,
                      untried_moves=legal_after)
        node.children[_move_key(move)] = child
        return child

    def _rollout(self, game) -> float:
        """Random play to terminal; returns reward from the root player's perspective."""
        depth = 0
        while not game.is_terminal() and depth < self.ROLLOUT_DEPTH_CAP:
            current = game.get_current_player()
            legal = game.get_legal_moves(current)
            if not legal:
                break
            move = self._rng.choice(legal)
            try:
                game.make_move(current, move)
            except Exception:
                break
            depth += 1
        return self._reward_for_root(game)

    def _reward_for_root(self, game) -> float:
        """
        Map terminal state to [0, 1] reward from `self._root_player`'s view.
        For zero-sum games: 1 win, 0 loss, 0.5 draw (or non-terminal hit cap).
        For payoff games (PD / Negotiation / Blind Auction): payoff fraction.
        """
        # Try payoff-based first (richer signal).
        for attr in ("cumulative_payoffs", "payoffs"):
            payoffs = getattr(game, attr, None)
            if payoffs:
                me = float(payoffs.get(self._root_player, 0))
                opp_key = "player_b" if self._root_player == "player_a" else "player_a"
                opp = float(payoffs.get(opp_key, 0))
                total = me + opp
                if total > 0:
                    return me / total
                if me == opp:
                    return 0.5
                return 1.0 if me > opp else 0.0

        winner = game.get_winner() if game.is_terminal() else None
        if winner == self._root_player:
            return 1.0
        if winner is None or winner == "draw":
            return 0.5
        return 0.0

    def _backpropagate(self, node: _Node, reward: float) -> None:
        """
        Push reward up the tree. Reward is from the ROOT player's view; flip
        sign for nodes whose player_to_move is the opponent so each node's
        mean_reward reflects "good move FOR this node's mover".
        """
        opponent_root = (
            "player_b" if self._root_player == "player_a" else "player_a"
        )
        while node is not None:
            node.visits += 1
            if node.parent is None:
                # Root: reward from root's perspective.
                node.total_reward += reward
            else:
                # The node's stored move was played BY node.parent.player_to_move.
                # If that mover is the root player, this move was "ours" and we
                # credit it with `reward`. Otherwise it was the opponent's move
                # and we credit the inverse so the search picks moves that
                # MINIMISE opponent reward.
                mover = node.parent.player_to_move
                node.total_reward += reward if mover == self._root_player else (1.0 - reward)
            node = node.parent
