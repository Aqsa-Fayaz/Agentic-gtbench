# 🛠️ IMPLEMENTATION PLAN — Agentic GTBench
## Step-by-Step Guide for Cursor IDE

> **Before starting**: Open this project in Cursor. Keep `README.md` and this file
> side-by-side. Execute phases in order — each phase builds on the last.

---

## 📋 Phase 0: Project Setup & Environment (Day 1)

### 0.1 — Initialize Python Environment
```bash
# In Cursor terminal
python -m venv venv
source venv/bin/activate          # or venv\Scripts\activate on Windows
pip install -r requirements.txt
cp .env.example .env
```

### 0.2 — Set API Keys in `.env`
```
OPENAI_API_KEY=sk-...
GROQ_API_KEY=gsk_...             # Free tier for Llama-3
ANTHROPIC_API_KEY=...            # Optional
LOG_LEVEL=INFO
RESULTS_DIR=./results
```

### 0.3 — Verify Setup
```bash
python -c "from config.settings import settings; print(settings.openai_api_key[:8])"
python -m pytest tests/ -v --tb=short
```

**Expected output**: All tests pass (they use mocks at this stage).

---

## 📋 Phase 1: Game Environments (Day 1–2)

> **Goal**: Build all 5 game environments with a consistent abstract interface.
> Each game must: accept a move, validate it, update state, detect terminal conditions.

### Step 1.1 — Implement `games/base_game.py`
The abstract `BaseGame` class defines the contract all games must follow:
- `reset()` → fresh game state
- `get_state()` → serializable state dict for LLM context
- `is_valid_move(player, move)` → bool
- `make_move(player, move)` → updated state
- `is_terminal()` → bool
- `get_winner()` → "player_a" | "player_b" | "draw" | None
- `get_legal_moves(player)` → list of valid moves
- `render()` → human-readable board string

### Step 1.2 — Implement `games/tictactoe.py`
- 3×3 board, players are 'X' and 'O'
- State: `{"board": [[...]], "current_player": "X", "turn": 3}`
- Move format: `{"row": 0, "col": 2}`
- Win check: rows, cols, diagonals
- Terminal: win or full board

### Step 1.3 — Implement `games/connect4.py`
- 6×7 board, gravity applies (pieces fall to lowest row)
- Move format: `{"col": 3}` — column only
- Win check: horizontal, vertical, both diagonals, 4-in-a-row
- Terminal: win or full board (42 pieces)

### Step 1.4 — Implement `games/nim.py`
- Start with configurable piles `[3, 5, 7]`
- Move format: `{"pile": 1, "count": 3}` — take from pile
- Classic misère variant: player who takes last piece **loses**
- Terminal: all piles empty

### Step 1.5 — Implement `games/prisoners_dilemma.py`
- **Iterated** Prisoner's Dilemma — multiple rounds
- Move format: `{"action": "cooperate"}` or `{"action": "defect"}`
- Payoff matrix: CC→(3,3), CD→(0,5), DC→(5,0), DD→(1,1)
- Track cumulative payoff per player across N rounds
- Terminal: after configured number of rounds (default 10)

### Step 1.6 — Implement `games/kuhn_poker.py`
- 3-card deck (J, Q, K), each player gets 1 card
- Actions: `{"action": "bet"}` or `{"action": "pass"}`
- Showdown logic: higher card wins if called
- Incomplete information: players see only their own card
- Terminal: one player folds or showdown occurs

### Step 1.7 — Write Tests `tests/test_games.py`
```python
# Test each game for:
# - Valid move accepted
# - Invalid move rejected
# - Terminal detection correct
# - get_legal_moves() returns non-empty for non-terminal
# - Winner correctly identified
```

**Cursor Prompt to use**:
> "Implement `games/tictactoe.py` following the `BaseGame` interface in `games/base_game.py`. 
> Include a `render()` method that prints a visual board. All methods must match the 
> abstract interface exactly. Write comprehensive docstrings."

---

## 📋 Phase 2: Agent Tools (Day 2–3)

> **Goal**: Build 4 standalone tools the agents will call. Tools must be
> pure functions or simple classes — stateless or minimally stateful.

### Step 2.1 — `tools/move_validator.py`
```python
class MoveValidatorTool:
    """
    LangChain-compatible tool. Agents call this BEFORE submitting a move.
    Input: {"game_name": str, "state": dict, "player": str, "move": dict}
    Output: {"valid": bool, "reason": str, "legal_moves": list}
    """
```

### Step 2.2 — `tools/state_tracker.py`
```python
class StateTrackerTool:
    """
    Converts raw game state dict into a rich natural-language description
    the LLM can reason about. Also extracts key features (empty cells, 
    threats, winning moves available).
    Input: {"game_name": str, "state": dict, "player": str}
    Output: {"description": str, "threats": list, "opportunities": list}
    """
```

### Step 2.3 — `tools/strategy_analyzer.py`
```python
class StrategyAnalyzerTool:
    """
    Analyzes opponent's move history to infer their strategy pattern.
    For Prisoner's Dilemma: detects Tit-for-Tat, Always Defect, etc.
    For games: detects aggressive vs defensive patterns.
    Input: {"move_history": list, "game_name": str}
    Output: {"inferred_strategy": str, "confidence": float, "recommendation": str}
    """
```

### Step 2.4 — `tools/history_manager.py`
```python
class HistoryManagerTool:
    """
    Persists game sessions to disk (JSON). Supports:
    - save_game(session_id, game_log)
    - load_game(session_id)
    - list_sessions()
    - get_stats(agent_id)  # aggregate win/loss/draw across sessions
    """
```

### Step 2.5 — Register Tools in LangChain Format
All tools must implement the LangChain `BaseTool` interface so they can be
bound to agents' tool-calling capability:
```python
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

class MoveInput(BaseModel):
    game_name: str = Field(description="Name of the current game")
    state: dict = Field(description="Current serialized game state")
    player: str = Field(description="'player_a' or 'player_b'")
    move: dict = Field(description="Proposed move as dict")

class MoveValidatorTool(BaseTool):
    name = "validate_move"
    description = "Validates if a proposed move is legal before playing it"
    args_schema = MoveInput
    def _run(self, **kwargs): ...
```

---

## 📋 Phase 3: Reasoning Strategies (Day 3)

> **Goal**: Implement 4 interchangeable reasoning strategies as prompt builders.
> Each strategy wraps an LLM call with different prompt construction logic.

### Step 3.1 — `agents/strategies/direct.py`
```
System: "You are a strategic game-playing AI. You must choose the best move."
User:   "[GAME STATE]\n{state_description}\nYour move options: {legal_moves}\nRespond with JSON: {\"move\": ...}"
```

### Step 3.2 — `agents/strategies/cot.py`
```
System: "You are a strategic game-playing AI. Think step by step before deciding."
User:   "[GAME STATE]\n{state_description}\n
         Step 1: What is the current board situation?
         Step 2: What are the immediate threats or opportunities?
         Step 3: Evaluate each legal move: {legal_moves}
         Step 4: Which move leads to the best outcome and why?
         Final Answer — respond with JSON: {\"move\": ..., \"reasoning\": ...}"
```

### Step 3.3 — `agents/strategies/tot.py`
Tree of Thoughts — generate N candidate moves, evaluate each, pick best:
```
Turn 1 (Generate): "Generate 3 distinct candidate moves from {legal_moves} with brief justification for each."
Turn 2 (Evaluate): "For each candidate move below, score it 1-10 for strategic value:\n{candidates}"
Turn 3 (Select):   "Based on scores, select the best move. Respond with JSON: {\"move\": ...}"
```
Implementation uses 3 sequential LLM calls per turn.

### Step 3.4 — `agents/strategies/react.py`
ReAct — interleaves Thought → Action → Observation cycles:
```
System: "You reason and act in cycles. Format: Thought: ... Action: tool_name(args) Observation: ..."
Cycle 1: Thought about state → Action: call StateTrackerTool → Observation: threats/opportunities
Cycle 2: Thought about move → Action: call MoveValidatorTool → Observation: valid/invalid
Cycle 3: Final decision → Action: MOVE({move_json})
```
Implementation uses LangChain's `AgentExecutor` with tool binding.

---

## 📋 Phase 4: Core Agents (Day 4)

> **Goal**: Build the three agent types. All agents use `langchain_openai.ChatOpenAI`.

### Step 4.1 — `agents/base_agent.py`
```python
class BaseAgent(ABC):
    def __init__(self, agent_id: str, model: str, temperature: float): ...
    @abstractmethod
    def decide(self, game_state: dict, game_name: str, legal_moves: list) -> dict: ...
    def get_history(self) -> list: ...
    def reset_history(self): ...
```

### Step 4.2 — `agents/player_agent.py`
```python
class PlayerAgent(BaseAgent):
    """
    Game-playing agent. Initialized with a strategy and a set of tools.
    
    decide() flow:
    1. Call StateTrackerTool to get natural-language state description
    2. If strategy is ReAct: use AgentExecutor with all tools bound
       Else: use strategy's prompt builder → direct LLM call → parse JSON
    3. Call MoveValidatorTool to verify chosen move
    4. If invalid: retry up to MAX_RETRIES (3) times with error feedback
    5. Return final move dict
    """
    def __init__(self, agent_id, model, strategy: str, tools: list, temperature=0.3):
        self.strategy = load_strategy(strategy)  # loads direct/cot/tot/react
        self.tools = tools
        self.invalid_move_count = 0
        self.move_history = []
```

### Step 4.3 — `agents/orchestrator_agent.py`
```python
class OrchestratorAgent(BaseAgent):
    """
    Manages game sessions. Does NOT play games.
    
    Responsibilities:
    - select_game(game_pool) → picks next game
    - assign_players(player_a, player_b, game) → initializes agents
    - manage_turn(game, current_player) → triggers player decision
    - handle_invalid_sequence(player, retries) → escalation logic
    - log_decision(turn, player, state, move, valid) → audit trail
    - broadcast_state(game_state) → sends state to both agents
    
    The orchestrator uses an LLM for post-game analysis:
    "Given this game log, what strategies did each player demonstrate?"
    """
```

### Step 4.4 — `agents/evaluator_agent.py`
```python
class EvaluatorAgent(BaseAgent):
    """
    Collects all game results and computes evaluation metrics.
    Does NOT play games — activated after each game completes.
    
    Methods:
    - record_game(session) → stores raw game log
    - compute_win_rates() → per agent/strategy breakdown
    - compute_invalid_rate() → invalid moves / total moves
    - compute_elo_ratings(results) → Elo after tournament
    - compute_pareto_efficiency(payoffs) → for Prisoner's Dilemma
    - generate_report(format="json"|"csv"|"html") → exports results
    
    Also uses LLM for qualitative analysis:
    "Summarize the strategic patterns observed across all games."
    """
```

---

## 📋 Phase 5: LangGraph Orchestration (Day 5)

> **Goal**: Wire everything into a LangGraph state machine.
> This is the core "Agentic" component satisfying the project requirements.

### Step 5.1 — Define State Schema
```python
from typing import TypedDict, Literal, Optional

class GameSessionState(TypedDict):
    session_id: str
    game_name: str
    game_instance: any           # BaseGame object
    player_a: PlayerAgent
    player_b: PlayerAgent
    current_player: Literal["player_a", "player_b"]
    turn_number: int
    move_history: list           # [{player, move, valid, timestamp}]
    retry_count: int
    terminal: bool
    winner: Optional[str]
    metrics: dict
    error_log: list
```

### Step 5.2 — Define Graph Nodes (functions)
Each node is a pure function: `state → state`.

```
node: init_session       → sets session_id, initializes game, assigns players
node: player_a_turn      → calls player_a.decide() → stores proposed_move
node: validate_move_a    → calls MoveValidatorTool → routes to valid/invalid
node: handle_invalid_a   → increments retry, adds error to state
node: apply_move_a       → calls game.make_move() → updates game state
node: player_b_turn      → same as player_a_turn
node: validate_move_b    → same as validate_move_a
node: handle_invalid_b   → same as handle_invalid_a
node: apply_move_b       → same as apply_move_a
node: check_terminal     → calls game.is_terminal() → routes game/done
node: record_result      → writes winner, turns to state
node: run_evaluator      → calls evaluator_agent.record_game()
node: generate_report    → calls evaluator_agent.generate_report()
```

### Step 5.3 — Define Graph Edges (routing)
```
init_session ──────────────────────────────────→ player_a_turn (always)
player_a_turn ─────────────────────────────────→ validate_move_a (always)
validate_move_a ──── "valid" ──────────────────→ apply_move_a
validate_move_a ──── "invalid" & retries < 3 ──→ handle_invalid_a → player_a_turn
validate_move_a ──── "invalid" & retries == 3 ─→ record_result (forfeit)
apply_move_a ──────────────────────────────────→ check_terminal
check_terminal ──── "terminal" ────────────────→ record_result
check_terminal ──── "ongoing" ─────────────────→ player_b_turn
[same pattern for player_b]
record_result ─────────────────────────────────→ run_evaluator
run_evaluator ─────────────────────────────────→ generate_report (if last game)
run_evaluator ─────────────────────────────────→ init_session (if more games)
```

### Step 5.4 — Build Graph
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(GameSessionState)
# Add all nodes
# Add all edges and conditional edges
# Compile
app = workflow.compile()

# Run
final_state = app.invoke(initial_state)
```

---

## 📋 Phase 6: Experiment Runner (Day 5–6)

### Step 6.1 — `experiments/configs/exp1_reasoning.yaml`
```yaml
experiment_name: "Strategy Comparison"
games: [tictactoe, connect4, nim, prisoners_dilemma, kuhn_poker]
rounds_per_matchup: 20
matchups:
  - player_a_strategy: direct
    player_b_strategy: cot
  - player_a_strategy: cot
    player_b_strategy: tot
  - player_a_strategy: tot
    player_b_strategy: react
  - player_a_strategy: direct
    player_b_strategy: react
model: "gpt-4o-mini"
temperature: 0.3
output_format: ["json", "csv", "html"]
```

### Step 6.2 — `experiments/configs/exp2_models.yaml`
```yaml
experiment_name: "Model Comparison"
games: [tictactoe, prisoners_dilemma, kuhn_poker]
rounds_per_matchup: 15
matchups:
  - player_a_model: gpt-4o
    player_b_model: gpt-3.5-turbo
  - player_a_model: gpt-4o
    player_b_model: llama-3-8b-8192   # via Groq
  - player_a_model: gpt-3.5-turbo
    player_b_model: llama-3-8b-8192
strategy: cot
temperature: 0.3
```

### Step 6.3 — `experiments/run_experiments.py`
```python
import argparse, yaml
from orchestration.game_graph import build_game_graph
from agents import PlayerAgent, OrchestratorAgent, EvaluatorAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    evaluator = EvaluatorAgent(...)
    
    for matchup in config["matchups"]:
        for game_name in config["games"]:
            for round_n in range(config["rounds_per_matchup"]):
                player_a = PlayerAgent(strategy=matchup["player_a_strategy"], ...)
                player_b = PlayerAgent(strategy=matchup["player_b_strategy"], ...)
                orchestrator = OrchestratorAgent(...)
                
                graph = build_game_graph(orchestrator, player_a, player_b, evaluator)
                initial_state = orchestrator.init_session(game_name, player_a, player_b)
                final_state = graph.invoke(initial_state)
                
                evaluator.record_game(final_state)
    
    evaluator.generate_report(format="html")
    print("✅ Experiment complete. Results in ./results/")

if __name__ == "__main__":
    main()
```

---

## 📋 Phase 7: Testing & Validation (Day 6)

### Step 7.1 — Unit Tests
```bash
# Test all games
python -m pytest tests/test_games.py -v

# Test agent tools
python -m pytest tests/test_agents.py -v

# Test evaluation metrics
python -m pytest tests/test_evaluation.py -v
```

### Step 7.2 — Integration Test (1 game, 2 rounds)
```bash
python experiments/run_experiments.py --config experiments/configs/exp1_reasoning.yaml
# Verify: results/Strategy_Comparison/ directory created
# Verify: results contain game logs, metrics, HTML report
```

### Step 7.3 — Smoke Test Checklist
- [ ] All 5 games run without error for 5 turns
- [ ] Invalid move triggers retry cycle (manually pass wrong move)
- [ ] Strategy Analyzer detects "Tit-for-Tat" in Prisoner's Dilemma
- [ ] Evaluator computes Elo after 10 games
- [ ] HTML report renders in browser

---

## 📋 Phase 8: Results & Metrics Collection (Day 7)

### Step 8.1 — Run Full Experiment 1 (Strategy Comparison)
```bash
python experiments/run_experiments.py --config experiments/configs/exp1_reasoning.yaml
# Expected runtime: ~2-3 hours with GPT-4o-mini (100 total games)
```

### Step 8.2 — Run Full Experiment 2 (Model Comparison)
```bash
python experiments/run_experiments.py --config experiments/configs/exp2_models.yaml
```

### Step 8.3 — Collect Key Results for Paper
From `results/` folder, extract:
- Win rate table per strategy per game
- Invalid move rate table
- Elo rating rankings
- Pareto efficiency scores (Prisoner's Dilemma)
- Sample game logs showing reasoning traces
- Statistical significance (t-test or Mann-Whitney U)

---

## 📋 Phase 9: GitHub Preparation (Day 7)

### Step 9.1 — Files to commit
```bash
git init
git add .
git commit -m "feat: initial agentic gtbench implementation"
git remote add origin https://github.com/YOUR_USERNAME/agentic-gtbench.git
git push -u origin main
```

### Step 9.2 — Files NOT to commit (`.gitignore`)
```
.env
venv/
__pycache__/
*.pyc
results/*.json
results/*.csv
results/*.html
.DS_Store
```

### Step 9.3 — GitHub Repository Settings
- Add README badges (Python version, License, Status)
- Create Issues for any known bugs
- Tag release `v1.0.0` after all experiments complete

---

## 🚨 Common Issues & Fixes

| Issue | Fix |
|---|---|
| `json.JSONDecodeError` when parsing LLM move | Add `try/except` with retry; include "respond ONLY with JSON" in prompt |
| Agent stuck in retry loop | Cap retries at 3; on 3rd failure, forfeit game and log |
| ToT too slow (3 LLM calls/turn) | Use `gpt-4o-mini` for ToT; use `gpt-4o` for Direct/CoT only |
| Kuhn Poker incomplete info leak | Never include opponent's card in state description |
| Rate limit errors | Add `time.sleep(1)` between games; use exponential backoff |
| LangGraph state mutation error | Always return new dict, never mutate `state` in-place |

---

## 📊 Expected Results (Hypotheses to Test)

1. **CoT > Direct** in probabilistic games (Kuhn Poker, Prisoner's Dilemma)
2. **ToT** has highest win rate but also highest cost (3x API calls)
3. **ReAct** has lowest invalid move rate (tool validation in loop)
4. **GPT-4o > GPT-3.5** in deterministic perfect-info games (Connect4, Nim)
5. **Llama-3** competitive with GPT-3.5 due to code pretraining

---

## 🗓️ Day-by-Day Schedule

| Day | Focus | Deliverable |
|---|---|---|
| 1 | Env setup + Games 1-3 | `tictactoe.py`, `connect4.py`, `nim.py` passing tests |
| 2 | Games 4-5 + Tools | `prisoners_dilemma.py`, `kuhn_poker.py`, all 4 tools |
| 3 | Strategies | All 4 strategies working in isolation |
| 4 | Agents | All 3 agents with unit tests |
| 5 | LangGraph graph | Full workflow running 1 game end-to-end |
| 6 | Experiment runner + full test | Both experiment configs running |
| 7 | Results collection + GitHub | Results exported, repo published |
