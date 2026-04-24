# 🎮 Agentic GTBench: LLM Strategic Reasoning via Multi-Agent Game-Theoretic Evaluation

> **FAST NUCES — Agentic AI Final Project**  
> Manal Aamir (22I-1940) · Aqsa Fayaz (22i-1865) · Arhum Khan (22i-1967)  
> Department of Data Science

---

## 📌 Project Overview

This project implements a **fully autonomous multi-agent system** that evaluates and compares the strategic reasoning capabilities of Large Language Models (LLMs) through game-theoretic environments. Extending the GTBench framework, we build a real **Agentic AI** where:

- **Autonomous LLM agents** compete across 5 classical games
- **Multiple reasoning strategies** (Direct, CoT, ToT, ReAct) are tested head-to-head
- A **dedicated Orchestrator Agent** manages sessions, coordinates players, and logs decisions
- An **Evaluator Agent** computes metrics, detects strategy patterns, and generates reports
- All communication flows through a **LangGraph state machine**

---

## 🗂️ Repository Structure

```
agentic-gtbench/
├── agents/
│   ├── base_agent.py           # Abstract LLM agent interface
│   ├── player_agent.py         # Game-playing agent with strategy injection
│   ├── orchestrator_agent.py   # Session manager & coordinator
│   ├── evaluator_agent.py      # Metric collector & result analyzer
│   └── strategies/
│       ├── direct.py           # Zero-shot prompting
│       ├── cot.py              # Chain-of-Thought
│       ├── tot.py              # Tree-of-Thoughts
│       └── react.py            # ReAct (Reason + Act)
├── games/
│   ├── base_game.py            # Abstract game interface
│   ├── tictactoe.py            # Complete-info, deterministic
│   ├── connect4.py             # Complete-info, deterministic
│   ├── nim.py                  # Perfect-info combinatorial
│   ├── prisoners_dilemma.py    # Repeated / iterated game
│   └── kuhn_poker.py           # Incomplete-info, probabilistic
├── tools/
│   ├── move_validator.py       # Legal move enforcement tool
│   ├── state_tracker.py        # Game state serialization tool
│   ├── strategy_analyzer.py    # Opponent modeling tool
│   └── history_manager.py      # Persistent game history tool
├── orchestration/
│   └── game_graph.py           # LangGraph workflow definition
├── evaluation/
│   ├── metrics.py              # Win rate, invalid rate, Elo, Pareto
│   └── reporter.py             # CSV/JSON/HTML result export
├── experiments/
│   ├── run_experiments.py      # Main experiment runner
│   └── configs/
│       ├── exp1_reasoning.yaml # Strategy comparison experiment
│       └── exp2_models.yaml    # Model family comparison
├── config/
│   └── settings.py             # Central config & env loading
├── tests/
│   ├── test_games.py
│   ├── test_agents.py
│   └── test_evaluation.py
├── diagrams/
│   ├── system_architecture.md  # Mermaid: full system diagram
│   └── agent_workflow.md       # Mermaid: LangGraph state flow
├── results/                    # Auto-generated experiment outputs
├── requirements.txt
├── .env.example
└── IMPLEMENTATION_PLAN.md      # Step-by-step dev guide for Cursor
```

---

## 🏗️ System Architecture

```mermaid
graph TB
    subgraph USER["👤 User / Experiment Runner"]
        EXP[run_experiments.py]
    end

    subgraph ORCH["🎯 Orchestrator Agent"]
        OM[Session Manager]
        OC[Agent Coordinator]
        OL[Decision Logger]
    end

    subgraph PLAYERS["⚔️ Player Agents (LLM-Powered)"]
        PA1["Player A\n(Strategy: CoT)"]
        PA2["Player B\n(Strategy: ToT)"]
    end

    subgraph STRATEGIES["🧠 Reasoning Strategies"]
        S1[Direct / Zero-Shot]
        S2[Chain-of-Thought]
        S3[Tree-of-Thoughts]
        S4[ReAct]
    end

    subgraph GAMES["🎲 Game Environments"]
        G1[Tic-Tac-Toe]
        G2[Connect-4]
        G3[Nim]
        G4[Prisoner's Dilemma]
        G5[Kuhn Poker]
    end

    subgraph TOOLS["🔧 Agent Tools"]
        T1[Move Validator]
        T2[State Tracker]
        T3[Strategy Analyzer]
        T4[History Manager]
    end

    subgraph EVAL["📊 Evaluator Agent"]
        E1[Metrics Engine]
        E2[Report Generator]
        E3[Elo Calculator]
    end

    subgraph LANGGRAPH["⚙️ LangGraph Orchestration"]
        LG[State Machine\ngame_graph.py]
    end

    EXP --> ORCH
    ORCH --> LANGGRAPH
    LANGGRAPH --> PLAYERS
    PLAYERS --> STRATEGIES
    PLAYERS --> GAMES
    PLAYERS --> TOOLS
    GAMES --> EVAL
    TOOLS --> EVAL
    EVAL --> EXP
```

---

## 🔄 LangGraph State Flow

```mermaid
stateDiagram-v2
    [*] --> INIT_SESSION
    INIT_SESSION --> SELECT_GAME
    SELECT_GAME --> ASSIGN_STRATEGIES
    ASSIGN_STRATEGIES --> PLAYER_A_TURN
    PLAYER_A_TURN --> VALIDATE_MOVE_A
    VALIDATE_MOVE_A --> INVALID_A: Invalid
    INVALID_A --> PLAYER_A_TURN: Retry (max 3)
    VALIDATE_MOVE_A --> UPDATE_STATE: Valid
    UPDATE_STATE --> CHECK_TERMINAL
    CHECK_TERMINAL --> PLAYER_B_TURN: Game ongoing
    PLAYER_B_TURN --> VALIDATE_MOVE_B
    VALIDATE_MOVE_B --> INVALID_B: Invalid
    INVALID_B --> PLAYER_B_TURN: Retry (max 3)
    VALIDATE_MOVE_B --> UPDATE_STATE: Valid
    CHECK_TERMINAL --> RECORD_RESULT: Game over
    RECORD_RESULT --> EVALUATE_GAME
    EVALUATE_GAME --> MORE_GAMES: Next game/round
    MORE_GAMES --> SELECT_GAME: Yes
    MORE_GAMES --> AGGREGATE_METRICS: No
    AGGREGATE_METRICS --> GENERATE_REPORT
    GENERATE_REPORT --> [*]
```

---

## 🎯 Gaps Addressed Beyond GTBench Paper

| Gap in Review Paper | Our Implementation Fix |
|---|---|
| No actual agent implementation | Full LangGraph multi-agent system |
| Only reviews prompting — no testing | Live head-to-head strategy tournaments |
| No tool use / API integration | 4 dedicated agent tools built |
| No multi-agent coordination | Orchestrator + Player + Evaluator agents |
| No evaluation framework | Elo ratings, win rates, invalid move %, Pareto efficiency |
| No dynamic/adaptive behavior | Opponent modeling via Strategy Analyzer tool |
| No persistence or memory | History Manager with JSON logs |
| Single reasoning path | All 4 strategies selectable per agent |

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.10+
- OpenAI API key (or Anthropic / Groq for open models)
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/agentic-gtbench.git
cd agentic-gtbench
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment
```bash
cp .env.example .env
# Edit .env and add your API keys
```

### 5. Run a Quick Test
```bash
python -m pytest tests/ -v
```

### 6. Run Experiments
```bash
# Strategy comparison experiment
python experiments/run_experiments.py --config experiments/configs/exp1_reasoning.yaml

# Model comparison experiment  
python experiments/run_experiments.py --config experiments/configs/exp2_models.yaml
```

---

## 📊 Evaluation Metrics

| Metric | Description | Games |
|---|---|---|
| **Win Rate** | % of games won per agent/strategy | All |
| **Invalid Move Rate** | % of illegal moves generated | All |
| **Elo Rating** | Relative strength across tournaments | All |
| **Avg Turns to Win** | Efficiency of reasoning | TicTacToe, Connect4, Nim |
| **Cooperation Rate** | % of cooperative choices | Prisoner's Dilemma |
| **Pareto Efficiency** | Closeness to optimal joint payoff | Prisoner's Dilemma |
| **Bluff Detection Rate** | Accuracy of opponent modeling | Kuhn Poker |

---

## 🔬 Experiments

### Experiment 1: Reasoning Strategy Comparison
- **Agents**: GPT-4o-mini vs GPT-4o-mini (different strategies)
- **Strategies**: Direct vs CoT vs ToT vs ReAct
- **Games**: All 5 games, 20 rounds each
- **Hypothesis**: CoT/ToT improve in probabilistic games; may hurt in deterministic

### Experiment 2: Model Family Comparison
- **Agents**: GPT-4o vs GPT-3.5-turbo vs Llama-3 (via Groq)
- **Strategy**: CoT for all
- **Games**: Tic-Tac-Toe, Connect-4, Prisoner's Dilemma
- **Hypothesis**: Code-pretrained models outperform in deterministic games

---

## 📝 NCEAC Complex Computing Requirements Checklist

- [x] **Autonomous Agents** — Player, Orchestrator, Evaluator agents
- [x] **Decision-making under uncertainty** — Kuhn Poker, Prisoner's Dilemma
- [x] **Tool usage** — 4 custom tools (validator, tracker, analyzer, history)
- [x] **Multi-agent coordination** — LangGraph orchestration with message passing
- [x] **Dynamic/unpredictable operation** — Opponent adapts based on history
- [x] **Research & Experimentation** — 2 controlled experiments with metrics
- [x] **Ethics discussion** — See paper section IX

---

## 📄 License

MIT License — For academic use only.
