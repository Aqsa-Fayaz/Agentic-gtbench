# System Architecture Diagrams

## 1. Full System Architecture

```mermaid
graph TB
    subgraph ENTRY["Entry Point"]
        EXP[run_experiments.py\nArgparse + YAML config]
    end

    subgraph LANGGRAPH["⚙️ LangGraph State Machine\norchestration/game_graph.py"]
        N1[init_session]
        N2[player_a_turn]
        N3[validate_move]
        N4[apply_move]
        N5[check_terminal]
        N6[player_b_turn]
        N7[handle_invalid\nForfeit]
        N8[record_result]
        N9[run_evaluator]
        N1-->N2-->N3
        N3--valid-->N4
        N3--invalid & retries<3-->N2
        N3--invalid & retries=3-->N7
        N4-->N5
        N5--ongoing-->N6-->N3
        N5--terminal-->N8
        N7-->N8-->N9
    end

    subgraph AGENTS["🤖 Agent Layer"]
        PA["PlayerAgent A\nstrategy: CoT/ToT/ReAct/Direct"]
        PB["PlayerAgent B\nstrategy: CoT/ToT/ReAct/Direct"]
        EV["EvaluatorAgent\nMetricsEngine wrapper"]
    end

    subgraph STRATEGIES["🧠 Reasoning Strategies"]
        S1["direct.py\n1 LLM call, zero-shot"]
        S2["cot.py\n1 LLM call + step-by-step prompt"]
        S3["tot.py\n3 LLM calls: Generate→Eval→Select"]
        S4["react.py\nThought→Action→Observation loop"]
    end

    subgraph GAMES["🎲 Game Environments"]
        G1["TicTacToe\n✓ Implemented"]
        G2["Connect4\n⬜ TODO"]
        G3["Nim\n⬜ TODO"]
        G4["PrisonersDilemma\n✓ Implemented"]
        G5["KuhnPoker\n⬜ TODO"]
    end

    subgraph TOOLS["🔧 LangChain Tools"]
        T1["MoveValidatorTool\nvalidate_move(game, state, move)"]
        T2["StateTrackerTool\ndescribe_state(game, state, player)"]
        T3["StrategyAnalyzerTool\nanalyze_opponent(history)"]
        T4["HistoryManagerTool\nsave/load/stats sessions"]
    end

    subgraph EVAL["📊 Evaluation Layer"]
        M1["MetricsEngine\nWin rate, Invalid rate, Elo"]
        M2["Pareto Efficiency\nCooperation Rate"]
        M3["Reporter\nJSON / CSV / HTML export"]
    end

    EXP --> LANGGRAPH
    LANGGRAPH --> PA
    LANGGRAPH --> PB
    LANGGRAPH --> EV
    PA --> STRATEGIES
    PB --> STRATEGIES
    PA --> TOOLS
    PB --> TOOLS
    LANGGRAPH --> GAMES
    EV --> EVAL
    EVAL --> M3
```

---

## 2. LangGraph State Machine (Detailed)

```mermaid
stateDiagram-v2
    [*] --> init_session : Graph invoked
    
    init_session --> player_a_turn : Always

    player_a_turn --> validate_move : Proposed move ready
    
    validate_move --> apply_move : Move valid
    validate_move --> player_a_turn : Invalid, retries < 3
    validate_move --> handle_invalid : Invalid, retries = 3
    
    apply_move --> check_terminal : State updated

    check_terminal --> player_b_turn : Game ongoing\ncurrent=player_a
    check_terminal --> player_a_turn : Game ongoing\ncurrent=player_b  
    check_terminal --> record_result : Game terminal

    player_b_turn --> validate_move : Proposed move ready
    
    handle_invalid --> record_result : Forfeit (loser = current player)
    
    record_result --> run_evaluator : Session packaged
    run_evaluator --> [*] : Session done
```

---

## 3. Agent Decision Flow (PlayerAgent)

```mermaid
flowchart TD
    A[Turn Start] --> B[Get game state\nfor player perspective]
    B --> C[Get legal moves list]
    C --> D{Strategy?}
    
    D -->|direct| E1[Build state prompt\nOne-shot LLM call]
    D -->|cot| E2[Build CoT prompt\nStep-by-step LLM call]
    D -->|tot| E3[Call 1: Generate 3 candidates\nCall 2: Score each\nCall 3: Select best]
    D -->|react| E4[Thought→Action loop\nUses tools inline]
    
    E1 --> F[Parse JSON move from response]
    E2 --> F
    E3 --> F
    E4 --> F
    
    F --> G{JSON valid?}
    G -->|No| H[Retry counter +1]
    H --> I{Retries < 3?}
    I -->|Yes| D
    I -->|No| J[Raise RuntimeError\n→ Forfeit in graph]
    
    G -->|Yes| K[Return move dict\nto LangGraph]
```

---

## 4. Evaluation Pipeline

```mermaid
flowchart LR
    S1[Game Session 1] --> E[EvaluatorWrapper\nrecord_session]
    S2[Game Session 2] --> E
    SN[Game Session N] --> E
    
    E --> M[MetricsEngine]
    
    M --> W[Win Rates\nper strategy × game]
    M --> I[Invalid Move Rates\nper agent]
    M --> EL[Elo Ratings\ntournament-style]
    M --> T[Avg Turns to Win]
    M --> P[Pareto Efficiency\nPrisoner's Dilemma only]
    M --> C[Cooperation Rates\nPrisoner's Dilemma only]
    
    W --> R[Reporter\nJSON + CSV + HTML]
    I --> R
    EL --> R
    T --> R
    P --> R
    C --> R
```

---

## 5. Game Taxonomy (from GTBench)

```mermaid
mindmap
  root((Games in\nThis System))
    Perfect Information
      Deterministic
        TicTacToe
          3×3 grid
          Zero-sum
        Connect4
          6×7 grid
          Gravity mechanic
        Nim
          Pile removal
          Misère variant
    Imperfect Information
      Probabilistic
        KuhnPoker
          3-card deck
          Belief modeling required
    Repeated Interaction
      Sequential
        PrisonersDilemma
          Iterated 10 rounds
          Payoff matrix
          Pareto efficiency measurable
```
