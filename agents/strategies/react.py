"""
ReAct (Reason + Act) reasoning strategy.

Prompt-based ReAct loop. The model produces Thought/Action/Observation cycles,
invoking bound tools by name, and finally emits an `Action: MOVE {json}` line.

If bound tools are provided we execute them inline and feed their output back to
the model as an Observation. The loop is capped at MAX_CYCLES to bound latency.
"""

from __future__ import annotations

import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agents.strategies._common import build_state_prompt, parse_move

NAME = "react"
MAX_CYCLES = 4

SYSTEM_PROMPT = (
    "You are a strategic game-playing AI that reasons and acts in cycles.\n"
    "On each cycle emit exactly two lines:\n"
    "  Thought: <your reasoning>\n"
    "  Action:  <tool_name>(<json_args>)   OR   MOVE <json_move>\n"
    "When you are ready to commit, emit `Action: MOVE {...}` and stop."
)

_TOOL_CALL_RE = re.compile(r"Action:\s*([A-Za-z_][A-Za-z0-9_]*)\s*\((.*)\)\s*$", re.MULTILINE)
_MOVE_RE = re.compile(r"Action:\s*MOVE\s*(\{.*\})", re.DOTALL)


def _call_tool(tools: list, tool_name: str, args_raw: str) -> str:
    """Look up a LangChain tool by name and invoke it with JSON args."""
    if not tools:
        return f"[error] no tools bound; cannot call {tool_name}."
    tool = next((t for t in tools if getattr(t, "name", None) == tool_name), None)
    if tool is None:
        return f"[error] unknown tool '{tool_name}'."
    try:
        args = json.loads(args_raw) if args_raw.strip() else {}
    except json.JSONDecodeError:
        args = {}
    try:
        return str(tool.invoke(args)) if hasattr(tool, "invoke") else str(tool.run(args))
    except Exception as exc:  # pragma: no cover - tools should be defensive
        return f"[error] tool '{tool_name}' raised: {exc}"


def _extract_move(text: str) -> dict:
    """Find an `Action: MOVE {...}` line and parse the JSON move."""
    match = _MOVE_RE.search(text)
    if match:
        return parse_move(match.group(1))
    return parse_move(text)


def run(
    llm,
    game_state: dict,
    game_name: str,
    legal_moves: list,
    error: Optional[str] = None,
    tools: Optional[list] = None,
) -> dict:
    tools = tools or []
    system = SystemMessage(content=SYSTEM_PROMPT)

    user_prompt = (
        build_state_prompt(game_state, game_name, legal_moves, error)
        + "\n\n"
        + "Available tools: "
        + (", ".join(t.name for t in tools) if tools else "(none)")
        + "\n\nBegin your Thought/Action loop."
    )

    transcript: list = [system, HumanMessage(content=user_prompt)]

    for _ in range(MAX_CYCLES):
        response = llm.invoke(transcript)
        content = response.content or ""
        transcript.append(response)

        if "MOVE" in content and _MOVE_RE.search(content):
            return _extract_move(content)

        tool_match = _TOOL_CALL_RE.search(content)
        if tool_match and tools:
            name, args_raw = tool_match.group(1), tool_match.group(2)
            observation = _call_tool(tools, name, args_raw)
            transcript.append(HumanMessage(content=f"Observation: {observation}"))
            continue

        # No tool call, no MOVE line -> nudge the model to commit.
        transcript.append(
            HumanMessage(
                content=(
                    "You did not emit `Action: MOVE {...}`. "
                    "Pick a legal move now and respond with `Action: MOVE {json}` only."
                )
            )
        )

    final = llm.invoke(
        transcript
        + [HumanMessage(content='Respond ONLY with JSON: {"move": {...}}')]
    )
    return parse_move(final.content)
