"""
EvaluatorAgent — metric collector & result analyzer.

Wraps `evaluation.metrics.MetricsEngine` and optionally consults an LLM to
produce a qualitative summary of tournament-level strategic patterns.

Does NOT play games. The LangGraph workflow calls `record_session()` once per
completed game; at experiment end, `generate_report()` exports JSON/CSV/HTML.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from agents.base_agent import BaseAgent
from config.settings import settings
from evaluation.metrics import MetricsEngine
from evaluation.reporter import Reporter

logger = logging.getLogger(__name__)


class EvaluatorAgent(BaseAgent):
    """
    Accumulates game results and produces evaluation artefacts.
    """

    SYSTEM_PROMPT = (
        "You are an evaluator for LLM game-playing agents. "
        "Given aggregate metrics from a tournament (win rates, invalid-move "
        "rates, Elo, cooperation rates, Pareto efficiency), produce a short "
        "(3-6 sentence) analytical summary. Highlight which reasoning "
        "strategies outperformed others and in which game categories. "
        "Do NOT invent numbers — only reference what is in the provided JSON."
    )

    def __init__(
        self,
        agent_id: str = "evaluator",
        model: str = None,
        temperature: float = 0.2,
    ):
        super().__init__(agent_id=agent_id, model=model, temperature=temperature)
        self.engine = MetricsEngine()
        self._llm: Optional[ChatOpenAI] = None

    @property
    def llm(self) -> ChatOpenAI:
        if self._llm is None:
            self._llm = settings.build_chat_openai_client(
                model=self.model,
                temperature=self.temperature,
            )
        return self._llm

    def decide(self, game_state: dict, game_name: str, legal_moves: list) -> dict:
        raise NotImplementedError("EvaluatorAgent does not play games.")

    def record_session(self, session: dict) -> None:
        """Forward a completed game session into the metrics engine."""
        self.engine.record_session(session)
        logger.debug(f"Recorded session: {session.get('session_id')}")

    @property
    def sessions(self) -> list:
        return self.engine.sessions

    def full_report(self) -> dict:
        """Return the aggregated metrics dict."""
        return self.engine.full_report()

    def qualitative_summary(self, report: Optional[dict] = None) -> str:
        """Ask the LLM for a brief analytical summary of the current metrics."""
        if report is None:
            report = self.full_report()
        if not settings.has_llm_credentials():
            return "[qualitative summary unavailable — set OPENAI_API_KEY or OPENROUTER_API_KEY]"
        try:
            import json

            messages = [
                SystemMessage(content=self.SYSTEM_PROMPT),
                HumanMessage(content=json.dumps(report, indent=2, default=str)),
            ]
            response = self.llm.invoke(messages)
            return (response.content or "").strip()
        except Exception as exc:  # pragma: no cover
            logger.warning(f"qualitative_summary failed: {exc}")
            return f"[summary error: {exc}]"

    def generate_report(
        self,
        output_dir: Path,
        formats: Optional[list[str]] = None,
        include_llm_summary: bool = False,
    ) -> dict:
        """
        Export all metrics + raw sessions to disk in the requested formats.
        Returns a manifest of written files.
        """
        formats = formats or ["json"]
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        report = self.full_report()
        sessions = list(self.engine.sessions)
        reporter = Reporter(output_dir=output_dir)

        written: dict[str, str] = {}
        if "json" in formats:
            written["json"] = str(reporter.write_json(report, sessions))
        if "csv" in formats:
            written["csv"] = str(reporter.write_csv(sessions))
        if "html" in formats:
            written["html"] = str(reporter.write_html(report, sessions))

        if include_llm_summary:
            summary = self.qualitative_summary(report)
            summary_path = output_dir / "qualitative_summary.txt"
            summary_path.write_text(summary, encoding="utf-8")
            written["summary"] = str(summary_path)

        logger.info(f"Report written: {written}")
        return written
