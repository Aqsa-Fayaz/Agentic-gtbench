"""
Post-hoc classifier: loads an existing experiment's report.json, runs the
ErrorProfileAgent over its sessions, and writes an enriched report back into
the SAME results directory.

Usage:
    python experiments/classify_existing.py results/Phase_2_Direct_vs_CoT_<ts>/

Why a separate driver:
    The classifier is a pure post-pass. Re-running run_experiments.py would
    re-spend the game-play API budget. This script only spends the classifier
    budget (~1 call per losing-side move).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.error_profile_agent import ErrorProfileAgent
from config.settings import settings
from evaluation.metrics import MetricsEngine
from evaluation.reporter import Reporter

logging.basicConfig(
    level=settings.log_level,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("classify_existing")


def main():
    parser = argparse.ArgumentParser(
        description="Run ErrorProfileAgent over an existing experiment's sessions."
    )
    parser.add_argument(
        "results_dir",
        help="Path to the experiment directory containing report.json",
    )
    parser.add_argument(
        "--all-moves",
        action="store_true",
        help="Classify every move (default: losing-side only).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    report_path = results_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"No report.json at {report_path}")

    logger.info(f"Loading sessions from {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    sessions = payload.get("sessions", [])
    logger.info(f"Loaded {len(sessions)} sessions; reconstructing metrics engine.")

    engine = MetricsEngine()
    for s in sessions:
        engine.record_session(s)

    judge = ErrorProfileAgent()
    only_losing = not args.all_moves
    logger.info(
        f"Classifying moves (only_losing_side={only_losing}). "
        "This will issue 1 LLM call per candidate move."
    )

    total_judged = 0
    for sess in sessions:
        findings = judge.classify_session(sess, only_losing_side=only_losing)
        total_judged += len(findings)
        logger.info(
            f"  session={sess.get('session_id')} game={sess.get('game_name')} "
            f"winner={sess.get('winner')} → {len(findings)} moves judged"
        )

    logger.info(f"Total moves judged: {total_judged}")
    aggregate = judge.aggregate_by_strategy()
    engine.attach_error_profile(aggregate)

    findings_path = results_dir / "error_findings.json"
    findings_path.write_text(
        json.dumps(judge.findings, indent=2, default=str), encoding="utf-8"
    )
    logger.info(f"Wrote per-move findings → {findings_path}")

    # Re-emit JSON + HTML with the new section embedded.
    report = engine.full_report()
    reporter = Reporter(output_dir=results_dir)
    json_path = reporter.write_json(report, sessions)
    html_path = reporter.write_html(report, sessions)
    logger.info(f"Updated report files: {json_path} and {html_path}")

    print()
    print("=== Error Profile Aggregate (per strategy) ===")
    for strat, counts in aggregate.items():
        total = counts.get("total", 0)
        cats = " | ".join(
            f"{c}={counts.get(c, 0)}" for c in ErrorProfileAgent.CATEGORIES
        )
        print(f"  {strat:8s} (total {total}): {cats}")


if __name__ == "__main__":
    main()
