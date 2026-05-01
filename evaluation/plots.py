"""
Paper-style figures for an experiment's report.

Reproduces the four key figures from the GTBench paper:
  * Fig 2/3 — NRA bar chart per strategy
  * Fig 4   — Win-rate heatmap per (strategy, game)
  * Fig 5a  — Ex-post regret per strategy (lower is better)
  * Table 5 — Error-profile stacked bar per strategy

Reads `report.json` from a results directory and writes PNGs alongside.

Usage:
    python evaluation/plots.py results/Phase_2_Direct_vs_CoT_<ts>/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")   # Don't try to open a window on Windows headless.
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger("plots")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


# ─────────────────────────────────────────────
# Individual chart helpers
# ─────────────────────────────────────────────

def plot_nra_by_strategy(report: dict, out_path: Path) -> bool:
    nra = (report.get("nra") or {}).get("by_strategy") or {}
    if not nra:
        return False
    strats = list(nra.keys())
    values = [nra[s].get("nra", 0) for s in strats]
    n_matches = [nra[s].get("n", 0) for s in strats]

    colors = ["#2ca02c" if v > 0 else "#d62728" for v in values]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(strats, values, color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylim(-1.05, 1.05)
    ax.set_ylabel("NRA  (range [-1, 1])")
    ax.set_title("Normalized Relative Advantage by Strategy")
    for bar, v, n in zip(bars, values, n_matches):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v + (0.04 if v >= 0 else -0.06),
            f"{v:+.2f}\n(n={n})",
            ha="center",
            va="bottom" if v >= 0 else "top",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_win_rate_heatmap(report: dict, out_path: Path) -> bool:
    wr = report.get("win_rates") or {}
    if not wr:
        return False
    # Each key is "('strategy', 'game')" — parse it.
    cells = []
    for key, v in wr.items():
        cells.append((v.get("strategy"), v.get("game"), v.get("win_rate", 0)))
    strategies = sorted({c[0] for c in cells})
    games = sorted({c[1] for c in cells})
    if not strategies or not games:
        return False
    matrix = np.full((len(strategies), len(games)), np.nan)
    for s, g, wr_val in cells:
        i = strategies.index(s)
        j = games.index(g)
        matrix[i, j] = wr_val

    fig, ax = plt.subplots(figsize=(1.4 * len(games) + 2, 0.8 * len(strategies) + 2))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(len(games)))
    ax.set_xticklabels(games, rotation=30, ha="right")
    ax.set_yticks(range(len(strategies)))
    ax.set_yticklabels(strategies)
    for i in range(len(strategies)):
        for j in range(len(games)):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="black" if 0.3 < v < 0.7 else "white", fontsize=9)
    ax.set_title("Win Rate by Strategy × Game")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04, label="Win rate")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_regret_by_strategy(report: dict, out_path: Path) -> bool:
    regret = (report.get("regret") or {}).get("by_strategy") or {}
    if not regret:
        return False
    strats = list(regret.keys())
    avg = [regret[s].get("avg_regret", 0) for s in strats]
    n = [regret[s].get("n", 0) for s in strats]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(strats, avg, color="#1f77b4")
    ax.set_ylabel("Average ex-post regret  (lower = closer to optimal)")
    ax.set_title("Ex-Post Regret by Strategy")
    for bar, v, samples in zip(bars, avg, n):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            v,
            f"{v:.2f}\n(n={samples})",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


def plot_error_profile(report: dict, out_path: Path) -> bool:
    profile = report.get("error_profile") or {}
    if not profile:
        return False
    categories = (
        "misinterpretation",
        "factual_error",
        "overconfidence",
        "calculation_mistake",
        "endgame_misdetection",
        "ok",
    )
    strats = list(profile.keys())
    data = np.array(
        [[profile[s].get(c, 0) for c in categories] for s in strats], dtype=float
    )

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bottoms = np.zeros(len(strats))
    palette = plt.colormaps.get_cmap("tab10")
    for i, cat in enumerate(categories):
        ax.bar(strats, data[:, i], bottom=bottoms, label=cat.replace("_", " "),
               color=palette(i / max(1, len(categories) - 1)))
        bottoms += data[:, i]

    ax.set_ylabel("Move count (losing-side moves judged)")
    ax.set_title("Error Profile by Strategy (LLM-as-judge)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return True


# ─────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────

PLOTS = [
    ("nra_by_strategy.png",   plot_nra_by_strategy,   "NRA bar chart"),
    ("win_rate_heatmap.png",  plot_win_rate_heatmap,  "Win-rate heatmap"),
    ("regret_by_strategy.png", plot_regret_by_strategy, "Regret bar chart"),
    ("error_profile.png",     plot_error_profile,     "Error-profile stacked bar"),
]


def render_all(results_dir: Path) -> dict:
    report_path = results_dir / "report.json"
    if not report_path.exists():
        raise SystemExit(f"No report.json at {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    report = payload.get("report", {})

    written: dict[str, str] = {}
    for filename, fn, label in PLOTS:
        out = results_dir / filename
        try:
            ok = fn(report, out)
        except Exception as exc:
            logger.warning(f"{label} failed: {exc}")
            ok = False
        if ok:
            logger.info(f"  wrote {label} -> {out.name}")
            written[label] = str(out)
        else:
            logger.info(f"  skipped {label} (no data in report)")
    return written


def main():
    parser = argparse.ArgumentParser(description="Generate paper-style figures.")
    parser.add_argument("results_dir")
    args = parser.parse_args()
    written = render_all(Path(args.results_dir))
    print(json.dumps({"written": written}, indent=2))


if __name__ == "__main__":
    main()
