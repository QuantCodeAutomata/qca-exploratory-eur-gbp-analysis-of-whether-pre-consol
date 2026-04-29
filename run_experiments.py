"""
Main experiment runner for:
  - Experiment 1: EUR/GBP exploratory heatmap analysis
  - Experiment 2: GBP/USD three-strategy out-of-sample comparison

Usage:
    python run_experiments.py

Results are saved to the 'results/' directory.
Metrics are saved to 'results/RESULTS.md'.
"""

import logging
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("results/experiment_run.log", mode="w"),
    ],
)
logger = logging.getLogger(__name__)

RESULTS_DIR = "results"
DATA_DIR = "data"


def run_all_experiments() -> None:
    """Run both experiments and save all results."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    results_summary = {}

    # =========================================================================
    # Experiment 1: EUR/GBP Exploratory Heatmap Analysis
    # =========================================================================
    logger.info("=" * 70)
    logger.info("EXPERIMENT 1: EUR/GBP Exploratory Heatmap Analysis")
    logger.info("=" * 70)

    try:
        from src.exp1_heatmap import run_exp1

        breakout_df, metrics_df, ambiguity_log = run_exp1(
            results_dir=RESULTS_DIR,
            cache_dir=DATA_DIR,
        )

        if not breakout_df.empty:
            results_summary["exp1"] = {
                "status": "success",
                "n_events": len(breakout_df),
                "metrics": metrics_df,
                "ambiguity_log": ambiguity_log,
            }
            logger.info("Experiment 1 completed: %d breakout events detected.", len(breakout_df))
        else:
            results_summary["exp1"] = {"status": "no_data", "n_events": 0}
            logger.warning("Experiment 1: No breakout events detected.")

    except Exception as e:
        logger.error("Experiment 1 failed: %s", str(e), exc_info=True)
        results_summary["exp1"] = {"status": "failed", "error": str(e)}

    # =========================================================================
    # Experiment 2: GBP/USD Three-Strategy Comparison
    # =========================================================================
    logger.info("=" * 70)
    logger.info("EXPERIMENT 2: GBP/USD Three-Strategy Out-of-Sample Comparison")
    logger.info("=" * 70)

    try:
        from src.exp2_strategies import PAPER_BENCHMARKS, run_exp2

        trades_df, metrics_df, ambiguity_log = run_exp2(
            results_dir=RESULTS_DIR,
            cache_dir=DATA_DIR,
        )

        if not trades_df.empty:
            results_summary["exp2"] = {
                "status": "success",
                "n_trades": len(trades_df),
                "metrics": metrics_df,
                "ambiguity_log": ambiguity_log,
                "paper_benchmarks": PAPER_BENCHMARKS,
            }
            logger.info("Experiment 2 completed: %d trade records.", len(trades_df))
        else:
            results_summary["exp2"] = {"status": "no_data", "n_trades": 0}
            logger.warning("Experiment 2: No trades simulated.")

    except Exception as e:
        logger.error("Experiment 2 failed: %s", str(e), exc_info=True)
        results_summary["exp2"] = {"status": "failed", "error": str(e)}

    # =========================================================================
    # Save RESULTS.md
    # =========================================================================
    _save_results_md(results_summary)
    logger.info("All experiments complete. Results saved to %s/", RESULTS_DIR)


def _save_results_md(results_summary: dict) -> None:
    """
    Save a comprehensive RESULTS.md file with all metrics and findings.

    Parameters
    ----------
    results_summary : dict
        Dictionary containing results from both experiments.
    """
    md_path = os.path.join(RESULTS_DIR, "RESULTS.md")
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# Experiment Results",
        "",
        f"**Generated:** {timestamp}",
        "",
        "---",
        "",
        "## Overview",
        "",
        "This repository replicates two experiments from a quantitative finance paper",
        "studying whether pre-consolidation ADX predicts breakout success on FX 1-minute data.",
        "",
        "### Ambiguity Resolutions (Common to Both Experiments)",
        "",
        "| # | Ambiguity | Resolution |",
        "|---|-----------|------------|",
        "| 1 | Paper says 'one year' but Oct 2017–Dec 2018 is 15 months | Use full 15-month span |",
        "| 2 | Width rule: flowchart says ≤15 pips, method says 8-15 pips | Use 8-15 pips (method section) |",
        "| 3 | ADX formulas in paper are nonstandard | Use Wilder-style 14-period ADX via `ta` library |",
        "| 4 | No rounding rule for 10% of consolidation length | Use floor() rounding |",
        "| 5 | Entry price not defined | Use close of breakout bar |",
        "| 6 | Intrabar tie-break not defined | Stop wins (conservative) |",
        "| 7 | GBP/USD dates not specified | Mirror EUR/GBP span (Oct 2017–Dec 2018) |",
        "| 8 | ADX >= 25 vs > 25 for Discrete strategy | Use ADX >= 25 as strong trend |",
        "",
        "---",
        "",
    ]

    # Experiment 1 results
    lines += [
        "## Experiment 1: EUR/GBP Exploratory Heatmap Analysis",
        "",
        "**Hypothesis:** Pre-consolidation ADX predicts breakout success.",
        "Higher ADX → higher win rate, especially for larger profit targets.",
        "",
    ]

    exp1 = results_summary.get("exp1", {})
    if exp1.get("status") == "success":
        lines += [
            f"**Status:** ✅ Success",
            f"**Breakout Events Detected:** {exp1['n_events']}",
            "",
            "### Win Rate Summary by Target Multiple",
            "",
        ]
        metrics_df = exp1.get("metrics", pd.DataFrame())
        if not metrics_df.empty:
            lines.append("| Target Multiple | ADX Bin | Total Trades | Wins | Win Rate |")
            lines.append("|----------------|---------|-------------|------|----------|")
            for _, row in metrics_df.iterrows():
                lines.append(
                    f"| {row.get('target_multiple', 'N/A')} | "
                    f"{row.get('adx_bin', 'N/A')} | "
                    f"{row.get('total_trades', 0)} | "
                    f"{row.get('wins', 0)} | "
                    f"{row.get('win_rate', float('nan')):.3f} |"
                )
            lines.append("")

        lines += [
            "### Qualitative Assessment",
            "",
            "Expected: Higher ADX bins should show higher win rates, especially for 2x target.",
            "Low-ADX groups should show more pronounced deterioration from 1x to 2x target.",
            "",
            "### Output Files",
            "",
            "- `results/exp1_breakout_dataset.csv` — Per-breakout dataset",
            "- `results/exp1_heatmap_target1x.png` — Win rate heatmap (target = 1x width)",
            "- `results/exp1_heatmap_target2x.png` — Win rate heatmap (target = 2x width)",
            "- `results/exp1_metrics.csv` — Summary metrics",
            "- `results/exp1_ambiguity_log.txt` — Ambiguity resolutions",
            "",
        ]
    elif exp1.get("status") == "no_data":
        lines += [
            "**Status:** ⚠️ No data / No events detected",
            "",
            "> Note: This may occur if the Massive API returns no data for the specified",
            "> date range and ticker. Check MASSIVE_TOKEN environment variable and API access.",
            "",
        ]
    else:
        lines += [
            f"**Status:** ❌ Failed",
            f"**Error:** {exp1.get('error', 'Unknown error')}",
            "",
        ]

    lines += ["---", ""]

    # Experiment 2 results
    lines += [
        "## Experiment 2: GBP/USD Three-Strategy Out-of-Sample Comparison",
        "",
        "**Hypothesis:** ADX-conditioned target strategies outperform standard breakout strategy.",
        "Expected ranking: Linear (highest profit) > Discrete (highest wins) > Standard (weakest).",
        "",
    ]

    exp2 = results_summary.get("exp2", {})
    if exp2.get("status") == "success":
        lines += [
            f"**Status:** ✅ Success",
            f"**Total Trade Records:** {exp2['n_trades']}",
            "",
            "### Strategy Performance Metrics",
            "",
            "| Strategy | Wins | Losses | Total | Avg Gain | Avg Loss | APPT | Total Profit |",
            "|----------|------|--------|-------|----------|----------|------|-------------|",
        ]
        metrics_df = exp2.get("metrics", pd.DataFrame())
        if not metrics_df.empty:
            for _, row in metrics_df.iterrows():
                lines.append(
                    f"| {row.get('strategy', 'N/A')} | "
                    f"{row.get('profitable_trades', 0)} | "
                    f"{row.get('non_profitable_trades', 0)} | "
                    f"{row.get('total_trades', 0)} | "
                    f"{row.get('average_gain', float('nan')):.4f} | "
                    f"{row.get('average_loss', float('nan')):.4f} | "
                    f"{row.get('appt', float('nan')):.4f} | "
                    f"{row.get('total_profit', 0.0):.3f} |"
                )
        lines.append("")

        lines += [
            "### Paper Benchmark Comparison",
            "",
            "| Strategy | Replicated APPT | Paper APPT | Sign Match | Replicated Total | Paper Total |",
            "|----------|----------------|------------|------------|-----------------|-------------|",
        ]
        paper_benchmarks = exp2.get("paper_benchmarks", {})
        if not metrics_df.empty:
            for _, row in metrics_df.iterrows():
                strategy = row.get("strategy", "N/A")
                bench = paper_benchmarks.get(strategy, {})
                paper_appt = bench.get("appt", float("nan"))
                paper_total = bench.get("total_profit", float("nan"))
                rep_appt = row.get("appt", float("nan"))
                rep_total = row.get("total_profit", float("nan"))
                sign_match = (
                    "✅" if (not np.isnan(rep_appt) and (rep_appt >= 0) == (paper_appt >= 0))
                    else "❌"
                )
                lines.append(
                    f"| {strategy} | {rep_appt:.4f} | {paper_appt:.4f} | "
                    f"{sign_match} | {rep_total:.3f} | {paper_total:.3f} |"
                )
        lines.append("")

        lines += [
            "### Paper Reference Benchmarks (327 trades each)",
            "",
            "| Strategy | Wins | Losses | Avg Gain | Avg Loss | APPT | Total Profit |",
            "|----------|------|--------|----------|----------|------|-------------|",
            "| Standard | 159 | 168 | 14.005 | -13.9601 | -0.363 | -118.5 |",
            "| Linear | 169 | 158 | 13.7997 | -14.0196 | 0.358 | 117.052 |",
            "| Discrete | 183 | 144 | 11.3495 | -13.9347 | 0.215 | 70.35 |",
            "",
            "### Output Files",
            "",
            "- `results/exp2_trades_dataset.csv` — Per-trade results for all strategies",
            "- `results/exp2_cumulative_pips.png` — Cumulative pips chart",
            "- `results/exp2_metrics.csv` — Strategy performance metrics",
            "- `results/exp2_benchmark_comparison.csv` — Comparison to paper benchmarks",
            "- `results/exp2_ambiguity_log.txt` — Ambiguity resolutions",
            "",
        ]
    elif exp2.get("status") == "no_data":
        lines += [
            "**Status:** ⚠️ No data / No events detected",
            "",
            "> Note: This may occur if the Massive API returns no data for the specified",
            "> date range and ticker. Check MASSIVE_TOKEN environment variable and API access.",
            "",
        ]
    else:
        lines += [
            f"**Status:** ❌ Failed",
            f"**Error:** {exp2.get('error', 'Unknown error')}",
            "",
        ]

    lines += [
        "---",
        "",
        "## Sensitivity Analyses (Optional)",
        "",
        "The following alternative width eligibility rules can be tested as sensitivity analyses:",
        "- 12-35 pips",
        "- 6-10 pips",
        "- ≤15 pips only",
        "",
        "These are clearly labeled as sensitivity analyses, not the primary replication.",
        "",
        "---",
        "",
        "## Implementation Notes",
        "",
        "- All computations are fully causal (no look-ahead bias).",
        "- ADX is computed from past data only; consolidation detection uses only historical bars.",
        "- Pre-consolidation ADX label uses only data before the consolidation start.",
        "- No transaction costs, slippage, or spread in the base replication.",
        "- Missing bars are logged but not filled.",
        "",
    ]

    with open(md_path, "w") as f:
        f.write("\n".join(lines))

    logger.info("Saved RESULTS.md to %s", md_path)


if __name__ == "__main__":
    run_all_experiments()
