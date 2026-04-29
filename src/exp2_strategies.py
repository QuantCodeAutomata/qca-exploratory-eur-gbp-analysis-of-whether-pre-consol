"""
Experiment 2: Out-of-sample GBP/USD comparison of Standard, Discrete, and Linear
ADX-conditioned breakout strategies.

Replicates the paper's out-of-sample strategy evaluation on GBP/USD 1-minute FX data.

Paper's reported benchmarks (327 trades each):
  Standard:  159 wins / 168 losses / avg gain 14.005 / avg loss -13.9601 / APPT -0.363 / total -118.5
  Linear:    169 wins / 158 losses / avg gain 13.7997 / avg loss -14.0196 / APPT 0.358 / total 117.052
  Discrete:  183 wins / 144 losses / avg gain 11.3495 / avg loss -13.9347 / APPT 0.215 / total 70.35

Ambiguity resolutions logged:
1. GBP/USD date range: paper says "15 months, previously unused pair" but gives no dates.
   We mirror the EUR/GBP calendar span: Oct 2017 – Dec 2018 (15 months).
2. Width rule: 8-15 pips (method section primary rule).
3. ADX: Wilder-style 14-period via `ta` library.
4. ADX averaging window: floor(0.1 * consolidation_length_minutes).
5. ADX >= 25 = strong trend for Discrete strategy.
6. Entry: close of breakout bar.
7. Intrabar tie-break: stop wins (conservative).
8. No transaction costs (paper's main table is in raw pips, pre-cost).
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.consolidation import (
    ConsolidationEvent,
    detect_consolidations,
    events_to_dataframe,
)
from src.data_loader import load_or_fetch, price_to_pips
from src.indicators import add_indicators, adx_label_window, rolling_mean_adx
from src.simulation import (
    STRATEGY_NAMES,
    compute_all_metrics,
    cumulative_pips,
    simulate_all_strategies,
)

logger = logging.getLogger(__name__)

# --- Experiment parameters ---
PAIR = "GBPUSD"
TICKER = "C:GBPUSD"
# Ambiguity: paper gives no dates for GBP/USD. We mirror EUR/GBP span.
EXP2_START = "2017-09-01"   # warm-up start
EXP2_EVAL_START = "2017-10-01"
EXP2_END = "2018-12-31"

# Paper's reference benchmarks for comparison
PAPER_BENCHMARKS = {
    "Standard": {
        "profitable_trades": 159,
        "non_profitable_trades": 168,
        "total_trades": 327,
        "average_gain": 14.005,
        "average_loss": -13.9601,
        "appt": -0.363,
        "total_profit": -118.5,
    },
    "Linear": {
        "profitable_trades": 169,
        "non_profitable_trades": 158,
        "total_trades": 327,
        "average_gain": 13.7997,
        "average_loss": -14.0196,
        "appt": 0.358,
        "total_profit": 117.052,
    },
    "Discrete": {
        "profitable_trades": 183,
        "non_profitable_trades": 144,
        "total_trades": 327,
        "average_gain": 11.3495,
        "average_loss": -13.9347,
        "appt": 0.215,
        "total_profit": 70.35,
    },
}


def run_exp2(
    results_dir: str = "results",
    cache_dir: str = "data",
    width_min_pips: float = 8.0,
    width_max_pips: float = 15.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run Experiment 2: GBP/USD three-strategy out-of-sample comparison.

    Parameters
    ----------
    results_dir : str
        Directory to save output files.
    cache_dir : str
        Directory for data cache files.
    width_min_pips : float
        Minimum consolidation width in pips.
    width_max_pips : float
        Maximum consolidation width in pips.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, Dict]
        (trades_dataset, metrics_summary, ambiguity_log)
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    ambiguity_log = _build_ambiguity_log()

    # --- 1. Load data ---
    logger.info("Loading GBP/USD 1-minute data...")
    cache_path = os.path.join(cache_dir, "gbpusd_1min.parquet")
    df = load_or_fetch(TICKER, EXP2_START, EXP2_END, cache_path=cache_path)

    if df.empty:
        logger.error("No data loaded for GBP/USD. Aborting Exp2.")
        return pd.DataFrame(), pd.DataFrame(), ambiguity_log

    logger.info("Loaded %d bars from %s to %s", len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    df = df.reset_index(drop=True)

    # --- 2. Compute indicators ---
    logger.info("Computing ATR and ADX indicators...")
    df = add_indicators(df, window=14)

    # Filter to evaluation window
    eval_mask = df["timestamp"] >= pd.Timestamp(EXP2_EVAL_START, tz="UTC")
    eval_start_idx = df[eval_mask].index[0] if eval_mask.any() else 0
    logger.info("Evaluation window starts at index %d (%s)", eval_start_idx, EXP2_EVAL_START)

    # --- 3. Detect consolidations ---
    logger.info("Detecting consolidation-breakout events...")
    df_eval = df.iloc[eval_start_idx:].reset_index(drop=True)
    df_full = df.copy()

    events = detect_consolidations(
        df_eval,
        pair=PAIR,
        width_min_pips=width_min_pips,
        width_max_pips=width_max_pips,
    )

    if not events:
        logger.warning("No consolidation events detected for GBP/USD.")
        return pd.DataFrame(), pd.DataFrame(), ambiguity_log

    logger.info("Detected %d events.", len(events))

    # Adjust event indices to full df
    for event in events:
        event.consolidation_start_idx += eval_start_idx
        event.breakout_idx += eval_start_idx

    # --- 4. Label each event with pre-consolidation ADX ---
    logger.info("Computing pre-consolidation ADX labels...")
    adx_series = df_full["adx"].reset_index(drop=True)

    for event in events:
        window_minutes = adx_label_window(event.consolidation_length_minutes)
        event.adx_label_window_minutes = window_minutes
        event.adx_label_value = rolling_mean_adx(
            adx_series,
            end_idx=event.consolidation_start_idx,
            window_minutes=window_minutes,
        )

    # Filter out events with NaN ADX labels
    valid_events = [e for e in events if not np.isnan(e.adx_label_value)]
    logger.info(
        "%d events with valid ADX labels (out of %d total).",
        len(valid_events),
        len(events),
    )

    # --- 5. Simulate all three strategies on the same event set ---
    logger.info("Simulating three strategies...")
    trades_df = simulate_all_strategies(df_full, valid_events, PAIR)

    if trades_df.empty:
        logger.warning("No trades simulated.")
        return pd.DataFrame(), pd.DataFrame(), ambiguity_log

    # --- 6. Compute performance metrics ---
    logger.info("Computing performance metrics...")
    metrics_df = compute_all_metrics(trades_df)

    # --- 7. Validate trade count equality across strategies ---
    _validate_trade_counts(trades_df, metrics_df)

    # --- 8. Compare to paper benchmarks ---
    comparison_df = _compare_to_benchmarks(metrics_df)

    # --- 9. Generate cumulative pips chart ---
    logger.info("Generating cumulative pips chart...")
    _plot_cumulative_pips(trades_df, results_dir)

    # --- 10. Save outputs ---
    trades_path = os.path.join(results_dir, "exp2_trades_dataset.csv")
    trades_df.to_csv(trades_path, index=False)
    logger.info("Saved trades dataset to %s", trades_path)

    metrics_path = os.path.join(results_dir, "exp2_metrics.csv")
    metrics_df.to_csv(metrics_path, index=False)
    logger.info("Saved metrics to %s", metrics_path)

    comparison_path = os.path.join(results_dir, "exp2_benchmark_comparison.csv")
    comparison_df.to_csv(comparison_path, index=False)
    logger.info("Saved benchmark comparison to %s", comparison_path)

    amb_path = os.path.join(results_dir, "exp2_ambiguity_log.txt")
    with open(amb_path, "w") as f:
        for k, v in ambiguity_log.items():
            f.write(f"{k}: {v}\n")
    logger.info("Saved ambiguity log to %s", amb_path)

    return trades_df, metrics_df, ambiguity_log


def _validate_trade_counts(trades_df: pd.DataFrame, metrics_df: pd.DataFrame) -> None:
    """
    Validate that all three strategies have the same number of closed trades.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Per-trade results.
    metrics_df : pd.DataFrame
        Summary metrics.
    """
    counts = {}
    for strategy in STRATEGY_NAMES:
        if strategy in trades_df["strategy_name"].values:
            n = len(
                trades_df[
                    (trades_df["strategy_name"] == strategy) &
                    (trades_df["outcome"].isin(["win", "loss"]))
                ]
            )
            counts[strategy] = n

    if len(set(counts.values())) > 1:
        logger.warning(
            "Trade count mismatch across strategies: %s. "
            "This may indicate an implementation issue.",
            counts,
        )
    else:
        logger.info("Trade count validation passed: all strategies have %d trades.", list(counts.values())[0] if counts else 0)

    # Validate APPT = total_profit / total_trades
    for _, row in metrics_df.iterrows():
        if row["total_trades"] > 0 and not np.isnan(row["appt"]):
            expected_appt = row["total_profit"] / row["total_trades"]
            if abs(expected_appt - row["appt"]) > 0.01:
                logger.warning(
                    "APPT validation failed for %s: computed %.4f, stored %.4f",
                    row["strategy"],
                    expected_appt,
                    row["appt"],
                )


def _compare_to_benchmarks(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare replicated metrics to paper's reported benchmarks.

    Parameters
    ----------
    metrics_df : pd.DataFrame
        Replicated metrics.

    Returns
    -------
    pd.DataFrame
        Comparison table.
    """
    rows = []
    for _, row in metrics_df.iterrows():
        strategy = row["strategy"]
        if strategy not in PAPER_BENCHMARKS:
            continue

        bench = PAPER_BENCHMARKS[strategy]
        rows.append(
            {
                "strategy": strategy,
                "replicated_wins": row["profitable_trades"],
                "paper_wins": bench["profitable_trades"],
                "replicated_losses": row["non_profitable_trades"],
                "paper_losses": bench["non_profitable_trades"],
                "replicated_total": row["total_trades"],
                "paper_total": bench["total_trades"],
                "replicated_avg_gain": row["average_gain"],
                "paper_avg_gain": bench["average_gain"],
                "replicated_avg_loss": row["average_loss"],
                "paper_avg_loss": bench["average_loss"],
                "replicated_appt": row["appt"],
                "paper_appt": bench["appt"],
                "replicated_total_profit": row["total_profit"],
                "paper_total_profit": bench["total_profit"],
                "appt_sign_match": (
                    (row["appt"] >= 0) == (bench["appt"] >= 0)
                    if not np.isnan(row["appt"]) else False
                ),
            }
        )

    return pd.DataFrame(rows)


def _plot_cumulative_pips(trades_df: pd.DataFrame, results_dir: str) -> None:
    """
    Plot cumulative pips over trades for all three strategies.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Per-trade results for all strategies.
    results_dir : str
        Directory to save the chart.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = {"Standard": "red", "Linear": "green", "Discrete": "blue"}
    linestyles = {"Standard": "--", "Linear": "-", "Discrete": "-."}

    for strategy in STRATEGY_NAMES:
        if strategy not in trades_df["strategy_name"].values:
            continue
        cum = cumulative_pips(trades_df, strategy)
        if cum.empty:
            continue
        ax.plot(
            cum.index,
            cum.values,
            label=strategy,
            color=colors.get(strategy, "black"),
            linestyle=linestyles.get(strategy, "-"),
            linewidth=2,
        )

    ax.axhline(y=0, color="black", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Total Trades", fontsize=12)
    ax.set_ylabel("Cumulative Returns (pips)", fontsize=12)
    ax.set_title(
        "GBP/USD Out-of-Sample: Cumulative Returns by Strategy\n"
        "(Standard vs. Discrete vs. Linear ADX-Conditioned)",
        fontsize=13,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fname = os.path.join(results_dir, "exp2_cumulative_pips.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved cumulative pips chart to %s", fname)


def _build_ambiguity_log() -> Dict:
    """Build the ambiguity resolution log for Experiment 2."""
    return {
        "date_range": (
            "Paper says '15 months, previously unused pair' but gives no dates. "
            "We mirror the EUR/GBP calendar span: Oct 2017 – Dec 2018."
        ),
        "width_rule": (
            "Flowchart says 'within 15 pips'; method section says 8-15 pips. "
            "Primary replication uses 8-15 pips."
        ),
        "adx_formula": (
            "Paper prints nonstandard ADX formulas. "
            "We use Wilder-style 14-period ADX via `ta` library."
        ),
        "adx_window_rounding": (
            "Paper gives no rounding rule for 10% of consolidation length. "
            "We use floor() as the deterministic integer conversion rule."
        ),
        "adx_strong_threshold": (
            "Paper alternates between 'above 25' and 'ADX of 25 indicating strong trend'. "
            "We use ADX >= 25 as the primary strong-trend rule."
        ),
        "entry_price": (
            "Paper does not define exact entry price semantics. "
            "We use the close of the breakout bar as the entry price."
        ),
        "intrabar_tiebreak": (
            "If both stop and target are hit in the same bar, stop wins (conservative rule). "
            "Paper gives no guidance."
        ),
        "open_trades": (
            "Trades still open at sample end are excluded from summary metrics."
        ),
        "no_costs": (
            "No transaction costs, slippage, or spread. "
            "Paper's main table is in raw pips, pre-cost."
        ),
        "data_source": (
            "No vendor, quote convention, or bid/ask/mid specification given in paper. "
            "We use Massive API with default quote type. This is an unresolved assumption."
        ),
    }
