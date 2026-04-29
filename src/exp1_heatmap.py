"""
Experiment 1: Exploratory EUR/GBP analysis of whether pre-consolidation ADX
predicts breakout success.

Replicates the paper's in-sample exploratory analysis on EUR/GBP 1-minute FX data.

Ambiguity resolutions logged:
1. Paper says "one year" but Oct 2017–Dec 2018 is 15 months. We use the full
   15-month span (Oct 2017 – Dec 2018) as stated in the paper's date range.
2. Width rule: 8-15 pips (method section), not 15 pips (flowchart).
3. ADX: Wilder-style 14-period via `ta` library.
4. ADX averaging window: floor(0.1 * consolidation_length_minutes).
5. Entry: close of breakout bar.
6. Intrabar tie-break: stop wins.
7. No transaction costs (pre-cost descriptive stage).
8. ADX bin edges: [0, 15, 20, 25, 30, 35, 100] (documented choice).
9. Width bins: [8, 10, 12, 15] pips (reflecting 8-15 pip range).
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.consolidation import (
    ConsolidationEvent,
    detect_consolidations,
    events_to_dataframe,
)
from src.data_loader import load_or_fetch, price_to_pips
from src.indicators import add_indicators, adx_label_window, rolling_mean_adx
from src.simulation import simulate_all_strategies

logger = logging.getLogger(__name__)

# --- Experiment parameters ---
PAIR = "EURGBP"
TICKER = "C:EURGBP"
# Ambiguity: paper says "one year" but Oct 2017–Dec 2018 is 15 months.
# We use the full stated date range. This ambiguity is logged.
EXP1_START = "2017-09-01"   # warm-up start (before Oct 2017 for ADX warm-up)
EXP1_EVAL_START = "2017-10-01"
EXP1_END = "2018-12-31"

# ADX bin edges for heatmap (documented choice — paper does not specify)
ADX_BINS = [0, 15, 20, 25, 30, 35, 100]
ADX_BIN_LABELS = ["0-15", "15-20", "20-25", "25-30", "30-35", "35+"]

# Width bins (reflecting 8-15 pip range)
WIDTH_BINS = [8, 10, 12, 15]
WIDTH_BIN_LABELS = ["8-10", "10-12", "12-15"]

TARGET_MULTIPLES = [1.0, 2.0]


def run_exp1(
    results_dir: str = "results",
    cache_dir: str = "data",
    width_min_pips: float = 8.0,
    width_max_pips: float = 15.0,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    """
    Run Experiment 1: EUR/GBP exploratory heatmap analysis.

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
        (breakout_dataset, metrics_summary, ambiguity_log)
    """
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    ambiguity_log = _build_ambiguity_log()

    # --- 1. Load data ---
    logger.info("Loading EUR/GBP 1-minute data...")
    cache_path = os.path.join(cache_dir, "eurgbp_1min.parquet")
    df = load_or_fetch(TICKER, EXP1_START, EXP1_END, cache_path=cache_path)

    if df.empty:
        logger.error("No data loaded for EUR/GBP. Aborting Exp1.")
        return pd.DataFrame(), pd.DataFrame(), ambiguity_log

    logger.info("Loaded %d bars from %s to %s", len(df), df["timestamp"].iloc[0], df["timestamp"].iloc[-1])

    # Filter to evaluation window (keep warm-up for indicators)
    df = df.reset_index(drop=True)

    # --- 2. Compute indicators ---
    logger.info("Computing ATR and ADX indicators...")
    df = add_indicators(df, window=14)

    # Filter to evaluation window after warm-up
    eval_mask = df["timestamp"] >= pd.Timestamp(EXP1_EVAL_START, tz="UTC")
    eval_start_idx = df[eval_mask].index[0] if eval_mask.any() else 0
    logger.info("Evaluation window starts at index %d (%s)", eval_start_idx, EXP1_EVAL_START)

    # --- 3. Detect consolidations ---
    logger.info("Detecting consolidation-breakout events...")
    # Only detect from eval_start_idx onwards to avoid warm-up period events
    df_eval = df.iloc[eval_start_idx:].reset_index(drop=True)
    # But keep full df for ADX lookback
    df_full = df.copy()

    events = detect_consolidations(
        df_eval,
        pair=PAIR,
        width_min_pips=width_min_pips,
        width_max_pips=width_max_pips,
    )

    if not events:
        logger.warning("No consolidation events detected for EUR/GBP.")
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

    # --- 5. Simulate breakout outcomes for target = 1x and 2x ---
    logger.info("Simulating breakout outcomes...")
    all_trades = []
    for target_mult in TARGET_MULTIPLES:
        trades = _simulate_with_target(df_full, events, PAIR, target_mult)
        all_trades.append(trades)

    trades_df = pd.concat(all_trades, ignore_index=True)

    # --- 6. Build per-breakout dataset ---
    breakout_df = _build_breakout_dataset(events, trades_df)

    # --- 7. Heatmap analysis ---
    logger.info("Generating heatmaps...")
    for target_mult in TARGET_MULTIPLES:
        _generate_heatmap(
            breakout_df,
            target_mult=target_mult,
            results_dir=results_dir,
        )

    # --- 8. Summary metrics ---
    metrics = _compute_summary_metrics(breakout_df)

    # --- 9. Save outputs ---
    breakout_path = os.path.join(results_dir, "exp1_breakout_dataset.csv")
    breakout_df.to_csv(breakout_path, index=False)
    logger.info("Saved breakout dataset to %s", breakout_path)

    metrics_path = os.path.join(results_dir, "exp1_metrics.csv")
    metrics.to_csv(metrics_path, index=False)
    logger.info("Saved metrics to %s", metrics_path)

    # Save ambiguity log
    amb_path = os.path.join(results_dir, "exp1_ambiguity_log.txt")
    with open(amb_path, "w") as f:
        for k, v in ambiguity_log.items():
            f.write(f"{k}: {v}\n")
    logger.info("Saved ambiguity log to %s", amb_path)

    return breakout_df, metrics, ambiguity_log


def _simulate_with_target(
    df: pd.DataFrame,
    events: List[ConsolidationEvent],
    pair: str,
    target_mult: float,
) -> pd.DataFrame:
    """
    Simulate breakout outcomes with a fixed target multiplier.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLC DataFrame.
    events : List[ConsolidationEvent]
        Accepted events with ADX labels.
    pair : str
        FX pair name.
    target_mult : float
        Target distance multiplier (1.0 or 2.0).

    Returns
    -------
    pd.DataFrame
        Trade records with target_multiple column added.
    """
    records = []
    for event in events:
        if np.isnan(event.adx_label_value):
            continue

        U = event.upper_edge
        D = event.lower_edge
        w_price = U - D
        direction = event.breakout_direction
        breakout_idx = event.breakout_idx

        entry_price = df["close"].iloc[breakout_idx]

        if direction == "bullish":
            stop_price = D
            target_price = entry_price + target_mult * w_price
        else:
            stop_price = U
            target_price = entry_price - target_mult * w_price

        outcome, exit_time, exit_price = _simulate_forward(
            df, breakout_idx + 1, direction, target_price, stop_price
        )

        if outcome in ("win", "loss"):
            pnl_price = (
                (exit_price - entry_price) if direction == "bullish"
                else (entry_price - exit_price)
            )
            pnl_pips = price_to_pips(pnl_price, pair)
        else:
            pnl_pips = float("nan")

        records.append(
            {
                "consolidation_start": event.consolidation_start,
                "breakout_time": event.breakout_time,
                "direction": direction,
                "width_pips": event.width_pips,
                "duration_minutes": event.consolidation_length_minutes,
                "adx_label_value": event.adx_label_value,
                "target_multiple": target_mult,
                "target_price": target_price,
                "stop_price": stop_price,
                "entry_price": entry_price,
                "outcome": outcome,
                "exit_time": exit_time,
                "pnl_pips": pnl_pips,
            }
        )

    return pd.DataFrame(records)


def _simulate_forward(
    df: pd.DataFrame,
    start_idx: int,
    direction: str,
    target_price: float,
    stop_price: float,
) -> Tuple[str, Optional[pd.Timestamp], Optional[float]]:
    """
    Simulate a trade forward from start_idx until target or stop is hit.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame.
    start_idx : int
        First bar to check after entry.
    direction : str
        'bullish' or 'bearish'.
    target_price : float
        Profit target price.
    stop_price : float
        Stop-loss price.

    Returns
    -------
    Tuple[str, Optional[pd.Timestamp], Optional[float]]
        (outcome, exit_time, exit_price)
    """
    n = len(df)
    for k in range(start_idx, n):
        h = df["high"].iloc[k]
        lo = df["low"].iloc[k]

        if direction == "bullish":
            hit_target = h >= target_price
            hit_stop = lo <= stop_price
        else:
            hit_target = lo <= target_price
            hit_stop = h >= stop_price

        if hit_target and hit_stop:
            # Tie-break: stop wins (conservative)
            return "loss", df["timestamp"].iloc[k], stop_price
        elif hit_target:
            return "win", df["timestamp"].iloc[k], target_price
        elif hit_stop:
            return "loss", df["timestamp"].iloc[k], stop_price

    return "open", None, None


def _build_breakout_dataset(
    events: List[ConsolidationEvent],
    trades_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the per-breakout dataset with outcomes for both target multiples.

    Parameters
    ----------
    events : List[ConsolidationEvent]
        Accepted events.
    trades_df : pd.DataFrame
        Simulated trade records.

    Returns
    -------
    pd.DataFrame
        Per-breakout dataset.
    """
    if trades_df.empty:
        return pd.DataFrame()

    # Pivot to have one row per breakout with outcomes for each target multiple
    pivot = trades_df.pivot_table(
        index="consolidation_start",
        columns="target_multiple",
        values=["outcome", "pnl_pips"],
        aggfunc="first",
    )
    pivot.columns = [f"{col[0]}_target{col[1]:.0f}x" for col in pivot.columns]
    pivot = pivot.reset_index()

    # Merge with event metadata
    event_df = events_to_dataframe(events)
    merged = event_df.merge(pivot, on="consolidation_start", how="left")

    # Add ADX bins
    merged["adx_bin"] = pd.cut(
        merged["adx_label_value"],
        bins=ADX_BINS,
        labels=ADX_BIN_LABELS,
        right=False,
    )

    # Add width bins
    merged["width_bin"] = pd.cut(
        merged["width_pips"],
        bins=WIDTH_BINS,
        labels=WIDTH_BIN_LABELS,
        right=False,
    )

    return merged


def _generate_heatmap(
    breakout_df: pd.DataFrame,
    target_mult: float,
    results_dir: str,
) -> None:
    """
    Generate and save a heatmap of win rates by ADX bin and width bin.

    Parameters
    ----------
    breakout_df : pd.DataFrame
        Per-breakout dataset with adx_bin, width_bin, and outcome columns.
    target_mult : float
        Target multiplier (1.0 or 2.0).
    results_dir : str
        Directory to save the heatmap image.
    """
    outcome_col = f"outcome_target{target_mult:.0f}x"
    if outcome_col not in breakout_df.columns:
        logger.warning("Column %s not found in breakout_df.", outcome_col)
        return

    df_valid = breakout_df[breakout_df[outcome_col].isin(["win", "loss"])].copy()
    df_valid["is_win"] = (df_valid[outcome_col] == "win").astype(float)

    if df_valid.empty:
        logger.warning("No valid trades for target_mult=%.1f heatmap.", target_mult)
        return

    # Compute win rate per cell
    pivot = df_valid.pivot_table(
        index="adx_bin",
        columns="width_bin",
        values="is_win",
        aggfunc="mean",
    )

    # Count per cell
    count_pivot = df_valid.pivot_table(
        index="adx_bin",
        columns="width_bin",
        values="is_win",
        aggfunc="count",
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Win rate heatmap
    sns.heatmap(
        pivot,
        ax=axes[0],
        annot=True,
        fmt=".2f",
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        linewidths=0.5,
        cbar_kws={"label": "Win Rate"},
    )
    axes[0].set_title(
        f"EUR/GBP Win Rate by ADX & Width\n(Target = {target_mult:.0f}x Width)",
        fontsize=12,
    )
    axes[0].set_xlabel("Consolidation Width (pips)", fontsize=10)
    axes[0].set_ylabel("Pre-Consolidation ADX", fontsize=10)

    # Count heatmap
    sns.heatmap(
        count_pivot,
        ax=axes[1],
        annot=True,
        fmt=".0f",
        cmap="Blues",
        linewidths=0.5,
        cbar_kws={"label": "Count"},
    )
    axes[1].set_title(
        f"EUR/GBP Trade Count by ADX & Width\n(Target = {target_mult:.0f}x Width)",
        fontsize=12,
    )
    axes[1].set_xlabel("Consolidation Width (pips)", fontsize=10)
    axes[1].set_ylabel("Pre-Consolidation ADX", fontsize=10)

    plt.tight_layout()
    fname = os.path.join(results_dir, f"exp1_heatmap_target{target_mult:.0f}x.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved heatmap to %s", fname)


def _compute_summary_metrics(breakout_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute overall win rates and per-ADX-bin win rates for both target multiples.

    Parameters
    ----------
    breakout_df : pd.DataFrame
        Per-breakout dataset.

    Returns
    -------
    pd.DataFrame
        Summary metrics table.
    """
    rows = []
    for target_mult in TARGET_MULTIPLES:
        outcome_col = f"outcome_target{target_mult:.0f}x"
        if outcome_col not in breakout_df.columns:
            continue

        df_valid = breakout_df[breakout_df[outcome_col].isin(["win", "loss"])].copy()
        if df_valid.empty:
            continue

        total = len(df_valid)
        wins = (df_valid[outcome_col] == "win").sum()
        overall_wr = wins / total if total > 0 else float("nan")

        rows.append(
            {
                "target_multiple": target_mult,
                "adx_bin": "Overall",
                "total_trades": total,
                "wins": wins,
                "win_rate": overall_wr,
            }
        )

        # Per ADX bin
        for adx_bin in ADX_BIN_LABELS:
            bin_df = df_valid[df_valid["adx_bin"] == adx_bin]
            if bin_df.empty:
                continue
            bin_total = len(bin_df)
            bin_wins = (bin_df[outcome_col] == "win").sum()
            bin_wr = bin_wins / bin_total if bin_total > 0 else float("nan")
            rows.append(
                {
                    "target_multiple": target_mult,
                    "adx_bin": adx_bin,
                    "total_trades": bin_total,
                    "wins": bin_wins,
                    "win_rate": bin_wr,
                }
            )

    return pd.DataFrame(rows)


def _build_ambiguity_log() -> Dict:
    """Build the ambiguity resolution log for Experiment 1."""
    return {
        "date_range_ambiguity": (
            "Paper says 'one year' but Oct 2017–Dec 2018 is 15 months. "
            "We use the full 15-month span as stated in the paper's date range."
        ),
        "width_rule": (
            "Flowchart says 'within 15 pips'; method section says 8-15 pips. "
            "Primary replication uses 8-15 pips."
        ),
        "adx_formula": (
            "Paper prints nonstandard ADX formulas. "
            "We use Wilder-style 14-period ADX via `ta` library as best paper-consistent implementation."
        ),
        "adx_window_rounding": (
            "Paper gives no rounding rule for 10% of consolidation length. "
            "We use floor() as the deterministic integer conversion rule."
        ),
        "entry_price": (
            "Paper does not define exact entry price semantics. "
            "We use the close of the breakout bar as the entry price."
        ),
        "intrabar_tiebreak": (
            "If both stop and target are hit in the same bar, stop wins (conservative rule). "
            "Paper gives no guidance."
        ),
        "adx_bins": (
            "Paper does not state exact ADX bin edges for heatmap. "
            "We use [0, 15, 20, 25, 30, 35, 100] with labels ['0-15','15-20','20-25','25-30','30-35','35+']."
        ),
        "width_bins": (
            "Width bins: [8, 10, 12, 15] pips, labels ['8-10','10-12','12-15'], "
            "reflecting the 8-15 pip range."
        ),
        "data_source": (
            "No vendor, quote convention, or bid/ask/mid specification given in paper. "
            "We use Massive API with default quote type. This is an unresolved assumption."
        ),
        "missing_bars": (
            "Weekend gaps and missing minute bars are NOT filled. "
            "Their presence is logged."
        ),
    }
