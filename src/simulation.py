"""
Trade simulation module for breakout strategies.

Implements three strategies on detected consolidation-breakout events:
  1. Standard: fixed target = 1.0 * w, stop = opposite edge
  2. Discrete: target = 1.5 * w if ADX >= 25 (strong), else 0.5 * w
  3. Linear: target = (ADX / 25) * w

Ambiguity resolutions:
- Entry: first bar close outside the consolidation range (breakout bar close).
- Stop-loss: opposite consolidation edge from breakout direction.
- Intrabar tie-break: if both stop and target are hit in the same bar, stop wins
  (conservative assumption; paper gives no guidance).
- ADX >= 25 = strong trend for Discrete strategy (paper alternates between
  'above 25' and 'ADX of 25 indicating strong trend'; >= 25 is primary rule).
- Trades open at sample end: excluded from summary metrics (marked 'open').
- No transaction costs, slippage, or spread (paper's descriptive stage is pre-cost).

# Custom — Context7 found no library equivalent (paper custom simulation)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.consolidation import ConsolidationEvent
from src.data_loader import price_to_pips

logger = logging.getLogger(__name__)

STRATEGY_NAMES = ["Standard", "Discrete", "Linear"]
ADX_STRONG_THRESHOLD = 25.0  # Ambiguity resolution: >= 25 = strong


def _target_multiplier(strategy: str, adx_value: float) -> float:
    """
    Compute the target distance multiplier for a given strategy and ADX value.

    Parameters
    ----------
    strategy : str
        One of 'Standard', 'Discrete', 'Linear'.
    adx_value : float
        Pre-consolidation ADX label.

    Returns
    -------
    float
        Target multiplier (target distance = multiplier * consolidation_width).
    """
    if strategy == "Standard":
        return 1.0
    elif strategy == "Discrete":
        # ADX >= 25 → strong → target = 1.5 * w; else target = 0.5 * w
        return 1.5 if adx_value >= ADX_STRONG_THRESHOLD else 0.5
    elif strategy == "Linear":
        # target = (ADX / 25) * w
        return adx_value / ADX_STRONG_THRESHOLD
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def simulate_trade(
    df: pd.DataFrame,
    event: ConsolidationEvent,
    strategy: str,
    pair: str,
) -> Dict:
    """
    Simulate a single trade for a given strategy and breakout event.

    Entry is at the close of the breakout bar (first bar outside the range).
    Stop-loss is at the opposite consolidation edge.
    Target is at entry ± target_multiplier * width.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLC DataFrame (integer-indexed, sorted chronologically).
    event : ConsolidationEvent
        The detected consolidation-breakout event.
    strategy : str
        Strategy name: 'Standard', 'Discrete', or 'Linear'.
    pair : str
        FX pair name for pip conversion.

    Returns
    -------
    Dict
        Trade record with all required output fields.
    """
    breakout_idx = event.breakout_idx
    direction = event.breakout_direction
    U = event.upper_edge
    D = event.lower_edge
    w_price = U - D
    adx_value = event.adx_label_value

    if np.isnan(adx_value):
        logger.warning(
            "ADX label is NaN for event at %s; skipping trade.", event.consolidation_start
        )
        return _build_trade_record(event, strategy, pair, outcome="skipped_no_adx")

    mult = _target_multiplier(strategy, adx_value)
    entry_price = df["close"].iloc[breakout_idx]

    if direction == "bullish":
        stop_price = D  # stop at lower edge
        target_price = entry_price + mult * w_price
    else:  # bearish
        stop_price = U  # stop at upper edge
        target_price = entry_price - mult * w_price

    # Simulate forward from the bar after entry
    n = len(df)
    outcome = "open"
    exit_time = None
    exit_price = None

    for k in range(breakout_idx + 1, n):
        h = df["high"].iloc[k]
        lo = df["low"].iloc[k]

        if direction == "bullish":
            hit_target = h >= target_price
            hit_stop = lo <= stop_price
        else:
            hit_target = lo <= target_price
            hit_stop = h >= stop_price

        if hit_target and hit_stop:
            # Intrabar tie-break: stop wins (conservative)
            outcome = "loss"
            exit_price = stop_price
            exit_time = df["timestamp"].iloc[k]
            logger.debug(
                "Intrabar tie-break at bar %d: stop wins (conservative rule).", k
            )
            break
        elif hit_target:
            outcome = "win"
            exit_price = target_price
            exit_time = df["timestamp"].iloc[k]
            break
        elif hit_stop:
            outcome = "loss"
            exit_price = stop_price
            exit_time = df["timestamp"].iloc[k]
            break

    if outcome == "open":
        logger.debug(
            "Trade at %s still open at sample end; excluded from metrics.",
            event.consolidation_start,
        )
        return _build_trade_record(
            event, strategy, pair, outcome="open",
            entry_price=entry_price, stop_price=stop_price,
            target_price=target_price, exit_time=None, exit_price=None,
        )

    pnl_price = (exit_price - entry_price) if direction == "bullish" else (entry_price - exit_price)
    pnl_pips = price_to_pips(pnl_price, pair)

    return _build_trade_record(
        event, strategy, pair,
        outcome=outcome,
        entry_price=entry_price,
        stop_price=stop_price,
        target_price=target_price,
        exit_time=exit_time,
        exit_price=exit_price,
        pnl_pips=pnl_pips,
    )


def _build_trade_record(
    event: ConsolidationEvent,
    strategy: str,
    pair: str,
    outcome: str = "unknown",
    entry_price: Optional[float] = None,
    stop_price: Optional[float] = None,
    target_price: Optional[float] = None,
    exit_time: Optional[pd.Timestamp] = None,
    exit_price: Optional[float] = None,
    pnl_pips: Optional[float] = None,
) -> Dict:
    """Build a complete trade record dictionary."""
    return {
        "pair": pair,
        "consolidation_start": event.consolidation_start,
        "breakout_time": event.breakout_time,
        "direction": event.breakout_direction,
        "width_pips": event.width_pips,
        "duration_minutes": event.consolidation_length_minutes,
        "adx_label_window_minutes": event.adx_label_window_minutes,
        "adx_label_value": event.adx_label_value,
        "strategy_name": strategy,
        "entry_price": entry_price,
        "stop_price": stop_price,
        "target_price": target_price,
        "exit_time": exit_time,
        "exit_reason": outcome,
        "outcome": outcome,
        "pnl_pips": pnl_pips,
    }


def simulate_all_strategies(
    df: pd.DataFrame,
    events: List[ConsolidationEvent],
    pair: str,
    strategies: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Simulate all three strategies on the same set of breakout events.

    Parameters
    ----------
    df : pd.DataFrame
        Full OHLC DataFrame.
    events : List[ConsolidationEvent]
        Accepted consolidation-breakout events with ADX labels.
    pair : str
        FX pair name.
    strategies : List[str], optional
        List of strategy names to simulate. Defaults to all three.

    Returns
    -------
    pd.DataFrame
        Per-trade results for all strategies.
    """
    if strategies is None:
        strategies = STRATEGY_NAMES

    records = []
    for event in events:
        for strategy in strategies:
            record = simulate_trade(df, event, strategy, pair)
            records.append(record)

    if not records:
        return pd.DataFrame()

    trades_df = pd.DataFrame(records)
    return trades_df


def compute_strategy_metrics(trades_df: pd.DataFrame, strategy: str) -> Dict:
    """
    Compute performance metrics for a single strategy.

    Metrics match the paper's reported evaluation:
    - profitable_trades: count of wins
    - non_profitable_trades: count of losses
    - average_gain: mean pnl_pips for wins
    - average_loss: mean pnl_pips for losses (negative value)
    - appt: Average Profit Per Trade = total_profit / total_trades
    - total_profit: sum of pnl_pips for all closed trades

    Parameters
    ----------
    trades_df : pd.DataFrame
        Per-trade results DataFrame.
    strategy : str
        Strategy name to filter on.

    Returns
    -------
    Dict
        Performance metrics dictionary.
    """
    strat_df = trades_df[
        (trades_df["strategy_name"] == strategy) &
        (trades_df["outcome"].isin(["win", "loss"]))
    ].copy()

    if strat_df.empty:
        return {
            "strategy": strategy,
            "profitable_trades": 0,
            "non_profitable_trades": 0,
            "total_trades": 0,
            "average_gain": float("nan"),
            "average_loss": float("nan"),
            "appt": float("nan"),
            "total_profit": 0.0,
        }

    wins = strat_df[strat_df["outcome"] == "win"]
    losses = strat_df[strat_df["outcome"] == "loss"]

    profitable = len(wins)
    non_profitable = len(losses)
    total_trades = profitable + non_profitable
    total_profit = float(strat_df["pnl_pips"].sum())
    avg_gain = float(wins["pnl_pips"].mean()) if len(wins) > 0 else float("nan")
    avg_loss = float(losses["pnl_pips"].mean()) if len(losses) > 0 else float("nan")
    appt = total_profit / total_trades if total_trades > 0 else float("nan")

    return {
        "strategy": strategy,
        "profitable_trades": profitable,
        "non_profitable_trades": non_profitable,
        "total_trades": total_trades,
        "average_gain": avg_gain,
        "average_loss": avg_loss,
        "appt": appt,
        "total_profit": total_profit,
    }


def compute_all_metrics(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute performance metrics for all strategies.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Per-trade results for all strategies.

    Returns
    -------
    pd.DataFrame
        Summary metrics table with one row per strategy.
    """
    metrics = []
    for strategy in STRATEGY_NAMES:
        if strategy in trades_df["strategy_name"].values:
            metrics.append(compute_strategy_metrics(trades_df, strategy))

    return pd.DataFrame(metrics)


def cumulative_pips(trades_df: pd.DataFrame, strategy: str) -> pd.Series:
    """
    Compute cumulative pips over trades for a given strategy.

    Parameters
    ----------
    trades_df : pd.DataFrame
        Per-trade results DataFrame.
    strategy : str
        Strategy name.

    Returns
    -------
    pd.Series
        Cumulative pips indexed by trade number (1-based).
    """
    strat_df = trades_df[
        (trades_df["strategy_name"] == strategy) &
        (trades_df["outcome"].isin(["win", "loss"]))
    ].copy()

    if strat_df.empty:
        return pd.Series(dtype=float)

    cum = strat_df["pnl_pips"].cumsum().reset_index(drop=True)
    cum.index = cum.index + 1  # 1-based trade number
    return cum
