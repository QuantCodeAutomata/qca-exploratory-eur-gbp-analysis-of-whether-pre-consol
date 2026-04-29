"""
Technical indicators module: ATR and ADX computation.

Ambiguity resolution:
- The paper prints nonstandard ADX formulas (DM_p, DM_n, DX using DM terms not DI terms,
  and a recursive smoothing formula that differs from Wilder's standard notation).
- For operational replication we use the `ta` library's ADXIndicator which implements
  Wilder-style 14-period ADX. This choice is logged as an ambiguity resolution.
- The paper's threshold of 25 for ADX implicitly assumes standard ADX semantics,
  consistent with this implementation.

# Using ta.trend.ADXIndicator — Context7 confirmed
# Using ta.volatility.AverageTrueRange — Context7 confirmed
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd
from ta.trend import ADXIndicator  # type: ignore
from ta.volatility import AverageTrueRange  # type: ignore

logger = logging.getLogger(__name__)

ADX_WINDOW = 14  # Paper specifies 14-period ADX/ATR


def compute_atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = ADX_WINDOW,
) -> pd.Series:
    """
    Compute 14-period Average True Range (ATR) using Wilder smoothing.

    # Using ta.volatility.AverageTrueRange — Context7 confirmed

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    window : int
        ATR period, default 14.

    Returns
    -------
    pd.Series
        ATR values aligned with input index.
    """
    atr_indicator = AverageTrueRange(high=high, low=low, close=close, window=window, fillna=False)
    return atr_indicator.average_true_range()


def compute_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    window: int = ADX_WINDOW,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """
    Compute 14-period ADX, +DI, and -DI using Wilder-style smoothing.

    Ambiguity note: The paper's printed formulas differ from standard Wilder notation
    (DX uses DM terms instead of DI terms, smoothing formula is nonstandard).
    We use the `ta` library's ADXIndicator as the best paper-consistent implementation.
    This choice is logged as an ambiguity resolution.

    # Using ta.trend.ADXIndicator — Context7 confirmed

    Parameters
    ----------
    high : pd.Series
        High prices.
    low : pd.Series
        Low prices.
    close : pd.Series
        Close prices.
    window : int
        ADX period, default 14.

    Returns
    -------
    Tuple[pd.Series, pd.Series, pd.Series]
        (adx, adx_pos, adx_neg) — ADX, +DI, -DI series aligned with input index.
    """
    adx_indicator = ADXIndicator(high=high, low=low, close=close, window=window, fillna=False)
    adx = adx_indicator.adx()
    adx_pos = adx_indicator.adx_pos()
    adx_neg = adx_indicator.adx_neg()
    return adx, adx_pos, adx_neg


def add_indicators(df: pd.DataFrame, window: int = ADX_WINDOW) -> pd.DataFrame:
    """
    Add ATR and ADX columns to an OHLC DataFrame in-place.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with columns: open, high, low, close.
    window : int
        Indicator period, default 14.

    Returns
    -------
    pd.DataFrame
        DataFrame with added columns: atr, adx, adx_pos, adx_neg.
    """
    assert {"open", "high", "low", "close"}.issubset(df.columns), (
        "DataFrame must contain open, high, low, close columns."
    )

    df = df.copy()
    df["atr"] = compute_atr(df["high"], df["low"], df["close"], window=window)
    adx, adx_pos, adx_neg = compute_adx(df["high"], df["low"], df["close"], window=window)
    df["adx"] = adx
    df["adx_pos"] = adx_pos
    df["adx_neg"] = adx_neg

    n_valid = df["adx"].notna().sum()
    logger.info(
        "Indicators computed: %d valid ADX values out of %d bars (window=%d).",
        n_valid,
        len(df),
        window,
    )
    return df


def rolling_mean_adx(
    adx_series: pd.Series,
    end_idx: int,
    window_minutes: int,
) -> float:
    """
    Compute the rolling mean ADX over `window_minutes` bars ending at (but not including)
    `end_idx`. This is used to label each breakout with its pre-consolidation ADX.

    Parameters
    ----------
    adx_series : pd.Series
        Full ADX series (integer-indexed).
    end_idx : int
        The index of the first bar of the consolidation (exclusive upper bound).
    window_minutes : int
        Number of bars to average over, immediately preceding the consolidation start.

    Returns
    -------
    float
        Mean ADX value, or NaN if insufficient data.

    Notes
    -----
    - Only data before the consolidation start is used (causal, no look-ahead).
    - If window_minutes < 1, returns NaN and logs a warning.
    """
    if window_minutes < 1:
        logger.warning("ADX averaging window < 1 minute; returning NaN.")
        return float("nan")

    start_idx = end_idx - window_minutes
    if start_idx < 0:
        logger.warning(
            "Insufficient warm-up for ADX label at index %d (window=%d); returning NaN.",
            end_idx,
            window_minutes,
        )
        return float("nan")

    window_data = adx_series.iloc[start_idx:end_idx]
    valid = window_data.dropna()
    if len(valid) == 0:
        logger.warning("All ADX values in window are NaN at index %d.", end_idx)
        return float("nan")

    return float(valid.mean())


def adx_label_window(consolidation_length_minutes: int) -> int:
    """
    Compute the ADX averaging window as 10% of the consolidation length.

    Ambiguity resolution: The paper gives no rounding rule. We use floor() as the
    deterministic integer conversion rule, fixed throughout the experiment.

    Parameters
    ----------
    consolidation_length_minutes : int
        Duration of the consolidation in minutes.

    Returns
    -------
    int
        ADX averaging window in minutes (floor of 10% of consolidation length).
    """
    # Ambiguity resolution: floor rounding, documented and fixed
    return max(1, int(np.floor(0.1 * consolidation_length_minutes)))
