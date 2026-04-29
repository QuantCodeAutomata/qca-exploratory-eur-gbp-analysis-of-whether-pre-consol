"""
Custom consolidation-breakout detection algorithm.

Implements the paper's event-driven consolidation detection with three structural rules:
  1. 30-minute edge-touch spacing rule
  2. Repeated-edge bounce-to-midpoint rule
  3. 80% bounce rule

Ambiguity resolutions documented:
- Width rule: paper says 15 pips in flowchart, 8-15 pips in method section.
  Primary replication uses 8-15 pips. Inconsistency is logged.
- Minimum consolidation duration: 180 minutes (3 hours), as stated in the paper.
- After an accepted breakout, scan resumes from one bar after the breakout bar.
- After a rejected candidate, scan advances by one bar from the original start.
- 'Touch' of upper edge uses bar high; touch of lower edge uses bar low.
- 30-minute edge-touch rule: if another edge is touched before 30 minutes have elapsed
  since the previous high/low point, the candidate is rejected.
- Breakout direction: bullish if close > U, bearish if close < D.
  (First bar whose close exits the range determines direction.)

# Custom — Context7 found no library equivalent (paper custom algorithm)
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Paper's primary width range (method section, 8-15 pips)
# Ambiguity: flowchart says "within 15 pips", method section says 8-15 pips.
# Primary replication uses 8-15 pips.
DEFAULT_WIDTH_MIN_PIPS = 8.0
DEFAULT_WIDTH_MAX_PIPS = 15.0
MIN_CONSOLIDATION_MINUTES = 180  # 3 hours, as stated in the paper
EDGE_TOUCH_SPACING_MINUTES = 30  # 30-minute edge-touch rule
BOUNCE_FRACTION = 0.80  # 80% bounce rule


@dataclass
class ConsolidationEvent:
    """Represents a detected consolidation-breakout event."""

    pair: str
    consolidation_start: pd.Timestamp
    consolidation_start_idx: int
    breakout_time: pd.Timestamp
    breakout_idx: int
    consolidation_length_minutes: int
    upper_edge: float  # U in pips-equivalent price
    lower_edge: float  # D in pips-equivalent price
    width_pips: float  # w = U - D in pips
    midpoint: float  # m = (U + D) / 2
    breakout_direction: str  # 'bullish' or 'bearish'
    adx_label_window_minutes: int = 0
    adx_label_value: float = float("nan")


def _price_to_pips(price_diff: float, pair: str) -> float:
    """Convert raw price difference to pips."""
    pair_upper = pair.upper().replace("C:", "").replace("/", "")
    if "JPY" in pair_upper:
        return price_diff / 0.01
    return price_diff / 0.0001


def _pips_to_price(pips: float, pair: str) -> float:
    """Convert pips to raw price difference."""
    pair_upper = pair.upper().replace("C:", "").replace("/", "")
    if "JPY" in pair_upper:
        return pips * 0.01
    return pips * 0.0001


def detect_consolidations(
    df: pd.DataFrame,
    pair: str,
    width_min_pips: float = DEFAULT_WIDTH_MIN_PIPS,
    width_max_pips: float = DEFAULT_WIDTH_MAX_PIPS,
    min_duration_minutes: int = MIN_CONSOLIDATION_MINUTES,
    edge_touch_spacing: int = EDGE_TOUCH_SPACING_MINUTES,
    bounce_fraction: float = BOUNCE_FRACTION,
) -> List[ConsolidationEvent]:
    """
    Detect consolidation-breakout events using the paper's custom algorithm.

    Algorithm overview:
    1. For each bar as a potential consolidation start, define U = max high over next
       180 bars, D = min low over next 180 bars. Check width eligibility.
    2. Extend the candidate bar by bar while price stays within [D, U].
    3. When price exits after >= 180 minutes, apply three structural rules.
    4. If all rules pass, record the event. Advance scan to one bar after breakout.
    5. If rules fail, advance scan by one bar from the original start.

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with columns: timestamp, open, high, low, close.
        Must be sorted chronologically with integer index.
    pair : str
        FX pair name for pip conversion, e.g. 'EURGBP'.
    width_min_pips : float
        Minimum consolidation width in pips (default 8).
    width_max_pips : float
        Maximum consolidation width in pips (default 15).
    min_duration_minutes : int
        Minimum consolidation duration in minutes (default 180).
    edge_touch_spacing : int
        Minimum minutes between consecutive edge touches (default 30).
    bounce_fraction : float
        Required bounce fraction after first opposite-edge touch (default 0.80).

    Returns
    -------
    List[ConsolidationEvent]
        List of accepted consolidation-breakout events.

    Notes
    -----
    - Width ambiguity: flowchart says "within 15 pips", method section says 8-15 pips.
      Primary replication uses 8-15 pips (width_min_pips=8, width_max_pips=15).
    - After accepted breakout: scan resumes from breakout_idx + 1.
    - After rejected candidate: scan advances by 1 from original start.
    """
    assert "timestamp" in df.columns, "DataFrame must have 'timestamp' column."
    assert {"high", "low", "close"}.issubset(df.columns), (
        "DataFrame must have high, low, close columns."
    )

    n = len(df)
    highs = df["high"].values
    lows = df["low"].values
    closes = df["close"].values
    timestamps = df["timestamp"].values

    events: List[ConsolidationEvent] = []
    i = 0  # current scan start index

    while i < n - min_duration_minutes:
        # --- Step 1: Define initial consolidation range from first 180 bars ---
        init_end = i + min_duration_minutes
        if init_end > n:
            break

        U_init = float(np.max(highs[i:init_end]))
        D_init = float(np.min(lows[i:init_end]))
        w_init_pips = _price_to_pips(U_init - D_init, pair)

        # Check width eligibility
        if not (width_min_pips <= w_init_pips <= width_max_pips):
            i += 1
            continue

        U = U_init
        D = D_init

        # --- Step 2: Extend candidate bar by bar ---
        j = init_end  # j is the current extension index
        while j < n:
            h = highs[j]
            lo = lows[j]
            c = closes[j]

            # Check if price exits the range
            if h > U or lo < D:
                # Price has exited; j is the breakout bar
                break
            j += 1
        else:
            # Reached end of data without breakout
            i += 1
            continue

        # j is the breakout bar index
        breakout_idx = j
        consolidation_length = breakout_idx - i  # in minutes

        if consolidation_length < min_duration_minutes:
            i += 1
            continue

        # Determine breakout direction
        if highs[breakout_idx] > U and lows[breakout_idx] >= D:
            breakout_direction = "bullish"
        elif lows[breakout_idx] < D and highs[breakout_idx] <= U:
            breakout_direction = "bearish"
        elif highs[breakout_idx] > U and lows[breakout_idx] < D:
            # Both edges breached in same bar — tie-break: use close
            if closes[breakout_idx] > (U + D) / 2:
                breakout_direction = "bullish"
            else:
                breakout_direction = "bearish"
            logger.debug(
                "Both edges breached at bar %d; tie-break by close → %s",
                breakout_idx,
                breakout_direction,
            )
        else:
            i += 1
            continue

        # Recompute U and D over the full consolidation window
        U = float(np.max(highs[i:breakout_idx]))
        D = float(np.min(lows[i:breakout_idx]))
        w_pips = _price_to_pips(U - D, pair)

        if not (width_min_pips <= w_pips <= width_max_pips):
            i += 1
            continue

        midpoint = (U + D) / 2.0

        # --- Step 3: Apply three structural rules ---
        passed, reason = _apply_structural_rules(
            highs=highs,
            lows=lows,
            start_idx=i,
            end_idx=breakout_idx,
            U=U,
            D=D,
            midpoint=midpoint,
            w_price=U - D,
            edge_touch_spacing=edge_touch_spacing,
            bounce_fraction=bounce_fraction,
        )

        if not passed:
            logger.debug(
                "Candidate at %s rejected: %s",
                pd.Timestamp(timestamps[i]),
                reason,
            )
            i += 1
            continue

        # --- Step 4: Record accepted event ---
        event = ConsolidationEvent(
            pair=pair,
            consolidation_start=pd.Timestamp(timestamps[i]),
            consolidation_start_idx=i,
            breakout_time=pd.Timestamp(timestamps[breakout_idx]),
            breakout_idx=breakout_idx,
            consolidation_length_minutes=consolidation_length,
            upper_edge=U,
            lower_edge=D,
            width_pips=w_pips,
            midpoint=midpoint,
            breakout_direction=breakout_direction,
        )
        events.append(event)
        logger.debug(
            "Accepted consolidation: start=%s, breakout=%s, dir=%s, w=%.1f pips",
            event.consolidation_start,
            event.breakout_time,
            event.breakout_direction,
            event.width_pips,
        )

        # Advance scan to one bar after breakout
        i = breakout_idx + 1

    logger.info("Detected %d consolidation-breakout events for %s.", len(events), pair)
    return events


def _apply_structural_rules(
    highs: np.ndarray,
    lows: np.ndarray,
    start_idx: int,
    end_idx: int,
    U: float,
    D: float,
    midpoint: float,
    w_price: float,
    edge_touch_spacing: int,
    bounce_fraction: float,
) -> Tuple[bool, str]:
    """
    Apply the three structural consolidation rules to a candidate.

    Rules:
    a. 30-minute edge-touch spacing: consecutive edge touches must be >= 30 minutes apart.
       Upper edge uses highs; lower edge uses lows.
    b. Repeated-edge bounce-to-midpoint: if the same edge is touched twice, price must
       subsequently reach the midpoint.
    c. 80% bounce rule: after the first touch of the opposite edge, price must reverse
       by at least 80% of the consolidation width.

    Parameters
    ----------
    highs, lows : np.ndarray
        Price arrays.
    start_idx, end_idx : int
        Consolidation bar range [start_idx, end_idx).
    U, D : float
        Upper and lower consolidation edges.
    midpoint : float
        Midpoint of the consolidation.
    w_price : float
        Consolidation width in price units.
    edge_touch_spacing : int
        Minimum bars between consecutive edge touches.
    bounce_fraction : float
        Required bounce fraction after first opposite-edge touch.

    Returns
    -------
    Tuple[bool, str]
        (passed, reason) — True if all rules pass, False with reason string otherwise.
    """
    # Tolerance for edge touch (floating point)
    tol = w_price * 1e-6

    # Track edge touches: list of (bar_index, edge_type) where edge_type is 'U' or 'D'
    touches: List[Tuple[int, str]] = []

    for k in range(start_idx, end_idx):
        h = highs[k]
        lo = lows[k]

        touched_U = h >= U - tol
        touched_D = lo <= D + tol

        if touched_U:
            touches.append((k, "U"))
        if touched_D:
            touches.append((k, "D"))

    if len(touches) < 2:
        # Not enough touches to form a valid consolidation
        return False, "fewer than 2 edge touches"

    # --- Rule a: 30-minute edge-touch spacing ---
    for idx_t in range(1, len(touches)):
        prev_bar, prev_edge = touches[idx_t - 1]
        curr_bar, curr_edge = touches[idx_t]
        spacing = curr_bar - prev_bar
        if spacing < edge_touch_spacing:
            return False, (
                f"edge touch spacing {spacing} < {edge_touch_spacing} "
                f"at bars {prev_bar},{curr_bar}"
            )

    # --- Rule b: Repeated-edge bounce-to-midpoint ---
    # Track last touch of each edge and whether midpoint was reached after
    last_U_touch: Optional[int] = None
    last_D_touch: Optional[int] = None
    U_touch_count = 0
    D_touch_count = 0

    for k in range(start_idx, end_idx):
        h = highs[k]
        lo = lows[k]
        c_mid = (h + lo) / 2.0  # use bar midpoint as proxy for price reaching midpoint

        # Check if midpoint was reached (use low for upward bounce, high for downward bounce)
        if last_U_touch is not None and U_touch_count >= 2:
            # After second U touch, price must reach midpoint (going down)
            if lo <= midpoint + tol:
                last_U_touch = None  # satisfied
                U_touch_count = 0

        if last_D_touch is not None and D_touch_count >= 2:
            # After second D touch, price must reach midpoint (going up)
            if h >= midpoint - tol:
                last_D_touch = None  # satisfied
                D_touch_count = 0

        if h >= U - tol:
            U_touch_count += 1
            last_U_touch = k

        if lo <= D + tol:
            D_touch_count += 1
            last_D_touch = k

    # After scanning, check if any unsatisfied repeated-edge bounce requirement remains
    if last_U_touch is not None and U_touch_count >= 2:
        return False, f"U edge touched {U_touch_count}x but midpoint not reached after"
    if last_D_touch is not None and D_touch_count >= 2:
        return False, f"D edge touched {D_touch_count}x but midpoint not reached after"

    # --- Rule c: 80% bounce rule ---
    # Find first touch of each edge, then check bounce
    first_U_touch_idx: Optional[int] = None
    first_D_touch_idx: Optional[int] = None

    for k in range(start_idx, end_idx):
        if first_U_touch_idx is None and highs[k] >= U - tol:
            first_U_touch_idx = k
        if first_D_touch_idx is None and lows[k] <= D + tol:
            first_D_touch_idx = k

    required_bounce = bounce_fraction * w_price

    # After first U touch, price must bounce down by at least 80% of w
    if first_U_touch_idx is not None:
        bounce_target = U - required_bounce
        bounced = False
        for k in range(first_U_touch_idx + 1, end_idx):
            if lows[k] <= bounce_target + tol:
                bounced = True
                break
        if not bounced:
            return False, (
                f"80% bounce rule failed after first U touch at bar {first_U_touch_idx}"
            )

    # After first D touch, price must bounce up by at least 80% of w
    if first_D_touch_idx is not None:
        bounce_target = D + required_bounce
        bounced = False
        for k in range(first_D_touch_idx + 1, end_idx):
            if highs[k] >= bounce_target - tol:
                bounced = True
                break
        if not bounced:
            return False, (
                f"80% bounce rule failed after first D touch at bar {first_D_touch_idx}"
            )

    return True, "all rules passed"


def events_to_dataframe(events: List[ConsolidationEvent]) -> pd.DataFrame:
    """
    Convert a list of ConsolidationEvent objects to a pandas DataFrame.

    Parameters
    ----------
    events : List[ConsolidationEvent]
        List of detected consolidation-breakout events.

    Returns
    -------
    pd.DataFrame
        DataFrame with one row per event.
    """
    if not events:
        return pd.DataFrame()

    records = []
    for e in events:
        records.append(
            {
                "pair": e.pair,
                "consolidation_start": e.consolidation_start,
                "consolidation_start_idx": e.consolidation_start_idx,
                "breakout_time": e.breakout_time,
                "breakout_idx": e.breakout_idx,
                "consolidation_length_minutes": e.consolidation_length_minutes,
                "upper_edge": e.upper_edge,
                "lower_edge": e.lower_edge,
                "width_pips": e.width_pips,
                "midpoint": e.midpoint,
                "breakout_direction": e.breakout_direction,
                "adx_label_window_minutes": e.adx_label_window_minutes,
                "adx_label_value": e.adx_label_value,
            }
        )
    return pd.DataFrame(records)
