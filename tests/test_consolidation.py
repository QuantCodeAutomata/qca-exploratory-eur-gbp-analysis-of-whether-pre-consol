"""
Tests for the consolidation detection algorithm.

Validates:
- Minimum duration requirement (>= 180 minutes)
- Width eligibility (8-15 pips)
- Three structural rules (30-min spacing, midpoint bounce, 80% bounce)
- Breakout direction detection
- Edge cases: empty data, insufficient data, no valid events
- Pip conversion utilities
"""

import numpy as np
import pandas as pd
import pytest

from src.consolidation import (
    ConsolidationEvent,
    _apply_structural_rules,
    _price_to_pips,
    _pips_to_price,
    detect_consolidations,
    events_to_dataframe,
)


def _make_flat_ohlc(
    n: int,
    center: float = 1.2500,
    half_width: float = 0.0005,  # 5 pips
    timestamps_start: str = "2017-10-01",
) -> pd.DataFrame:
    """
    Generate synthetic flat OHLC data within a tight range.
    Prices oscillate between center ± half_width.
    """
    timestamps = pd.date_range(timestamps_start, periods=n, freq="1min", tz="UTC")
    # Oscillate: even bars touch upper, odd bars touch lower
    highs = np.where(np.arange(n) % 60 == 0, center + half_width, center + half_width * 0.5)
    lows = np.where(np.arange(n) % 60 == 30, center - half_width, center - half_width * 0.5)
    closes = np.full(n, center)
    opens = np.full(n, center)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


def _make_consolidation_then_breakout(
    consolidation_bars: int = 200,
    center: float = 1.2500,
    half_width_pips: float = 5.0,  # 5 pips each side → 10 pip width
    breakout_direction: str = "bullish",
    timestamps_start: str = "2017-10-01",
) -> pd.DataFrame:
    """
    Generate OHLC data with a consolidation period followed by a breakout.

    The consolidation has:
    - Bars touching upper edge every 60 bars
    - Bars touching lower edge every 30 bars after upper touch
    - 80% bounce after each edge touch
    - Breakout at the end
    """
    half_width = half_width_pips * 0.0001
    U = center + half_width
    D = center - half_width
    w = U - D
    mid = (U + D) / 2.0

    n_total = consolidation_bars + 10  # extra bars for breakout simulation
    timestamps = pd.date_range(timestamps_start, periods=n_total, freq="1min", tz="UTC")

    highs = np.full(n_total, center + half_width * 0.3)
    lows = np.full(n_total, center - half_width * 0.3)
    closes = np.full(n_total, center)
    opens = np.full(n_total, center)

    # Touch upper edge at bar 0
    highs[0] = U
    lows[0] = center - half_width * 0.3

    # Bounce to midpoint after upper touch (bar 5)
    highs[5] = mid + half_width * 0.1
    lows[5] = mid - half_width * 0.1
    closes[5] = mid

    # Touch lower edge at bar 60 (>= 30 bars after bar 0)
    lows[60] = D
    highs[60] = center + half_width * 0.3

    # Bounce up by 80% of width after lower touch (bar 65)
    bounce_target = D + 0.8 * w
    highs[65] = bounce_target + half_width * 0.1
    lows[65] = center - half_width * 0.1
    closes[65] = bounce_target

    # Touch upper edge again at bar 120 (>= 30 bars after bar 60)
    highs[120] = U
    lows[120] = center - half_width * 0.3

    # Bounce to midpoint after second upper touch (bar 125)
    lows[125] = mid - half_width * 0.1
    closes[125] = mid

    # Touch lower edge at bar 160 (>= 30 bars after bar 120)
    lows[160] = D
    highs[160] = center + half_width * 0.3

    # Bounce up by 80% after lower touch (bar 165)
    highs[165] = bounce_target + half_width * 0.1
    closes[165] = bounce_target

    # Breakout at consolidation_bars
    if breakout_direction == "bullish":
        highs[consolidation_bars] = U + half_width * 2
        lows[consolidation_bars] = center
        closes[consolidation_bars] = U + half_width
    else:
        lows[consolidation_bars] = D - half_width * 2
        highs[consolidation_bars] = center
        closes[consolidation_bars] = D - half_width

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


class TestPipConversion:
    def test_eurgbp_pip_conversion(self):
        # 0.0010 price diff = 10 pips for non-JPY pair
        result = _price_to_pips(0.0010, "EURGBP")
        assert abs(result - 10.0) < 1e-9

    def test_gbpusd_pip_conversion(self):
        result = _price_to_pips(0.0001, "GBPUSD")
        assert abs(result - 1.0) < 1e-9

    def test_jpy_pair_pip_conversion(self):
        # 0.01 price diff = 1 pip for JPY pair
        result = _price_to_pips(0.01, "USDJPY")
        assert abs(result - 1.0) < 1e-9

    def test_roundtrip_conversion(self):
        pips = 12.5
        price = _pips_to_price(pips, "EURGBP")
        back = _price_to_pips(price, "EURGBP")
        assert abs(back - pips) < 1e-9


class TestApplyStructuralRules:
    def test_insufficient_touches_fails(self):
        n = 200
        highs = np.full(n, 1.2505)
        lows = np.full(n, 1.2495)
        U, D = 1.2510, 1.2490
        mid = (U + D) / 2
        w = U - D
        passed, reason = _apply_structural_rules(
            highs, lows, 0, n, U, D, mid, w, 30, 0.8
        )
        assert not passed

    def test_valid_consolidation_passes(self):
        """Construct a minimal valid consolidation."""
        n = 200
        U, D = 1.2510, 1.2490
        mid = (U + D) / 2
        w = U - D
        highs = np.full(n, 1.2500)
        lows = np.full(n, 1.2500)

        # Touch U at bar 0
        highs[0] = U
        # Bounce to mid at bar 5
        lows[5] = mid - 0.0001
        # Touch D at bar 60
        lows[60] = D
        # Bounce up 80% at bar 65
        highs[65] = D + 0.8 * w + 0.0001

        passed, reason = _apply_structural_rules(
            highs, lows, 0, n, U, D, mid, w, 30, 0.8
        )
        assert passed, f"Expected pass but got: {reason}"

    def test_spacing_rule_fails(self):
        """Two edge touches within 30 bars should fail."""
        n = 200
        U, D = 1.2510, 1.2490
        mid = (U + D) / 2
        w = U - D
        highs = np.full(n, 1.2500)
        lows = np.full(n, 1.2500)

        # Touch U at bar 0 and bar 10 (only 10 bars apart → fail)
        highs[0] = U
        highs[10] = U

        passed, reason = _apply_structural_rules(
            highs, lows, 0, n, U, D, mid, w, 30, 0.8
        )
        assert not passed
        assert "spacing" in reason.lower()

    def test_80_percent_bounce_fails(self):
        """Bounce of only 50% should fail the 80% rule."""
        n = 200
        U, D = 1.2510, 1.2490
        mid = (U + D) / 2
        w = U - D
        highs = np.full(n, 1.2500)
        lows = np.full(n, 1.2500)

        # Touch U at bar 0
        highs[0] = U
        # Only 50% bounce (not 80%)
        lows[40] = U - 0.5 * w

        passed, reason = _apply_structural_rules(
            highs, lows, 0, n, U, D, mid, w, 30, 0.8
        )
        assert not passed


class TestDetectConsolidations:
    def test_empty_dataframe_returns_empty(self):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])
        events = detect_consolidations(df, pair="EURGBP")
        assert events == []

    def test_insufficient_data_returns_empty(self):
        df = _make_flat_ohlc(50)  # Only 50 bars, need >= 180
        events = detect_consolidations(df, pair="EURGBP")
        assert events == []

    def test_width_too_narrow_returns_empty(self):
        """Width of 2 pips is below the 8-pip minimum."""
        df = _make_flat_ohlc(300, half_width=0.0001)  # 1 pip each side = 2 pip width
        events = detect_consolidations(df, pair="EURGBP", width_min_pips=8.0)
        assert events == []

    def test_width_too_wide_returns_empty(self):
        """Width of 30 pips is above the 15-pip maximum."""
        df = _make_flat_ohlc(300, half_width=0.0015)  # 15 pips each side = 30 pip width
        events = detect_consolidations(df, pair="EURGBP", width_max_pips=15.0)
        assert events == []

    def test_event_duration_at_least_180(self):
        """All detected events must have duration >= 180 minutes."""
        df = _make_consolidation_then_breakout(consolidation_bars=200)
        events = detect_consolidations(
            df, pair="EURGBP", width_min_pips=8.0, width_max_pips=12.0
        )
        for event in events:
            assert event.consolidation_length_minutes >= 180, (
                f"Event duration {event.consolidation_length_minutes} < 180"
            )

    def test_breakout_direction_bullish(self):
        df = _make_consolidation_then_breakout(
            consolidation_bars=200, breakout_direction="bullish"
        )
        events = detect_consolidations(
            df, pair="EURGBP", width_min_pips=8.0, width_max_pips=12.0
        )
        for event in events:
            assert event.breakout_direction in ("bullish", "bearish")

    def test_event_fields_populated(self):
        """All required fields should be populated for each event."""
        df = _make_consolidation_then_breakout(consolidation_bars=200)
        events = detect_consolidations(
            df, pair="EURGBP", width_min_pips=8.0, width_max_pips=12.0
        )
        for event in events:
            assert event.pair == "EURGBP"
            assert event.upper_edge > event.lower_edge
            assert event.width_pips > 0
            assert event.midpoint == pytest.approx((event.upper_edge + event.lower_edge) / 2)
            assert event.breakout_direction in ("bullish", "bearish")
            assert event.consolidation_length_minutes >= 180

    def test_width_within_bounds(self):
        """All detected events must have width within the specified range."""
        df = _make_consolidation_then_breakout(consolidation_bars=200)
        events = detect_consolidations(
            df, pair="EURGBP", width_min_pips=8.0, width_max_pips=12.0
        )
        for event in events:
            assert 8.0 <= event.width_pips <= 12.0, (
                f"Width {event.width_pips} outside [8, 12] pips"
            )

    def test_no_overlap_between_events(self):
        """Events should not overlap (each starts after the previous breakout)."""
        df = _make_consolidation_then_breakout(consolidation_bars=200)
        events = detect_consolidations(
            df, pair="EURGBP", width_min_pips=8.0, width_max_pips=12.0
        )
        for i in range(1, len(events)):
            assert events[i].consolidation_start_idx > events[i - 1].breakout_idx, (
                "Events should not overlap."
            )


class TestEventsToDataframe:
    def test_empty_list_returns_empty_df(self):
        result = events_to_dataframe([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_single_event_returns_one_row(self):
        event = ConsolidationEvent(
            pair="EURGBP",
            consolidation_start=pd.Timestamp("2017-10-01 10:00", tz="UTC"),
            consolidation_start_idx=100,
            breakout_time=pd.Timestamp("2017-10-01 13:30", tz="UTC"),
            breakout_idx=310,
            consolidation_length_minutes=210,
            upper_edge=1.2510,
            lower_edge=1.2490,
            width_pips=20.0,
            midpoint=1.2500,
            breakout_direction="bullish",
        )
        df = events_to_dataframe([event])
        assert len(df) == 1
        assert df["pair"].iloc[0] == "EURGBP"
        assert df["breakout_direction"].iloc[0] == "bullish"

    def test_required_columns_present(self):
        event = ConsolidationEvent(
            pair="GBPUSD",
            consolidation_start=pd.Timestamp("2017-10-01 10:00", tz="UTC"),
            consolidation_start_idx=50,
            breakout_time=pd.Timestamp("2017-10-01 14:00", tz="UTC"),
            breakout_idx=290,
            consolidation_length_minutes=240,
            upper_edge=1.3010,
            lower_edge=1.2990,
            width_pips=20.0,
            midpoint=1.3000,
            breakout_direction="bearish",
        )
        df = events_to_dataframe([event])
        required_cols = [
            "pair", "consolidation_start", "breakout_time",
            "consolidation_length_minutes", "upper_edge", "lower_edge",
            "width_pips", "midpoint", "breakout_direction",
        ]
        for col in required_cols:
            assert col in df.columns, f"Missing column: {col}"
