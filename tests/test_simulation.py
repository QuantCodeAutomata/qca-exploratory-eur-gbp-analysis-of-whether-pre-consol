"""
Tests for the trade simulation module.

Validates:
- Target multiplier computation for all three strategies
- Trade simulation correctness (win/loss/open outcomes)
- Intrabar tie-break rule (stop wins)
- Performance metrics computation (APPT, total profit, win/loss counts)
- Cumulative pips computation
- Edge cases: NaN ADX, empty events, open trades at sample end
"""

import numpy as np
import pandas as pd
import pytest

from src.consolidation import ConsolidationEvent
from src.simulation import (
    ADX_STRONG_THRESHOLD,
    STRATEGY_NAMES,
    _target_multiplier,
    compute_all_metrics,
    compute_strategy_metrics,
    cumulative_pips,
    simulate_all_strategies,
    simulate_trade,
)


def _make_event(
    adx_value: float = 30.0,
    direction: str = "bullish",
    upper_edge: float = 1.2510,
    lower_edge: float = 1.2490,
    breakout_idx: int = 200,
    consolidation_start_idx: int = 0,
    consolidation_length: int = 200,
) -> ConsolidationEvent:
    """Create a synthetic ConsolidationEvent for testing."""
    return ConsolidationEvent(
        pair="EURGBP",
        consolidation_start=pd.Timestamp("2017-10-01 10:00", tz="UTC"),
        consolidation_start_idx=consolidation_start_idx,
        breakout_time=pd.Timestamp("2017-10-01 13:20", tz="UTC"),
        breakout_idx=breakout_idx,
        consolidation_length_minutes=consolidation_length,
        upper_edge=upper_edge,
        lower_edge=lower_edge,
        width_pips=20.0,
        midpoint=(upper_edge + lower_edge) / 2,
        breakout_direction=direction,
        adx_label_window_minutes=20,
        adx_label_value=adx_value,
    )


def _make_ohlc_for_trade(
    n: int = 300,
    breakout_idx: int = 200,
    direction: str = "bullish",
    upper_edge: float = 1.2510,
    lower_edge: float = 1.2490,
    target_hit_bar: int = 210,
    stop_hit_bar: int = None,
) -> pd.DataFrame:
    """
    Generate OHLC data where a trade is entered at breakout_idx.
    Target is hit at target_hit_bar (if provided), stop at stop_hit_bar.
    """
    w = upper_edge - lower_edge
    timestamps = pd.date_range("2017-10-01", periods=n, freq="1min", tz="UTC")
    highs = np.full(n, (upper_edge + lower_edge) / 2 + w * 0.1)
    lows = np.full(n, (upper_edge + lower_edge) / 2 - w * 0.1)
    closes = np.full(n, (upper_edge + lower_edge) / 2)
    opens = closes.copy()

    # Breakout bar
    if direction == "bullish":
        closes[breakout_idx] = upper_edge + w * 0.1
        highs[breakout_idx] = upper_edge + w * 0.2
        lows[breakout_idx] = upper_edge
    else:
        closes[breakout_idx] = lower_edge - w * 0.1
        lows[breakout_idx] = lower_edge - w * 0.2
        highs[breakout_idx] = lower_edge

    # Target hit bar
    if target_hit_bar is not None:
        entry = closes[breakout_idx]
        if direction == "bullish":
            target = entry + 1.0 * w  # Standard target
            highs[target_hit_bar] = target + w * 0.1
        else:
            target = entry - 1.0 * w
            lows[target_hit_bar] = target - w * 0.1

    # Stop hit bar
    if stop_hit_bar is not None:
        if direction == "bullish":
            lows[stop_hit_bar] = lower_edge - w * 0.1
        else:
            highs[stop_hit_bar] = upper_edge + w * 0.1

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
        }
    )


class TestTargetMultiplier:
    def test_standard_always_1x(self):
        for adx in [0, 10, 25, 50, 100]:
            assert _target_multiplier("Standard", adx) == 1.0

    def test_discrete_strong_adx(self):
        # ADX >= 25 → 1.5x
        assert _target_multiplier("Discrete", 25.0) == 1.5
        assert _target_multiplier("Discrete", 30.0) == 1.5
        assert _target_multiplier("Discrete", 100.0) == 1.5

    def test_discrete_weak_adx(self):
        # ADX < 25 → 0.5x
        assert _target_multiplier("Discrete", 24.9) == 0.5
        assert _target_multiplier("Discrete", 0.0) == 0.5

    def test_linear_proportional(self):
        # target = ADX / 25
        assert _target_multiplier("Linear", 25.0) == pytest.approx(1.0)
        assert _target_multiplier("Linear", 50.0) == pytest.approx(2.0)
        assert _target_multiplier("Linear", 12.5) == pytest.approx(0.5)

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError):
            _target_multiplier("Unknown", 30.0)


class TestSimulateTrade:
    def test_bullish_win(self):
        event = _make_event(adx_value=30.0, direction="bullish")
        df = _make_ohlc_for_trade(
            direction="bullish",
            target_hit_bar=210,
            stop_hit_bar=None,
        )
        result = simulate_trade(df, event, "Standard", "EURGBP")
        assert result["outcome"] == "win"
        assert result["pnl_pips"] > 0

    def test_bullish_loss(self):
        event = _make_event(adx_value=30.0, direction="bullish")
        df = _make_ohlc_for_trade(
            direction="bullish",
            target_hit_bar=None,
            stop_hit_bar=205,
        )
        # Make stop hit before target
        w = event.upper_edge - event.lower_edge
        entry = df["close"].iloc[event.breakout_idx]
        # Ensure stop is hit: low goes below lower_edge
        df.loc[205, "low"] = event.lower_edge - w * 0.1
        result = simulate_trade(df, event, "Standard", "EURGBP")
        assert result["outcome"] == "loss"
        assert result["pnl_pips"] < 0

    def test_nan_adx_returns_skipped(self):
        event = _make_event(adx_value=float("nan"))
        df = _make_ohlc_for_trade()
        result = simulate_trade(df, event, "Standard", "EURGBP")
        assert result["outcome"] == "skipped_no_adx"

    def test_intrabar_tiebreak_stop_wins(self):
        """When both stop and target are hit in the same bar, stop should win."""
        event = _make_event(adx_value=30.0, direction="bullish")
        df = _make_ohlc_for_trade(direction="bullish")
        w = event.upper_edge - event.lower_edge
        entry = df["close"].iloc[event.breakout_idx]
        target = entry + 1.0 * w
        stop = event.lower_edge

        # Bar 205: both target and stop hit
        df.loc[205, "high"] = target + w * 0.1
        df.loc[205, "low"] = stop - w * 0.1

        result = simulate_trade(df, event, "Standard", "EURGBP")
        assert result["outcome"] == "loss", "Tie-break: stop should win."

    def test_open_trade_at_sample_end(self):
        """Trade that never hits target or stop should be marked 'open'."""
        event = _make_event(adx_value=30.0, direction="bullish")
        # Create data where price never reaches target or stop
        n = 250
        timestamps = pd.date_range("2017-10-01", periods=n, freq="1min", tz="UTC")
        mid = (event.upper_edge + event.lower_edge) / 2
        w = event.upper_edge - event.lower_edge
        entry = event.upper_edge + w * 0.1
        # Price stays between entry and target, never hitting either
        highs = np.full(n, entry + w * 0.3)
        lows = np.full(n, entry + w * 0.05)
        closes = np.full(n, entry + w * 0.1)
        opens = closes.copy()
        highs[200] = entry + w * 0.2  # breakout bar
        lows[200] = event.upper_edge
        closes[200] = entry

        df = pd.DataFrame(
            {"timestamp": timestamps, "open": opens, "high": highs, "low": lows, "close": closes}
        )
        result = simulate_trade(df, event, "Standard", "EURGBP")
        assert result["outcome"] == "open"
        assert result["pnl_pips"] is None

    def test_pnl_pips_positive_for_win(self):
        event = _make_event(adx_value=30.0, direction="bullish")
        df = _make_ohlc_for_trade(direction="bullish", target_hit_bar=210)
        result = simulate_trade(df, event, "Standard", "EURGBP")
        if result["outcome"] == "win":
            assert result["pnl_pips"] > 0

    def test_pnl_pips_negative_for_loss(self):
        event = _make_event(adx_value=30.0, direction="bullish")
        df = _make_ohlc_for_trade(direction="bullish", target_hit_bar=None)
        w = event.upper_edge - event.lower_edge
        # Force stop hit
        df.loc[205, "low"] = event.lower_edge - w * 0.1
        result = simulate_trade(df, event, "Standard", "EURGBP")
        if result["outcome"] == "loss":
            assert result["pnl_pips"] < 0


class TestComputeStrategyMetrics:
    def _make_trades_df(self, wins: int, losses: int, strategy: str = "Standard") -> pd.DataFrame:
        records = []
        for _ in range(wins):
            records.append({"strategy_name": strategy, "outcome": "win", "pnl_pips": 14.0})
        for _ in range(losses):
            records.append({"strategy_name": strategy, "outcome": "loss", "pnl_pips": -14.0})
        return pd.DataFrame(records)

    def test_basic_metrics(self):
        df = self._make_trades_df(wins=10, losses=5)
        metrics = compute_strategy_metrics(df, "Standard")
        assert metrics["profitable_trades"] == 10
        assert metrics["non_profitable_trades"] == 5
        assert metrics["total_trades"] == 15
        assert abs(metrics["total_profit"] - (10 * 14.0 + 5 * (-14.0))) < 1e-9
        assert abs(metrics["appt"] - metrics["total_profit"] / 15) < 1e-9

    def test_appt_equals_total_profit_over_total_trades(self):
        df = self._make_trades_df(wins=159, losses=168)
        metrics = compute_strategy_metrics(df, "Standard")
        expected_appt = metrics["total_profit"] / metrics["total_trades"]
        assert abs(metrics["appt"] - expected_appt) < 1e-9

    def test_empty_strategy_returns_zeros(self):
        df = pd.DataFrame(columns=["strategy_name", "outcome", "pnl_pips"])
        metrics = compute_strategy_metrics(df, "Standard")
        assert metrics["total_trades"] == 0
        assert np.isnan(metrics["appt"])

    def test_all_wins(self):
        df = self._make_trades_df(wins=10, losses=0)
        metrics = compute_strategy_metrics(df, "Standard")
        assert metrics["profitable_trades"] == 10
        assert metrics["non_profitable_trades"] == 0
        assert metrics["total_profit"] > 0

    def test_all_losses(self):
        df = self._make_trades_df(wins=0, losses=10)
        metrics = compute_strategy_metrics(df, "Standard")
        assert metrics["profitable_trades"] == 0
        assert metrics["non_profitable_trades"] == 10
        assert metrics["total_profit"] < 0


class TestComputeAllMetrics:
    def test_returns_one_row_per_strategy(self):
        records = []
        for strategy in STRATEGY_NAMES:
            for _ in range(5):
                records.append({"strategy_name": strategy, "outcome": "win", "pnl_pips": 10.0})
            for _ in range(5):
                records.append({"strategy_name": strategy, "outcome": "loss", "pnl_pips": -10.0})
        df = pd.DataFrame(records)
        metrics = compute_all_metrics(df)
        assert len(metrics) == len(STRATEGY_NAMES)

    def test_strategy_names_in_output(self):
        records = []
        for strategy in STRATEGY_NAMES:
            records.append({"strategy_name": strategy, "outcome": "win", "pnl_pips": 10.0})
        df = pd.DataFrame(records)
        metrics = compute_all_metrics(df)
        assert set(metrics["strategy"].values) == set(STRATEGY_NAMES)


class TestCumulativePips:
    def test_cumulative_increases_with_wins(self):
        records = [
            {"strategy_name": "Standard", "outcome": "win", "pnl_pips": 10.0},
            {"strategy_name": "Standard", "outcome": "win", "pnl_pips": 10.0},
            {"strategy_name": "Standard", "outcome": "win", "pnl_pips": 10.0},
        ]
        df = pd.DataFrame(records)
        cum = cumulative_pips(df, "Standard")
        assert list(cum.values) == [10.0, 20.0, 30.0]

    def test_cumulative_decreases_with_losses(self):
        records = [
            {"strategy_name": "Standard", "outcome": "loss", "pnl_pips": -10.0},
            {"strategy_name": "Standard", "outcome": "loss", "pnl_pips": -10.0},
        ]
        df = pd.DataFrame(records)
        cum = cumulative_pips(df, "Standard")
        assert list(cum.values) == [-10.0, -20.0]

    def test_empty_strategy_returns_empty(self):
        df = pd.DataFrame(columns=["strategy_name", "outcome", "pnl_pips"])
        cum = cumulative_pips(df, "Standard")
        assert cum.empty

    def test_index_is_1_based(self):
        records = [
            {"strategy_name": "Standard", "outcome": "win", "pnl_pips": 5.0},
            {"strategy_name": "Standard", "outcome": "win", "pnl_pips": 5.0},
        ]
        df = pd.DataFrame(records)
        cum = cumulative_pips(df, "Standard")
        assert cum.index[0] == 1
        assert cum.index[-1] == 2

    def test_open_trades_excluded(self):
        """Open trades should not appear in cumulative pips."""
        records = [
            {"strategy_name": "Standard", "outcome": "win", "pnl_pips": 10.0},
            {"strategy_name": "Standard", "outcome": "open", "pnl_pips": None},
            {"strategy_name": "Standard", "outcome": "win", "pnl_pips": 10.0},
        ]
        df = pd.DataFrame(records)
        cum = cumulative_pips(df, "Standard")
        assert len(cum) == 2
        assert cum.iloc[-1] == 20.0


class TestSimulateAllStrategies:
    def test_same_event_set_for_all_strategies(self):
        """All three strategies must be applied to the same event set."""
        event = _make_event(adx_value=30.0, direction="bullish")
        df = _make_ohlc_for_trade(direction="bullish", target_hit_bar=210)
        trades = simulate_all_strategies(df, [event], "EURGBP")

        for strategy in STRATEGY_NAMES:
            strategy_trades = trades[trades["strategy_name"] == strategy]
            assert len(strategy_trades) == 1, (
                f"Strategy {strategy} should have exactly 1 trade."
            )

    def test_empty_events_returns_empty(self):
        df = _make_ohlc_for_trade()
        trades = simulate_all_strategies(df, [], "EURGBP")
        assert trades.empty

    def test_output_has_required_columns(self):
        event = _make_event(adx_value=30.0)
        df = _make_ohlc_for_trade()
        trades = simulate_all_strategies(df, [event], "EURGBP")
        required_cols = [
            "pair", "consolidation_start", "breakout_time", "direction",
            "width_pips", "duration_minutes", "adx_label_window_minutes",
            "adx_label_value", "strategy_name", "entry_price", "stop_price",
            "target_price", "exit_time", "exit_reason", "outcome", "pnl_pips",
        ]
        for col in required_cols:
            assert col in trades.columns, f"Missing column: {col}"
