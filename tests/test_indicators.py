"""
Tests for the indicators module.

Validates:
- ATR and ADX computation correctness
- Rolling mean ADX labeling (causal, no look-ahead)
- ADX label window calculation (floor rounding)
- Edge cases: empty data, single bar, extreme values
"""

import numpy as np
import pandas as pd
import pytest

from src.indicators import (
    add_indicators,
    adx_label_window,
    compute_adx,
    compute_atr,
    rolling_mean_adx,
)


def _make_ohlc(n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLC data for testing."""
    rng = np.random.default_rng(seed)
    close = 1.2 + np.cumsum(rng.normal(0, 0.0001, n))
    high = close + rng.uniform(0.0001, 0.0005, n)
    low = close - rng.uniform(0.0001, 0.0005, n)
    open_ = close + rng.normal(0, 0.0001, n)
    timestamps = pd.date_range("2017-10-01", periods=n, freq="1min", tz="UTC")
    return pd.DataFrame(
        {"timestamp": timestamps, "open": open_, "high": high, "low": low, "close": close}
    )


class TestComputeATR:
    def test_returns_series_same_length(self):
        df = _make_ohlc(100)
        atr = compute_atr(df["high"], df["low"], df["close"])
        assert len(atr) == 100

    def test_atr_positive_values_after_warmup(self):
        """ATR values after warm-up period should be positive (ta library fills 0 during warm-up)."""
        df = _make_ohlc(100)
        atr = compute_atr(df["high"], df["low"], df["close"], window=14)
        # ta library fills 0 during warm-up; check values after warm-up are positive
        valid = atr.iloc[14:].dropna()
        valid_nonzero = valid[valid > 0]
        assert len(valid_nonzero) > 0, "ATR should have positive values after warm-up."

    def test_atr_warmup_period_is_zero_or_nan(self):
        """ta library fills 0 (not NaN) during warm-up period with fillna=False."""
        df = _make_ohlc(100)
        atr = compute_atr(df["high"], df["low"], df["close"], window=14)
        # Early values should be 0 or NaN (ta library behavior)
        early = atr.iloc[:13]
        assert (early.isna() | (early == 0)).all(), (
            "ATR warm-up values should be 0 or NaN."
        )

    def test_atr_single_bar_raises(self):
        """ta library raises IndexError for fewer bars than window size."""
        df = _make_ohlc(1)
        with pytest.raises(Exception):
            compute_atr(df["high"], df["low"], df["close"])

    def test_atr_empty_raises(self):
        empty = pd.Series(dtype=float)
        with pytest.raises(Exception):
            compute_atr(empty, empty, empty)


class TestComputeADX:
    def test_returns_three_series(self):
        df = _make_ohlc(100)
        adx, adx_pos, adx_neg = compute_adx(df["high"], df["low"], df["close"])
        assert len(adx) == 100
        assert len(adx_pos) == 100
        assert len(adx_neg) == 100

    def test_adx_range_0_to_100(self):
        df = _make_ohlc(200)
        adx, _, _ = compute_adx(df["high"], df["low"], df["close"])
        valid = adx.dropna()
        assert (valid >= 0).all(), "ADX must be >= 0."
        assert (valid <= 100).all(), "ADX must be <= 100."

    def test_adx_pos_neg_range(self):
        df = _make_ohlc(200)
        _, adx_pos, adx_neg = compute_adx(df["high"], df["low"], df["close"])
        valid_pos = adx_pos.dropna()
        valid_neg = adx_neg.dropna()
        assert (valid_pos >= 0).all()
        assert (valid_neg >= 0).all()

    def test_adx_warmup_period_is_zero_or_nan(self):
        """ta library fills 0 (not NaN) during warm-up period with fillna=False."""
        df = _make_ohlc(100)
        adx, _, _ = compute_adx(df["high"], df["low"], df["close"], window=14)
        # ta library fills 0 during warm-up; early values should be 0 or NaN
        early = adx.iloc[:14]
        assert (early.isna() | (early == 0)).all(), (
            "ADX warm-up values should be 0 or NaN."
        )

    def test_adx_sufficient_data(self):
        df = _make_ohlc(200)
        adx, _, _ = compute_adx(df["high"], df["low"], df["close"])
        valid = adx.dropna()
        assert len(valid) > 100, "Should have many valid ADX values with 200 bars."


class TestAddIndicators:
    def test_adds_required_columns(self):
        df = _make_ohlc(100)
        result = add_indicators(df)
        assert "atr" in result.columns
        assert "adx" in result.columns
        assert "adx_pos" in result.columns
        assert "adx_neg" in result.columns

    def test_does_not_modify_original(self):
        df = _make_ohlc(100)
        original_cols = set(df.columns)
        _ = add_indicators(df)
        assert set(df.columns) == original_cols, "Original DataFrame should not be modified."

    def test_missing_column_raises(self):
        df = _make_ohlc(100).drop(columns=["high"])
        with pytest.raises(AssertionError):
            add_indicators(df)


class TestADXLabelWindow:
    def test_floor_rounding(self):
        # 10% of 180 = 18.0 → 18
        assert adx_label_window(180) == 18
        # 10% of 185 = 18.5 → floor = 18
        assert adx_label_window(185) == 18
        # 10% of 190 = 19.0 → 19
        assert adx_label_window(190) == 19

    def test_minimum_window_is_1(self):
        assert adx_label_window(1) == 1
        assert adx_label_window(5) == 1

    def test_large_consolidation(self):
        # 10% of 1000 = 100
        assert adx_label_window(1000) == 100

    def test_returns_int(self):
        result = adx_label_window(200)
        assert isinstance(result, int)


class TestRollingMeanADX:
    def test_basic_computation(self):
        adx_series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        # Window of 3 ending at index 3 → mean of [10, 20, 30] = 20
        result = rolling_mean_adx(adx_series, end_idx=3, window_minutes=3)
        assert abs(result - 20.0) < 1e-9

    def test_causal_no_lookahead(self):
        """Verify that only data before end_idx is used."""
        adx_series = pd.Series([10.0, 20.0, 30.0, 999.0, 999.0])
        # Window of 2 ending at index 3 → mean of [20, 30] = 25
        result = rolling_mean_adx(adx_series, end_idx=3, window_minutes=2)
        assert abs(result - 25.0) < 1e-9, "Should not use data at or after end_idx."

    def test_insufficient_warmup_returns_nan(self):
        adx_series = pd.Series([10.0, 20.0, 30.0])
        # Window of 5 ending at index 2 → start_idx = -3 → NaN
        result = rolling_mean_adx(adx_series, end_idx=2, window_minutes=5)
        assert np.isnan(result)

    def test_window_zero_returns_nan(self):
        adx_series = pd.Series([10.0, 20.0, 30.0])
        result = rolling_mean_adx(adx_series, end_idx=3, window_minutes=0)
        assert np.isnan(result)

    def test_all_nan_window_returns_nan(self):
        adx_series = pd.Series([float("nan"), float("nan"), float("nan"), 30.0])
        result = rolling_mean_adx(adx_series, end_idx=3, window_minutes=3)
        assert np.isnan(result)

    def test_partial_nan_window(self):
        adx_series = pd.Series([float("nan"), 20.0, 30.0, 40.0])
        # Window of 3 ending at index 3 → [nan, 20, 30] → mean of valid = 25
        result = rolling_mean_adx(adx_series, end_idx=3, window_minutes=3)
        assert abs(result - 25.0) < 1e-9

    def test_single_bar_window(self):
        adx_series = pd.Series([10.0, 25.0, 30.0])
        result = rolling_mean_adx(adx_series, end_idx=2, window_minutes=1)
        assert abs(result - 25.0) < 1e-9

    def test_adx_value_in_range(self):
        """ADX values should be between 0 and 100."""
        rng = np.random.default_rng(0)
        adx_vals = rng.uniform(0, 100, 200)
        adx_series = pd.Series(adx_vals)
        result = rolling_mean_adx(adx_series, end_idx=100, window_minutes=20)
        assert 0 <= result <= 100
