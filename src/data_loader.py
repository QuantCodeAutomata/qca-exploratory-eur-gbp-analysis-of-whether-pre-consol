"""
Data loading module for FX 1-minute OHLC data via the Massive REST API.

Ambiguity resolutions documented here:
- No vendor, quote convention, or bid/ask/mid specification is given in the paper.
  We use the Massive API with default quote type (mid). This is an unresolved assumption.
- Weekend gaps and missing minute bars are NOT filled; their presence is logged.
- Sufficient warm-up history (at least 14 bars before the evaluation window) is ensured
  by fetching from a date slightly before the stated start.
"""

import os
import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


def fetch_ohlc(
    ticker: str,
    from_date: str,
    to_date: str,
    timespan: str = "minute",
    multiplier: int = 1,
    limit: int = 50000,
) -> pd.DataFrame:
    """
    Fetch 1-minute OHLC data from the Massive REST API.

    Parameters
    ----------
    ticker : str
        FX pair ticker, e.g. 'C:EURGBP' or 'C:GBPUSD'.
    from_date : str
        Start date in 'YYYY-MM-DD' format.
    to_date : str
        End date in 'YYYY-MM-DD' format.
    timespan : str
        Bar timespan, default 'minute'.
    multiplier : int
        Bar multiplier, default 1.
    limit : int
        Max bars per API call, default 50000.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: timestamp, open, high, low, close.
        Sorted chronologically. Missing bars are NOT filled.

    Notes
    -----
    - Data source assumption: Massive API, default quote type (mid).
    - No bid/ask/mid specification given in the paper; this is an unresolved assumption.
    """
    api_key = os.getenv("MASSIVE_API_KEY") or os.getenv("MASSIVE_TOKEN")
    if not api_key:
        raise EnvironmentError(
            "Neither MASSIVE_API_KEY nor MASSIVE_TOKEN environment variable is set."
        )

    from massive import RESTClient  # type: ignore

    client = RESTClient(api_key=api_key)

    aggs = []
    for a in client.list_aggs(
        ticker=ticker,
        multiplier=multiplier,
        timespan=timespan,
        from_=from_date,
        to=to_date,
        limit=limit,
    ):
        aggs.append(a)

    if not aggs:
        logger.warning("No data returned for %s from %s to %s", ticker, from_date, to_date)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close"])

    records = []
    for a in aggs:
        records.append(
            {
                "timestamp": pd.Timestamp(a.timestamp, unit="ms", tz="UTC"),
                "open": float(a.open),
                "high": float(a.high),
                "low": float(a.low),
                "close": float(a.close),
            }
        )

    df = pd.DataFrame(records)
    df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Log missing bars
    _log_missing_bars(df, ticker)

    logger.info(
        "Loaded %d bars for %s from %s to %s",
        len(df),
        ticker,
        df["timestamp"].iloc[0] if len(df) > 0 else "N/A",
        df["timestamp"].iloc[-1] if len(df) > 0 else "N/A",
    )
    return df


def _log_missing_bars(df: pd.DataFrame, ticker: str) -> None:
    """
    Log the presence of missing 1-minute bars (gaps > 1 minute on weekdays).

    Parameters
    ----------
    df : pd.DataFrame
        OHLC DataFrame with 'timestamp' column.
    ticker : str
        Ticker name for logging context.
    """
    if len(df) < 2:
        return

    diffs = df["timestamp"].diff().dropna()
    # Expected gap is 1 minute; gaps > 2 minutes on weekdays are suspicious
    large_gaps = diffs[diffs > pd.Timedelta(minutes=2)]
    if not large_gaps.empty:
        logger.warning(
            "%s: Found %d gaps > 2 minutes. First gap at index %d (%s).",
            ticker,
            len(large_gaps),
            large_gaps.index[0],
            df["timestamp"].iloc[large_gaps.index[0]],
        )
    else:
        logger.info("%s: No significant bar gaps detected.", ticker)


def price_to_pips(price_diff: float, pair: str) -> float:
    """
    Convert a price difference to pips for a given FX pair.

    For most FX pairs, 1 pip = 0.0001 (4th decimal place).
    For JPY pairs, 1 pip = 0.01 (2nd decimal place).

    Parameters
    ----------
    price_diff : float
        Raw price difference.
    pair : str
        FX pair string, e.g. 'EURGBP', 'GBPUSD'.

    Returns
    -------
    float
        Price difference expressed in pips.
    """
    pair_upper = pair.upper().replace("C:", "").replace("/", "")
    if "JPY" in pair_upper:
        return price_diff / 0.01
    return price_diff / 0.0001


def pips_to_price(pips: float, pair: str) -> float:
    """
    Convert pips to a raw price difference for a given FX pair.

    Parameters
    ----------
    pips : float
        Number of pips.
    pair : str
        FX pair string.

    Returns
    -------
    float
        Raw price difference.
    """
    pair_upper = pair.upper().replace("C:", "").replace("/", "")
    if "JPY" in pair_upper:
        return pips * 0.01
    return pips * 0.0001


def load_or_fetch(
    ticker: str,
    from_date: str,
    to_date: str,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load OHLC data from cache if available, otherwise fetch from Massive API.

    Parameters
    ----------
    ticker : str
        FX pair ticker.
    from_date : str
        Start date 'YYYY-MM-DD'.
    to_date : str
        End date 'YYYY-MM-DD'.
    cache_path : str, optional
        Path to a parquet cache file. If provided and exists, load from it.
        If provided and does not exist, fetch and save to it.

    Returns
    -------
    pd.DataFrame
        OHLC DataFrame.
    """
    if cache_path and os.path.exists(cache_path):
        logger.info("Loading cached data from %s", cache_path)
        df = pd.read_parquet(cache_path)
        return df

    df = fetch_ohlc(ticker, from_date, to_date)

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        df.to_parquet(cache_path, index=False)
        logger.info("Saved data cache to %s", cache_path)

    return df
