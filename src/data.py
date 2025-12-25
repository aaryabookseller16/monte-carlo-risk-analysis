from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
import yfinance as yf

DATA_DIR = Path("data")


def _cache_path(ticker: str) -> Path:
    """Return the CSV cache path for a ticker."""
    return DATA_DIR / f"{ticker.lower()}.csv"


def load_prices(
    ticker: str = "SPY",
    start: str = "2015-01-01",
    end: Optional[str] = None,
    *,
    force_download: bool = False,
) -> pd.DataFrame:
    """
    Load adjusted close prices for a ticker using a local CSV cache for reproducibility.

    Returns a DataFrame indexed by 'date' with a single column: 'adj_close'.

    Parameters
    ----------
    ticker : str
        Asset ticker (e.g., 'SPY', 'QQQ', 'BTC-USD').
    start : str
        Start date (YYYY-MM-DD).
    end : Optional[str]
        Optional end date (YYYY-MM-DD). If None, downloads up to latest.
    force_download : bool
        If True, bypasses the cache and re-downloads data.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = _cache_path(ticker)

    # Load cache if present and not forcing download
    if csv_path.exists() and not force_download:
        df = pd.read_csv(csv_path, parse_dates=["date"], index_col="date")
        df = df.sort_index().dropna()
        _validate_prices_df(df, ticker=ticker)
        return df

    # Download from Yahoo via yfinance
    raw = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker='{ticker}'. "
            "Check ticker symbol and date range."
        )

    # Extract adjusted close robustly (Series vs DataFrame / MultiIndex)
    adj = raw["Adj Close"]
    if isinstance(adj, pd.DataFrame):
        if adj.shape[1] == 1:
            adj = adj.iloc[:, 0]
        elif ticker in adj.columns:
            adj = adj[ticker]
        else:
            raise ValueError(f"Unexpected Adj Close columns: {list(adj.columns)}")

    prices = adj.rename("adj_close")
    df = prices.to_frame().dropna()
    df.index.name = "date"
    df = df.sort_index()

    _validate_prices_df(df, ticker=ticker)

    # Cache to disk
    df.to_csv(csv_path)

    return df


def _validate_prices_df(df: pd.DataFrame, *, ticker: str) -> None:
    """Lightweight sanity checks to prevent silent downstream bugs."""
    if list(df.columns) != ["adj_close"]:
        raise ValueError(
            f"Expected a single column ['adj_close'] for ticker='{ticker}', "
            f"got columns={list(df.columns)}"
        )
    if df.index.name != "date":
        raise ValueError(
            f"Expected index name 'date' for ticker='{ticker}', got '{df.index.name}'"
        )
    if not pd.api.types.is_datetime64_any_dtype(df.index):
        raise TypeError(
            f"Expected datetime index for ticker='{ticker}', got {df.index.dtype}"
        )
    if df["adj_close"].isna().any():
        raise ValueError(f"Found NaNs in adj_close for ticker='{ticker}'")
    if len(df) < 50:
        raise ValueError(
            f"Too few rows ({len(df)}) for ticker='{ticker}'. "
            "Check ticker/date range."
        )


if __name__ == "__main__":
    # Simple sanity check (does not run on import)
    df = load_prices("SPY", start="2015-01-01")
    print(df.head())
    print(df.tail())
    print(df.columns)
    print(df.index.dtype)
    print(df.isna().sum())
