"""Stock price retrieval.

`fetch_prices` pulls daily OHLCV bars for one or many tickers from Yahoo Finance (via the
`yfinance` package) and returns them in one tidy, long-format DataFrame. Results are cached
as CSV under `.cache/prices/` so repeated runs on the same day do not re-download.

Prices are *unadjusted* closes (what a chart on StockCharts or a broker shows for each day)
with the dividend/split-adjusted close kept alongside in `adj_close`. Chart patterns are
drawn on the prices traders actually saw, which is why unadjusted is the default.
"""

from datetime import date, timedelta

import pandas as pd
import yfinance as yf

from rag_charts import config
from rag_charts.guardrails import ensure_dir, safe_path

PRICE_CACHE_DIR = config.CACHE_DIR / "prices"
COLUMNS = ["date", "ticker", "open", "high", "low", "close", "adj_close", "volume"]

# yfinance keeps a timezone/cookie cache in the user's AppData folder by default; point it
# inside the project so the app never writes outside it.
ensure_dir(config.CACHE_DIR / "yfinance")
yf.set_tz_cache_location(str(safe_path(config.CACHE_DIR / "yfinance")))


def _resolve_dates(years: float | None, start: str | date | None,
                   end: str | date | None) -> tuple[date, date]:
    """Work out the request window from either `years` or explicit dates.

    Args:
        years: Look-back length in years ending at `end` (ignored if `start` is given).
        start: First date to include.
        end: Last date to include; defaults to today.

    Returns:
        (start_date, end_date) as `date` objects.

    Raises:
        ValueError: If neither `years` nor `start` is given, or start is after end.
    """
    end_date = pd.Timestamp(end).date() if end else date.today()
    if start:
        start_date = pd.Timestamp(start).date()
    elif years:
        # 365.25 days per year keeps leap years from shaving a day off long windows.
        start_date = end_date - timedelta(days=round(365.25 * years))
    else:
        raise ValueError("give either `years` or `start`")
    if start_date >= end_date:
        raise ValueError(f"start {start_date} must be before end {end_date}")
    return start_date, end_date


def _download(tickers: list[str], start_date: date, end_date: date) -> pd.DataFrame:
    """Download bars from Yahoo Finance and reshape them to the long format.

    Args:
        tickers: Upper-case ticker symbols.
        start_date: First date (inclusive).
        end_date: Last date (inclusive).

    Returns:
        DataFrame with `COLUMNS`, sorted by ticker then date. Tickers Yahoo returns no
        data for are simply absent.
    """
    # yfinance treats `end` as exclusive, so ask for one extra day.
    raw = yf.download(
        tickers, start=start_date.isoformat(), end=(end_date + timedelta(days=1)).isoformat(),
        interval="1d", auto_adjust=False, group_by="ticker", progress=False, threads=False,
    )
    if raw is None or raw.empty:
        return pd.DataFrame(columns=COLUMNS)

    # With group_by="ticker" the columns are a (ticker, field) MultiIndex; stacking the
    # ticker level turns the wide table into one row per (date, ticker).
    long = raw.stack(level=0, future_stack=True).reset_index()
    long.columns = [str(c) for c in long.columns]
    long = long.rename(columns={
        "Date": "date", "Ticker": "ticker", "level_1": "ticker", "Open": "open",
        "High": "high", "Low": "low", "Close": "close", "Adj Close": "adj_close",
        "Volume": "volume",
    })
    long = long.dropna(subset=["close"])
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long[COLUMNS].sort_values(["ticker", "date"]).reset_index(drop=True)


def fetch_prices(tickers: str | list[str], years: float | None = None,
                 start: str | date | None = None, end: str | date | None = None,
                 use_cache: bool = True) -> pd.DataFrame:
    """Fetch daily OHLCV price data for one or more tickers.

    Examples:
        fetch_prices("SPY", years=10)
        fetch_prices(["SPY", "QQQ"], start="2020-01-01", end="2024-12-31")

    Args:
        tickers: A ticker ("SPY"), a comma/space separated string ("SPY, QQQ") or a list.
        years: Look-back in years, ending at `end` (or today).
        start: Explicit first date; overrides `years`.
        end: Last date; defaults to today.
        use_cache: Reuse a cached CSV for the identical request if present.

    Returns:
        Long-format DataFrame with columns
        date, ticker, open, high, low, close, adj_close, volume
        (one row per ticker per trading day, sorted by ticker then date).

    Raises:
        ValueError: If no tickers are given, the window is invalid, or Yahoo returns no
            data for any requested ticker.

    Side effects:
        Network call to Yahoo Finance on a cache miss; writes `.cache/prices/*.csv`.
    """
    if isinstance(tickers, str):
        tickers = tickers.replace(",", " ").split()
    symbols = sorted({t.strip().upper() for t in tickers if t.strip()})
    if not symbols:
        raise ValueError("no tickers given")
    start_date, end_date = _resolve_dates(years, start, end)

    cache_file = safe_path(PRICE_CACHE_DIR / f"{'_'.join(symbols)}_{start_date}_{end_date}.csv")
    if use_cache and cache_file.exists():
        cached = pd.read_csv(cache_file, parse_dates=["date"])
        return cached[COLUMNS]

    prices = _download(symbols, start_date, end_date)
    missing = sorted(set(symbols) - set(prices["ticker"].unique()))
    if len(missing) == len(symbols):
        raise ValueError(f"no price data returned for {', '.join(symbols)}")
    if missing:
        print(f"warning: no price data for {', '.join(missing)}")

    ensure_dir(PRICE_CACHE_DIR)
    prices.to_csv(cache_file, index=False)
    return prices


def ohlcv(prices: pd.DataFrame, ticker: str, bars: str = "daily") -> pd.DataFrame:
    """Extract one ticker as a date-indexed OHLCV frame, optionally resampled to weekly.

    Weekly bars use standard OHLC aggregation (first open, max high, min low, last close,
    summed volume) on weeks ending Friday, matching how charting sites build weekly charts.
    Each weekly bar is dated by its last actual trading day, so a week in progress is not
    labeled with a future Friday.

    Args:
        prices: Output of `fetch_prices`.
        ticker: Ticker to extract.
        bars: "daily" or "weekly".

    Returns:
        DataFrame indexed by date with open, high, low, close, volume columns.

    Raises:
        ValueError: If the ticker is absent or `bars` is unknown.
    """
    one = prices[prices["ticker"] == ticker.upper()]
    if one.empty:
        raise ValueError(f"{ticker} not in price data")
    frame = one.set_index("date")[["open", "high", "low", "close", "volume"]].astype(float)
    if bars == "daily":
        return frame
    if bars == "weekly":
        frame = frame.assign(last_day=frame.index)
        weekly = frame.resample("W-FRI").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
             "last_day": "last"})
        weekly = weekly.dropna(subset=["close"])
        return weekly.set_index("last_day").rename_axis("date")
    raise ValueError(f"unknown bar size '{bars}'")
