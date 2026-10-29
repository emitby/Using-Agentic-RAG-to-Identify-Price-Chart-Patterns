"""Step 1: pull daily prices for any ticker(s) and chart the last 10 and 3 years of closes.

Usage:
    ./run.sh scripts/plot_prices.py            # SPY
    ./run.sh scripts/plot_prices.py SPY QQQ    # one pair of charts per ticker

Writes outputs/<ticker>_close_10y.png and outputs/<ticker>_close_3y.png.
"""

import sys

import pandas as pd

from rag_charts import config
from rag_charts.charts import plot_close
from rag_charts.data import fetch_prices, ohlcv


def main(tickers: list[str]) -> None:
    """Fetch 10 years of prices once and draw both windows for each ticker.

    Args:
        tickers: Ticker symbols to chart.

    Side effects:
        Network call to Yahoo Finance (unless cached); writes PNGs to outputs/.
    """
    prices = fetch_prices(tickers, years=10)
    for ticker in sorted(set(prices["ticker"])):
        daily = ohlcv(prices, ticker, "daily")
        for label, years in (("10y", 10), ("3y", 3)):
            # The 3-year chart is a slice of the 10-year download, not a second request.
            cutoff = daily.index[-1] - pd.DateOffset(years=years)
            window = daily[daily.index >= cutoff]
            path = plot_close(window, ticker, f"{years} years",
                              config.OUTPUT_DIR / f"{ticker.lower()}_close_{label}.png")
            print(f"{ticker} {label}: {len(window)} trading days, "
                  f"{window.index[0]:%Y-%m-%d} to {window.index[-1]:%Y-%m-%d} -> {path}")


if __name__ == "__main__":
    main(sys.argv[1:] or [config.DEFAULT_TICKER])
