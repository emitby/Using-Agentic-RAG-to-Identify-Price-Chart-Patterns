"""Tests for price fetching (with Yahoo mocked out) and chart rendering."""

from pathlib import Path

import pandas as pd
import pytest
from helpers import path_series, test_output_dir

from rag_charts import data
from rag_charts.charts import plot_annotated, plot_close
from rag_charts.patterns import detect


def fake_download(tickers, start, end, **kwargs):
    """Mimic yf.download(group_by='ticker'): (ticker, field) MultiIndex columns."""
    dates = pd.bdate_range(start, pd.Timestamp(end) - pd.Timedelta(days=1))
    frames = {}
    for i, t in enumerate(tickers):
        base = 100 + 10 * i
        frames[t] = pd.DataFrame({"Open": base, "High": base + 1, "Low": base - 1,
                                  "Close": base + 0.5, "Adj Close": base + 0.4,
                                  "Volume": 1000}, index=pd.DatetimeIndex(dates, name="Date"))
    return pd.concat(frames, axis=1)


@pytest.fixture
def mocked_yahoo(monkeypatch):
    monkeypatch.setattr(data.yf, "download", fake_download)


def test_fetch_prices_multiple_tickers_long_format(mocked_yahoo):
    prices = data.fetch_prices("spy, qqq", start="2024-01-01", end="2024-03-01", use_cache=False)
    assert list(prices.columns) == data.COLUMNS
    assert set(prices["ticker"]) == {"SPY", "QQQ"}
    assert prices.groupby("ticker")["date"].is_monotonic_increasing.all()


def test_fetch_prices_validates_inputs():
    with pytest.raises(ValueError):
        data.fetch_prices([], years=1)
    with pytest.raises(ValueError):
        data.fetch_prices("SPY")                      # no window given
    with pytest.raises(ValueError):
        data.fetch_prices("SPY", start="2024-02-01", end="2024-01-01")


def test_weekly_bars_dated_by_last_trading_day(mocked_yahoo):
    prices = data.fetch_prices("SPY", start="2024-01-01", end="2024-01-17", use_cache=False)
    weekly = data.ohlcv(prices, "SPY", "weekly")
    # Jan 17 2024 is a Wednesday: the last partial week must be dated Jan 17, not Jan 19.
    assert weekly.index[-1] == pd.Timestamp("2024-01-17")


def test_plot_close_and_annotated_write_files():
    out = Path(test_output_dir())
    bars = path_series([(0, 90), (60, 120), (90, 105), (120, 120), (170, 95)])
    assert Path(plot_close(bars, "TEST", "1 year", out / "close.png")).exists()
    report = detect(bars, "daily")
    annotations = [{"label": c["type"], "description": "Costs $105 to test $ escaping.",
                    "role": "bearish", "shapes": c["shapes"], "citations": []}
                   for c in report["candidates"]]
    path = plot_annotated(bars, "TEST", "test", annotations,
                          {"bias": "bearish", "confidence": "low", "rationale": "Test $1 and $2."},
                          report["trend"], out / "annotated.png")
    assert Path(path).stat().st_size > 10_000
