"""Chart rendering: plain closing-price charts and fully annotated pattern charts.

Two entry points:
  * `plot_close` draws a clean daily-close line chart (step 1 of the project).
  * `plot_annotated` draws the final chart: price, moving averages, the support/resistance
    levels, trendlines and pattern shapes the agents approved, numbered callouts, a notes
    panel with each annotation's description and source, and a bullish/bearish badge.

Geometry always comes from the deterministic detector (`patterns.py`); the agents only
choose which shapes to show and write their labels. That split keeps every line on the
chart tied to real prices.
"""

import textwrap

import matplotlib

matplotlib.use("Agg")  # file output only; no GUI windows
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from rag_charts import config  # noqa: E402
from rag_charts.guardrails import ensure_dir, safe_path  # noqa: E402

# Colors keyed by an annotation's role, so meaning is consistent across every chart.
ROLE_COLORS = {
    "support": "#1a9850",      # green
    "resistance": "#d73027",   # red
    "bullish": "#1a9850",
    "bearish": "#d73027",
    "neutral": "#4a6fa5",      # muted blue
}
VERDICT_COLORS = {"bullish": "#1a9850", "bearish": "#d73027", "neutral": "#7f7f7f"}
SITE_LABELS = {"stockcharts": "StockCharts ChartSchool", "investopedia": "Investopedia",
               "thepatternsite": "ThePatternSite (Bulkowski)"}


def _save(fig, out_path) -> str:
    """Save a figure inside the project and close it.

    Args:
        fig: Matplotlib figure.
        out_path: Destination PNG path (sandboxed).

    Returns:
        The saved path as a string.
    """
    target = safe_path(out_path)
    ensure_dir(target.parent)
    fig.savefig(target, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(target)


def _style_date_axis(ax, span_days: int) -> None:
    """Pick tick spacing that stays readable for both 3-year and 10-year windows."""
    if span_days > 5 * 365:
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    else:
        ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 4, 7, 10]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.grid(True, alpha=0.25)


def plot_close(bars: pd.DataFrame, ticker: str, label: str, out_path) -> str:
    """Draw a plain closing-price line chart.

    Args:
        bars: Date-indexed frame with a `close` column (from `data.ohlcv`).
        ticker: Ticker symbol for the title.
        label: Window label for the title, e.g. "10 years".
        out_path: Destination PNG path inside the project.

    Returns:
        Path of the saved PNG.

    Side effects:
        Writes the PNG file.
    """
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(bars.index, bars["close"], color="#1f3b73", linewidth=1.1)
    first, last = bars.index[0], bars.index[-1]
    change = bars["close"].iloc[-1] / bars["close"].iloc[0] - 1
    ax.set_title(f"{ticker} daily close — last {label} "
                 f"({first:%b %d, %Y} to {last:%b %d, %Y}, {change:+.0%})", fontsize=13)
    ax.set_ylabel("Price (USD)")
    _style_date_axis(ax, (last - first).days)
    ax.annotate(f"${bars['close'].iloc[-1]:,.2f}", xy=(last, bars["close"].iloc[-1]),
                xytext=(6, 0), textcoords="offset points", va="center", fontsize=9)
    return _save(fig, out_path)


def _source_note(citations: list[dict]) -> str:
    """Turn citation dicts into a short "Source: Site (title)" line."""
    if not citations:
        return ""
    first = citations[0]
    site = SITE_LABELS.get(first.get("site", ""), first.get("site", ""))
    extra = f" +{len(citations) - 1} more" if len(citations) > 1 else ""
    return f"Source: {site} — {first.get('title', '')[:60]}{extra}"


def _draw_shape(ax, shape: dict, color: str, number: int | None) -> None:
    """Draw one piece of an annotation's geometry, with its numbered callout if given.

    An annotation can have several shapes (a head-and-shoulders outline plus its
    neckline); only the first carries the number.

    Shape kinds:
      * "level": horizontal line across the window (support/resistance), with a faint
        band of +/- the clustering tolerance.
      * "line": a straight segment between two dated points (trendline, channel side,
        neckline). Dotted when the detector marks it `style: "secondary"` (a neckline or
        confirmation level, as opposed to the pattern's main lines).
      * "polyline": connected pivot points outlining a pattern (e.g. head and shoulders).
      * "marker": a single dated point (e.g. a moving-average cross).
    """
    kind = shape["kind"]
    points = [(pd.Timestamp(p["date"]), p["price"]) for p in shape.get("points", [])]
    badge = dict(boxstyle="circle,pad=0.25", fc=color, ec="none")
    if kind == "level":
        price = shape["price"]
        ax.axhline(price, color=color, linestyle="--", linewidth=1.3, alpha=0.9)
        band = shape.get("band", 0)
        if band:
            ax.axhspan(price - band, price + band, color=color, alpha=0.07)
        if number is not None:
            ax.annotate(f"{number}", xy=(ax.get_xlim()[1], price), xytext=(4, 0),
                        textcoords="offset points", va="center", fontsize=9,
                        fontweight="bold", color="white", bbox=badge, annotation_clip=False)
        return
    if kind in ("line", "polyline") and points:
        xs, ys = zip(*points)
        secondary = shape.get("style") == "secondary"
        ax.plot(xs, ys, color=color, linewidth=1.2 if secondary else 1.8,
                linestyle=":" if secondary else "-",
                marker="o" if kind == "polyline" else None, markersize=4, alpha=0.95)
        anchor = points[len(points) // 2] if kind == "polyline" else points[-1]
    elif kind == "marker" and points:
        anchor = points[0]
        ax.plot([anchor[0]], [anchor[1]], marker="D", color=color, markersize=8,
                markeredgecolor="black", zorder=5)
    else:
        return
    if number is not None:
        ax.annotate(f"{number}", xy=anchor, xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=9, fontweight="bold", color="white", bbox=badge,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8))


def plot_annotated(bars: pd.DataFrame, ticker: str, label: str, annotations: list[dict],
                   verdict: dict, trend: dict, out_path) -> str:
    """Draw the final annotated chart.

    Args:
        bars: Date-indexed OHLCV frame for the window (daily or weekly).
        ticker: Ticker symbol.
        label: Window label, e.g. "10 years (weekly bars)".
        annotations: Approved annotations, each with keys `label`, `description`, `role`
            (support/resistance/bullish/bearish/neutral), `shapes` (list; see
            `_draw_shape`) and `citations` (list of {site, title, url}).
        verdict: {"bias": bullish|bearish|neutral, "confidence": str, "rationale": str}.
        trend: Trend summary from `patterns.trend_regime`, used for the moving averages
            (`sma_fast_series`, `sma_slow_series`, `fast_len`, `slow_len`).
        out_path: Destination PNG inside the project.

    Returns:
        Path of the saved PNG.

    Side effects:
        Writes the PNG file.
    """
    # Height of the notes panel scales with the number of wrapped text lines it holds.
    note_lines = sum(len(textwrap.wrap(f"{a['label']}: {a['description']}", 150)) + 1.4
                     for a in annotations) + 1.5
    notes_h = 0.19 * note_lines
    fig = plt.figure(figsize=(14, 8.5 + notes_h))
    grid = fig.add_gridspec(2, 1, height_ratios=[8.5, notes_h], hspace=0.12)
    ax = fig.add_subplot(grid[0])
    notes_ax = fig.add_subplot(grid[1])
    notes_ax.axis("off")

    ax.plot(bars.index, bars["close"], color="#1f3b73", linewidth=1.1, label="Close")
    if trend.get("sma_fast_series") is not None:
        ax.plot(bars.index, trend["sma_fast_series"], color="#f39c12", linewidth=1,
                alpha=0.85, label=f"{trend['fast_len']}-bar SMA")
        ax.plot(bars.index, trend["sma_slow_series"], color="#8e44ad", linewidth=1,
                alpha=0.85, label=f"{trend['slow_len']}-bar SMA")
    # Freeze the x-range before drawing levels so right-edge callouts anchor correctly.
    ax.set_xlim(bars.index[0], bars.index[-1] + (bars.index[-1] - bars.index[0]) * 0.02)

    for number, item in enumerate(annotations, start=1):
        color = ROLE_COLORS.get(item.get("role", "neutral"), ROLE_COLORS["neutral"])
        for i, shape in enumerate(item["shapes"]):
            _draw_shape(ax, shape, color, number if i == 0 else None)

    first, last = bars.index[0], bars.index[-1]
    ax.set_title(f"{ticker} — chart patterns, last {label} "
                 f"({first:%b %Y} to {last:%b %d, %Y})", fontsize=13)
    ax.set_ylabel("Price (USD)")
    _style_date_axis(ax, (last - first).days)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)

    # Verdict badge in the top-left corner of the price panel.
    bias = verdict.get("bias", "neutral")
    badge = (f"{bias.upper()} ({verdict.get('confidence', '?')} confidence)\n"
             + textwrap.fill(_plain(verdict.get("rationale", "")), 70))
    ax.text(0.01, 0.98, badge, transform=ax.transAxes, va="top", ha="left", fontsize=8.5,
            color="white", bbox=dict(boxstyle="round,pad=0.5",
                                     fc=VERDICT_COLORS.get(bias, "#7f7f7f"), alpha=0.92))

    # Notes panel: one numbered entry per annotation, with its source. Entries are laid
    # out by counting wrapped lines so multi-line descriptions never overlap the next one.
    entries = []
    for item in annotations:
        lines = textwrap.wrap(_plain(f"{item['label']}: {item['description']}"), 150)
        source = _plain(_source_note(item.get("citations", [])))
        entries.append(lines + ([source] if source else []))
    total_lines = sum(len(e) for e in entries) + len(entries) * 0.4 + 1.5
    line_h = 1.0 / max(total_lines, 1)
    y = 1.0
    for number, (item, lines) in enumerate(zip(annotations, entries), start=1):
        color = ROLE_COLORS.get(item.get("role", "neutral"), ROLE_COLORS["neutral"])
        notes_ax.text(0.0, y, f" {number} ", transform=notes_ax.transAxes, va="top",
                      fontsize=9, fontweight="bold", color="white",
                      bbox=dict(boxstyle="circle,pad=0.2", fc=color, ec="none"))
        notes_ax.text(0.03, y, "\n".join(lines), transform=notes_ax.transAxes, va="top",
                      fontsize=8.3, linespacing=1.25)
        y -= line_h * (len(lines) + 0.4)
    notes_ax.text(0.0, 0.0, config.DISCLAIMER, transform=notes_ax.transAxes, fontsize=7.5,
                  style="italic", color="#555555")
    return _save(fig, out_path)


def _plain(text: str) -> str:
    """Escape dollar signs so matplotlib prints prices instead of parsing math mode.

    Without this, "$759.57 ... $779.09" renders the text between the two dollar signs
    as an italic math expression with its spaces removed.
    """
    return text.replace("$", r"\$")
