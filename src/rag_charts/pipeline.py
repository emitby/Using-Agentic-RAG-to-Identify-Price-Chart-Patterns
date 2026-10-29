"""The orchestrator: a fixed, code-driven pipeline that runs the three agents.

    prices -> windows (10y weekly, 3y daily) -> deterministic detection
           -> Researcher (once, for every pattern type found in either window)
           -> per window: Chart Analyst -> Verifier -> annotations -> annotated chart

The orchestrator itself is plain Python, not an agent: the order of steps never changes,
so letting a model decide it would only add cost and failure modes. Model judgment is used
where it is needed: interpreting definitions, choosing what to show, and checking it.
"""

from datetime import datetime

import pandas as pd

from rag_charts import config
from rag_charts.agents import (Session, analyst, build_annotations, researcher, verifier)
from rag_charts.charts import plot_annotated
from rag_charts.costs import CostTracker
from rag_charts.data import fetch_prices, ohlcv
from rag_charts.guardrails import write_json
from rag_charts.patterns import detect, report_for_model
from rag_charts.retrieval import load_index

# Human-readable names the Researcher searches for, per detector candidate type.
PATTERN_NAMES = {
    "support": "support and resistance levels",
    "resistance": "support and resistance levels",
    "rising_support_trendline": "trend lines (rising support / uptrend line)",
    "falling_resistance_trendline": "trend lines (falling resistance / downtrend line)",
    "double_top": "double top reversal", "double_bottom": "double bottom reversal",
    "triple_top": "triple top reversal", "triple_bottom": "triple bottom reversal",
    "head_and_shoulders_top": "head and shoulders top",
    "inverse_head_and_shoulders": "head and shoulders bottom (inverse)",
    "ascending_triangle": "ascending triangle", "descending_triangle": "descending triangle",
    "symmetrical_triangle": "symmetrical triangle", "rising_wedge": "rising wedge",
    "falling_wedge": "falling wedge", "rising_channel": "price channel (rising)",
    "falling_channel": "price channel (falling)", "rectangle": "rectangle",
    "broadening_formation": "broadening formation (megaphone)",
    "golden_cross": "moving average crossovers (golden cross)",
    "death_cross": "moving average crossovers (death cross)",
}


def build_windows(ticker: str) -> tuple[dict, dict]:
    """Fetch prices once and slice them into the configured windows.

    Args:
        ticker: Ticker symbol.

    Returns:
        (bars by window key, metadata by window key with title and bar size).
    """
    prices = fetch_prices(ticker, years=max(years for _, years, _ in config.TIMEFRAMES))
    bars_by_window, meta = {}, {}
    for key, years, bar_size in config.TIMEFRAMES:
        bars = ohlcv(prices, ticker, bar_size)
        bars = bars[bars.index >= bars.index[-1] - pd.DateOffset(years=years)]
        bars_by_window[key] = bars
        meta[key] = {"bar_size": bar_size, "years": years,
                     "title": f"{ticker}, last {years} years, {bar_size} bars"}
    return bars_by_window, meta


def run_pipeline(ticker: str = config.DEFAULT_TICKER, budget_usd: float = config.RUN_BUDGET_USD,
                 client=None, run_id: str | None = None) -> dict:
    """Run detection, the three agents, and chart rendering for one ticker.

    Args:
        ticker: Ticker symbol.
        budget_usd: Spend cap for this run (stops before the next API call once reached).
        client: Anthropic client (default constructs one from the environment).
        run_id: Identifier for logs/outputs (default: timestamp).

    Returns:
        Result dict with, per window, the detection report (model view), analyst and
        verifier outputs, annotations, chart path; plus research output and cost summary.

    Side effects:
        Network calls (prices, Claude API); writes charts and outputs/runs/<run_id>/.
    """
    run_id = run_id or datetime.now().strftime("%Y%m%d-%H%M%S")
    tracker = CostTracker(run_id, budget_usd)
    session = Session(index=load_index())
    bars_by_window, meta = build_windows(ticker)

    views = {}
    for key, bars in bars_by_window.items():
        session.bars[key] = bars
        session.reports[key] = detect(bars, meta[key]["bar_size"])
        views[key] = report_for_model(session.reports[key])

    types = {c["type"] for r in session.reports.values() for c in r["candidates"]}
    names = sorted({PATTERN_NAMES.get(t, t.replace("_", " ")) for t in types})
    print(f"[{run_id}] researching {len(names)} pattern types")
    research = researcher(names, session, tracker, client)

    result = {"run_id": run_id, "ticker": ticker, "model": config.MODEL,
              "research": research.model_dump(), "windows": {}}
    for key, bars in bars_by_window.items():
        title = meta[key]["title"]
        print(f"[{run_id}] {key}: analyst")
        analysis = analyst(key, title, research, session, tracker, client, views[key])
        print(f"[{run_id}] {key}: verifier")
        review = verifier(key, title, research, analysis, session, tracker, client, views[key])
        annotations = build_annotations(review, key, session)
        chart = plot_annotated(
            bars, ticker, f"{meta[key]['years']} years ({meta[key]['bar_size']} bars)",
            annotations, review.verdict.model_dump(), session.reports[key]["trend"],
            config.OUTPUT_DIR / f"{ticker.lower()}_annotated_{key}.png")
        result["windows"][key] = {
            "title": title, "detection": views[key], "analysis": analysis.model_dump(),
            "review": review.model_dump(),
            "annotations": [{k: v for k, v in a.items() if k != "shapes"} for a in annotations],
            "chart": chart,
        }

    result["cost"] = tracker.summary()
    write_json(config.OUTPUT_DIR / "runs" / run_id / "result.json", result)
    write_json(config.OUTPUT_DIR / "latest_result.json", result)
    return result
