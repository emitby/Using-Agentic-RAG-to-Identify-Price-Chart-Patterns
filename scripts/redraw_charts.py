"""Redraw the annotated charts from the latest pipeline result without calling the API.

Useful after changing chart styling: the agents' decisions are reused from
outputs/latest_result.json, and the geometry is rebuilt by re-running the deterministic
detector on the same price data.

Usage:
    ./run.sh scripts/redraw_charts.py
"""

import json

from rag_charts import config
from rag_charts.charts import plot_annotated
from rag_charts.guardrails import safe_path
from rag_charts.patterns import detect
from rag_charts.pipeline import build_windows


def main() -> None:
    """Rebuild each window's annotations from saved reviews and redraw its chart.

    Raises:
        ValueError: If detection no longer produces a candidate a saved review refers to
            (e.g. the price data changed since the run).
    """
    result = json.loads(safe_path(config.OUTPUT_DIR / "latest_result.json").read_text(encoding="utf-8"))
    bars_by_window, meta = build_windows(result["ticker"])
    for key, window in result["windows"].items():
        bars = bars_by_window[key]
        report = detect(bars, meta[key]["bar_size"])
        by_id = {c["id"]: c for c in report["candidates"]}
        annotations = []
        for saved in window["annotations"]:
            cand = by_id.get(saved["candidate_id"])
            if cand is None or cand["type"] != saved["type"]:
                raise ValueError(f"{key}: candidate {saved['candidate_id']} changed; rerun the pipeline")
            annotations.append({**saved, "shapes": cand["shapes"]})
        path = plot_annotated(bars, result["ticker"],
                              f"{meta[key]['years']} years ({meta[key]['bar_size']} bars)",
                              annotations, window["review"]["verdict"], report["trend"],
                              config.OUTPUT_DIR / f"{result['ticker'].lower()}_annotated_{key}.png")
        print(f"{key}: {path}")


if __name__ == "__main__":
    main()
