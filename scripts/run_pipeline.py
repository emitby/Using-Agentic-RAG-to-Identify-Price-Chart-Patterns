"""Run the full agentic RAG pipeline and draw the annotated charts.

Usage:
    ./run.sh scripts/run_pipeline.py            # SPY
    ./run.sh scripts/run_pipeline.py QQQ        # another ticker

Needs ANTHROPIC_API_KEY and a built corpus (scripts/build_corpus.py). Spending stops at
config.RUN_BUDGET_USD. Writes outputs/<ticker>_annotated_{10y,3y}.png,
outputs/latest_result.json and outputs/runs/<run_id>/result.json.
"""

import json
import sys

from rag_charts import config
from rag_charts.pipeline import run_pipeline


def main(ticker: str) -> None:
    """Run the pipeline and print verdicts, annotation counts, and the cost summary."""
    result = run_pipeline(ticker)
    for key, window in result["windows"].items():
        verdict = window["review"]["verdict"]
        decisions = [r["decision"] for r in window["review"]["reviews"]]
        print(f"{key}: {verdict['bias']} ({verdict['confidence']}) | "
              f"{len(window['annotations'])} drawn, decisions {decisions} | {window['chart']}")
    print(json.dumps(result["cost"], indent=2))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else config.DEFAULT_TICKER)
