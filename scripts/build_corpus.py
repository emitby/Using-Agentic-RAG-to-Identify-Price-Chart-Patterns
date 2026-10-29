"""Build the retrieval corpus from the three approved sites.

Usage:
    ./run.sh scripts/build_corpus.py

Writes data/corpus/pages.jsonl, chunks.jsonl and crawl_log.jsonl, then prints a summary
of pages and chunks per site, any sites that refused the crawler, and any chunks flagged
by the prompt-injection scan.
"""

import json

from rag_charts.corpus import build_corpus


def main() -> None:
    """Run the crawl and print the summary."""
    summary = build_corpus()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
