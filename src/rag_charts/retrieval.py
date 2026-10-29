"""Keyword retrieval over the corpus with BM25.

Why BM25 rather than embeddings: the Claude API has no embeddings endpoint, the corpus is
small (~1,000 chunks), and the queries are full of exact technical terms ("descending
triangle", "neckline", "throwback") that lexical matching handles well. BM25 runs locally,
needs no extra API key, and is deterministic, which makes retrieval evals repeatable.

Chunks flagged by the prompt-injection scan are excluded from results by default.
"""

import re
from functools import lru_cache

from rank_bm25 import BM25Okapi

from rag_charts import config
from rag_charts.guardrails import read_jsonl, wrap_untrusted

# Common words that carry no meaning for pattern lookups. Kept short on purpose: words
# like "top", "bottom", "up" and "down" are stopwords elsewhere but are the whole point here.
STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "to", "in", "on", "for", "is", "are", "was",
    "be", "it", "this", "that", "with", "as", "at", "by", "from", "what", "how", "when",
    "which", "does", "do", "its", "i", "you", "can", "will",
}
SITE_ALIASES = {"any": None, "stockcharts": "stockcharts", "investopedia": "investopedia",
                "thepatternsite": "thepatternsite"}


def tokenize(text: str) -> list[str]:
    """Lower-case, split on non-alphanumerics, drop stopwords, and fold simple plurals.

    Plural folding ("tops" -> "top", "triangles" -> "triangle") matters because page
    titles are plural and queries are often singular.

    Args:
        text: Any text.

    Returns:
        Token list.
    """
    tokens = []
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        if word in STOPWORDS:
            continue
        # Fold "tops", "gaps", "lines" but leave "cross", "bonus", "analysis" intact.
        if len(word) > 3 and word.endswith("s") and not word.endswith(("ss", "us", "is")):
            word = word[:-1]
        tokens.append(word)
    return tokens


TITLE_WEIGHT = 3  # times the page title is repeated in each chunk's indexed text


def clean_title(title: str) -> str:
    """Strip site-name suffixes ("Double Top | ChartSchool | StockCharts.com" -> "Double Top").

    The suffix appears on every ChartSchool page, so it adds noise to every match.
    """
    return title.split(" | ")[0].strip()


class CorpusIndex:
    """BM25 index over corpus chunks.

    Each chunk is indexed as (page title x TITLE_WEIGHT) + heading + text. Weighting the
    title makes a query that names a pattern rank that pattern's own page above pages that
    merely mention it in passing (e.g. "double top" vs. a head-and-shoulders page that
    compares itself to double tops).

    Attributes:
        chunks: All chunk records, including flagged ones.
        by_id: Chunk lookup by id (used to check citations).
    """

    def __init__(self, chunks: list[dict]):
        """Build the index.

        Args:
            chunks: Chunk records (id, url, site, title, heading, text, injection_flags).

        Raises:
            ValueError: If there are no chunks (the corpus has not been built).
        """
        if not chunks:
            raise ValueError("corpus is empty; run ./run.sh scripts/build_corpus.py first")
        self.chunks = chunks
        self.by_id = {c["id"]: c for c in chunks}
        self._bm25 = BM25Okapi([
            tokenize(f"{(clean_title(c['title']) + ' ') * TITLE_WEIGHT}"
                     f"{clean_title(c['heading'])} {c['text']}")
            for c in chunks])

    def search(self, query: str, k: int = 5, site: str | None = None,
               include_flagged: bool = False) -> list[dict]:
        """Return the top-k chunks for a query.

        Args:
            query: Natural-language or keyword query.
            k: Number of results (clamped to 1..10).
            site: Restrict to one site ("stockcharts", "investopedia", "thepatternsite").
            include_flagged: Include chunks flagged as possible prompt injection (only for
                guardrail tests).

        Returns:
            Chunk dicts with an added `score`, best first. Zero-score chunks are dropped.
        """
        k = max(1, min(int(k), 10))
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.chunks)), key=lambda i: scores[i], reverse=True)
        results = []
        for i in ranked:
            chunk = self.chunks[i]
            if scores[i] <= 0:
                break
            if site and chunk["site"] != site:
                continue
            if chunk.get("injection_flags") and not include_flagged:
                continue
            results.append({**chunk, "score": round(float(scores[i]), 3)})
            if len(results) == k:
                break
        return results

    def get(self, chunk_id: str) -> dict | None:
        """Look up a chunk by id (None if it does not exist)."""
        return self.by_id.get(chunk_id)


@lru_cache(maxsize=1)
def load_index() -> CorpusIndex:
    """Load data/corpus/chunks.jsonl and build the index once per process.

    Returns:
        The shared `CorpusIndex`.

    Raises:
        ValueError: If the corpus has not been built.
    """
    return CorpusIndex(read_jsonl(config.CHUNKS_FILE))


def format_results(results: list[dict]) -> str:
    """Render search results for a model, each wrapped as untrusted reference material.

    Args:
        results: Output of `CorpusIndex.search`.

    Returns:
        Text block listing each chunk's id, site, title and wrapped text.
    """
    if not results:
        return "No matching passages found."
    parts = []
    for r in results:
        source = f"{r['id']} | {r['site']} | {r['title']} | {r['url']}"
        parts.append(wrap_untrusted(f"[section: {r['heading']}]\n{r['text']}", source))
    return "\n\n".join(parts)
