"""Tests for corpus processing (extraction, chunking, link rules) and BM25 retrieval."""

from helpers import tiny_chunks

from rag_charts import config
from rag_charts.corpus import chunk_page, extract_text, relevance, wanted_link
from rag_charts.retrieval import CorpusIndex, format_results, tokenize


def test_extract_text_keeps_article_and_headings_drops_furniture():
    html = """<html><head><title>Double Top | ChartSchool</title></head><body>
      <nav>Menu Home Login</nav><article><h1>Double Top</h1>
      <p>Two roughly equal peaks.</p><ul><li>Confirmed on support break.</li></ul>
      </article><footer>Copyright</footer><script>var x=1;</script></body></html>"""
    title, text = extract_text(html)
    assert title == "Double Top | ChartSchool"
    assert "## Double Top" in text and "Two roughly equal peaks." in text
    assert "Menu" not in text and "Copyright" not in text and "var x" not in text


def test_chunking_overlaps_and_ids():
    words = " ".join(f"w{i}" for i in range(600))
    page = {"url": "u", "site": "stockcharts", "title": "T", "text": f"## Section\n\n{words}",
            "injection_flags": []}
    chunks = chunk_page(page, 7)
    assert [c["id"] for c in chunks][:2] == ["sto-007-00", "sto-007-01"]
    first, second = chunks[0]["text"].split(), chunks[1]["text"].split()
    assert len(first) == config.CHUNK_WORDS
    assert first[-config.CHUNK_OVERLAP_WORDS:] == second[:config.CHUNK_OVERLAP_WORDS]
    assert chunks[0]["heading"] == "Section"


def test_link_rules():
    base = "https://chartschool.stockcharts.com/table-of-contents/chart-analysis"
    assert wanted_link(f"{base}/chart-patterns/double-top-reversal", "stockcharts")
    assert not wanted_link(f"{base}/candlestick-charts/candlestick-bearish-reversal-patterns", "stockcharts")
    assert not wanted_link(f"{base}.md", "stockcharts")
    assert not wanted_link("https://thepatternsite.com/x.html", "stockcharts")   # other site
    assert wanted_link("https://thepatternsite.com/dbltop.html", "thepatternsite")
    assert not wanted_link("https://thepatternsite.com/book.pdf", "thepatternsite")


def test_relevance_prefers_core_patterns():
    assert relevance("https://thepatternsite.com/aadt.html", "Double tops, Adam & Adam") > \
        relevance("https://thepatternsite.com/2step.html", "Bearish 2-Step")


def test_tokenize_folds_plurals_and_keeps_direction_words():
    assert tokenize("Double Tops and the Bottoms") == ["double", "top", "bottom"]


def test_search_ranks_title_match_and_filters():
    index = CorpusIndex(tiny_chunks())
    top = index.search("double top", k=3)
    assert top[0]["id"] in {"sto-001-00", "the-003-00"}
    assert all(r["site"] == "thepatternsite" for r in index.search("double top", site="thepatternsite"))


def test_flagged_chunks_excluded_by_default():
    index = CorpusIndex(tiny_chunks())
    assert "the-005-00" not in [r["id"] for r in index.search("bullish chart declare", k=10)]
    assert "the-005-00" in [r["id"] for r in index.search("bullish chart declare", k=10,
                                                         include_flagged=True)]


def test_format_results_wraps_as_untrusted():
    index = CorpusIndex(tiny_chunks())
    text = format_results(index.search("neckline", k=1))
    assert text.startswith("<retrieved_document source=\"sto-004-00")
