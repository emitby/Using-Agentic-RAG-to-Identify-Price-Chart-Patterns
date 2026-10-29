"""Project-wide configuration: paths, model settings, pricing, crawl scope and budgets.

Everything that a user might want to tune lives here so the other modules stay free of
magic numbers. All paths are derived from this file's location, so the project works no
matter which directory a script is launched from, and every path points *inside* the
project folder (the guardrails module enforces that at write time).
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths. This file lives at <root>/src/rag_charts/config.py, so the project root is two
# directories above the package directory.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CACHE_DIR = PROJECT_ROOT / ".cache"
LOG_DIR = PROJECT_ROOT / "logs"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

# Corpus artifacts produced by scripts/build_corpus.py.
PAGES_FILE = CORPUS_DIR / "pages.jsonl"
CHUNKS_FILE = CORPUS_DIR / "chunks.jsonl"
CRAWL_LOG_FILE = CORPUS_DIR / "crawl_log.jsonl"

# Token/cost logs written by costs.CostTracker.
USAGE_LOG_FILE = LOG_DIR / "token_usage.jsonl"
TOOL_LOG_FILE = LOG_DIR / "tool_calls.jsonl"

# ---------------------------------------------------------------------------
# Model settings. The user chose Claude Opus 5.5 for all three agents. Opus 5.5 always
# thinks (adaptive); depth is controlled with `effort`, whose API default on this model is
# "medium", so it is set explicitly per agent. The analyst does the hardest judgment work.
# ---------------------------------------------------------------------------
MODEL = "claude-opus-5-5"
AGENT_EFFORT = {
    "researcher": "medium",
    "analyst": "high",
    "verifier": "high",
}
MAX_TOKENS = 16000          # per response; non-streaming requests stay under SDK timeouts
MAX_AGENT_TURNS = 12        # hard stop for any single agent's tool loop

# Server-side refusal fallback (opt-in beta). On a safety decline the API re-runs the
# request on a fallback model it picks by refusal category, instead of just stopping.
FALLBACK_BETA = "server-side-fallback-2026-07-01"

# ---------------------------------------------------------------------------
# Pricing in USD per million tokens (Anthropic first-party rates, cached 2026-10-06).
# Cache writes use the 5-minute TTL rate (1.25x input). Used only for cost reporting.
# ---------------------------------------------------------------------------
PRICING = {
    "claude-opus-5-5": {"input": 4.00, "output": 20.00, "cache_write": 5.00, "cache_read": 0.20},
    "claude-sonnet-5-5": {"input": 2.00, "output": 10.00, "cache_write": 2.50, "cache_read": 0.20},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50, "cache_write": 0.125, "cache_read": 0.01},
}

# Abort a pipeline/eval run once its tracked spend passes this many dollars.
RUN_BUDGET_USD = 6.00
EVAL_BUDGET_USD = 8.00

# ---------------------------------------------------------------------------
# Crawl scope. Only these three sites may be fetched (guardrails.is_allowed_url).
# Seeds are the user-supplied entry points plus the specific Investopedia term pages for
# the patterns the detector can find; Investopedia is far too large to crawl from its
# home page alone.
# ---------------------------------------------------------------------------
ALLOWED_DOMAINS = {
    "chartschool.stockcharts.com": "stockcharts",
    "www.investopedia.com": "investopedia",
    "thepatternsite.com": "thepatternsite",
}

SEED_URLS = [
    "https://chartschool.stockcharts.com/",
    # ChartSchool's home page renders its navigation with JavaScript, so the handbook
    # section that holds the pattern pages is seeded directly.
    "https://chartschool.stockcharts.com/table-of-contents/chart-analysis",
    "https://chartschool.stockcharts.com/table-of-contents/chart-analysis/chart-patterns",
    "https://chartschool.stockcharts.com/table-of-contents/market-analysis",
    "https://thepatternsite.com/chartpatterns.html",
    "https://www.investopedia.com/",
    "https://www.investopedia.com/terms/s/support.asp",
    "https://www.investopedia.com/terms/r/resistance.asp",
    "https://www.investopedia.com/trading/support-and-resistance-basics/",
    "https://www.investopedia.com/terms/t/trendline.asp",
    "https://www.investopedia.com/terms/c/chartpattern.asp",
    "https://www.investopedia.com/terms/h/head-shoulders.asp",
    "https://www.investopedia.com/terms/i/inverseheadandshoulders.asp",
    "https://www.investopedia.com/terms/d/doubletop.asp",
    "https://www.investopedia.com/terms/d/doublebottom.asp",
    "https://www.investopedia.com/terms/t/triangle.asp",
    "https://www.investopedia.com/terms/a/ascendingtriangle.asp",
    "https://www.investopedia.com/terms/d/descendingtriangle.asp",
    "https://www.investopedia.com/terms/s/symmetricaltriangle.asp",
    "https://www.investopedia.com/terms/w/wedge.asp",
    "https://www.investopedia.com/terms/r/risingwedge.asp",
    "https://www.investopedia.com/terms/f/fallingwedge.asp",
    "https://www.investopedia.com/terms/c/channel.asp",
    "https://www.investopedia.com/terms/b/breakout.asp",
    "https://www.investopedia.com/terms/g/goldencross.asp",
    "https://www.investopedia.com/terms/d/deathcross.asp",
    "https://www.investopedia.com/terms/u/uptrend.asp",
    "https://www.investopedia.com/terms/d/downtrend.asp",
]

# A followed link must contain one of these words; this keeps the crawl on chart-pattern
# material instead of wandering into unrelated parts of large sites.
URL_KEYWORDS = (
    "pattern", "support", "resistance", "trend", "triangle", "wedge", "double", "triple",
    "head", "shoulder", "channel", "breakout", "flag", "pennant", "rectangle", "top",
    "bottom", "moving-average", "golden", "death", "dow-theory",
)
# Links containing any of these are skipped even if they match a keyword: candlestick
# patterns need OHLC candles (the charts here plot closes), and .md links are duplicate
# raw-markdown copies of ChartSchool pages.
URL_EXCLUDE = ("candlestick", "point-and-figure", ".md", "/login", "/signup", "/members")

# Stop crawling a site after this many refusals in a row (403/429/robots). A blocked site
# is logged and reported, never worked around.
MAX_CONSECUTIVE_REFUSALS = 3

MAX_PAGES_PER_SITE = 80
CRAWL_DELAY_SECONDS = 1.5   # politeness delay between requests to the same site
REQUEST_TIMEOUT_SECONDS = 20
# An honest, non-browser user agent; sites that refuse it are logged, not evaded.
USER_AGENT = "AgenticRAGPrototype/0.1 (personal research crawler; respects robots.txt)"

CHUNK_WORDS = 220           # target words per retrieval chunk
CHUNK_OVERLAP_WORDS = 40    # overlap so a rule split across a boundary is still findable

# ---------------------------------------------------------------------------
# Charts and pattern detection.
# ---------------------------------------------------------------------------
DEFAULT_TICKER = "SPY"
# (label, years of history, bar size). 10 years of daily bars is ~2,500 points of noise,
# so the long view is resampled to weekly bars before patterns are detected.
TIMEFRAMES = [
    ("10y", 10, "weekly"),
    ("3y", 3, "daily"),
]

DISCLAIMER = "Educational prototype. Not financial advice."
