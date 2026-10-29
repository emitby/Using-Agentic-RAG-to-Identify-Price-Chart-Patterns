"""Corpus builder: crawl the three approved sites, clean the pages, and chunk them.

Pipeline: seed URLs -> polite relevance-ordered crawl (robots.txt, rate limit, allowlist,
no binaries) -> main-text extraction -> injection scan -> overlapping word chunks.

Outputs (all under data/corpus/):
  * pages.jsonl      one cleaned page per line (url, site, title, text, flags)
  * chunks.jsonl     retrieval chunks (id, url, site, title, heading, text, flags)
  * crawl_log.jsonl  every fetch attempt with its status, so blocked sites are visible

Crawl depth differs per site because the sites are shaped differently:
  * ThePatternSite's seed page is an index linking straight to every pattern page, so one
    hop from it is enough and URL keyword filtering is skipped (its file names are
    abbreviations that keywords would miss).
  * ChartSchool is a nested handbook, so the crawl goes three hops deep, keyword-filtered.
  * Investopedia is huge; seeds point at the relevant term pages and only one hop is taken.
"""

import heapq
import itertools
import json
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urldefrag, urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from rag_charts import config
from rag_charts.guardrails import (append_jsonl, ensure_dir, is_allowed_url,
                                   scan_for_injection, site_of, write_text)

MAX_DEPTH = {"stockcharts": 3, "investopedia": 1, "thepatternsite": 1}
KEYWORD_FILTER = {"stockcharts": True, "investopedia": True, "thepatternsite": False}

# Patterns the detector can find. Links whose text or URL mentions these are crawled
# first, so a per-site page cap is spent on the pages the agents will actually need
# (ThePatternSite's index is alphabetical and would otherwise fill the cap with "2-Step",
# "3-Bar" and similar before reaching "Double tops").
PRIORITY_TERMS = (
    "double top", "double bottom", "head-and-shoulders", "head and shoulders",
    "triangle", "wedge", "channel", "rectangle", "broadening", "trendline", "trend line",
    "support", "resistance", "rounding", "flag", "pennant", "measured move",
    "moving average", "golden cross", "death cross", "triple top", "triple bottom",
    "breakout", "busted",
)


def relevance(url: str, anchor_text: str = "") -> int:
    """Score how relevant a link is to the detectable patterns (higher crawls sooner).

    Args:
        url: Link URL.
        anchor_text: Visible link text, which on ThePatternSite names the pattern.

    Returns:
        Number of priority terms present in the URL or anchor text.
    """
    haystack = f"{url} {anchor_text}".lower().replace("-", " ")
    return sum(term.replace("-", " ") in haystack for term in PRIORITY_TERMS)

# Page furniture that is never article text.
STRIP_TAGS = ["script", "style", "noscript", "nav", "header", "footer", "aside", "form",
              "iframe", "svg", "button"]


def normalize_url(url: str) -> str:
    """Drop fragments and query strings and trailing slashes so each page is crawled once.

    Args:
        url: Absolute URL.

    Returns:
        Canonical form of the URL.
    """
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    path = parsed.path.rstrip("/") or "/"
    return f"{parsed.scheme}://{parsed.netloc.lower()}{path}"


def wanted_link(url: str, site: str) -> bool:
    """Decide whether a discovered link is worth following.

    Args:
        url: Normalized absolute URL.
        site: Short site name of the page the link was found on.

    Returns:
        True if the URL is allowed, on the same site, and (where the site uses keyword
        filtering) mentions a chart-pattern keyword.
    """
    if not is_allowed_url(url) or site_of(url) != site:
        return False
    if any(word in url.lower() for word in config.URL_EXCLUDE):
        return False
    if not KEYWORD_FILTER[site]:
        return urlparse(url).path.lower().endswith((".html", ".htm", "/"))
    return any(word in url.lower() for word in config.URL_KEYWORDS)


def extract_text(html: str) -> tuple[str, str]:
    """Pull the title and readable main text out of an HTML page.

    Prefers <article> or <main>; falls back to <body> for older sites (ThePatternSite).
    Headings are kept on their own lines and prefixed with '## ' so the chunker can carry
    section names into each chunk.

    Args:
        html: Raw page HTML.

    Returns:
        (title, text). Text is blank-line separated paragraphs.
    """
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.get_text(" ", strip=True) if soup.title else "").strip()
    for tag in soup(STRIP_TAGS):
        tag.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    lines = []
    for element in root.find_all(["h1", "h2", "h3", "h4", "p", "li", "td", "blockquote"]):
        # Skip containers whose text is already captured by a nested element.
        if element.find(["p", "li", "h1", "h2", "h3", "h4"]):
            continue
        text = " ".join(element.get_text(" ", strip=True).split())
        if len(text) < 3:
            continue
        lines.append(f"## {text}" if element.name in ("h1", "h2", "h3", "h4") else text)
    # Deduplicate repeated lines (menus rendered twice, repeated table cells) in order.
    seen, unique = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
    return title, "\n\n".join(unique)


def chunk_page(page: dict, page_index: int) -> list[dict]:
    """Split one page into overlapping ~CHUNK_WORDS-word chunks, tracking the heading.

    Overlap means a rule that straddles a chunk boundary still appears whole in one chunk.

    Args:
        page: Page record with url, site, title, text, injection_flags.
        page_index: Page number, used to build stable chunk ids.

    Returns:
        Chunk records.
    """
    chunks, buffer, heading = [], [], page["title"]

    def emit(words: list[str]) -> None:
        text = " ".join(words)
        chunks.append({
            "id": f"{page['site'][:3]}-{page_index:03d}-{len(chunks):02d}",
            "url": page["url"], "site": page["site"], "title": page["title"],
            "heading": heading, "text": text,
            "injection_flags": scan_for_injection(text),
        })

    for paragraph in page["text"].split("\n\n"):
        if paragraph.startswith("## "):
            # A new section starts a new chunk once the current one has some substance.
            if len(buffer) > config.CHUNK_WORDS // 3:
                emit(buffer)
                buffer = []
            heading = paragraph[3:]
            continue
        buffer.extend(paragraph.split())
        while len(buffer) >= config.CHUNK_WORDS:
            head, rest = buffer[:config.CHUNK_WORDS], buffer[config.CHUNK_WORDS:]
            emit(head)
            buffer = head[-config.CHUNK_OVERLAP_WORDS:] + rest
    if len(buffer) > config.CHUNK_OVERLAP_WORDS:
        emit(buffer)
    return chunks


class Crawler:
    """Polite crawler restricted to the approved sites, most relevant links first.

    Attributes:
        session: Shared HTTP session with the project's honest user agent.
        robots: Cached robots.txt parsers per host.
        last_hit: Time of the last request per host, for rate limiting.
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = config.USER_AGENT
        self.robots: dict[str, RobotFileParser | None] = {}
        self.last_hit: dict[str, float] = {}

    def allowed_by_robots(self, url: str) -> bool:
        """Check robots.txt for the URL's host (fetched once per host).

        If robots.txt cannot be fetched, crawling is allowed, which is the standard
        convention for a missing robots file.

        Args:
            url: URL to check.

        Returns:
            True if our user agent may fetch the URL.
        """
        host = urlparse(url).netloc
        if host not in self.robots:
            parser = RobotFileParser()
            try:
                resp = self.session.get(f"https://{host}/robots.txt",
                                        timeout=config.REQUEST_TIMEOUT_SECONDS)
                parser.parse(resp.text.splitlines() if resp.ok else [])
            except requests.RequestException:
                parser.parse([])
            self.robots[host] = parser
        return self.robots[host].can_fetch(config.USER_AGENT, url)

    def fetch(self, url: str) -> tuple[int, str]:
        """GET a page, honoring the per-host delay and refusing non-HTML responses.

        Args:
            url: Allowed URL.

        Returns:
            (status_code, html). html is "" for errors or non-HTML content; status 0 means
            a network error.
        """
        host = urlparse(url).netloc
        wait = config.CRAWL_DELAY_SECONDS - (time.monotonic() - self.last_hit.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        self.last_hit[host] = time.monotonic()
        try:
            resp = self.session.get(url, timeout=config.REQUEST_TIMEOUT_SECONDS)
        except requests.RequestException:
            return 0, ""
        # Second binary guard: the URL may look like HTML but serve a PDF.
        if "text/html" not in resp.headers.get("Content-Type", ""):
            return resp.status_code, ""
        return resp.status_code, resp.text if resp.ok else ""

    def crawl(self, seeds: list[str], max_pages: int) -> list[dict]:
        """Crawl from the seeds and return cleaned page records.

        Args:
            seeds: Starting URLs (each must be on an allowed site).
            max_pages: Page cap per site.

        Returns:
            Page records (url, site, title, text, word_count, injection_flags).

        Side effects:
            Network requests; appends every attempt to crawl_log.jsonl.
        """
        # Priority queue of (-relevance, order, url, depth): most relevant first, then
        # discovery order. Seeds get a large score so they are always fetched first.
        queue: list[tuple[int, int, str, int]] = []
        order = itertools.count()
        for seed in seeds:
            if is_allowed_url(seed):
                heapq.heappush(queue, (-1000, next(order), normalize_url(seed), 0))
        seen = {item[2] for item in queue}
        pages, per_site = [], {}
        refusals: dict[str, int] = {}   # consecutive refusals per site
        self.blocked_sites: set[str] = set()
        while queue:
            _, _, url, depth = heapq.heappop(queue)
            site = site_of(url)
            if per_site.get(site, 0) >= max_pages or site in self.blocked_sites:
                continue
            log = {"time": datetime.now(timezone.utc).isoformat(), "url": url, "site": site,
                   "depth": depth}
            if not self.allowed_by_robots(url):
                append_jsonl(config.CRAWL_LOG_FILE, {**log, "status": "robots_disallowed"})
                continue
            status, html = self.fetch(url)
            append_jsonl(config.CRAWL_LOG_FILE, {**log, "status": status})
            if status in (401, 403, 429):
                refusals[site] = refusals.get(site, 0) + 1
                if refusals[site] >= config.MAX_CONSECUTIVE_REFUSALS:
                    self.blocked_sites.add(site)
                    append_jsonl(config.CRAWL_LOG_FILE, {**log, "status": "site_blocked"})
                    print(f"  [{site}] refused {refusals[site]} times in a row "
                          f"(HTTP {status}); skipping the rest of this site")
                continue
            refusals[site] = 0
            if not html:
                continue
            title, text = extract_text(html)
            words = len(text.split())
            if words >= 80:  # skip near-empty shells (JS-only pages, redirects)
                per_site[site] = per_site.get(site, 0) + 1
                pages.append({"url": url, "site": site, "title": title, "text": text,
                              "word_count": words,
                              "injection_flags": scan_for_injection(text)})
                print(f"  [{site}] {words:5d} words  {title[:70]}")
            if depth >= MAX_DEPTH[site]:
                continue
            for link in BeautifulSoup(html, "html.parser").find_all("a", href=True):
                target = normalize_url(urljoin(url, link["href"]))
                if target not in seen and wanted_link(target, site):
                    seen.add(target)
                    score = relevance(target, link.get_text(" ", strip=True))
                    heapq.heappush(queue, (-score, next(order), target, depth + 1))
        return pages


def build_corpus(seeds: list[str] | None = None, max_pages: int | None = None) -> dict:
    """Crawl, chunk and save the corpus.

    Args:
        seeds: Override seed URLs (defaults to config.SEED_URLS).
        max_pages: Override the per-site page cap.

    Returns:
        Summary dict: pages and chunks per site, flagged chunk count.

    Side effects:
        Network requests; overwrites pages.jsonl and chunks.jsonl; appends crawl_log.jsonl.
    """
    ensure_dir(config.CORPUS_DIR)
    crawler = Crawler()
    pages = crawler.crawl(seeds or config.SEED_URLS, max_pages or config.MAX_PAGES_PER_SITE)
    chunks = [c for i, page in enumerate(pages) for c in chunk_page(page, i)]
    write_text(config.PAGES_FILE, "".join(json.dumps(p) + "\n" for p in pages))
    write_text(config.CHUNKS_FILE, "".join(json.dumps(c) + "\n" for c in chunks))
    summary = {"pages": {}, "chunks": {}, "flagged_chunks": 0,
               "blocked_sites": sorted(crawler.blocked_sites)}
    for page in pages:
        summary["pages"][page["site"]] = summary["pages"].get(page["site"], 0) + 1
    for chunk in chunks:
        summary["chunks"][chunk["site"]] = summary["chunks"].get(chunk["site"], 0) + 1
        summary["flagged_chunks"] += bool(chunk["injection_flags"])
    return summary
