"""Quick reachability probe: fetch each seed site once and report status and text size.

Run before a full crawl to see which sites serve readable HTML to our honest user agent
(some sites block non-browser clients or render everything with JavaScript).

Usage:
    ./run.sh scripts/probe_sites.py
"""

from rag_charts.corpus import Crawler, extract_text

PROBES = [
    "https://chartschool.stockcharts.com/",
    "https://thepatternsite.com/chartpatterns.html",
    "https://www.investopedia.com/terms/d/doubletop.asp",
]


def main() -> None:
    """Fetch each probe URL and print status, robots decision, and extracted word count."""
    crawler = Crawler()
    for url in PROBES:
        robots_ok = crawler.allowed_by_robots(url)
        status, html = crawler.fetch(url) if robots_ok else (None, "")
        title, text = extract_text(html) if html else ("", "")
        print(f"{url}\n  robots ok={robots_ok} status={status} "
              f"html={len(html)} chars words={len(text.split())} title={title[:60]!r}")


if __name__ == "__main__":
    main()
