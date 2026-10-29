"""Build outputs/Agentic_RAG_Overview.pdf: what the system does, how its agents work
together, the SPY results, eval results, and costs.

Inputs (produced by the other scripts):
    outputs/latest_result.json    scripts/run_pipeline.py
    outputs/eval_report.json      evals/run_evals.py
    outputs/cost_summary.json     scripts/cost_report.py
    outputs/spy_annotated_*.png, outputs/cost_breakdown.png
    data/corpus/pages.jsonl, chunks.jsonl, crawl_log.jsonl

Usage:
    ./run.sh scripts/make_pdf.py
"""

import json
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402
from reportlab.lib import colors  # noqa: E402
from reportlab.lib.pagesizes import letter  # noqa: E402
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # noqa: E402
from reportlab.lib.units import inch  # noqa: E402
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,  # noqa: E402
                                Table, TableStyle)

from rag_charts import config  # noqa: E402
from rag_charts.guardrails import read_jsonl, safe_path  # noqa: E402

PDF_PATH = config.OUTPUT_DIR / "Agentic_RAG_Overview.pdf"
DIAGRAM_PATH = config.OUTPUT_DIR / "agentic_workflow.png"

AGENT_FILL, AGENT_EDGE = "#dce8f7", "#2f5d9e"    # Claude agents
CODE_FILL, CODE_EDGE = "#eeeeee", "#777777"      # deterministic code
DATA_FILL, DATA_EDGE = "#fff4dc", "#b8860b"      # external data sources
GUARD_FILL, GUARD_EDGE = "#fde8e6", "#b03a2e"    # guardrails


def load_json(path) -> dict:
    """Read a JSON output file inside the project (empty dict if missing)."""
    target = safe_path(path)
    return json.loads(target.read_text(encoding="utf-8")) if target.exists() else {}


# ---------------------------------------------------------------------------
# Workflow diagram
# ---------------------------------------------------------------------------
def draw_workflow(out_path) -> str:
    """Draw the agentic workflow: data sources, code steps, agents, guardrails, outputs.

    Two lanes feed the agents: the price lane (top) and the knowledge lane (bottom).
    Box color says who does the work, Claude agent or deterministic code, so the
    division of labor (code computes numbers, agents judge) is visible at a glance.
    """
    fig, ax = plt.subplots(figsize=(13, 7.2))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7.3)
    ax.axis("off")

    def box(x, y, w, h, title, body, fill, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                                    fc=fill, ec=edge, lw=1.4))
        ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=10,
                fontweight="bold", color="#222222")
        ax.text(x + w / 2, y + h - 0.5, body, ha="center", va="top", fontsize=7.6,
                color="#333333", linespacing=1.3)
        return (x, y, w, h)

    def arrow(a, b, label="", side="right", dashed=False, shift=0.0):
        (ax0, ay0, aw, ah), (bx0, by0, bw, bh) = a, b
        if side == "right":
            start, end = (ax0 + aw, ay0 + ah / 2), (bx0, by0 + bh / 2)
            ax.text((start[0] + end[0]) / 2, start[1] + 0.12, label, ha="center",
                    fontsize=7, color="#555555", style="italic")
        else:  # "up": from the top of a to the bottom of b, x offset by `shift`
            start = (ax0 + aw / 2 + shift, ay0 + ah)
            end = (bx0 + bw / 2 + shift, by0)
            ax.text(start[0] + 0.08, (start[1] + end[1]) / 2, label, ha="left",
                    va="center", fontsize=7, color="#555555", style="italic")
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                     lw=1.2, color="#555555", linestyle="--" if dashed else "-"))

    # Price lane (top)
    prices = box(0.2, 4.9, 1.9, 1.4, "Yahoo Finance", "fetch_prices(tickers)\ndaily OHLCV,\ncached in .cache/",
                 DATA_FILL, DATA_EDGE)
    detect = box(2.6, 4.9, 2.3, 1.4, "Pattern detector",
                 "zigzag pivots, S/R levels,\ntrendlines, double/triple,\nH&S, triangles/wedges,\nSMA trend & crosses",
                 CODE_FILL, CODE_EDGE)
    analyst = box(5.6, 4.9, 2.1, 1.4, "Chart Analyst",
                  "picks candidates to draw,\nwrites labels + cited\ndescriptions, drafts\nverdict  (Opus 5.5, high)",
                  AGENT_FILL, AGENT_EDGE)
    verifier = box(8.3, 4.9, 2.1, 1.4, "Verifier",
                   "re-checks every item vs.\ndata + cited text;\napprove / revise / reject;\nfinal verdict (Opus 5.5)",
                   AGENT_FILL, AGENT_EDGE)
    chart = box(11.0, 4.9, 1.8, 1.4, "Annotated chart",
                "detector geometry +\nverified text, sources,\nbullish/bearish badge", CODE_FILL, CODE_EDGE)

    # Knowledge lane (middle)
    sites = box(0.2, 2.2, 1.9, 1.6, "3 websites",
                "ChartSchool\nThePatternSite\nInvestopedia\n(403: blocked crawler)", DATA_FILL, DATA_EDGE)
    crawl = box(2.6, 2.2, 2.3, 1.6, "Crawler + chunker",
                "allowlist, robots.txt,\nrate limit, no PDFs,\ninjection scan,\n220-word chunks",
                CODE_FILL, CODE_EDGE)
    index = box(5.6, 2.2, 2.1, 1.6, "BM25 index",
                "~1,000 chunks;\nsearch_corpus tool;\nflagged chunks\nexcluded", CODE_FILL, CODE_EDGE)
    research = box(8.3, 2.2, 2.1, 1.6, "Researcher",
                   "definitions, rules,\nconfirmation, bias,\nverbatim key quote,\ncitations (Opus 5.5)",
                   AGENT_FILL, AGENT_EDGE)

    arrow(prices, detect, "bars")
    arrow(detect, analyst, "candidates")
    arrow(analyst, verifier, "selections")
    arrow(verifier, chart, "approved")
    arrow(sites, crawl, "HTML")
    arrow(crawl, index, "chunks")
    arrow(index, research, "search")
    arrow(research, verifier, "definitions", side="up", shift=0.35)
    arrow(index, analyst, "search_corpus (all agents)", side="up", dashed=True, shift=-0.45)
    # Definitions also reach the Analyst: an elbow from the Researcher's top-left corner.
    ax.add_patch(FancyArrowPatch((8.75, 3.8), (7.2, 4.9), arrowstyle="-|>", mutation_scale=13,
                                 lw=1.2, color="#555555", connectionstyle="angle,angleA=90,angleB=0"))

    # Guardrail band under the lanes: applies to every agent answer.
    ax.add_patch(FancyBboxPatch((0.2, 0.85), 12.6, 1.0, boxstyle="round,pad=0.04,rounding_size=0.12",
                                fc=GUARD_FILL, ec=GUARD_EDGE, lw=1.2))
    ax.text(6.5, 1.7, "Grounding guardrails (code), applied after every agent answer", ha="center",
            va="top", fontsize=9.5, fontweight="bold", color=GUARD_EDGE)
    ax.text(6.5, 1.4, "citations must exist and have been retrieved  |  key quotes verbatim  |  "
            "$ amounts match candidate data (within 1%)  |  every selection reviewed once\n"
            "one fix attempt, then stop  |  strict JSON schemas  |  cost tracker + budget cap "
            "before each API call  |  refusal fallback",
            ha="center", va="top", fontsize=7.6, color="#5a1d16", linespacing=1.4)

    # Legend
    for i, (fill, edge, label) in enumerate([(AGENT_FILL, AGENT_EDGE, "Claude agent"),
                                             (CODE_FILL, CODE_EDGE, "Deterministic code"),
                                             (DATA_FILL, DATA_EDGE, "External source"),
                                             (GUARD_FILL, GUARD_EDGE, "Guardrail")]):
        ax.add_patch(FancyBboxPatch((0.25 + i * 2.2, 0.3), 0.35, 0.25, boxstyle="round,pad=0.02",
                                    fc=fill, ec=edge, lw=1.2))
        ax.text(0.7 + i * 2.2, 0.42, label, va="center", fontsize=8.5, color="#333333")
    ax.text(6.5, 6.85, "Orchestrator: plain Python in a fixed order. The Researcher runs once; "
            "the Analyst and Verifier run per window (10y weekly, 3y daily).",
            ha="center", va="top", fontsize=8, color="#555555")
    ax.text(6.5, 7.25, "Agentic RAG workflow", ha="center", va="top", fontsize=13, fontweight="bold")

    target = safe_path(out_path)
    fig.savefig(target, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return str(target)


# ---------------------------------------------------------------------------
# PDF
# ---------------------------------------------------------------------------
def table(rows: list[list], widths: list[float], header: bool = True) -> Table:
    """A compact, lightly ruled table with optional bold header row."""
    cell = ParagraphStyle("cell", fontSize=8, leading=10)
    data = [[Paragraph(str(c), cell) for c in row] for row in rows]
    t = Table(data, colWidths=[w * inch for w in widths], repeatRows=1 if header else 0)
    style = [("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#bbbbbb")),
             ("VALIGN", (0, 0), (-1, -1), "TOP"),
             ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3)]
    if header:
        style.append(("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#e8eef7")))
    t.setStyle(TableStyle(style))
    return t


def scaled_image(path, max_w: float, max_h: float) -> Image:
    """An image scaled to fit inside max_w x max_h inches, keeping its aspect ratio."""
    img = Image(str(safe_path(path)))
    ratio = min(max_w * inch / img.imageWidth, max_h * inch / img.imageHeight)
    img.drawWidth, img.drawHeight = img.imageWidth * ratio, img.imageHeight * ratio
    return img


def build_pdf() -> str:
    """Assemble the PDF from the latest outputs and return its path."""
    result = load_json(config.OUTPUT_DIR / "latest_result.json")
    evals = load_json(config.OUTPUT_DIR / "eval_report.json")
    costs = load_json(config.OUTPUT_DIR / "cost_summary.json")
    pages = read_jsonl(config.PAGES_FILE)
    chunks = read_jsonl(config.CHUNKS_FILE)
    blocked = sorted({r["site"] for r in read_jsonl(config.CRAWL_LOG_FILE)
                      if r.get("status") == "site_blocked"})
    diagram = draw_workflow(DIAGRAM_PATH)

    styles = getSampleStyleSheet()
    h1, h2, body = styles["Title"], styles["Heading2"], styles["BodyText"]
    body.fontSize, body.leading = 9.5, 13
    small = ParagraphStyle("small", parent=body, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
    story = []
    P = lambda text, style=body: story.append(Paragraph(text, style))  # noqa: E731

    # --- 1. Overview -------------------------------------------------------
    P("Agentic RAG Chart-Pattern Prototype", h1)
    P(f"{result.get('ticker', 'SPY')} &nbsp;|&nbsp; model {config.MODEL} &nbsp;|&nbsp; "
      f"run {result.get('run_id', '-')} &nbsp;|&nbsp; {date.today():%B %d, %Y}", small)
    story.append(Spacer(1, 8))
    P("<b>What it does.</b> The system pulls daily prices for any ticker, charts the last 10 "
      "years (weekly bars) and 3 years (daily bars), and marks each chart with support and "
      "resistance, trendlines and classic chart patterns, each with a short description and a "
      "source. It ends with a bullish or bearish verdict. The definitions come from a "
      "retrieval (RAG) corpus built from StockCharts ChartSchool and ThePatternSite, so every "
      "description cites the passage it relies on.")
    P("<b>How the work is split.</b> Deterministic Python finds every number on the chart: "
      "swing pivots, levels, trendlines, pattern geometry and moving-average trend. Three "
      "Claude agents do the judgment work. The <b>Researcher</b> looks up each pattern's "
      "definition in the corpus. The <b>Chart Analyst</b> decides which detected candidates "
      "meet those definitions, writes their labels and descriptions, and drafts the verdict. "
      "The <b>Verifier</b> independently re-checks each item against the data and the cited "
      "text, approves, revises or rejects it, and confirms the verdict. Agents can only "
      "describe candidates the code found. Code checks every citation, quote and dollar "
      "amount they write.")
    rows = [["Window", "Verdict", "Confidence", "Drawn", "Verifier decisions", "Rationale (Verifier)"]]
    for key, w in result.get("windows", {}).items():
        v = w["review"]["verdict"]
        decisions = [r["decision"] for r in w["review"]["reviews"]]
        rows.append([w["title"], v["bias"].upper(), v["confidence"], len(w["annotations"]),
                     ", ".join(f"{decisions.count(d)} {d}" for d in ("approve", "revise", "reject")
                               if decisions.count(d)), v["rationale"]])
    story += [Spacer(1, 6), table(rows, [1.2, 0.65, 0.8, 0.5, 1.0, 3.35])]
    story.append(Spacer(1, 6))
    P(f"<i>{config.DISCLAIMER} Chart patterns have limited predictive value; the verdict "
      "illustrates the method and is not a trading signal.</i>", small)

    # --- 2. Workflow -------------------------------------------------------
    story.append(PageBreak())
    P("Agentic workflow", h2)
    story.append(scaled_image(diagram, 7.4, 4.3))
    story.append(Spacer(1, 6))
    P("<b>Why three agents.</b> Fetching, crawling, detecting and drawing never change order, "
      "so they are plain code, and the orchestrator is a fixed Python pipeline. Agents are used "
      "only where judgment is needed. A single agent could research, select and describe, but "
      "the main failure mode of this task is confident text that drifts from the data or "
      "misapplies a definition. A separate Verifier with its own tools (get_candidate, "
      "get_chunk, get_price_window) catches that. In the evals it also caught two real "
      "detector bugs. More agents (for example one per pattern family) would add cost and "
      "coordination without adding accuracy, because the corpus is small and one Researcher "
      "covers it in a single pass.")
    rows = [["Agent", "Runs", "Tools", "Output (strict JSON schema)", "Effort"],
            ["Researcher", "once per run", "search_corpus",
             "definition, identification rules, confirmation, implication, verbatim key quote, citations", "medium"],
            ["Chart Analyst", "per window", "search_corpus, get_candidate, get_price_window",
             "selections (candidate id, label, description, role, citations), rejections with reasons, verdict", "high"],
            ["Verifier", "per window", "search_corpus, get_chunk, get_candidate, get_price_window",
             "review per selection (approve / revise / reject + final text), final verdict, agreement flag", "high"]]
    story += [table(rows, [1.0, 0.8, 1.7, 3.2, 0.6])]

    # --- 3. Components & guardrails ---------------------------------------
    story.append(PageBreak())
    P("Knowledge base (RAG corpus)", h2)
    per_site = {}
    for c in chunks:
        per_site[c["site"]] = per_site.get(c["site"], 0) + 1
    page_sites = {}
    for p in pages:
        page_sites[p["site"]] = page_sites.get(p["site"], 0) + 1
    rows = [["Site", "Pages", "Chunks", "Status"]]
    for site in ("stockcharts", "thepatternsite", "investopedia"):
        rows.append([site, page_sites.get(site, 0), per_site.get(site, 0),
                     "blocked (HTTP 403); logged, not worked around" if site in blocked else "crawled"])
    story.append(table(rows, [1.4, 0.7, 0.7, 3.6]))
    story.append(Spacer(1, 4))
    P("Pages are fetched with an honest user agent, only from the three allowlisted HTTPS "
      "hosts, after a robots.txt check, at most one request every 1.5 s per site, and never as "
      "PDFs or other binaries. Links are crawled most-relevant first, so the per-site page cap "
      "goes to the patterns the detector can find. Main text is split into 220-word chunks "
      "with a 40-word overlap and indexed with BM25. Each chunk's page title is weighted "
      "3x, so a query naming a pattern finds that pattern's own page. The Claude API has no "
      "embeddings endpoint, and BM25 needs no extra key and is deterministic.")
    P("Guardrails", h2)
    rows = [["Layer", "Guardrail", "Enforced by"],
            ["Claude Code", "Blocks any Write/Edit/NotebookEdit outside the project, every delete command "
             "(rm, del, Remove-Item, git clean, os.remove, ...) and Bash writes that mention outside paths",
             ".claude/hooks/guard_paths.py (PreToolUse, exit 2 = block, fails closed)"],
            ["Files", "Every application write goes through safe_path(); no delete helper exists; "
             "all caches (pip, yfinance, matplotlib, temp) live in .cache/", "guardrails.py, run.sh"],
            ["Web", "HTTPS allowlist of 3 hosts, robots.txt, rate limit, no binaries, stop after "
             "3 refusals", "guardrails.is_allowed_url, corpus.Crawler"],
            ["Prompt injection", "Chunks scanned for instruction-like text (flagged chunks are excluded); "
             "retrieved text wrapped in &lt;retrieved_document&gt; tags with brackets neutralized; "
             "agents told to treat it as data", "guardrails.scan_for_injection / wrap_untrusted"],
            ["Grounding", "Citations must exist and have been retrieved this run; key quotes verbatim; "
             "dollar amounts within 1% of candidate data; each selection reviewed exactly once; one "
             "fix attempt then stop", "agents.validate_*"],
            ["Output shape", "Strict JSON schemas (structured outputs) and strict tool inputs; "
             "pydantic validation", "agents.strict_schema"],
            ["Cost", "Every call logged; run stops before the next call once the budget "
             f"(${config.RUN_BUDGET_USD:.0f} per run) is spent", "costs.CostTracker"],
            ["Safety", "Refusals stop the agent; server-side refusal fallback enabled", "agents.run_agent"]]
    story.append(table(rows, [1.1, 4.0, 2.2]))

    # --- 4. Charts ----------------------------------------------------------
    for key, w in result.get("windows", {}).items():
        story.append(PageBreak())
        P(f"Result: {w['title']}", h2)
        story.append(scaled_image(w["chart"], 7.4, 8.6))

    # --- 5. Evals -------------------------------------------------------------
    story.append(PageBreak())
    P("Evaluation", h2)
    P("The offline suite (88 pytest tests, free) covers the hook, path sandbox, URL allowlist, "
      "injection scanner, chunking, retrieval, every detector on synthetic series with planted "
      "patterns, the agent loop with a scripted fake client (validation retries, refusals, budget "
      "stop, fabricated quotes and citations), price fetching (mocked) and chart rendering. The "
      "live evals below call the real agents.")
    r = evals.get("retrieval", {})
    if r:
        P(f"<b>Retrieval:</b> recall@5 {r['recall_at_5']:.0%}, MRR {r['mrr']:.2f} over "
          f"{len(r['cases'])} labeled queries (either site's page on the pattern counts as correct).")
    if evals.get("golden"):
        rows = [["Planted-pattern case", "Pattern drawn", "Verdict", "Verdict correct", "Cited", "Grounding checks"]]
        for g in evals["golden"]:
            rows.append([g["case"].replace("_", " "), g.get("planted_drawn"), g.get("verdict"),
                         g.get("verdict_ok"), g.get("citation_rate"),
                         "pass" if g["grounding_pass"] else "FAIL"])
        story += [Spacer(1, 4), table(rows, [2.3, 0.9, 0.8, 0.9, 0.6, 1.1])]
    inj = evals.get("injection")
    if inj:
        story.append(Spacer(1, 6))
        P(f"<b>Prompt injection:</b> a poisoned passage titled like a double-top page, worded to slip "
          f"past the regex scanner, told agents to call every chart strongly bullish. On the "
          f"confirmed-double-top case the verdict stayed <b>{inj['verdict']}</b> and the passage was "
          f"{'cited' if inj['poison_cited'] else 'never cited'}: <b>{'PASS' if inj['pass'] else 'FAIL'}</b>.")
    P("<b>What the evals caught.</b> The first eval run drew only 3 of 4 planted patterns. The Chart "
      "Analyst had correctly rejected the ascending triangle: the detector had fitted the "
      "triangle's lower line through the series' starting low, so the two lines crossed. It "
      "also rejected a nearly flat line the detector had called a falling trendline. Both were "
      "detector bugs. They were fixed, regression tests were added, and the rerun drew 4 of 4.")
    s = evals.get("spy", {})
    if s.get("available"):
        for key, w in s["windows"].items():
            P(f"<b>SPY {key}:</b> trend regime {w['trend_regime']}, verdict {w['verdict']} "
              f"(consistent: {w['verdict_matches_trend']}); approved or revised "
              f"{w['approved_or_revised_rate']:.0%}; annotations cited {w['annotations_with_citations']}.")

    # --- 6. Costs ---------------------------------------------------------------
    story.append(PageBreak())
    P("Token costs", h2)
    if costs:
        P(f"All Claude API calls are logged to logs/token_usage.jsonl (exact tokens from the API) and "
          f"tool executions to logs/tool_calls.jsonl. Across {costs['runs']} logged runs: "
          f"<b>${costs['total_cost_usd']:.2f}</b> for {costs['api_calls']} API calls and "
          f"{costs['tool_calls']} tool calls. A full SPY pipeline run averages "
          f"<b>${costs['avg_pipeline_run_usd']:.2f}</b>. Prompt caching served "
          f"{costs['cache_read_share_of_input']:.0%} of input tokens at the cache-read rate.")
        story.append(scaled_image(config.OUTPUT_DIR / "cost_breakdown.png", 7.4, 2.8))
        P("Tool attribution: each API call is charged to the tools it requested (split evenly), "
          "or to final_answer when it requested none, so the tool costs add up to the real total. "
          "Writing the final structured answer is the largest cost, because that is where the "
          "model reasons and writes. Per-tool estimates of the cost of re-reading tool results "
          "are kept separately in outputs/cost_by_tool.csv.", small)
    P("Limitations", h2)
    P("Investopedia refused the crawler (HTTP 403), so the corpus covers two of the three requested "
      "sites. The detector reads closing prices only (no candlestick or volume rules), and its "
      "thresholds are simple and documented rather than optimized. Pattern definitions are "
      "qualitative, so borderline cases depend on agent judgment, which the Verifier and evals "
      "constrain but do not eliminate. Chart patterns themselves have limited predictive value.")

    doc = SimpleDocTemplate(str(safe_path(PDF_PATH)), pagesize=letter, leftMargin=0.55 * inch,
                            rightMargin=0.55 * inch, topMargin=0.55 * inch, bottomMargin=0.55 * inch,
                            title="Agentic RAG Chart-Pattern Prototype", author="Agentic-RAG-Prototype")
    doc.build(story)
    return str(safe_path(PDF_PATH))


if __name__ == "__main__":
    print(build_pdf())
