# Agentic RAG Chart-Pattern Prototype

Pulls daily prices for any ticker, charts the last 10 years (weekly bars) and 3 years (daily bars),
and annotates each chart with support/resistance, trendlines and chart patterns. Every note cites
StockCharts ChartSchool or ThePatternSite, retrieved through a RAG corpus, and each chart gets a
bullish/bearish verdict.

**Educational prototype. Not financial advice.**

## How it works

- **Deterministic code computes every number**: swing pivots, levels, trendlines,
  double/triple tops and bottoms, head and shoulders, triangles/wedges/channels, and the
  SMA trend. See `src/rag_charts/patterns.py`.
- **Three Claude agents (Opus 5.5) do the judgment work.** See `src/rag_charts/agents.py`.
  - **Researcher**: finds each pattern's definition in the corpus, with verbatim quotes and citations.
  - **Chart Analyst**: picks which candidates to draw, writes their labels and descriptions, and drafts the verdict.
  - **Verifier**: re-checks every item against the data and the cited text, approves, revises or rejects it, and gives the final verdict.
- **Code-enforced grounding**:
  - citations must exist and have been retrieved in the run
  - quotes must be verbatim
  - every `$` amount must match the candidate's data
  - an invalid answer gets one fix attempt, then the run stops

The PDF has the workflow diagram, both annotated charts, the guardrails, the evals and the
costs: `outputs/Agentic_RAG_Overview.pdf`.

## Commands

Always go through `./run.sh`. It keeps the pip, yfinance and matplotlib caches and the
temp files inside `.cache/`.

| Step | Command | Cost |
|---|---|---|
| Plain 10y and 3y close charts for any ticker(s) | `./run.sh scripts/plot_prices.py SPY QQQ` | free |
| Build the RAG corpus (crawl about 4 min) | `./run.sh scripts/build_corpus.py` | free |
| Full pipeline and annotated charts | `./run.sh scripts/run_pipeline.py SPY` | ~$0.70 |
| Redraw charts from the last run | `./run.sh scripts/redraw_charts.py` | free |
| Offline test suite (88 tests) | `./run.sh pytest` | free |
| Evals, free parts only | `./run.sh evals/run_evals.py --offline` | free |
| Evals, all | `./run.sh evals/run_evals.py` | ~$1.20 |
| Token cost report | `./run.sh scripts/cost_report.py` | free |
| PDF overview | `./run.sh scripts/make_pdf.py` | free |

The pipeline and the evals need `ANTHROPIC_API_KEY`. Each run stops before its next API
call once it reaches its budget: $6 per pipeline run and $8 per eval run, set in
`config.py`.

## Guardrails

- **Claude Code hook** (`.claude/settings.json` → `.claude/hooks/guard_paths.py`):
  - blocks file writes outside this folder
  - blocks every delete command, inside or outside
  - blocks Bash writes that mention outside paths
  - fails closed (blocks the call if it can't read it)

  Pattern matching on shell commands is a safety net, not a sandbox.
- **Application**:
  - every write goes through `guardrails.safe_path`
  - HTTPS allowlist of the three sites, robots.txt, rate limiting, no PDFs
  - prompt-injection scan of every chunk, and retrieved text is wrapped as untrusted data
  - strict JSON schemas
  - cost budget per run
  - refusal fallback

## Token cost tracking

Every API call goes to `logs/token_usage.jsonl` with exact token counts, cost, agent, turn,
and the tools it requested. Every tool execution goes to `logs/tool_calls.jsonl`.
`scripts/cost_report.py` writes per-run, per-agent, per-tool and per-model CSVs plus
`outputs/cost_breakdown.png`.

## Known limits

- Investopedia returns HTTP 403 to the crawler. The crawler logs it and skips the site;
  it does not get around the block. The corpus therefore covers two of the three sites.
- The detector reads closes only (no candlestick or volume rules), with simple documented thresholds.
