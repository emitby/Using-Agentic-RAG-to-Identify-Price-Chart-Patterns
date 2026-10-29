# Agentic RAG Chart-Pattern Prototype — Plan

Goal: pull SPY daily prices, chart 10y and 3y closes, then use a RAG system grounded in
StockCharts ChartSchool, Investopedia and ThePatternSite to draw and annotate chart patterns
(support/resistance, trendlines, named patterns) and give a cited bullish/bearish verdict.

## Design principles
- **Code finds the numbers, agents use judgment.** Pivots, support/resistance levels, trendlines
  and pattern candidates are computed by deterministic Python. Agents never invent price
  levels; they check candidates against the retrieved definitions, pick which ones to draw,
  write the annotation text, and cite sources.
- **All state stays inside the project.** venv, pip cache, yfinance cache, matplotlib cache,
  corpus, logs and outputs all live under this folder (`.cache/`, `data/`, `logs/`, `outputs/`).
- Python: `py -3.12` → project `.venv`. No `import os` (use `pathlib`).

## Recommended agents: 3 LLM agents + a code orchestrator
| # | Agent | Job | Tools |
|---|-------|-----|-------|
| 1 | Researcher | For each pattern candidate, retrieve and summarize the definition / confirmation rules / bullish-bearish implication, with citations | `search_corpus` |
| 2 | Chart Analyst | Given price stats + candidates + definitions, decide which patterns are valid, emit annotations as validated JSON (lines, zones, labels, citations) and a draft verdict | `get_price_stats`, `get_candidates`, `search_corpus` |
| 3 | Verifier | Independently check each annotation against the data (dates exist, prices in range, rules met) and the cited text; reject or fix; confirm the final verdict | `check_annotation`, `search_corpus` |

The orchestrator is plain Python (fixed pipeline: data → detect → research → analyze → verify → render),
run once per timeframe (10y weekly-resampled, 3y daily). Why not more: crawling, detection and
drawing are deterministic and belong in code; extra agents add cost and coordination failure
points without improving accuracy. Why not one: a separate verifier catches hallucinated levels
and misapplied definitions, which is the main failure mode of this task.

## Steps
- [x] 1. Scaffold: `.venv`, `requirements.txt`, layout below, `tasks/lessons.md`, `README.md`
      → verify: `py -m pytest` runs (0 tests)
- [x] 2. Guard hooks: `.claude/settings.json` + `.claude/hooks/guard_paths.py` (PreToolUse on
      Write/Edit/NotebookEdit/Bash): block any write outside the project and any delete
      (rm, rmdir, del, Remove-Item, unlink, shutil.rmtree…)
      → verify: pytest feeds the hook allowed + blocked payloads, checks exit code 2 on blocks
- [x] 3. `data.py`: `fetch_prices(tickers, years=None, start=None, end=None)` → tidy DataFrame
      (one or many tickers, yfinance, cached under `.cache/`)
      → verify: tests with a mocked download + one live SPY pull
- [x] 4. `charts.py`: SPY 10y and 3y daily close charts → `outputs/spy_close_10y.png`, `outputs/spy_close_3y.png`
      → verify: files exist, date ranges correct, visual check
- [x] 5. Corpus: polite crawler limited to the 3 domains (robots.txt, rate limit, seed URLs for
      chart-pattern / support-resistance / trendline pages, no PDFs) → clean text → chunks
      with URL + title → `data/corpus/chunks.jsonl`. If Investopedia blocks scraping, log it and continue.
      → verify: chunk counts per site, sample spot checks
- [x] 6. Retrieval: BM25 index (local, no embeddings key needed)
      → verify: retrieval eval (labeled queries → expected page in top 5)
- [x] 7. `patterns.py`: pivots, S/R clusters, trendlines/channels, double top/bottom,
      head & shoulders (+ inverse), triangles, trend regime (50/200 SMA)
      → verify: synthetic series with planted patterns are detected; clean series yield none
- [x] 8. `costs.py`: cost tracker (agent loop in `agents.py`). Every call logs run_id, agent, model,
      input/output/cache tokens, tools requested, tool-result size, $ cost →
      `logs/token_usage.jsonl`. `scripts/cost_report.py` → summary by agent/tool/model (CSV + chart).
      Per-run $ budget cap aborts the run if exceeded.
- [x] 9. Agents + orchestrator (Researcher, Analyst, Verifier) with pydantic-validated outputs
      → verify: end-to-end run on SPY produces annotation JSON for both timeframes
- [x] 10. Annotated charts: S/R lines, trendlines, pattern shapes/labels, callout text with
       source, verdict badge → `outputs/spy_annotated_10y.png`, `outputs/spy_annotated_3y.png`
- [x] 11. Guardrails (runtime): path sandbox on every write, URL allowlist (3 domains),
       retrieved text treated as data not instructions (prompt-injection test), schema + numeric
       grounding validation, required citations, budget cap, "not financial advice" note
- [x] 12. Evals: `pytest` (offline, free) + `evals/` (LLM, opt-in, costs money): pattern
       accuracy on synthetic golden charts, grounding rate, citation validity, verdict consistency
       with trend indicators, injection resistance. Results → `outputs/eval_report.md`
- [x] 13. PDF: `outputs/Agentic_RAG_Overview.pdf`: what it does, agentic workflow diagram,
       both annotated charts, guardrails, eval results, cost summary
- [x] 14. Review section below; lessons captured

## Layout
```
.claude/ settings.json, hooks/guard_paths.py
src/rag_charts/ config.py data.py charts.py patterns.py corpus.py retrieval.py
                costs.py guardrails.py agents.py pipeline.py
scripts/ build_corpus.py run_pipeline.py cost_report.py make_pdf.py
tests/   evals/   data/   logs/   outputs/   .cache/   tasks/
```

## Decisions (user, 2026-10-29)
- Model: Claude Opus 5.5 (`claude-opus-5-5`) for all three agents
- Auth: `ANTHROPIC_API_KEY` is set in the environment
- Hook delete policy: block all deletes, inside or outside the project

## Risks
- Investopedia often blocks scrapers (403); the other two sites are usually fine.
- Pattern detection is heuristic; the verifier and evals exist to keep claims honest.
- Est. cost per full run on Opus 5.5: roughly $1–3 (to be measured by the tracker).

## Review (2026-10-29)
- All 14 steps done. 88 offline tests pass. Live evals: retrieval recall@5 100% (MRR 0.93);
  4/4 golden charts drawn with correct verdicts, all grounding checks passed; injection test passed.
- SPY: both windows bullish, medium confidence. 10y flags a forming rising wedge as a bearish caution.
- Spend: $3.90 total across 2 pipeline runs and 3 eval runs; a pipeline run is ~$0.68.
- Deviations from plan: Investopedia returns 403, so it was logged and skipped rather than
  worked around. The planned `llm.py` was folded into `agents.py` + `costs.py`.
- Bugs found by the agents/evals and fixed: triangle fitted through an old starting low (lines
  crossed); near-flat line labeled a falling trendline. Bugs found by tests: plural folding
  ("tops"), provisional pivot used as a peak. Rendering: `$` math mode, note overlap, future
  weekly date.
- Hook verified with its exact configured command (exit 2 on outside write and on `rm`,
  exit 0 inside). It takes effect after Claude Code reloads project settings.
