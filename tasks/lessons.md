# Lessons

Patterns learned from corrections and bugs. Review at session start.

## Environment
- Use `py` (Python 3.12) on this machine; `python` is the Microsoft Store stub.
- Run everything through `./run.sh`: it points pip, yfinance, matplotlib and temp dirs at `.cache/`
  (Python here may not `import os`, so env vars are set by the wrapper).
- pytest runs with `-p no:tmpdir` because its temp dirs delete old runs; tests write to
  `.cache/test_outputs/<timestamp>/` instead.

## Editing
- Multi-line Python edits via a bash heredoc + `str.replace` mangled `\n` escapes inside string
  literals (they became real newlines). For code containing escapes, use the Edit/Write tools or
  splice in a file written with Write.
- After a scripted edit, `compile()` the file before writing it back.

## Detector
- Never fit a formation through the provisional (unconfirmed) last pivot; it is a running extreme,
  not a peak.
- Formation fits must reject crossing lines and large residuals, retrying with fewer pivots; an old
  starting low once dragged a triangle line through the pattern (caught by the Chart Analyst in evals).
- Trendlines need a minimum slope; near-flat lines are horizontal levels.

## Retrieval
- Plural folding must cover short words ("tops" -> "top"); weight page titles; strip site-name
  suffixes from titles.
- Retrieval eval labels should accept the right page from either site.

## Rendering
- Escape `$` in matplotlib text (`\$`), or two prices in one string render as math italics.
- Lay out multi-line note panels by counting wrapped lines.
- Weekly bars must be dated by their last actual trading day, not the week-ending Friday.

## Costs
- Estimated tool-result cost is already inside later calls' input cost; never add it to
  call-attributed cost.
