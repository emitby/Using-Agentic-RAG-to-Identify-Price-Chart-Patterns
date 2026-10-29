#!/usr/bin/env bash
# Run a project command with every tool cache and temp folder kept inside the project.
#
# Python code here may not use `import os` (project rule), so environment variables are
# set by this wrapper instead:
#   MPLCONFIGDIR       matplotlib's font cache (default: ~/.matplotlib)
#   PIP_CACHE_DIR      pip's download cache (default: %LOCALAPPDATA%\pip)
#   TMP/TEMP/TMPDIR    Python tempfile and pytest temp dirs (default: system temp)
#   PYTHONPATH         makes `import rag_charts` work from scripts and tests
#
# Usage:
#   ./run.sh pytest [args]           run the offline test suite
#   ./run.sh scripts/<name>.py ...   run a project script
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$ROOT/.cache/matplotlib" "$ROOT/.cache/pip" "$ROOT/.cache/tmp"
export MPLCONFIGDIR="$ROOT/.cache/matplotlib"
export PIP_CACHE_DIR="$ROOT/.cache/pip"
export TMP="$ROOT/.cache/tmp" TEMP="$ROOT/.cache/tmp" TMPDIR="$ROOT/.cache/tmp"
export PYTHONPATH="$ROOT/src"
export PYTHONUTF8=1
PY="$ROOT/.venv/Scripts/python.exe"
if [ "${1:-}" = "pytest" ]; then
  shift
  exec "$PY" -m pytest "$@"
fi
exec "$PY" "$@"
