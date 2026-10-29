"""Tests for the Claude Code guard hook (.claude/hooks/guard_paths.py).

Each test feeds the hook a payload shaped like the JSON Claude Code sends and checks the
decision. The subprocess test at the end runs the real script to confirm exit codes, since
exit code 2 is what actually blocks a tool call.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

HOOK_PATH = Path(__file__).resolve().parents[1] / ".claude" / "hooks" / "guard_paths.py"
sys.path.insert(0, str(HOOK_PATH.parent))
import guard_paths as hook  # noqa: E402  (import after path setup)

ROOT = hook.PROJECT_ROOT
OUTSIDE = ROOT.parent / "some-other-project" / "file.txt"


def payload(tool: str, **tool_input) -> dict:
    """Build a minimal PreToolUse payload with the project root as cwd."""
    return {"tool_name": tool, "tool_input": tool_input, "cwd": str(ROOT)}


# --- file tools -------------------------------------------------------------
@pytest.mark.parametrize("tool", ["Write", "Edit", "MultiEdit"])
def test_file_tool_inside_project_allowed(tool):
    assert hook.evaluate(payload(tool, file_path=str(ROOT / "src" / "x.py"))) is None


def test_relative_path_inside_allowed():
    assert hook.evaluate(payload("Write", file_path="outputs/chart.png")) is None


@pytest.mark.parametrize("tool", ["Write", "Edit", "MultiEdit"])
def test_file_tool_outside_project_blocked(tool):
    assert "write blocked" in hook.evaluate(payload(tool, file_path=str(OUTSIDE)))


def test_parent_hop_escape_blocked():
    assert hook.evaluate(payload("Write", file_path="../escape.txt")) is not None


def test_notebook_outside_blocked():
    assert hook.evaluate(payload("NotebookEdit", notebook_path=str(OUTSIDE))) is not None


def test_home_memory_dir_blocked():
    reason = hook.evaluate(payload("Write", file_path="~/.claude/memory/note.md"))
    assert reason is not None


# --- Bash deletes (blocked everywhere) -------------------------------------
@pytest.mark.parametrize("command", [
    "rm outputs/old.png",
    "rm -rf .cache",
    "rmdir data",
    "del notes.txt",
    "powershell Remove-Item foo.txt",
    "find . -name '*.tmp' -delete",
    "git clean -fd",
    "git reset --hard HEAD",
    "py -c \"import shutil; shutil.rmtree('data')\"",
    "py -c \"from pathlib import Path; Path('a').unlink()\"",
])
def test_deletes_blocked_even_inside(command):
    assert "delete blocked" in hook.evaluate(payload("Bash", command=command))


# --- Bash writes ------------------------------------------------------------
@pytest.mark.parametrize("command", [
    "echo hi > /c/Users/someone/file.txt",
    "cp data.csv C:/Users/someone/Desktop/",
    "mv report.pdf ../",
    "mkdir ~/newdir",
    "echo x | tee /tmp/out.txt",
    "curl -o $HOME/file https://example.com",
])
def test_writes_outside_blocked(command):
    assert "write blocked" in hook.evaluate(payload("Bash", command=command))


@pytest.mark.parametrize("command", [
    "ls -la",
    "./run.sh pytest",
    "echo hi > outputs/note.txt",
    "mkdir -p outputs/charts",
    f"cp data/a.csv {ROOT.as_posix()}/outputs/",
    "git status 2>&1",
    "grep -r pattern src 2>/dev/null",
])
def test_safe_commands_allowed(command):
    assert hook.evaluate(payload("Bash", command=command)) is None


def test_other_tools_ignored():
    assert hook.evaluate(payload("Read", file_path=str(OUTSIDE))) is None


# --- real process: exit codes ----------------------------------------------
def run_hook(data) -> subprocess.CompletedProcess:
    """Run the hook script as Claude Code would, returning the finished process."""
    text = data if isinstance(data, str) else json.dumps(data)
    return subprocess.run([sys.executable, str(HOOK_PATH)], input=text,
                          capture_output=True, text=True, timeout=30)


def test_process_exit_codes():
    assert run_hook(payload("Write", file_path="outputs/x.txt")).returncode == 0
    blocked = run_hook(payload("Write", file_path=str(OUTSIDE)))
    assert blocked.returncode == 2
    assert "write blocked" in blocked.stderr


def test_unparseable_payload_fails_closed():
    assert run_hook("not json").returncode == 2
