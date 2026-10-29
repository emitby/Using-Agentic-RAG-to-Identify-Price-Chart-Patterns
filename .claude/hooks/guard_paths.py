"""Claude Code PreToolUse hook: block deletes everywhere and writes outside the project.

Claude Code runs this script before every Write, Edit, MultiEdit, NotebookEdit and Bash
tool call, passing the call as JSON on stdin. Exit code 0 lets the call run; exit code 2
blocks it and shows the stderr message to Claude, which is how Claude Code hooks signal a
denial.

Rules (from the project CLAUDE.md and the user's decision on 2026-10-29):
  * File tools (Write/Edit/MultiEdit/NotebookEdit) may only target paths inside the project.
  * Bash commands that delete or discard files are blocked everywhere, inside or outside.
  * Bash commands that write (redirections, cp, mv, mkdir, tee, ...) are blocked when the
    command mentions any path outside the project.

Limits, stated honestly: a shell is a full programming language, so pattern matching on a
command string is a safety net, not a sandbox. Obfuscated commands (variables holding
paths, base64-decoded scripts) can slip past. The design leans toward false positives:
when in doubt it blocks, and the user can run the command themselves.

Only the standard library is used, and `os` is not imported (project rule); `pathlib`
covers everything needed.
"""

import json
import re
import sys
from pathlib import Path

# This file lives at <project>/.claude/hooks/guard_paths.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

FILE_TOOLS = {"Write", "Edit", "MultiEdit", "NotebookEdit"}

# Commands or code that delete or discard files. Matched anywhere in the command, so a
# Python one-liner calling os.remove is caught as well as a plain `rm`.
DELETE_PATTERNS = [
    r"\brm\b", r"\brmdir\b", r"\bdel\b", r"\berase\b", r"\brd\b", r"\bunlink\b",
    r"\bshred\b", r"Remove-Item", r"Clear-Content", r"\brmtree\b", r"os\.remove",
    r"\.unlink\s*\(", r"\bfind\b[^|;&]*\s-delete\b", r"\bgit\s+clean\b", r"\bgit\s+rm\b",
    r"\bgit\s+reset\s+--hard\b", r"\bgit\s+checkout\s+--\s", r"\bgit\s+restore\b",
]
_DELETE_RE = re.compile("|".join(DELETE_PATTERNS), re.IGNORECASE)

# Words that mean the command can create or modify files. Paired with an outside path,
# any of these causes a block.
WRITE_WORDS = [
    r"\bcp\b", r"\bmv\b", r"\btouch\b", r"\bmkdir\b", r"\btee\b", r"\binstall\b",
    r"\bln\b", r"\bdd\b", r"\brsync\b", r"\bcurl\b", r"\bwget\b", r"\btruncate\b",
    r"\bsed\s+-i", r"\bperl\s+-i", r"\btar\b", r"\bunzip\b", r"\bchmod\b",
    r"Out-File", r"Set-Content", r"Add-Content", r"New-Item", r"Copy-Item",
    r"Move-Item", r"Rename-Item", r"\bpy(thon3?)?(\.exe)?\b", r"\bpip\b", r"\bgit\b",
]
_WRITE_RE = re.compile("|".join(WRITE_WORDS), re.IGNORECASE)

# Output redirection (`>` / `>>`), excluding `2>&1`-style file-descriptor duplication.
_REDIRECT_RE = re.compile(r"(?<![0-9&])>>?(?!&)|[0-9]>>?(?!&)")

# Substrings that look like filesystem paths: Windows drive paths, Git Bash drive paths
# (/c/...), home-relative paths, parent-directory hops, and common shell temp/home
# variables.
_PATH_RE = re.compile(
    r"""(?:[A-Za-z]:[\\/][^\s'"`;|&<>()]*)"""
    r"""|(?:(?<![\w.])/[a-zA-Z]/[^\s'"`;|&<>()]*)"""
    r"""|(?:(?<![\w.])/(?:tmp|etc|usr|var|home|Users|mnt)\b[^\s'"`;|&<>()]*)"""
    r"""|(?:~[^\s'"`;|&<>()]*)"""
    r"""|(?:\.\.[\\/][^\s'"`;|&<>()]*)"""
    r"""|(?:\$HOME\b|\$TMP\b|\$TEMP\b|\$TMPDIR\b|%USERPROFILE%|%TEMP%|%TMP%|%APPDATA%|\$env:\w+)"""
)

# Paths that are always safe to mention.
SAFE_PATHS = {"/dev/null", "nul", "NUL"}


def to_path(raw: str, cwd: Path) -> Path | None:
    """Turn a path-like string from a tool call into an absolute `Path`.

    Handles Git Bash drive paths (`/c/Users/...`), `~`, and paths relative to `cwd`.
    Shell variables (`$HOME`, `%TEMP%`, ...) cannot be expanded safely here, so they
    return None, which callers treat as "outside the project".

    Args:
        raw: The path text.
        cwd: Directory that relative paths are relative to.

    Returns:
        The resolved absolute path, or None if it cannot be determined.
    """
    text = raw.strip().strip("'\"")
    if text.startswith(("$", "%")):
        return None
    drive = re.match(r"^/([a-zA-Z])(/.*)?$", text)
    if drive:
        text = f"{drive.group(1).upper()}:{drive.group(2) or '/'}"
    if text.startswith("~"):
        text = str(Path.home()) + text[1:]
    candidate = Path(text)
    if not candidate.is_absolute():
        candidate = cwd / candidate
    try:
        return candidate.resolve()
    except (OSError, RuntimeError):
        return None


def is_inside(path: Path | None) -> bool:
    """Return True if `path` is the project root or inside it (case-insensitive on Windows).

    Args:
        path: An absolute path, or None (unknown, treated as outside).

    Returns:
        Whether the path is inside the project.
    """
    if path is None:
        return False
    root = str(PROJECT_ROOT.resolve()).replace("\\", "/").rstrip("/").lower()
    target = str(path).replace("\\", "/").rstrip("/").lower()
    return target == root or target.startswith(root + "/")


def check_file_tool(tool_input: dict, cwd: Path) -> str | None:
    """Check a Write/Edit/MultiEdit/NotebookEdit call.

    Args:
        tool_input: The tool's input dict (uses `file_path` or `notebook_path`).
        cwd: The session's current directory, for relative paths.

    Returns:
        A denial reason, or None if the call is allowed.
    """
    raw = tool_input.get("file_path") or tool_input.get("notebook_path") or ""
    target = to_path(raw, cwd)
    if not is_inside(target):
        return f"write blocked: '{raw}' is outside the project folder {PROJECT_ROOT}"
    return None


def check_bash(command: str, cwd: Path) -> str | None:
    """Check a Bash command for deletes (anywhere) and writes that touch outside paths.

    Args:
        command: The full command string.
        cwd: The session's current directory, for relative paths.

    Returns:
        A denial reason, or None if the command is allowed.
    """
    deletion = _DELETE_RE.search(command)
    if deletion:
        return (f"delete blocked: '{deletion.group(0)}' found. Deleting files is not "
                f"allowed in this project, inside or outside the folder.")

    outside = []
    for match in _PATH_RE.finditer(command):
        raw = match.group(0)
        if raw in SAFE_PATHS:
            continue
        if not is_inside(to_path(raw, cwd)):
            outside.append(raw)

    if outside and (_WRITE_RE.search(command) or _REDIRECT_RE.search(command)):
        return (f"write blocked: the command can write files and mentions paths outside "
                f"the project: {', '.join(outside[:3])}")
    return None


def evaluate(payload: dict) -> str | None:
    """Decide whether a PreToolUse payload should be blocked.

    Args:
        payload: The JSON object Claude Code sends on stdin (`tool_name`, `tool_input`,
            `cwd`, ...).

    Returns:
        A denial reason, or None to allow.
    """
    tool = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    # Relative paths in a tool call are relative to the session's cwd; fall back to the
    # project root if the payload does not say.
    cwd = to_path(payload.get("cwd") or str(PROJECT_ROOT), PROJECT_ROOT) or PROJECT_ROOT
    if tool in FILE_TOOLS:
        return check_file_tool(tool_input, cwd)
    if tool == "Bash":
        return check_bash(tool_input.get("command", ""), cwd)
    return None


def main() -> int:
    """Read the hook payload from stdin, print a reason on stderr if blocked.

    Returns:
        0 to allow the tool call, 2 to block it. A payload that cannot be parsed is
        blocked (fail closed): an unreadable request is not provably safe.
    """
    try:
        payload = json.loads(sys.stdin.read() or "{}")
    except json.JSONDecodeError:
        print("guard hook: could not parse tool call; blocking to be safe", file=sys.stderr)
        return 2
    reason = evaluate(payload)
    if reason:
        print(reason, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
