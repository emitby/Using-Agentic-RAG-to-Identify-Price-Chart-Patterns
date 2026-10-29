"""Runtime guardrails shared by every module.

Four kinds of protection live here:

1. **Path sandbox.** Every file the application writes goes through `safe_path`, which
   refuses any path that resolves outside the project folder (including via `..` or
   symlinks). The write helpers below are the only sanctioned way to write files.
2. **URL allowlist.** The crawler may only fetch HTTPS pages on the three approved sites,
   and never PDFs or other binary files.
3. **Prompt-injection hygiene.** Web text is untrusted data. It is scanned for
   instruction-like phrases and always handed to the model wrapped in tags that mark it as
   reference material, never as instructions.
4. **Output grounding checks** live in `agents.py` (they need the corpus and detection
   report), but raise the `GuardrailError` defined here.

There is deliberately no delete helper anywhere in the project: the user's rules forbid
deleting files.
"""

import json
import re
from pathlib import Path
from urllib.parse import urlparse

from rag_charts import config


class GuardrailError(Exception):
    """Raised when an action would violate a project guardrail.

    The message always says which rule was hit so the caller (or a log reader) can tell a
    guardrail stop apart from an ordinary bug.
    """


# ---------------------------------------------------------------------------
# Path sandbox
# ---------------------------------------------------------------------------
def safe_path(path: str | Path) -> Path:
    """Resolve `path` and confirm it lies inside the project folder.

    Relative paths are interpreted relative to the project root (not the process's working
    directory) so scripts behave the same wherever they are launched from. Resolution
    follows symlinks and collapses `..`, so neither can be used to escape.

    Args:
        path: A file or directory path, absolute or relative to the project root.

    Returns:
        The fully resolved absolute `Path`.

    Raises:
        GuardrailError: If the resolved path is outside the project root.
    """
    candidate = Path(path)
    if not candidate.is_absolute():
        candidate = config.PROJECT_ROOT / candidate
    resolved = candidate.resolve()
    root = config.PROJECT_ROOT.resolve()
    if resolved != root and root not in resolved.parents:
        raise GuardrailError(f"path sandbox: refusing to touch '{resolved}' outside '{root}'")
    return resolved


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) inside the project if it does not exist.

    Args:
        path: Directory path, absolute or relative to the project root.

    Returns:
        The resolved directory path.

    Raises:
        GuardrailError: If the directory would be outside the project.

    Side effects:
        Creates directories on disk.
    """
    target = safe_path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def write_text(path: str | Path, text: str) -> Path:
    """Write a UTF-8 text file inside the project, creating parent folders as needed.

    Args:
        path: Destination file path.
        text: Full file contents (overwrites any existing file).

    Returns:
        The resolved path written.

    Raises:
        GuardrailError: If the destination is outside the project.
    """
    target = safe_path(path)
    ensure_dir(target.parent)
    target.write_text(text, encoding="utf-8")
    return target


def write_json(path: str | Path, data) -> Path:
    """Write `data` as pretty-printed JSON inside the project.

    Args:
        path: Destination file path.
        data: Any JSON-serializable object. Dates and other objects fall back to `str`.

    Returns:
        The resolved path written.

    Raises:
        GuardrailError: If the destination is outside the project.
    """
    return write_text(path, json.dumps(data, indent=2, default=str))


def append_jsonl(path: str | Path, record: dict) -> Path:
    """Append one JSON record as a line to a JSON Lines file inside the project.

    Append-only logs (token usage, crawl log) use this so history is never rewritten.

    Args:
        path: Destination `.jsonl` file.
        record: JSON-serializable dict.

    Returns:
        The resolved path written.

    Raises:
        GuardrailError: If the destination is outside the project.
    """
    target = safe_path(path)
    ensure_dir(target.parent)
    with target.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    return target


def read_jsonl(path: str | Path) -> list[dict]:
    """Read a JSON Lines file inside the project; a missing file reads as empty.

    Args:
        path: Source `.jsonl` file.

    Returns:
        One dict per non-blank line.

    Raises:
        GuardrailError: If the path is outside the project.
    """
    target = safe_path(path)
    if not target.exists():
        return []
    with target.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---------------------------------------------------------------------------
# URL allowlist
# ---------------------------------------------------------------------------
# File types the crawler must never download. PDFs are banned by the user's rules; the
# rest are binaries that would be useless to a text retriever anyway.
BINARY_EXTENSIONS = (
    ".pdf", ".zip", ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".mp4", ".mp3",
    ".xls", ".xlsx", ".doc", ".docx", ".ppt", ".pptx", ".exe", ".dmg", ".gz", ".ico",
)


def site_of(url: str) -> str | None:
    """Return the short site name for an allowed URL, or None if the host is not allowed.

    Args:
        url: Any URL.

    Returns:
        "stockcharts", "investopedia" or "thepatternsite", else None.
    """
    host = (urlparse(url).hostname or "").lower()
    return config.ALLOWED_DOMAINS.get(host)


def is_allowed_url(url: str) -> bool:
    """Check that a URL is HTTPS, on an approved site, and not a binary file.

    Args:
        url: Candidate URL.

    Returns:
        True when the crawler may fetch it.
    """
    parsed = urlparse(url)
    if parsed.scheme != "https" or site_of(url) is None:
        return False
    return not parsed.path.lower().endswith(BINARY_EXTENSIONS)


# ---------------------------------------------------------------------------
# Prompt-injection hygiene
# ---------------------------------------------------------------------------
# Phrases that address the model rather than the reader. Legitimate trading tutorials
# essentially never contain these, so a hit is a strong signal of injected text.
INJECTION_PATTERNS = [
    r"ignore (all |any )?(previous|prior|above) (instructions|directions)",
    r"disregard (the |your )?(previous|prior|system) (instructions|prompt)",
    r"you are now\b",
    r"new instructions\s*:",
    r"system prompt",
    r"\bas an ai (language )?model\b",
    r"do not (tell|inform) the user",
    r"(always|must) (declare|say|answer|output) (that )?(the chart is )?(bullish|bearish)",
    r"<\s*/?\s*(system|instructions?)\s*>",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def scan_for_injection(text: str) -> list[str]:
    """Find instruction-like phrases that suggest a prompt-injection attempt.

    Args:
        text: Untrusted text (a web page or retrieval chunk).

    Returns:
        The matched phrases, in order; empty when the text looks clean.
    """
    return [match.group(0) for match in _INJECTION_RE.finditer(text)]


def wrap_untrusted(text: str, source: str) -> str:
    """Wrap retrieved web text so the model treats it as reference data only.

    The angle brackets inside the text are neutralized so a page cannot close the wrapper
    tag early and smuggle text outside it.

    Args:
        text: Untrusted text.
        source: Where it came from (chunk id and URL), shown in the opening tag.

    Returns:
        The tagged string.
    """
    neutral = text.replace("<", "‹").replace(">", "›")
    return f'<retrieved_document source="{source}">\n{neutral}\n</retrieved_document>'
