"""claudemem.files — Fork A markdown I/O (L1, file-truth side).

Fork A markdown files are the **source of truth** (PRD G-4/C-11); ``forkA.db`` is
a rebuildable cache. This module owns the *file* side of that contract: it reads
and writes one-record-per-file frontmatter + body, parses the closed IN-1
frontmatter field set (snake_case keys), serializes the ISO-8601 ⇄ epoch
boundary (tech-design §3.2/§3.12 MF-1), derives the three alias forms (§3.11),
enumerates Fork A files for ``reindex`` (architecture §5.5), and flushes the
mutable counter / stale / superseded fields back into frontmatter (IN-4/IN-6/
IN-10/IN-14).

Module discipline (architecture §2.2 — enforced by the import-linter contract):
this module imports ``config`` and the **standard library only**. It must NOT
call a model, perform ranking, open ``forkA.db``/``forkB.db``, nor import
``index``/``store``/``recall``/``enrich``/``anthropic``. It is the file truth,
not the index cache.

**Frontmatter format ClaudeMem writes** (the form ``write_record`` emits and the
canonical form ``read_record`` round-trips): a leading ``---`` line, then flat
top-level ``snake_case: value`` lines in the deterministic IN-1 order of
:data:`_FRONTMATTER_ORDER`, an ``aliases`` JSON-array line
(``["a", "b", "c"]`` — the §3.11 canonical durable form, so commas/quotes inside
an alias round-trip exactly per SC-4), a closing ``---``, a blank line, then the
body. The parser is intentionally more permissive than the writer (AS-7
defensiveness for the importer): it prefers ``json.loads`` for the ``aliases``
value but falls back to a lenient bare inline-list parse (``[a, b]``), also
tolerates block-list aliases (``- a`` lines) and a ``metadata:``-nested block
(the alternate shape shown in ``brief.md``), but never grows a YAML dependency
(C-2, zero-dep core).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

if TYPE_CHECKING:
    from claudemem.config import ScopeContext

_log = logging.getLogger("claudemem.files")

# --------------------------------------------------------------------------- #
# IN-1 closed field set + deterministic serialization order.                    #
# --------------------------------------------------------------------------- #

#: The IN-1 frontmatter keys in the deterministic write order (tech-design §3.12
#: mapping-table order). ``aliases`` is emitted last among the scalars because it
#: is the only list-valued key. This order is what makes round-trips git-stable.
_FRONTMATTER_ORDER: tuple[str, ...] = (
    "type",
    "scope",
    "importance",
    "pinned",
    "source",
    "created",
    "last_accessed",
    "access_count",
    "hit_count",
    "summary",
    "aliases",
    "superseded_by",
    "stale",
)

# Locked IN-1 defaults for absent optional fields (AS-7 defensive read). The
# required-on-write fields (``type``/``scope``/``source``/timestamps) get
# spec-aligned fallbacks so a hand-authored minimal file never crashes the read.
_DEFAULT_TYPE = "reference"
_DEFAULT_SCOPE = "global"
_DEFAULT_IMPORTANCE = 3
_DEFAULT_SOURCE = "explicit"

_FRONTMATTER_DELIM = "---"


# --------------------------------------------------------------------------- #
# RecordFile dataclass.                                                          #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RecordFile:
    """One Fork A markdown record (frozen; mutate via :func:`dataclasses.replace`).

    Carries every IN-1 frontmatter field plus the body and the on-disk
    ``path``/``name``. ``name`` is the filename stem — the durable ``a:<name>``
    id (IN-21). Timestamps are the human-readable ISO-8601 TEXT form as they live
    in the file; the ISO ⇄ epoch boundary is crossed only via
    :func:`iso_to_epoch` / :func:`epoch_to_iso` at index time (MF-1). ``aliases``
    is the list source-of-truth (§3.11); the JSON / flat forms are derived on
    demand via :func:`aliases_json` / :func:`aliases_flat`.
    """

    path: Path
    name: str
    type: str
    scope: str
    importance: int
    pinned: bool
    source: str
    created: str
    last_accessed: str
    access_count: int
    hit_count: int
    summary: str | None
    aliases: list[str]
    superseded_by: str | None
    stale: bool
    body: str


# --------------------------------------------------------------------------- #
# ISO-8601 ⇄ epoch boundary helpers (MF-1; tech-design §3.2/§3.12).             #
# --------------------------------------------------------------------------- #


def iso_to_epoch(s: str) -> int:
    """Parse ISO-8601 TEXT → UTC epoch seconds (MF-1). Exact at second grain.

    Accepts a trailing ``Z`` (Zulu/UTC) as well as explicit offsets. A
    timezone-naive value is interpreted as UTC (the file convention is UTC).
    """
    text = s.strip()
    if text.endswith("Z") or text.endswith("z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def epoch_to_iso(n: int) -> str:
    """Format UTC epoch seconds → ISO-8601 TEXT with a ``Z`` suffix (MF-1).

    The inverse of :func:`iso_to_epoch` at second granularity: emits
    ``YYYY-MM-DDTHH:MM:SSZ`` so the written file stays human-readable and the
    round-trip ``epoch_to_iso(iso_to_epoch(x))`` is stable for any UTC ``Z`` time.
    """
    dt = datetime.fromtimestamp(n, tz=timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# Alias dual-form helpers (§3.11).                                              #
# --------------------------------------------------------------------------- #


def aliases_json(aliases: list[str]) -> str:
    """Canonical JSON-array form of the alias list (``Record.AliasesJson``).

    The durable mirror of the file list and the SC-4 round-trip target. Multi-word
    aliases survive verbatim here even though :func:`aliases_flat` tokenizes them
    apart. ``ensure_ascii=False`` keeps non-ASCII aliases human-readable.
    """
    return json.dumps(aliases, ensure_ascii=False)


def aliases_flat(aliases: list[str]) -> str:
    """Space-joined FTS-only form (``Record.AliasesFlat``); never a source of truth.

    Multi-word aliases are joined with the rest by single spaces so FTS5
    tokenizes each word independently for recall (§3.11). The list / JSON forms
    remain authoritative for reconstruction.
    """
    return " ".join(aliases)


def aliases_from_json(text: str | None) -> list[str]:
    """Reconstruct the alias list from the canonical JSON-array form (§3.11).

    Inverse of :func:`aliases_json`. A ``None``/empty value yields ``[]``. A
    non-list JSON payload is treated as no aliases (defensive, AS-7).
    """
    if not text:
        return []
    value = json.loads(text)
    if not isinstance(value, list):
        return []
    return [str(item) for item in value]


# --------------------------------------------------------------------------- #
# Frontmatter scalar parsing / formatting (minimal, stdlib-only — C-2).         #
# --------------------------------------------------------------------------- #

_TRUE_LITERALS = frozenset({"true", "yes", "on"})
_FALSE_LITERALS = frozenset({"false", "no", "off"})


def _strip_quotes(raw: str) -> str:
    """Strip a single matching pair of surrounding quotes, if present."""
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {'"', "'"}:
        return raw[1:-1]
    return raw


def _parse_scalar(raw: str) -> Any:
    """Parse a frontmatter scalar into bool / int / str (stdlib, no YAML).

    Recognizes boolean and integer literals; everything else is an unquoted or
    quoted string. Only the minimal type set ClaudeMem writes is supported (C-2,
    simplicity); floats and nested structures are out of scope for frontmatter.
    """
    text = raw.strip()
    if text == "" or text in {"null", "~"}:
        return None
    lowered = text.lower()
    if lowered in _TRUE_LITERALS:
        return True
    if lowered in _FALSE_LITERALS:
        return False
    unquoted = _strip_quotes(text)
    if unquoted == text:
        # Bare (unquoted) token — try an int literal before falling back to str.
        try:
            return int(text)
        except ValueError:
            return text
    return unquoted


def _parse_inline_list(raw: str) -> list[str]:
    """Parse an inline ``[a, b, c]`` list into a list of trimmed strings.

    Handles quoted items (``["image generation", b]``) and empty lists. This is
    the form ClaudeMem writes for ``aliases`` (matches the ``brief.md`` sample).
    """
    inner = raw.strip()[1:-1].strip()
    if not inner:
        return []
    return [_strip_quotes(item.strip()) for item in inner.split(",") if item.strip()]


def _parse_bracketed_list(raw: str) -> list[str]:
    """Parse a ``[...]`` aliases value, preferring the canonical JSON-array form.

    The writer emits ``aliases`` as a JSON array (§3.11) so commas/quotes inside
    an alias round-trip exactly (SC-4). Parse with :func:`json.loads` first; fall
    back to the lenient :func:`_parse_inline_list` for hand-authored or
    memsearch-imported files that use the bare ``[a, b]`` shape (AS-7).
    """
    try:
        value = json.loads(raw.strip())
    except ValueError:
        return _parse_inline_list(raw)
    if isinstance(value, list):
        return [str(item) for item in value]
    return _parse_inline_list(raw)


def _format_scalar(value: Any) -> str:
    """Format a Python scalar back to its frontmatter text form (write side).

    Booleans → ``true``/``false``; ints → bare digits; strings are quoted only
    when they would otherwise be ambiguous (leading/trailing space, a leading
    ``[``, or a value that would re-parse as a bool/int/null), so the common case
    stays clean and git-friendly.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    needs_quote = (
        text != text.strip()
        or text == ""
        or text.startswith("[")
        or text.lower() in _TRUE_LITERALS
        or text.lower() in _FALSE_LITERALS
        or text in {"null", "~"}
        or _is_int_literal(text)
    )
    if needs_quote:
        escaped = text.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return text


def _is_int_literal(text: str) -> bool:
    try:
        int(text)
    except ValueError:
        return False
    return True


def _split_frontmatter(content: str) -> tuple[str, str]:
    """Split raw file text into (frontmatter-block, body).

    The frontmatter is delimited by a leading ``---`` line and the next ``---``
    line. A file without a leading delimiter is treated as all-body (no
    frontmatter) — defensive for the importer (AS-7).
    """
    lines = content.splitlines()
    if not lines or lines[0].strip() != _FRONTMATTER_DELIM:
        return "", content
    for idx in range(1, len(lines)):
        if lines[idx].strip() == _FRONTMATTER_DELIM:
            fm = "\n".join(lines[1:idx])
            body = "\n".join(lines[idx + 1 :])
            # Drop a single leading blank line that separates frontmatter/body.
            if body.startswith("\n"):
                body = body[1:]
            return fm, body
    # Unterminated frontmatter: treat the remainder as frontmatter, empty body.
    return "\n".join(lines[1:]), ""


def _parse_frontmatter(block: str) -> dict[str, Any]:
    """Parse a frontmatter block into a flat ``{snake_case: value}`` dict.

    Supports the canonical flat form ClaudeMem writes and, defensively (AS-7),
    tolerates: block-list ``aliases`` (``- item`` lines under ``aliases:``), and a
    ``metadata:``-nested block whose indented keys are hoisted to the top level
    (the alternate shape in ``brief.md``). Unknown keys are kept; the caller maps
    only the IN-1 subset onto :class:`RecordFile`.
    """
    out: dict[str, Any] = {}
    lines = block.splitlines()
    idx = 0
    pending_list_key: str | None = None
    while idx < len(lines):
        line = lines[idx]
        stripped = line.strip()
        if stripped == "" or stripped.startswith("#"):
            idx += 1
            continue

        # Block-list continuation (``- item``) for the most recent list key.
        if stripped.startswith("- ") and pending_list_key is not None:
            item = _strip_quotes(stripped[2:].strip())
            existing = out.setdefault(pending_list_key, [])
            if isinstance(existing, list):
                existing.append(item)
            idx += 1
            continue
        pending_list_key = None

        if ":" not in line:
            idx += 1
            continue
        key, _, raw_value = line.partition(":")
        key = key.strip()
        raw_value = raw_value.strip()
        # Strip an inline trailing comment (``value   # note``) on scalar lines
        # that are not quoted strings. Quoted values keep ``#`` verbatim.
        raw_value = _strip_inline_comment(raw_value)

        # A ``metadata:`` (or any bare key with no value) that introduces an
        # indented block: hoist nested keys to top level on subsequent lines.
        if raw_value == "" and key == "metadata":
            idx += 1
            continue

        if raw_value.startswith("["):
            out[key] = _parse_bracketed_list(raw_value)
        elif raw_value == "":
            # Bare key — could be a block-list header (e.g. ``aliases:``).
            out[key] = []
            pending_list_key = key
        else:
            out[key] = _parse_scalar(raw_value)
        idx += 1
    return out


def _strip_inline_comment(raw: str) -> str:
    """Remove a trailing ``# comment`` from an unquoted scalar value.

    Quoted and bracketed values are left untouched (a ``#`` inside them is data).
    """
    if raw.startswith(('"', "'", "[")):
        return raw
    hash_idx = raw.find("#")
    if hash_idx == -1:
        return raw
    return raw[:hash_idx].rstrip()


# --------------------------------------------------------------------------- #
# read_record / write_record.                                                   #
# --------------------------------------------------------------------------- #


def _as_str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_LITERALS
    return False


def _as_int(value: Any, default: int) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, str) and _is_int_literal(value.strip()):
        return int(value.strip())
    return default


def read_record(path: Path) -> RecordFile:
    """Parse one Fork A markdown file into a :class:`RecordFile` (T1.1, IN-1).

    The frontmatter is parsed with the minimal stdlib parser (no YAML dep, C-2);
    missing optional fields fall back to their IN-1 defaults (AS-7 defensive).
    ``name`` is the filename stem (the ``a:<name>`` id, IN-21). Timestamps are
    kept in their on-disk ISO-8601 TEXT form (MF-1).
    """
    content = path.read_text(encoding="utf-8")
    block, body = _split_frontmatter(content)
    fm = _parse_frontmatter(block)

    raw_aliases = fm.get("aliases", [])
    aliases = (
        [str(a) for a in raw_aliases] if isinstance(raw_aliases, list) else []
    )

    return RecordFile(
        path=path,
        name=path.stem,
        type=str(fm.get("type", _DEFAULT_TYPE)),
        scope=str(fm.get("scope", _DEFAULT_SCOPE)),
        importance=_as_int(fm.get("importance"), _DEFAULT_IMPORTANCE),
        pinned=_as_bool(fm.get("pinned")),
        source=str(fm.get("source", _DEFAULT_SOURCE)),
        created=str(fm.get("created", "")),
        last_accessed=str(fm.get("last_accessed", "")),
        access_count=_as_int(fm.get("access_count"), 0),
        hit_count=_as_int(fm.get("hit_count"), 0),
        summary=_as_str_or_none(fm.get("summary")),
        aliases=aliases,
        superseded_by=_as_str_or_none(fm.get("superseded_by")),
        stale=_as_bool(fm.get("stale")),
        body=body,
    )


def _frontmatter_value(record: RecordFile, key: str) -> Any:
    """Return the raw Python value for an IN-1 frontmatter key off the record."""
    return getattr(record, key)


def render_frontmatter(record: RecordFile) -> str:
    """Render a :class:`RecordFile`'s IN-1 fields to a frontmatter block string.

    Deterministic key order (:data:`_FRONTMATTER_ORDER`) for git-stable
    round-trips. Optional fields whose value is ``None`` are omitted (so a record
    that never had a ``summary``/``superseded_by`` does not gain an empty key).
    ``aliases`` is always written as a JSON array (even when empty), the §3.11
    canonical durable form (commas/quotes inside an alias round-trip exactly).
    """
    out_lines: list[str] = [_FRONTMATTER_DELIM]
    for key in _FRONTMATTER_ORDER:
        value = _frontmatter_value(record, key)
        if key == "aliases":
            # Serialize as a JSON array (the §3.11 canonical durable form). JSON
            # is valid frontmatter here and makes commas/quotes/any special
            # character round-trip exactly (SC-4), unlike a bare inline list.
            out_lines.append(f"aliases: {json.dumps(value, ensure_ascii=False)}")
            continue
        if value is None:
            # Omit absent optional scalars (summary / superseded_by).
            continue
        out_lines.append(f"{key}: {_format_scalar(value)}")
    out_lines.append(_FRONTMATTER_DELIM)
    return "\n".join(out_lines)


def write_record(record: RecordFile) -> None:
    """Serialize a :class:`RecordFile` back to ``frontmatter + body`` on disk (T1.2).

    One record per file at ``record.path``. Deterministic frontmatter key order
    keeps round-trips byte-stable and git-friendly. The body is preserved
    verbatim with a single trailing newline. The parent directory is created if
    absent.
    """
    record.path.parent.mkdir(parents=True, exist_ok=True)
    frontmatter = render_frontmatter(record)
    body = record.body
    text = f"{frontmatter}\n\n{body}"
    if not text.endswith("\n"):
        text += "\n"
    record.path.write_text(text, encoding="utf-8")


# --------------------------------------------------------------------------- #
# iter_records (T1.3; architecture §5.5).                                       #
# --------------------------------------------------------------------------- #


def _iter_dir(directory: Path | None) -> Iterator[RecordFile]:
    """Yield every parseable ``.md`` record in ``directory`` (absent dir → none).

    Tolerant of a missing directory (yields nothing, never errors — a hand-deleted
    scope dir is simply empty per C-11/SC-4). The reserved ``MEMORY.md`` index
    file is skipped: it is Claude Code's own memory index, not a Fork A record.
    """
    if directory is None or not directory.is_dir():
        return
    for entry in sorted(directory.glob("*.md")):
        if entry.name == "MEMORY.md" or not entry.is_file():
            continue
        try:
            yield read_record(entry)
        except OSError as exc:  # file vanished mid-iteration (hand-deleted)
            _log.warning("skipping unreadable Fork A file %s: %s", entry, exc)


def iter_records(scope_ctx: ScopeContext) -> Iterator[RecordFile]:
    """Enumerate Fork A records across the scope's global + project dirs (T1.3).

    Reads the global dir (``~/.claude/memory/``) and, for project scope, the
    cwd-derived project dir, both carried on the :class:`~claudemem.config.ScopeContext`.
    A missing directory yields nothing (hand-deletion reconciliation is implicit
    in the rebuild-from-files model, architecture §5.5 / C-11 / SC-4). Order is
    global dir first, then project dir, each sorted by filename for determinism.
    """
    yield from _iter_dir(scope_ctx.global_dir)
    yield from _iter_dir(scope_ctx.project_dir)


# --------------------------------------------------------------------------- #
# Batch-writeback helpers (T1.3; IN-4/IN-6/IN-10/IN-14).                         #
# --------------------------------------------------------------------------- #


def _safe_rewrite(record: RecordFile) -> bool:
    """Re-read from disk and rewrite ``record``, skipping a hand-deleted file.

    Robust to a file removed between the original read and writeback (no crash —
    logs and skips, returns ``False``). Returns ``True`` on a successful write.
    """
    if not record.path.is_file():
        _log.warning("writeback skipped — file gone: %s", record.path)
        return False
    try:
        write_record(record)
    except OSError as exc:  # vanished between the is_file check and the write
        _log.warning("writeback skipped — write failed for %s: %s", record.path, exc)
        return False
    return True


def writeback_counters(
    record: RecordFile,
    *,
    hit_count: int | None = None,
    last_accessed: str | None = None,
    access_count: int | None = None,
) -> RecordFile | None:
    """Flush counter fields into a record's frontmatter (IN-4/IN-6/IN-14, SC-4).

    Updates only the provided fields (``hit_count``/``last_accessed``/
    ``access_count``), preserving the body and every other frontmatter field, and
    rewrites the file. Returns the updated :class:`RecordFile`, or ``None`` if the
    file was hand-deleted before writeback (no crash, robust per task).

    ``last_accessed`` is the ISO-8601 TEXT form (callers use :func:`epoch_to_iso`
    to convert from the epoch column at flush time).
    """
    changes: dict[str, Any] = {}
    if hit_count is not None:
        changes["hit_count"] = hit_count
    if last_accessed is not None:
        changes["last_accessed"] = last_accessed
    if access_count is not None:
        changes["access_count"] = access_count
    updated = replace(record, **changes) if changes else record
    return updated if _safe_rewrite(updated) else None


def set_superseded(record: RecordFile, superseded_by: str | None) -> RecordFile | None:
    """Set/clear the ``superseded_by`` frontmatter field (IN-10; supersede-not-delete).

    ``None`` clears the field (record becomes active again). Preserves the body
    and all other fields. Returns the updated record, or ``None`` if the file was
    hand-deleted before writeback.
    """
    updated = replace(record, superseded_by=superseded_by)
    return updated if _safe_rewrite(updated) else None


def set_stale(record: RecordFile, stale: bool) -> RecordFile | None:
    """Set the ``stale`` trust flag in frontmatter (IN-16; SessionEnd/reindex flush).

    Preserves the body and all other fields. Returns the updated record, or
    ``None`` if the file was hand-deleted before writeback.
    """
    updated = replace(record, stale=stale)
    return updated if _safe_rewrite(updated) else None


__all__ = [
    "RecordFile",
    "read_record",
    "write_record",
    "render_frontmatter",
    "iter_records",
    "writeback_counters",
    "set_superseded",
    "set_stale",
    "iso_to_epoch",
    "epoch_to_iso",
    "aliases_json",
    "aliases_flat",
    "aliases_from_json",
]
