"""claudemem.hooks — the single Claude Code hook dispatch entry (L4, T5.4).

``claudemem hook <event>`` (the only hook wiring, architecture §2.8 / §6.1) is
the one place a Claude Code hook reaches ClaudeMem. ``cli._cmd_hook`` lazily
imports this module and calls :func:`dispatch` — so a *read* command never drags
``hooks`` (and its SessionEnd-only ``enrich`` reach) into ``sys.modules``.

**Recursion guard is the literal FIRST statement of :func:`dispatch`**
(tech-design §7.1 / §6.3 MF-3, architecture §6.3): ``CLAUDEMEM_DISABLE_HOOKS``
set → return 0 *before* reading stdin, importing anything, or opening a DB. This
bounds recursion at depth 1 — the inner ``claude -p`` ClaudeMem spawns inherits
the env var (MF-2 merge), so its own hooks all no-op and cannot spawn a third
level (`SC-1`).

**ALWAYS exit 0** (`SC-3`, architecture §2.8 / §7.1). The whole body after the
guard is wrapped: any failure — malformed/empty stdin, a DB-locked / corrupt /
missing-key condition — is logged to ``~/.claude/claudemem/claudemem.log`` and
the hook returns 0. A hook NEVER raises to Claude Code; worst case it loses one
turn's archive row or one menu injection and continues.

**Event → flow** (architecture §6.1, tech-design §7.6):

* ``SessionStart`` → :func:`recall.menu` (source-aware: skip on ``resume``);
  emit the menu as the SessionStart ``additionalContext`` (≤10,000-char ceiling).
* ``UserPromptSubmit`` → ``log`` the prompt (model-free; NO enrich import).
* ``Stop`` → ``log`` the new transcript turns from the per-session ``Cursor``
  watermark forward (defensive JSONL parse, model-free; NO enrich import).
* ``SessionEnd`` → **reflection FIRST, then EnrichPending backfill** (§5.7
  two-spawn order) — the ONLY path that imports ``enrich`` (lazily).

**Read enrich LAZILY, only on the SessionEnd path** (architecture §2.8 "Must
NOT" / §4). The import-linter layering contract and the read-path firewall both
depend on ``menu``/``log`` never reaching ``enrich``.

**No hook-side state** (tech-design §7.6): ``cwd`` and ``session_id`` derive from
the payload alone. ``session_id`` falls back ``session_id`` →
``Path(transcript_path).stem`` → ``f"unknown-{os.getpid()}"`` (§7.4) so two
concurrent sessions that both lose the id still get distinct ids.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import sqlite3

    from claudemem.config import ScopeContext, Settings

#: The recursion-guard env var (tech-design §7.1); mirrors ``cli.GUARD_ENV_VAR``.
GUARD_ENV_VAR = "CLAUDEMEM_DISABLE_HOOKS"

#: SC-5 hard char ceiling on the emitted SessionStart ``additionalContext``.
#: ``recall.menu`` already caps to ≤600 tokens / ≤30 entries / ≤10,000 chars;
#: this is the belt-and-suspenders enforcement of the same 10 KB ceiling on the
#: text actually emitted (the task's "enforce the hard ceiling" requirement).
_ADDITIONAL_CONTEXT_CHAR_CEILING = 10_000

_log = logging.getLogger("claudemem")

#: Canonical event keys → handler dispatch. Both the Claude Code PascalCase names
#: (``SessionStart``) and the kebab-case CLI aliases (``session-start``, the form
#: the §6.1 wiring table registers) map to one normalized key.
_EVENT_ALIASES = {
    "sessionstart": "session-start",
    "session-start": "session-start",
    "userpromptsubmit": "user-prompt-submit",
    "user-prompt-submit": "user-prompt-submit",
    "stop": "stop",
    "sessionend": "session-end",
    "session-end": "session-end",
}


def dispatch(event: str) -> int:
    """The single hook entry: guard → read stdin → map event → flow → exit 0.

    The recursion guard is the literal FIRST statement (tech-design §7.1 / §6.3
    MF-3): set → return 0 before reading stdin / importing / opening a DB. After
    the guard, the entire body is wrapped so the hook ALWAYS returns 0 (`SC-3`):
    malformed/empty stdin, a DB error, a missing key — all are logged and
    swallowed. Never raises to ``cli`` / Claude Code (architecture §2.8).
    """
    # ── Recursion guard — MUST be first (tech-design §7.1 / §6.3 MF-3). ──────
    # Before ANY stdin read, import, or DB open: a guarded inner `claude -p`
    # session inherits this env var (MF-2 merge), so every event no-ops at 0.
    if os.environ.get(GUARD_ENV_VAR):
        return 0

    try:
        payload = _read_payload()
        return _route(event, payload)
    except Exception:  # noqa: BLE001 — SC-3: a hook NEVER raises; log + exit 0.
        _log.error("claudemem hook %s failed (exit 0 per SC-3)", event, exc_info=True)
        _log_to_file(event)
        return 0


# --------------------------------------------------------------------------- #
# Payload + routing                                                             #
# --------------------------------------------------------------------------- #


def _read_payload() -> dict[str, Any]:
    """Read + parse the hook JSON payload from stdin, defensively (AS-10).

    Empty or malformed stdin yields an empty dict rather than raising — a hook
    handed garbage stdin must still no-op cleanly (`SC-3`). Every downstream
    accessor uses ``.get()`` so a missing field is never a ``KeyError``.
    """
    raw = sys.stdin.read()
    if not raw or not raw.strip():
        return {}
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        _log.warning("claudemem hook: unparseable stdin payload; treating as empty")
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _route(event: str, payload: dict[str, Any]) -> int:
    """Map a Claude Code event to its internal flow (architecture §6.1, §7.6).

    Unknown events log and no-op at 0 (`SC-3`) — a future Claude Code hook we do
    not handle must never error. ``cwd`` / ``session_id`` come from the payload
    (no hook-side state, §7.6).
    """
    key = _EVENT_ALIASES.get(event.strip().lower())
    if key is None:
        _log.info("claudemem hook: unhandled event %r (no-op)", event)
        return 0

    if key == "session-start":
        return _on_session_start(payload)
    if key == "user-prompt-submit":
        return _on_user_prompt_submit(payload)
    if key == "stop":
        return _on_stop(payload)
    # key == "session-end"
    return _on_session_end(payload)


def _scope_ctx(payload: dict[str, Any]) -> ScopeContext:
    """Resolve the active scope from the payload ``cwd`` (no hook-side state, §7.4).

    Falls back to the process cwd when the payload omits ``cwd`` (defensive — a
    malformed payload must not abort the hook). No ``--scope`` / ``--project``
    override on the hook path; the scope is purely cwd-derived (`AS-8`, `IN-3`).
    """
    from claudemem import config

    cwd_raw = payload.get("cwd")
    cwd = Path(cwd_raw) if isinstance(cwd_raw, str) and cwd_raw else Path.cwd()
    return config.resolve_scope(cwd)


def _session_id(payload: dict[str, Any]) -> str:
    """Derive the session id with the §7.4 fallback chain (never errors).

    ``session_id`` → ``Path(transcript_path).stem`` → ``f"unknown-{os.getpid()}"``.
    The last-resort id is pid-suffixed (not a literal ``"unknown"``) so two
    concurrent sessions that both lose ``session_id`` get distinct ids and never
    merge each other's rows in reflection (§7.4).
    """
    sid = payload.get("session_id")
    if isinstance(sid, str) and sid:
        return sid
    transcript = payload.get("transcript_path")
    if isinstance(transcript, str) and transcript:
        stem = Path(transcript).stem
        if stem:
            return stem
    return f"unknown-{os.getpid()}"


def _settings() -> Settings:
    """Load the frozen settings (locked defaults when ``config.toml`` is absent)."""
    from claudemem import config

    return config.load_config()


# --------------------------------------------------------------------------- #
# SessionStart → recall.menu (model-free; NO enrich import) — architecture §5.3 #
# --------------------------------------------------------------------------- #


def _on_session_start(payload: dict[str, Any]) -> int:
    """``SessionStart`` → emit the salience menu as ``additionalContext`` (IN-11).

    Source-aware (IN-11): ``source == 'resume'`` → ``menu`` returns ``""`` and
    nothing is injected. Otherwise the menu (already ≤600 tokens / ≤30 entries /
    ≤10 KB chars, `SC-5`) is emitted in the documented SessionStart hook output
    shape, with the 10 KB ceiling re-enforced on the emitted text.

    Model-free: imports only ``index`` / ``recall`` (NO ``enrich``, architecture
    §2.8 / §4).
    """
    from claudemem import index
    from claudemem.recall import menu as menu_mod

    scope_ctx = _scope_ctx(payload)
    source = payload.get("source")
    source = source if isinstance(source, str) else None

    conn_a = index.open_forkA()
    try:
        text = menu_mod.menu(conn_a, scope_ctx, source, settings=_settings())
    finally:
        conn_a.close()

    _emit_additional_context(text)
    return 0


def _emit_additional_context(text: str) -> None:
    """Print the SessionStart ``additionalContext`` JSON (≤10 KB ceiling, `SC-5`).

    The empty string (no records / ``resume`` skip) prints nothing — Claude sees
    no injection, not an empty block. Otherwise we emit the documented Claude
    Code SessionStart hook output shape::

        {"hookSpecificOutput": {"hookEventName": "SessionStart",
                                 "additionalContext": "<menu>"}}

    Claude Code adds ``additionalContext`` to the session context on exit 0. The
    text is truncated to the 10,000-char ceiling defensively — ``menu`` already
    caps to the same ceiling, so this only ever trims a pathological overflow.
    """
    if not text:
        return
    capped = text[:_ADDITIONAL_CONTEXT_CHAR_CEILING]
    output = {
        "hookSpecificOutput": {
            "hookEventName": "SessionStart",
            "additionalContext": capped,
        }
    }
    print(json.dumps(output))


# --------------------------------------------------------------------------- #
# UserPromptSubmit → log (model-free; NO enrich import) — architecture §5.7     #
# --------------------------------------------------------------------------- #


def _on_user_prompt_submit(payload: dict[str, Any]) -> int:
    """``UserPromptSubmit`` → append one model-free Fork B prompt row (IN-12).

    Captures the submitted prompt (``role='user'``, ``kind='prompt'``) via
    ``store.forkb.append_activity`` — the cap is applied at write time inside
    forkb, model-free. The 30 s budget is comfortably met (one INSERT). Imports
    only ``index`` / ``store`` (NO ``enrich``, `SC-6`/`IN-12`).

    A payload with no prompt text is a clean no-op (nothing to log) — exit 0.
    """
    from claudemem import index
    from claudemem.store import forkb

    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt:
        return 0

    session_id = _session_id(payload)
    settings = _settings()
    conn_b = index.open_forkB()
    try:
        forkb.append_activity(
            conn_b,
            session_id=session_id,
            ts=int(time.time()),
            role="user",
            kind="prompt",
            body=prompt,
            settings=settings,
        )
    finally:
        conn_b.close()
    return 0


# --------------------------------------------------------------------------- #
# Stop → log (model-free; NO enrich import) — architecture §5.7, tech-design §7.5
# --------------------------------------------------------------------------- #


def _on_stop(payload: dict[str, Any]) -> int:
    """``Stop`` → append new transcript turns from the ``Cursor`` watermark (IN-12).

    Reads the transcript JSONL at ``transcript_path`` line-by-line from
    ``Cursor.LastLine`` forward (defensive, AS-10 — skip unparseable lines,
    ``.get()`` everywhere), maps each entry to ``(role, kind, body)``, and lets
    ``store.forkb.append_activity`` apply the head+tail cap + tool-output→ToolRef
    transforms MODEL-FREE at write time (§3.5). Advances the watermark to the new
    line count, then opportunistically prunes the 45-day window (+ reclaim).
    MODEL-FREE — zero model calls (`SC-6`/`IN-12`); NO ``enrich`` import.

    A missing / unreadable transcript is a clean no-op (exit 0). The watermark is
    only advanced past lines actually consumed, so a partial read re-reads the
    tail on the next ``Stop`` rather than skipping turns.
    """
    from claudemem import index
    from claudemem.store import forkb

    transcript_path = payload.get("transcript_path")
    if not isinstance(transcript_path, str) or not transcript_path:
        return 0
    path = Path(transcript_path)
    if not path.is_file():
        return 0

    session_id = _session_id(payload)
    settings = _settings()
    conn_b = index.open_forkB()
    try:
        last_line = forkb.get_cursor(conn_b, session_id)
        new_line, entries = _read_transcript_from(path, last_line)
        for role, kind, body in entries:
            forkb.append_activity(
                conn_b,
                session_id=session_id,
                ts=int(time.time()),
                role=role,
                kind=kind,
                body=body,
                settings=settings,
            )
        if new_line > last_line:
            forkb.advance_cursor(conn_b, session_id, new_line)
        # Opportunistic inline maintenance — never a daemon (§3.7, IN-2).
        forkb.prune_window(conn_b, settings=settings)
    finally:
        conn_b.close()
    return 0


def _read_transcript_from(
    path: Path, start_line: int
) -> tuple[int, list[tuple[str, str, str]]]:
    """Parse transcript JSONL from ``start_line`` forward → ``(new_line, entries)``.

    Defensive per AS-10 / §7.5: line-by-line, skip-on-failure (a malformed line
    never aborts the read), ``.get()`` everywhere. ``new_line`` is the total
    line count read (the next watermark); ``entries`` is the ordered list of
    ``(role, kind, body)`` tuples extracted from the new lines only.

    The transcript schema is external + undocumented (the very reason for the
    defensive posture). Each line is a JSON object with a top-level ``type``
    (``user`` / ``assistant`` / control rows); the message lives under
    ``message`` with ``content`` either a plain string (user prompts) or a list
    of typed blocks (``text`` / ``thinking`` / ``tool_use`` / ``tool_result``).
    Tool-result blocks are surfaced as ``role='tool'`` so forkb skips the body to
    a ``ToolRef`` (§3.5); control rows (no usable message) are skipped.
    """
    entries: list[tuple[str, str, str]] = []
    line_count = 0
    try:
        with path.open("r", encoding="utf-8", errors="replace") as fh:
            for line_count, raw in enumerate(fh, start=1):
                if line_count <= start_line:
                    continue
                entries.extend(_parse_transcript_line(raw))
    except OSError as exc:
        _log.warning("claudemem hook Stop: transcript read failed: %s", exc)
        # Return whatever was parsed so far; do not advance past unread lines.
        return start_line, entries
    return line_count, entries


def _parse_transcript_line(raw: str) -> list[tuple[str, str, str]]:
    """Extract ``(role, kind, body)`` tuples from one transcript JSONL line.

    Skip-on-failure (AS-10): an unparseable line, a non-dict line, or a line with
    no usable message yields ``[]`` — never raises. ``.get()`` everywhere guards
    the undocumented, version-unstable schema.
    """
    line = raw.strip()
    if not line:
        return []
    try:
        obj = json.loads(line)
    except (ValueError, TypeError):
        return []
    if not isinstance(obj, dict):
        return []

    entry_type = obj.get("type")
    if entry_type not in ("user", "assistant"):
        # Control rows (queue-operation, summary, etc.) carry no turn content.
        return []

    message = obj.get("message")
    if not isinstance(message, dict):
        return []
    role = message.get("role")
    role = role if isinstance(role, str) and role else entry_type
    content = message.get("content")

    if isinstance(content, str):
        text = content.strip()
        kind = "prompt" if role == "user" else "text"
        return [(role, kind, text)] if text else []

    if isinstance(content, list):
        return _parse_content_blocks(role, content)

    return []


def _parse_content_blocks(
    role: str, blocks: list[Any]
) -> list[tuple[str, str, str]]:
    """Map a content-block list to ``(role, kind, body)`` tuples (defensive).

    Block ``type`` drives the mapping (skip-on-failure on any non-dict block):

    * ``text`` → ``(role, 'text', <text>)``.
    * ``thinking`` → ``(role, 'thinking', <thinking>)``.
    * ``tool_use`` → ``('tool', 'tool_use', <name + input>)`` — forkb skips the
      body to a ``ToolRef`` (`role='tool'`, §3.5).
    * ``tool_result`` → ``('tool', 'tool_result', <result text>)`` — likewise
      ToolRef-skipped; this is the (usually large) tool output we never retain.

    Empty bodies are dropped so an empty block never produces a blank row.
    """
    out: list[tuple[str, str, str]] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "text":
            body = _coerce_text(block.get("text"))
            if body:
                out.append((role, "text", body))
        elif btype == "thinking":
            body = _coerce_text(block.get("thinking"))
            if body:
                out.append((role, "thinking", body))
        elif btype == "tool_use":
            name = _coerce_text(block.get("name")) or "tool"
            tool_input = block.get("input")
            body = f"{name} {json.dumps(tool_input, default=str)}".strip()
            out.append(("tool", "tool_use", body))
        elif btype == "tool_result":
            body = _coerce_text(block.get("content"))
            out.append(("tool", "tool_result", body or ""))
        # Unknown block types are skipped (forward-compatible, AS-10).
    return out


def _coerce_text(value: Any) -> str:
    """Coerce a content field to a plain string (defensive, AS-10).

    A ``tool_result`` ``content`` is usually a string but may be a list of
    ``{"type": "text", "text": ...}`` blocks; flatten that to text. Anything else
    is stringified so a structured body still produces a (skipped) ToolRef.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, dict):
                txt = item.get("text")
                if isinstance(txt, str):
                    parts.append(txt)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts).strip()
    return str(value).strip()


# --------------------------------------------------------------------------- #
# SessionEnd → reflect THEN backfill (the ONLY enrich-importing path) — §5.4/5.7
# --------------------------------------------------------------------------- #


def _on_session_end(payload: dict[str, Any]) -> int:
    """``SessionEnd`` → reflection FIRST, then EnrichPending backfill (§5.7 order).

    The two-spawn SessionEnd order (tech-design §5.7, architecture §5.4):

    1. ``routine.reflect`` — the reflection routine (Haiku call #2): reinforce
       confirmed passive hits + propose Fork B→A promotion candidates over this
       session's bounded Fork B rows. Degrades to a clean no-op when no transport
       is available (`SC-3`).
    2. ``routine.enrich_batch`` — backfill the ``EnrichPending=1`` Fork A records
       (the carry-forward from degraded saves) — routine #1, not a third code
       path (`SC-6`).

    This is the ONLY hook path that imports ``enrich`` — and it does so LAZILY
    here, inside the handler (architecture §2.8 / §4). ``cli`` opens the
    connections (`index` owns connection opening); a degraded backend or a DB
    error is logged and the hook still exits 0 (`SC-3`).
    """
    from claudemem import index
    from claudemem.enrich import routine  # lazy — the ONLY enrich import (§4)

    scope_ctx = _scope_ctx(payload)
    session_id = _session_id(payload)
    settings = _settings()

    conn_a = index.open_forkA()
    conn_b = index.open_forkB()
    try:
        # 1. Reflection FIRST (§5.7) — Haiku call #2 over this session's rows.
        routine.reflect(conn_b, session_id, conn_a, scope_ctx, settings)
        # 2. THEN the EnrichPending backfill (routine #1 — carry-forward saves).
        _backfill_enrich_pending(conn_a, routine, scope_ctx, settings)
    finally:
        conn_a.close()
        conn_b.close()
    return 0


def _backfill_enrich_pending(
    conn_a: sqlite3.Connection,
    routine: Any,
    scope_ctx: ScopeContext,
    settings: Settings,
) -> None:
    """Backfill ``EnrichPending=1`` Fork A records via ``enrich_batch`` (§5.4 step 2).

    Selects the degraded-save records (those persisted lexical-only while a
    backend was unavailable / over-cap), re-reads their markdown (files-as-truth),
    and routes them through ``enrich_batch`` — the SAME shared routine #1, never a
    third model code path (`SC-6`). A record whose file was hand-deleted is simply
    skipped (`C-11`). Nothing pending → clean no-op.
    """
    from claudemem import files
    from claudemem.store import forka

    rows = forka.pending_names(conn_a)
    if not rows:
        return

    records: list[files.RecordFile] = []
    for name, _scope_kind in rows:
        target = _read_forka_file(scope_ctx, name)
        if target is not None:
            records.append(target)
    if not records:
        return

    routine.enrich_batch(conn_a, records, scope_ctx, settings)


def _read_forka_file(scope_ctx: ScopeContext, name: str) -> Any:
    """Read a Fork A record's markdown by name from the scope dirs, or ``None``.

    Mirrors ``cli._read_forka_file`` (the backfill needs the file body to
    re-enrich). A hand-deleted file returns ``None`` (`C-11` / `SC-4`).
    """
    from claudemem import files

    for directory in (scope_ctx.project_dir, scope_ctx.global_dir):
        if directory is None:
            continue
        path = directory / f"{name}.md"
        if path.is_file():
            return files.read_record(path)
    return None


# --------------------------------------------------------------------------- #
# Local logfile (SC-3 / architecture §7.1)                                      #
# --------------------------------------------------------------------------- #


def _log_to_file(event: str) -> None:
    """Append a hook-failure trace to ``~/.claude/claudemem/claudemem.log`` (`SC-3`).

    The §7.1 local logfile: hook errors are written here, never raised to Claude
    Code. Best-effort — if even this write fails (unwritable home), it is
    swallowed (the hook must still exit 0). The log dir is the resolved
    ``CLAUDEMEM_HOME`` so a test pointing it at ``tmp_path`` keeps the real
    ``~/.claude`` untouched.
    """
    try:
        from claudemem import config

        home = os.environ.get(config.CONFIG_HOME_ENV)
        log_dir = Path(home).expanduser() if home else config.DEFAULT_HOME
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        with (log_dir / "claudemem.log").open("a", encoding="utf-8") as fh:
            fh.write(f"{ts} hook {event} failed (logged, exit 0 per SC-3)\n")
    except Exception:  # noqa: BLE001 — last-resort; the hook must still exit 0.
        pass


__all__ = ["dispatch", "GUARD_ENV_VAR"]
