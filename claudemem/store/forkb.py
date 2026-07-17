"""claudemem.store.forkb — Fork B activity-archive persistence (L1, model-free).

The Fork B write/maintenance layer over :mod:`claudemem.index` (architecture
§2.4): append one ``Activity`` row per parsed turn event, advance the per-session
``Cursor`` watermark for incremental ``Stop`` transcript extraction, prune the
45-day rolling window, and expose the bounded per-session row set that the
Phase-4 reflection routine consumes.

**Model-free firewall (SC-6, IN-12, IN-2).** This is the SC-6-protected log path:
the 4,000-char head+tail cap and the tool-output→``ToolRef`` skip are applied
*here, at write time, with no model call*. Per architecture §2.4 this module
imports only :mod:`claudemem.index`, :mod:`claudemem.config`, and the standard
library — never ``enrich`` / ``recall`` / ``files`` / ``anthropic``, and it never
constructs a model request. The ``import-linter`` layering contract and the
``test_store_forkb`` ``sys.modules`` assertion both depend on that absence.

**Session scoping (§7.4).** Every read and write carries ``SessionId`` so two
concurrent sessions never collide: each session appends only its own rows, its
``Cursor`` watermark advances independently (``SessionId`` PRIMARY KEY), and
reflection reads a single session's in-window rows. Deriving the ``session_id``
string (payload ``session_id`` → ``transcript_path`` stem → ``unknown-{pid}``)
is the **caller's** (hooks') responsibility, per §7.4 — this module takes a
``session_id`` string as given.
"""

from __future__ import annotations

import hashlib
import sqlite3
import time

from claudemem import config, index

# --------------------------------------------------------------------------- #
# Head+tail cap split (§3.5, Q-3, IN-2, AS-4) — model-free, applied at write   #
# time. The spec fixes the split for the 4,000-char cap as first ~2,800 + last #
# ~1,000 with an elision marker between. Both halves are derived from the cap   #
# as fractions so the split tracks any configured `entry_char_cap` while        #
# reproducing the spec exactly at the 4,000 default:                            #
#                                                                               #
#   head = round(0.70 * cap) = 2,800   tail = round(0.25 * cap) = 1,000         #
#                                                                               #
# 0.70 + 0.25 = 0.95 leaves 5% of the cap (200 chars at 4,000) as headroom for  #
# the elision marker, so head + tail + marker always stays within the cap.      #
# --------------------------------------------------------------------------- #

_HEAD_FRACTION = 0.70
_TAIL_FRACTION = 0.25

#: Roles whose body is a tool result/output and must be skipped (§3.5/IN-2).
_TOOL_ROLE = "tool"


def _head_tail_split(cap: int) -> tuple[int, int]:
    """Return ``(head_chars, tail_chars)`` for ``cap`` (2,800 / 1,000 at 4,000)."""
    return round(_HEAD_FRACTION * cap), round(_TAIL_FRACTION * cap)


def _elision_marker(elided: int) -> str:
    """The marker placed between the kept head and tail of a truncated body."""
    return f"\n…[{elided} chars elided]…\n"


def _apply_cap(body: str, cap: int) -> tuple[str, int, int]:
    """Apply the §3.5 head+tail cap, model-free.

    Returns ``(stored_body, full_len, truncated)``. Under the cap the body is
    returned verbatim with ``truncated=0``; over the cap it becomes
    ``head + elision-marker + tail`` with ``truncated=1`` and ``full_len`` set to
    the original length. The stored result is guaranteed within ``cap`` because
    head+tail consume 95% of the cap and the marker fits the 5% headroom.
    """
    full_len = len(body)
    if full_len <= cap:
        return body, full_len, 0
    head_chars, tail_chars = _head_tail_split(cap)
    head = body[:head_chars]
    tail = body[full_len - tail_chars :]
    elided = full_len - head_chars - tail_chars
    stored = head + _elision_marker(elided) + tail
    return stored, full_len, 1


def _tool_ref(kind: str, body: str) -> str:
    """Build the compact reference line stored in place of a skipped tool body.

    Format ``tool:<kind> len=<n> sha=<8 hex>`` — informative (which tool event,
    how large, content fingerprint) without retaining the body. The sha lets a
    later read distinguish/deduplicate identical tool outputs cheaply.
    """
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()[:8]
    return f"tool:{kind} len={len(body)} sha={digest}"


# --------------------------------------------------------------------------- #
# T2.3 — append one Activity row                                               #
# --------------------------------------------------------------------------- #


def append_activity(
    conn: sqlite3.Connection,
    *,
    session_id: str,
    ts: int,
    role: str,
    kind: str,
    body: str,
    settings: config.Settings | None = None,
) -> int:
    """Append one ``Activity`` row from a parsed turn event (§3.5, model-free).

    Applies the two write-time transforms with no model call:

    * **Tool-output skip** — when ``role`` is ``'tool'`` the body is *not* stored:
      ``Body`` is ``NULL`` and only a compact ``ToolRef`` reference line is kept
      (``FullLen`` = original length, ``Truncated`` = 0).
    * **Head+tail cap** — otherwise a body longer than ``entry_char_cap`` (4,000)
      is stored as ``head + elision-marker + tail`` with ``Truncated=1`` and
      ``FullLen`` = original length; a shorter body is stored verbatim.

    ``settings`` supplies ``forkb.entry_char_cap``; when ``None`` the locked
    defaults are loaded. Returns the new row's ``Id`` (the ``b:<rowid>`` id base).
    """
    if settings is None:
        settings = config.load_config()
    cap = settings.forkb.entry_char_cap

    if role == _TOOL_ROLE:
        stored_body: str | None = None
        tool_ref: str | None = _tool_ref(kind, body)
        full_len = len(body)
        truncated = 0
    else:
        stored_body, full_len, truncated = _apply_cap(body, cap)
        tool_ref = None

    with index.write_tx(conn):
        cur = conn.execute(
            "INSERT INTO Activity "
            "(SessionId, Ts, Role, Kind, Body, ToolRef, FullLen, Truncated) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?);",
            (session_id, ts, role, kind, stored_body, tool_ref, full_len, truncated),
        )
        row_id = cur.lastrowid
    assert row_id is not None  # INSERT always yields a rowid on Activity
    return row_id


# --------------------------------------------------------------------------- #
# T2.4 — Cursor watermark (per-session, §3.5/§7.4)                              #
# --------------------------------------------------------------------------- #


def get_cursor(conn: sqlite3.Connection, session_id: str) -> int:
    """Return the ``Stop`` transcript watermark for ``session_id`` (0 if unset).

    A session with no prior ``Stop`` extraction has no ``Cursor`` row; the
    watermark defaults to 0 so the first extraction reads the transcript from
    line 0 forward (§3.5).
    """
    row = conn.execute(
        "SELECT LastLine FROM Cursor WHERE SessionId = ?;", (session_id,)
    ).fetchone()
    return int(row[0]) if row is not None else 0


def advance_cursor(conn: sqlite3.Connection, session_id: str, last_line: int) -> None:
    """Set ``session_id``'s watermark to ``last_line`` (per-session, idempotent).

    UPSERT on the ``SessionId`` PRIMARY KEY so the first ``Stop`` for a session
    inserts and subsequent ones update — concurrent sessions advance disjoint
    rows and never overwrite each other (§7.4).
    """
    with index.write_tx(conn):
        conn.execute(
            "INSERT INTO Cursor (SessionId, LastLine) VALUES (?, ?) "
            "ON CONFLICT(SessionId) DO UPDATE SET LastLine = excluded.LastLine;",
            (session_id, last_line),
        )


# --------------------------------------------------------------------------- #
# T2.4 — 45-day window prune + opportunistic reclaim (IN-2)                     #
# --------------------------------------------------------------------------- #


def prune_window(
    conn: sqlite3.Connection, *, now_epoch: int | None = None,
    settings: config.Settings | None = None,
) -> int:
    """Delete ``Activity`` rows older than the 45-day window; reclaim (IN-2).

    Opportunistic inline maintenance — **never a scheduled daemon** (§8.1, IN-2).
    The cutoff is ``now - window_days * 86400`` (``now`` defaults to the current
    UTC epoch seconds, matching ``Activity.Ts``); rows with ``Ts < cutoff`` are
    deleted. After the delete it calls :func:`index.reclaim` (a bounded
    ``incremental_vacuum(64)``) to return freed pages, also opportunistically.
    Returns the number of rows pruned.
    """
    if settings is None:
        settings = config.load_config()
    if now_epoch is None:
        now_epoch = int(time.time())
    cutoff = now_epoch - settings.forkb.window_days * 86400

    with index.write_tx(conn):
        cur = conn.execute("DELETE FROM Activity WHERE Ts < ?;", (cutoff,))
        pruned = cur.rowcount
    index.reclaim(conn)
    return pruned


# --------------------------------------------------------------------------- #
# T2.4 — per-session in-window read (reflection consumes this, §7.4)            #
# --------------------------------------------------------------------------- #


def rows_for_session(
    conn: sqlite3.Connection, session_id: str, *, now_epoch: int | None = None,
    settings: config.Settings | None = None,
) -> list[sqlite3.Row]:
    """Return ``session_id``'s in-window ``Activity`` rows, oldest first (§7.4).

    This is exactly what the Phase-4 reflection routine reads — already bounded
    (45-day window), capped, and tool-output-skipped (the write-time transforms
    in :func:`append_activity`), **not** the raw transcript. Scoped
    ``WHERE SessionId = ?`` so concurrent sessions never reflect over each
    other's rows. Rows are returned as :class:`sqlite3.Row` (column access by
    name) ordered by ``Ts`` then ``Id`` for stable turn order.
    """
    if settings is None:
        settings = config.load_config()
    if now_epoch is None:
        now_epoch = int(time.time())
    cutoff = now_epoch - settings.forkb.window_days * 86400

    prior_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT Id, SessionId, Ts, Role, Kind, Body, ToolRef, FullLen, "
            "Truncated FROM Activity "
            "WHERE SessionId = ? AND Ts >= ? ORDER BY Ts, Id;",
            (session_id, cutoff),
        ).fetchall()
    finally:
        conn.row_factory = prior_factory
    return rows


# --------------------------------------------------------------------------- #
# Read accessors for the recall archive fallback (§4.3, §5.2) — model-free.    #
# These are the typed by-this-layer reads ``recall.search`` / ``recall.get``   #
# consume; the SQL lives here (the sole persistence layer, architecture §2.4)  #
# so it tracks the §3.5 ``Activity`` DDL, not in the read modules above it.    #
# All three return :class:`sqlite3.Row` (column access by name) and stay pure  #
# reads — no model, no write, mirroring this module's column-enumerated,       #
# parameterized SQL idioms (never ``SELECT *``).                               #
# --------------------------------------------------------------------------- #


def _like_escape(token: str) -> str:
    """Escape LIKE wildcards in a literal token (paired with ``ESCAPE '\\'``).

    Doubles the escape char, then escapes ``%`` and ``_`` so a query token such
    as ``50%`` matches the literal substring rather than acting as a wildcard.
    """
    return token.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def archive_matching(
    conn: sqlite3.Connection,
    tokens: list[str],
    *,
    cutoff_epoch: int,
    limit: int,
) -> list[sqlite3.Row]:
    """In-window ``Activity`` rows whose ``Body`` LIKE-matches any token (§4.3).

    The bounded lexical backstop for ``recall.search``'s A→B fallback: a
    case-insensitive ``Body LIKE '%tok%' ESCAPE '\\'`` OR-scan over the rows with
    ``Ts >= cutoff_epoch``, most-recent first (``ORDER BY Ts DESC, Id DESC``),
    capped at ``limit``. Each token is wildcard-escaped (:func:`_like_escape`) and
    bound as a parameter, so there is no injection surface. Tool-output rows are
    excluded via ``Body IS NULL`` alone — a tool row always has ``Body`` NULL
    (§3.5), so no ``Role`` predicate is needed. ``tokens`` is assumed non-empty
    (the caller routes the empty case to :func:`archive_recent`).
    """
    like_clause = " OR ".join("Body LIKE ? ESCAPE '\\'" for _ in tokens)
    params: list[object] = [cutoff_epoch]
    params.extend(f"%{_like_escape(t)}%" for t in tokens)
    params.append(limit)
    prior_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT Id, Body FROM Activity "
            "WHERE Ts >= ? AND Body IS NOT NULL "
            f"AND ({like_clause}) "
            "ORDER BY Ts DESC, Id DESC LIMIT ?;",
            params,
        ).fetchall()
    finally:
        conn.row_factory = prior_factory
    return rows


def archive_recent(
    conn: sqlite3.Connection, *, cutoff_epoch: int, limit: int
) -> list[sqlite3.Row]:
    """Most-recent in-window non-tool ``Activity`` rows (the no-token fallback).

    A query with no usable token still triggers the archive (§4.3), so this
    surfaces the most-recent raw activity (``Ts >= cutoff_epoch``, ``Body`` not
    NULL, ``ORDER BY Ts DESC, Id DESC``, capped at ``limit``) rather than nothing.
    """
    prior_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT Id, Body FROM Activity "
            "WHERE Ts >= ? AND Body IS NOT NULL "
            "ORDER BY Ts DESC, Id DESC LIMIT ?;",
            (cutoff_epoch, limit),
        ).fetchall()
    finally:
        conn.row_factory = prior_factory
    return rows


def get_activity(conn: sqlite3.Connection, rowid: int) -> sqlite3.Row | None:
    """Fetch one ``Activity`` row by rowid for ``recall.get`` (``b:<rowid>``).

    Column-enumerated by-rowid SELECT (Id, Body, Role, Kind, Ts). Returns the
    :class:`sqlite3.Row` or ``None`` when the rowid is absent (pruned out of the
    45-day window) — the caller maps both ``None`` and a NULL ``Body`` (tool
    row, §3.5) to the SC-3 "not found" reply.
    """
    prior_factory = conn.row_factory
    conn.row_factory = sqlite3.Row
    try:
        row: sqlite3.Row | None = conn.execute(
            "SELECT Id, Body, Role, Kind, Ts FROM Activity WHERE Id = ?;",
            (rowid,),
        ).fetchone()
    finally:
        conn.row_factory = prior_factory
    return row


__all__ = [
    "append_activity",
    "get_cursor",
    "advance_cursor",
    "prune_window",
    "rows_for_session",
    "archive_matching",
    "archive_recent",
    "get_activity",
]
