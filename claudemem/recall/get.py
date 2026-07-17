"""claudemem.recall.get — the ``get`` read command (L2, model-free).

Full-body fetch of a single record by its unified id (PRD IN-4, architecture
§2.5). ``output.parse_id`` splits ``a:<name>`` / ``b:<rowid>`` and dispatches by
fork: Fork A returns the full ``Record`` body (and refreshes the in-session
access counters in the **index only**, IN-4); Fork B returns the stored
``Activity`` body from the archive. An unparseable id, an unknown Fork A name, or
a pruned/unknown Fork B rowid all resolve to a clean "not found" string with
**exit-0 semantics** (SC-3) — this command never raises.

**Firewall (architecture §2.5, §4; SC-6/SC-2).** Imports only ``recall``
(``output``), ``store`` (``forka``), ``config``, and the standard library
(``sqlite3`` for the passed connections). NEVER ``enrich`` / ``anthropic``; no
model request, no ``claude`` spawn.

**Connection ownership.** As with ``search``, the caller opens the DBs and
threads them in: ``conn_a`` is the Fork A index (the ``a:`` path + the IN-4
counter refresh), ``conn_b`` is the Fork B archive (the ``b:`` path).
"""

from __future__ import annotations

import json as _json
import sqlite3
import time

from claudemem import config
from claudemem.recall import output
from claudemem.store import forka, forkb

#: The clean SC-3 "not found" reply — a single human line, exit 0. The id is
#: echoed so the caller/Claude can see which id missed.
_NOT_FOUND = "not found"


def get(
    conn_a: sqlite3.Connection,
    conn_b: sqlite3.Connection,
    raw_id: str,
    scope_ctx: config.ScopeContext,
    *,
    json: bool = False,
    now_epoch: int | None = None,
) -> str:
    """Return the full body of one record by id, or a "not found" line (IN-4).

    ``output.parse_id`` splits the unified id; an :class:`output.InvalidId`
    (no/unknown fork prefix, empty key) is caught and becomes the clean
    "not found" reply (SC-3 — never an exception). Dispatch:

    * ``a:<name>`` → :func:`store.select_record` within ``scope_ctx``. A hit is
      serialized full-body via :func:`output.serialize_get`, and its in-session
      access counters are bumped **in the index only** (:func:`_refresh_access`,
      IN-4) — the file flush is owned by SessionEnd / reindex, not here.
    * ``b:<rowid>`` → :func:`_get_archive` against the Fork B ``Activity`` table.
      A non-numeric or pruned rowid → "not found" (SC-3).

    ``now_epoch`` (defaults to the current UTC epoch seconds) sets the refreshed
    ``LastAccessed`` value and is injectable for deterministic tests.
    """
    try:
        fork, key = output.parse_id(raw_id)
    except output.InvalidId:
        return _not_found(raw_id)

    if fork == "a":
        return _get_forka(
            conn_a, key, raw_id, scope_ctx, json=json, now_epoch=now_epoch
        )
    return _get_archive(conn_b, key, raw_id, json=json)


def _not_found(raw_id: str) -> str:
    """The SC-3 clean "not found" reply for ``raw_id`` (a single human line)."""
    return f"{_NOT_FOUND}: {raw_id}"


def _get_forka(
    conn_a: sqlite3.Connection,
    name: str,
    raw_id: str,
    scope_ctx: config.ScopeContext,
    *,
    json: bool,
    now_epoch: int | None,
) -> str:
    """Fork A full-body fetch + IN-4 in-session access refresh."""
    record = forka.select_record(conn_a, scope_ctx, name)
    if record is None:
        return _not_found(raw_id)
    if now_epoch is None:
        now_epoch = int(time.time())
    _refresh_access(conn_a, record.id, now_epoch)
    return output.serialize_get(record, json=json)


def _refresh_access(conn_a: sqlite3.Connection, record_id: int, now_epoch: int) -> None:
    """IN-4: bump ``LastAccessed``/``AccessCount`` in the INDEX ONLY (best-effort).

    Delegates the write to :func:`store.forka.refresh_access` (which runs the
    UPDATE under ``index.write_tx``, so the §3.3 FTS sync triggers fire as on any
    Fork A write). A failure here must never sink the read — IN-4 is best-effort
    access tracking (SC-3) — so a :class:`sqlite3.Error` (e.g. a read-only or
    locked connection) is swallowed; the body is still returned to the caller.
    """
    try:
        forka.refresh_access(conn_a, record_id, now_epoch=now_epoch)
    except sqlite3.Error:
        pass


def _get_archive(
    conn_b: sqlite3.Connection, key: str, raw_id: str, *, json: bool
) -> str:
    """Fork B full-body fetch by rowid; pruned/unknown/non-numeric → "not found".

    Delegates the by-rowid read to :func:`store.forkb.get_activity` (the sole
    persistence layer, architecture §2.4). A non-integer key, an absent rowid
    (pruned out of the 45-day window), or a tool-output row (``Body IS NULL``)
    all resolve to the clean "not found" reply (SC-3, §8.2).
    """
    try:
        rowid = int(key)
    except ValueError:
        return _not_found(raw_id)

    row = forkb.get_activity(conn_b, rowid)
    if row is None or row["Body"] is None:
        return _not_found(raw_id)

    body = row["Body"]
    title = output.archive_title(body)
    if json:
        obj: dict[str, object] = {
            "id": output.make_id_b(rowid),
            "archive": True,
            "summary": title,
            "body": body,
        }
        return _json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    header = f"{output.make_id_b(rowid)}{output.ID_TITLE_SEP}{title}"
    return f"{header}\n\n{body}"


__all__ = ["get"]
