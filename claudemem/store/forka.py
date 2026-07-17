"""claudemem.store.forka — typed persistence over Fork A's ``Record``/``RecordFts``.

The L2 ``store`` layer's Fork A half (architecture §2.4): typed CRUD against the
``Record`` table, the supersede-not-delete soft delete (PRD SC-7/C-10/IN-8), the
partial active-set scan (``IX_Record_Active``, tech-design §3.8), and the
**model-free** FTS5 candidate query (§4.3) reused by both ``recall.search`` and
``enrich`` dedup (§5.4, IN-13/SC-12).

Module discipline (architecture §2.4 — enforced by the import-linter layering +
read-path-firewall contracts): this module imports ``index``, ``files``,
``config``, and the standard library **only**. It must NOT call a model, format
human/JSON output, decide an enrichment backend, nor import
``recall``/``enrich``/``anthropic``. ``store`` provides the candidate query that
``enrich`` *consumes*; it never constructs a model request itself.

**Timestamp boundary (MF-1, §3.2/§3.12).** ``Record.Created``/``LastAccessed``
are ``INTEGER`` UTC-epoch seconds in the table; the file frontmatter stays
ISO-8601 TEXT. The ISO → epoch crossing happens here at index time via
:func:`files.iso_to_epoch`; epoch → ISO writeback is owned by ``files``.

**Upsert key (§3.2).** ``Record`` carries ``UNIQUE(Scope, ProjectId, Name)`` but
``ProjectId`` is ``NULL`` for global scope, and SQLite treats ``NULL`` as
distinct in a UNIQUE index — so a plain ``ON CONFLICT(Scope, ProjectId, Name)``
upsert never fires for global records and would insert duplicates. We therefore
resolve the existing row with NULL-aware matching (``ProjectId IS NULL`` vs
``ProjectId = ?``) and branch INSERT/UPDATE explicitly. RecordFts stays in sync
automatically via the §3.3 triggers — this module never writes ``RecordFts``.
"""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from typing import Any

from claudemem import files, index
from claudemem.config import ScopeContext

# --------------------------------------------------------------------------- #
# bm25 column order (§4.1) — Name, Summary, AliasesFlat, Body. Load-bearing.     #
# --------------------------------------------------------------------------- #

#: ``candidate_k`` default (§4.3); a param so dedup can pass ``dedup_k=5`` (§5.4).
DEFAULT_CANDIDATE_K = 64

#: The ``Record`` columns selected by row accessors, in a fixed positional order
#: that :func:`_row_to_record` maps onto :class:`Record`. Enumerated explicitly
#: (never ``SELECT *``) so the row contract is stable and auditable.
_RECORD_COLUMNS = (
    "Id",
    "Name",
    "Scope",
    "ProjectId",
    "Type",
    "Importance",
    "Pinned",
    "Source",
    "Created",
    "LastAccessed",
    "AccessCount",
    "HitCount",
    "Summary",
    "AliasesJson",
    "AliasesFlat",
    "SupersededBy",
    "Stale",
    "EnrichPending",
    "Body",
)
_RECORD_COLUMN_LIST = ", ".join(_RECORD_COLUMNS)

#: Tokenizer for the injection-safe MATCH builder: any run of characters that is
#: NOT a Unicode word character is a separator. This strips FTS5 operator syntax
#: (``OR``-as-bareword survives as a literal token but is quoted, ``*``/``"``/
#: ``(``/``:`` are dropped), leaving only literal terms to double-quote.
_TOKEN_SPLIT = re.compile(r"\W+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class Record:
    """One ``Record`` row, typed (architecture §2.4 "typed persistence layer").

    Mirrors the §3.2 column set. Timestamps are ``INTEGER`` UTC-epoch seconds (the
    in-table MF-1 representation); ``Pinned``/``Stale``/``EnrichPending`` are the
    raw 0/1 ``INTEGER`` storage values. ``AliasesJson`` is the canonical alias
    mirror (reconstruct the list via :func:`files.aliases_from_json`).
    """

    id: int
    name: str
    scope: str
    project_id: str | None
    type: str
    importance: int
    pinned: int
    source: str
    created: int
    last_accessed: int
    access_count: int
    hit_count: int
    summary: str | None
    aliases_json: str | None
    aliases_flat: str | None
    superseded_by: str | None
    stale: int
    enrich_pending: int
    body: str


def _row_to_record(row: tuple[Any, ...]) -> Record:
    """Map a positional row (selected in :data:`_RECORD_COLUMNS` order) to a Record."""
    return Record(
        id=row[0],
        name=row[1],
        scope=row[2],
        project_id=row[3],
        type=row[4],
        importance=row[5],
        pinned=row[6],
        source=row[7],
        created=row[8],
        last_accessed=row[9],
        access_count=row[10],
        hit_count=row[11],
        summary=row[12],
        aliases_json=row[13],
        aliases_flat=row[14],
        superseded_by=row[15],
        stale=row[16],
        enrich_pending=row[17],
        body=row[18],
    )


# --------------------------------------------------------------------------- #
# CRUD (T2.1)                                                                   #
# --------------------------------------------------------------------------- #


def _existing_id(
    conn: sqlite3.Connection, scope: str, project_id: str | None, name: str
) -> int | None:
    """Resolve the rowid of an existing record on the ``(Scope, ProjectId, Name)``
    natural key with NULL-aware ``ProjectId`` matching (see module docstring)."""
    if project_id is None:
        row = conn.execute(
            "SELECT Id FROM Record "
            "WHERE Scope = ? AND ProjectId IS NULL AND Name = ?;",
            (scope, name),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT Id FROM Record "
            "WHERE Scope = ? AND ProjectId = ? AND Name = ?;",
            (scope, project_id, name),
        ).fetchone()
    return row[0] if row is not None else None


def upsert_record(
    conn: sqlite3.Connection,
    record_file: files.RecordFile,
    scope_ctx: ScopeContext,
    *,
    enrich_pending: bool = False,
) -> int:
    """Insert or update the ``Record`` row for ``record_file`` + scope (§3.2/§3.12).

    Maps every §3.12 column off the :class:`files.RecordFile`: ISO → epoch for
    ``Created``/``LastAccessed`` via :func:`files.iso_to_epoch`; the alias list →
    ``AliasesJson`` (canonical) + ``AliasesFlat`` (FTS-fed) via the ``files``
    dual-form helpers (§3.11). ``Scope``/``ProjectId`` come from ``scope_ctx``
    (the derived columns are not frontmatter-backed, §3.12); ``EnrichPending`` is
    the runtime degraded-save marker.

    Keyed on ``(Scope, ProjectId, Name)`` with NULL-aware matching. The whole
    write runs under :func:`index.write_tx` (``BEGIN IMMEDIATE``, C-18); RecordFts
    syncs via the §3.3 triggers. Returns the row ``Id``.
    """
    scope = scope_ctx.kind
    project_id = scope_ctx.project_id if scope_ctx.kind == "project" else None

    values = (
        record_file.name,
        scope,
        project_id,
        record_file.type,
        record_file.importance,
        1 if record_file.pinned else 0,
        record_file.source,
        files.iso_to_epoch(record_file.created),
        files.iso_to_epoch(record_file.last_accessed),
        record_file.access_count,
        record_file.hit_count,
        record_file.summary,
        files.aliases_json(record_file.aliases),
        files.aliases_flat(record_file.aliases),
        record_file.superseded_by,
        1 if record_file.stale else 0,
        1 if enrich_pending else 0,
        record_file.body,
    )

    with index.write_tx(conn):
        existing = _existing_id(conn, scope, project_id, record_file.name)
        if existing is None:
            cursor = conn.execute(
                "INSERT INTO Record ("
                "Name, Scope, ProjectId, Type, Importance, Pinned, Source, "
                "Created, LastAccessed, AccessCount, HitCount, Summary, "
                "AliasesJson, AliasesFlat, SupersededBy, Stale, EnrichPending, Body"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);",
                values,
            )
            row_id = cursor.lastrowid
            assert row_id is not None  # INSERT always sets lastrowid
            return row_id
        conn.execute(
            "UPDATE Record SET "
            "Name = ?, Scope = ?, ProjectId = ?, Type = ?, Importance = ?, "
            "Pinned = ?, Source = ?, Created = ?, LastAccessed = ?, "
            "AccessCount = ?, HitCount = ?, Summary = ?, AliasesJson = ?, "
            "AliasesFlat = ?, SupersededBy = ?, Stale = ?, EnrichPending = ?, "
            "Body = ? WHERE Id = ?;",
            (*values, existing),
        )
        return existing


def select_record(
    conn: sqlite3.Connection, scope_ctx: ScopeContext, name: str
) -> Record | None:
    """Fetch one record by ``name`` within the scope, or ``None`` (T2.1).

    Resolves against the same ``(Scope, ProjectId, Name)`` natural key as
    :func:`upsert_record` (global → ``ProjectId IS NULL``; project →
    ``ProjectId = scope_ctx.project_id``), so an upsert round-trips back through
    this accessor. Superseded rows are still returned (this is a by-name lookup,
    not the active set); callers filter on ``superseded_by`` if they need active.
    """
    scope = scope_ctx.kind
    project_id = scope_ctx.project_id if scope_ctx.kind == "project" else None
    if project_id is None:
        row = conn.execute(
            f"SELECT {_RECORD_COLUMN_LIST} FROM Record "
            "WHERE Scope = ? AND ProjectId IS NULL AND Name = ?;",
            (scope, name),
        ).fetchone()
    else:
        row = conn.execute(
            f"SELECT {_RECORD_COLUMN_LIST} FROM Record "
            "WHERE Scope = ? AND ProjectId = ? AND Name = ?;",
            (scope, project_id, name),
        ).fetchone()
    return _row_to_record(row) if row is not None else None


def mark_superseded(
    conn: sqlite3.Connection,
    name: str,
    superseded_by: str | None,
    scope_ctx: ScopeContext,
) -> None:
    """Set/clear ``SupersededBy`` — the soft delete (SC-7/C-10/IN-8).

    This is how ``forget`` and conflict-supersede retire a record: the row is
    **never** DELETEd (the SC-7 trail survives), it just drops out of the
    ``SupersededBy IS NULL`` active set (§3.8). ``superseded_by=None`` reactivates
    the record. Resolved on the same NULL-aware natural key as the rest of the
    module; a no-op if no such record exists. Runs under ``BEGIN IMMEDIATE``.
    """
    scope = scope_ctx.kind
    project_id = scope_ctx.project_id if scope_ctx.kind == "project" else None

    with index.write_tx(conn):
        if project_id is None:
            conn.execute(
                "UPDATE Record SET SupersededBy = ? "
                "WHERE Scope = ? AND ProjectId IS NULL AND Name = ?;",
                (superseded_by, scope, name),
            )
        else:
            conn.execute(
                "UPDATE Record SET SupersededBy = ? "
                "WHERE Scope = ? AND ProjectId = ? AND Name = ?;",
                (superseded_by, scope, project_id, name),
            )


def refresh_access(conn: sqlite3.Connection, record_id: int, *, now_epoch: int) -> None:
    """IN-4: bump ``LastAccessed``/``AccessCount`` in the INDEX ONLY.

    The access-tracking write behind ``recall.get`` (distinct from confirmed-hit
    reinforcement): set ``LastAccessed = now_epoch`` (epoch seconds, MF-1) and
    increment ``AccessCount`` by one for ``record_id``. The file frontmatter is
    NOT touched — those counters are flushed to markdown at the next batch
    writeback (SessionEnd reflection, IN-14, or ``reindex``, IN-10).

    Runs under :func:`index.write_tx` (``BEGIN IMMEDIATE``, C-18) like any other
    Fork A write, so the §3.3 FTS sync triggers fire exactly as on any
    ``Record`` UPDATE. This is **not** best-effort here: a DB error propagates so
    the caller decides. ``recall.get`` keeps its SC-3 swallow-on-error contract
    (a failed refresh must never sink a read) around this call.
    """
    with index.write_tx(conn):
        conn.execute(
            "UPDATE Record SET LastAccessed = ?, AccessCount = AccessCount + 1 "
            "WHERE Id = ?;",
            (now_epoch, record_id),
        )


def set_pinned(conn: sqlite3.Connection, record_id: int, pinned: bool) -> None:
    """IN-7: set/clear the immortal ranking flag (``Pinned``) in the INDEX.

    The index half of ``pin``/``unpin`` (cli flushes the markdown frontmatter
    separately, SC-4). Stores the raw 0/1 ``INTEGER`` for ``record_id``. Runs
    under :func:`index.write_tx` (``BEGIN IMMEDIATE``, C-18) like every Fork A
    write, so the §3.3 FTS sync triggers fire as on any ``Record`` UPDATE. A
    no-op if no such row exists.
    """
    with index.write_tx(conn):
        conn.execute(
            "UPDATE Record SET Pinned = ? WHERE Id = ?;",
            (1 if pinned else 0, record_id),
        )


def bump_hit(conn: sqlite3.Connection, record_id: int) -> int:
    """SC-9/SC-13 confirmed-hit reinforcement: ``HitCount`` +1 + clear stale.

    The index half of ``used`` for a Fork A record: increment ``HitCount`` by
    exactly one and clear ``Stale`` for ``record_id`` (the cli flushes the new
    count + stale flag to markdown separately). The increment is done atomically
    in SQL (``HitCount = HitCount + 1``) rather than read-then-write — the same
    observable effect (one increment per ``used`` call) without a TOCTOU window.
    Returns the new ``HitCount``. Runs under :func:`index.write_tx`
    (``BEGIN IMMEDIATE``, C-18) so the §3.3 FTS sync triggers fire. Returns ``0``
    if no such row exists (the cli guards on ``select_record`` first, so this is
    unreached in practice).
    """
    with index.write_tx(conn):
        conn.execute(
            "UPDATE Record SET HitCount = HitCount + 1, Stale = 0 WHERE Id = ?;",
            (record_id,),
        )
        row = conn.execute(
            "SELECT HitCount FROM Record WHERE Id = ?;",
            (record_id,),
        ).fetchone()
    return int(row[0]) if row is not None else 0


def pending_names(
    conn: sqlite3.Connection, *, all_active: bool = False
) -> list[tuple[str, str]]:
    """The EnrichPending backfill select: active records' ``(Name, Scope)`` (IN-10).

    Returns ``(Name, Scope)`` for the degraded-save records still awaiting
    enrichment (``EnrichPending = 1 AND SupersededBy IS NULL``) — the carry-forward
    set re-read from markdown and routed through ``enrich_batch`` by both
    ``reindex`` (PHASE B) and the SessionEnd backfill. With ``all_active=True``
    the predicate drops the ``EnrichPending`` clause and returns every active row
    (``SupersededBy IS NULL``), the ``reindex --reenrich-all`` path (§3.9).

    No explicit ``ORDER BY``: the callers do not depend on a particular order
    (each name is independently re-read and grouped by its own scope), so the
    natural rowid scan order is preserved unchanged from the prior inline queries.
    """
    if all_active:
        rows = conn.execute(
            "SELECT Name, Scope FROM Record WHERE SupersededBy IS NULL;"
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT Name, Scope FROM Record "
            "WHERE EnrichPending = 1 AND SupersededBy IS NULL;"
        ).fetchall()
    return [(row[0], row[1]) for row in rows]


def count_pending(conn: sqlite3.Connection) -> int:
    """Count active records still ``EnrichPending`` (IN-10 convergence report).

    ``COUNT(*)`` over ``EnrichPending = 1 AND SupersededBy IS NULL`` — the
    still-pending tally ``reindex`` reports after a backfill pass when the spend
    cap leaves records carried forward.
    """
    return int(
        conn.execute(
            "SELECT COUNT(*) FROM Record "
            "WHERE EnrichPending = 1 AND SupersededBy IS NULL;"
        ).fetchone()[0]
    )


def active_set(conn: sqlite3.Connection, scope_ctx: ScopeContext) -> list[Record]:
    """Active records for the scope, scope-merged (``IX_Record_Active``, §3.8).

    The "active" set is ``SupersededBy IS NULL`` (the partial-index predicate is
    the single definition of active, §3.8). Scope merge is the §4.3/SC-12 rule:
    global plus the active project (``ProjectId = scope_ctx.project_id``); for a
    global-scope context only the global rows match. This is the **no-MATCH**
    path that feeds ``menu`` and dedup (§3.8 MF-2) — it should ride
    ``IX_Record_Active``. Order matches the index key columns
    (Scope, ProjectId, Pinned, Importance) so the plan can stay index-only for
    the scan; the salience tie-break ordering is ``recall``'s job, not here.
    """
    pid = scope_ctx.project_id if scope_ctx.kind == "project" else None
    rows = conn.execute(
        f"SELECT {_RECORD_COLUMN_LIST} FROM Record "
        "WHERE SupersededBy IS NULL "
        "AND (Scope = 'global' OR (Scope = 'project' AND ProjectId = ?)) "
        "ORDER BY Scope, ProjectId, Pinned, Importance;",
        (pid,),
    ).fetchall()
    return [_row_to_record(row) for row in rows]


# --------------------------------------------------------------------------- #
# Model-free FTS5 candidate query (T2.2; §4.3, reused by §5.4 dedup)            #
# --------------------------------------------------------------------------- #


def _build_match_expr(query: str) -> str | None:
    """Build an injection-safe FTS5 MATCH expression from free-text ``query``.

    Load-bearing security boundary: a naive f-string MATCH lets a query like
    ``a* NEAR b`` or ``foo OR bar`` invoke FTS5 query operators (and a stray
    ``"`` raises a syntax error that would surface as a crash). We instead split
    the query into literal terms on any non-word run, **double-quote each term**
    (FTS5 string literals — operators inside a quoted phrase are inert), escape
    embedded double-quotes by doubling them (FTS5 rule), and join with ``OR``.

    Returns ``None`` when the query yields no usable token (empty or
    punctuation-only) — the §4.3 contract: the caller (``recall.search``) treats
    that as a relevance-floor miss and falls back to the Fork B archive.
    """
    tokens = [t for t in _TOKEN_SPLIT.split(query) if t]
    if not tokens:
        return None
    quoted = [f'"{t.replace(chr(34), chr(34) * 2)}"' for t in tokens]
    return " OR ".join(quoted)


def fts_candidates(
    conn: sqlite3.Connection,
    query: str,
    scope_ctx: ScopeContext,
    k: int = DEFAULT_CANDIDATE_K,
) -> list[tuple[int, float]]:
    """The shared model-free lexical candidate query (§4.3; reused by §5.4 dedup).

    Runs the exact §4.3 SELECT: the FTS5 ``MATCH`` of OR-ed double-quoted literal
    tokens drives candidate selection, ``Record`` is joined by its rowid PK, the
    set is restricted to active rows (``SupersededBy IS NULL``) and scope-merged
    (global + active project, SC-12), ordered by raw ``bm25`` ascending
    (most-negative = best match first, §4.1) and capped at ``k``.

    ``k`` defaults to ``candidate_k = 64`` (§4.3) but is a parameter: dedup passes
    ``dedup_k = 5`` (§5.4). The bm25 weight vector is the §4.1 constant
    ``(10.0, 5.0, 8.0, 1.0)`` in the fixed RecordFts column order (Name, Summary,
    AliasesFlat, Body) — positional, so it is **not** caller-tunable here.

    Returns ``[(Id, raw_bm25)]`` — the **raw** SQLite ``bm25`` value (non-positive;
    more negative = better), ascending. No salience/logistic transform is applied
    (that is ``recall.rank``'s job); this primitive stays pure and model-free so
    both ``recall.search`` and ``enrich`` dedup can reuse it (IN-13). An empty /
    punctuation-only ``query`` → no usable tokens → ``[]``.
    """
    match_expr = _build_match_expr(query)
    if match_expr is None:
        return []
    pid = scope_ctx.project_id if scope_ctx.kind == "project" else None
    rows = conn.execute(
        "SELECT r.Id, bm25(RecordFts, 10.0, 5.0, 8.0, 1.0) AS raw "
        "FROM RecordFts JOIN Record r ON r.Id = RecordFts.rowid "
        "WHERE RecordFts MATCH ? "
        "AND r.SupersededBy IS NULL "
        "AND (r.Scope = 'global' OR (r.Scope = 'project' AND r.ProjectId = ?)) "
        "ORDER BY bm25(RecordFts, 10.0, 5.0, 8.0, 1.0) "
        "LIMIT ?;",
        (match_expr, pid, k),
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


__all__ = [
    "Record",
    "DEFAULT_CANDIDATE_K",
    "upsert_record",
    "select_record",
    "mark_superseded",
    "refresh_access",
    "set_pinned",
    "bump_hit",
    "pending_names",
    "count_pending",
    "active_set",
    "fts_candidates",
]
