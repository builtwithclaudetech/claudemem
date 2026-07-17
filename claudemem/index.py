"""claudemem.index — SQLite schema lifecycle + low-level DB primitives (L1).

This module owns the *storage substrate* shared by both database files
(architecture §2.3): connection opening with the standard PRAGMA block
(tech-design §3.6, C-18), the Fork A / Fork B DDL (§3.2–§3.5), the FTS5 sync
triggers (§3.3), the FTS5 startup probe (§3.10), ``auto_vacuum=INCREMENTAL`` +
bounded reclaim (§3.7), and the forkB ``PRAGMA user_version`` in-place migration
framework (§3.13).

It is a **foundation primitive** that ``store`` / ``recall`` / ``enrich`` build
on. Per architecture §2.3 it must NOT:

* call a model;
* know about ranking weights, salience, or output format;
* import ``recall``, ``enrich``, ``files``, or ``store``.

It imports ``config`` (for the ``CLAUDEMEM_HOME`` override + default home dir) and
the standard library only. The ``import-linter`` layering + read-path firewall
contracts depend on this absence staying true.

**Home-dir override mechanism.** DB file locations derive from
``config.CONFIG_HOME_ENV`` (``$CLAUDEMEM_HOME``), defaulting to
``config.DEFAULT_HOME`` (``~/.claude/claudemem``) — exactly the mechanism
``config`` itself uses, so tests can point ``CLAUDEMEM_HOME`` at a ``tmp_path``
and never touch the real ``~/.claude``.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from claudemem import config

_log = logging.getLogger("claudemem")

# --------------------------------------------------------------------------- #
# DB file names + schema versions                                             #
# --------------------------------------------------------------------------- #

#: Fork A curated-store index file (Record/RecordFts/SpendLog/Meta), §3.1.
FORKA_FILENAME = "forkA.db"
#: Fork B activity-archive file (Activity/Cursor), §3.1.
FORKB_FILENAME = "forkB.db"

#: Fork A ``Meta('schema_version', …)`` target — drives reindex-from-files on
#: mismatch (§3.13). v1 = 1.
FORKA_SCHEMA_VERSION = 1
#: Fork B ``PRAGMA user_version`` code target — drives the in-place migration
#: (§3.13). v1 = 1, so no migration step runs yet; the framework is exercised by
#: tests that register fake bumps.
FORKB_CODE_VERSION = 1

#: Pages reclaimed per opportunistic ``incremental_vacuum`` run (§3.7).
_RECLAIM_PAGES = 64

# --------------------------------------------------------------------------- #
# DDL — reproduced verbatim from tech-design §3.2–§3.5                          #
# --------------------------------------------------------------------------- #

_FORKA_RECORD_DDL = """
CREATE TABLE IF NOT EXISTS Record (
    Id            INTEGER PRIMARY KEY,
    Name          TEXT    NOT NULL,
    Scope         TEXT    NOT NULL,
    ProjectId     TEXT,
    Type          TEXT    NOT NULL,
    Importance    INTEGER NOT NULL DEFAULT 3,
    Pinned        INTEGER NOT NULL DEFAULT 0,
    Source        TEXT    NOT NULL,
    Created       INTEGER NOT NULL,
    LastAccessed  INTEGER NOT NULL,
    AccessCount   INTEGER NOT NULL DEFAULT 0,
    HitCount      INTEGER NOT NULL DEFAULT 0,
    Summary       TEXT,
    AliasesJson   TEXT,
    AliasesFlat   TEXT,
    SupersededBy  TEXT,
    Stale         INTEGER NOT NULL DEFAULT 0,
    EnrichPending INTEGER NOT NULL DEFAULT 0,
    Body          TEXT    NOT NULL,
    UNIQUE (Scope, ProjectId, Name)
) STRICT;
"""

# External-content FTS5 over Record. Column order Name, Summary, AliasesFlat,
# Body is load-bearing: the §4.1 bm25 weight vector (10/5/8/1) is positional.
_FORKA_RECORDFTS_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS RecordFts USING fts5(
    Name,
    Summary,
    AliasesFlat,
    Body,
    content='Record',
    content_rowid='Id',
    tokenize='porter unicode61 remove_diacritics 1'
);
"""

# The 3 sync triggers (§3.3): external-content FTS5 does NOT auto-sync. The
# 'delete' special-insert form uses the OLD row values to drop the prior entry.
_FORKA_TRIGGERS_DDL = """
CREATE TRIGGER IF NOT EXISTS Record_ai AFTER INSERT ON Record BEGIN
    INSERT INTO RecordFts(rowid, Name, Summary, AliasesFlat, Body)
    VALUES (new.Id, new.Name, new.Summary, new.AliasesFlat, new.Body);
END;

CREATE TRIGGER IF NOT EXISTS Record_ad AFTER DELETE ON Record BEGIN
    INSERT INTO RecordFts(RecordFts, rowid, Name, Summary, AliasesFlat, Body)
    VALUES ('delete', old.Id, old.Name, old.Summary, old.AliasesFlat, old.Body);
END;

CREATE TRIGGER IF NOT EXISTS Record_au AFTER UPDATE ON Record BEGIN
    INSERT INTO RecordFts(RecordFts, rowid, Name, Summary, AliasesFlat, Body)
    VALUES ('delete', old.Id, old.Name, old.Summary, old.AliasesFlat, old.Body);
    INSERT INTO RecordFts(rowid, Name, Summary, AliasesFlat, Body)
    VALUES (new.Id, new.Name, new.Summary, new.AliasesFlat, new.Body);
END;
"""

_FORKA_SPENDLOG_DDL = """
CREATE TABLE IF NOT EXISTS SpendLog (
    Id               INTEGER PRIMARY KEY,
    Ts               INTEGER NOT NULL,
    CallSite         TEXT    NOT NULL,
    Model            TEXT    NOT NULL,
    InputTokens      INTEGER NOT NULL DEFAULT 0,
    OutputTokens     INTEGER NOT NULL DEFAULT 0,
    CacheWriteTokens INTEGER NOT NULL DEFAULT 0,
    CacheReadTokens  INTEGER NOT NULL DEFAULT 0,
    IdempotencyKey   TEXT,
    Backend          TEXT    NOT NULL,
    Latency          INTEGER NOT NULL DEFAULT 0,
    RetryCount       INTEGER NOT NULL DEFAULT 0,
    Outcome          TEXT    NOT NULL DEFAULT 'ok'
) STRICT;
"""

_FORKA_META_DDL = """
CREATE TABLE IF NOT EXISTS Meta (
    Key   TEXT PRIMARY KEY,
    Value TEXT NOT NULL
) STRICT;
"""

_FORKA_INDEXES_DDL = """
CREATE INDEX IF NOT EXISTS IX_SpendLog_Ts ON SpendLog(Ts);

CREATE UNIQUE INDEX IF NOT EXISTS UX_SpendLog_Idem
    ON SpendLog(IdempotencyKey) WHERE IdempotencyKey IS NOT NULL;

CREATE INDEX IF NOT EXISTS IX_Record_Active
    ON Record(Scope, ProjectId, Pinned, Importance)
    WHERE SupersededBy IS NULL;
"""

_FORKB_ACTIVITY_DDL = """
CREATE TABLE IF NOT EXISTS Activity (
    Id        INTEGER PRIMARY KEY,
    SessionId TEXT    NOT NULL,
    Ts        INTEGER NOT NULL,
    Role      TEXT    NOT NULL,
    Kind      TEXT    NOT NULL,
    Body      TEXT,
    ToolRef   TEXT,
    FullLen   INTEGER NOT NULL DEFAULT 0,
    Truncated INTEGER NOT NULL DEFAULT 0
) STRICT;
"""

_FORKB_CURSOR_DDL = """
CREATE TABLE IF NOT EXISTS Cursor (
    SessionId TEXT PRIMARY KEY,
    LastLine  INTEGER NOT NULL DEFAULT 0
) STRICT;
"""

_FORKB_INDEX_DDL = """
CREATE INDEX IF NOT EXISTS IX_Activity_Ts ON Activity(Ts);
"""

# --------------------------------------------------------------------------- #
# Path resolution                                                              #
# --------------------------------------------------------------------------- #


def _home_dir() -> Path:
    """Resolve the claudemem home dir from ``$CLAUDEMEM_HOME`` or the default.

    Mirrors ``config._home_dir`` (same env var + default) so DB files and the
    config file always co-locate under one home dir, and tests can redirect both
    with a single ``CLAUDEMEM_HOME`` override.
    """
    override = os.environ.get(config.CONFIG_HOME_ENV)
    if override:
        return Path(override).expanduser()
    return config.DEFAULT_HOME


def forka_path() -> Path:
    """Absolute path to ``forkA.db`` under the resolved home dir."""
    return _home_dir() / FORKA_FILENAME


def forkb_path() -> Path:
    """Absolute path to ``forkB.db`` under the resolved home dir."""
    return _home_dir() / FORKB_FILENAME


# --------------------------------------------------------------------------- #
# Connection opening + PRAGMAs (§3.6, C-18)                                     #
# --------------------------------------------------------------------------- #


def _apply_standard_pragmas(conn: sqlite3.Connection, *, cold_path: bool) -> None:
    """Apply the §3.6 standard PRAGMA block on every connection, before any work.

    ``journal_mode``/``synchronous``/``temp_store``/``cache_size``/
    ``wal_autocheckpoint`` are set every open; ``foreign_keys``/``busy_timeout``
    are connection-scoped and must be re-set each open. ``cold_path=True`` raises
    ``cache_size`` to ~64 MB (§3.6 D) *after* the standard block, for the
    not-SC-2-bound reindex/import connection.
    """
    conn.execute("PRAGMA journal_mode = WAL;")
    conn.execute("PRAGMA busy_timeout = 5000;")
    conn.execute("PRAGMA synchronous = NORMAL;")
    conn.execute("PRAGMA foreign_keys = ON;")
    conn.execute("PRAGMA temp_store = MEMORY;")
    conn.execute("PRAGMA cache_size = -8000;")
    conn.execute("PRAGMA wal_autocheckpoint = 256;")
    if cold_path:
        # §3.6 D: per-connection override on the cold reindex/import path only.
        conn.execute("PRAGMA cache_size = -65536;")


def _new_connection(path: Path) -> sqlite3.Connection:
    """Open a raw connection, ensuring ``auto_vacuum=INCREMENTAL`` before any DDL.

    ``auto_vacuum`` only takes effect if set **before the first table is
    created** (§3.7). A brand-new DB file is empty until ``ensure_schema``/
    ``migrate_forkB`` create tables, so issuing the PRAGMA on every open is
    correct: it is honored on the first-ever open (no tables yet) and is a
    harmless no-op on subsequent opens.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA auto_vacuum = INCREMENTAL;")
    return conn


def open_forkA(*, cold_path: bool = False) -> sqlite3.Connection:
    """Open ``forkA.db`` with the standard PRAGMA block + schema ensured (§3.6).

    ``cold_path=True`` opens the larger-cache reindex/import connection (§3.6 D).
    Writes must go through :func:`write_tx` (``BEGIN IMMEDIATE``, C-18).
    """
    conn = _new_connection(forka_path())
    _apply_standard_pragmas(conn, cold_path=cold_path)
    ensure_schema(conn)
    return conn


def open_forkA_at(path: Path, *, cold_path: bool = False) -> sqlite3.Connection:
    """Open a forkA-schema DB at an *explicit* ``path`` (the reindex sidecar, §3.9).

    Same standard PRAGMA block + ``ensure_schema`` as :func:`open_forkA`, but
    targeting an arbitrary path (``forkA.db.rebuild``) rather than
    :func:`forka_path`. Shares ``_new_connection`` / ``_apply_standard_pragmas``
    so the PRAGMA logic is never duplicated. ``cold_path=True`` selects the larger
    reindex/import cache (§3.6 D).
    """
    conn = _new_connection(path)
    _apply_standard_pragmas(conn, cold_path=cold_path)
    ensure_schema(conn)
    return conn


def open_forkB() -> sqlite3.Connection:
    """Open ``forkB.db`` with the standard PRAGMA block, ensure schema, migrate.

    Per §3.13 the ``PRAGMA user_version`` check runs on every forkB open
    (immediately after the PRAGMA block); on the common version-matches path the
    migration body is skipped. Writes must go through :func:`write_tx`.
    """
    conn = _new_connection(forkb_path())
    _apply_standard_pragmas(conn, cold_path=False)
    _ensure_forkb_schema(conn)
    migrate_forkB(conn)
    return conn


@contextmanager
def write_tx(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Run a write transaction under ``BEGIN IMMEDIATE`` (§3.6, C-18).

    Every write path acquires the writer lock up front to avoid the
    deferred→immediate upgrade deadlock under concurrent writers;
    ``busy_timeout=5000`` lets a concurrent writer wait rather than erroring.
    The connection opens in ``isolation_level=None`` semantics here by issuing
    ``BEGIN IMMEDIATE`` explicitly and committing/rolling back by hand, so the
    sqlite3 module's implicit transaction handling does not double-begin.
    """
    # sqlite3's default isolation_level would auto-BEGIN (deferred) before DML;
    # disable it for the duration so our explicit BEGIN IMMEDIATE is the only one.
    prior_isolation = conn.isolation_level
    conn.isolation_level = None
    conn.execute("BEGIN IMMEDIATE;")
    try:
        yield conn
    except BaseException:
        # A nested ``executescript`` may have already auto-committed; only roll
        # back if a transaction is still open, else this raises spuriously.
        if conn.in_transaction:
            conn.execute("ROLLBACK;")
        raise
    else:
        if conn.in_transaction:
            conn.execute("COMMIT;")
    finally:
        conn.isolation_level = prior_isolation


# --------------------------------------------------------------------------- #
# Schema creation (§3.2–§3.4, §3.8, §3.10)                                      #
# --------------------------------------------------------------------------- #


def probe_fts5(conn: sqlite3.Connection) -> None:
    """Probe FTS5 availability once at DB creation (§3.10).

    Creates ``temp.__probe`` as an FTS5 virtual table; raises if FTS5 is absent.
    This is the **one allowed hard error** — FTS5 *is* the lexical core, so its
    absence is not an SC-3 degradation case. On the uv-shipped interpreter
    (SQLite 3.50.4) this never fires.
    """
    conn.execute("CREATE VIRTUAL TABLE temp.__probe USING fts5(x);")
    conn.execute("DROP TABLE temp.__probe;")


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Idempotently create the forkA schema, probe FTS5 once, write Meta (§3.10).

    Creates ``Record`` + external-content ``RecordFts`` + the 3 sync triggers +
    ``SpendLog`` + ``Meta`` + the §3.8 indexes (all ``IF NOT EXISTS``), runs the
    FTS5 probe, and records ``Meta('fts5_ok','1')`` / ``Meta('schema_version','1')``.
    The hot path never re-probes — ``executescript`` of ``IF NOT EXISTS`` DDL is a
    cheap no-op once the schema exists, and the Meta rows are upserted.
    """
    conn.executescript(_FORKA_RECORD_DDL)
    conn.executescript(_FORKA_RECORDFTS_DDL)
    conn.executescript(_FORKA_TRIGGERS_DDL)
    conn.executescript(_FORKA_SPENDLOG_DDL)
    conn.executescript(_FORKA_META_DDL)
    conn.executescript(_FORKA_INDEXES_DDL)
    probe_fts5(conn)
    conn.execute(
        "INSERT INTO Meta(Key, Value) VALUES ('fts5_ok', '1') "
        "ON CONFLICT(Key) DO UPDATE SET Value = excluded.Value;"
    )
    conn.execute(
        "INSERT INTO Meta(Key, Value) VALUES ('schema_version', ?) "
        "ON CONFLICT(Key) DO UPDATE SET Value = excluded.Value;",
        (str(FORKA_SCHEMA_VERSION),),
    )
    conn.commit()


def _ensure_forkb_schema(conn: sqlite3.Connection) -> None:
    """Idempotently create the forkB schema (``Activity`` + ``Cursor`` + index)."""
    conn.executescript(_FORKB_ACTIVITY_DDL)
    conn.executescript(_FORKB_CURSOR_DDL)
    conn.executescript(_FORKB_INDEX_DDL)
    conn.commit()


# --------------------------------------------------------------------------- #
# Reclaim (§3.7)                                                               #
# --------------------------------------------------------------------------- #


def reclaim(conn: sqlite3.Connection) -> None:
    """Run a bounded ``PRAGMA incremental_vacuum(64)`` (§3.7).

    Opportunistic, never a daemon: called after a Fork B prune and at the end of
    ``reindex``. The bound keeps any single reclaim well inside an interactive
    budget; residual free pages are reclaimed across subsequent runs.
    """
    conn.execute(f"PRAGMA incremental_vacuum({_RECLAIM_PAGES});")
    conn.commit()


# --------------------------------------------------------------------------- #
# forkB migration framework (§3.13)                                            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Migration:
    """A single forkB ``user_version`` bump step (§3.13).

    ``target`` is the ``user_version`` this step produces (it migrates a DB at
    ``target - 1`` up to ``target``). ``apply`` performs the schema change inside
    the caller's already-open ``BEGIN IMMEDIATE`` transaction; it must be
    idempotent (Class-1 ``ADD COLUMN`` guarded against re-entry; Class-2
    create-new + copy + swap). It must NOT begin/commit a transaction — the
    framework owns the single ``BEGIN IMMEDIATE … COMMIT`` and the
    ``user_version`` bump.
    """

    target: int
    cls: int  # 1 = additive ALTER ADD COLUMN; 2 = incompatible in-window copy
    apply: Callable[[sqlite3.Connection], None]


#: Ordered registry of forkB migrations (target ascending). EMPTY for v1
#: (FORKB_CODE_VERSION == 1): there is no real bump yet. The framework is proven
#: by tests that monkeypatch this list with fake Class-1 / Class-2 steps.
_MIGRATIONS: list[Migration] = []


def _column_exists(conn: sqlite3.Connection, table: str, column: str) -> bool:
    """Return whether ``column`` exists on ``table`` (Class-1 idempotency guard)."""
    rows = conn.execute(f"PRAGMA table_info({table});").fetchall()
    return any(row[1] == column for row in rows)


def add_column_if_absent(
    conn: sqlite3.Connection, table: str, column: str, decl: str
) -> None:
    """Class-1 helper: ``ALTER TABLE … ADD COLUMN`` guarded for idempotency.

    STRICT tables require an explicit datatype and a literal-typed default and
    forbid ``UNIQUE``/``PRIMARY KEY`` via ``ADD COLUMN`` (§3.13) — ``decl`` must
    honor that (e.g. ``"INTEGER NOT NULL DEFAULT 0"``). Re-entry is harmless: a
    column-existence check skips the ALTER, and an ``OperationalError`` (the race
    where another path added it) is swallowed.
    """
    if _column_exists(conn, table, column):
        return
    try:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {decl};")
    except sqlite3.OperationalError:
        # Duplicate-column race under concurrent sessions — already present.
        pass


def _pending_migrations(current: int, target: int) -> Sequence[Migration]:
    """Migrations whose ``target`` is in ``(current, target]``, ascending."""
    return [m for m in sorted(_MIGRATIONS, key=lambda m: m.target)
            if current < m.target <= target]


def _run_forkb_migration(conn: sqlite3.Connection, target: int) -> None:
    """Apply all pending steps in one ``BEGIN IMMEDIATE`` and bump user_version.

    ``user_version`` is set INSIDE the same transaction as the schema changes so
    schema + version flip atomically (§3.13): a crash leaves the DB fully-old or
    fully-new. ``BEGIN IMMEDIATE`` serializes concurrent sessions; a second
    session re-reads ``user_version`` (already bumped) and finds nothing pending.
    """
    with write_tx(conn):
        # Re-read inside the writer lock: a peer session may have just bumped.
        current = conn.execute("PRAGMA user_version;").fetchone()[0]
        for migration in _pending_migrations(current, target):
            migration.apply(conn)
        conn.execute(f"PRAGMA user_version = {target};")


def migrate_forkB(conn: sqlite3.Connection) -> None:
    """Best-effort in-place forkB migration via ``PRAGMA user_version`` (§3.13).

    Read ``user_version``; if it already equals the code target, no-op (the hot
    common path — single PRAGMA read, no transaction). Otherwise run the pending
    steps atomically. The whole thing is wrapped so a hook NEVER sees an
    exception (SC-3): on any failure → warn, roll back (handled by
    :func:`write_tx`), then **drop+recreate forkB from scratch at the current
    schema** — the only sanctioned data-loss path, justified because forkB is
    ephemeral (45-day window, no files-as-truth).
    """
    target = FORKB_CODE_VERSION
    try:
        current = conn.execute("PRAGMA user_version;").fetchone()[0]
        if current == target:
            return
        _run_forkb_migration(conn, target)
    except Exception:  # noqa: BLE001 — never let a migration failure reach a hook
        _log.warning(
            "forkB migration failed; recreating forkB at current schema "
            "(in-window archive rows lost — SC-3 fallback)",
            exc_info=True,
        )
        _recreate_forkb(conn, target)


def _recreate_forkb(conn: sqlite3.Connection, target: int) -> None:
    """Last-resort SC-3 fallback: drop every forkB object, recreate, set version.

    Must itself never raise (the hook must exit 0 regardless, §7.1). The DB stays
    usable afterward at the current schema; only in-window rows are lost.
    """
    try:
        with write_tx(conn):
            conn.execute("DROP TABLE IF EXISTS Activity;")
            conn.execute("DROP TABLE IF EXISTS Cursor;")
        _ensure_forkb_schema(conn)
        with write_tx(conn):
            conn.execute(f"PRAGMA user_version = {target};")
    except Exception:  # noqa: BLE001 — absolute last resort; swallow + log
        _log.error("forkB recreate fallback failed", exc_info=True)


__all__ = [
    "FORKA_FILENAME",
    "FORKB_FILENAME",
    "FORKA_SCHEMA_VERSION",
    "FORKB_CODE_VERSION",
    "Migration",
    "forka_path",
    "forkb_path",
    "open_forkA",
    "open_forkA_at",
    "open_forkB",
    "write_tx",
    "ensure_schema",
    "probe_fts5",
    "migrate_forkB",
    "reclaim",
    "add_column_if_absent",
]
