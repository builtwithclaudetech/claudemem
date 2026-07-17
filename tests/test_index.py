"""Tests for claudemem.index (T1.5–T1.9).

Covers: the standard PRAGMA block on every connection (§3.6, C-18); forkA schema
creation — every table/index/trigger exists, ``Record`` is STRICT, ``RecordFts``
is external-content (§3.2–§3.4, §3.8); the 3 FTS5 sync triggers keep ``RecordFts``
in sync on INSERT/UPDATE/DELETE (§3.3); the FTS5 probe success path + Meta write
(§3.10); ``auto_vacuum=INCREMENTAL`` + bounded reclaim (§3.7); and the forkB
``PRAGMA user_version`` migration framework (§3.13) — Class-1 idempotency, a
Class-2 in-window copy, and the never-error drop+recreate fallback (SC-3).

The claudemem home dir is pointed at ``tmp_path`` via ``CLAUDEMEM_HOME`` so no
test touches the real ``~/.claude``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from claudemem import config, index


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME at a tmp dir; return it (no DB files written yet)."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    return tmp_path


def _scalar(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> object:
    return conn.execute(sql, params).fetchone()[0]


def _objects(conn: sqlite3.Connection, obj_type: str) -> set[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type = ?;", (obj_type,)
    ).fetchall()
    return {r[0] for r in rows}


# --------------------------------------------------------------------------- #
# Path resolution + PRAGMAs                                                    #
# --------------------------------------------------------------------------- #


def test_db_paths_follow_home_override(home: Path) -> None:
    assert index.forka_path() == home / index.FORKA_FILENAME
    assert index.forkb_path() == home / index.FORKB_FILENAME


def test_standard_pragmas_applied_forka(home: Path) -> None:
    conn = index.open_forkA()
    assert str(_scalar(conn, "PRAGMA journal_mode;")).lower() == "wal"
    assert _scalar(conn, "PRAGMA synchronous;") == 1  # NORMAL
    assert _scalar(conn, "PRAGMA foreign_keys;") == 1
    assert _scalar(conn, "PRAGMA cache_size;") == -8000
    assert _scalar(conn, "PRAGMA wal_autocheckpoint;") == 256
    assert _scalar(conn, "PRAGMA auto_vacuum;") == 2  # INCREMENTAL
    conn.close()


def test_cold_path_overrides_cache_size(home: Path) -> None:
    conn = index.open_forkA(cold_path=True)
    assert _scalar(conn, "PRAGMA cache_size;") == -65536
    conn.close()


def test_standard_pragmas_applied_forkb(home: Path) -> None:
    conn = index.open_forkB()
    assert str(_scalar(conn, "PRAGMA journal_mode;")).lower() == "wal"
    assert _scalar(conn, "PRAGMA auto_vacuum;") == 2  # INCREMENTAL
    conn.close()


# --------------------------------------------------------------------------- #
# forkA schema creation (§3.2–§3.4, §3.8, §3.10)                                #
# --------------------------------------------------------------------------- #


def test_forka_tables_indexes_triggers_exist(home: Path) -> None:
    conn = index.open_forkA()
    tables = _objects(conn, "table")
    assert {"Record", "RecordFts", "SpendLog", "Meta"} <= tables
    indexes = _objects(conn, "index")
    assert {"IX_SpendLog_Ts", "UX_SpendLog_Idem", "IX_Record_Active"} <= indexes
    triggers = _objects(conn, "trigger")
    assert {"Record_ai", "Record_ad", "Record_au"} == triggers
    conn.close()


def test_record_is_strict(home: Path) -> None:
    conn = index.open_forkA()
    ddl = str(_scalar(
        conn, "SELECT sql FROM sqlite_master WHERE name = 'Record';"
    ))
    assert ddl.rstrip().endswith("STRICT")
    # STRICT enforces declared types: a non-integer into an INTEGER column errors.
    with pytest.raises(sqlite3.IntegrityError):
        with index.write_tx(conn):
            conn.execute(
                "INSERT INTO Record(Name, Scope, Type, Source, Created, "
                "LastAccessed, Importance, Body) VALUES "
                "('n', 'global', 'user', 'explicit', 0, 0, 'not-an-int', 'b');"
            )
    conn.close()


def test_recordfts_is_external_content(home: Path) -> None:
    conn = index.open_forkA()
    ddl = str(_scalar(
        conn, "SELECT sql FROM sqlite_master WHERE name = 'RecordFts';"
    ))
    assert "content='Record'" in ddl
    assert "content_rowid='Id'" in ddl
    assert "porter unicode61 remove_diacritics 1" in ddl
    conn.close()


def test_partial_indexes_have_predicates(home: Path) -> None:
    conn = index.open_forkA()
    active = str(_scalar(
        conn, "SELECT sql FROM sqlite_master WHERE name = 'IX_Record_Active';"
    ))
    assert "WHERE SupersededBy IS NULL" in active
    idem = str(_scalar(
        conn, "SELECT sql FROM sqlite_master WHERE name = 'UX_SpendLog_Idem';"
    ))
    assert "WHERE IdempotencyKey IS NOT NULL" in idem
    conn.close()


def test_meta_written_on_create(home: Path) -> None:
    conn = index.open_forkA()
    assert _scalar(conn, "SELECT Value FROM Meta WHERE Key = 'fts5_ok';") == "1"
    assert (
        _scalar(conn, "SELECT Value FROM Meta WHERE Key = 'schema_version';")
        == str(index.FORKA_SCHEMA_VERSION)
    )
    conn.close()


def test_ensure_schema_idempotent(home: Path) -> None:
    conn = index.open_forkA()
    index.ensure_schema(conn)  # second call must not raise
    index.ensure_schema(conn)
    assert {"Record", "RecordFts", "SpendLog", "Meta"} <= _objects(conn, "table")
    conn.close()


def test_probe_fts5_success(home: Path) -> None:
    conn = index.open_forkA()
    index.probe_fts5(conn)  # raises only if FTS5 absent; here it must succeed
    conn.close()


# --------------------------------------------------------------------------- #
# FTS5 sync triggers (§3.3)                                                     #
# --------------------------------------------------------------------------- #


def _insert_record(conn: sqlite3.Connection, name: str, body: str) -> int:
    with index.write_tx(conn):
        cur = conn.execute(
            "INSERT INTO Record(Name, Scope, Type, Source, Created, "
            "LastAccessed, Body) VALUES (?, 'global', 'user', 'explicit', 0, 0, ?);",
            (name, body),
        )
    return int(cur.lastrowid or 0)


def _fts_match_names(conn: sqlite3.Connection, term: str) -> set[str]:
    rows = conn.execute(
        "SELECT r.Name FROM RecordFts JOIN Record r ON r.Id = RecordFts.rowid "
        "WHERE RecordFts MATCH ?;",
        (term,),
    ).fetchall()
    return {r[0] for r in rows}


def test_trigger_insert_syncs_fts(home: Path) -> None:
    conn = index.open_forkA()
    _insert_record(conn, "widget", "alpha bravo charlie")
    assert "widget" in _fts_match_names(conn, "bravo")
    conn.close()


def test_trigger_update_syncs_fts(home: Path) -> None:
    conn = index.open_forkA()
    rid = _insert_record(conn, "widget", "alpha bravo charlie")
    with index.write_tx(conn):
        conn.execute(
            "UPDATE Record SET Body = ? WHERE Id = ?;", ("delta echo foxtrot", rid)
        )
    assert _fts_match_names(conn, "bravo") == set()  # old text gone
    assert "widget" in _fts_match_names(conn, "echo")  # new text present
    conn.close()


def test_trigger_delete_syncs_fts(home: Path) -> None:
    conn = index.open_forkA()
    rid = _insert_record(conn, "widget", "alpha bravo charlie")
    with index.write_tx(conn):
        conn.execute("DELETE FROM Record WHERE Id = ?;", (rid,))
    assert _fts_match_names(conn, "bravo") == set()  # gone from FTS
    conn.close()


# --------------------------------------------------------------------------- #
# forkB schema + reclaim                                                       #
# --------------------------------------------------------------------------- #


def test_forkb_tables_and_index_exist(home: Path) -> None:
    conn = index.open_forkB()
    assert {"Activity", "Cursor"} <= _objects(conn, "table")
    assert "IX_Activity_Ts" in _objects(conn, "index")
    conn.close()


def test_reclaim_runs(home: Path) -> None:
    conn = index.open_forkB()
    index.reclaim(conn)  # bounded incremental_vacuum(64); must not raise
    conn.close()


# --------------------------------------------------------------------------- #
# forkB migration framework (§3.13)                                            #
# --------------------------------------------------------------------------- #


def test_migrate_forkb_noop_when_versions_match(home: Path) -> None:
    conn = index.open_forkB()  # open already ran migrate_forkB at target
    assert _scalar(conn, "PRAGMA user_version;") == index.FORKB_CODE_VERSION
    index.migrate_forkB(conn)  # idempotent re-run
    assert _scalar(conn, "PRAGMA user_version;") == index.FORKB_CODE_VERSION
    conn.close()


def test_class1_migration_and_idempotency(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fake Class-1 ADD COLUMN bump applies once and re-runs cleanly."""

    def add_flag(conn: sqlite3.Connection) -> None:
        index.add_column_if_absent(
            conn, "Activity", "Reviewed", "INTEGER NOT NULL DEFAULT 0"
        )

    fake = index.Migration(target=2, cls=1, apply=add_flag)
    monkeypatch.setattr(index, "_MIGRATIONS", [fake])
    monkeypatch.setattr(index, "FORKB_CODE_VERSION", 2)

    conn = index.open_forkB()  # open() runs migrate_forkB at target 2
    cols = {r[1] for r in conn.execute("PRAGMA table_info(Activity);").fetchall()}
    assert "Reviewed" in cols
    assert _scalar(conn, "PRAGMA user_version;") == 2

    # Re-run: column-existence guard + already-bumped version → clean no-op.
    index.migrate_forkB(conn)
    assert _scalar(conn, "PRAGMA user_version;") == 2
    conn.close()


def test_class2_in_window_copy(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A Class-2 create-new + INSERT…SELECT WHERE Ts>=cutoff copies in-window rows.

    Seed Activity at v1 with one in-window and one stale row, then register a
    Class-2 step that rebuilds Activity copying only ``Ts >= cutoff``; assert the
    stale row is dropped and the in-window row survives.
    """
    cutoff = 1_000
    conn0 = index.open_forkB()
    with index.write_tx(conn0):
        conn0.execute(
            "INSERT INTO Activity(SessionId, Ts, Role, Kind, Body) "
            "VALUES ('s', 500, 'user', 'prompt', 'stale');"
        )
        conn0.execute(
            "INSERT INTO Activity(SessionId, Ts, Role, Kind, Body) "
            "VALUES ('s', 2000, 'user', 'prompt', 'fresh');"
        )
    conn0.close()

    def rebuild(conn: sqlite3.Connection) -> None:
        # Discrete execute() calls only — NOT executescript, which would
        # auto-commit and break the framework's single-transaction invariant
        # (§3.13: schema + user_version must flip atomically).
        conn.execute(
            "CREATE TABLE Activity_new ("
            "Id INTEGER PRIMARY KEY, SessionId TEXT NOT NULL, Ts INTEGER NOT NULL, "
            "Role TEXT NOT NULL, Kind TEXT NOT NULL, Body TEXT, ToolRef TEXT, "
            "FullLen INTEGER NOT NULL DEFAULT 0, "
            "Truncated INTEGER NOT NULL DEFAULT 0) STRICT;"
        )
        conn.execute(
            "INSERT INTO Activity_new(Id, SessionId, Ts, Role, Kind, Body, "
            "ToolRef, FullLen, Truncated) "
            "SELECT Id, SessionId, Ts, Role, Kind, Body, ToolRef, FullLen, "
            "Truncated FROM Activity WHERE Ts >= ?;",
            (cutoff,),
        )
        conn.execute("DROP TABLE Activity;")
        conn.execute("ALTER TABLE Activity_new RENAME TO Activity;")
        conn.execute("CREATE INDEX IF NOT EXISTS IX_Activity_Ts ON Activity(Ts);")

    fake = index.Migration(target=2, cls=2, apply=rebuild)
    monkeypatch.setattr(index, "_MIGRATIONS", [fake])
    monkeypatch.setattr(index, "FORKB_CODE_VERSION", 2)

    conn = index.open_forkB()  # runs the Class-2 migration on open
    bodies = {r[0] for r in conn.execute("SELECT Body FROM Activity;").fetchall()}
    assert bodies == {"fresh"}
    assert _scalar(conn, "PRAGMA user_version;") == 2
    conn.close()


def test_never_error_fallback_recreates_forkb(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A failing migration step must NOT raise; forkB is recreated + usable."""
    # Seed a row at v1 first.
    conn0 = index.open_forkB()
    with index.write_tx(conn0):
        conn0.execute(
            "INSERT INTO Activity(SessionId, Ts, Role, Kind, Body) "
            "VALUES ('s', 2000, 'user', 'prompt', 'doomed');"
        )
    conn0.close()

    def boom(_conn: sqlite3.Connection) -> None:
        raise RuntimeError("migration blew up")

    fake = index.Migration(target=2, cls=1, apply=boom)
    monkeypatch.setattr(index, "_MIGRATIONS", [fake])
    monkeypatch.setattr(index, "FORKB_CODE_VERSION", 2)

    # open_forkB calls migrate_forkB; the failure must be swallowed (no raise).
    conn = index.open_forkB()
    # forkB usable afterward: schema present, version set, rows wiped (SC-3 cost).
    assert {"Activity", "Cursor"} <= _objects(conn, "table")
    assert _scalar(conn, "SELECT COUNT(*) FROM Activity;") == 0
    assert _scalar(conn, "PRAGMA user_version;") == 2
    # And still writable.
    with index.write_tx(conn):
        conn.execute(
            "INSERT INTO Activity(SessionId, Ts, Role, Kind, Body) "
            "VALUES ('s', 3000, 'user', 'prompt', 'after');"
        )
    assert _scalar(conn, "SELECT COUNT(*) FROM Activity;") == 1
    conn.close()
