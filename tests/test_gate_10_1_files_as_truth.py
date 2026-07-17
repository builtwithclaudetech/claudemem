"""§10.1 files-as-truth round-trip GATE — certifies SC-4 (T2.8).

This is the deterministic build-phase **gate** that discharges PRD **SC-4**
("Files-as-truth round-trip"): delete ``forkA.db``, run the §3.9 reindex, and
prove that the rebuilt index is a faithful, lossless reconstruction of the Fork A
markdown files (the source of truth, G-4/C-11). It is a hard GREEN gate — no
``xfail`` — because a failure here is a real round-trip defect in
``files``/``forka``/``reindex``, not a known gap.

What the gate asserts (mapped to SC-4 + tech-design §10.1):

* **(a) searchable + active set (SC-4a / SC-12 / SC-7).** Every *active* Fork A
  file is retrievable post-reindex — both through the model-free FTS5 candidate
  query (:func:`store.forka.fts_candidates`, the §4.3 "returns in search" proxy
  for the not-yet-built Phase-3 ``recall.search``) and present in the scope-merged
  active set (:func:`store.forka.active_set`). The *superseded* record is NOT in
  the active set but is still selectable by name (the SC-7 supersede trail).
* **(b) every IN-1 field round-trips (SC-4b).** For every seeded file,
  :func:`store.forka.select_record` returns a row whose every IN-1 frontmatter
  field equals the file's value — one assertion per field, exhaustively:
  ``type``/``scope``/``importance``/``pinned``/``source``/``created`` (ISO→epoch
  exact)/``last_accessed`` (ISO→epoch exact)/``access_count``/``hit_count``/
  ``summary``/``superseded_by``/``stale``, plus the alias list reconstructed from
  ``AliasesJson`` (§3.11; the multi-word and comma-bearing aliases are the
  Phase-1-fix lock-in).
* **No Fork A data loss.** Rebuilt-record count == seeded-file count, and each
  record's body matches the file.
* **Idempotent re-reindex.** A second ``rebuild_index`` produces a byte-identical
  index (no duplication, no drift) — SC-11-adjacent re-runnability of the rebuild.

Determinism: fixed ``CLAUDEMEM_HOME`` under ``tmp_path``, fixed ISO-8601
timestamps chosen to round-trip exactly at second grain, fixed scope.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from claudemem import config, files, index
from claudemem.store import forka, reindex

# --------------------------------------------------------------------------- #
# Fixed, deterministic timestamps (second-grain; round-trip exactly via MF-1). #
# --------------------------------------------------------------------------- #

_CREATED_ISO = "2026-01-02T03:04:05Z"
_LAST_ACCESSED_ISO = "2026-05-29T12:00:00Z"

# The distinctive alias list for the fully-populated record: a multi-word alias
# ("image generation") tokenizes apart in AliasesFlat but reconstructs verbatim
# from AliasesJson; the comma-bearing alias ("comma, separated") locks in the
# Phase-1 JSON-array fix (a bare inline list would split it at the comma).
_FULL_ALIASES = ["image generation", "comma, separated", "img-gen"]


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME at a tmp dir; return it (no DB files written yet)."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    return tmp_path


def _scope(home: Path) -> config.ScopeContext:
    """A project scope whose global + project dirs live under the tmp home.

    Both dirs are under ``home`` (never the real ``~/.claude``) so
    ``iter_records`` only ever sees files this gate wrote.
    """
    return config.ScopeContext(
        kind="project",
        project_id="proj-1",
        global_dir=home / "memory",
        project_dir=home / "projects" / "proj-1" / "memory",
    )


def _global_ctx(scope: config.ScopeContext) -> config.ScopeContext:
    """The global-scope view of ``scope`` (for selecting global-scope records)."""
    return config.ScopeContext(
        kind="global",
        project_id=None,
        global_dir=scope.global_dir,
        project_dir=None,
    )


def _seed_record(
    directory: Path,
    name: str,
    *,
    scope: str,
    type: str = "reference",
    importance: int = 3,
    pinned: bool = False,
    source: str = "explicit",
    created: str = _CREATED_ISO,
    last_accessed: str = _LAST_ACCESSED_ISO,
    access_count: int = 0,
    hit_count: int = 0,
    summary: str | None = "a useful summary",
    aliases: list[str] | None = None,
    superseded_by: str | None = None,
    stale: bool = False,
    body: str = "the body text",
) -> files.RecordFile:
    """Write one Fork A markdown file and return its :class:`files.RecordFile`.

    A thin, readable seeding helper so each assertion can be checked against the
    exact value written (the assertions below compare against these arguments).
    """
    record = files.RecordFile(
        path=directory / f"{name}.md",
        name=name,
        type=type,
        scope=scope,
        importance=importance,
        pinned=pinned,
        source=source,
        created=created,
        last_accessed=last_accessed,
        access_count=access_count,
        hit_count=hit_count,
        summary=summary,
        aliases=aliases if aliases is not None else [],
        superseded_by=superseded_by,
        stale=stale,
        body=body,
    )
    files.write_record(record)
    return record


def _seed_corpus(scope: config.ScopeContext) -> dict[str, files.RecordFile]:
    """Seed the realistic gate corpus; return ``{name: RecordFile}`` for assertions.

    Four records spanning every dimension the gate must certify:

    * ``full-global`` — a **global** record with EVERY IN-1 field set to a
      distinctive non-default value (pinned, stale, importance 5, source
      ``session``, type ``user``, access_count 7, hit_count 4, multi-word + comma
      aliases). The exhaustive per-field round-trip target.
    * ``minimal-proj`` — a **project** record with only the required fields
      (no summary, no aliases, default counters/flags).
    * ``active-proj`` — a plain active project record (searchable corpus member).
    * ``superseded-proj`` — a project record superseded by ``active-proj`` (the
      SC-7 trail target: selectable by name, absent from the active set).
    """
    return {
        "full-global": _seed_record(
            scope.global_dir,
            "full-global",
            scope="global",
            type="user",
            importance=5,
            pinned=True,
            source="session",
            created=_CREATED_ISO,
            last_accessed=_LAST_ACCESSED_ISO,
            access_count=7,
            hit_count=4,
            summary="comprehensive enrichment summary",
            aliases=_FULL_ALIASES,
            superseded_by=None,
            stale=True,
            body="full global body about photosynthesis pipelines",
        ),
        "minimal-proj": _seed_record(
            scope.project_dir,
            "minimal-proj",
            scope="project",
            summary=None,
            aliases=[],
            body="minimal project body about turbines",
        ),
        "active-proj": _seed_record(
            scope.project_dir,
            "active-proj",
            scope="project",
            summary="active project summary",
            aliases=["alpha"],
            body="active project body about kubernetes scheduling",
        ),
        "superseded-proj": _seed_record(
            scope.project_dir,
            "superseded-proj",
            scope="project",
            summary="superseded project summary",
            aliases=["beta"],
            superseded_by="active-proj",
            body="superseded project body about deprecated cron jobs",
        ),
    }


def _delete_forka_db() -> None:
    """Delete ``forkA.db`` (+ its ``-wal``/``-shm``) — the SC-4 'lost index' step."""
    live = index.forka_path()
    base = str(live)
    for suffix in ("", "-wal", "-shm"):
        Path(base + suffix).unlink(missing_ok=True)


# --------------------------------------------------------------------------- #
# The gate.                                                                     #
# --------------------------------------------------------------------------- #


def test_gate_files_as_truth_round_trip(home: Path) -> None:
    """SC-4 GATE: delete forkA.db → reindex → assert no Fork A data loss.

    The single, exhaustive gate. It seeds the corpus, builds an index, **deletes
    ``forkA.db``** (simulating a lost SQLite cache), rebuilds from files, and then
    asserts (a) searchability/active-set + (b) every IN-1 field round-trips + no
    data loss. Kept as one test so the 'delete → reindex → assert the whole
    contract' flow is read top-to-bottom as the gate it certifies.
    """
    scope = _scope(home)
    seeded = _seed_corpus(scope)
    global_ctx = _global_ctx(scope)

    # Build the index once so there is a real forkA.db to lose.
    index.open_forkA().close()
    reindex.rebuild_index(scope)
    assert index.forka_path().exists()

    # ---- SC-4 step: DELETE forkA.db, then reindex from files. ----------------
    _delete_forka_db()
    assert not index.forka_path().exists()
    result = reindex.rebuild_index(scope)

    # ---- No Fork A data loss: rebuilt count == seeded file count. ------------
    assert result.records == len(seeded) == 4

    conn = index.open_forkA()
    try:
        # ---- (a) searchable + active-set membership (SC-4a / SC-12 / SC-7). --
        # Every ACTIVE record returns via the model-free FTS5 candidate query
        # (the §4.3 "returns in search" proxy) on a distinctive body/summary token.
        full = forka.select_record(conn, global_ctx, "full-global")
        assert full is not None
        assert _matches(conn, "photosynthesis", scope, full.id)
        assert _matches(conn, "enrichment", scope, full.id)  # summary token

        minimal = forka.select_record(conn, scope, "minimal-proj")
        assert minimal is not None
        assert _matches(conn, "turbines", scope, minimal.id)

        active = forka.select_record(conn, scope, "active-proj")
        assert active is not None
        assert _matches(conn, "kubernetes", scope, active.id)

        # Active set: the three active records present; the superseded one absent.
        active_names = {r.name for r in forka.active_set(conn, scope)}
        assert active_names == {"full-global", "minimal-proj", "active-proj"}
        assert "superseded-proj" not in active_names

        # SC-7 trail: the superseded record is still selectable by name (history
        # preserved) even though it dropped out of the active set.
        superseded = forka.select_record(conn, scope, "superseded-proj")
        assert superseded is not None
        assert superseded.superseded_by == "active-proj"

        # ---- (b) every IN-1 field round-trips on the fully-populated record. -
        # One assertion per IN-1 field — a missing/wrong field is a gate failure.
        src = seeded["full-global"]
        assert full.type == src.type == "user"
        assert full.scope == src.scope == "global"
        assert full.importance == src.importance == 5
        assert full.pinned == 1 and src.pinned is True          # bool -> 0/1
        assert full.source == src.source == "session"
        assert full.created == files.iso_to_epoch(src.created)  # ISO -> epoch
        assert full.last_accessed == files.iso_to_epoch(src.last_accessed)
        assert full.access_count == src.access_count == 7
        assert full.hit_count == src.hit_count == 4
        assert full.summary == src.summary == "comprehensive enrichment summary"
        assert full.superseded_by is None and src.superseded_by is None
        assert full.stale == 1 and src.stale is True            # bool -> 0/1
        # Alias list reconstructs exactly from AliasesJson (§3.11), incl. the
        # multi-word and comma-bearing aliases (the Phase-1 fix lock-in).
        assert files.aliases_from_json(full.aliases_json) == _FULL_ALIASES
        assert files.aliases_from_json(full.aliases_json) == src.aliases

        # ---- No data loss: body matches the file for every seeded record. ----
        for name, record_file in seeded.items():
            ctx = global_ctx if record_file.scope == "global" else scope
            row = forka.select_record(conn, ctx, name)
            assert row is not None, f"{name} missing from rebuilt index"
            assert row.body == record_file.body, f"{name} body mismatch"

        # ---- IN-1 fields on the MINIMAL record (defaults round-trip too). ----
        assert minimal.type == "reference"
        assert minimal.scope == "project"
        assert minimal.importance == 3
        assert minimal.pinned == 0
        assert minimal.source == "explicit"
        assert minimal.access_count == 0
        assert minimal.hit_count == 0
        assert minimal.summary is None
        assert minimal.superseded_by is None
        assert minimal.stale == 0
        assert files.aliases_from_json(minimal.aliases_json) == []
    finally:
        conn.close()


def test_gate_reindex_is_idempotent(home: Path) -> None:
    """A second reindex yields a byte-identical index — no duplication / drift.

    SC-11-adjacent re-runnability of the rebuild: after the SC-4 delete→reindex,
    running ``rebuild_index`` again must reproduce the exact same logical content
    (same record count, same per-record column tuple). Compared via a stable
    snapshot of every Record row (ordered) so any duplication, dropped row, or
    field drift surfaces as an inequality.
    """
    scope = _scope(home)
    _seed_corpus(scope)

    reindex.rebuild_index(scope)
    first = _record_snapshot(index.forka_path())

    second_result = reindex.rebuild_index(scope)
    second = _record_snapshot(index.forka_path())

    assert second_result.records == 4
    assert first == second


# --------------------------------------------------------------------------- #
# Helpers.                                                                       #
# --------------------------------------------------------------------------- #


def _matches(
    conn: sqlite3.Connection, token: str, scope: config.ScopeContext, record_id: int
) -> bool:
    """Whether ``record_id`` is among the FTS5 candidates for ``token`` (§4.3)."""
    return any(rid == record_id for rid, _raw in forka.fts_candidates(conn, token, scope))


def _record_snapshot(db_path: Path) -> list[tuple[object, ...]]:
    """A stable, ordered snapshot of every Record row's persisted columns.

    Excludes the autoincrement ``Id`` (a re-derived rowid is not part of the
    files-as-truth contract) and orders by the natural key so two independent
    rebuilds compare equal iff they hold the same logical records with the same
    field values — the idempotency check.
    """
    raw = sqlite3.connect(db_path)
    try:
        return raw.execute(
            "SELECT Name, Scope, ProjectId, Type, Importance, Pinned, Source, "
            "Created, LastAccessed, AccessCount, HitCount, Summary, AliasesJson, "
            "AliasesFlat, SupersededBy, Stale, EnrichPending, Body "
            "FROM Record ORDER BY Scope, ProjectId, Name;"
        ).fetchall()
    finally:
        raw.close()
