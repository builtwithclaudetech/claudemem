"""Tests for claudemem.store.forka (T2.1 + T2.2).

Covers: ``upsert_record`` → ``select_record`` round-trips every §3.2/§3.12 column
including the ISO → epoch timestamp boundary (MF-1) and the §3.11 alias dual-form;
upsert is an update (not a duplicate insert) for both project and global scope
(the NULL-``ProjectId`` UNIQUE gotcha); ``mark_superseded`` drops a record from the
active set while the row survives (SC-7 trail); ``active_set`` scope-merges global +
project and excludes superseded (§3.8/SC-12); and ``fts_candidates`` — the model-free
§4.3 lexical primitive — matches the right rows, is **injection-safe** against FTS5
operators/quotes, returns ``[]`` for empty/punctuation queries, respects the scope
merge + ``LIMIT k``, and orders better matches first (§4.1 bm25 ascending).

The claudemem home dir is pointed at ``tmp_path`` via ``CLAUDEMEM_HOME`` so no test
touches the real ``~/.claude``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from claudemem import config, files, index
from claudemem.store import forka

NOW_EPOCH = files.iso_to_epoch("2026-05-30T12:00:00Z")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME at a tmp dir; return it (no DB files written yet)."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def conn(home: Path) -> sqlite3.Connection:
    """An open forkA connection with the schema ensured."""
    return index.open_forkA()


def _project_scope(pid: str = "proj-1") -> config.ScopeContext:
    return config.ScopeContext(
        kind="project",
        project_id=pid,
        global_dir=Path("/g"),
        project_dir=Path("/p"),
    )


def _global_scope() -> config.ScopeContext:
    return config.ScopeContext(
        kind="global", project_id=None, global_dir=Path("/g"), project_dir=None
    )


def _record_file(
    name: str,
    *,
    summary: str | None = "a summary",
    aliases: list[str] | None = None,
    body: str = "the body text",
    pinned: bool = False,
    importance: int = 3,
    superseded_by: str | None = None,
    created: str = "2026-05-29T12:00:00Z",
    last_accessed: str = "2026-05-30T08:30:00Z",
) -> files.RecordFile:
    return files.RecordFile(
        path=Path(f"/tmp/{name}.md"),
        name=name,
        type="reference",
        scope="project",
        importance=importance,
        pinned=pinned,
        source="explicit",
        created=created,
        last_accessed=last_accessed,
        access_count=2,
        hit_count=1,
        summary=summary,
        aliases=aliases if aliases is not None else ["alpha", "image generation"],
        superseded_by=superseded_by,
        stale=False,
        body=body,
    )


# --------------------------------------------------------------------------- #
# upsert / select round-trip (T2.1)                                            #
# --------------------------------------------------------------------------- #


def test_upsert_select_round_trips_all_columns(conn: sqlite3.Connection) -> None:
    scope = _project_scope()
    rf = _record_file("greeting")
    row_id = forka.upsert_record(conn, rf, scope)
    assert row_id > 0

    rec = forka.select_record(conn, scope, "greeting")
    assert rec is not None
    assert rec.id == row_id
    assert rec.name == "greeting"
    assert rec.scope == "project"
    assert rec.project_id == "proj-1"
    assert rec.type == "reference"
    assert rec.importance == 3
    assert rec.pinned == 0
    assert rec.source == "explicit"
    # ISO → epoch boundary (MF-1): column holds epoch seconds.
    assert rec.created == files.iso_to_epoch("2026-05-29T12:00:00Z")
    assert rec.last_accessed == files.iso_to_epoch("2026-05-30T08:30:00Z")
    assert rec.access_count == 2
    assert rec.hit_count == 1
    assert rec.summary == "a summary"
    # Alias dual-form (§3.11): canonical JSON mirror + space-joined flat form.
    assert files.aliases_from_json(rec.aliases_json) == ["alpha", "image generation"]
    assert rec.aliases_flat == "alpha image generation"
    assert rec.superseded_by is None
    assert rec.stale == 0
    assert rec.enrich_pending == 0
    assert rec.body == "the body text"


def test_upsert_enrich_pending_flag(conn: sqlite3.Connection) -> None:
    scope = _project_scope()
    forka.upsert_record(conn, _record_file("deg"), scope, enrich_pending=True)
    rec = forka.select_record(conn, scope, "deg")
    assert rec is not None
    assert rec.enrich_pending == 1


def test_upsert_is_update_not_duplicate_project_scope(
    conn: sqlite3.Connection,
) -> None:
    scope = _project_scope()
    first = forka.upsert_record(conn, _record_file("dup", body="v1"), scope)
    second = forka.upsert_record(conn, _record_file("dup", body="v2"), scope)
    assert first == second
    count = conn.execute(
        "SELECT count(*) FROM Record WHERE Name = 'dup';"
    ).fetchone()[0]
    assert count == 1
    rec = forka.select_record(conn, scope, "dup")
    assert rec is not None and rec.body == "v2"


def test_upsert_is_update_not_duplicate_global_scope(
    conn: sqlite3.Connection,
) -> None:
    # The NULL-ProjectId gotcha: a plain ON CONFLICT upsert would insert a
    # duplicate here because NULLs are distinct in the UNIQUE index.
    scope = _global_scope()
    rf = files.RecordFile(
        path=Path("/tmp/g.md"), name="g", type="reference", scope="global",
        importance=3, pinned=False, source="explicit",
        created="2026-05-29T12:00:00Z", last_accessed="2026-05-29T12:00:00Z",
        access_count=0, hit_count=0, summary=None, aliases=[],
        superseded_by=None, stale=False, body="g1",
    )
    first = forka.upsert_record(conn, rf, scope)
    from dataclasses import replace

    second = forka.upsert_record(conn, replace(rf, body="g2"), scope)
    assert first == second
    count = conn.execute(
        "SELECT count(*) FROM Record WHERE Name = 'g';"
    ).fetchone()[0]
    assert count == 1
    rec = forka.select_record(conn, scope, "g")
    assert rec is not None and rec.body == "g2" and rec.project_id is None


def test_select_missing_returns_none(conn: sqlite3.Connection) -> None:
    assert forka.select_record(conn, _project_scope(), "nope") is None


# --------------------------------------------------------------------------- #
# mark_superseded — soft delete (SC-7/C-10/IN-8)                                #
# --------------------------------------------------------------------------- #


def test_mark_superseded_drops_from_active_but_row_survives(
    conn: sqlite3.Connection,
) -> None:
    scope = _project_scope()
    forka.upsert_record(conn, _record_file("old"), scope)
    forka.mark_superseded(conn, "old", "new", scope)

    # Row still exists (SC-7 trail) ...
    rec = forka.select_record(conn, scope, "old")
    assert rec is not None
    assert rec.superseded_by == "new"
    # ... but is excluded from the active set.
    active_names = {r.name for r in forka.active_set(conn, scope)}
    assert "old" not in active_names


def test_mark_superseded_none_reactivates(conn: sqlite3.Connection) -> None:
    scope = _project_scope()
    forka.upsert_record(conn, _record_file("rec"), scope)
    forka.mark_superseded(conn, "rec", "x", scope)
    forka.mark_superseded(conn, "rec", None, scope)
    active_names = {r.name for r in forka.active_set(conn, scope)}
    assert "rec" in active_names


# --------------------------------------------------------------------------- #
# active_set — scope merge + active filter (§3.8/SC-12)                         #
# --------------------------------------------------------------------------- #


def test_active_set_scope_merges_global_and_project(
    conn: sqlite3.Connection,
) -> None:
    proj = _project_scope("proj-1")
    other = _project_scope("proj-2")
    glob = _global_scope()

    forka.upsert_record(conn, _record_file("p1"), proj)
    forka.upsert_record(conn, _record_file("p2"), other)
    g = files.RecordFile(
        path=Path("/tmp/g.md"), name="g1", type="reference", scope="global",
        importance=3, pinned=False, source="explicit",
        created="2026-05-29T12:00:00Z", last_accessed="2026-05-29T12:00:00Z",
        access_count=0, hit_count=0, summary=None, aliases=[],
        superseded_by=None, stale=False, body="b",
    )
    forka.upsert_record(conn, g, glob)

    names = {r.name for r in forka.active_set(conn, proj)}
    assert names == {"p1", "g1"}  # own project + global; NOT proj-2's record


def test_active_set_global_scope_excludes_project_rows(
    conn: sqlite3.Connection,
) -> None:
    forka.upsert_record(conn, _record_file("p1"), _project_scope("proj-1"))
    g = files.RecordFile(
        path=Path("/tmp/g.md"), name="g1", type="reference", scope="global",
        importance=3, pinned=False, source="explicit",
        created="2026-05-29T12:00:00Z", last_accessed="2026-05-29T12:00:00Z",
        access_count=0, hit_count=0, summary=None, aliases=[],
        superseded_by=None, stale=False, body="b",
    )
    forka.upsert_record(conn, g, _global_scope())
    names = {r.name for r in forka.active_set(conn, _global_scope())}
    assert names == {"g1"}


def test_active_set_excludes_superseded(conn: sqlite3.Connection) -> None:
    scope = _project_scope()
    forka.upsert_record(conn, _record_file("a"), scope)
    forka.upsert_record(conn, _record_file("b"), scope)
    forka.mark_superseded(conn, "b", "a", scope)
    names = {r.name for r in forka.active_set(conn, scope)}
    assert names == {"a"}


# --------------------------------------------------------------------------- #
# fts_candidates — model-free §4.3 lexical primitive (T2.2)                     #
# --------------------------------------------------------------------------- #


def _seed_searchable(conn: sqlite3.Connection, scope: config.ScopeContext) -> None:
    forka.upsert_record(
        conn,
        _record_file(
            "dragon", summary="about dragons", aliases=["wyrm"],
            body="a tale of a dragon and its hoard",
        ),
        scope,
    )
    forka.upsert_record(
        conn,
        _record_file(
            "garden", summary="about gardens", aliases=["plants"],
            body="a quiet garden with flowers",
        ),
        scope,
    )


def test_fts_candidates_matches_expected_record(conn: sqlite3.Connection) -> None:
    scope = _project_scope()
    _seed_searchable(conn, scope)
    hits = forka.fts_candidates(conn, "dragon", scope)
    ids = [h[0] for h in hits]
    dragon = forka.select_record(conn, scope, "dragon")
    garden = forka.select_record(conn, scope, "garden")
    assert dragon is not None and garden is not None
    assert dragon.id in ids
    assert garden.id not in ids


def test_fts_candidates_returns_raw_bm25_ascending(
    conn: sqlite3.Connection,
) -> None:
    scope = _project_scope()
    # "dragon" appears in Name + Summary + Body of one record (strong) and not
    # the other; a record matching on Name (weight 10) should rank above one
    # matching only on Body (weight 1).
    forka.upsert_record(
        conn, _record_file("dragon", summary="dragon lore", body="dragon dragon"),
        scope,
    )
    forka.upsert_record(
        conn,
        _record_file("notes", summary="misc", aliases=[],
                     body="a passing mention of a dragon once"),
        scope,
    )
    hits = forka.fts_candidates(conn, "dragon", scope)
    assert len(hits) == 2
    # Ascending raw bm25 (more negative = better); the Name-matching record first.
    assert hits[0][1] <= hits[1][1]
    best = forka.select_record(conn, scope, "dragon")
    assert best is not None and hits[0][0] == best.id


def test_fts_candidates_respects_limit_k(conn: sqlite3.Connection) -> None:
    scope = _project_scope()
    for i in range(5):
        forka.upsert_record(
            conn, _record_file(f"r{i}", body="shared keyword here"), scope
        )
    hits = forka.fts_candidates(conn, "keyword", scope, k=3)
    assert len(hits) == 3


def test_fts_candidates_scope_merge(conn: sqlite3.Connection) -> None:
    proj = _project_scope("proj-1")
    other = _project_scope("proj-2")
    forka.upsert_record(conn, _record_file("mine", body="needle term"), proj)
    forka.upsert_record(conn, _record_file("theirs", body="needle term"), other)
    g = files.RecordFile(
        path=Path("/tmp/g.md"), name="shared", type="reference", scope="global",
        importance=3, pinned=False, source="explicit",
        created="2026-05-29T12:00:00Z", last_accessed="2026-05-29T12:00:00Z",
        access_count=0, hit_count=0, summary=None, aliases=[],
        superseded_by=None, stale=False, body="needle term",
    )
    forka.upsert_record(conn, g, _global_scope())

    found = {h[0] for h in forka.fts_candidates(conn, "needle", proj)}
    mine = forka.select_record(conn, proj, "mine")
    shared = forka.select_record(conn, _global_scope(), "shared")
    theirs = forka.select_record(conn, other, "theirs")
    assert mine is not None and shared is not None and theirs is not None
    assert mine.id in found
    assert shared.id in found  # global merges in
    assert theirs.id not in found  # other project excluded


def test_fts_candidates_excludes_superseded(conn: sqlite3.Connection) -> None:
    scope = _project_scope()
    forka.upsert_record(conn, _record_file("live", body="findme"), scope)
    forka.upsert_record(conn, _record_file("dead", body="findme"), scope)
    forka.mark_superseded(conn, "dead", "live", scope)
    found = {h[0] for h in forka.fts_candidates(conn, "findme", scope)}
    dead = forka.select_record(conn, scope, "dead")
    assert dead is not None and dead.id not in found


# --- injection safety (load-bearing) --------------------------------------- #


@pytest.mark.parametrize(
    "evil_query",
    [
        "foo OR bar",
        '"',
        'a* NEAR b',
        "'); DROP TABLE Record;--",
        "(unbalanced",
        "col:value",
        "NEAR(a b 3)",
        "^prefix",
        "***",
    ],
)
def test_fts_candidates_injection_safe(
    conn: sqlite3.Connection, evil_query: str
) -> None:
    scope = _project_scope()
    _seed_searchable(conn, scope)
    # Must not raise — operators/quotes are neutralized into literal tokens.
    hits = forka.fts_candidates(conn, evil_query, scope)
    assert isinstance(hits, list)
    # And the table is intact (no DROP took effect).
    assert (
        conn.execute("SELECT count(*) FROM Record;").fetchone()[0] == 2
    )


def test_fts_candidates_quoted_term_matches_literally(
    conn: sqlite3.Connection,
) -> None:
    # "dragon OR garden" must be tokenized to the literal terms dragon, OR,
    # garden and still find the dragon record (OR-of-literals), not be parsed as
    # an FTS5 OR operator that errors or behaves specially.
    scope = _project_scope()
    _seed_searchable(conn, scope)
    hits = forka.fts_candidates(conn, "dragon OR garden", scope)
    found = {h[0] for h in hits}
    dragon = forka.select_record(conn, scope, "dragon")
    garden = forka.select_record(conn, scope, "garden")
    assert dragon is not None and garden is not None
    # Both match (OR-of-literal-tokens), neither errors.
    assert dragon.id in found and garden.id in found


@pytest.mark.parametrize("empty_query", ["", "   ", "!!!", "@#$%", "...---..."])
def test_fts_candidates_empty_or_punctuation_returns_empty(
    conn: sqlite3.Connection, empty_query: str
) -> None:
    scope = _project_scope()
    _seed_searchable(conn, scope)
    assert forka.fts_candidates(conn, empty_query, scope) == []


# --------------------------------------------------------------------------- #
# refresh_access (IN-4) + the SC-3 best-effort contract                         #
# --------------------------------------------------------------------------- #


def test_refresh_access_bumps_last_accessed_and_count(conn: sqlite3.Connection) -> None:
    """``refresh_access`` sets ``LastAccessed`` and increments ``AccessCount`` (IN-4)."""
    scope = _project_scope()
    rid = forka.upsert_record(conn, _record_file("notes"), scope)
    before = forka.select_record(conn, scope, "notes")
    assert before is not None

    new_now = files.iso_to_epoch("2026-06-01T00:00:00Z")
    forka.refresh_access(conn, rid, now_epoch=new_now)

    after = forka.select_record(conn, scope, "notes")
    assert after is not None
    assert after.last_accessed == new_now
    assert after.access_count == before.access_count + 1


def test_get_returns_body_when_refresh_access_fails(
    conn: sqlite3.Connection, home: Path
) -> None:
    """SC-3 best-effort: a failed access bump must never sink the read.

    ``recall.get`` opens the Fork A record fine but the in-session refresh hits a
    locked/read-only connection; the body must still be returned. We force the
    failure by opening a *second, read-only* connection to the same forkA.db and
    routing the read through it — the IN-4 ``UPDATE`` raises ``sqlite3.Error``
    which ``recall.get`` swallows, returning the body regardless.
    """
    from claudemem.recall import get as get_mod

    scope = _project_scope()
    forka.upsert_record(conn, _record_file("locked", body="the durable body"), scope)
    conn.commit()

    ro = sqlite3.connect(f"file:{index.forka_path()}?mode=ro", uri=True)
    try:
        out = get_mod.get(ro, conn, "a:locked", scope, now_epoch=NOW_EPOCH)
    finally:
        ro.close()
    assert "the durable body" in out  # read survived the failed access bump
