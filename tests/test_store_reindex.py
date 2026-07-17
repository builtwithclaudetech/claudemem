"""Tests for claudemem.store.reindex (T2.7) — the §3.9 rebuild/flush phase.

Covers the SC-4 files-as-truth round-trip (every IN-1 field re-derived from the
markdown into the rebuilt index), the §3.9 atomic-swap mechanics (sidecar gone,
no stale ``-wal``/``-shm`` from the old inode), crash safety (a mid-rebuild crash
leaves the live ``forkA.db`` untouched/valid and a later clean rebuild removes the
orphan sidecar and succeeds), hand-deletion reconciliation (NG-7/C-11), the
EnrichPending marking of summary-less records, and the absence of any
mismatched-WAL corruption after the swap (the step-3 fix).

``CLAUDEMEM_HOME`` is pointed at ``tmp_path`` so no test touches the real
``~/.claude``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from claudemem import config, files, index
from claudemem.store import forka, reindex


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME at a tmp dir; return it (no DB files written yet)."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    return tmp_path


def _scope(home: Path) -> config.ScopeContext:
    """A project scope whose global + project dirs live under the tmp home.

    Both dirs are under ``home`` (not the real ``~/.claude``) so ``iter_records``
    only ever sees files this test wrote.
    """
    return config.ScopeContext(
        kind="project",
        project_id="proj-1",
        global_dir=home / "memory",
        project_dir=home / "projects" / "proj-1" / "memory",
    )


def _write_record(
    directory: Path,
    name: str,
    *,
    scope: str,
    summary: str | None = "a useful summary",
    aliases: list[str] | None = None,
    body: str = "the body text",
    pinned: bool = False,
    importance: int = 3,
    superseded_by: str | None = None,
    stale: bool = False,
    access_count: int = 2,
    hit_count: int = 1,
    created: str = "2026-05-29T12:00:00Z",
    last_accessed: str = "2026-05-30T08:30:00Z",
) -> files.RecordFile:
    """Write one Fork A markdown file into ``directory`` and return its RecordFile."""
    record = files.RecordFile(
        path=directory / f"{name}.md",
        name=name,
        type="reference",
        scope=scope,
        importance=importance,
        pinned=pinned,
        source="explicit",
        created=created,
        last_accessed=last_accessed,
        access_count=access_count,
        hit_count=hit_count,
        summary=summary,
        aliases=aliases if aliases is not None else ["alpha", "image generation"],
        superseded_by=superseded_by,
        stale=stale,
        body=body,
    )
    files.write_record(record)
    return record


# --------------------------------------------------------------------------- #
# SC-4 core — files-as-truth round-trip                                         #
# --------------------------------------------------------------------------- #


def test_rebuild_round_trips_every_in1_field(home: Path) -> None:
    """Every IN-1 frontmatter field of every file lands in the rebuilt index."""
    scope = _scope(home)
    _write_record(
        scope.global_dir,
        "global-rec",
        scope="global",
        summary="global summary",
        aliases=["g-alias", "comma, alias"],
        body="global body",
        pinned=True,
        importance=5,
        stale=True,
        access_count=7,
        hit_count=4,
    )
    _write_record(
        scope.project_dir,
        "proj-rec",
        scope="project",
        summary="project summary",
        aliases=["p-alias"],
        body="project body",
    )

    result = reindex.rebuild_index(scope)
    assert result.records == 2

    global_ctx = config.ScopeContext(
        kind="global", project_id=None, global_dir=scope.global_dir, project_dir=None
    )
    conn = index.open_forkA()
    try:
        rec = forka.select_record(conn, global_ctx, "global-rec")
        assert rec is not None
        assert rec.summary == "global summary"
        assert rec.body == "global body"
        assert rec.pinned == 1
        assert rec.importance == 5
        assert rec.stale == 1
        assert rec.access_count == 7
        assert rec.hit_count == 4
        # Timestamps crossed ISO -> epoch (MF-1).
        assert rec.created == files.iso_to_epoch("2026-05-29T12:00:00Z")
        assert rec.last_accessed == files.iso_to_epoch("2026-05-30T08:30:00Z")
        # Alias target round-trips through AliasesJson (commas/quotes preserved).
        assert files.aliases_from_json(rec.aliases_json) == ["g-alias", "comma, alias"]

        proj = forka.select_record(conn, scope, "proj-rec")
        assert proj is not None
        assert proj.summary == "project summary"
        assert files.aliases_from_json(proj.aliases_json) == ["p-alias"]
    finally:
        conn.close()


def test_rebuild_record_is_searchable(home: Path) -> None:
    """A rebuilt record is reachable through the FTS5 candidate query (triggers ran)."""
    scope = _scope(home)
    _write_record(
        scope.project_dir,
        "searchable",
        scope="project",
        summary="quantum widget calibration",
        body="details about the widget",
    )
    reindex.rebuild_index(scope)

    conn = index.open_forkA()
    try:
        hits = forka.fts_candidates(conn, "quantum widget", scope)
        assert len(hits) == 1
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# EnrichPending marking                                                         #
# --------------------------------------------------------------------------- #


def test_summary_less_record_marked_enrich_pending(home: Path) -> None:
    """A file with no summary is flagged EnrichPending=1 for Phase-B backfill."""
    scope = _scope(home)
    _write_record(scope.project_dir, "no-summary", scope="project", summary=None)
    _write_record(scope.project_dir, "has-summary", scope="project", summary="ok")

    result = reindex.rebuild_index(scope)
    assert result.enrich_pending == 1

    conn = index.open_forkA()
    try:
        no_sum = forka.select_record(conn, scope, "no-summary")
        has_sum = forka.select_record(conn, scope, "has-summary")
        assert no_sum is not None and no_sum.enrich_pending == 1
        assert has_sum is not None and has_sum.enrich_pending == 0
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Atomic swap mechanics                                                         #
# --------------------------------------------------------------------------- #


def test_atomic_swap_leaves_clean_filesystem(home: Path) -> None:
    """After a rebuild: forkA.db exists, sidecar gone, no stale -wal/-shm."""
    scope = _scope(home)
    _write_record(scope.project_dir, "rec", scope="project")

    reindex.rebuild_index(scope)

    live = index.forka_path()
    sidecar = live.with_name(live.name + ".rebuild")
    assert live.exists()
    assert not sidecar.exists()
    assert not Path(str(sidecar) + "-wal").exists()
    assert not Path(str(sidecar) + "-shm").exists()
    # The freshly-built forkA.db was checkpoint+closed clean in step 2/the swap,
    # and any old-inode -wal/-shm were removed in step 5.
    assert not Path(str(live) + "-wal").exists()
    assert not Path(str(live) + "-shm").exists()


def test_rebuild_over_existing_live_db(home: Path) -> None:
    """A second rebuild over an existing live DB (with a real WAL) succeeds + swaps.

    Exercises step 3 against a populated live DB that has an active WAL.
    """
    scope = _scope(home)
    _write_record(scope.project_dir, "v1", scope="project", body="first")
    reindex.rebuild_index(scope)

    # Touch the live DB so it has an undrained WAL at the next swap.
    conn = index.open_forkA()
    try:
        with index.write_tx(conn):
            conn.execute("UPDATE Record SET HitCount = HitCount + 1;")
    finally:
        conn.close()

    # Add a second file, rebuild again.
    _write_record(scope.project_dir, "v2", scope="project", body="second")
    result = reindex.rebuild_index(scope)
    assert result.records == 2

    # No mismatched-WAL corruption: the swapped DB opens + reads cleanly.
    conn = index.open_forkA()
    try:
        names = {r.name for r in forka.active_set(conn, scope)}
        assert names == {"v1", "v2"}
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Crash safety                                                                  #
# --------------------------------------------------------------------------- #


def test_crash_before_swap_leaves_live_db_intact(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash injected before os.replace leaves the live forkA.db untouched."""
    scope = _scope(home)
    _write_record(scope.project_dir, "original", scope="project", body="original body")
    reindex.rebuild_index(scope)  # establish a valid live index

    # Add a new file, then inject a failure at the os.replace boundary.
    _write_record(scope.project_dir, "newcomer", scope="project")

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated crash before swap")

    monkeypatch.setattr(reindex.os, "replace", _boom)
    with pytest.raises(RuntimeError, match="simulated crash"):
        reindex.rebuild_index(scope)

    # The live DB is untouched: still the pre-crash content (no "newcomer").
    conn = index.open_forkA()
    try:
        names = {r.name for r in forka.active_set(conn, scope)}
        assert names == {"original"}
    finally:
        conn.close()


def test_clean_rebuild_after_crash_removes_orphan_and_succeeds(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """After a crashed rebuild, a subsequent clean rebuild clears the orphan + works."""
    scope = _scope(home)
    _write_record(scope.project_dir, "original", scope="project")
    reindex.rebuild_index(scope)

    _write_record(scope.project_dir, "newcomer", scope="project")

    def _boom(*_a: object, **_k: object) -> None:
        raise RuntimeError("simulated crash before swap")

    sidecar = index.forka_path().with_name(index.forka_path().name + ".rebuild")
    with monkeypatch.context() as patched:
        patched.setattr(reindex.os, "replace", _boom)
        with pytest.raises(RuntimeError):
            reindex.rebuild_index(scope)
        # A crashed run left the sidecar behind.
        assert sidecar.exists()

    # Clean rebuild (os.replace restored): orphan removed at start, swap succeeds.
    result = reindex.rebuild_index(scope)
    assert result.records == 2
    assert not sidecar.exists()

    conn = index.open_forkA()
    try:
        names = {r.name for r in forka.active_set(conn, scope)}
        assert names == {"original", "newcomer"}
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# Hand-deletion reconciliation (NG-7/C-11)                                      #
# --------------------------------------------------------------------------- #


def test_hand_deleted_file_absent_from_rebuilt_index(home: Path) -> None:
    """A file removed before reindex is simply absent from the rebuilt index."""
    scope = _scope(home)
    keep = _write_record(scope.project_dir, "keep", scope="project")
    gone = _write_record(scope.project_dir, "gone", scope="project")
    reindex.rebuild_index(scope)

    # Hand-delete one file, rebuild.
    gone.path.unlink()
    result = reindex.rebuild_index(scope)
    assert result.records == 1

    conn = index.open_forkA()
    try:
        assert forka.select_record(conn, scope, "gone") is None
        assert forka.select_record(conn, scope, "keep") is not None
        _ = keep
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# No mismatched-WAL corruption (the step-3 fix)                                 #
# --------------------------------------------------------------------------- #


def test_no_wal_replay_error_after_rebuild(home: Path) -> None:
    """Opening forkA.db after a rebuild raises no WAL-replay/corruption error."""
    scope = _scope(home)
    _write_record(scope.project_dir, "rec", scope="project")
    reindex.rebuild_index(scope)

    live = index.forka_path()
    # A raw open + integrity check: a mismatched WAL would surface here.
    raw = sqlite3.connect(live)
    try:
        result = raw.execute("PRAGMA integrity_check;").fetchone()
        assert result is not None and result[0] == "ok"
        count = raw.execute("SELECT count(*) FROM Record;").fetchone()[0]
        assert count == 1
    finally:
        raw.close()


def test_first_ever_rebuild_with_no_live_db(home: Path) -> None:
    """A first-ever rebuild (no pre-existing forkA.db) builds + swaps cleanly."""
    scope = _scope(home)
    _write_record(scope.project_dir, "rec", scope="project")
    assert not index.forka_path().exists()

    result = reindex.rebuild_index(scope)
    assert result.records == 1
    assert index.forka_path().exists()
