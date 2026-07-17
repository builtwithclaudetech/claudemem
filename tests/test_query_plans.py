"""T6.4 — EXPLAIN QUERY PLAN build-time validation (tech-design §3.8 MF-2).

DETERMINISTIC, runs in the normal suite (this is a query-plan structural check,
NOT a timing/live-backend cert — those are execution-deferred). It discharges the
§3.8 "Build-time validation (MF-2)" requirement: run ``EXPLAIN QUERY PLAN``
against

* **(a)** the §4.3 ``search`` query (the SQL ``store.forka.fts_candidates``
  issues), and
* **(b)** the ``menu`` active-set scan (the SQL ``store.forka.active_set``
  issues),

over the ClaudeMem-2k fixture, and assert the plans match the §3.8 expectation:

* the **search** plan shows the **FTS5 virtual table (``RecordFts``) driving**
  with a **rowid / INTEGER PRIMARY KEY join into ``Record``**, and does **NOT**
  use ``IX_Record_Active``;
* the **menu** plan shows **``IX_Record_Active``** serving the active scan
  (not a full ``Record`` table scan).

A deviation (search relying on the partial index, or menu full-scanning) is a
genuine indexing finding that must be revisited before SC-2 certification — the
assertions FAIL with the actual plan rather than being weakened to force green.

**Replicated SQL (assumption).** ``fts_candidates`` / ``active_set`` embed their
SQL inline (not as exported constants), so the two query strings here are
**replicated verbatim** from ``claudemem/store/forka.py``. They are pinned by
:func:`test_search_sql_matches_forka_source` / :func:`test_menu_sql_matches_forka_source`,
which assert the replicated text still appears in the live source — if either
query drifts in ``forka.py`` without updating this file, those guards fail loudly
rather than silently EXPLAIN-ing stale SQL.

**Corpus size (assumption).** The §3.8 expectation is structural (which table/
index drives), and the plan is **identical at count=50 and the full count=2000**
(verified — both captured into the build artifact). To keep the assertion honest
against the SC-2 cert reality we EXPLAIN at the full 2,000-row index with
``ANALYZE`` applied (so the planner sees the real ``sqlite_stat1`` stats); the
count=50 plan is also captured to document the equivalence. 2,000 records with
``write_files=False`` / ``seed_forkb=False`` materializes in ~1s, fast enough for
the normal suite.

**Build artifact.** Both captured plans (at count=50 and count=2000) are written
to ``docs/build-artifacts/query-plans.txt`` for morning review.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from claudemem import index
from claudemem.store import forka
from tests.fixtures.claudemem_2k import SEED, generate_corpus, materialize_corpus

# --------------------------------------------------------------------------- #
# Replicated SQL — verbatim from claudemem/store/forka.py (see module docstring).
# Pinned by the two _matches_forka_source guards below.                         #
# --------------------------------------------------------------------------- #

#: The §4.3 search SQL, exactly as ``forka.fts_candidates`` issues it.
_SEARCH_SQL = (
    "SELECT r.Id, bm25(RecordFts, 10.0, 5.0, 8.0, 1.0) AS raw "
    "FROM RecordFts JOIN Record r ON r.Id = RecordFts.rowid "
    "WHERE RecordFts MATCH ? "
    "AND r.SupersededBy IS NULL "
    "AND (r.Scope = 'global' OR (r.Scope = 'project' AND r.ProjectId = ?)) "
    "ORDER BY bm25(RecordFts, 10.0, 5.0, 8.0, 1.0) "
    "LIMIT ?;"
)

#: The ``menu`` active-set scan, exactly as ``forka.active_set`` issues it — but
#: projecting ``Id`` only (the plan is identical to the full column list; we keep
#: the plan-shaping WHERE / ORDER BY clauses verbatim). The full SELECT list lives
#: in ``forka.active_set``; here we EXPLAIN the index-selecting shape.
_MENU_SQL = (
    "SELECT Id FROM Record "
    "WHERE SupersededBy IS NULL "
    "AND (Scope = 'global' OR (Scope = 'project' AND ProjectId = ?)) "
    "ORDER BY Scope, ProjectId, Pinned, Importance;"
)

#: A representative scope + match expression for the EXPLAIN (a real project slug
#: from the fixture so the scope-merge branch is exercised; "postgres" is a real
#: topic token, REPRESENTATIVE_QUERIES[0]).
_PID = "-home-you-projects-my-app"
_MATCH_EXPR = forka._build_match_expr("postgres")

_ARTIFACT_PATH = (
    Path(__file__).resolve().parents[1] / "docs" / "build-artifacts" / "query-plans.txt"
)

_FULL_COUNT = 2000
_SMALL_COUNT = 50


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #


def _explain(conn, sql: str, params: tuple) -> list[tuple]:
    """Return the EXPLAIN QUERY PLAN rows (id, parent, notused, detail) for ``sql``."""
    return conn.execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()


def _plan_text(rows: list[tuple]) -> str:
    """The concatenated ``detail`` column of an EXPLAIN QUERY PLAN result, lowercased.

    Assertions match on key tokens in this text (robust to minor SQLite plan
    wording) rather than brittle exact-row compares.
    """
    return "\n".join(str(row[3]) for row in rows).lower()


def _build_index(tmp_path: Path, count: int):
    """Materialize a ``count``-record ClaudeMem-2k index and open + ANALYZE it.

    ``write_files=False`` / ``seed_forkb=False`` — only the ``forkA.db`` index is
    needed for the plan check, so we skip the markdown + Fork B writes (the slow
    part). ``ANALYZE`` populates ``sqlite_stat1`` so the planner sees the real
    row-count stats the SC-2 cert depends on. Returns the open connection.
    """
    home = tmp_path / f"home_{count}"
    records = generate_corpus(seed=SEED, count=count)
    materialize_corpus(records, home, write_files=False, seed_forkb=False)
    conn = index.open_forkA_at(home / "forkA.db")
    conn.execute("ANALYZE;")
    return conn


# --------------------------------------------------------------------------- #
# Source-pin guards: the replicated SQL must still live in forka.py.            #
# --------------------------------------------------------------------------- #


def _normalize(sql: str) -> str:
    """Collapse whitespace + drop adjacent-string-literal joins for the source pin.

    ``inspect.getsource`` returns the SQL as Python source, where a multi-line
    query is several adjacent string literals (``"... " "..."``). Removing the
    ``" "`` / ``" "`` boundaries and collapsing whitespace yields the same logical
    SQL text the interpreter concatenates, so the replicated query can be matched
    as a substring of the live source.
    """
    collapsed = " ".join(sql.split())
    dejoined = collapsed.replace('" "', " ").replace("' '", " ")
    return " ".join(dejoined.split())


def test_search_sql_matches_forka_source() -> None:
    """The replicated search SQL is still the SQL ``fts_candidates`` issues (§4.3)."""
    src = _normalize(inspect.getsource(forka.fts_candidates))
    assert _normalize(_SEARCH_SQL) in src, (
        "_SEARCH_SQL has drifted from forka.fts_candidates; update the replicated "
        "SQL in test_query_plans.py to match the live §4.3 query."
    )


def test_menu_sql_matches_forka_source() -> None:
    """The replicated menu WHERE/ORDER shape is still what ``active_set`` issues (§4.3)."""
    src = _normalize(inspect.getsource(forka.active_set))
    where_order = _normalize(
        "WHERE SupersededBy IS NULL "
        "AND (Scope = 'global' OR (Scope = 'project' AND ProjectId = ?)) "
        "ORDER BY Scope, ProjectId, Pinned, Importance;"
    )
    assert where_order in src, (
        "_MENU_SQL plan-shaping clauses have drifted from forka.active_set; update "
        "the replicated SQL in test_query_plans.py to match the live §4.3 scan."
    )


# --------------------------------------------------------------------------- #
# The §3.8 build-time validation (the real assertions).                         #
# --------------------------------------------------------------------------- #


def test_search_plan_is_fts5_driven_no_partial_index(tmp_path: Path) -> None:
    """(a) Search: RecordFts drives + rowid join into Record, NOT IX_Record_Active.

    §3.8 MF-2: in ``search`` the FTS5 ``MATCH`` drives candidate selection and
    ``Record`` is joined by its rowid PK, so the partial active index must NOT be
    consulted. A plan that relies on ``IX_Record_Active`` here is a deviation to
    revisit before SC-2.
    """
    conn = _build_index(tmp_path, _FULL_COUNT)
    try:
        rows = _explain(conn, _SEARCH_SQL, (_MATCH_EXPR, _PID, 64))
    finally:
        conn.close()
    text = _plan_text(rows)

    # FTS5 virtual table must drive.
    assert "recordfts" in text and "virtual table" in text, (
        "search plan does not show RecordFts driving as a VIRTUAL TABLE — "
        f"DEVIATION from §3.8. Actual plan:\n{_render(rows)}"
    )
    # Record must be reached by its rowid / INTEGER PRIMARY KEY, not a partial scan.
    assert ("primary key" in text and "rowid" in text), (
        "search plan does not show a rowid / INTEGER PRIMARY KEY join into Record — "
        f"DEVIATION from §3.8. Actual plan:\n{_render(rows)}"
    )
    # The partial active index must NOT appear on the search path.
    assert "ix_record_active" not in text, (
        "search plan unexpectedly relies on IX_Record_Active — DEVIATION from "
        f"§3.8 MF-2 (search must be FTS5-driven). Actual plan:\n{_render(rows)}"
    )


def test_menu_plan_uses_partial_active_index(tmp_path: Path) -> None:
    """(b) Menu: IX_Record_Active serves the active scan, NOT a full table scan.

    §3.8 MF-2: ``menu`` does no MATCH; it scans the active set and must ride the
    partial ``IX_Record_Active``. A full ``Record`` table scan here is a deviation
    to revisit before SC-2.
    """
    conn = _build_index(tmp_path, _FULL_COUNT)
    try:
        rows = _explain(conn, _MENU_SQL, (_PID,))
    finally:
        conn.close()
    text = _plan_text(rows)

    assert "ix_record_active" in text, (
        "menu plan does not use IX_Record_Active — DEVIATION from §3.8 "
        f"(the partial active index must serve the menu scan). Actual plan:\n{_render(rows)}"
    )
    # Guard against a full table scan slipping through (a SCAN of Record with no
    # index named is the failure mode the spec warns about).
    assert "using index" in text or "using covering index" in text, (
        "menu plan appears to be a full Record table scan (no index used) — "
        f"DEVIATION from §3.8. Actual plan:\n{_render(rows)}"
    )


def test_plan_identical_at_small_and_full_corpus(tmp_path: Path) -> None:
    """The §3.8 plans are size-independent: identical at count=50 and count=2000.

    Documents (and enforces) the corpus-size assumption in the module docstring —
    the structural plan does not change with row count / stats, so the small-corpus
    plan is a faithful stand-in and the full-corpus EXPLAIN is honest about the
    cert reality.
    """
    conn_small = _build_index(tmp_path, _SMALL_COUNT)
    try:
        search_small = _plan_text(_explain(conn_small, _SEARCH_SQL, (_MATCH_EXPR, _PID, 64)))
        menu_small = _plan_text(_explain(conn_small, _MENU_SQL, (_PID,)))
    finally:
        conn_small.close()

    conn_full = _build_index(tmp_path, _FULL_COUNT)
    try:
        search_full = _plan_text(_explain(conn_full, _SEARCH_SQL, (_MATCH_EXPR, _PID, 64)))
        menu_full = _plan_text(_explain(conn_full, _MENU_SQL, (_PID,)))
    finally:
        conn_full.close()

    assert search_small == search_full, (
        "search plan changed between count=50 and count=2000 — the corpus-size "
        f"assumption is invalid.\nsmall:\n{search_small}\nfull:\n{search_full}"
    )
    assert menu_small == menu_full, (
        "menu plan changed between count=50 and count=2000 — the corpus-size "
        f"assumption is invalid.\nsmall:\n{menu_small}\nfull:\n{menu_full}"
    )


# --------------------------------------------------------------------------- #
# Build artifact: capture both plans for morning review.                        #
# --------------------------------------------------------------------------- #


def _render(rows: list[tuple]) -> str:
    """Render EXPLAIN QUERY PLAN rows the way the sqlite3 CLI does (indented tree)."""
    return "\n".join(f"  {row[0]:>3}|{row[1]:>3}|{row[2]:>3}| {row[3]}" for row in rows)


def test_write_query_plan_artifact(tmp_path: Path) -> None:
    """Write the captured search + menu plans (count=50 and 2000) to the build artifact.

    Not an assertion of plan shape (the two tests above own that) — this produces
    the §3.8 "record both plans in the build artifacts" deliverable so the user can
    eyeball the actual EXPLAIN output in the morning.
    """
    sections: list[str] = [
        "ClaudeMem T6.4 — EXPLAIN QUERY PLAN build-time validation (tech-design §3.8 MF-2)",
        "",
        "Generated by tests/test_query_plans.py (deterministic; SEED="
        f"{SEED}). ANALYZE applied before each EXPLAIN so the planner sees real",
        "sqlite_stat1 stats. (a) = §4.3 search query; (b) = menu active-set scan.",
        "",
        "Expectation (§3.8): (a) RecordFts VIRTUAL TABLE drives + rowid/INTEGER",
        "PRIMARY KEY join into Record, NO IX_Record_Active; (b) IX_Record_Active",
        "serves the active scan (no full Record table scan).",
        "",
    ]
    for count in (_SMALL_COUNT, _FULL_COUNT):
        conn = _build_index(tmp_path, count)
        try:
            search_rows = _explain(conn, _SEARCH_SQL, (_MATCH_EXPR, _PID, 64))
            menu_rows = _explain(conn, _MENU_SQL, (_PID,))
        finally:
            conn.close()
        sections.extend(
            [
                "=" * 76,
                f"corpus count = {count}",
                "=" * 76,
                "",
                "(a) SEARCH query (§4.3) — EXPLAIN QUERY PLAN:",
                f"    SQL: {_SEARCH_SQL}",
                "",
                _render(search_rows),
                "",
                "(b) MENU active-set scan — EXPLAIN QUERY PLAN:",
                f"    SQL: {_MENU_SQL}",
                "",
                _render(menu_rows),
                "",
            ]
        )

    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ARTIFACT_PATH.write_text("\n".join(sections) + "\n", encoding="utf-8")
    assert _ARTIFACT_PATH.is_file()


if __name__ == "__main__":  # pragma: no cover - convenience for manual capture
    raise SystemExit(pytest.main([__file__, "-q"]))
