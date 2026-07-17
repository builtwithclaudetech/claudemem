"""Tests for the ClaudeMem-2k seeded fixture generator (T6.1; tech-design §4.6).

These verify the *generator* contract (determinism, scale arithmetic,
materialization, the 5 representative queries) WITHOUT running the 50×5 latency
certification (that is T6.2, execution-deferred to morning-verify). Everything
here runs against a small count so the suite stays fast.
"""

from __future__ import annotations

from pathlib import Path

from claudemem import index
from claudemem.config import ScopeContext
from claudemem.store import forka
from tests.fixtures.claudemem_2k import (
    DEFAULT_COUNT,
    DEFAULT_TARGET_BODY_BYTES,
    NO_MATCH_QUERY_INDEX,
    NO_MATCH_TOKEN,
    REPRESENTATIVE_QUERIES,
    SEED,
    GeneratedRecord,
    generate_corpus,
    materialize_corpus,
)

_SMALL = 50


def _total_body_bytes(records: list[GeneratedRecord]) -> int:
    return sum(len(r.body.encode("utf-8")) for r in records)


# --------------------------------------------------------------------------- #
# Determinism (C-13: same seed → byte-identical corpus).                        #
# --------------------------------------------------------------------------- #


def test_generate_corpus_is_deterministic() -> None:
    a = generate_corpus(seed=SEED, count=_SMALL)
    b = generate_corpus(seed=SEED, count=_SMALL)
    assert a == b  # frozen dataclasses → structural equality over every field


def test_different_seed_changes_corpus() -> None:
    a = generate_corpus(seed=SEED, count=_SMALL)
    b = generate_corpus(seed=SEED + 1, count=_SMALL)
    assert a != b


def test_corpus_count_matches_request() -> None:
    records = generate_corpus(seed=SEED, count=_SMALL)
    assert len(records) == _SMALL
    # Names are collision-free (the upsert natural key would otherwise clash).
    assert len({(r.scope, r.project_id, r.name) for r in records}) == _SMALL


# --------------------------------------------------------------------------- #
# Scale sanity — the 2,000/6 MB target is reachable by arithmetic (no slow run). #
# --------------------------------------------------------------------------- #


def test_small_corpus_total_tracks_target_fraction() -> None:
    """At count=50 the total body ≈ target × (50/2000), within a few percent.

    This proves the per-record mean is ``target/count`` for ANY count, so the
    full 2,000-record corpus reaches ~6 MB — without generating 2,000 records.
    """
    records = generate_corpus(
        seed=SEED, count=_SMALL, target_body_bytes=DEFAULT_TARGET_BODY_BYTES
    )
    total = _total_body_bytes(records)
    expected = DEFAULT_TARGET_BODY_BYTES * _SMALL / DEFAULT_COUNT  # 150 KB
    # Sentence-granularity rounding pushes each body up to the next whole
    # sentence; allow a generous band but assert it tracks the fraction.
    assert 0.85 * expected <= total <= 1.20 * expected


def test_target_scales_to_six_mb_at_2000() -> None:
    """The configured target IS ~6 MB at the pinned 2,000 count (the §4.6 spec)."""
    assert DEFAULT_COUNT == 2000
    assert 5_500_000 <= DEFAULT_TARGET_BODY_BYTES <= 6_500_000


def test_metadata_is_varied() -> None:
    records = generate_corpus(seed=SEED, count=_SMALL)
    assert {r.importance for r in records} & {1, 2, 3, 4, 5}
    assert any(r.pinned for r in records)
    assert any(not r.pinned for r in records)
    assert {r.type for r in records} == {"user", "feedback", "project", "reference"}
    # Scope merge is exercised: at least one global AND one project record.
    assert any(r.scope == "global" for r in records)
    assert any(r.scope == "project" for r in records)
    # Recency spread: created timestamps are not all identical.
    assert len({r.created for r in records}) > 1
    # At least one multi-word alias exists (the task requires it).
    assert any(" " in alias for r in records for alias in r.aliases)


# --------------------------------------------------------------------------- #
# The 5 representative queries (§4.6 / §10.3).                                   #
# --------------------------------------------------------------------------- #


def test_five_representative_queries_defined() -> None:
    assert len(REPRESENTATIVE_QUERIES) == 5
    assert REPRESENTATIVE_QUERIES[NO_MATCH_QUERY_INDEX] == NO_MATCH_TOKEN
    # A mix: at least one single-token and one multi-token matching query.
    assert any(" " not in q for q in REPRESENTATIVE_QUERIES[:NO_MATCH_QUERY_INDEX])
    assert any(" " in q for q in REPRESENTATIVE_QUERIES[:NO_MATCH_QUERY_INDEX])


def test_no_match_token_absent_from_corpus() -> None:
    """The no-match token appears in NO name/summary/alias/body (genuine miss)."""
    records = generate_corpus(seed=SEED, count=_SMALL)
    for r in records:
        haystack = " ".join([r.name, r.summary, *r.aliases, r.body]).lower()
        assert NO_MATCH_TOKEN not in haystack


# --------------------------------------------------------------------------- #
# Materialization → a searchable corpus under a tmp CLAUDEMEM_HOME.              #
# --------------------------------------------------------------------------- #


def test_materialize_produces_searchable_corpus(tmp_path: Path) -> None:
    home = tmp_path / "claudemem_home"
    records = generate_corpus(seed=SEED, count=_SMALL)
    materialize_corpus(records, home, write_files=True, seed_forkb=True)

    # forkA.db exists and files were written.
    assert (home / "forkA.db").is_file()
    md_files = list(home.rglob("*.md"))
    assert len(md_files) == _SMALL

    # A known topic token returns Fork A hits via the model-free candidate query.
    # The §4.3 candidate query is scope-merged (global + the active project), so
    # query the union of every materialized scope — exactly what the harness does
    # when it picks a representative scope.
    project_slugs = {r.project_id for r in records if r.project_id is not None}
    scopes = [
        ScopeContext(
            kind="global",
            project_id=None,
            global_dir=home / "memory",
            project_dir=None,
        ),
        *[
            ScopeContext(
                kind="project",
                project_id=slug,
                global_dir=home / "memory",
                project_dir=home / "projects" / slug / "memory",
            )
            for slug in sorted(project_slugs)
        ],
    ]

    conn = index.open_forkA_at(home / "forkA.db")
    try:
        # A single-token matching query (postgres) returns candidates in at least
        # one scope (the topic exists in the corpus).
        matching = REPRESENTATIVE_QUERIES[0]
        any_hits = any(forka.fts_candidates(conn, matching, ctx) for ctx in scopes)
        assert any_hits, f"expected Fork A hits for {matching!r} in some scope"

        # The no-match token returns NOTHING from Fork A in ANY scope (this is
        # what drives the Fork B archive fallback).
        for ctx in scopes:
            assert forka.fts_candidates(conn, NO_MATCH_TOKEN, ctx) == []
    finally:
        conn.close()


def test_materialize_seeds_forkb_for_fallback(tmp_path: Path) -> None:
    """seed_forkb=True populates Fork B so the no-match query has a fallback."""
    home = tmp_path / "home"
    records = generate_corpus(seed=SEED, count=_SMALL)
    materialize_corpus(records, home, write_files=False, seed_forkb=True)

    assert (home / "forkB.db").is_file()
    conn_b = index.open_forkB()  # CLAUDEMEM_HOME unset → default; reopen explicitly
    conn_b.close()
    # Reopen at the materialized home to count rows.
    import sqlite3

    conn = sqlite3.connect(home / "forkB.db")
    try:
        count = conn.execute("SELECT COUNT(*) FROM Activity;").fetchone()[0]
        assert count > 0
    finally:
        conn.close()


def test_materialize_is_deterministic_across_homes(tmp_path: Path) -> None:
    """Same seed → identical on-disk markdown bytes in two separate homes."""
    records = generate_corpus(seed=SEED, count=_SMALL)
    home_a = tmp_path / "a"
    home_b = tmp_path / "b"
    materialize_corpus(records, home_a, write_files=True, seed_forkb=False)
    materialize_corpus(records, home_b, write_files=True, seed_forkb=False)

    files_a = sorted(p.relative_to(home_a) for p in home_a.rglob("*.md"))
    files_b = sorted(p.relative_to(home_b) for p in home_b.rglob("*.md"))
    assert files_a == files_b
    for rel in files_a:
        assert (home_a / rel).read_bytes() == (home_b / rel).read_bytes()
