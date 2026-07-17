"""Tests for claudemem.enrich.routine (T4.7 + T4.8) — the two enrichment routines.

Covers the public surface ``cli`` / ``hooks`` call, against REAL store / files /
index on a tmp ``CLAUDEMEM_HOME`` with a **fake backend injected** so no test
makes a real model call (the only model touch is the injected stub):

* ``enrich_batch`` ``new`` verdict — model-free dedup assembly calls
  ``fts_candidates`` (k=5), ONE backend call, record persisted with
  summary+aliases, spend recorded, ``EnrichPending`` cleared atomically.
* ``duplicate`` verdict — fields still applied, no supersede.
* ``conflict`` verdict — target superseded (``SupersededBy`` set in index + file),
  conflict surfaced NON-BLOCKING (no exception, no ``input()``), keep-both-and-flag.
* degraded (``LexicalOnlyBackend`` / defer-all fake) — record persists
  ``EnrichPending=1``, no error, the save still happened (``SC-10`` / ``SC-3``).
* dedup assembly is model-free — ``fts_candidates`` used; the fake backend is the
  only model touch.
* ``reflect`` — valid passive hit → ``HitCount`` bumped + flushed to frontmatter +
  stale cleared (``SC-9`` / ``SC-13``); out-of-set id ignored; promotion
  candidates returned, NOT applied (``SC-8``); empty / no-key → no-op no error.

``CLAUDEMEM_HOME`` is pointed at ``tmp_path`` so no test touches the real
``~/.claude``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest

from claudemem import config, files, index
from claudemem.enrich import routine
from claudemem.enrich.backend import (
    BackendOutcome,
    DeferralEntry,
    EnrichRequest,
    EnrichResult,
    LexicalOnlyBackend,
    PassiveHit,
    PromotionCandidate,
    ReflectOutcome,
    ReflectRequest,
    SpendEntry,
)
from claudemem.store import forka, forkb

NOW_EPOCH = files.iso_to_epoch("2026-05-30T12:00:00Z")


# --------------------------------------------------------------------------- #
# Fixtures + fake backend                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME at a tmp dir; return it."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def conn_a(home: Path) -> Iterator[sqlite3.Connection]:
    connection = index.open_forkA()
    yield connection
    connection.close()


@pytest.fixture
def conn_b(home: Path) -> Iterator[sqlite3.Connection]:
    connection = index.open_forkB()
    yield connection
    connection.close()


@pytest.fixture
def settings() -> config.Settings:
    return config.load_config()


@pytest.fixture
def scope(tmp_path: Path) -> config.ScopeContext:
    """Project scope whose memory dirs live under tmp_path (so file I/O is sandboxed)."""
    proj = tmp_path / "proj_mem"
    glob = tmp_path / "global_mem"
    proj.mkdir(parents=True, exist_ok=True)
    glob.mkdir(parents=True, exist_ok=True)
    return config.ScopeContext(
        kind="project", project_id="proj-1", global_dir=glob, project_dir=proj
    )


class FakeBackend:
    """A scripted EnrichmentBackend — records calls, returns canned outcomes.

    This is the injected seam: ``enrich_batch`` / ``reflect`` accept a ``backend=``
    so the routine never resolves a real transport. The fake is the *only* model
    touch in the suite (no ``anthropic`` import, no ``claude`` spawn).
    """

    def __init__(
        self,
        *,
        enrich_outcome: BackendOutcome | None = None,
        reflect_outcome: ReflectOutcome | None = None,
    ) -> None:
        self.enrich_outcome = enrich_outcome
        self.reflect_outcome = reflect_outcome
        self.enrich_reqs: list[list[EnrichRequest]] = []
        self.reflect_reqs: list[ReflectRequest] = []

    @staticmethod
    def detect() -> bool:
        return True

    def enrich_batch(self, reqs: list[EnrichRequest]) -> BackendOutcome:
        self.enrich_reqs.append(reqs)
        assert self.enrich_outcome is not None
        return self.enrich_outcome

    def reflect(self, req: ReflectRequest) -> ReflectOutcome:
        self.reflect_reqs.append(req)
        assert self.reflect_outcome is not None
        return self.reflect_outcome


def _record_file(
    scope: config.ScopeContext,
    name: str,
    *,
    body: str = "the body text about pgvector and embeddings",
    summary: str | None = None,
    aliases: list[str] | None = None,
    superseded_by: str | None = None,
    hit_count: int = 0,
    stale: bool = False,
) -> files.RecordFile:
    assert scope.project_dir is not None
    return files.RecordFile(
        path=scope.project_dir / f"{name}.md",
        name=name,
        type="reference",
        scope="project",
        importance=3,
        pinned=False,
        source="explicit",
        created="2026-05-29T12:00:00Z",
        last_accessed="2026-05-30T08:30:00Z",
        access_count=0,
        hit_count=hit_count,
        summary=summary,
        aliases=aliases if aliases is not None else [],
        superseded_by=superseded_by,
        stale=stale,
        body=body,
    )


def _seed_existing(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, name: str, **kw: object
) -> files.RecordFile:
    """Write a record to disk + index so it becomes a dedup candidate."""
    rf = _record_file(scope, name, **kw)  # type: ignore[arg-type]
    files.write_record(rf)
    forka.upsert_record(conn_a, rf, scope)
    return rf


def _ok_spend(record_id_int: int | None = None) -> SpendEntry:
    return SpendEntry(
        call_site="save",
        model="haiku",
        backend="sdk",
        input_tokens=100,
        output_tokens=20,
        record_id_int=record_id_int,
    )


def _enrich_pending(conn_a: sqlite3.Connection, scope: config.ScopeContext, name: str) -> int:
    rec = forka.select_record(conn_a, scope, name)
    assert rec is not None
    return rec.enrich_pending


# --------------------------------------------------------------------------- #
# enrich_batch — new verdict (T4.7)                                             #
# --------------------------------------------------------------------------- #


def test_new_verdict_persists_summary_aliases_and_clears_pending(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, settings: config.Settings
) -> None:
    rf = _record_file(scope, "vectors")
    backend = FakeBackend(
        enrich_outcome=BackendOutcome(
            results=[
                EnrichResult(
                    record_id="vectors",
                    summary="how ClaudeMem may add vectors",
                    aliases=["pgvector", "embeddings"],
                    dedup_verdict="new",
                    dedup_target_name=None,
                    conflict_explanation=None,
                )
            ],
            deferred=[],
            spend=[_ok_spend()],
        )
    )

    result = routine.enrich_batch(conn_a, [rf], scope, settings, backend=backend)

    assert result.enriched == 1
    assert result.deferred == []
    assert result.conflicts == []
    assert result.spend_rows == 1

    # Exactly ONE backend call (IN-13).
    assert len(backend.enrich_reqs) == 1

    # Persisted to the index with summary + aliases, EnrichPending cleared.
    rec = forka.select_record(conn_a, scope, "vectors")
    assert rec is not None
    assert rec.summary == "how ClaudeMem may add vectors"
    assert files.aliases_from_json(rec.aliases_json) == ["pgvector", "embeddings"]
    assert rec.enrich_pending == 0

    # Persisted to the file (Fork A = truth).
    assert rf.path.is_file()
    on_disk = files.read_record(rf.path)
    assert on_disk.summary == "how ClaudeMem may add vectors"

    # Spend row recorded.
    spend_count = conn_a.execute("SELECT COUNT(*) FROM SpendLog;").fetchone()[0]
    assert spend_count == 1


def test_dedup_assembly_is_model_free_and_uses_fts_k5(
    conn_a: sqlite3.Connection,
    scope: config.ScopeContext,
    settings: config.Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Seed an existing active record so there is a real candidate to assemble.
    _seed_existing(
        conn_a, scope, "existing_vec", summary="existing vector note",
        aliases=["vec"], body="pgvector embeddings note",
    )

    calls: list[int] = []
    real_fts = forka.fts_candidates

    def spy_fts(
        conn: sqlite3.Connection,
        query: str,
        sc: config.ScopeContext,
        k: int = 64,
    ) -> list[tuple[int, float]]:
        calls.append(k)
        return real_fts(conn, query, sc, k=k)

    # routine imported the forka MODULE (``from claudemem.store import forka``),
    # so patching the module attribute here is what the routine resolves at call.
    monkeypatch.setattr(forka, "fts_candidates", spy_fts)

    rf = _record_file(scope, "newvec", body="pgvector embeddings new note")
    backend = FakeBackend(
        enrich_outcome=BackendOutcome(
            results=[
                EnrichResult(
                    record_id="newvec",
                    summary="s",
                    aliases=["a"],
                    dedup_verdict="new",
                    dedup_target_name=None,
                    conflict_explanation=None,
                )
            ],
            spend=[_ok_spend()],
        )
    )

    routine.enrich_batch(conn_a, [rf], scope, settings, backend=backend)

    # Candidate assembly used fts_candidates with k = dedup_k = 5 (model-free).
    assert calls == [config.LLM_DEDUP_K_DEFAULT]
    # The candidate was passed to the backend (the only model touch).
    req = backend.enrich_reqs[0][0]
    assert any(c.name == "existing_vec" for c in req.candidates)
    # Excerpt is head-only, capped at dedup_excerpt_chars.
    cand = next(c for c in req.candidates if c.name == "existing_vec")
    assert len(cand.excerpt) <= settings.llm.dedup_excerpt_chars


# --------------------------------------------------------------------------- #
# enrich_batch — duplicate verdict                                              #
# --------------------------------------------------------------------------- #


def test_duplicate_verdict_applies_fields_and_leaves_target_active(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, settings: config.Settings
) -> None:
    _seed_existing(conn_a, scope, "target_dup", summary="prior", body="duplicate body content")

    rf = _record_file(scope, "dupe", body="duplicate body content")
    backend = FakeBackend(
        enrich_outcome=BackendOutcome(
            results=[
                EnrichResult(
                    record_id="dupe",
                    summary="merged summary",
                    aliases=["dup_alias"],
                    dedup_verdict="duplicate",
                    dedup_target_name="target_dup",
                    conflict_explanation=None,
                )
            ],
            spend=[_ok_spend()],
        )
    )

    result = routine.enrich_batch(conn_a, [rf], scope, settings, backend=backend)

    assert result.enriched == 1
    assert result.conflicts == []

    # Summary/aliases still applied (IN-13).
    rec = forka.select_record(conn_a, scope, "dupe")
    assert rec is not None
    assert rec.summary == "merged summary"

    # The dedup target stays ACTIVE (a duplicate is not a supersede).
    target = forka.select_record(conn_a, scope, "target_dup")
    assert target is not None
    assert target.superseded_by is None


# --------------------------------------------------------------------------- #
# enrich_batch — conflict verdict (non-blocking supersede)                      #
# --------------------------------------------------------------------------- #


def test_conflict_verdict_supersedes_target_non_blocking(
    conn_a: sqlite3.Connection,
    scope: config.ScopeContext,
    settings: config.Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _seed_existing(
        conn_a, scope, "old_fact", summary="old", body="the old contradicting fact"
    )

    rf = _record_file(scope, "new_fact", body="the new contradicting fact")
    backend = FakeBackend(
        enrich_outcome=BackendOutcome(
            results=[
                EnrichResult(
                    record_id="new_fact",
                    summary="the corrected fact",
                    aliases=["fact"],
                    dedup_verdict="conflict",
                    dedup_target_name="old_fact",
                    conflict_explanation="contradicts the old value",
                )
            ],
            spend=[_ok_spend()],
        )
    )

    # Must NOT block on input() — a patched input raises if called.
    def _no_input(*_a: object, **_k: object) -> str:
        raise AssertionError("enrich_batch must not block on input()")

    monkeypatch.setattr("builtins.input", _no_input)
    result = routine.enrich_batch(conn_a, [rf], scope, settings, backend=backend)

    # Conflict surfaced, non-blocking (returned for cli to print).
    assert len(result.conflicts) == 1
    rep = result.conflicts[0]
    assert rep.record_name == "new_fact"
    assert rep.target_name == "old_fact"
    assert rep.explanation == "contradicts the old value"

    # Target superseded in the INDEX (SupersededBy = new record's name) — SC-7.
    old = forka.select_record(conn_a, scope, "old_fact")
    assert old is not None
    assert old.superseded_by == "new_fact"

    # Target superseded in the FILE too (keep-both-and-flag, unattended path).
    assert scope.project_dir is not None
    old_on_disk = files.read_record(scope.project_dir / "old_fact.md")
    assert old_on_disk.superseded_by == "new_fact"

    # New record persisted active.
    new = forka.select_record(conn_a, scope, "new_fact")
    assert new is not None
    assert new.superseded_by is None


def test_conflict_target_out_of_candidate_set_is_not_superseded(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, settings: config.Settings
) -> None:
    # The named target was never offered as a candidate → defensive: no supersede.
    _seed_existing(conn_a, scope, "unrelated", body="totally unrelated content")

    rf = _record_file(scope, "lonely", body="a unique body with no overlap zzqqxx")
    backend = FakeBackend(
        enrich_outcome=BackendOutcome(
            results=[
                EnrichResult(
                    record_id="lonely",
                    summary="s",
                    aliases=["a"],
                    dedup_verdict="conflict",
                    dedup_target_name="unrelated",  # not in candidate set
                    conflict_explanation="x",
                )
            ],
            spend=[_ok_spend()],
        )
    )

    result = routine.enrich_batch(conn_a, [rf], scope, settings, backend=backend)

    assert result.conflicts == []  # not surfaced — target wasn't a candidate
    unrelated = forka.select_record(conn_a, scope, "unrelated")
    assert unrelated is not None
    assert unrelated.superseded_by is None


# --------------------------------------------------------------------------- #
# enrich_batch — degraded / deferred (SC-3 / SC-10)                             #
# --------------------------------------------------------------------------- #


def test_lexical_only_backend_defers_with_enrich_pending(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, settings: config.Settings
) -> None:
    rf = _record_file(scope, "degraded")
    result = routine.enrich_batch(
        conn_a, [rf], scope, settings, backend=LexicalOnlyBackend()
    )

    assert result.enriched == 0
    assert len(result.deferred) == 1
    assert result.deferred[0].record_name == "degraded"
    assert result.deferred[0].reason == "auth"
    assert result.spend_rows == 0  # no model call attempted

    # The save STILL happened (file + index), EnrichPending=1 (SC-10 / SC-3).
    assert rf.path.is_file()
    assert _enrich_pending(conn_a, scope, "degraded") == 1


def test_defer_all_fake_backend_persists_without_error(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, settings: config.Settings
) -> None:
    rf = _record_file(scope, "deferred_rec")
    backend = FakeBackend(
        enrich_outcome=BackendOutcome(
            results=[],
            deferred=[DeferralEntry(record_id="deferred_rec", reason="transient")],
            spend=[],
        )
    )

    result = routine.enrich_batch(conn_a, [rf], scope, settings, backend=backend)

    assert result.deferred[0].reason == "transient"
    assert _enrich_pending(conn_a, scope, "deferred_rec") == 1


def test_empty_records_is_noop(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, settings: config.Settings
) -> None:
    backend = FakeBackend(enrich_outcome=BackendOutcome())
    result = routine.enrich_batch(conn_a, [], scope, settings, backend=backend)
    assert result.enriched == 0
    assert backend.enrich_reqs == []  # backend not even called


# --------------------------------------------------------------------------- #
# reflect — passive hits + promotion candidates (T4.8)                          #
# --------------------------------------------------------------------------- #


def _seed_activity(conn_b: sqlite3.Connection) -> None:
    forkb.append_activity(
        conn_b, session_id="sess-1", ts=NOW_EPOCH, role="user", kind="prompt",
        body="working on the vectors note",
    )


def test_reflect_reinforces_valid_hit_and_proposes_promotions(
    conn_a: sqlite3.Connection,
    conn_b: sqlite3.Connection,
    scope: config.ScopeContext,
    settings: config.Settings,
) -> None:
    _seed_existing(conn_a, scope, "hit_rec", summary="s", hit_count=2, stale=True)
    _seed_activity(conn_b)

    backend = FakeBackend(
        reflect_outcome=ReflectOutcome(
            passive_hits=[
                PassiveHit(record_id="a:hit_rec", evidence="used in the session"),
                PassiveHit(record_id="a:ghost", evidence="not a real record"),
            ],
            promotion_candidates=[
                PromotionCandidate(
                    archive_id="b:1", proposed_summary="promote me", rationale="recurs"
                )
            ],
            spend=[SpendEntry(call_site="reflect", model="haiku", backend="sdk",
                              input_tokens=50, output_tokens=10)],
        )
    )

    result = routine.reflect(
        conn_b, "sess-1", conn_a, scope, settings, backend=backend
    )

    # Exactly one valid hit reinforced (the ghost id is out-of-set, ignored).
    assert result.reinforced == 1

    # HitCount bumped in the INDEX.
    rec = forka.select_record(conn_a, scope, "hit_rec")
    assert rec is not None
    assert rec.hit_count == 3

    # Flushed to frontmatter + stale cleared (SC-9 / SC-13).
    assert scope.project_dir is not None
    on_disk = files.read_record(scope.project_dir / "hit_rec.md")
    assert on_disk.hit_count == 3
    assert on_disk.stale is False

    # Promotion candidate is PROPOSED, never auto-applied (SC-8): no a:promote_me record.
    assert len(result.proposed_promotions) == 1
    assert result.proposed_promotions[0].archive_id == "b:1"
    # No Fork A record was created from the promotion candidate.
    assert forka.select_record(conn_a, scope, "promote me") is None

    # Reflection spend recorded (call_site='reflect', no Record row cleared).
    assert result.spend_rows == 1
    spend = conn_a.execute(
        "SELECT CallSite FROM SpendLog WHERE CallSite = 'reflect';"
    ).fetchone()
    assert spend is not None


def test_reflect_lexical_only_is_noop_no_error(
    conn_a: sqlite3.Connection,
    conn_b: sqlite3.Connection,
    scope: config.ScopeContext,
    settings: config.Settings,
) -> None:
    _seed_existing(conn_a, scope, "untouched", hit_count=5)
    _seed_activity(conn_b)

    result = routine.reflect(
        conn_b, "sess-1", conn_a, scope, settings, backend=LexicalOnlyBackend()
    )

    assert result.reinforced == 0
    assert result.proposed_promotions == []
    assert result.spend_rows == 0

    # Nothing changed — the record is untouched (reindex is the backstop, SC-9).
    rec = forka.select_record(conn_a, scope, "untouched")
    assert rec is not None
    assert rec.hit_count == 5
