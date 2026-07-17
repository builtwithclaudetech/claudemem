"""GATE — tech-design §10.6 degradation gate → certifies ``SC-3`` / ``NG-5``.

This is the **deterministic degradation GATE** at the **enrich layer** (Phase 4).
It reproduces the §10.6 conditions EXACTLY and asserts the load-bearing
graceful-degradation contract:

* **(a)** no ``ANTHROPIC_API_KEY`` in the environment, and
* **(b)** the ``anthropic`` package not importable — simulated WITHOUT
  uninstalling by setting ``sys.modules["anthropic"] = None`` so the
  function-local ``import anthropic`` in ``backend_sdk`` raises ``ImportError``
  on cue (the exact mechanism §10.6 prescribes).

Under either condition: ``select_backend`` resolves :class:`LexicalOnlyBackend`,
``enrich_batch`` over real RecordFiles against a real tmp store persists every
record lexical-only with ``EnrichPending=1`` (Fork A file + index), records NO
spend (no model call was attempted), and never raises; ``reflect`` is a clean
no-op. **Both conditions degrade identically** (``NG-5``).

**Phase scope (Phase 4 vs Phase 5).** The full ``SC-3`` command list
(``search``/``get``/``save``/``used``/``pin``/``unpin``/``forget``/``promote``/
``reindex``/``menu``/``log``/``import``) includes CLI commands that do not exist
until Phase 5. At THIS phase the gate certifies §10.6 at the **enrich layer** by
driving the two enrichment routines (``enrich_batch`` — the save/import/backfill
routine — and ``reflect``) directly. The CLI-command-level *exit-0* assertions
for that command list are re-covered at **T5.5 / T5.6** (Phase 5) once the CLI
exists — see the carry-forward note in the task report. This gate does NOT fake
a CLI that does not yet exist.

**Robust simulation + clean teardown.** Two context managers isolate the
simulated degradation so it never leaks to other tests:

* :func:`_no_api_key` removes ``ANTHROPIC_API_KEY`` and restores it after.
* :func:`_anthropic_absent` installs ``sys.modules["anthropic"] = None`` and
  restores the prior entry (present or absent) on exit, even on exception.

Both also reset the per-process backend-selection cache + warn-once set
(``backend._reset_cache_for_tests``) and the CLI detect cache on entry and exit,
so a forced-unavailable resolution from one case never serves a stale cached
backend to the next (NG-1 cache is process-lifetime by design).

**CLI parse/repair (§5.6) + SDK idempotency (§5.8) consolidation.** These are
exhaustively unit-tested in ``test_enrich_backend_cli.py`` /
``test_enrich_backend_sdk.py``; this gate does NOT duplicate them. It adds two
focused reference assertions confirming the §10.6 degradation gate sits on top
of the same backend boundary those suites cover (the deferral reason taxonomy
and the lexical floor), keeping this file scoped to the degradation contract.
"""

from __future__ import annotations

import os
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from claudemem import config, files, index
from claudemem.enrich import backend as be
from claudemem.enrich import routine
from claudemem.enrich.backend import LexicalOnlyBackend
from claudemem.enrich.backend_cli import ClaudeCliBackend
from claudemem.store import forka, forkb

_API_KEY_ENV = "ANTHROPIC_API_KEY"

NOW_EPOCH = files.iso_to_epoch("2026-05-30T12:00:00Z")


# --------------------------------------------------------------------------- #
# Fixtures — a REAL tmp store (no model anywhere) on a tmp CLAUDEMEM_HOME       #
# --------------------------------------------------------------------------- #


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME at a tmp dir so no test touches the real ~/.claude."""
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
def scope(tmp_path: Path) -> config.ScopeContext:
    """Project scope whose memory dirs live under tmp_path (file I/O sandboxed)."""
    proj = tmp_path / "proj_mem"
    glob = tmp_path / "global_mem"
    proj.mkdir(parents=True, exist_ok=True)
    glob.mkdir(parents=True, exist_ok=True)
    return config.ScopeContext(
        kind="project", project_id="proj-1", global_dir=glob, project_dir=proj
    )


def _record_file(scope: config.ScopeContext, name: str, *, body: str) -> files.RecordFile:
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
        hit_count=0,
        summary=None,
        aliases=[],
        superseded_by=None,
        stale=False,
        body=body,
    )


def _two_new_records(scope: config.ScopeContext) -> list[files.RecordFile]:
    return [
        _record_file(scope, "redis-port", body="Redis listens on 6380 on this VPS."),
        _record_file(scope, "vectors", body="ClaudeMem may add pgvector embeddings later."),
    ]


# --------------------------------------------------------------------------- #
# Degradation simulators (robust; clean teardown — restores prior state)        #
# --------------------------------------------------------------------------- #


def _reset_backend_caches() -> None:
    """Clear all per-process backend caches so a forced resolution never leaks.

    The selection cache + warn-once set (``backend._reset_cache_for_tests``) AND
    the ``ClaudeCliBackend`` detect cache. Both are process-lifetime by design
    (NG-1); the test hooks exist precisely so a simulated condition does not
    serve a stale cached backend to the next case or to an unrelated test.
    """
    be._reset_cache_for_tests()
    ClaudeCliBackend._reset_detect_cache_for_tests()


@contextmanager
def _no_api_key() -> Iterator[None]:
    """Run the body with ``ANTHROPIC_API_KEY`` removed; restore it after.

    Condition (a) of §10.6. Resets the backend caches on entry and exit so the
    no-key resolution is re-derived inside the block and discarded after.
    """
    saved = os.environ.pop(_API_KEY_ENV, None)
    _reset_backend_caches()
    try:
        yield
    finally:
        if saved is not None:
            os.environ[_API_KEY_ENV] = saved
        _reset_backend_caches()


@contextmanager
def _anthropic_absent() -> Iterator[None]:
    """Run the body with the ``anthropic`` package made un-importable.

    Condition (b) of §10.6, simulated WITHOUT uninstalling: setting
    ``sys.modules["anthropic"] = None`` makes the function-local
    ``import anthropic`` in ``backend_sdk`` raise ``ImportError`` on cue (CPython
    treats a ``None`` entry in ``sys.modules`` as a poisoned import). Restores the
    prior entry — whether the real module was present or genuinely absent — on
    exit, even on exception. Resets the backend caches on entry and exit.
    """
    sentinel = object()
    prior = sys.modules.get("anthropic", sentinel)
    sys.modules["anthropic"] = None  # type: ignore[assignment]  # poison the import.
    _reset_backend_caches()
    try:
        yield
    finally:
        if prior is sentinel:
            sys.modules.pop("anthropic", None)
        else:
            sys.modules["anthropic"] = prior  # type: ignore[assignment]
        _reset_backend_caches()


# --------------------------------------------------------------------------- #
# Helpers to assert the persisted degraded state                                #
# --------------------------------------------------------------------------- #


def _assert_persisted_lexical_only(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, records: list[files.RecordFile]
) -> None:
    """Assert every record persisted Fork A file + index with ``EnrichPending=1``.

    The SC-3 degraded-save contract: a save ALWAYS persists (file is truth, index
    is the rebuildable cache), with ``EnrichPending=1`` so the next ``reindex``
    backfills enrichment, and NO model-side enrichment (no summary, no aliases).
    """
    for rf in records:
        # File written (Fork A = source of truth).
        assert rf.path.is_file(), f"{rf.name} file not persisted"
        # Index row present, EnrichPending=1, no enrichment.
        rec = forka.select_record(conn_a, scope, rf.name)
        assert rec is not None, f"{rf.name} not in index"
        assert rec.enrich_pending == 1, f"{rf.name} EnrichPending != 1"
        assert rec.summary is None, f"{rf.name} got a summary while degraded"
        assert files.aliases_from_json(rec.aliases_json) == [], (
            f"{rf.name} got aliases while degraded"
        )


def _no_spend_rows(conn_a: sqlite3.Connection) -> int:
    return conn_a.execute("SELECT COUNT(*) FROM SpendLog;").fetchone()[0]


# --------------------------------------------------------------------------- #
# 1. Condition (a) — no API key, no authed CLI → lexical floor (SC-3)           #
# --------------------------------------------------------------------------- #


def test_condition_a_no_key_selects_lexical(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No key + no authed CLI + ``backend=auto`` → ``select_backend`` is lexical."""
    monkeypatch.setattr(ClaudeCliBackend, "detect", staticmethod(lambda: False))
    settings = config.load_config(overrides={"llm": {"backend": "auto"}})
    with _no_api_key():
        resolved = be.select_backend(settings)
    assert isinstance(resolved, LexicalOnlyBackend)


def test_condition_a_enrich_batch_persists_pending_no_spend_no_raise(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) ``enrich_batch`` over new records degrades cleanly: pending, no spend."""
    monkeypatch.setattr(ClaudeCliBackend, "detect", staticmethod(lambda: False))
    settings = config.load_config(overrides={"llm": {"backend": "auto"}})
    records = _two_new_records(scope)

    with _no_api_key():
        # Resolve through select_backend exactly as production would (no injection).
        result = routine.enrich_batch(conn_a, records, scope, settings)

    # Returned normally (no exception), every record deferred, nothing enriched.
    assert result.enriched == 0
    assert {d.record_name for d in result.deferred} == {"redis-port", "vectors"}
    assert result.spend_rows == 0

    _assert_persisted_lexical_only(conn_a, scope, records)
    # Spend NOT recorded for the never-attempted calls.
    assert _no_spend_rows(conn_a) == 0


def test_condition_a_reflect_is_clean_noop(
    conn_a: sqlite3.Connection,
    conn_b: sqlite3.Connection,
    scope: config.ScopeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(a) ``reflect`` under no key → clean no-op, no exception, no spend."""
    monkeypatch.setattr(ClaudeCliBackend, "detect", staticmethod(lambda: False))
    settings = config.load_config(overrides={"llm": {"backend": "auto"}})
    # Seed one active record + one activity row so the no-op is meaningful.
    rf = _record_file(scope, "untouched", body="a durable note")
    files.write_record(rf)
    forka.upsert_record(conn_a, rf, scope)
    forkb.append_activity(
        conn_b, session_id="sess-1", ts=NOW_EPOCH, role="user", kind="prompt",
        body="working on the untouched note",
    )

    with _no_api_key():
        result = routine.reflect(conn_b, "sess-1", conn_a, scope, settings)

    assert result.reinforced == 0
    assert result.proposed_promotions == []
    assert result.spend_rows == 0
    # Nothing changed — reindex is the backstop (SC-9).
    rec = forka.select_record(conn_a, scope, "untouched")
    assert rec is not None
    assert rec.hit_count == 0
    assert _no_spend_rows(conn_a) == 0


# --------------------------------------------------------------------------- #
# 2. Condition (b) — anthropic absent, even WITH a key + forced sdk → lexical   #
# --------------------------------------------------------------------------- #


def test_condition_b_sdk_detect_false_when_package_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``anthropic`` poisoned, ``AnthropicSdkBackend.detect`` is False even WITH a key.

    Proves the function-local guarded import is what makes the missing package a
    clean ``False`` (architecture §4.2) rather than an ImportError at load.
    """
    from claudemem.enrich.backend_sdk import AnthropicSdkBackend

    monkeypatch.setenv(_API_KEY_ENV, "sk-ant-fake-present")
    with _anthropic_absent():
        assert AnthropicSdkBackend.detect() is False


def test_condition_b_forced_sdk_unavailable_warns_once_and_falls_to_lexical(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """(b) forced ``backend=sdk`` + key present + package absent → warn once → lexical.

    Forced-but-unavailable degrades; it never raises (SC-3, §5.9). The CLI detect
    is forced False so a real authed CLI on the box cannot interfere — the gate
    isolates the SDK-absent path deterministically.
    """
    import logging

    monkeypatch.setenv(_API_KEY_ENV, "sk-ant-fake-present")
    monkeypatch.setattr(ClaudeCliBackend, "detect", staticmethod(lambda: False))
    settings = config.load_config(overrides={"llm": {"backend": "sdk"}})

    with _anthropic_absent(), caplog.at_level(logging.WARNING, logger="claudemem"):
        resolved = be.select_backend(settings)

    assert isinstance(resolved, LexicalOnlyBackend)
    warns = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warns) == 1
    assert "backend=sdk" in warns[0].getMessage()


def test_condition_b_enrich_batch_persists_pending_no_spend_no_raise(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) ``enrich_batch`` with package absent + key present + forced sdk degrades."""
    monkeypatch.setenv(_API_KEY_ENV, "sk-ant-fake-present")
    monkeypatch.setattr(ClaudeCliBackend, "detect", staticmethod(lambda: False))
    settings = config.load_config(overrides={"llm": {"backend": "sdk"}})
    records = _two_new_records(scope)

    with _anthropic_absent():
        result = routine.enrich_batch(conn_a, records, scope, settings)

    assert result.enriched == 0
    assert {d.record_name for d in result.deferred} == {"redis-port", "vectors"}
    assert result.spend_rows == 0
    _assert_persisted_lexical_only(conn_a, scope, records)
    assert _no_spend_rows(conn_a) == 0


def test_condition_b_reflect_is_clean_noop(
    conn_a: sqlite3.Connection,
    conn_b: sqlite3.Connection,
    scope: config.ScopeContext,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """(b) ``reflect`` with package absent + forced sdk → clean no-op, no spend."""
    monkeypatch.setenv(_API_KEY_ENV, "sk-ant-fake-present")
    monkeypatch.setattr(ClaudeCliBackend, "detect", staticmethod(lambda: False))
    settings = config.load_config(overrides={"llm": {"backend": "sdk"}})
    rf = _record_file(scope, "untouched", body="a durable note")
    files.write_record(rf)
    forka.upsert_record(conn_a, rf, scope)
    forkb.append_activity(
        conn_b, session_id="sess-1", ts=NOW_EPOCH, role="user", kind="prompt",
        body="working on the untouched note",
    )

    with _anthropic_absent():
        result = routine.reflect(conn_b, "sess-1", conn_a, scope, settings)

    assert result.reinforced == 0
    assert result.proposed_promotions == []
    assert result.spend_rows == 0
    rec = forka.select_record(conn_a, scope, "untouched")
    assert rec is not None
    assert rec.hit_count == 0
    assert _no_spend_rows(conn_a) == 0


# --------------------------------------------------------------------------- #
# 3. Both conditions degrade IDENTICALLY (NG-5)                                 #
# --------------------------------------------------------------------------- #


def _degraded_state_snapshot(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, records: list[files.RecordFile]
) -> list[tuple[str, int, str | None, list[str]]]:
    """Capture the persisted degraded state per record for an apples-to-apples compare."""
    snapshot: list[tuple[str, int, str | None, list[str]]] = []
    for rf in records:
        rec = forka.select_record(conn_a, scope, rf.name)
        assert rec is not None
        snapshot.append(
            (rec.name, rec.enrich_pending, rec.summary, files.aliases_from_json(rec.aliases_json))
        )
    return sorted(snapshot)


def test_both_conditions_degrade_identically(
    home: Path, scope: config.ScopeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The persisted state under (a) and (b) is byte-for-byte the same (NG-5).

    Each condition runs ``enrich_batch`` over the SAME two new records against a
    FRESH Fork A index, then the persisted ``(name, EnrichPending, summary,
    aliases)`` tuples are compared. ``deferred``/``spend`` ledgers are compared
    too. ``reason="auth"`` is asserted identical (a never-attempted lexical defer
    is an ``auth`` defer in both, tech-design §5.1).
    """
    monkeypatch.setattr(ClaudeCliBackend, "detect", staticmethod(lambda: False))

    # --- Condition (a): no key, backend=auto, fresh index. ---
    conn_a_a = index.open_forkA()
    try:
        settings_a = config.load_config(overrides={"llm": {"backend": "auto"}})
        records_a = _two_new_records(scope)
        with _no_api_key():
            result_a = routine.enrich_batch(conn_a_a, records_a, scope, settings_a)
        snap_a = _degraded_state_snapshot(conn_a_a, scope, records_a)
        spend_a = _no_spend_rows(conn_a_a)
    finally:
        conn_a_a.close()

    # Wipe Fork A so condition (b) writes onto a fresh index (apples-to-apples).
    (home / "forkA.db").unlink()
    # Re-clear the sandboxed Fork A files so the second pass re-persists from scratch.
    assert scope.project_dir is not None
    for md in scope.project_dir.glob("*.md"):
        md.unlink()

    # --- Condition (b): key present, backend=sdk, package absent, fresh index. ---
    monkeypatch.setenv(_API_KEY_ENV, "sk-ant-fake-present")
    conn_a_b = index.open_forkA()
    try:
        settings_b = config.load_config(overrides={"llm": {"backend": "sdk"}})
        records_b = _two_new_records(scope)
        with _anthropic_absent():
            result_b = routine.enrich_batch(conn_a_b, records_b, scope, settings_b)
        snap_b = _degraded_state_snapshot(conn_a_b, scope, records_b)
        spend_b = _no_spend_rows(conn_a_b)
    finally:
        conn_a_b.close()

    # Identical persisted state, identical empty ledgers, identical (no) spend.
    assert snap_a == snap_b
    assert result_a.enriched == result_b.enriched == 0
    assert sorted(d.record_name for d in result_a.deferred) == sorted(
        d.record_name for d in result_b.deferred
    )
    assert {d.reason for d in result_a.deferred} == {d.reason for d in result_b.deferred} == {"auth"}
    assert result_a.spend_rows == result_b.spend_rows == 0
    assert spend_a == spend_b == 0


# --------------------------------------------------------------------------- #
# 4. Deferral-ledger reasons are sensible, and a no-key/no-SDK defer never raises #
# --------------------------------------------------------------------------- #


def test_deferral_reasons_are_in_the_taxonomy_and_never_raise(
    conn_a: sqlite3.Connection, scope: config.ScopeContext, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Every degraded defer carries a reason ∈ {parse,cap,auth,transient}; never raises.

    The §5.1 deferral taxonomy. A never-attempted lexical-floor defer (no key / no
    SDK) is specifically ``auth`` (tech-design §5.1: auth/cap defers that never
    reach the model carry a DeferralEntry but no SpendLog row). Asserting the
    reason is in-taxonomy guards against a future change emitting a bogus reason.
    """
    valid_reasons = {"parse", "cap", "auth", "transient"}
    monkeypatch.setattr(ClaudeCliBackend, "detect", staticmethod(lambda: False))
    settings = config.load_config(overrides={"llm": {"backend": "auto"}})
    records = _two_new_records(scope)

    with _no_api_key():
        result = routine.enrich_batch(conn_a, records, scope, settings)

    assert result.deferred  # something deferred
    for report in result.deferred:
        assert report.reason in valid_reasons
        # A lexical-floor defer (never-attempted) is specifically 'auth'.
        assert report.reason == "auth"


# --------------------------------------------------------------------------- #
# 5. Consolidation reference — the gate sits on the unit-tested backend boundary #
#    (CLI parse/repair §5.6 + SDK idempotency §5.8 are NOT re-tested here)        #
# --------------------------------------------------------------------------- #


def test_lexical_floor_is_the_degradation_target_of_both_transports() -> None:
    """Reference: the §10.6 gate's degraded floor is ``LexicalOnlyBackend`` (no model).

    The CLI defensive parse/repair (§5.6) and SDK idempotency/retry (§5.8) are
    exhaustively unit-tested in ``test_enrich_backend_cli.py`` /
    ``test_enrich_backend_sdk.py``; this gate deliberately does NOT duplicate
    them. What it pins is the **floor** both transports fall to when unavailable:
    a model-free backend that defers every record (reason ``auth``) and records
    NO spend — the structural basis of the identical-degradation contract above.
    """
    lex = LexicalOnlyBackend()
    out = lex.enrich_batch(
        [be.EnrichRequest(record_id="r", name="r", body="b", candidates=[])]
    )
    assert out.results == []
    assert [d.reason for d in out.deferred] == ["auth"]
    assert out.spend == []
    # And reflect degrades to a fully empty outcome (reindex is the backstop, SC-9).
    empty = lex.reflect(be.ReflectRequest(session_id="s", activity=[], active_record_ids=[]))
    assert empty.passive_hits == [] and empty.promotion_candidates == [] and empty.spend == []
