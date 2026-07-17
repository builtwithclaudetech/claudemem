"""Tests for the L4 hook dispatch entry (T5.4) — ``claudemem.hooks``.

Drives ``claudemem.hooks.dispatch`` against a **real** tmp store
(``CLAUDEMEM_HOME`` → ``tmp_path`` so nothing touches the real ``~/.claude``),
feeding hook payloads via a monkeypatched ``sys.stdin``. **No test spawns a real
``claude`` or makes a real model call** — the SessionEnd path uses an injected
fake backend, and the model-free paths are asserted model-free via a
``sys.modules`` check (``enrich`` / ``anthropic`` absent after a fresh import).

Coverage maps to the spec ids:

* **SC-1 / §7.1 / §6.3** — recursion guard: ``CLAUDEMEM_DISABLE_HOOKS=1`` → every
  event exits 0 as a no-op BEFORE reading stdin or opening a DB.
* **SC-3** — always exit 0: malformed/empty stdin and a DB error both → exit 0.
* **IN-11 / SC-5** — SessionStart emits the menu ``additionalContext`` within the
  10 KB ceiling; ``source=resume`` → no injection.
* **IN-12 / SC-6** — UserPromptSubmit appends a Fork B prompt row, model-free.
* **AS-10 / IN-12** — Stop reads from the ``Cursor`` watermark, appends rows
  (cap / tool-skip applied), advances the watermark; a malformed line is skipped.
* **§5.7 / IN-14** — SessionEnd runs reflect THEN backfill in that order; a
  degraded backend → no-op, exit 0.
* **§7.4** — session_id fallback chain; two missing-id sessions get distinct ids.
"""

from __future__ import annotations

import io
import json
import sqlite3
from pathlib import Path

import pytest

from claudemem import config, files, hooks, index
from claudemem.enrich import backend as backend_mod
from claudemem.enrich.backend import (
    BackendOutcome,
    ReflectOutcome,
    ReflectRequest,
)

NOW_ISO = "2026-05-30T12:00:00Z"


# --------------------------------------------------------------------------- #
# Fixtures                                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME (DB files) at tmp_path; cwd → tmp_path for scope."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    monkeypatch.delenv(hooks.GUARD_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def scoped_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Redirect the global + project Fork A memory dirs into tmp_path."""
    global_dir = tmp_path / "global_memory"
    projects_root = tmp_path / "projects"
    global_dir.mkdir(parents=True, exist_ok=True)
    projects_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "GLOBAL_MEMORY_DIR", global_dir)
    monkeypatch.setattr(config, "PROJECTS_ROOT", projects_root)
    return global_dir, projects_root


def _feed_stdin(monkeypatch: pytest.MonkeyPatch, payload: str | dict[str, object]) -> None:
    """Replace sys.stdin with a StringIO carrying the hook JSON payload."""
    text = payload if isinstance(payload, str) else json.dumps(payload)
    monkeypatch.setattr("sys.stdin", io.StringIO(text))


def _project_dir(scoped_dirs: tuple[Path, Path]) -> Path:
    """The cwd-derived project memory dir for the active scope."""
    _global_dir, projects_root = scoped_dirs
    scope_ctx = config.resolve_scope(Path.cwd(), None, None)
    assert scope_ctx.project_id is not None
    return projects_root / scope_ctx.project_id / "memory"


def _write_record(
    directory: Path,
    name: str,
    *,
    scope: str,
    summary: str | None,
    importance: int = 3,
    pinned: bool = False,
    enrich_pending: bool = False,
) -> None:
    """Write a Fork A markdown record + upsert it into the index."""
    directory.mkdir(parents=True, exist_ok=True)
    record = files.RecordFile(
        path=directory / f"{name}.md",
        name=name,
        type="reference",
        scope=scope,
        importance=importance,
        pinned=pinned,
        source="explicit",
        created=NOW_ISO,
        last_accessed=NOW_ISO,
        access_count=0,
        hit_count=0,
        summary=summary,
        aliases=[],
        superseded_by=None,
        stale=False,
        body=f"body of {name}",
    )
    files.write_record(record)
    scope_ctx = config.resolve_scope(Path.cwd(), None, None)
    conn = index.open_forkA()
    try:
        forka = __import__("claudemem.store.forka", fromlist=["forka"])
        forka.upsert_record(conn, record, scope_ctx, enrich_pending=enrich_pending)
    finally:
        conn.close()


class SpyBackend:
    """Records call order so the SessionEnd reflect-then-backfill order is testable."""

    def __init__(self) -> None:
        self.order: list[str] = []

    @staticmethod
    def detect() -> bool:
        return True

    def enrich_batch(self, reqs: list[object]) -> BackendOutcome:
        self.order.append("enrich_batch")
        return BackendOutcome(results=[], deferred=[], spend=[])

    def reflect(self, req: ReflectRequest) -> ReflectOutcome:
        self.order.append("reflect")
        return ReflectOutcome()


# --------------------------------------------------------------------------- #
# Recursion guard (SC-1 / §7.1 / §6.3) — first statement, before stdin/DB        #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "event", ["session-start", "user-prompt-submit", "stop", "session-end"]
)
def test_guard_noop_before_stdin_or_db(
    event: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard set → exit 0 BEFORE any stdin read or DB open (SC-1, §6.3 MF-3)."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    monkeypatch.setenv(hooks.GUARD_ENV_VAR, "1")

    # stdin that would explode if read; DB opens that would explode if reached.
    class _BoomStdin:
        def read(self) -> str:
            raise AssertionError("stdin must not be read under the guard")

    monkeypatch.setattr("sys.stdin", _BoomStdin())

    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("DB must not open under the guard")

    monkeypatch.setattr(index, "open_forkA", _boom)
    monkeypatch.setattr(index, "open_forkB", _boom)

    assert hooks.dispatch(event) == 0
    # And no DB file was created under the guard.
    assert not (tmp_path / index.FORKA_FILENAME).exists()
    assert not (tmp_path / index.FORKB_FILENAME).exists()


def test_guard_uppercase_event_noop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The PascalCase Claude Code event name also no-ops under the guard."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    monkeypatch.setenv(hooks.GUARD_ENV_VAR, "1")
    assert hooks.dispatch("SessionStart") == 0


# --------------------------------------------------------------------------- #
# Always exit 0 (SC-3)                                                           #
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("bad", ["", "not json", "{unterminated", "\x00\x01 noise", "[1,2,3]"])
def test_malformed_stdin_exits_zero(
    bad: str, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Malformed / empty / non-object stdin → exit 0, never raises (SC-3)."""
    _feed_stdin(monkeypatch, bad)
    assert hooks.dispatch("user-prompt-submit") == 0


def test_db_error_exits_zero_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A DB error (unwritable home) → exit 0 + a logfile line, never raises (SC-3)."""
    home_dir = tmp_path / "home"
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(home_dir))
    monkeypatch.delenv(hooks.GUARD_ENV_VAR, raising=False)

    def _boom() -> sqlite3.Connection:
        raise sqlite3.OperationalError("disk I/O error (simulated)")

    monkeypatch.setattr(index, "open_forkB", _boom)
    _feed_stdin(monkeypatch, {"prompt": "hello"})

    assert hooks.dispatch("user-prompt-submit") == 0
    # The §7.1 local logfile captured the failure.
    log = home_dir / "claudemem.log"
    assert log.is_file()
    assert "user-prompt-submit" in log.read_text(encoding="utf-8")


def test_unknown_event_noop(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unhandled event → clean no-op, exit 0 (SC-3)."""
    _feed_stdin(monkeypatch, {})
    assert hooks.dispatch("PreToolUse") == 0


# --------------------------------------------------------------------------- #
# SessionStart → menu additionalContext (IN-11 / SC-5)                           #
# --------------------------------------------------------------------------- #


def test_session_start_emits_additional_context(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SessionStart emits the menu in the documented additionalContext shape."""
    _write_record(
        _project_dir(scoped_dirs), "alpha-note", scope="project",
        summary="Alpha summary", importance=5, pinned=True,
    )
    _feed_stdin(monkeypatch, {"source": "startup", "cwd": str(home)})

    assert hooks.dispatch("SessionStart") == 0
    out = capsys.readouterr().out
    parsed = json.loads(out)
    hso = parsed["hookSpecificOutput"]
    assert hso["hookEventName"] == "SessionStart"
    assert "a:alpha-note" in hso["additionalContext"]
    assert len(hso["additionalContext"]) <= 10_000


def test_session_start_resume_no_injection(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """source=resume → menu returns "" → nothing emitted (IN-11)."""
    _write_record(
        _project_dir(scoped_dirs), "beta-note", scope="project", summary="Beta",
    )
    _feed_stdin(monkeypatch, {"source": "resume", "cwd": str(home)})

    assert hooks.dispatch("SessionStart") == 0
    assert capsys.readouterr().out == ""


def test_session_start_empty_store_no_injection(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """An empty store → empty menu → nothing emitted (clean, not a blank block)."""
    _feed_stdin(monkeypatch, {"source": "startup", "cwd": str(home)})
    assert hooks.dispatch("SessionStart") == 0
    assert capsys.readouterr().out == ""


# --------------------------------------------------------------------------- #
# UserPromptSubmit → log (IN-12 / SC-6) — model-free                             #
# --------------------------------------------------------------------------- #


def _activity_rows(home: Path) -> list[sqlite3.Row]:
    conn = index.open_forkB()
    try:
        conn.row_factory = sqlite3.Row
        return conn.execute(
            "SELECT SessionId, Role, Kind, Body, ToolRef FROM Activity ORDER BY Id;"
        ).fetchall()
    finally:
        conn.close()


def test_user_prompt_submit_appends_row(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """UserPromptSubmit appends one Fork B row (role user, kind prompt)."""
    _feed_stdin(monkeypatch, {"session_id": "S1", "prompt": "remember the deploy steps"})
    assert hooks.dispatch("UserPromptSubmit") == 0

    rows = _activity_rows(home)
    assert len(rows) == 1
    assert rows[0]["SessionId"] == "S1"
    assert rows[0]["Role"] == "user"
    assert rows[0]["Kind"] == "prompt"
    assert rows[0]["Body"] == "remember the deploy steps"


def test_user_prompt_submit_no_prompt_is_noop(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A payload with no prompt → clean no-op, no row written."""
    _feed_stdin(monkeypatch, {"session_id": "S1"})
    assert hooks.dispatch("UserPromptSubmit") == 0
    assert _activity_rows(home) == []


def _assert_model_free(event: str, payload: dict[str, object]) -> None:
    """Dispatch ``event`` and assert no enrich / anthropic was imported (SC-6/IN-12).

    Snapshots and **restores** the ``claudemem.enrich`` / ``anthropic`` modules
    around the deletion so the destructive ``sys.modules`` clear never leaks into
    later tests (otherwise a re-imported ``routine`` binds a *different* ``backend``
    object than a test's top-level import, breaking the SessionEnd spy).
    """
    import io
    import json
    import sys

    snapshot = {
        m: sys.modules[m]
        for m in list(sys.modules)
        if m.startswith("claudemem.enrich") or m == "anthropic"
    }
    for m in snapshot:
        del sys.modules[m]
    try:
        sys.stdin = io.StringIO(json.dumps(payload))  # noqa: PTH123 — StringIO stdin
        assert hooks.dispatch(event) == 0
        assert "anthropic" not in sys.modules
        assert not any(m.startswith("claudemem.enrich") for m in sys.modules)
    finally:
        sys.modules.update(snapshot)


def test_log_path_is_model_free(home: Path) -> None:
    """The UserPromptSubmit log path imports NO enrich / anthropic (SC-6/IN-12)."""
    _assert_model_free("UserPromptSubmit", {"session_id": "S1", "prompt": "p"})


def test_session_start_path_is_model_free(home: Path) -> None:
    """The SessionStart menu path imports NO enrich / anthropic (read-path firewall)."""
    _assert_model_free("SessionStart", {"session_id": "S1"})


# --------------------------------------------------------------------------- #
# Stop → log transcript from Cursor watermark (AS-10 / IN-12)                    #
# --------------------------------------------------------------------------- #


def _transcript(path: Path, lines: list[dict[str, object]], *, raw_extra: str = "") -> None:
    """Write a JSONL transcript fixture (+ optional raw malformed line)."""
    text = "\n".join(json.dumps(line) for line in lines)
    if raw_extra:
        text += "\n" + raw_extra
    path.write_text(text + "\n", encoding="utf-8")


def test_stop_reads_transcript_appends_rows_advances_cursor(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stop ingests new turns, applies tool-skip, advances the watermark (AS-10)."""
    transcript = tmp_path / "transcript.jsonl"
    _transcript(
        transcript,
        [
            {"type": "user", "message": {"role": "user", "content": "user turn one"}},
            {"type": "assistant", "message": {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "pondering"},
                {"type": "text", "text": "assistant reply"},
                {"type": "tool_use", "name": "Read", "input": {"file": "x"}},
            ]}},
            {"type": "user", "message": {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "t1", "content": "BIG TOOL OUTPUT" * 50},
            ]}},
            {"type": "queue-operation", "operation": "enqueue"},  # control row → skipped
        ],
        raw_extra="{this is a malformed line",  # AS-10 — skipped, not fatal
    )
    _feed_stdin(monkeypatch, {"session_id": "S1", "transcript_path": str(transcript)})

    assert hooks.dispatch("Stop") == 0

    rows = _activity_rows(home)
    kinds = [(r["Role"], r["Kind"]) for r in rows]
    assert ("user", "prompt") in kinds
    assert ("assistant", "thinking") in kinds
    assert ("assistant", "text") in kinds
    # Tool blocks are role='tool' → Body skipped to a ToolRef (model-free §3.5).
    tool_rows = [r for r in rows if r["Role"] == "tool"]
    assert len(tool_rows) == 2
    for tr in tool_rows:
        assert tr["Body"] is None
        assert tr["ToolRef"] and tr["ToolRef"].startswith("tool:")

    # The watermark advanced past every line (including the malformed tail).
    conn = index.open_forkB()
    try:
        forkb = __import__("claudemem.store.forkb", fromlist=["forkb"])
        assert forkb.get_cursor(conn, "S1") == 5
    finally:
        conn.close()


def test_stop_incremental_from_watermark(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A second Stop reads only the lines after the prior watermark (incremental)."""
    transcript = tmp_path / "t.jsonl"
    _transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "first"}},
    ])
    _feed_stdin(monkeypatch, {"session_id": "S2", "transcript_path": str(transcript)})
    assert hooks.dispatch("Stop") == 0
    assert len(_activity_rows(home)) == 1

    # Append a second turn; the next Stop ingests only the new line.
    _transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "first"}},
        {"type": "assistant", "message": {"role": "assistant", "content": "second"}},
    ])
    _feed_stdin(monkeypatch, {"session_id": "S2", "transcript_path": str(transcript)})
    assert hooks.dispatch("Stop") == 0

    rows = _activity_rows(home)
    assert len(rows) == 2
    assert rows[1]["Body"] == "second"


def test_stop_missing_transcript_is_noop(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing transcript_path → clean no-op, exit 0."""
    _feed_stdin(monkeypatch, {"session_id": "S3", "transcript_path": str(tmp_path / "nope.jsonl")})
    assert hooks.dispatch("Stop") == 0
    assert _activity_rows(home) == []


def test_stop_path_is_model_free(home: Path, tmp_path: Path) -> None:
    """The Stop log path imports NO enrich / anthropic (SC-6/IN-12)."""
    transcript = tmp_path / "t.jsonl"
    _transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "hi"}},
    ])
    _assert_model_free("Stop", {"session_id": "S1", "transcript_path": str(transcript)})


# --------------------------------------------------------------------------- #
# SessionEnd → reflect THEN backfill (§5.7) — fake backend, no model            #
# --------------------------------------------------------------------------- #


def test_session_end_reflect_then_backfill_order(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SessionEnd runs reflect FIRST then enrich_batch backfill (§5.7 order)."""
    # One degraded record so the backfill actually calls enrich_batch.
    _write_record(
        _project_dir(scoped_dirs), "pending-rec", scope="project",
        summary=None, enrich_pending=True,
    )
    spy = SpyBackend()
    monkeypatch.setattr(backend_mod, "select_backend", lambda settings: spy)
    backend_mod._reset_cache_for_tests()

    _feed_stdin(monkeypatch, {"session_id": "S1", "cwd": str(home)})
    assert hooks.dispatch("SessionEnd") == 0

    # Reflection precedes the EnrichPending backfill (the carry-forward order).
    assert spy.order == ["reflect", "enrich_batch"]


def test_session_end_no_pending_only_reflects(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With nothing EnrichPending, SessionEnd reflects and skips the backfill."""
    spy = SpyBackend()
    monkeypatch.setattr(backend_mod, "select_backend", lambda settings: spy)
    backend_mod._reset_cache_for_tests()

    _feed_stdin(monkeypatch, {"session_id": "S1", "cwd": str(home)})
    assert hooks.dispatch("SessionEnd") == 0
    assert spy.order == ["reflect"]


def test_session_end_degraded_backend_noop_exit_zero(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No transport (LexicalOnlyBackend) → SessionEnd is a clean no-op, exit 0 (SC-3)."""
    # Force the real lazy selection to land on lexical-only (no key, no CLI).
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setattr(backend_mod, "select_backend", lambda settings: backend_mod.LexicalOnlyBackend())
    backend_mod._reset_cache_for_tests()

    _feed_stdin(monkeypatch, {"session_id": "S1", "cwd": str(home)})
    assert hooks.dispatch("SessionEnd") == 0


# --------------------------------------------------------------------------- #
# session_id fallback chain (§7.4)                                              #
# --------------------------------------------------------------------------- #


def test_session_id_falls_back_to_transcript_stem(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing session_id → Path(transcript_path).stem (§7.4)."""
    transcript = tmp_path / "abc-123-def.jsonl"
    _transcript(transcript, [
        {"type": "user", "message": {"role": "user", "content": "hello"}},
    ])
    _feed_stdin(monkeypatch, {"transcript_path": str(transcript)})
    assert hooks.dispatch("Stop") == 0

    rows = _activity_rows(home)
    assert rows and rows[0]["SessionId"] == "abc-123-def"


def test_session_id_unknown_pid_fallback_distinct(monkeypatch: pytest.MonkeyPatch) -> None:
    """No session_id and no transcript → ``unknown-{pid}`` (distinct per pid, §7.4)."""
    sid = hooks._session_id({})
    assert sid.startswith("unknown-")
    # The pid suffix makes two id-less concurrent sessions distinct.
    import os

    assert sid == f"unknown-{os.getpid()}"
