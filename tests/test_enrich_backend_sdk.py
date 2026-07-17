"""Tests for ``claudemem.enrich.backend_sdk`` — the funneled Anthropic SDK backend.

Covers T4.3 (``detect`` + ``enrich_batch`` + the funneled import) and T4.4
(``reflect``). **No real Anthropic call is ever made** — every test injects a
fake client through the ``client_factory`` seam (or hides ``anthropic`` from
``sys.modules`` to exercise the ImportError branch), and ``sleep`` is injected so
backoff is instantaneous.

Mocking seam (documented):

* ``AnthropicSdkBackend(client_factory=..., sleep=...)`` — ``client_factory``
  returns a fake object exposing ``.messages.create(...)``; ``sleep`` replaces
  ``time.sleep`` so retries don't wait. No real ``anthropic.Anthropic()`` is
  constructed and no network request happens.
* The funneled-import / ImportError branch is exercised by setting
  ``sys.modules["anthropic"] = None`` (which makes ``import anthropic`` raise
  ImportError) and using a backend WITHOUT a ``client_factory``.
"""

from __future__ import annotations

import importlib
import subprocess
import sys
from typing import Any

import pytest

from claudemem import config
from claudemem.enrich import backend as be
from claudemem.enrich import backend_sdk as sdk


# --------------------------------------------------------------------------- #
# Fakes — a fake Anthropic client + response, no SDK, no network               #
# --------------------------------------------------------------------------- #


class _FakeToolUse:
    def __init__(self, name: str, tool_input: dict[str, Any]) -> None:
        self.type = "tool_use"
        self.name = name
        self.input = tool_input


class _FakeUsage:
    def __init__(self, input_tokens: int, output_tokens: int) -> None:
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeResponse:
    def __init__(self, content: list[Any], input_tokens: int = 100, output_tokens: int = 20) -> None:
        self.content = content
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeStatusError(Exception):
    """Mimics an anthropic ``APIStatusError`` subclass: carries ``status_code``."""

    def __init__(self, status_code: int) -> None:
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _FakeMessages:
    def __init__(self, outcomes: list[Any]) -> None:
        # Each entry is either a _FakeResponse (returned) or an Exception (raised).
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        outcome = self._outcomes[min(len(self.calls) - 1, len(self._outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _FakeClient:
    def __init__(self, outcomes: list[Any]) -> None:
        self.messages = _FakeMessages(outcomes)


def _enrich_tool_input(
    *,
    summary: str = "Redis runs on 6380.",
    aliases: list[str] | None = None,
    verdict: str = "new",
    target: str | None = None,
    conflict: str | None = None,
) -> dict[str, Any]:
    return {
        "summary": summary,
        "aliases": aliases if aliases is not None else ["redis"],
        "dedup_verdict": verdict,
        "dedup_target_name": target,
        "conflict_explanation": conflict,
    }


def _enrich_response(tool_input: dict[str, Any]) -> _FakeResponse:
    return _FakeResponse([_FakeToolUse(sdk._ENRICH_TOOL, tool_input)])


def _reflect_response(tool_input: dict[str, Any]) -> _FakeResponse:
    return _FakeResponse([_FakeToolUse(sdk._REFLECT_TOOL, tool_input)])


def _backend(outcomes: list[Any]) -> tuple[sdk.AnthropicSdkBackend, _FakeClient]:
    client = _FakeClient(outcomes)
    backend = sdk.AnthropicSdkBackend(
        settings=config.load_config(),
        client_factory=lambda: client,
        sleep=lambda _s: None,  # never actually sleep.
    )
    return backend, client


def _req(name: str = "redis-port", body: str = "Redis listens on 6380.", candidates: list[be.Candidate] | None = None) -> be.EnrichRequest:
    return be.EnrichRequest(
        record_id=name,
        name=name,
        body=body,
        candidates=candidates if candidates is not None else [],
    )


# --------------------------------------------------------------------------- #
# detect()                                                                      #
# --------------------------------------------------------------------------- #


def test_detect_true_with_key_and_package(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # anthropic IS installed in the .venv, so detect should be True.
    assert sdk.AnthropicSdkBackend.detect() is True


def test_detect_false_without_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert sdk.AnthropicSdkBackend.detect() is False


def test_detect_false_on_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    # Hiding anthropic behind a None entry makes `import anthropic` raise ImportError.
    monkeypatch.setitem(sys.modules, "anthropic", None)
    assert sdk.AnthropicSdkBackend.detect() is False


# --------------------------------------------------------------------------- #
# C-17: no top-level anthropic import                                           #
# --------------------------------------------------------------------------- #


def test_importing_module_does_not_import_anthropic() -> None:
    """Importing backend_sdk must NOT pull anthropic into sys.modules (C-17)."""
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, claudemem.enrich.backend_sdk; "
            "assert 'anthropic' not in sys.modules, 'SDK leaked at import time'",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------- #
# enrich_batch — happy path                                                     #
# --------------------------------------------------------------------------- #


def test_enrich_happy_path_populates_result_and_spend() -> None:
    backend, client = _backend([_enrich_response(_enrich_tool_input())])
    out = backend.enrich_batch([_req()])

    assert len(out.results) == 1
    assert out.deferred == []
    res = out.results[0]
    assert res.record_id == "redis-port"
    assert res.summary == "Redis runs on 6380."
    assert res.aliases == ["redis"]
    assert res.dedup_verdict == "new"
    assert res.dedup_target_name is None

    assert len(out.spend) == 1
    se = out.spend[0]
    assert se.call_site == "save"
    assert se.backend == "sdk"
    assert se.outcome == "ok"
    assert se.retry_count == 0
    assert se.idempotency_key is not None
    assert se.input_tokens == 100
    assert se.output_tokens == 20

    # forced tool-use was requested with the exact schema + tool_choice.
    create_kwargs = client.messages.calls[0]
    assert create_kwargs["tool_choice"] == {"type": "tool", "name": "record_memory_analysis"}
    assert create_kwargs["tools"][0]["name"] == "record_memory_analysis"
    assert create_kwargs["model"] == "claude-haiku-4-5-20251001"
    assert create_kwargs["extra_headers"]["Idempotency-Key"] == se.idempotency_key


def test_enrich_duplicate_verdict_still_has_summary_and_aliases() -> None:
    cand = be.Candidate(name="redis-port", summary="Redis on 6380.", aliases=["redis"], excerpt="…")
    tool_input = _enrich_tool_input(verdict="duplicate", target="redis-port")
    backend, _ = _backend([_enrich_response(tool_input)])
    out = backend.enrich_batch([_req(name="redis-2", candidates=[cand])])

    res = out.results[0]
    assert res.dedup_verdict == "duplicate"
    assert res.dedup_target_name == "redis-port"
    assert res.summary  # present regardless of verdict (IN-13)
    assert res.aliases  # 1..8, present regardless of verdict


def test_enrich_one_call_per_record() -> None:
    backend, client = _backend([_enrich_response(_enrich_tool_input())])
    backend.enrich_batch([_req(name="a"), _req(name="b"), _req(name="c")])
    assert len(client.messages.calls) == 3  # IN-13: one model call per record.


# --------------------------------------------------------------------------- #
# enrich_batch — defensive validation                                           #
# --------------------------------------------------------------------------- #


def test_dedup_target_out_of_set_coerced_to_new() -> None:
    cand = be.Candidate(name="real-cand", summary="s", aliases=[], excerpt="e")
    tool_input = _enrich_tool_input(verdict="duplicate", target="hallucinated-name")
    backend, _ = _backend([_enrich_response(tool_input)])
    out = backend.enrich_batch([_req(candidates=[cand])])

    res = out.results[0]
    assert res.dedup_verdict == "new"  # out-of-set target → coerced (§5.2)
    assert res.dedup_target_name is None
    assert res.conflict_explanation is None


def test_aliases_over_eight_are_clamped() -> None:
    tool_input = _enrich_tool_input(aliases=[f"a{i}" for i in range(12)])
    backend, _ = _backend([_enrich_response(tool_input)])
    out = backend.enrich_batch([_req()])
    assert len(out.results[0].aliases) == 8


def test_empty_aliases_get_a_floor_entry() -> None:
    tool_input = _enrich_tool_input(aliases=[])
    backend, _ = _backend([_enrich_response(tool_input)])
    out = backend.enrich_batch([_req()])
    assert len(out.results[0].aliases) == 1  # minItems:1 invariant upheld


def test_conflict_verdict_with_valid_target_keeps_explanation() -> None:
    cand = be.Candidate(name="redis-port", summary="Redis on 6379.", aliases=[], excerpt="e")
    tool_input = _enrich_tool_input(
        verdict="conflict", target="redis-port", conflict="6379 vs 6380"
    )
    backend, _ = _backend([_enrich_response(tool_input)])
    out = backend.enrich_batch([_req(candidates=[cand])])
    res = out.results[0]
    assert res.dedup_verdict == "conflict"
    assert res.dedup_target_name == "redis-port"
    assert res.conflict_explanation == "6379 vs 6380"


# --------------------------------------------------------------------------- #
# Retry / defer matrix (§5.8)                                                   #
# --------------------------------------------------------------------------- #


def test_transient_then_success_records_retry_count() -> None:
    backend, client = _backend(
        [_FakeStatusError(503), _enrich_response(_enrich_tool_input())]
    )
    out = backend.enrich_batch([_req()])
    assert len(out.results) == 1
    assert out.spend[0].outcome == "ok"
    assert out.spend[0].retry_count == 1  # one transient retry consumed
    assert len(client.messages.calls) == 2


def test_429_is_transient_and_retried() -> None:
    backend, client = _backend(
        [_FakeStatusError(429), _enrich_response(_enrich_tool_input())]
    )
    out = backend.enrich_batch([_req()])
    assert len(out.results) == 1
    assert out.spend[0].retry_count == 1
    assert len(client.messages.calls) == 2


def test_three_transient_failures_defer_transient() -> None:
    backend, client = _backend([_FakeStatusError(500)])  # always fails
    out = backend.enrich_batch([_req()])
    assert out.results == []
    assert len(out.deferred) == 1
    assert out.deferred[0].reason == "transient"
    # 1 initial + 3 retries = 4 attempts.
    assert len(client.messages.calls) == 4
    assert out.spend[0].outcome == "deferred"
    assert out.spend[0].retry_count == sdk._MAX_RETRIES


def test_401_defers_auth_without_retry() -> None:
    backend, client = _backend([_FakeStatusError(401)])
    out = backend.enrich_batch([_req()])
    assert out.results == []
    assert len(out.deferred) == 1
    assert out.deferred[0].reason == "auth"
    assert len(client.messages.calls) == 1  # NOT retried (§5.8)
    # No SpendLog row for an auth defer that never produced a usable call result.
    assert out.spend == []


def test_auth_failure_short_circuits_remaining_records() -> None:
    backend, client = _backend([_FakeStatusError(401)])
    out = backend.enrich_batch([_req(name="a"), _req(name="b"), _req(name="c")])
    assert len(out.deferred) == 3
    assert all(d.reason == "auth" for d in out.deferred)
    # Only the first record attempted a call; auth-dead cache short-circuits rest.
    assert len(client.messages.calls) == 1


def test_missing_tool_block_is_transient() -> None:
    # A response with no forced tool_use block → transient → retried then deferred.
    bad = _FakeResponse([])
    backend, client = _backend([bad])
    out = backend.enrich_batch([_req()])
    assert out.deferred[0].reason == "transient"
    assert len(client.messages.calls) == 4


# --------------------------------------------------------------------------- #
# Idempotency key (§5.8)                                                        #
# --------------------------------------------------------------------------- #


def test_idempotency_key_deterministic_and_body_sensitive() -> None:
    k1 = sdk._idempotency_key("save", "redis-port", "Redis on 6380.")
    k2 = sdk._idempotency_key("save", "redis-port", "Redis on 6380.")
    assert k1 == k2  # deterministic

    # Whitespace-only differences normalize to the same key.
    k3 = sdk._idempotency_key("save", "redis-port", "  Redis   on\t6380. ")
    assert k1 == k3

    # A genuine body change → different key.
    k4 = sdk._idempotency_key("save", "redis-port", "Redis on 6379.")
    assert k1 != k4

    # call_site participates.
    k5 = sdk._idempotency_key("reflect", "redis-port", "Redis on 6380.")
    assert k1 != k5


# --------------------------------------------------------------------------- #
# reflect (T4.4, §5.3)                                                          #
# --------------------------------------------------------------------------- #


def _reflect_req() -> be.ReflectRequest:
    return be.ReflectRequest(
        session_id="sess-1",
        activity=[
            be.ActivityRow(archive_id="b:1", role="user", kind="msg", body="use redis 6380"),
            be.ActivityRow(archive_id="b:2", role="assistant", kind="msg", body="done"),
        ],
        active_record_ids=["redis-port", "vps-host"],
    )


def test_reflect_parses_hits_and_candidates() -> None:
    tool_input = {
        "passive_hits": [{"record_id": "redis-port", "evidence": "used port 6380"}],
        "promotion_candidates": [
            {"archive_id": "b:1", "proposed_summary": "Redis on 6380", "rationale": "recurring"}
        ],
    }
    backend, client = _backend([_reflect_response(tool_input)])
    out = backend.reflect(_reflect_req())

    assert len(out.passive_hits) == 1
    assert out.passive_hits[0].record_id == "redis-port"
    assert len(out.promotion_candidates) == 1
    assert out.promotion_candidates[0].archive_id == "b:1"
    assert len(out.spend) == 1
    assert out.spend[0].call_site == "reflect"
    assert out.spend[0].outcome == "ok"
    assert client.messages.calls[0]["tool_choice"] == {"type": "tool", "name": "session_reflection"}


def test_reflect_drops_out_of_set_ids() -> None:
    tool_input = {
        "passive_hits": [
            {"record_id": "redis-port", "evidence": "ok"},
            {"record_id": "ghost-record", "evidence": "hallucinated"},
        ],
        "promotion_candidates": [
            {"archive_id": "b:1", "proposed_summary": "s", "rationale": "r"},
            {"archive_id": "b:999", "proposed_summary": "s", "rationale": "r"},
        ],
    }
    backend, _ = _backend([_reflect_response(tool_input)])
    out = backend.reflect(_reflect_req())
    assert [h.record_id for h in out.passive_hits] == ["redis-port"]
    assert [c.archive_id for c in out.promotion_candidates] == ["b:1"]


def test_reflect_empty_arrays_ok() -> None:
    tool_input = {"passive_hits": [], "promotion_candidates": []}
    backend, _ = _backend([_reflect_response(tool_input)])
    out = backend.reflect(_reflect_req())
    assert out.passive_hits == []
    assert out.promotion_candidates == []
    assert out.spend[0].outcome == "ok"


def test_reflect_auth_failure_returns_empty_no_raise() -> None:
    backend, client = _backend([_FakeStatusError(401)])
    out = backend.reflect(_reflect_req())
    assert out.passive_hits == []
    assert out.promotion_candidates == []
    assert out.spend == []  # auth defer, never-attempted-usable → no spend row
    assert len(client.messages.calls) == 1


def test_reflect_transient_exhausted_defers_with_spend() -> None:
    backend, _ = _backend([_FakeStatusError(500)])
    out = backend.reflect(_reflect_req())
    assert out.passive_hits == []
    assert len(out.spend) == 1
    assert out.spend[0].outcome == "deferred"
    assert out.spend[0].retry_count == sdk._MAX_RETRIES


# --------------------------------------------------------------------------- #
# ImportError path — no client_factory, anthropic hidden → defer, no raise      #
# --------------------------------------------------------------------------- #


def test_enrich_import_error_defers_all_no_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # import anthropic → ImportError
    backend = sdk.AnthropicSdkBackend(
        settings=config.load_config(), sleep=lambda _s: None
    )  # NO client_factory → real funneled import path.
    out = backend.enrich_batch([_req(name="a"), _req(name="b")])
    assert out.results == []
    assert len(out.deferred) == 2
    assert all(d.reason == "auth" for d in out.deferred)
    assert out.spend == []


def test_reflect_import_error_returns_empty_no_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setitem(sys.modules, "anthropic", None)
    backend = sdk.AnthropicSdkBackend(
        settings=config.load_config(), sleep=lambda _s: None
    )
    out = backend.reflect(_reflect_req())
    assert out.passive_hits == []
    assert out.promotion_candidates == []
    assert out.spend == []


def test_missing_key_without_factory_defers_auth(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    # anthropic importable, but no key → _AuthUnavailable → auth defer.
    importlib.import_module("anthropic")  # ensure it is importable in this run
    backend = sdk.AnthropicSdkBackend(
        settings=config.load_config(), sleep=lambda _s: None
    )
    out = backend.enrich_batch([_req()])
    assert out.deferred[0].reason == "auth"
    assert out.spend == []
