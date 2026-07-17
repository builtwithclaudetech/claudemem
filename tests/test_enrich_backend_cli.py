"""Tests for ``claudemem.enrich.backend_cli`` — the ``ClaudeCliBackend``.

Covers T4.5 (spawn + recursion-guard env MERGE + cached auth probe) and T4.6
(defensive JSON-array parse/repair + chunking + ``enrich_batch`` / ``reflect``).

**No real ``claude`` spawn and no real sleep.** ``subprocess.run`` is replaced
with a recording fake (``FakeRunner``) injected via ``runner=``; ``time.sleep``
is replaced with a no-op via ``sleeper=``; the auth probe is monkeypatched at
``shutil.which`` + ``subprocess.run``. Everything runs offline.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from typing import Any

import pytest

from claudemem import config
from claudemem.enrich import backend_cli as bc
from claudemem.enrich.backend import (
    ActivityRow,
    Candidate,
    EnrichRequest,
    ReflectRequest,
)


# --------------------------------------------------------------------------- #
# Test doubles — a recording fake subprocess.run, no real spawn.                #
# --------------------------------------------------------------------------- #


@dataclass
class SpawnCall:
    """One captured invocation of the fake runner."""

    argv: list[str]
    input: str | None
    env: dict[str, str] | None


class FakeRunner:
    """Records each call and replays a scripted sequence of outcomes.

    Each scripted item is either:
      * a ``str`` → returned as the ``--output-format json`` envelope's stdout
        with returncode 0 (the str is the full envelope JSON);
      * an ``int`` → a non-zero returncode (transient failure);
      * the ``TIMEOUT`` sentinel → raises ``subprocess.TimeoutExpired``.
    The last item repeats once the script is exhausted.
    """

    TIMEOUT = object()

    def __init__(self, script: list[Any]) -> None:
        self._script = script
        self.calls: list[SpawnCall] = []

    def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        self.calls.append(
            SpawnCall(argv=argv, input=kwargs.get("input"), env=kwargs.get("env"))
        )
        item = self._script[min(len(self.calls) - 1, len(self._script) - 1)]
        if item is FakeRunner.TIMEOUT:
            raise subprocess.TimeoutExpired(cmd="claude", timeout=120)
        if isinstance(item, int):
            return subprocess.CompletedProcess(argv, returncode=item, stdout="", stderr="boom")
        return subprocess.CompletedProcess(argv, returncode=0, stdout=item, stderr="")


def _envelope(result_text: str, *, in_tok: int = 100, out_tok: int = 50) -> str:
    """Build a ``claude -p --output-format json`` envelope around a result text."""
    return json.dumps(
        {
            "type": "result",
            "result": result_text,
            "usage": {"input_tokens": in_tok, "output_tokens": out_tok},
        }
    )


def _element(record_id: str, *, verdict: str = "new", target: Any = None) -> dict[str, Any]:
    """A valid §5.2 enrichment element for a record."""
    return {
        "record_id": record_id,
        "summary": f"summary for {record_id}",
        "aliases": [record_id, "alias2"],
        "dedup_verdict": verdict,
        "dedup_target_name": target,
        "conflict_explanation": "they disagree" if verdict == "conflict" else None,
    }


def _settings(**llm_overrides: Any) -> config.Settings:
    return config.load_config(overrides={"llm": llm_overrides}) if llm_overrides else config.Settings()


def _backend(script: list[Any], **llm_overrides: Any) -> tuple[bc.ClaudeCliBackend, FakeRunner]:
    runner = FakeRunner(script)
    backend = bc.ClaudeCliBackend(
        _settings(**llm_overrides), runner=runner, sleeper=lambda _s: None
    )
    return backend, runner


def _reqs(n: int, *, candidates: list[Candidate] | None = None) -> list[EnrichRequest]:
    return [
        EnrichRequest(
            record_id=f"r{i}",
            name=f"r{i}",
            body=f"body {i}",
            candidates=candidates if candidates is not None else [],
        )
        for i in range(n)
    ]


@pytest.fixture(autouse=True)
def _clear_detect_cache() -> None:
    """Reset the per-process auth-probe cache before each test (§7.3 caching)."""
    bc.ClaudeCliBackend._reset_detect_cache_for_tests()


# --------------------------------------------------------------------------- #
# T4.5 — detect() availability probe (cached, env-merge probe spawn).           #
# --------------------------------------------------------------------------- #


def test_detect_true_when_which_and_auth_status_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc.shutil, "which", lambda _name: "/usr/bin/claude")
    captured: dict[str, Any] = {}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        captured["argv"] = argv
        captured["env"] = kwargs.get("env")
        return subprocess.CompletedProcess(argv, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(bc.subprocess, "run", fake_run)
    assert bc.ClaudeCliBackend.detect() is True
    # The auth probe spawn uses the env MERGE: guard set AND PATH preserved (MF-2).
    assert captured["env"][bc._DISABLE_HOOKS_ENV] == "1"
    assert "PATH" in captured["env"]
    assert captured["argv"][:3] == ["claude", "auth", "status"]


def test_detect_false_when_claude_not_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc.shutil, "which", lambda _name: None)
    # auth-status must never be spawned when which() is None.
    monkeypatch.setattr(
        bc.subprocess,
        "run",
        lambda *a, **k: pytest.fail("auth status spawned despite missing claude"),
    )
    assert bc.ClaudeCliBackend.detect() is False


def test_detect_false_when_auth_status_nonzero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc.shutil, "which", lambda _name: "/usr/bin/claude")
    monkeypatch.setattr(
        bc.subprocess,
        "run",
        lambda argv, **k: subprocess.CompletedProcess(argv, returncode=1, stdout="", stderr=""),
    )
    assert bc.ClaudeCliBackend.detect() is False


def test_detect_is_cached_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(bc.shutil, "which", lambda _name: "/usr/bin/claude")
    calls = {"n": 0}

    def fake_run(argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        return subprocess.CompletedProcess(argv, returncode=0, stdout="{}", stderr="")

    monkeypatch.setattr(bc.subprocess, "run", fake_run)
    assert bc.ClaudeCliBackend.detect() is True
    assert bc.ClaudeCliBackend.detect() is True
    assert calls["n"] == 1  # probed once, cached thereafter (§7.3 / NG-1).


# --------------------------------------------------------------------------- #
# T4.5 — recursion-guard env MERGE + spawn argv + stdin (the load-bearing test).#
# --------------------------------------------------------------------------- #


def test_spawn_env_is_a_merge_not_a_bare_dict() -> None:
    backend, runner = _backend([_envelope(json.dumps([_element("r0")]))])
    backend.enrich_batch(_reqs(1))
    env = runner.calls[0].env
    assert env is not None
    # The guard is set AND the inherited env (PATH/HOME) is preserved — a bare
    # env={"CLAUDEMEM_DISABLE_HOOKS": "1"} regression would fail these two lines.
    assert env[bc._DISABLE_HOOKS_ENV] == "1"
    assert "PATH" in env
    assert "HOME" in env


def test_spawn_argv_carries_recursion_guard_flags() -> None:
    backend, runner = _backend([_envelope(json.dumps([_element("r0")]))])
    backend.enrich_batch(_reqs(1))
    argv = runner.calls[0].argv
    assert argv[:2] == ["claude", "-p"]
    # `--bare` is intentionally absent: it forces API-key-only auth and breaks
    # subscription/OAuth enrichment (the design's billing path). Lock the fix.
    assert "--bare" not in argv
    assert "--max-turns" in argv and argv[argv.index("--max-turns") + 1] == "1"
    assert "--no-session-persistence" in argv
    assert "--output-format" in argv and argv[argv.index("--output-format") + 1] == "json"
    assert "--model" in argv


def test_prompt_goes_via_stdin_not_argv() -> None:
    backend, runner = _backend([_envelope(json.dumps([_element("r0")]))])
    backend.enrich_batch(_reqs(1))
    call = runner.calls[0]
    assert call.input is not None and "RECORDS" in call.input
    # The prompt text must NOT appear as an argv element.
    assert not any("RECORDS" in arg for arg in call.argv)


# --------------------------------------------------------------------------- #
# T4.6 — enrich_batch happy path.                                               #
# --------------------------------------------------------------------------- #


def test_enrich_batch_happy_path_keys_results_back_to_record_ids() -> None:
    reqs = _reqs(3)
    array = json.dumps([_element("r0"), _element("r1"), _element("r2")])
    backend, runner = _backend([_envelope(array, in_tok=200, out_tok=120)])
    outcome = backend.enrich_batch(reqs)
    assert len(runner.calls) == 1
    assert {r.record_id for r in outcome.results} == {"r0", "r1", "r2"}
    assert outcome.deferred == []
    r0 = next(r for r in outcome.results if r.record_id == "r0")
    assert r0.summary == "summary for r0"
    assert 1 <= len(r0.aliases) <= 8
    assert r0.dedup_verdict == "new"
    # One spend row, outcome 'ok', tokens from the usage block, CLI key None.
    assert len(outcome.spend) == 1
    spend = outcome.spend[0]
    assert spend.outcome == "ok"
    assert spend.backend == "cli"
    assert spend.input_tokens == 200 and spend.output_tokens == 120
    assert spend.idempotency_key is None


# --------------------------------------------------------------------------- #
# T4.6 — defensive parse / repair.                                              #
# --------------------------------------------------------------------------- #


def test_parse_strips_code_fences_and_prose_around_array() -> None:
    reqs = _reqs(1)
    array = json.dumps([_element("r0")])
    wrapped = f"Here is the result:\n```json\n{array}\n```\nDone."
    backend, runner = _backend([_envelope(wrapped)])
    outcome = backend.enrich_batch(reqs)
    assert len(runner.calls) == 1
    assert {r.record_id for r in outcome.results} == {"r0"}
    assert outcome.deferred == []


def test_one_invalid_element_among_valid_accepts_valid_defers_invalid() -> None:
    reqs = _reqs(2)
    bad = _element("r1")
    bad["aliases"] = []  # violates minItems:1 → invalid element.
    array = json.dumps([_element("r0"), bad])
    backend, _ = _backend([_envelope(array)])
    outcome = backend.enrich_batch(reqs)
    assert {r.record_id for r in outcome.results} == {"r0"}
    assert [d.record_id for d in outcome.deferred] == ["r1"]
    assert outcome.deferred[0].reason == "parse"


def test_dropped_record_not_in_array_defers_parse() -> None:
    reqs = _reqs(2)
    array = json.dumps([_element("r0")])  # r1 dropped by the model.
    backend, _ = _backend([_envelope(array)])
    outcome = backend.enrich_batch(reqs)
    assert {r.record_id for r in outcome.results} == {"r0"}
    assert [d.record_id for d in outcome.deferred] == ["r1"]
    assert outcome.deferred[0].reason == "parse"


def test_total_garbage_respawns_once_then_defers_whole_chunk_parse() -> None:
    reqs = _reqs(2)
    backend, runner = _backend([_envelope("totally not json at all")])
    outcome = backend.enrich_batch(reqs)
    # Exactly 2 spawns: original + one repair re-spawn (cli_parse_retries=1).
    assert len(runner.calls) == 2
    assert outcome.results == []
    assert {d.record_id for d in outcome.deferred} == {"r0", "r1"}
    assert all(d.reason == "parse" for d in outcome.deferred)
    # A spend row records the burned tokens with Outcome='deferred'.
    assert len(outcome.spend) == 1
    assert outcome.spend[0].outcome == "deferred"


def test_repaired_on_retry_sets_outcome_repaired() -> None:
    reqs = _reqs(1)
    array = json.dumps([_element("r0")])
    # First spawn = garbage, second (repair) spawn = valid → Outcome='repaired'.
    backend, runner = _backend([_envelope("garbage"), _envelope(array)])
    outcome = backend.enrich_batch(reqs)
    assert len(runner.calls) == 2
    assert {r.record_id for r in outcome.results} == {"r0"}
    assert len(outcome.spend) == 1
    assert outcome.spend[0].outcome == "repaired"


def test_no_repair_when_parse_retries_zero() -> None:
    reqs = _reqs(1)
    backend, runner = _backend([_envelope("garbage")], cli_parse_retries=0)
    outcome = backend.enrich_batch(reqs)
    assert len(runner.calls) == 1  # no repair re-spawn.
    assert {d.reason for d in outcome.deferred} == {"parse"}


# --------------------------------------------------------------------------- #
# T4.6 — chunking.                                                              #
# --------------------------------------------------------------------------- #


def test_chunking_splits_into_groups_of_chunk_size() -> None:
    reqs = _reqs(60)  # 60 / 25 = 3 chunks (25, 25, 10).

    # Each spawn returns a valid array for the 25/25/10 record ids it was sent.
    def script_item_for_call(prompt: str) -> str:
        ids = [rec["record_id"] for rec in json.loads(prompt.split("RECORDS:\n", 1)[1])]
        return _envelope(json.dumps([_element(rid) for rid in ids]))

    class DynamicRunner(FakeRunner):
        def __call__(self, argv: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
            self.calls.append(
                bc_call := SpawnCall(argv=argv, input=kwargs.get("input"), env=kwargs.get("env"))
            )
            assert bc_call.input is not None
            return subprocess.CompletedProcess(
                argv, returncode=0, stdout=script_item_for_call(bc_call.input), stderr=""
            )

    runner = DynamicRunner([])
    backend = bc.ClaudeCliBackend(_settings(), runner=runner, sleeper=lambda _s: None)
    outcome = backend.enrich_batch(reqs)
    assert len(runner.calls) == 3
    # Each spawn carried at most 25 records.
    for call in runner.calls:
        assert call.input is not None
        sent = json.loads(call.input.split("RECORDS:\n", 1)[1])
        assert len(sent) <= 25
    assert len(outcome.results) == 60
    assert outcome.deferred == []


# --------------------------------------------------------------------------- #
# T4.6 — transient failure (non-zero exit / timeout) → defer transient.         #
# --------------------------------------------------------------------------- #


def test_nonzero_exit_retries_then_defers_transient() -> None:
    reqs = _reqs(2)
    # Always non-zero → exhausts the 2 retries (3 spawns) → defer transient.
    backend, runner = _backend([1])
    outcome = backend.enrich_batch(reqs)
    assert len(runner.calls) == 3  # 1 + 2 transient retries (§5.8).
    assert outcome.results == []
    assert {d.record_id for d in outcome.deferred} == {"r0", "r1"}
    assert all(d.reason == "transient" for d in outcome.deferred)
    assert outcome.spend == []  # no usable usage → no spend row.


def test_timeout_retries_then_defers_transient_no_real_sleep() -> None:
    reqs = _reqs(1)
    backend, runner = _backend([FakeRunner.TIMEOUT])
    outcome = backend.enrich_batch(reqs)
    assert len(runner.calls) == 3
    assert {d.reason for d in outcome.deferred} == {"transient"}


def test_exit0_envelope_is_error_defers_transient() -> None:
    # Belt-and-braces: exit 0 but `is_error:true` (a real API error the CLI ever
    # surfaces on a zero exit) must defer as transient, not parse as a result.
    reqs = _reqs(1)
    err_envelope = json.dumps(
        {"type": "result", "is_error": True, "result": "Not logged in"}
    )
    backend, runner = _backend([err_envelope])
    outcome = backend.enrich_batch(reqs)
    assert len(runner.calls) == 3  # transient retry budget exhausted.
    assert outcome.results == []
    assert {d.reason for d in outcome.deferred} == {"transient"}


def test_transient_then_success_returns_results() -> None:
    reqs = _reqs(1)
    array = json.dumps([_element("r0")])
    backend, runner = _backend([1, _envelope(array)])  # fail once, then succeed.
    outcome = backend.enrich_batch(reqs)
    assert len(runner.calls) == 2
    assert {r.record_id for r in outcome.results} == {"r0"}
    # One retry consumed → recorded on the spend row.
    assert outcome.spend[0].retry_count == 1


# --------------------------------------------------------------------------- #
# T4.6 — dedup_target_name validation.                                          #
# --------------------------------------------------------------------------- #


def test_dedup_target_out_of_set_coerced_to_new() -> None:
    cands = [Candidate(name="known-cand", summary="s", aliases=[], excerpt="e")]
    reqs = _reqs(1, candidates=cands)
    element = _element("r0", verdict="duplicate", target="hallucinated-name")
    backend, _ = _backend([_envelope(json.dumps([element]))])
    outcome = backend.enrich_batch(reqs)
    res = outcome.results[0]
    assert res.dedup_verdict == "new"
    assert res.dedup_target_name is None
    assert res.conflict_explanation is None


def test_dedup_target_in_set_preserved() -> None:
    cands = [Candidate(name="known-cand", summary="s", aliases=[], excerpt="e")]
    reqs = _reqs(1, candidates=cands)
    element = _element("r0", verdict="duplicate", target="known-cand")
    backend, _ = _backend([_envelope(json.dumps([element]))])
    outcome = backend.enrich_batch(reqs)
    res = outcome.results[0]
    assert res.dedup_verdict == "duplicate"
    assert res.dedup_target_name == "known-cand"


# --------------------------------------------------------------------------- #
# T4.6 — reflect.                                                               #
# --------------------------------------------------------------------------- #


def _reflect_req() -> ReflectRequest:
    return ReflectRequest(
        session_id="s1",
        activity=[
            ActivityRow(archive_id="b:1", role="user", kind="prompt", body="hi"),
            ActivityRow(archive_id="b:2", role="assistant", kind="text", body="ok"),
        ],
        active_record_ids=["a:redis", "a:port"],
    )


def test_reflect_parses_valid_arrays() -> None:
    obj = {
        "passive_hits": [{"record_id": "a:redis", "evidence": "cited the redis port"}],
        "promotion_candidates": [
            {"archive_id": "b:2", "proposed_summary": "ok was said", "rationale": "useful"}
        ],
    }
    backend, runner = _backend([_envelope(json.dumps(obj), in_tok=80, out_tok=40)])
    outcome = backend.reflect(_reflect_req())
    assert len(runner.calls) == 1
    assert [h.record_id for h in outcome.passive_hits] == ["a:redis"]
    assert [c.archive_id for c in outcome.promotion_candidates] == ["b:2"]
    assert outcome.spend[0].call_site == "reflect"
    assert outcome.spend[0].input_tokens == 80


def test_reflect_drops_out_of_set_ids() -> None:
    obj = {
        "passive_hits": [
            {"record_id": "a:redis", "evidence": "ok"},
            {"record_id": "a:NOT-A-RECORD", "evidence": "hallucinated"},
        ],
        "promotion_candidates": [
            {"archive_id": "b:999", "proposed_summary": "x", "rationale": "y"}
        ],
    }
    backend, _ = _backend([_envelope(json.dumps(obj))])
    outcome = backend.reflect(_reflect_req())
    assert [h.record_id for h in outcome.passive_hits] == ["a:redis"]
    assert outcome.promotion_candidates == []  # b:999 out of set, dropped.


def test_reflect_parse_failure_returns_empty_no_raise() -> None:
    backend, runner = _backend([_envelope("not json at all")])
    outcome = backend.reflect(_reflect_req())
    assert len(runner.calls) == 1  # reflect does not parse-repair-respawn.
    assert outcome.passive_hits == []
    assert outcome.promotion_candidates == []
    assert outcome.spend == []


def test_reflect_transient_failure_returns_empty_no_raise() -> None:
    backend, runner = _backend([1])
    outcome = backend.reflect(_reflect_req())
    assert len(runner.calls) == 3  # transient retry budget, then give up.
    assert outcome.passive_hits == []
    assert outcome.spend == []


# --------------------------------------------------------------------------- #
# Envelope robustness.                                                          #
# --------------------------------------------------------------------------- #


def test_envelope_without_result_field_is_transient() -> None:
    reqs = _reqs(1)
    bad_envelope = json.dumps({"type": "result", "usage": {}})  # no 'result'.
    backend, _ = _backend([bad_envelope])
    outcome = backend.enrich_batch(reqs)
    assert {d.reason for d in outcome.deferred} == {"transient"}


def test_non_json_envelope_is_transient() -> None:
    reqs = _reqs(1)
    backend, _ = _backend(["this is not the json envelope"])
    outcome = backend.enrich_batch(reqs)
    assert {d.reason for d in outcome.deferred} == {"transient"}
