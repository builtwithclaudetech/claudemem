"""Tests for ``claudemem.enrich.backend`` — the EnrichmentBackend boundary.

Covers T4.1 (transport-neutral dataclasses + the protocol) and T4.2
(``select_backend`` resolution matrix, lazy import, per-process cache,
LexicalOnlyBackend deferral, and the boundary-purity assertion that importing
the module pulls in no Anthropic SDK).

The concrete ``ClaudeCliBackend`` / ``AnthropicSdkBackend`` are built in
parallel tasks (T4.3-T4.6) and may not exist when this runs; the resolution
matrix monkeypatches the lazy-import helper ``_try_detect`` so the tests never
require a real key, the ``claude`` CLI, or those modules.
"""

from __future__ import annotations

import dataclasses
import logging
import subprocess
import sys
from typing import get_args

import pytest

from claudemem import config
from claudemem.enrich import backend as be


@pytest.fixture(autouse=True)
def _clear_backend_cache() -> None:
    """Reset the per-process selection cache before each test (T4.2 caching)."""
    be._reset_cache_for_tests()


# --------------------------------------------------------------------------- #
# T4.1 — dataclasses construct, are frozen, and carry the spec fields           #
# --------------------------------------------------------------------------- #


def test_candidate_constructs_with_name_summary_aliases_excerpt() -> None:
    cand = be.Candidate(
        name="redis-port", summary="Redis runs on 6380.", aliases=["redis"], excerpt="…"
    )
    assert cand.name == "redis-port"
    assert cand.aliases == ["redis"]


def test_enrich_request_carries_correlation_key_and_candidates() -> None:
    req = be.EnrichRequest(
        record_id="redis-port",
        name="redis-port",
        body="Redis listens on 6380 here.",
        candidates=[be.Candidate(name="a", summary="s", aliases=[], excerpt="e")],
    )
    assert req.record_id == "redis-port"
    assert len(req.candidates) == 1


def test_enrich_result_has_all_three_job_fields() -> None:
    res = be.EnrichResult(
        record_id="r1",
        summary="A short summary.",
        aliases=["alias-1"],
        dedup_verdict="new",
        dedup_target_name=None,
        conflict_explanation=None,
    )
    assert res.dedup_verdict == "new"
    assert res.dedup_target_name is None


@pytest.mark.parametrize(
    "dc, kwargs",
    [
        (be.Candidate, dict(name="n", summary="s", aliases=[], excerpt="e")),
        (
            be.EnrichRequest,
            dict(record_id="r", name="n", body="b", candidates=[]),
        ),
        (
            be.EnrichResult,
            dict(
                record_id="r",
                summary="s",
                aliases=["a"],
                dedup_verdict="new",
                dedup_target_name=None,
                conflict_explanation=None,
            ),
        ),
        (be.DeferralEntry, dict(record_id="r", reason="parse")),
        (be.SpendEntry, dict(call_site="save", model="haiku", backend="cli")),
        (be.BackendOutcome, dict()),
        (be.ReflectRequest, dict(session_id="s", activity=[], active_record_ids=[])),
        (be.PassiveHit, dict(record_id="r", evidence="e")),
        (
            be.PromotionCandidate,
            dict(archive_id="b:1", proposed_summary="s", rationale="r"),
        ),
        (be.ReflectOutcome, dict()),
        (be.ActivityRow, dict(archive_id="b:1", role="user", kind="prompt", body="hi")),
    ],
)
def test_value_dataclasses_are_frozen(dc: type, kwargs: dict) -> None:
    """Every transport-neutral value dataclass is frozen (tech-design §5.1)."""
    obj = dc(**kwargs)
    field_name = next(iter(dataclasses.fields(obj))).name
    with pytest.raises(dataclasses.FrozenInstanceError):
        setattr(obj, field_name, "mutated")


def test_backend_outcome_defaults_are_empty_lists() -> None:
    out = be.BackendOutcome()
    assert out.results == [] and out.deferred == [] and out.spend == []


def test_deferral_reason_literal_has_exactly_the_four_values() -> None:
    """``DeferralEntry.reason`` is the 4-value Literal (tech-design §5.1 MF-1)."""
    assert set(get_args(be.DeferralReason)) == {"parse", "cap", "auth", "transient"}
    # Each of the four values constructs cleanly (runtime smoke; type-level is mypy).
    for reason in ("parse", "cap", "auth", "transient"):
        assert be.DeferralEntry(record_id="r", reason=reason).reason == reason  # type: ignore[arg-type]


def test_dedup_verdict_literal_values() -> None:
    assert set(get_args(be.DedupVerdict)) == {"new", "duplicate", "conflict"}


def test_spend_entry_aligns_with_record_spend_fields() -> None:
    """SpendEntry carries the post-call usage the store needs (tech-design §5.1)."""
    entry = be.SpendEntry(
        call_site="save",
        model="haiku",
        backend="sdk",
        input_tokens=2350,
        output_tokens=150,
        idempotency_key="abc",
        latency_ms=420,
        retry_count=0,
        outcome="ok",
        record_id_int=7,
    )
    names = {f.name for f in dataclasses.fields(entry)}
    # The fields that map onto store.spend.record_spend_and_clear_pending.
    assert {
        "call_site",
        "model",
        "backend",
        "input_tokens",
        "output_tokens",
        "idempotency_key",
        "latency_ms",
        "retry_count",
        "outcome",
        "record_id_int",
    } <= names


def test_protocol_runtime_checkable_against_lexical_backend() -> None:
    """LexicalOnlyBackend satisfies the EnrichmentBackend protocol shape."""
    assert isinstance(be.LexicalOnlyBackend(), be.EnrichmentBackend)


# --------------------------------------------------------------------------- #
# T4.2 — LexicalOnlyBackend defers everything; reflect is empty; no model        #
# --------------------------------------------------------------------------- #


def test_lexical_backend_defers_all_records_with_no_spend() -> None:
    lex = be.LexicalOnlyBackend()
    reqs = [
        be.EnrichRequest(record_id=f"r{i}", name=f"n{i}", body="b", candidates=[])
        for i in range(3)
    ]
    out = lex.enrich_batch(reqs)
    assert out.results == []
    assert [d.record_id for d in out.deferred] == ["r0", "r1", "r2"]
    assert all(d.reason == "auth" for d in out.deferred)  # documented mapping
    assert out.spend == []  # no model call attempted → no SpendLog row


def test_lexical_backend_reflect_is_empty() -> None:
    lex = be.LexicalOnlyBackend()
    out = lex.reflect(be.ReflectRequest(session_id="s", activity=[], active_record_ids=[]))
    assert out.passive_hits == []
    assert out.promotion_candidates == []
    assert out.spend == []


def test_lexical_backend_detect_is_true() -> None:
    assert be.LexicalOnlyBackend.detect() is True


# --------------------------------------------------------------------------- #
# T4.2 — select_backend resolution matrix (monkeypatch the lazy detect helper)   #
# --------------------------------------------------------------------------- #


class _StubBackend:
    """A minimal stand-in concrete backend for resolution tests."""

    def __init__(self, tag: str) -> None:
        self.tag = tag

    @staticmethod
    def detect() -> bool:  # pragma: no cover - not exercised via the stub path
        return True

    def enrich_batch(self, reqs: list[be.EnrichRequest]) -> be.BackendOutcome:
        return be.BackendOutcome()

    def reflect(self, req: be.ReflectRequest) -> be.ReflectOutcome:
        return be.ReflectOutcome()


def _settings(backend: str) -> config.Settings:
    return config.load_config(overrides={"llm": {"backend": backend}})


def _patch_detect(
    monkeypatch: pytest.MonkeyPatch, *, cli: object | None, sdk: object | None
) -> None:
    """Patch the lazy-import probe so it returns the given backends per module."""

    def fake_try_detect(module_name: str, class_name: str) -> object | None:
        if module_name == be._CLI_MODULE:
            return cli
        if module_name == be._SDK_MODULE:
            return sdk
        return None

    monkeypatch.setattr(be, "_try_detect", fake_try_detect)


def test_auto_prefers_cli_when_available(monkeypatch: pytest.MonkeyPatch) -> None:
    cli = _StubBackend("cli")
    _patch_detect(monkeypatch, cli=cli, sdk=_StubBackend("sdk"))
    assert be.select_backend(_settings("auto")) is cli


def test_auto_falls_to_sdk_when_no_cli(monkeypatch: pytest.MonkeyPatch) -> None:
    sdk = _StubBackend("sdk")
    _patch_detect(monkeypatch, cli=None, sdk=sdk)
    assert be.select_backend(_settings("auto")) is sdk


def test_auto_falls_to_lexical_when_neither(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch_detect(monkeypatch, cli=None, sdk=None)
    assert isinstance(be.select_backend(_settings("auto")), be.LexicalOnlyBackend)


def test_forced_cli_unavailable_warns_once_and_falls_to_lexical(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_detect(monkeypatch, cli=None, sdk=_StubBackend("sdk"))
    with caplog.at_level(logging.WARNING, logger="claudemem"):
        resolved = be.select_backend(_settings("cli"))
    # Never raised, fell through to lexical, even though an SDK was available
    # (a forced backend does NOT silently pick a different transport).
    assert isinstance(resolved, be.LexicalOnlyBackend)
    warn_lines = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warn_lines) == 1
    assert "backend=cli" in warn_lines[0].getMessage()


def test_forced_cli_warns_only_once_across_calls(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    _patch_detect(monkeypatch, cli=None, sdk=None)
    with caplog.at_level(logging.WARNING, logger="claudemem"):
        be.select_backend(_settings("cli"))
        # Second call hits the per-process cache → no re-resolve, no second warn.
        be.select_backend(_settings("cli"))
    assert sum(r.levelno == logging.WARNING for r in caplog.records) == 1


def test_forced_sdk_unavailable_falls_to_lexical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_detect(monkeypatch, cli=_StubBackend("cli"), sdk=None)
    assert isinstance(be.select_backend(_settings("sdk")), be.LexicalOnlyBackend)


def test_none_is_always_lexical(monkeypatch: pytest.MonkeyPatch) -> None:
    # Even if both transports would detect, `none` is lexical-only.
    _patch_detect(monkeypatch, cli=_StubBackend("cli"), sdk=_StubBackend("sdk"))
    assert isinstance(be.select_backend(_settings("none")), be.LexicalOnlyBackend)


def test_unknown_backend_value_degrades_to_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli = _StubBackend("cli")
    _patch_detect(monkeypatch, cli=cli, sdk=None)
    # A stray/forward-compat value never errors; it is treated as `auto` (SC-3).
    assert be.select_backend(_settings("bogus")) is cli


def test_select_backend_is_cached_per_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """The resolved backend is cached so detect() runs at most once per name."""
    calls: list[str] = []

    def counting_try_detect(module_name: str, class_name: str) -> object | None:
        calls.append(module_name)
        return _StubBackend("cli") if module_name == be._CLI_MODULE else None

    monkeypatch.setattr(be, "_try_detect", counting_try_detect)
    first = be.select_backend(_settings("auto"))
    second = be.select_backend(_settings("auto"))
    assert first is second
    # Only the first resolve probed; the second was served from cache.
    assert calls == [be._CLI_MODULE]


def test_missing_concrete_module_falls_through_to_lexical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A not-yet-built concrete module is a clean ImportError → lexical (SC-3)."""

    def raise_import_error(name: str) -> object:
        raise ImportError(f"no module {name}")

    monkeypatch.setattr(be.importlib, "import_module", raise_import_error)
    assert isinstance(be.select_backend(_settings("auto")), be.LexicalOnlyBackend)


def test_try_detect_swallows_probe_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """A backend whose detect() raises is treated as unavailable, never errors."""

    class _Boom:
        @staticmethod
        def detect() -> bool:
            raise RuntimeError("probe blew up")

    import types

    fake_mod = types.ModuleType("claudemem.enrich.backend_cli")
    fake_mod.ClaudeCliBackend = _Boom  # type: ignore[attr-defined]
    monkeypatch.setattr(
        be.importlib,
        "import_module",
        lambda name: fake_mod if name == be._CLI_MODULE else raise_(ImportError(name)),
    )
    assert isinstance(be.select_backend(_settings("cli")), be.LexicalOnlyBackend)


def raise_(exc: BaseException) -> object:
    raise exc


# --------------------------------------------------------------------------- #
# Boundary purity (SC-6 / C-17) — importing backend.py imports NO anthropic      #
# --------------------------------------------------------------------------- #


def test_importing_backend_does_not_import_anthropic_in_this_process() -> None:
    """In this already-loaded process, `anthropic` is absent (funneled to SDK mod)."""
    assert "anthropic" not in sys.modules


def test_fresh_import_of_backend_does_not_import_anthropic() -> None:
    """A clean interpreter importing only backend.py never imports the SDK (C-17).

    This is the load-bearing boundary assertion: the SDK import is funneled into
    backend_sdk.py (function-local), so importing backend.py — and constructing
    every dataclass + LexicalOnlyBackend — touches neither `anthropic` nor the
    CLI spawn machinery.
    """
    code = (
        "import sys\n"
        "import claudemem.enrich.backend as be\n"
        "be.LexicalOnlyBackend().enrich_batch([])\n"
        "assert 'anthropic' not in sys.modules, sorted(m for m in sys.modules if 'anthropic' in m)\n"
        "print('OK')\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "OK"
