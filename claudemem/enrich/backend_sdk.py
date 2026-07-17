"""claudemem.enrich.backend_sdk — the single funneled Anthropic SDK backend (L3).

This module is the **one and only importer of the ``anthropic`` SDK** in the
codebase (tech-design §6.1, architecture §2.6, C-17). The import is **function-
local, wrapped in ``try/except ImportError``** — there is deliberately NO
module-level ``import anthropic`` here. Two consequences fall out of that rule:

* Merely importing ``claudemem.enrich.backend_sdk`` does **not** put ``anthropic``
  in ``sys.modules`` — the SDK is only touched the first time a method that needs
  a client runs. The firewall test asserts this.
* An adopter who installs ClaudeMem **without** the ``[llm]`` extra (no
  ``anthropic`` on the path) never sees an import error at module load; instead
  the first ``enrich_batch`` / ``reflect`` call degrades cleanly to a deferral
  (``ImportError`` → defer every record, ``reason="auth"``) rather than raising
  (SC-3 / NG-5).

It implements the :class:`~claudemem.enrich.backend.EnrichmentBackend` protocol
via **forced tool-use** (tech-design §5.2 ``record_memory_analysis`` /
§5.3 ``session_reflection``), with **bounded retry + idempotency** (§5.8) and
**no prompt caching** (§2.3 — Haiku 4.5's 4,096-token cache prefix exceeds the
~1,050-token system prompt, so a ``cache_control`` breakpoint would be a no-op;
cache token classes are always 0).

Per IN-13, ``enrich_batch`` makes **exactly one model call per record** (it
iterates the batch length-1 inline). The K dedup candidates are laid in the
**user channel only**, never in the tool schema (§5.2).
"""

from __future__ import annotations

import hashlib
import logging
import os
import random
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from claudemem import config
from claudemem.enrich.backend import (
    BackendOutcome,
    Candidate,
    DeferralEntry,
    EnrichRequest,
    EnrichResult,
    PassiveHit,
    PromotionCandidate,
    ReflectOutcome,
    ReflectRequest,
    SpendEntry,
)

if TYPE_CHECKING:
    # Type-checking-only reference. ``TYPE_CHECKING`` is False at runtime, so this
    # never executes an ``import anthropic`` — it exists purely so mypy can name
    # the client type without violating the C-17 funneled-import rule.
    from anthropic import Anthropic

_log = logging.getLogger("claudemem")

# --------------------------------------------------------------------------- #
# Constants                                                                     #
# --------------------------------------------------------------------------- #

#: The Anthropic API-key environment variable the SDK reads (§7.3 / detect()).
_API_KEY_ENV = "ANTHROPIC_API_KEY"

#: Model pin (tech-design §5.4). ``[llm].model`` defaults to the ``"haiku"``
#: alias (config §9); we map that alias to this pinned id. Any other configured
#: value is passed through verbatim so the pin stays config-overridable.
_HAIKU_PIN = "claude-haiku-4-5-20251001"
_HAIKU_ALIASES = frozenset({"haiku", "claude-haiku", "claude-3-5-haiku"})

#: Schema version baked into the idempotency key (§5.8). Bump when either tool
#: schema below changes shape so a re-run after a schema change is not silently
#: deduped against the old call.
_SCHEMA_VER = "1"

#: Output token caps (Rule: never default to "unlimited"). Enrichment output is
#: ~150 tokens (§5.5); reflection is a short list of hits/candidates. Generous
#: but bounded so a runaway generation cannot bill open-endedly.
_ENRICH_MAX_TOKENS = 1024
_REFLECT_MAX_TOKENS = 2048

#: Retry policy (tech-design §5.8, SDK column): 3 retries, base 0.5 s, ×2,
#: 8 s cap, full jitter.
_MAX_RETRIES = 3
_BACKOFF_BASE_S = 0.5
_BACKOFF_FACTOR = 2.0
_BACKOFF_CAP_S = 8.0

#: Tool names (§5.2 / §5.3).
_ENRICH_TOOL = "record_memory_analysis"
_REFLECT_TOOL = "session_reflection"

# --------------------------------------------------------------------------- #
# Forced tool-use schemas — reproduced EXACTLY from tech-design §5.2 / §5.3     #
# --------------------------------------------------------------------------- #

#: §5.2 ``record_memory_analysis``. All fields required regardless of verdict
#: (IN-13); ``dedup_target_name`` / ``conflict_explanation`` nullable-but-present.
#: Dedup candidates are NOT in this schema — they go in the user channel (§5.2).
_ENRICH_SCHEMA: dict[str, Any] = {
    "name": _ENRICH_TOOL,
    "description": "Enrich, dedup, and contradiction-check a single memory record.",
    "input_schema": {
        "type": "object",
        "properties": {
            "summary": {"type": "string"},
            "aliases": {
                "type": "array",
                "items": {"type": "string"},
                "minItems": 1,
                "maxItems": 8,
            },
            "dedup_verdict": {
                "type": "string",
                "enum": ["new", "duplicate", "conflict"],
            },
            "dedup_target_name": {"type": ["string", "null"]},
            "conflict_explanation": {"type": ["string", "null"]},
        },
        "required": [
            "summary",
            "aliases",
            "dedup_verdict",
            "dedup_target_name",
            "conflict_explanation",
        ],
    },
}

#: §5.3 ``session_reflection``. All fields required; empty arrays allowed.
_REFLECT_SCHEMA: dict[str, Any] = {
    "name": _REFLECT_TOOL,
    "description": (
        "Identify passive hits to reinforce and propose Fork B->A "
        "promotion candidates."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "passive_hits": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "record_id": {"type": "string"},
                        "evidence": {"type": "string"},
                    },
                    "required": ["record_id", "evidence"],
                },
            },
            "promotion_candidates": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "archive_id": {"type": "string"},
                        "proposed_summary": {"type": "string"},
                        "rationale": {"type": "string"},
                    },
                    "required": ["archive_id", "proposed_summary", "rationale"],
                },
            },
        },
        "required": ["passive_hits", "promotion_candidates"],
    },
}

# --------------------------------------------------------------------------- #
# System prompts (concise; no caching v1 — §2.3)                                #
# --------------------------------------------------------------------------- #

_ENRICH_SYSTEM = (
    "You enrich a single durable memory record for a developer's personal "
    "memory system. Call the record_memory_analysis tool exactly once.\n"
    "- summary: one tight factual sentence capturing the record.\n"
    "- aliases: 1-8 short alternate names/keywords someone might search by.\n"
    "- dedup_verdict: 'duplicate' if an existing candidate already states the "
    "same fact, 'conflict' if a candidate states a contradicting fact, else "
    "'new'.\n"
    "- dedup_target_name: the EXACT name of the matching candidate for "
    "duplicate/conflict, else null. Use only a name from the supplied "
    "candidate list.\n"
    "- conflict_explanation: one sentence on the contradiction for 'conflict', "
    "else null."
)

_REFLECT_SYSTEM = (
    "You review one Claude Code session's bounded activity log against the "
    "developer's existing memory records. Call the session_reflection tool "
    "exactly once.\n"
    "- passive_hits: existing record_ids the session evidently relied on, each "
    "with brief evidence. Use only record_ids from the supplied list.\n"
    "- promotion_candidates: archive_ids from the log worth promoting to a "
    "durable record, each with a proposed_summary and rationale. Use only "
    "archive_ids from the supplied log.\n"
    "Both arrays may be empty. Propose only; never assume anything is applied."
)


# --------------------------------------------------------------------------- #
# Errors raised internally to drive the retry/defer matrix                      #
# --------------------------------------------------------------------------- #


class _AuthUnavailable(Exception):
    """Auth/import failure — NOT retried; the record defers with reason='auth'."""


class _Transient(Exception):
    """A 429/5xx/timeout/connection error — retried within budget (§5.8)."""


# --------------------------------------------------------------------------- #
# Normalization + idempotency (tech-design §5.8)                                #
# --------------------------------------------------------------------------- #


def _norm_body(body: str) -> str:
    """Normalize a record body for the idempotency key (§5.8).

    Rule: strip leading/trailing whitespace, then collapse every internal run of
    whitespace (spaces, tabs, newlines) to a single space. So bodies that differ
    only in incidental whitespace/formatting hash to the same key — a retry of
    the "same" save is deduped — while a genuine content edit produces a
    different key.
    """
    return " ".join(body.split())


def _idempotency_key(call_site: str, name: str, body: str) -> str:
    """Compute ``sha256("call_site|name|norm_body|schema_ver")`` (§5.8).

    Recorded on the :class:`SpendEntry`; the ``store.spend`` UNIQUE index turns a
    retry-after-network-blip into a clean no-op rather than a double-bill.
    """
    payload = f"{call_site}|{name}|{_norm_body(body)}|{_SCHEMA_VER}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------- #
# Error classification (no anthropic import — inspect duck-typed attributes)    #
# --------------------------------------------------------------------------- #


def _classify(exc: BaseException) -> type[_AuthUnavailable] | type[_Transient]:
    """Map an SDK exception to auth (no-retry) vs transient (retry) (§5.8).

    Deliberately does NOT import ``anthropic`` to inspect its exception classes
    (C-17). Instead it inspects the duck-typed ``status_code`` the SDK attaches
    to ``APIStatusError`` subclasses, and the class-name lineage for the
    no-status connection/timeout errors:

    * ``status_code`` 401 / 403 → **auth** (never retried).
    * ``status_code`` 429 or any 5xx → **transient** (retried within budget).
    * a connection / timeout error (``APIConnectionError`` / ``APITimeoutError``
      lineage, which carry no ``status_code``) → **transient**.
    * anything else → **transient** (conservative: a retry is cheap and the
      budget is bounded; a genuinely fatal error exhausts the budget and defers).
    """
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        if status in (401, 403):
            return _AuthUnavailable
        return _Transient
    return _Transient


# --------------------------------------------------------------------------- #
# The backend                                                                   #
# --------------------------------------------------------------------------- #


class AnthropicSdkBackend:
    """Forced-tool-use Anthropic SDK enrichment backend (tech-design §5.2/§5.3).

    The ``anthropic`` import is funneled function-locally (C-17) through
    :meth:`_client`. For tests, a ``client_factory`` may be injected so no real
    ``anthropic.Anthropic()`` is constructed and no network call happens; a
    ``sleep`` callable is injectable so backoff is instantaneous under test.
    """

    def __init__(
        self,
        *,
        settings: config.Settings | None = None,
        client_factory: Callable[[], Anthropic] | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings if settings is not None else config.load_config()
        self._client_factory = client_factory
        self._sleep = sleep
        self._client: Anthropic | None = None
        self._auth_dead = False  # cache an auth/import failure for the process.

    # ----------------------------------------------------------------- #
    # Availability probe (write-path only, §5.9)                          #
    # ----------------------------------------------------------------- #

    @staticmethod
    def detect() -> bool:
        """True iff a key is present AND ``anthropic`` is importable (§5.9).

        The ONLY place (besides ``backend_cli`` for the claude-spawn path) the
        SDK availability is probed. The import is function-local + guarded so a
        missing ``[llm]`` extra is a clean ``False``, never an ImportError at
        module load (C-17 / SC-3).
        """
        if not os.environ.get(_API_KEY_ENV):
            return False
        try:
            import anthropic  # noqa: F401  (function-local funneled import, C-17)
        except ImportError:
            return False
        return True

    # ----------------------------------------------------------------- #
    # Funneled client construction (C-17)                                 #
    # ----------------------------------------------------------------- #

    def _model(self) -> str:
        configured = self._settings.llm.model
        return _HAIKU_PIN if configured in _HAIKU_ALIASES else configured

    def _client_or_raise(self) -> Anthropic:
        """Lazily build (and cache) the Anthropic client; the funneled import.

        Raises :class:`_AuthUnavailable` on ImportError (missing ``[llm]`` extra)
        or a missing key — both mean "this transport cannot run", which defers
        with ``reason="auth"`` and never raises out of a public method (SC-3).
        """
        if self._client is not None:
            return self._client
        if self._client_factory is not None:
            self._client = self._client_factory()
            return self._client
        try:
            import anthropic  # noqa: PLC0415  (function-local funneled import, C-17)
        except ImportError as exc:
            raise _AuthUnavailable("anthropic SDK not installed") from exc
        if not os.environ.get(_API_KEY_ENV):
            raise _AuthUnavailable("ANTHROPIC_API_KEY not set")
        self._client = anthropic.Anthropic()
        return self._client

    # ----------------------------------------------------------------- #
    # The retrying call core (§5.8)                                       #
    # ----------------------------------------------------------------- #

    def _call_with_retry(
        self,
        *,
        system: str,
        user_text: str,
        tool: dict[str, Any],
        max_tokens: int,
        idempotency_key: str,
    ) -> tuple[dict[str, Any], int, int, int]:
        """Make one forced-tool call, retrying transient failures (§5.8).

        Returns ``(tool_input, retry_count, input_tokens, output_tokens)`` on
        success. Raises :class:`_AuthUnavailable` (no retry) or :class:`_Transient`
        (budget exhausted) for the caller to map to a deferral.
        """
        client = self._client_or_raise()
        attempt = 0
        while True:
            # Build kwargs as a dynamically-typed dict and splat it: the SDK's
            # strict ``create`` overloads expect ``*Param`` TypedDicts, but the
            # forced-tool schema / tool_choice are plain dicts here, so a direct
            # keyword call trips the overload matcher. The wire shape is correct
            # (verified by the SDK at runtime); the dict-splat keeps mypy strict
            # over the rest of the module.
            create_kwargs: dict[str, Any] = {
                "model": self._model(),
                "max_tokens": max_tokens,
                "system": system,
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": tool["name"]},
                "messages": [{"role": "user", "content": user_text}],
                "extra_headers": {"Idempotency-Key": idempotency_key},
            }
            try:
                resp = client.messages.create(**create_kwargs)
                # Extracting the forced tool_use block is inside the retry try so
                # a malformed / tool-block-missing response (model refused or
                # returned text despite tool_choice) is retried as transient
                # rather than crashing the save (SC-3).
                tool_input = _extract_tool_input(resp, tool["name"])
            except _Transient as exc:
                if attempt >= _MAX_RETRIES:
                    raise
                self._backoff(attempt)
                attempt += 1
                _log.debug("claudemem sdk: transient (malformed response), retrying: %s", exc)
                continue
            except Exception as exc:  # noqa: BLE001 — classified below; never escapes raw.
                kind = _classify(exc)
                if kind is _AuthUnavailable:
                    raise _AuthUnavailable(str(exc)) from exc
                if attempt >= _MAX_RETRIES:
                    raise _Transient(str(exc)) from exc
                self._backoff(attempt)
                attempt += 1
                continue
            in_tok, out_tok = _usage(resp)
            return tool_input, attempt, in_tok, out_tok

    def _backoff(self, attempt: int) -> None:
        """Sleep base×factor^attempt, capped, with full jitter (§5.8)."""
        ceiling = min(_BACKOFF_CAP_S, _BACKOFF_BASE_S * (_BACKOFF_FACTOR**attempt))
        self._sleep(random.uniform(0.0, ceiling))  # noqa: S311 — jitter, not crypto.

    # ----------------------------------------------------------------- #
    # enrich_batch (T4.3, IN-13 — one model call per record)              #
    # ----------------------------------------------------------------- #

    def enrich_batch(self, reqs: list[EnrichRequest]) -> BackendOutcome:
        """Enrich + dedup + contradiction-check, one model call per record.

        Never raises (SC-3): an ImportError / missing key / exhausted-retry
        failure becomes a :class:`DeferralEntry`. Once an auth failure is seen it
        is cached for the process so the remaining records short-circuit to an
        ``auth`` defer without re-attempting.
        """
        results: list[EnrichResult] = []
        deferred: list[DeferralEntry] = []
        spend: list[SpendEntry] = []
        model = self._model()

        for req in reqs:
            if self._auth_dead:
                deferred.append(DeferralEntry(record_id=req.record_id, reason="auth"))
                continue

            idem = _idempotency_key("save", req.name, req.body)
            user_text = _enrich_user_message(req)
            t0 = time.monotonic()
            try:
                tool_input, retries, in_tok, out_tok = self._call_with_retry(
                    system=_ENRICH_SYSTEM,
                    user_text=user_text,
                    tool=_ENRICH_SCHEMA,
                    max_tokens=_ENRICH_MAX_TOKENS,
                    idempotency_key=idem,
                )
            except _AuthUnavailable:
                self._auth_dead = True
                deferred.append(DeferralEntry(record_id=req.record_id, reason="auth"))
                continue
            except _Transient:
                latency_ms = int((time.monotonic() - t0) * 1000)
                deferred.append(
                    DeferralEntry(record_id=req.record_id, reason="transient")
                )
                spend.append(
                    SpendEntry(
                        call_site="save",
                        model=model,
                        backend="sdk",
                        idempotency_key=idem,
                        latency_ms=latency_ms,
                        retry_count=_MAX_RETRIES,
                        outcome="deferred",
                    )
                )
                continue

            latency_ms = int((time.monotonic() - t0) * 1000)
            results.append(_parse_enrich(req, tool_input))
            spend.append(
                SpendEntry(
                    call_site="save",
                    model=model,
                    backend="sdk",
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    idempotency_key=idem,
                    latency_ms=latency_ms,
                    retry_count=retries,
                    outcome="ok",
                )
            )

        return BackendOutcome(results=results, deferred=deferred, spend=spend)

    # ----------------------------------------------------------------- #
    # reflect (T4.4, IN-14)                                               #
    # ----------------------------------------------------------------- #

    def reflect(self, req: ReflectRequest) -> ReflectOutcome:
        """One forced ``session_reflection`` call over bounded Fork B rows (§5.3).

        Validates every ``record_id`` / ``archive_id`` against the supplied log;
        out-of-set ids are dropped. Propose-only. Never raises (SC-3): an auth /
        exhausted-retry failure returns an empty outcome (reindex is the backstop,
        SC-9), with a ``deferred``-outcome SpendEntry only where a call was
        actually attempted.
        """
        model = self._model()
        # Idempotency key over the session's normalized activity (the "body" here
        # is the concatenated bounded log) keyed by session_id as the name.
        body = "\n".join(f"{r.archive_id}\t{r.role}\t{r.kind}\t{r.body}" for r in req.activity)
        idem = _idempotency_key("reflect", req.session_id, body)
        user_text = _reflect_user_message(req)

        t0 = time.monotonic()
        try:
            tool_input, retries, in_tok, out_tok = self._call_with_retry(
                system=_REFLECT_SYSTEM,
                user_text=user_text,
                tool=_REFLECT_SCHEMA,
                max_tokens=_REFLECT_MAX_TOKENS,
                idempotency_key=idem,
            )
        except _AuthUnavailable:
            self._auth_dead = True
            return ReflectOutcome()
        except _Transient:
            latency_ms = int((time.monotonic() - t0) * 1000)
            return ReflectOutcome(
                spend=[
                    SpendEntry(
                        call_site="reflect",
                        model=model,
                        backend="sdk",
                        idempotency_key=idem,
                        latency_ms=latency_ms,
                        retry_count=_MAX_RETRIES,
                        outcome="deferred",
                    )
                ]
            )

        latency_ms = int((time.monotonic() - t0) * 1000)
        hits, candidates = _parse_reflect(req, tool_input)
        return ReflectOutcome(
            passive_hits=hits,
            promotion_candidates=candidates,
            spend=[
                SpendEntry(
                    call_site="reflect",
                    model=model,
                    backend="sdk",
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    idempotency_key=idem,
                    latency_ms=latency_ms,
                    retry_count=retries,
                    outcome="ok",
                )
            ],
        )


# --------------------------------------------------------------------------- #
# Response parsing (module-level pure helpers)                                  #
# --------------------------------------------------------------------------- #


def _extract_tool_input(resp: Any, tool_name: str) -> dict[str, Any]:
    """Pull the forced ``tool_use`` block's ``input`` dict out of a response.

    With ``tool_choice`` forcing a specific tool, the model returns a single
    ``tool_use`` content block. A response missing it (the model refused or
    returned text) is treated as a transient failure so the retry/defer matrix
    handles it rather than crashing the save (SC-3).
    """
    for block in getattr(resp, "content", []) or []:
        if getattr(block, "type", None) == "tool_use" and getattr(block, "name", None) == tool_name:
            tool_input = getattr(block, "input", None)
            if isinstance(tool_input, dict):
                return tool_input
    raise _Transient(f"forced tool_use block '{tool_name}' missing from response")


def _usage(resp: Any) -> tuple[int, int]:
    """Read ``(input_tokens, output_tokens)`` from a response, 0 if absent."""
    usage = getattr(resp, "usage", None)
    in_tok = int(getattr(usage, "input_tokens", 0) or 0)
    out_tok = int(getattr(usage, "output_tokens", 0) or 0)
    return in_tok, out_tok


def _coerce_aliases(raw: Any) -> list[str]:
    """Coerce the tool's ``aliases`` to a 1..8 list of non-empty strings (§5.2).

    The schema declares ``minItems: 1, maxItems: 8`` but a model can still emit
    a non-conforming array; we defensively keep only string entries, clamp to 8,
    and guarantee at least one entry so :class:`EnrichResult` never violates its
    documented 1..8 invariant.
    """
    items: list[str] = []
    if isinstance(raw, list):
        for entry in raw:
            if isinstance(entry, str) and entry.strip():
                items.append(entry)
    if not items:
        items = ["(unaliased)"]
    return items[:8]


def _parse_enrich(req: EnrichRequest, tool_input: dict[str, Any]) -> EnrichResult:
    """Parse a ``record_memory_analysis`` tool input into an EnrichResult (§5.2).

    Defensive: ``dedup_target_name`` must be ∈ the request's candidate names; an
    out-of-set name coerces the verdict to ``"new"`` with target ``None`` (the
    model occasionally hallucinates a target).
    """
    summary = tool_input.get("summary")
    summary = summary if isinstance(summary, str) else ""
    aliases = _coerce_aliases(tool_input.get("aliases"))

    verdict = tool_input.get("dedup_verdict")
    if verdict not in ("new", "duplicate", "conflict"):
        verdict = "new"

    target = tool_input.get("dedup_target_name")
    target = target if isinstance(target, str) else None
    conflict = tool_input.get("conflict_explanation")
    conflict = conflict if isinstance(conflict, str) else None

    candidate_names = {c.name for c in req.candidates}
    if verdict in ("duplicate", "conflict"):
        if target is None or target not in candidate_names:
            # Out-of-set / missing target → defensive coercion to 'new' (§5.2).
            verdict = "new"
            target = None
            conflict = None
    else:
        target = None
        conflict = None

    return EnrichResult(
        record_id=req.record_id,
        summary=summary,
        aliases=aliases,
        dedup_verdict=verdict,  # type: ignore[arg-type]  # narrowed to the 3 literals above.
        dedup_target_name=target,
        conflict_explanation=conflict,
    )


def _parse_reflect(
    req: ReflectRequest, tool_input: dict[str, Any]
) -> tuple[list[PassiveHit], list[PromotionCandidate]]:
    """Parse ``session_reflection`` output, dropping out-of-set ids (§5.3).

    ``passive_hits[].record_id`` is validated against ``req.active_record_ids``;
    ``promotion_candidates[].archive_id`` against the supplied activity log. Empty
    arrays are valid. Propose-only.
    """
    valid_record_ids = set(req.active_record_ids)
    valid_archive_ids = {row.archive_id for row in req.activity}

    hits: list[PassiveHit] = []
    raw_hits = tool_input.get("passive_hits")
    if isinstance(raw_hits, list):
        for item in raw_hits:
            if not isinstance(item, dict):
                continue
            rid = item.get("record_id")
            evidence = item.get("evidence")
            if isinstance(rid, str) and rid in valid_record_ids and isinstance(evidence, str):
                hits.append(PassiveHit(record_id=rid, evidence=evidence))

    candidates: list[PromotionCandidate] = []
    raw_cands = tool_input.get("promotion_candidates")
    if isinstance(raw_cands, list):
        for item in raw_cands:
            if not isinstance(item, dict):
                continue
            aid = item.get("archive_id")
            proposed = item.get("proposed_summary")
            rationale = item.get("rationale")
            if (
                isinstance(aid, str)
                and aid in valid_archive_ids
                and isinstance(proposed, str)
                and isinstance(rationale, str)
            ):
                candidates.append(
                    PromotionCandidate(
                        archive_id=aid, proposed_summary=proposed, rationale=rationale
                    )
                )

    return hits, candidates


# --------------------------------------------------------------------------- #
# User-channel message builders (§5.2: candidates in the USER channel only)     #
# --------------------------------------------------------------------------- #


def _format_candidate(cand: Candidate) -> str:
    aliases = ", ".join(cand.aliases) if cand.aliases else "(none)"
    return (
        f"- name: {cand.name}\n"
        f"  summary: {cand.summary}\n"
        f"  aliases: {aliases}\n"
        f"  excerpt: {cand.excerpt}"
    )


def _enrich_user_message(req: EnrichRequest) -> str:
    """Build the user message: the new record body + the K dedup candidates.

    The candidates live here (user channel) and NOT in the tool schema (§5.2),
    so user/record content is never substituted into the system prompt (no
    prompt-injection trust of record bodies).
    """
    if req.candidates:
        cand_block = "\n".join(_format_candidate(c) for c in req.candidates)
    else:
        cand_block = "(no existing candidates)"
    return (
        f"NEW RECORD\nname: {req.name}\nbody:\n{req.body}\n\n"
        f"EXISTING DEDUP CANDIDATES (compare against these only):\n{cand_block}"
    )


def _reflect_user_message(req: ReflectRequest) -> str:
    """Build the reflection user message from the bounded Fork B rows (§5.3)."""
    if req.activity:
        rows = "\n".join(
            f"- archive_id: {r.archive_id}\n  role: {r.role}\n  kind: {r.kind}\n  body: {r.body}"
            for r in req.activity
        )
    else:
        rows = "(empty session log)"
    if req.active_record_ids:
        ids = ", ".join(req.active_record_ids)
    else:
        ids = "(none)"
    return (
        f"SESSION ACTIVITY LOG (Fork B rows):\n{rows}\n\n"
        f"EXISTING ACTIVE RECORD IDS (valid passive-hit targets):\n{ids}"
    )


__all__ = ["AnthropicSdkBackend"]
