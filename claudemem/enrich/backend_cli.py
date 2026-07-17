"""claudemem.enrich.backend_cli — the ``ClaudeCliBackend`` transport (L3).

The **preferred** enrichment transport (tech-design §5.9 ``auto`` order): spawn
the headless ``claude -p`` CLI so enrichment runs under the user's subscription with
**no Python dependency** (a ``claude`` binary on ``PATH`` is enough; §6.4). This
module — together with ``backend_sdk.py`` — is the **only spawner of ``claude``**
(C-17, §6.1): both the enrichment spawn *and* the ``claude auth status``
availability probe (§7.3) live here so "only spawner of ``claude``" stays
literally true. It is reached **lazily** by ``backend.select_backend`` and is
**write-path only** (read paths never import it; the import-linter firewall
contract forbids ``recall -> enrich``).

**The load-bearing invariant — recursion guard (§7.1 / §6.3, MF-2).** Every
``claude`` spawn from this module (enrichment, reflection, AND the auth probe)
sets ``CLAUDEMEM_DISABLE_HOOKS=1`` via an **environment MERGE**:
``env = {**os.environ, "CLAUDEMEM_DISABLE_HOOKS": "1"}`` — never a bare dict. A
bare ``env={"CLAUDEMEM_DISABLE_HOOKS": "1"}`` would wipe ``PATH``/``HOME`` and
the child could not even locate ``claude``. The flag makes the spawned session's
own ClaudeMem hooks no-op (they exit 0 on this env var, §6.3), which bounds
recursion at depth 1 and prevents Fork B pollution — this env merge is the
**primary** guard. ``--max-turns 1`` and ``--no-session-persistence`` remain as
secondary bounds. ``--bare`` was **removed**: it forces API-key-only auth
(strictly ``ANTHROPIC_API_KEY`` / ``apiKeyHelper``, never OAuth/keychain), which
breaks subscription/OAuth enrichment — the design's intended billing path.

**Never raises (SC-3).** A non-zero exit, timeout, rate-limit, or unparseable
result is converted to a **deferral** (``DeferralEntry`` with a triageable
``reason``) — never an exception that could error a ``save``. Records that can't
be enriched defer to the next ``reindex``.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import time
from collections.abc import Callable
from typing import Any

from claudemem import config
from claudemem.enrich.backend import (
    BackendOutcome,
    DedupVerdict,
    DeferralEntry,
    EnrichRequest,
    EnrichResult,
    PassiveHit,
    PromotionCandidate,
    ReflectOutcome,
    ReflectRequest,
    SpendEntry,
)

_log = logging.getLogger("claudemem")

#: The recursion-guard env var (§7.1 / §6.3). Set to ``"1"`` in every ``claude``
#: spawn this module makes so the spawned session's hooks no-op.
_DISABLE_HOOKS_ENV = "CLAUDEMEM_DISABLE_HOOKS"

#: Transient-retry policy for a chunk spawn (§5.8 CLI path: 2 retries, base 1 s
#: ×2, ~10 s cap). The cap bounds the worst-case backoff well under the 600 s
#: SessionEnd budget.
_TRANSIENT_RETRIES = 2
_BACKOFF_BASE_S = 1.0
_BACKOFF_CAP_S = 10.0

#: Outermost ``[...]`` JSON-array extractor for defensive parse (§5.6): greedy so
#: it spans the entire array even when the model wraps it in prose / code fences.
_ARRAY_RE = re.compile(r"\[.*\]", re.DOTALL)

#: Code-fence stripper (``` and ```json) for defensive parse (§5.6).
_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class _ChunkParseError(Exception):
    """Internal: the chunk result text yielded no usable JSON array (§5.6).

    Raised by the defensive parser when no outermost ``[...]`` is extractable or
    ``json.loads`` fails / the top level is not a list. Never escapes this
    module — it is caught and converted to a chunk-level re-spawn + defer.
    """


class _SpawnTransientError(Exception):
    """Internal: a spawn failed transiently (non-zero exit / timeout, §7.2).

    Never escapes this module — the caller converts it to a ``transient``
    deferral after the §5.8 retry budget is exhausted.
    """


class ClaudeCliBackend:
    """Spawn ``claude -p`` for enrichment under the subscription (§5.6 / §5.7).

    Injectable seams keep every test offline (no real spawn, no real sleep):

    * ``runner`` — stands in for :func:`subprocess.run`. Tests pass a fake that
      returns a canned :class:`subprocess.CompletedProcess` (or raises
      :class:`subprocess.TimeoutExpired`) and captures the ``env=`` / ``input=``
      / argv the production code passed.
    * ``sleeper`` — stands in for :func:`time.sleep` so the transient backoff
      consumes no wall-clock in tests.
    """

    def __init__(
        self,
        settings: config.Settings | None = None,
        *,
        runner: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._settings = settings if settings is not None else config.load_config()
        self._runner = runner
        self._sleeper = sleeper

    # ------------------------------------------------------------------ #
    # Availability probe (§7.3) — write-path only, cached per-process.    #
    # ------------------------------------------------------------------ #

    #: Per-process detection cache (NG-1: process-lifetime only, never
    #: persisted). ``None`` = not yet probed this process. Shared at the class
    #: level because ``detect()`` is a staticmethod with no instance.
    _detect_cache: bool | None = None

    @staticmethod
    def detect() -> bool:
        """Is an authenticated ``claude`` CLI available? (§7.3, write-path only).

        Two checks, cached per-process (NG-1): ``shutil.which("claude")`` is not
        ``None`` **and** ``claude auth status --json`` exits 0 (0 = authed, 1 =
        not, §1.3). The auth-status spawn ALSO uses the recursion-guard env merge
        (it is a ``claude`` spawn — keeps the "only spawner" invariant true and
        stops the probe itself from firing hooks). Any unexpected failure
        degrades to ``False`` so a probe never errors a save (SC-3).
        """
        if ClaudeCliBackend._detect_cache is not None:
            return ClaudeCliBackend._detect_cache
        available = ClaudeCliBackend._probe()
        ClaudeCliBackend._detect_cache = available
        return available

    @staticmethod
    def _probe() -> bool:
        """Run the uncached availability probe (factored out for the cache)."""
        if shutil.which("claude") is None:
            return False
        try:
            proc = subprocess.run(
                ["claude", "auth", "status", "--json"],
                env={**os.environ, _DISABLE_HOOKS_ENV: "1"},  # MERGE (MF-2).
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            # which() said claude exists but the spawn blew up: not available.
            return False
        return proc.returncode == 0

    @staticmethod
    def _reset_detect_cache_for_tests() -> None:
        """Clear the per-process detection cache (test hook only)."""
        ClaudeCliBackend._detect_cache = None

    # ------------------------------------------------------------------ #
    # Spawn helper (§7.1 / §7.2) — the single point that builds argv +    #
    # the recursion-guard env merge and invokes the runner.               #
    # ------------------------------------------------------------------ #

    def _spawn(self, prompt: str, *, max_output_tokens: int, timeout: int) -> tuple[str, int, int]:
        """Spawn ``claude -p``, returning ``(result_text, input_tokens, output_tokens)``.

        Prompt goes via **stdin** (``input=prompt``) — never argv (§7.2;
        prevents argv-length limits and keeps the prompt out of process tables).
        The env is the recursion-guard MERGE (§7.1, MF-2). A non-zero exit or a
        :class:`subprocess.TimeoutExpired` raises :class:`_SpawnTransientError`
        (the caller defers as ``transient``). The ``--output-format json``
        envelope is parsed for the ``result`` text and the ``usage`` block;
        ``usage`` tokens default to 0 when absent.

        ``max_output_tokens`` is passed via ``--max-output-tokens`` if the CLI
        accepts it. ASSUMPTION (noted to the caller): the public ``claude -p``
        surface does not document a per-call output-token flag, so the cap is
        primarily enforced by the §5.7 reasoning (16k headroom + whole-chunk
        defer on truncation) rather than a hard CLI flag; the value is still
        threaded through so the moment a flag exists it is a one-line addition.
        """
        argv = [
            "claude",
            "-p",
            "--output-format",
            "json",
            "--model",
            self._settings.llm.model,
            "--max-turns",
            "1",
            "--no-session-persistence",
        ]
        env = {**os.environ, _DISABLE_HOOKS_ENV: "1"}  # MERGE — MF-2, load-bearing.
        try:
            proc = self._runner(
                argv,
                input=prompt,
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise _SpawnTransientError("claude -p timed out") from exc
        except OSError as exc:
            raise _SpawnTransientError("claude -p failed to spawn") from exc
        if proc.returncode != 0:
            raise _SpawnTransientError(f"claude -p exited {proc.returncode}")
        return self._parse_envelope(proc.stdout)

    @staticmethod
    def _parse_envelope(stdout: str) -> tuple[str, int, int]:
        """Extract ``(result_text, input_tokens, output_tokens)`` from the JSON envelope.

        ``claude -p --output-format json`` emits an object with a ``result``
        text field and a ``usage`` block (§1.3). A non-JSON envelope, one missing
        ``result``, or one flagged ``is_error: true`` is treated as a transient
        spawn failure — the CLI misbehaved at the transport layer, distinct from
        the inner result text being malformed (which is a ``parse`` deferral
        handled upstream).
        """
        try:
            envelope = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise _SpawnTransientError("claude -p envelope was not JSON") from exc
        if not isinstance(envelope, dict) or "result" not in envelope:
            raise _SpawnTransientError("claude -p envelope had no result field")
        # Belt-and-braces against CLI drift: a real API error that ever returns
        # exit 0 + `is_error:true` is deferred as transient, not parsed (§7.2).
        if envelope.get("is_error") is True:
            raise _SpawnTransientError("claude -p envelope reported is_error")
        result_text = envelope.get("result")
        if not isinstance(result_text, str):
            raise _SpawnTransientError("claude -p result field was not a string")
        usage = envelope.get("usage")
        in_tok, out_tok = 0, 0
        if isinstance(usage, dict):
            in_tok = _as_int(usage.get("input_tokens"))
            out_tok = _as_int(usage.get("output_tokens"))
        return result_text, in_tok, out_tok

    # ------------------------------------------------------------------ #
    # enrich_batch (§5.6 / §5.7) — chunk, spawn, defensive-parse, defer.  #
    # ------------------------------------------------------------------ #

    def enrich_batch(self, reqs: list[EnrichRequest]) -> BackendOutcome:
        """Enrich a batch of records via chunked ``claude -p`` spawns (IN-13).

        Chunks ``reqs`` into groups of ``cli_chunk_size`` (25), one spawn per
        chunk, array-in / array-out keyed by ``record_id`` (§5.7). Each chunk's
        outcome is independent: a parse-failed chunk defers without affecting the
        others. Never raises (SC-3).
        """
        results: list[EnrichResult] = []
        deferred: list[DeferralEntry] = []
        spend: list[SpendEntry] = []
        chunk_size = max(1, self._settings.llm.cli_chunk_size)
        for start in range(0, len(reqs), chunk_size):
            chunk = reqs[start : start + chunk_size]
            self._enrich_chunk(chunk, results, deferred, spend)
        return BackendOutcome(results=results, deferred=deferred, spend=spend)

    def _enrich_chunk(
        self,
        chunk: list[EnrichRequest],
        results: list[EnrichResult],
        deferred: list[DeferralEntry],
        spend: list[SpendEntry],
    ) -> None:
        """Run one chunk: transient-retry the spawn, then defensive-parse/repair.

        Appends into the shared ``results`` / ``deferred`` / ``spend`` ledgers.
        A spawn that exhausts its transient budget defers the whole chunk with
        reason ``transient`` (no spend row — no usable usage block). A spawn that
        succeeds but yields unparseable text re-spawns once (``cli_parse_retries``)
        and, on a successful repair, stamps the spend row ``Outcome='repaired'``.
        """
        prompt = _build_enrich_prompt(chunk)
        timeout = 120
        max_out = self._settings.llm.cli_max_output_tokens

        # --- Spawn with transient retry (§5.8 CLI: 2 retries, base 1s ×2). ---
        try:
            result_text, in_tok, out_tok, retries = self._spawn_with_transient_retry(
                prompt, max_output_tokens=max_out, timeout=timeout
            )
        except _SpawnTransientError:
            deferred.extend(DeferralEntry(record_id=r.record_id, reason="transient") for r in chunk)
            return

        # --- Defensive parse; on chunk-level failure re-spawn once (repair). ---
        repaired = False
        start = time.monotonic()
        try:
            elements = _defensive_parse_array(result_text)
        except _ChunkParseError:
            if self._settings.llm.cli_parse_retries >= 1:
                try:
                    result_text, in2, out2, retries2 = self._spawn_with_transient_retry(
                        prompt, max_output_tokens=max_out, timeout=timeout
                    )
                except _SpawnTransientError:
                    self._defer_chunk_parse(chunk, deferred, spend, in_tok, out_tok, retries, start)
                    return
                in_tok += in2
                out_tok += out2
                retries += retries2
                try:
                    elements = _defensive_parse_array(result_text)
                    repaired = True
                except _ChunkParseError:
                    self._defer_chunk_parse(chunk, deferred, spend, in_tok, out_tok, retries, start)
                    return
            else:
                self._defer_chunk_parse(chunk, deferred, spend, in_tok, out_tok, retries, start)
                return

        # --- Element-level: accept valid, defer invalid / dropped (§5.6). ---
        latency_ms = int((time.monotonic() - start) * 1000)
        by_id = _index_elements_by_record_id(elements)
        for req in chunk:
            element = by_id.get(req.record_id)
            parsed = _validate_element(element, req) if element is not None else None
            if parsed is None:
                deferred.append(DeferralEntry(record_id=req.record_id, reason="parse"))
            else:
                results.append(parsed)
        spend.append(
            SpendEntry(
                call_site="save",
                model=self._settings.llm.model,
                backend="cli",
                input_tokens=in_tok,
                output_tokens=out_tok,
                idempotency_key=None,  # CLI path: EnrichPending IS the boundary (§5.8).
                latency_ms=latency_ms,
                retry_count=retries,
                outcome="repaired" if repaired else "ok",
            )
        )

    def _defer_chunk_parse(
        self,
        chunk: list[EnrichRequest],
        deferred: list[DeferralEntry],
        spend: list[SpendEntry],
        in_tok: int,
        out_tok: int,
        retries: int,
        start: float,
    ) -> None:
        """Defer a whole chunk on an unrecoverable parse failure (§5.6).

        Every record → ``DeferralEntry(reason="parse")``; one ``SpendEntry`` with
        ``Outcome='deferred'`` records the tokens actually burned (the spawn(s)
        did consume usage even though the result was unusable).
        """
        latency_ms = int((time.monotonic() - start) * 1000)
        deferred.extend(DeferralEntry(record_id=r.record_id, reason="parse") for r in chunk)
        spend.append(
            SpendEntry(
                call_site="save",
                model=self._settings.llm.model,
                backend="cli",
                input_tokens=in_tok,
                output_tokens=out_tok,
                idempotency_key=None,
                latency_ms=latency_ms,
                retry_count=retries,
                outcome="deferred",
            )
        )

    def _spawn_with_transient_retry(
        self, prompt: str, *, max_output_tokens: int, timeout: int
    ) -> tuple[str, int, int, int]:
        """Spawn with the §5.8 CLI transient-retry budget; return tokens + retry count.

        On a transient spawn failure, back off (base 1 s ×2, ~10 s cap) and retry
        up to ``_TRANSIENT_RETRIES`` times; re-raise :class:`_SpawnTransientError`
        if all attempts fail. The returned int is the number of retries consumed
        (0 on first-attempt success) for ``SpendLog.RetryCount`` (§3.4).
        """
        last_exc: _SpawnTransientError | None = None
        for attempt in range(_TRANSIENT_RETRIES + 1):
            try:
                result_text, in_tok, out_tok = self._spawn(
                    prompt, max_output_tokens=max_output_tokens, timeout=timeout
                )
            except _SpawnTransientError as exc:
                last_exc = exc
                if attempt < _TRANSIENT_RETRIES:
                    self._sleeper(min(_BACKOFF_BASE_S * (2**attempt), _BACKOFF_CAP_S))
                continue
            return result_text, in_tok, out_tok, attempt
        assert last_exc is not None  # loop always sets it before falling through.
        raise last_exc

    # ------------------------------------------------------------------ #
    # reflect (§5.3 / §5.7) — one spawn, defensive-parse two arrays.      #
    # ------------------------------------------------------------------ #

    def reflect(self, req: ReflectRequest) -> ReflectOutcome:
        """Reflect over one session's bounded Fork B rows (IN-14, §5.3).

        One ``claude -p`` spawn carrying the bounded activity; defensive-parse
        the ``passive_hits`` / ``promotion_candidates`` arrays out of an object
        result; validate ids ∈ the supplied log and drop out-of-set ones
        (§5.3); propose-only. Parse / transient failure → empty
        :class:`ReflectOutcome` (no raise — reindex is the backstop, SC-9).
        """
        prompt = _build_reflect_prompt(req)
        start = time.monotonic()
        try:
            result_text, in_tok, out_tok, retries = self._spawn_with_transient_retry(
                prompt, max_output_tokens=self._settings.llm.cli_max_output_tokens, timeout=120
            )
        except _SpawnTransientError:
            return ReflectOutcome()

        try:
            obj = _defensive_parse_object(result_text)
        except _ChunkParseError:
            return ReflectOutcome()

        valid_record_ids = set(req.active_record_ids)
        valid_archive_ids = {row.archive_id for row in req.activity}
        passive_hits = _parse_passive_hits(obj.get("passive_hits"), valid_record_ids)
        promotion_candidates = _parse_promotion_candidates(
            obj.get("promotion_candidates"), valid_archive_ids
        )
        latency_ms = int((time.monotonic() - start) * 1000)
        spend = SpendEntry(
            call_site="reflect",
            model=self._settings.llm.model,
            backend="cli",
            input_tokens=in_tok,
            output_tokens=out_tok,
            idempotency_key=None,
            latency_ms=latency_ms,
            retry_count=retries,
            outcome="ok",
        )
        return ReflectOutcome(
            passive_hits=passive_hits,
            promotion_candidates=promotion_candidates,
            spend=[spend],
        )


# --------------------------------------------------------------------------- #
# Prompt construction (CLI-flavored — array-in / array-out, §5.6).              #
# --------------------------------------------------------------------------- #

#: The CLI enrichment instruction. CLI-flavored (the SDK path uses forced
#: tool-use; the CLI path can only ask in-prompt for a strict JSON array keyed by
#: record_id, §5.6). User-record bodies are laid in the data block, never
#: substituted into instructions — no prompt-injection surface for the model's
#: own behavior (the records are the user's own memory text, but the discipline
#: holds: instructions and data are kept in distinct sections).
_ENRICH_SYSTEM = """\
You enrich, dedup, and contradiction-check memory records. You are given a JSON \
array of records, each with a record_id, name, body, and a list of dedup \
candidates (existing active records that may be the same or in conflict).

Respond with ONLY a JSON array (no prose, no code fences). The array MUST have \
exactly one element per input record, each element an object with ALL of these \
fields:
  - "record_id": echo the input record_id unchanged.
  - "summary": a one-to-two sentence summary of the record body.
  - "aliases": an array of 1 to 8 short alternate names/keywords for the record.
  - "dedup_verdict": one of "new", "duplicate", "conflict".
  - "dedup_target_name": the candidate name this duplicates/conflicts with, or \
null when the verdict is "new". It MUST be one of the candidate names supplied \
for that record.
  - "conflict_explanation": a sentence explaining the conflict, or null unless \
the verdict is "conflict".

Every field is required for every element. Emit the array and nothing else.

A one-element example:
[{"record_id":"redis-port","summary":"Redis listens on port 6380 on this host.",\
"aliases":["redis","cache port"],"dedup_verdict":"new","dedup_target_name":null,\
"conflict_explanation":null}]
"""

_REFLECT_SYSTEM = """\
You review one session's activity log to (1) identify passive hits — existing \
memory records the session relied on — and (2) propose Fork B->A promotion \
candidates worth saving as durable memory. You propose only; nothing is \
auto-applied.

You are given a JSON object with "active_record_ids" (the valid record ids you \
may cite as passive hits) and "activity" (rows, each with an archive_id you may \
cite as a promotion candidate).

Respond with ONLY a JSON object (no prose, no code fences) with exactly these \
two fields:
  - "passive_hits": an array of {"record_id": <one of active_record_ids>, \
"evidence": <short reason>}. May be empty.
  - "promotion_candidates": an array of {"archive_id": <one of the activity \
archive_ids>, "proposed_summary": <text>, "rationale": <text>}. May be empty.

Emit the object and nothing else.
"""


def _build_enrich_prompt(chunk: list[EnrichRequest]) -> str:
    """Build the CLI enrichment prompt: instruction + a JSON data block (§5.6)."""
    records = [
        {
            "record_id": req.record_id,
            "name": req.name,
            "body": req.body,
            "candidates": [
                {
                    "name": c.name,
                    "summary": c.summary,
                    "aliases": c.aliases,
                    "excerpt": c.excerpt,
                }
                for c in req.candidates
            ],
        }
        for req in chunk
    ]
    return f"{_ENRICH_SYSTEM}\n\nRECORDS:\n{json.dumps(records, ensure_ascii=False)}\n"


def _build_reflect_prompt(req: ReflectRequest) -> str:
    """Build the CLI reflection prompt: instruction + a JSON data block (§5.3)."""
    payload = {
        "active_record_ids": req.active_record_ids,
        "activity": [
            {"archive_id": row.archive_id, "role": row.role, "kind": row.kind, "body": row.body}
            for row in req.activity
        ],
    }
    return f"{_REFLECT_SYSTEM}\n\nSESSION:\n{json.dumps(payload, ensure_ascii=False)}\n"


# --------------------------------------------------------------------------- #
# Defensive parse / repair (§5.6) — strip fences, regex-extract, validate.      #
# --------------------------------------------------------------------------- #


def _strip_fences(text: str) -> str:
    """Remove a leading/trailing markdown code fence if the model wrapped output."""
    stripped = text.strip()
    stripped = _FENCE_RE.sub("", stripped)
    return stripped.strip()


def _defensive_parse_array(result_text: str) -> list[Any]:
    """Extract the outermost ``[...]`` array from the result text (§5.6).

    Strip code fences, regex-extract the OUTERMOST array, ``json.loads`` it, and
    require a list at the top level. Any failure raises :class:`_ChunkParseError`
    (the caller re-spawns once then defers the chunk).
    """
    candidate = _strip_fences(result_text)
    match = _ARRAY_RE.search(candidate)
    if match is None:
        raise _ChunkParseError("no outermost [...] array found")
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise _ChunkParseError("array region was not valid JSON") from exc
    if not isinstance(parsed, list):
        raise _ChunkParseError("top-level JSON was not an array")
    return parsed


def _defensive_parse_object(result_text: str) -> dict[str, Any]:
    """Extract the outermost ``{...}`` object from the result text (§5.3 reflect).

    Mirror of :func:`_defensive_parse_array` for the reflection object result.
    """
    candidate = _strip_fences(result_text)
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise _ChunkParseError("no outermost {...} object found")
    try:
        parsed = json.loads(candidate[start : end + 1])
    except json.JSONDecodeError as exc:
        raise _ChunkParseError("object region was not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise _ChunkParseError("top-level JSON was not an object")
    return parsed


def _index_elements_by_record_id(elements: list[Any]) -> dict[str, dict[str, Any]]:
    """Map ``record_id`` → element for the well-formed dict elements only.

    Elements that are not dicts or lack a string ``record_id`` are skipped (the
    record they would have matched defers as ``parse`` — handled by the caller
    finding no entry for it). On duplicate ids the first wins.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for element in elements:
        if not isinstance(element, dict):
            continue
        rid = element.get("record_id")
        if isinstance(rid, str) and rid not in by_id:
            by_id[rid] = element
    return by_id


def _validate_element(element: dict[str, Any], req: EnrichRequest) -> EnrichResult | None:
    """Validate one array element against the §5.2 fields → ``EnrichResult``.

    Returns ``None`` (→ caller defers reason ``parse``) on any schema violation:
    missing/typed-wrong ``summary``, ``aliases`` not a 1..8 list of strings, an
    out-of-enum ``dedup_verdict``, or a non-null-non-string nullable field. A
    ``dedup_target_name`` that is not ∈ this record's candidate names is coerced
    to ``new`` with a null target (§5.2 defensive rule — the model occasionally
    hallucinates a target), rather than deferred.
    """
    summary = element.get("summary")
    if not isinstance(summary, str) or not summary:
        return None

    aliases = element.get("aliases")
    if (
        not isinstance(aliases, list)
        or not (1 <= len(aliases) <= 8)
        or not all(isinstance(a, str) for a in aliases)
    ):
        return None

    raw_verdict = element.get("dedup_verdict")
    if raw_verdict not in ("new", "duplicate", "conflict"):
        return None
    verdict: DedupVerdict = raw_verdict

    target = element.get("dedup_target_name")
    if target is not None and not isinstance(target, str):
        return None
    explanation = element.get("conflict_explanation")
    if explanation is not None and not isinstance(explanation, str):
        return None

    # §5.2 defensive: an out-of-set target (or one supplied with a 'new' verdict)
    # is coerced to a clean 'new' rather than trusted.
    candidate_names = {c.name for c in req.candidates}
    if target is not None and target not in candidate_names:
        verdict = "new"
        target = None
        explanation = None
    if verdict == "new":
        target = None
        explanation = None

    return EnrichResult(
        record_id=req.record_id,
        summary=summary,
        aliases=list(aliases),
        dedup_verdict=verdict,
        dedup_target_name=target,
        conflict_explanation=explanation,
    )


def _parse_passive_hits(raw: Any, valid_record_ids: set[str]) -> list[PassiveHit]:
    """Parse + validate the ``passive_hits`` array, dropping out-of-set ids (§5.3)."""
    if not isinstance(raw, list):
        return []
    hits: list[PassiveHit] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        rid = item.get("record_id")
        evidence = item.get("evidence")
        if isinstance(rid, str) and rid in valid_record_ids and isinstance(evidence, str):
            hits.append(PassiveHit(record_id=rid, evidence=evidence))
    return hits


def _parse_promotion_candidates(
    raw: Any, valid_archive_ids: set[str]
) -> list[PromotionCandidate]:
    """Parse + validate ``promotion_candidates``, dropping out-of-set ids (§5.3)."""
    if not isinstance(raw, list):
        return []
    candidates: list[PromotionCandidate] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        archive_id = item.get("archive_id")
        proposed = item.get("proposed_summary")
        rationale = item.get("rationale")
        if (
            isinstance(archive_id, str)
            and archive_id in valid_archive_ids
            and isinstance(proposed, str)
            and isinstance(rationale, str)
        ):
            candidates.append(
                PromotionCandidate(
                    archive_id=archive_id, proposed_summary=proposed, rationale=rationale
                )
            )
    return candidates


def _as_int(value: Any) -> int:
    """Coerce a usage-block value to a non-negative int; 0 on anything else."""
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return max(0, value)
    return 0


__all__ = ["ClaudeCliBackend"]
