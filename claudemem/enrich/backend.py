"""claudemem.enrich.backend — the EnrichmentBackend boundary (L3).

This module defines the **transport-neutral seam** between ClaudeMem's two
enrichment routines (``enrich_batch`` / ``reflect``, tech-design §5.1/§5.3) and
whichever concrete transport is live (``ClaudeCliBackend`` spawning ``claude``,
``AnthropicSdkBackend`` importing ``anthropic``, or the no-model
``LexicalOnlyBackend`` defined here). The protocol's request/result dataclasses
carry **no ``anthropic`` types and no ``subprocess``/``claude`` types** — that
absence is what lets the two routines and the ``cli``/``hooks`` call sites stay
oblivious to the transport (architecture §2.6, §7.2; SC-6 restated against
transport methods, tech-design §2.2).

**Import discipline (the load-bearing rule, C-17 / SC-6).** This module imports
``config`` + the standard library ONLY. It MUST NOT import ``anthropic`` and
MUST NOT import ``subprocess`` or spawn ``claude``:

* The Anthropic SDK import is funneled **function-locally** into
  ``backend_sdk.py`` wrapped in ``try/except ImportError`` (C-17, tech-design
  §6.1) — never here.
* The ``claude`` subprocess spawn (and the ``claude auth status`` availability
  probe, §7.3) live in ``backend_cli.py`` — never here.

:func:`select_backend` reaches those concrete backends by **lazy ``importlib``
import inside the function body**, guarded by ``try/except ImportError`` so that
merely importing ``claudemem.enrich.backend`` pulls in neither the SDK nor the
CLI spawn machinery, and a missing concrete module degrades cleanly to
lexical-only (SC-3 / NG-5). The firewall test asserts ``anthropic`` is absent
from ``sys.modules`` after importing this module.

**Backend selection** (:func:`select_backend`, tech-design §5.9) is
**write-path only, lazy, and cached per-process** (NG-1: no persistence). A
forced-but-unavailable backend warns once and falls through to lexical-only —
it never raises (SC-3).
"""

from __future__ import annotations

import importlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Literal, Protocol, runtime_checkable

from claudemem import config

_log = logging.getLogger("claudemem")

#: ``[llm].backend`` values (tech-design §5.9). ``auto`` resolves CLI→SDK→none.
BackendName = Literal["auto", "cli", "sdk", "none"]

#: ``EnrichResult.dedup_verdict`` values (tech-design §5.1/§5.2).
DedupVerdict = Literal["new", "duplicate", "conflict"]

#: ``DeferralEntry.reason`` values — the IN-13 deferral triggers, 1:1
#: (tech-design §5.1/§5.8): a model-free / never-attempted defer is ``auth`` or
#: ``cap``; a CLI JSON parse/repair failure is ``parse``; a transient
#: (429/5xx/timeout) defer after retries is ``transient``.
DeferralReason = Literal["parse", "cap", "auth", "transient"]


# --------------------------------------------------------------------------- #
# Transport-neutral request / result dataclasses (tech-design §5.1, §5.3)       #
#                                                                               #
# Frozen where they are pure values crossing the boundary. NONE of these carry  #
# an ``anthropic`` or ``subprocess``/``claude`` type — the seam is transport-   #
# neutral by construction (SC-6).                                               #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class Candidate:
    """One model-free dedup candidate (tech-design §5.4).

    Assembled before the single enrichment call from the top-``dedup_k`` similar
    active Fork A records in scope via the FTS5 lexical path (no model). Carries
    name + summary + aliases + a head-only excerpt so the model can judge
    duplicate/conflict without a second call (IN-13, exactly one model call per
    record).
    """

    name: str
    summary: str
    aliases: list[str]
    excerpt: str


@dataclass(frozen=True, slots=True)
class EnrichRequest:
    """One record to enrich + dedup + contradiction-check (tech-design §5.1).

    ``record_id`` is the caller's correlation key (the Fork A ``name`` or a
    batch index) — it is echoed back on the matching :class:`EnrichResult` /
    :class:`DeferralEntry` so the caller can reconcile the array-in / array-out
    keyed result (§5.6/§5.7).
    """

    record_id: str
    name: str
    body: str
    candidates: list[Candidate]


@dataclass(frozen=True, slots=True)
class EnrichResult:
    """The 3-job enrichment result for one record (tech-design §5.1/§5.2).

    All fields are present regardless of ``dedup_verdict`` (IN-13): a
    ``duplicate``/``conflict`` verdict never suppresses ``summary``/``aliases``.
    ``dedup_target_name`` / ``conflict_explanation`` are nullable but always
    present (``None`` when not applicable). ``aliases`` is 1..8 entries (§5.2
    schema ``minItems: 1, maxItems: 8``).
    """

    record_id: str
    summary: str
    aliases: list[str]
    dedup_verdict: DedupVerdict
    dedup_target_name: str | None
    conflict_explanation: str | None


@dataclass(frozen=True, slots=True)
class DeferralEntry:
    """A record the backend could not enrich (tech-design §5.1, MF-1).

    Listed in :attr:`BackendOutcome.deferred`; the caller marks the record
    ``EnrichPending=1`` for the next ``reindex``. ``reason`` is the triageable
    deferral cause (the IN-13 deferral triggers, 1:1) — it maps onto the
    ``SpendLog.Outcome='deferred'`` row where a call was actually attempted
    (§5.6/§5.8); ``auth``/``cap`` defers that never reach the model produce no
    ``SpendLog`` row but still carry a ``DeferralEntry``.
    """

    record_id: str
    reason: DeferralReason


@dataclass(frozen=True, slots=True)
class SpendEntry:
    """Post-call usage to write to ``SpendLog`` (tech-design §5.1, §3.4).

    Transport-neutral mirror of the ``store.spend.record_spend`` /
    ``record_spend_and_clear_pending`` parameters: the backend reports what it
    actually spent and the caller persists it. ``record_id_int`` is the
    ``Record.Id`` to clear ``EnrichPending`` on in the SAME ``BEGIN IMMEDIATE``
    transaction as the spend insert (§3.6 atomic pairing); ``None`` for
    ``reflect`` (which clears no ``Record`` row) and for never-attempted defers.
    Cache token classes are not carried — they are always 0 in v1 (§2.3) and the
    store forces them to 0.
    """

    call_site: Literal["save", "reflect"]
    model: str
    backend: Literal["sdk", "cli"]
    input_tokens: int = 0
    output_tokens: int = 0
    idempotency_key: str | None = None
    latency_ms: int = 0
    retry_count: int = 0
    outcome: Literal["ok", "deferred", "repaired"] = "ok"
    record_id_int: int | None = None


@dataclass(frozen=True, slots=True)
class BackendOutcome:
    """The result of one ``enrich_batch`` call (tech-design §5.1).

    Three disjoint ledgers: ``results`` (records enriched), ``deferred`` (the
    unified deferral ledger — each carries a triageable ``reason``), and
    ``spend`` (post-call usage rows to persist). A purely-degraded outcome
    (lexical-only / over-cap) has empty ``results`` and every input record in
    ``deferred``.
    """

    results: list[EnrichResult] = field(default_factory=list)
    deferred: list[DeferralEntry] = field(default_factory=list)
    spend: list[SpendEntry] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Reflection request / outcome (tech-design §5.3 — session_reflection schema)   #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ActivityRow:
    """One bounded Fork B row fed to reflection (tech-design §5.3, §3.5).

    Reflection runs over the already-capped / tool-output-skipped Fork B rows
    for one ``session_id`` (NOT the raw transcript). ``archive_id`` is the
    ``b:<rowid>`` id the model may cite as a promotion candidate.
    """

    archive_id: str
    role: str
    kind: str
    body: str


@dataclass(frozen=True, slots=True)
class ReflectRequest:
    """The SessionEnd reflection input (tech-design §5.3, IN-14).

    ``active_record_ids`` and the ``archive_id`` of each row in ``activity`` are
    the validation sets: the model's ``passive_hits[].record_id`` and
    ``promotion_candidates[].archive_id`` are checked against them and
    out-of-set ids are dropped (§5.3).
    """

    session_id: str
    activity: list[ActivityRow]
    active_record_ids: list[str]


@dataclass(frozen=True, slots=True)
class PassiveHit:
    """A record the session passively used, to reinforce (tech-design §5.3)."""

    record_id: str
    evidence: str


@dataclass(frozen=True, slots=True)
class PromotionCandidate:
    """A proposed (never auto-applied) Fork B→A promotion (tech-design §5.3).

    Surfaces for approval at session-end / ``reindex`` (NG-6, SC-8).
    """

    archive_id: str
    proposed_summary: str
    rationale: str


@dataclass(frozen=True, slots=True)
class ReflectOutcome:
    """The reflection result (tech-design §5.3, IN-14).

    Both lists may be empty (a session with no hits/candidates returns ``[]`` /
    ``[]``). ``spend`` carries the one reflection-call ``SpendEntry`` (empty for
    the lexical-only no-model path).
    """

    passive_hits: list[PassiveHit] = field(default_factory=list)
    promotion_candidates: list[PromotionCandidate] = field(default_factory=list)
    spend: list[SpendEntry] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# The EnrichmentBackend protocol (tech-design §5.1)                             #
# --------------------------------------------------------------------------- #


@runtime_checkable
class EnrichmentBackend(Protocol):
    """The single transport-neutral adapter the two routines call (§5.1).

    The two enrichment routines (``enrich_batch`` / ``reflect``) call this and
    never know which transport is live (C-3 amended §2.1, SC-6). ``detect()`` is
    a static availability probe run only on the write path (§5.9) — read paths
    never call it (C-17).
    """

    @staticmethod
    def detect() -> bool:
        """Is this transport available right now? (write-path only, §5.9)."""
        ...

    def enrich_batch(self, reqs: list[EnrichRequest]) -> BackendOutcome:
        """Enrich + dedup + contradiction-check a batch of records (IN-13)."""
        ...

    def reflect(self, req: ReflectRequest) -> ReflectOutcome:
        """Reflect over one session's bounded Fork B rows (IN-14)."""
        ...


# --------------------------------------------------------------------------- #
# LexicalOnlyBackend — the no-model fallback (pure; lives here, §5.9)           #
# --------------------------------------------------------------------------- #


class LexicalOnlyBackend:
    """The no-model fallback backend (tech-design §5.9, SC-3 / NG-5).

    Constructs **no** model request (it imports nothing beyond this module), so
    it lives in ``backend.py`` rather than in a transport module. It is the
    resolved backend whenever no transport is available (missing key/SDK,
    missing/unauthenticated CLI, ``backend=none``, or a forced-but-unavailable
    backend after the warn-once fallthrough).

    * :meth:`enrich_batch` defers **every** input record with a
      :class:`DeferralEntry`, so the caller marks each ``EnrichPending=1`` and
      the next ``reindex`` backfills the enrichment (SC-3 degraded-save path,
      tech-design §5.1 / architecture §5.1). It records **no** spend (no call
      was attempted).
    * :meth:`reflect` returns an empty :class:`ReflectOutcome` (no hits, no
      candidates, no spend) — the SessionEnd reflection simply does nothing when
      degraded; ``reindex`` is the backstop (SC-9).

    **Deferral reason = ``"auth"``.** A model-free fallback is, by definition, a
    transport that is unavailable / not authenticated, so ``auth`` is the
    spec-consistent reason for these never-attempted defers (tech-design §5.1:
    "auth/cap deferrals that never reach the model produce no SpendLog row but
    still carry a DeferralEntry"; an over-cap throttle is a separate caller-side
    decision that uses ``cap``). This is the most defensible reading; documented
    here as an assumption.
    """

    @staticmethod
    def detect() -> bool:
        """Always available — it is the floor of the fallback chain (§5.9)."""
        return True

    def enrich_batch(self, reqs: list[EnrichRequest]) -> BackendOutcome:
        """Defer every record; record no spend (no model call attempted)."""
        return BackendOutcome(
            results=[],
            deferred=[DeferralEntry(record_id=req.record_id, reason="auth") for req in reqs],
            spend=[],
        )

    def reflect(self, req: ReflectRequest) -> ReflectOutcome:
        """Return an empty reflection outcome — reindex is the backstop (SC-9)."""
        return ReflectOutcome()


# --------------------------------------------------------------------------- #
# Backend selection (write-path only, lazy, cached per-process — §5.9)          #
# --------------------------------------------------------------------------- #

#: Lazy-import targets for the concrete transports. Resolved by module path so
#: importing ``backend.py`` alone pulls in neither the SDK nor the CLI spawn
#: machinery (C-17). Built in parallel tasks (T4.3-T4.6); a missing module is a
#: clean ``ImportError`` fallthrough to lexical-only (SC-3).
_CLI_MODULE = "claudemem.enrich.backend_cli"
_CLI_CLASS = "ClaudeCliBackend"
_SDK_MODULE = "claudemem.enrich.backend_sdk"
_SDK_CLASS = "AnthropicSdkBackend"

# Per-process cache (NG-1: no persistence — process-lifetime ONLY). Keyed by the
# resolved ``[llm].backend`` setting so a test or a long-lived process that
# changes the setting re-resolves. A lock guards parallel-session re-entry
# within one process (two threads resolving at once never race the dict).
_backend_cache: dict[BackendName, EnrichmentBackend] = {}
_cache_lock = threading.Lock()

#: Forced-but-unavailable backends already warned about this process — so the
#: "warn once" contract (§5.9) holds even though selection is cached per setting.
_warned_unavailable: set[BackendName] = set()


def _try_detect(module_name: str, class_name: str) -> EnrichmentBackend | None:
    """Lazily import a concrete backend and probe it; ``None`` if unavailable.

    The import is **inside the function** (C-17: importing ``backend.py`` must
    not import the SDK/CLI machinery). A missing module — the concrete backend
    has not been built yet, or the ``[llm]`` extra is not installed — is a clean
    ``ImportError`` fallthrough returning ``None`` (SC-3 / NG-5 degradation).
    Any failure inside ``detect()`` itself is also swallowed → ``None``, so a
    probe never errors a save (SC-3).
    """
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        return None
    cls = getattr(module, class_name, None)
    if cls is None:
        return None
    try:
        if cls.detect():
            return cls()  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001 — a probe must never error a save (SC-3).
        return None
    return None


def _warn_once_unavailable(forced: BackendName) -> None:
    """Emit a single warning that a forced backend was unavailable (§5.9)."""
    if forced in _warned_unavailable:
        return
    _warned_unavailable.add(forced)
    _log.warning(
        "claudemem: [llm].backend=%s requested but that transport is "
        "unavailable; falling back to lexical-only (enrichment defers to "
        "reindex). SC-3: this is a degradation, not an error.",
        forced,
    )


def _resolve(backend: BackendName) -> EnrichmentBackend:
    """Resolve a backend name to a concrete backend (uncached; tech-design §5.9).

    ``auto`` order = authenticated ``claude`` CLI → SDK key present →
    lexical-only. A forced-but-unavailable backend (``cli``/``sdk``) warns once
    and falls through to lexical-only — never raises (SC-3).
    """
    if backend == "none":
        return LexicalOnlyBackend()

    if backend == "cli":
        cli = _try_detect(_CLI_MODULE, _CLI_CLASS)
        if cli is not None:
            return cli
        _warn_once_unavailable("cli")
        return LexicalOnlyBackend()

    if backend == "sdk":
        sdk = _try_detect(_SDK_MODULE, _SDK_CLASS)
        if sdk is not None:
            return sdk
        _warn_once_unavailable("sdk")
        return LexicalOnlyBackend()

    # auto: prefer subscription CLI, then SDK key, then lexical (§5.9).
    cli = _try_detect(_CLI_MODULE, _CLI_CLASS)
    if cli is not None:
        return cli
    sdk = _try_detect(_SDK_MODULE, _SDK_CLASS)
    if sdk is not None:
        return sdk
    return LexicalOnlyBackend()


def select_backend(settings: config.Settings) -> EnrichmentBackend:
    """Select the active enrichment backend (tech-design §5.9; WRITE-PATH ONLY).

    Reads ``settings.llm.backend`` (``auto|cli|sdk|none``) and resolves to a
    concrete :class:`EnrichmentBackend`. The result is **cached per-process**
    (NG-1: process-lifetime only, never persisted) keyed by the resolved backend
    name, so ``detect()`` (and therefore the ``claude auth status`` probe and
    any SDK import) runs at most once per name per process. Read paths
    (``search``/``get``/``menu``/``log``) MUST NEVER call this — detection is
    write-path only (C-17, SC-6).

    Never raises: an unknown ``backend`` value, a forced-but-unavailable
    transport, or a not-yet-built concrete module all degrade to
    :class:`LexicalOnlyBackend` (SC-3 / NG-5).
    """
    name = settings.llm.backend
    # Narrow an arbitrary config string to the BackendName Literal; a stray /
    # forward-compat value is treated as `auto` and never errors (SC-3).
    backend: BackendName
    if name == "cli":
        backend = "cli"
    elif name == "sdk":
        backend = "sdk"
    elif name == "none":
        backend = "none"
    else:
        backend = "auto"
    with _cache_lock:
        cached = _backend_cache.get(backend)
        if cached is not None:
            return cached
        resolved = _resolve(backend)
        _backend_cache[backend] = resolved
        return resolved


def _reset_cache_for_tests() -> None:
    """Clear the per-process selection cache + warn-once set (test hook only).

    Not part of the public surface — exposed so unit tests that monkeypatch
    ``detect()`` between cases re-resolve cleanly. Production never calls this
    (the cache is process-lifetime by design, NG-1).
    """
    with _cache_lock:
        _backend_cache.clear()
        _warned_unavailable.clear()


__all__ = [
    "BackendName",
    "DedupVerdict",
    "DeferralReason",
    "Candidate",
    "EnrichRequest",
    "EnrichResult",
    "DeferralEntry",
    "SpendEntry",
    "BackendOutcome",
    "ActivityRow",
    "ReflectRequest",
    "PassiveHit",
    "PromotionCandidate",
    "ReflectOutcome",
    "EnrichmentBackend",
    "LexicalOnlyBackend",
    "select_backend",
]
