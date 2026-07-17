"""claudemem.enrich.routine — the two (and only two) enrichment routines (L3).

This module is the **public surface of the ``enrich`` package** (architecture
§2.6): the two model-request constructors that ``cli`` / ``hooks`` call, and the
only two Haiku code paths in the system (PRD ``SC-6``).

* :func:`enrich_batch` — the shared **save-time** routine (``IN-13``): assemble
  the model-free dedup candidates (§5.4, ``store.fts_candidates``, no model),
  build one :class:`~claudemem.enrich.backend.EnrichRequest` per record, make the
  **one** backend call per record, then apply the verdict
  (new / duplicate / conflict→supersede), persist Fork A markdown + index, and
  record spend with the atomic ``EnrichPending``-clear (§3.6). A save **always**
  persists — a degraded / over-cap / deferred outcome marks the record
  ``EnrichPending=1`` and never blocks (``SC-3`` / ``SC-10``). Reused unchanged
  for one record (``save``) or many (``import`` / ``reindex`` backfill).
* :func:`reflect` — the **SessionEnd** routine (``IN-14``): reflect over one
  session's bounded Fork B rows (``store.forkb.rows_for_session`` — NOT the raw
  transcript), reinforce the confirmed passive hits (validated against the active
  Fork A set, ``SC-9``), and **propose** Fork B→A promotion candidates (never
  auto-applied, ``SC-8`` / ``NG-6``). Degrades to a clean no-op when no transport
  is available (``SC-3``).

**Backend selection seam.** Both routines resolve the live backend lazily via
:func:`~claudemem.enrich.backend.select_backend` (write-path only, §5.9). A
``backend=`` parameter overrides that selection — the injection seam tests use to
supply a fake backend so **no test makes a real model call**.

**The two-spawn SessionEnd ORDER** (reflection-first, then the
``EnrichPending`` backfill via :func:`enrich_batch`) is the *caller's* (hooks')
responsibility (§5.7); :func:`reflect` is just the reflection half.

Module discipline (architecture §4): ``enrich`` is the model side of the SC-6
firewall, so it may import ``store`` / ``files`` / ``config`` / ``backend``. It
must be reachable only from write flows — the read-path firewall forbids
``recall`` from importing ``enrich`` (import-linter contract).
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field, replace

from claudemem import config, files
from claudemem.enrich import backend as backend_mod
from claudemem.enrich.backend import (
    ActivityRow,
    Candidate,
    DeferralReason,
    EnrichmentBackend,
    EnrichRequest,
    EnrichResult,
    PassiveHit,
    PromotionCandidate,
    ReflectRequest,
    SpendEntry,
)
from claudemem.store import forka

_log = logging.getLogger("claudemem")


# --------------------------------------------------------------------------- #
# Result objects (what cli / hooks consume to report)                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class ConflictReport:
    """One surfaced contradiction for ``cli`` to print, NON-BLOCKING (``IN-13``).

    Emitted when the backend returns a ``conflict`` verdict: the prior ``target``
    record was superseded by the new record and the user is offered resolution
    options. The routine never blocks on ``input()`` — it returns this report so
    ``cli`` can print the options and the resolution is re-issued via a separate
    ``--resolve replace|keep-both|supersede`` invocation. The unattended path
    keeps both records and flags the conflict (this report being the flag).
    """

    record_name: str
    target_name: str
    explanation: str | None


@dataclass(frozen=True, slots=True)
class DeferralReport:
    """One record that deferred to the next ``reindex`` (``SC-3`` / ``SC-10``)."""

    record_name: str
    reason: DeferralReason


@dataclass(frozen=True, slots=True)
class EnrichBatchResult:
    """Summary of one :func:`enrich_batch` run (for ``cli`` / ``hooks`` to report).

    Three disjoint tallies plus the conflict reports: ``enriched`` records were
    persisted with a model summary/aliases; ``deferred`` records persisted
    lexical-only with ``EnrichPending=1`` (each :class:`DeferralReport` carries a
    triageable reason); ``conflicts`` are the non-blocking contradiction reports
    to surface; ``spend_rows`` counts the ``SpendLog`` rows recorded.
    """

    enriched: int = 0
    deferred: list[DeferralReport] = field(default_factory=list)
    conflicts: list[ConflictReport] = field(default_factory=list)
    spend_rows: int = 0


@dataclass(frozen=True, slots=True)
class ReflectResult:
    """Summary of one :func:`reflect` run (for ``hooks`` to report).

    ``reinforced`` is the count of confirmed passive hits whose index counters
    were bumped and flushed to frontmatter (``SC-9``). ``proposed_promotions`` are
    the Fork B→A promotion candidates surfaced for approval — never auto-applied
    (``SC-8`` / ``NG-6``). ``spend_rows`` counts the ``SpendLog`` rows recorded.
    """

    reinforced: int = 0
    proposed_promotions: list[PromotionCandidate] = field(default_factory=list)
    spend_rows: int = 0


# --------------------------------------------------------------------------- #
# T4.7 — enrich_batch: the shared save-time routine (IN-13)                     #
# --------------------------------------------------------------------------- #


def _dedup_query(record: files.RecordFile) -> str:
    """The model-free dedup-candidate query for one record (§5.4).

    The name plus the body is the lexical signal we hand to ``fts_candidates``;
    the FTS5 MATCH builder (``store.forka._build_match_expr``) tokenizes and
    quotes it injection-safely, so passing the raw name + body is sound.
    """
    return f"{record.name} {record.body}"


def _build_candidates(
    conn: sqlite3.Connection,
    record: files.RecordFile,
    scope_ctx: config.ScopeContext,
    settings: config.Settings,
) -> list[Candidate]:
    """Assemble the ``dedup_k`` model-free dedup candidates for one record (§5.4).

    Runs the **model-free** FTS5 lexical path (``store.fts_candidates``, reusing
    §4.3 with ``k=dedup_k=5``) to find the top-similar **active** Fork A records
    in the same scope, then loads each and builds a :class:`Candidate` with its
    name + summary + aliases + a ``dedup_excerpt_chars=500`` **head-only** excerpt.
    The record itself is excluded (a record never dedups against itself). No model
    is touched here — this is the IN-13 "exactly one model call per record" rule.
    """
    dedup_k = settings.llm.dedup_k
    excerpt_chars = settings.llm.dedup_excerpt_chars
    hits = forka.fts_candidates(conn, _dedup_query(record), scope_ctx, k=dedup_k)

    candidates: list[Candidate] = []
    for record_id, _bm25 in hits:
        existing = _select_by_id(conn, record_id, scope_ctx)
        if existing is None or existing.name == record.name:
            # Skip the record itself (re-saves) and any row that vanished.
            continue
        candidates.append(
            Candidate(
                name=existing.name,
                summary=existing.summary or "",
                aliases=files.aliases_from_json(existing.aliases_json),
                excerpt=(existing.body or "")[:excerpt_chars],
            )
        )
    return candidates


def _select_by_id(
    conn: sqlite3.Connection, record_id: int, scope_ctx: config.ScopeContext
) -> forka.Record | None:
    """Fetch a candidate :class:`~claudemem.store.forka.Record` by its rowid.

    ``fts_candidates`` returns ``(Id, bm25)`` pairs; the dedup candidate carries
    the record's *name*, so we resolve the rowid to its name here. Column-
    enumerated by-rowid SELECT (never ``SELECT *``), restricted to active rows so
    a just-superseded candidate is not re-offered.
    """
    row = conn.execute(
        "SELECT Name FROM Record WHERE Id = ? AND SupersededBy IS NULL;",
        (record_id,),
    ).fetchone()
    if row is None:
        return None
    return forka.select_record(conn, scope_ctx, row[0])


def _persist_enriched(
    conn: sqlite3.Connection,
    record: files.RecordFile,
    result: EnrichResult,
    scope_ctx: config.ScopeContext,
    candidate_names: set[str],
) -> tuple[ConflictReport | None, int]:
    """Persist one enriched record + handle its verdict (§5.1 / §5.2).

    Applies the model's ``summary``/``aliases`` to the record regardless of
    verdict (``IN-13``: a duplicate/conflict never suppresses them), then:

    * ``new`` → persist as-is.
    * ``duplicate`` → persist (summary/aliases applied); the dedup target is left
      active — a duplicate is a no-op supersede here (the merge is the model
      having already folded the content; we keep the existing target unchanged).
    * ``conflict`` → ``store.mark_superseded`` the validated target (``SupersededBy``
      = this record's name) + ``files.set_superseded`` the target file, and return
      a NON-BLOCKING :class:`ConflictReport` (exit 0; resolution via ``--resolve``).

    ``dedup_target_name`` is validated ∈ the candidate set (defensive — the
    backend already coerces an out-of-set name to ``new``, §5.2). The file is
    written (source of truth) then the index is upserted with
    ``enrich_pending=False`` (a successful enrichment clears the flag directly;
    the §3.6 atomic spend pairing in :func:`enrich_batch` re-confirms it). Returns
    ``(conflict_report_or_None, row_id)`` — the upserted ``Record.Id`` so the
    caller can pair the spend insert atomically.
    """
    enriched = replace(record, summary=result.summary, aliases=result.aliases)

    target = result.dedup_target_name
    target_valid = target is not None and target in candidate_names

    conflict: ConflictReport | None = None
    if result.dedup_verdict == "conflict" and target_valid:
        # Supersede the prior record (soft delete; the SC-7 trail survives).
        assert target is not None  # narrowed by target_valid
        forka.mark_superseded(conn, target, enriched.name, scope_ctx)
        target_file = _read_target_file(scope_ctx, target)
        if target_file is not None:
            files.set_superseded(target_file, enriched.name)
        conflict = ConflictReport(
            record_name=enriched.name,
            target_name=target,
            explanation=result.conflict_explanation,
        )

    # Persist Fork A markdown (source of truth) then the index (rebuildable cache).
    files.write_record(enriched)
    row_id = forka.upsert_record(conn, enriched, scope_ctx, enrich_pending=False)
    return conflict, row_id


def _read_target_file(
    scope_ctx: config.ScopeContext, name: str
) -> files.RecordFile | None:
    """Read a superseded target's markdown file so its frontmatter can be flushed.

    The target lives in the scope's project dir (project scope) or the global dir.
    A hand-deleted target (file gone) returns ``None`` — the index ``mark_superseded``
    still recorded the trail, so a missing file is not an error (``C-11`` / ``SC-4``).
    """
    for directory in (scope_ctx.project_dir, scope_ctx.global_dir):
        if directory is None:
            continue
        path = directory / f"{name}.md"
        if path.is_file():
            return files.read_record(path)
    return None


def _persist_deferred(
    conn: sqlite3.Connection,
    record: files.RecordFile,
    scope_ctx: config.ScopeContext,
) -> None:
    """Persist a deferred record lexical-only with ``EnrichPending=1`` (``SC-3``).

    A save ALWAYS persists (``SC-10`` warn-not-block): the file is written and the
    index upserted with ``enrich_pending=True`` so the next ``reindex`` backfills
    the enrichment (``IN-20``). No spend is recorded here — the backend reports a
    ``SpendEntry`` only for a call it actually attempted (auth/cap defers never
    reach the model; transient/parse defers carry their spend in the outcome).
    """
    files.write_record(record)
    forka.upsert_record(conn, record, scope_ctx, enrich_pending=True)


def _record_spend_for(
    conn: sqlite3.Connection,
    entry: SpendEntry,
    *,
    record_id_int: int | None,
) -> bool:
    """Map one :class:`SpendEntry` to the §3.6 atomic spend + EnrichPending-clear.

    The save-site spend is paired with the ``EnrichPending`` clear for
    ``record_id_int`` in ONE ``BEGIN IMMEDIATE`` transaction (§3.6): a crash can
    no longer leave a record counted-but-unbilled or billed-but-still-pending.
    For ``reflect`` (no ``Record`` row) the id is ``None`` and only the spend
    insert runs. Returns the underlying insert result (``False`` on an idempotent
    duplicate skip, §5.8).
    """
    from claudemem.store import spend

    return spend.record_spend_and_clear_pending(
        conn,
        record_id=record_id_int,
        call_site=entry.call_site,
        model=entry.model,
        backend=entry.backend,
        input_tokens=entry.input_tokens,
        output_tokens=entry.output_tokens,
        idempotency_key=entry.idempotency_key,
        latency_ms=entry.latency_ms,
        retry_count=entry.retry_count,
        outcome=entry.outcome,
    )


def _record_batch_spend(
    conn: sqlite3.Connection,
    spend_entries: list[SpendEntry],
    row_id_by_name: dict[str, int],
    result_order: list[str],
) -> int:
    """Record the save-batch spend, pairing each row with the §3.6 atomic clear.

    Resolves each :class:`SpendEntry`'s ``record_id_int``: the backend sets it
    where it already knows the row id, else we resolve it positionally against the
    persisted results (the SDK length-1 path emits one spend per result in result
    order; a CLI chunk's single spend pairs with the chunk's first result, the
    rest having had their ``EnrichPending`` cleared by their own upsert).
    Returns the count of ``SpendLog`` rows actually inserted (a duplicate
    ``IdempotencyKey`` is an idempotent skip, §5.8).
    """
    spend_rows = 0
    for idx, entry in enumerate(spend_entries):
        if entry.record_id_int is not None:
            rid: int | None = entry.record_id_int
        elif idx < len(result_order):
            rid = row_id_by_name.get(result_order[idx])
        else:
            rid = None
        if _record_spend_for(conn, entry, record_id_int=rid):
            spend_rows += 1
    return spend_rows


def enrich_batch(
    conn: sqlite3.Connection,
    records: list[files.RecordFile],
    scope_ctx: config.ScopeContext,
    settings: config.Settings,
    *,
    backend: EnrichmentBackend | None = None,
) -> EnrichBatchResult:
    """The shared save-time enrich + dedup + contradiction routine (T4.7, ``IN-13``).

    For each record in ``records``:

    1. **Model-free dedup assembly** (§5.4): ``store.fts_candidates`` with
       ``k=dedup_k=5`` over active Fork A rows in scope → build
       :class:`Candidate` s (name + summary + aliases + ``dedup_excerpt_chars=500``
       head excerpt). NO model.
    2. Build one :class:`EnrichRequest` (``record_id`` = the record's ``name``).

    Then the **one** backend call per record (``backend.enrich_batch`` — the
    backend makes exactly one model request per record, ``IN-13``). The
    :class:`~claudemem.enrich.backend.BackendOutcome` is applied:

    * ``results`` → :func:`_persist_enriched` (verdict handling: ``new`` proceeds,
      ``duplicate`` merges, ``conflict`` supersedes the target + surfaces a
      NON-BLOCKING :class:`ConflictReport`), file + index write, atomic spend.
    * ``deferred`` → :func:`_persist_deferred` (lexical-only, ``EnrichPending=1``).

    The backend is resolved lazily via ``select_backend`` (write-path only) unless
    a ``backend`` is injected (the test seam). A save **always** persists and
    **never** blocks (``SC-10`` / ``SC-3``). Returns an :class:`EnrichBatchResult`
    summarizing enriched / deferred / conflicts / spend for ``cli`` / ``hooks``.
    """
    if not records:
        return EnrichBatchResult()

    active_backend = backend if backend is not None else backend_mod.select_backend(settings)

    # 1-2. Model-free candidate assembly + request build (one per record).
    reqs: list[EnrichRequest] = []
    candidate_names_by_id: dict[str, set[str]] = {}
    record_by_id: dict[str, files.RecordFile] = {}
    for record in records:
        candidates = _build_candidates(conn, record, scope_ctx, settings)
        candidate_names_by_id[record.name] = {c.name for c in candidates}
        record_by_id[record.name] = record
        reqs.append(
            EnrichRequest(
                record_id=record.name,
                name=record.name,
                body=record.body,
                candidates=candidates,
            )
        )

    # The ONE model call per record (the backend owns batching/retry/idempotency).
    outcome = active_backend.enrich_batch(reqs)

    enriched_count = 0
    deferred_reports: list[DeferralReport] = []
    conflicts: list[ConflictReport] = []

    # Persist every enriched record first, recording the upserted Record.Id per
    # name so the spend insert (below) can pair the §3.6 atomic EnrichPending-clear
    # against the right row. The upsert already wrote enrich_pending=False; the
    # spend pairing re-confirms it atomically with the SpendLog insert.
    row_id_by_name: dict[str, int] = {}
    result_order: list[str] = []
    for result in outcome.results:
        target_record = record_by_id.get(result.record_id)
        if target_record is None:
            # Defensive: a backend echoed an unknown record_id; ignore it.
            continue
        conflict, row_id = _persist_enriched(
            conn,
            target_record,
            result,
            scope_ctx,
            candidate_names_by_id.get(result.record_id, set()),
        )
        row_id_by_name[result.record_id] = row_id
        result_order.append(result.record_id)
        enriched_count += 1
        if conflict is not None:
            conflicts.append(conflict)

    spend_rows = _record_batch_spend(conn, outcome.spend, row_id_by_name, result_order)

    for entry in outcome.deferred:
        deferred_record = record_by_id.get(entry.record_id)
        if deferred_record is None:
            continue
        _persist_deferred(conn, deferred_record, scope_ctx)
        deferred_reports.append(
            DeferralReport(record_name=entry.record_id, reason=entry.reason)
        )

    return EnrichBatchResult(
        enriched=enriched_count,
        deferred=deferred_reports,
        conflicts=conflicts,
        spend_rows=spend_rows,
    )


# --------------------------------------------------------------------------- #
# T4.8 — reflect: the SessionEnd reflection routine (IN-14)                     #
# --------------------------------------------------------------------------- #


def _activity_rows(rows: list[sqlite3.Row]) -> list[ActivityRow]:
    """Map bounded Fork B ``Activity`` rows to transport-neutral :class:`ActivityRow` s.

    ``store.forkb.rows_for_session`` returns the already-bounded / capped /
    tool-output-skipped rows (§3.5). A tool row has a NULL ``Body`` and only a
    ``ToolRef`` line; we surface ``ToolRef`` as the body in that case so a
    reflection can still reference the event without ever retaining the tool
    output. ``archive_id`` is the ``b:<rowid>`` id the model may cite (§5.3).
    """
    activity: list[ActivityRow] = []
    for row in rows:
        body = row["Body"] if row["Body"] is not None else (row["ToolRef"] or "")
        activity.append(
            ActivityRow(
                archive_id=f"b:{row['Id']}",
                role=row["Role"],
                kind=row["Kind"],
                body=body,
            )
        )
    return activity


def reflect(
    conn_b: sqlite3.Connection,
    session_id: str,
    conn_a: sqlite3.Connection,
    scope_ctx: config.ScopeContext,
    settings: config.Settings,
    *,
    backend: EnrichmentBackend | None = None,
) -> ReflectResult:
    """The SessionEnd reflection routine (T4.8, ``IN-14`` / ``SC-9`` / ``SC-8``).

    Reads ``session_id``'s **bounded** Fork B rows
    (``store.forkb.rows_for_session`` — capped, tool-output-skipped, NOT the raw
    transcript), builds the active Fork A id set as the validation set, and makes
    the one reflection call (``backend.reflect``). Then:

    * **passive_hits** — each ``record_id`` is re-validated against the active
      Fork A set (the backend validates against the supplied log; we re-resolve
      the id to a real active record here). For each valid hit: bump
      ``HitCount`` + recency in the INDEX, flush via ``files.writeback_counters``
      to frontmatter (``SC-9`` — reinforce on confirmed hit only), and clear the
      ``stale`` flag via ``files.set_stale`` (``SC-13``).
    * **promotion_candidates** — collected and returned for surfacing; NEVER
      auto-applied (``SC-8`` / ``NG-6``).

    Spend from the outcome is recorded (``call_site='reflect'``; no ``Record`` row
    to clear). Degrades to a clean no-op when no transport is available (empty
    outcome → nothing reinforced, nothing proposed, no error — ``SC-3``); the next
    ``reindex`` is the backstop. The two-spawn SessionEnd ORDER (reflection then
    ``EnrichPending`` backfill) is the caller's responsibility (§5.7).
    """
    active = forka.active_set(conn_a, scope_ctx)
    active_by_name: dict[str, forka.Record] = {rec.name: rec for rec in active}

    rows = forkb_rows(conn_b, session_id, settings)
    req = ReflectRequest(
        session_id=session_id,
        activity=_activity_rows(rows),
        active_record_ids=[f"a:{name}" for name in active_by_name],
    )

    active_backend = backend if backend is not None else backend_mod.select_backend(settings)
    outcome = active_backend.reflect(req)

    reinforced = _reinforce_hits(conn_a, outcome.passive_hits, active_by_name, scope_ctx)

    spend_rows = 0
    for entry in outcome.spend:
        recorded = _record_spend_for(conn_a, entry, record_id_int=None)
        spend_rows += 1 if recorded else 0

    return ReflectResult(
        reinforced=reinforced,
        proposed_promotions=list(outcome.promotion_candidates),
        spend_rows=spend_rows,
    )


def forkb_rows(
    conn_b: sqlite3.Connection, session_id: str, settings: config.Settings
) -> list[sqlite3.Row]:
    """Thin wrapper over ``store.forkb.rows_for_session`` (kept for the test seam)."""
    from claudemem.store import forkb

    return forkb.rows_for_session(conn_b, session_id, settings=settings)


def _reinforce_hits(
    conn_a: sqlite3.Connection,
    passive_hits: list[PassiveHit],
    active_by_name: dict[str, forka.Record],
    scope_ctx: config.ScopeContext,
) -> int:
    """Reinforce confirmed passive hits in the index + flush to frontmatter (``SC-9``).

    Each hit's ``record_id`` is an ``a:<name>`` id; we strip the prefix and
    re-validate it resolves to a real **active** Fork A record (an out-of-set /
    superseded id is silently ignored, §5.3). For each valid hit: bump
    ``HitCount`` and recency in the INDEX, then flush ``hit_count`` to frontmatter
    via :func:`files.writeback_counters` and clear the ``stale`` flag via
    :func:`files.set_stale` (``SC-13`` — a confirmed hit re-establishes trust).
    A hand-deleted file (writeback returns ``None``) is not an error (``C-11``).
    Returns the count of reinforced records.
    """
    reinforced = 0
    seen: set[str] = set()
    for hit in passive_hits:
        name = _strip_a_prefix(hit.record_id)
        if name is None or name in seen:
            continue
        record = active_by_name.get(name)
        if record is None:
            # Out-of-set or superseded id — drop it (§5.3 validation).
            continue
        seen.add(name)

        new_hit_count = record.hit_count + 1
        _bump_hit_index(conn_a, record.id, new_hit_count)

        target_file = _read_target_file(scope_ctx, name)
        if target_file is None:
            continue
        updated = files.writeback_counters(target_file, hit_count=new_hit_count)
        if updated is None:
            continue
        files.set_stale(updated, False)
        reinforced += 1
    return reinforced


def _bump_hit_index(conn_a: sqlite3.Connection, record_id: int, new_hit_count: int) -> None:
    """Bump ``HitCount`` + recency in the INDEX only (``SC-9`` confirmed-hit reinforce).

    The confirmed-hit reinforcement write (distinct from ``recall.get``'s access
    bump): set ``HitCount`` to ``new_hit_count`` for ``record_id``. The file
    frontmatter flush is the caller's next step (:func:`files.writeback_counters`).
    Runs under :func:`~claudemem.index.write_tx` (``BEGIN IMMEDIATE``, ``C-18``)
    so the §3.3 FTS sync triggers fire as on any ``Record`` UPDATE.
    """
    from claudemem import index

    with index.write_tx(conn_a):
        conn_a.execute(
            "UPDATE Record SET HitCount = ? WHERE Id = ?;",
            (new_hit_count, record_id),
        )


def _strip_a_prefix(record_id: str) -> str | None:
    """Strip the ``a:`` Fork A id prefix; return the bare ``name`` (or ``None``).

    The reflection schema's ``passive_hits[].record_id`` is an ``a:<name>`` id
    (§8.2). A value missing the prefix (or naming Fork B, ``b:``) is not a Fork A
    record_id and is dropped — only Fork A records are reinforced (``IN-6``).
    """
    if record_id.startswith("a:"):
        return record_id[2:] or None
    return None


__all__ = [
    "ConflictReport",
    "DeferralReport",
    "EnrichBatchResult",
    "ReflectResult",
    "enrich_batch",
    "reflect",
]
