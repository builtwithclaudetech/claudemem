"""claudemem.recall.search — the ``search`` read command (L2, model-free).

The salience-ranked recall flow (PRD IN-3, architecture §2.5/§5.2, tech-design
§4.3): take a free-text query, run the model-free FTS5 candidate prefilter
(``store.fts_candidates``, ``candidate_k = 64``, scope-merged global + active
project, SC-12), re-rank the ≤64 candidates in Python by the §4.4 salience
formula (``recall.rank``), drop sub-floor hits, and serialize id-first text (or
``--json`` JSONL) via ``recall.output``. When nothing clears the relevance floor
— or the query has no usable MATCH token — fall back to the Fork B activity
archive, labelling each hit ``[archive]`` with a ``b:<rowid>`` id (IN-3).

**Firewall (architecture §2.5, §4; SC-6/SC-2 — load-bearing).** This module
imports only ``recall`` (``rank``/``output``), ``store`` (``forka``), ``config``,
and the standard library (``sqlite3`` for the passed connections only). It NEVER
imports ``enrich`` or ``anthropic``, never spawns ``claude``, never constructs a
model request. It is the SC-2 cold path's protected side and a missing API key
changes nothing here (``search`` is model-free by construction, SC-3).

**Connection ownership.** Per architecture §2.5 ``recall`` does not open SQLite
(``index`` is below it and outside its import set); the caller (``cli``/``hooks``,
Phase 5) opens ``forkA``/``forkB`` and threads the connections in. ``conn_a`` is
the Fork A index; ``conn_b`` is the Fork B archive. The Phase-5 cli must always
pass a **live** ``conn_b`` to :func:`search`: the archive fallback can fire on any
query (any below-floor or token-less query routes to Fork B), so ``conn_b`` is
not optional and must not be opened lazily on the fallback branch alone.
"""

from __future__ import annotations

import json as _json
import re
import sqlite3
import time

from claudemem import config
from claudemem.recall import output, rank
from claudemem.store import forka, forkb

#: Hard cap on the number of in-window Fork B rows scanned for the archive
#: fallback. The fallback is a cheap lexical backstop, not a second index, so the
#: scan is bounded (most-recent first) to keep the SC-2 path thin even when the
#: 45-day window holds many rows.
_ARCHIVE_SCAN_LIMIT = 500

#: How many archive hits to surface — same order of magnitude as a Fork A result
#: page; the fallback is a coarse "nothing curated matched, here is the raw log"
#: signal, not an exhaustive search.
_ARCHIVE_RESULT_LIMIT = 20

#: Word-token splitter for the archive scan — the same "split on any non-word
#: run" rule the FTS MATCH builder uses (store.forka._TOKEN_SPLIT), so a
#: punctuation-only query (``!!!``) yields no token here exactly as it yields no
#: FTS MATCH token, and both paths agree on "no usable token → recent rows".
_TOKEN_SPLIT = re.compile(r"\W+", re.UNICODE)


def search(
    conn_a: sqlite3.Connection,
    conn_b: sqlite3.Connection,
    query: str,
    scope_ctx: config.ScopeContext,
    *,
    json: bool = False,
    now_epoch: int | None = None,
    settings: config.Settings | None = None,
) -> str:
    """Salience-ranked recall with Fork A → Fork B archive fallback (IN-3).

    Flow (architecture §5.2): ``store.fts_candidates`` (FTS5 MATCH, OR-ed quoted
    tokens, ``candidate_k = 64``, scope merge SC-12) → per-candidate
    ``rank.normalized_relevance`` + ``rank.salience`` → ``rank.sort_by_ranking``
    (pinned-first, salience desc) → drop salience < ``salience_floor`` (0.05).

    **A→B fallback (IN-3, §4.3):** if there are no usable Fork A hits — because
    the query yielded no MATCH token (empty / punctuation), or the top surviving
    candidate's ``normalized_relevance`` is below ``relevance_floor`` (0.30) —
    fall back to a bounded lexical scan of the in-window Fork B ``Activity`` rows
    (:func:`_archive_fallback`), emitting ``b:<rowid>`` ids each prefixed
    ``[archive]``. Otherwise serialize the Fork A hits via
    :func:`output.serialize_search` (human id-first text by default, JSONL when
    ``json``).

    ``now_epoch`` (defaults to the current UTC epoch seconds) and ``settings``
    (defaults to :func:`config.load_config`) are injectable so the salience clock
    and the floors are deterministic under test.
    """
    if settings is None:
        settings = config.load_config()
    if now_epoch is None:
        now_epoch = int(time.time())
    ranking = settings.ranking

    candidates = forka.fts_candidates(conn_a, query, scope_ctx, k=ranking.candidate_k)

    if not candidates:
        # No usable MATCH token (empty / punctuation-only query) → §4.3 says treat
        # as a relevance-floor miss and go straight to the Fork B archive.
        return _archive_fallback(conn_b, query, settings, now_epoch=now_epoch, json=json)

    by_id = {rec.id: rec for rec in forka.active_set(conn_a, scope_ctx)}

    scored: list[tuple[forka.Record, float]] = []
    relevances: dict[int, float] = {}
    for cand_id, raw_bm25 in candidates:
        record = by_id.get(cand_id)
        if record is None:
            # A candidate that is no longer in the active set (raced supersede);
            # skip it rather than crash — the active set is the source of truth.
            continue
        rel = rank.normalized_relevance(raw_bm25)
        relevances[record.id] = rel
        sal = rank.salience(
            normalized_relevance=rel,
            importance=record.importance,
            last_accessed_epoch=record.last_accessed,
            now_epoch=now_epoch,
            pinned=bool(record.pinned),
            settings=ranking,
        )
        scored.append((record, sal))

    ranked = rank.sort_by_ranking(scored)
    surviving = [rec for rec, sal in ranked if sal >= ranking.salience_floor]

    # A→B fallback (§4.3, IN-3): nothing survived the salience floor, or the best
    # surviving hit's *relevance* is below the relevance floor → the curated store
    # had no good match, so serve the raw activity archive instead.
    if not surviving or relevances[surviving[0].id] < ranking.relevance_floor:
        return _archive_fallback(conn_b, query, settings, now_epoch=now_epoch, json=json)

    return output.serialize_search(surviving, json=json)


# --------------------------------------------------------------------------- #
# Fork B archive fallback (IN-3, §4.3, §5.2)                                    #
# --------------------------------------------------------------------------- #
#
# Fork B has NO FTS index (forkb.py is a model-free write/maintenance layer;
# architecture §2.4). The fallback is therefore a deliberately simple, bounded
# lexical scan over the in-window ``Activity`` bodies: case-insensitive substring
# (SQL ``LIKE``) of each query token against the stored ``Body``, OR-ed across
# tokens, most-recent first, capped at ``_ARCHIVE_SCAN_LIMIT`` rows examined and
# ``_ARCHIVE_RESULT_LIMIT`` rows returned. Tool-output rows (``Body IS NULL`` —
# only a ``ToolRef`` was kept, §3.5) are excluded. This is a coarse "nothing
# curated matched, here is the recent raw log" backstop, not a second ranked
# index, so it does not compute salience and is intentionally cheap.


def _archive_tokens(query: str) -> list[str]:
    """Split a free-text query into lowercase literal tokens for the LIKE scan.

    Splits on any non-word run (matching the FTS MATCH tokenizer) and lowercases;
    the tokens feed parameterized ``LIKE`` patterns (``%token%``) so there is no
    SQL-injection surface (values are bound, never interpolated). Returns ``[]``
    for an empty / whitespace-only / punctuation-only query — which routes the
    fallback to the most-recent rows, mirroring the empty-MATCH case (§4.3).
    """
    return [t.lower() for t in _TOKEN_SPLIT.split(query) if t]


def _archive_fallback(
    conn_b: sqlite3.Connection,
    query: str,
    settings: config.Settings,
    *,
    now_epoch: int,
    json: bool,
) -> str:
    """Bounded lexical scan of the in-window Fork B archive (IN-3, §4.3).

    Delegates the SQL to the ``store.forkb`` read accessors (the sole persistence
    layer, architecture §2.4): :func:`forkb.archive_matching` for the
    token-LIKE scan, or :func:`forkb.archive_recent` for the no-usable-token
    fallback. Both exclude tool-output rows (``Body IS NULL``, §3.5) and are
    bounded by ``_ARCHIVE_RESULT_LIMIT``, recent-first. Each surviving row is
    emitted as a ``[archive]``-prefixed line/object with a ``b:<rowid>`` id —
    formatted here because archive rows are not Fork A
    :class:`~claudemem.store.forka.Record` values.

    An empty/punctuation query (no tokens) or no matching rows yields the empty
    string — the same "no results" signal Fork A search returns, so the caller
    (and Claude) sees a clean empty result rather than an error (SC-3).

    ``now_epoch`` is the caller's already-resolved "now" (:func:`search`'s
    ``now_epoch`` param) and governs the archive window cutoff below — it must
    stay the single clock for the whole request, not re-read from the wall clock.
    """
    tokens = _archive_tokens(query)
    cutoff = now_epoch - settings.forkb.window_days * 86400
    if not tokens:
        rows = forkb.archive_recent(
            conn_b, cutoff_epoch=cutoff, limit=_ARCHIVE_RESULT_LIMIT
        )
    else:
        rows = forkb.archive_matching(
            conn_b, tokens, cutoff_epoch=cutoff, limit=_ARCHIVE_RESULT_LIMIT
        )

    if json:
        return "\n".join(_archive_obj_line(row["Id"], row["Body"]) for row in rows)
    return "\n".join(_archive_text_line(row["Id"], row["Body"]) for row in rows)


def _archive_text_line(rowid: int, body: str) -> str:
    """One id-first human line for an archive hit: ``b:<rowid>␠[archive] <title>``."""
    return (
        f"{output.make_id_b(rowid)}{output.ID_TITLE_SEP}{output.archive_title(body)}"
    )


def _archive_obj_line(rowid: int, body: str) -> str:
    """One JSONL object for an archive hit (the ``[archive]`` JSONL counterpart).

    Mirrors the Fork A search object's id-first shape but marks the source so a
    consumer can distinguish a curated hit from a raw archive line: ``id`` is the
    ``b:<rowid>`` id, ``archive`` is ``true``, ``summary`` is the one-line title.
    """
    obj: dict[str, object] = {
        "id": output.make_id_b(rowid),
        "archive": True,
        "summary": output.archive_title(body),
    }
    return _json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


__all__ = ["search"]
