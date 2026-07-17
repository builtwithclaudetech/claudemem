"""claudemem.recall.menu — the SessionStart ``menu`` read command (L2, model-free).

The token-budgeted session-start menu (PRD IN-11/SC-5, architecture §2.5/§5.3,
tech-design §4.3). There is **no query** at session start, so ``menu`` does NO
FTS MATCH: it scans the active set (``store.active_set``, ``IX_Record_Active``),
ranks salience-only (importance × recency, pinned-first), drops sub-floor
records, and emits compact ``id␠title`` lines (no bodies) within the SC-5 budget.
On a ``resume`` session it injects nothing (source-aware skip, IN-11).

**Firewall (architecture §2.5, §4; SC-6/SC-2).** Imports only ``recall``
(``rank``/``output``), ``store`` (``forka``), ``config``, and the standard
library. NEVER ``enrich`` / ``anthropic``; no model request, no ``claude`` spawn.

**Connection ownership.** The caller (the SessionStart ``hooks`` path) opens the
Fork A index and threads ``conn_a`` in; ``menu`` reads no other store.
"""

from __future__ import annotations

import sqlite3
import time

from claudemem import config
from claudemem.recall import output, rank
from claudemem.store import forka

#: Neutral relevance for the no-query menu (§4.3/§4.4): with no query text there
#: is nothing to MATCH, so the relevance factor is 1.0 and salience reduces to
#: ``importance/5 × recency_decay`` (pinned ⇒ decay 1.0), pinned-first.
_NEUTRAL_RELEVANCE = 1.0

#: SC-5 hard char ceiling for the whole emitted menu block.
_CHAR_CEILING = 10_000

#: Token estimator divisor: ~4 characters per token. A deliberately simple,
#: dependency-free heuristic (no tokenizer) that over- rather than under-counts
#: for typical English, so the ≤600-token SC-5 budget is respected conservatively.
_CHARS_PER_TOKEN = 4

#: The ``source`` value for a resumed session — IN-11 says skip injection.
_RESUME_SOURCE = "resume"


def menu(
    conn_a: sqlite3.Connection,
    scope_ctx: config.ScopeContext,
    source: str | None = None,
    *,
    now_epoch: int | None = None,
    settings: config.Settings | None = None,
) -> str:
    """The session-start menu: salience-only, pinned-first, SC-5-budgeted (IN-11).

    Source-aware (IN-11): ``source == 'resume'`` → return the empty string (no
    menu injection on a resumed session). Otherwise:

    1. ``store.active_set`` — the scope-merged active records (NO FTS MATCH).
    2. Per record, ``rank.salience`` with a **neutral relevance of 1.0** (no query
       at session start, §4.3/§4.4) so salience is ``importance/5 × recency``
       (pinned ⇒ recency 1.0); ``rank.sort_by_ranking`` orders pinned-first then
       salience desc.
    3. Drop salience < ``salience_floor`` (0.05).
    4. Apply all three SC-5 caps (:func:`_apply_caps`): ≤ ``max_entries`` (30)
       entries AND ≤ ``token_ceiling`` (600) estimated tokens AND ≤ 10,000 chars,
       keeping the most-salient entries that fit.
    5. Serialize via :func:`output.serialize_menu` (compact ``id␠title`` lines,
       no bodies).

    ``now_epoch`` / ``settings`` are injectable for deterministic ordering and
    budgets under test (default: current UTC epoch / :func:`config.load_config`).
    """
    if source == _RESUME_SOURCE:
        return ""

    if settings is None:
        settings = config.load_config()
    if now_epoch is None:
        now_epoch = int(time.time())
    ranking = settings.ranking

    scored: list[tuple[forka.Record, float]] = []
    for record in forka.active_set(conn_a, scope_ctx):
        sal = rank.salience(
            normalized_relevance=_NEUTRAL_RELEVANCE,
            importance=record.importance,
            last_accessed_epoch=record.last_accessed,
            now_epoch=now_epoch,
            pinned=bool(record.pinned),
            settings=ranking,
        )
        if sal >= ranking.salience_floor:
            scored.append((record, sal))

    ranked = rank.sort_by_ranking(scored)
    entries = [
        (output.make_id(record), record.summary or record.name)
        for record, _sal in ranked
    ]
    capped = _apply_caps(entries, settings.menu)
    return output.serialize_menu(capped)


def _apply_caps(
    entries: list[tuple[str, str]], menu_settings: config.MenuSettings
) -> list[tuple[str, str]]:
    """Keep the most-salient entries that satisfy all three SC-5 caps (IN-11).

    ``entries`` arrive most-salient-first. We add them one at a time and stop at
    the first entry that would breach any of: the ``max_entries`` count (30), the
    ``token_ceiling`` estimated-token budget (600, ``chars/4``), or the 10,000-char
    hard ceiling. The estimate is taken against the **actually emitted** text
    (the ``id␠title`` line plus the joining newline) so the returned list, once
    serialized by :func:`output.serialize_menu`, is guaranteed within every cap.
    Truncation drops the least-salient entries (those that did not fit), never
    re-orders.
    """
    kept: list[tuple[str, str]] = []
    char_total = 0
    for eid, title in entries:
        if len(kept) >= menu_settings.max_entries:
            break
        line = f"{eid}{output.ID_TITLE_SEP}{title}"
        # Newline join cost: every line after the first adds one separator char.
        added_chars = len(line) + (1 if kept else 0)
        new_char_total = char_total + added_chars
        if new_char_total > _CHAR_CEILING:
            break
        if _estimate_tokens(new_char_total) > menu_settings.token_ceiling:
            break
        kept.append((eid, title))
        char_total = new_char_total
    return kept


def _estimate_tokens(char_count: int) -> int:
    """Estimate token count from a character count (~4 chars/token, ceil-rounded).

    Dependency-free and intentionally conservative (rounds up) so the ≤600-token
    SC-5 budget is never under-counted: ``ceil(char_count / 4)``.
    """
    return -(-char_count // _CHARS_PER_TOKEN)


__all__ = ["menu"]
