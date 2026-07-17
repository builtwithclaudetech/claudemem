"""claudemem.recall.rank — the model-free ranking math (L2, read path).

The pure-math half of the read path (architecture §2.5, tech-design §4): map a
raw SQLite ``bm25`` score to a bounded relevance, fold relevance × importance ×
recency into a single salience scalar, and define the **one** pinned-first
ordering that both ``search`` (salience re-rank over ≤64 FTS candidates) and
``menu`` (salience-only over the active set) reuse.

**Firewall (architecture §2.5, §4; SC-6/SC-2).** This module imports the
standard library **only** — never ``enrich``, never ``anthropic``, never the
``store``/``files``/``index`` layers. It performs no DB access and no I/O: it is
pure functions over numbers and :class:`~claudemem.store.forka.Record` values
passed in by the caller. Keeping it import-light is the SC-2 cold-path property
and the SC-6-protected side of the read-path firewall (``lint-imports`` asserts
it). The ``Record`` type used by :func:`ranking_key` is referenced only under
``TYPE_CHECKING`` so importing ``rank`` never drags in the ``store`` layer.

**Code constants vs. config (tech-design §4.2/§4.5).** The logistic *slope*
(0.35) and *center* (6.0) are calibration-coupled to the bm25 weights and stay
fixed in code — they are NOT exposed through ``[ranking]`` config. The tunable
knobs (``recency_half_life_days``, ``salience_floor``, ``relevance_floor``,
``importance_curve``, ...) live on :class:`~claudemem.config.RankingSettings`
and are consumed by the *callers* (``search``/``menu``); ``rank`` only supplies
the math.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from claudemem.config import RankingSettings
    from claudemem.store.forka import Record

# --------------------------------------------------------------------------- #
# Logistic code constants (tech-design §4.2) — NOT config-exposed.              #
# --------------------------------------------------------------------------- #

#: Logistic slope; calibration-coupled to the §4.1 bm25 weights (tech-design §4.2).
LOGISTIC_SLOPE = 0.35
#: Logistic center; the sign-flipped bm25 score that maps to relevance 0.5 (§4.2).
LOGISTIC_CENTER = 6.0
#: Seconds per day — the §4.4 epoch → ``age_days`` divisor.
SECONDS_PER_DAY = 86400.0
#: Importance normalization divisor for ``importance_curve == "normalized"`` (§4.5).
IMPORTANCE_MAX = 5.0


def normalized_relevance(raw_bm25: float) -> float:
    """Map a raw SQLite ``bm25`` value to a bounded ``[0, 1)`` relevance (§4.2).

    SQLite's ``bm25()`` is **non-positive** (more negative = better match), so we
    sign-flip to ``score = -raw_bm25`` (larger = better, §4.1) then squash with
    the fixed logistic ``1 / (1 + exp(-slope * (score - center)))``. The slope
    (0.35) and center (6.0) are code constants, not config (§4.2). The result is
    set-independent, so the ``relevance_floor`` (§4.5) is an *absolute* threshold.

    Bounded ``[0, 1)``: strictly in ``(0, 1)`` for the realistic bm25 score
    range; at an absurdly strong match (raw bm25 well below ~-2860, which no
    real corpus produces) ``exp()`` underflows and the value saturates to exactly
    ``1.0``. It never goes below 0 or above 1, so the §4.5 floors — which live
    far below saturation — are unaffected.
    """
    score = -raw_bm25
    return 1.0 / (1.0 + math.exp(-LOGISTIC_SLOPE * (score - LOGISTIC_CENTER)))


def recency_decay(
    *, last_accessed_epoch: int, now_epoch: int, half_life_days: int, pinned: bool
) -> float:
    """The §4.4 exponential recency factor (``0.5 ** (age_days / half_life)``).

    ``age_days = (now_epoch - last_accessed_epoch) / 86400.0`` (both UTC epoch
    seconds, MF-1; float division). **Pinned records short-circuit to ``1.0``**
    (tech-design §4.4 rule 1: decay can never sink a pin). A negative age (clock
    skew where ``last_accessed`` is in the future) yields a factor ``> 1.0``,
    which is harmless — it only ever helps a record's salience.
    """
    if pinned:
        return 1.0
    age_days = (now_epoch - last_accessed_epoch) / SECONDS_PER_DAY
    return float(0.5 ** (age_days / half_life_days))


def salience(
    *,
    normalized_relevance: float,
    importance: int,
    last_accessed_epoch: int,
    now_epoch: int,
    pinned: bool,
    settings: RankingSettings,
) -> float:
    """The single-factor salience scalar (tech-design §4.4, Q-4, IN-19).

    ``salience = normalized_relevance * (importance / 5.0) * recency_decay``.
    The importance term follows ``settings.importance_curve``: ``"normalized"``
    (the only supported curve, §4.5) maps importance 1→0.2 … 5→1.0 via
    ``importance / 5.0``; any other value falls back to the same normalized curve
    (forward-compatible, never raises — SC-3 posture). ``recency_decay`` uses
    ``settings.recency_half_life_days`` and is forced to ``1.0`` for pinned
    records (§4.4). This is single-factor by construction: raising any one of
    relevance, importance, or recency (lower age) raises salience monotonically,
    holding the others fixed (the IN-19 done-condition).
    """
    decay = recency_decay(
        last_accessed_epoch=last_accessed_epoch,
        now_epoch=now_epoch,
        half_life_days=settings.recency_half_life_days,
        pinned=pinned,
    )
    importance_term = importance / IMPORTANCE_MAX
    return normalized_relevance * importance_term * decay


def ranking_key(record: Record, salience_value: float) -> tuple[int, float]:
    """The **one** pinned-first sort key shared by ``search`` and ``menu`` (IN-19).

    Returns ``(pinned_rank, salience)`` where ``pinned_rank`` is ``0`` for pinned
    and ``1`` for unpinned. Callers sort **ascending by this key reversed** — see
    :func:`sort_by_ranking` — so pinned records (rank 0) sort ahead of every
    unpinned record (rank 1) regardless of computed salience, with ties broken by
    salience descending (tech-design §4.4 rule 2). Centralizing the key here means
    ``search``'s salience re-rank and ``menu``'s salience-only ordering share a
    single, audited definition of "pinned-first, then salience desc".
    """
    return (0 if record.pinned else 1, salience_value)


def sort_by_ranking(scored: list[tuple[Record, float]]) -> list[tuple[Record, float]]:
    """Order ``(record, salience)`` pairs pinned-first, then salience desc (IN-19).

    The single ordering definition reused by both ``search`` (re-rank of ≤64 FTS
    candidates) and ``menu`` (salience-only scan of the active set). Pinned rows
    come first (regardless of salience), then everything is ordered by salience
    descending; the sort is stable so equal keys keep their input order. Built on
    :func:`ranking_key`, so the pinned-first contract has exactly one home.
    """
    return sorted(
        scored,
        key=lambda pair: (ranking_key(pair[0], pair[1])[0], -ranking_key(pair[0], pair[1])[1]),
    )


__all__ = [
    "LOGISTIC_SLOPE",
    "LOGISTIC_CENTER",
    "SECONDS_PER_DAY",
    "IMPORTANCE_MAX",
    "normalized_relevance",
    "recency_decay",
    "salience",
    "ranking_key",
    "sort_by_ranking",
]
