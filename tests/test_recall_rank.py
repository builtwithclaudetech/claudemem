"""Tests for claudemem.recall.rank (T3.1) — the model-free ranking math (§4).

Covers: the logistic ``normalized_relevance`` is monotonic in the raw bm25 score
and bounded ``(0, 1)`` with exact spot-checks at the center and a known offset;
single-factor salience monotonicity (higher relevance / higher importance /
lower age each raise salience — IN-19); ``recency_decay == 1.0`` for pinned
records (tech-design §4.4 rule 1); and the IN-19 *done-condition* — an old pinned
record outranks a fresh, otherwise-higher-salience unpinned record via the shared
pinned-first ordering (§4.4 rule 2).

``rank`` is pure (no DB, no I/O); these tests construct ``Record`` values and
``RankingSettings`` directly with no fixtures or temp dirs.
"""

from __future__ import annotations

import math

import pytest

from claudemem.config import RankingSettings
from claudemem.recall import rank
from claudemem.store.forka import Record

NOW = 1_700_000_000  # arbitrary fixed "now" epoch for deterministic decay math.


def _record(*, name: str = "r", importance: int = 3, pinned: bool = False) -> Record:
    """Minimal Record carrying only the fields ranking touches."""
    return Record(
        id=1,
        name=name,
        scope="project",
        project_id="p",
        type="fact",
        importance=importance,
        pinned=1 if pinned else 0,
        source="user",
        created=NOW,
        last_accessed=NOW,
        access_count=0,
        hit_count=0,
        summary=None,
        aliases_json=None,
        aliases_flat=None,
        superseded_by=None,
        stale=0,
        enrich_pending=0,
        body="",
    )


# --------------------------------------------------------------------------- #
# normalized_relevance — logistic                                              #
# --------------------------------------------------------------------------- #


def test_normalized_relevance_bounded() -> None:
    # The §4.2 logistic is bounded [0, 1). Across the realistic bm25 score range
    # it stays strictly in (0, 1). At absurdly negative raw bm25 (a score no real
    # corpus produces, ~ -2860+) IEEE-754 exp() underflows to 0.0 and the value
    # saturates to exactly 1.0 — still within [0, 1] and never below 0, so floors
    # (which live well below saturation, §4.5) are unaffected.
    for raw in (-50.0, -6.0, 0.0, 50.0, 1000.0):
        val = rank.normalized_relevance(raw)
        assert 0.0 < val < 1.0
    # Saturation boundary: extreme match never exceeds 1.0 and never goes < 0.
    assert 0.0 < rank.normalized_relevance(-1.0e6) <= 1.0


def test_normalized_relevance_monotonic_in_score() -> None:
    # bm25 is non-positive; a *more negative* raw (better match) → larger score
    # after sign-flip → higher relevance. So relevance strictly decreases as raw
    # increases toward 0 and beyond.
    raws = [-30.0, -20.0, -10.0, -6.0, 0.0, 10.0]
    vals = [rank.normalized_relevance(r) for r in raws]
    assert vals == sorted(vals, reverse=True)
    assert all(a > b for a, b in zip(vals, vals[1:], strict=False))


def test_normalized_relevance_center_is_half() -> None:
    # score = -raw = center (6.0) => logistic = 0.5 exactly.
    assert rank.normalized_relevance(-rank.LOGISTIC_CENTER) == pytest.approx(0.5)


def test_normalized_relevance_spot_value() -> None:
    # Exact logistic at a known offset: score = 6.0 + (1/slope) above center
    # gives 1/(1+e^-1). raw = -(center + 1/slope).
    raw = -(rank.LOGISTIC_CENTER + 1.0 / rank.LOGISTIC_SLOPE)
    expected = 1.0 / (1.0 + math.exp(-1.0))
    assert rank.normalized_relevance(raw) == pytest.approx(expected)


# --------------------------------------------------------------------------- #
# recency_decay                                                                #
# --------------------------------------------------------------------------- #


def test_recency_decay_half_life() -> None:
    s = RankingSettings()
    last = NOW - s.recency_half_life_days * 86400  # exactly one half-life old
    decay = rank.recency_decay(
        last_accessed_epoch=last,
        now_epoch=NOW,
        half_life_days=s.recency_half_life_days,
        pinned=False,
    )
    assert decay == pytest.approx(0.5)


def test_recency_decay_pinned_is_one_regardless_of_age() -> None:
    s = RankingSettings()
    ancient = NOW - 10 * s.recency_half_life_days * 86400
    decay = rank.recency_decay(
        last_accessed_epoch=ancient,
        now_epoch=NOW,
        half_life_days=s.recency_half_life_days,
        pinned=True,
    )
    assert decay == 1.0


# --------------------------------------------------------------------------- #
# salience — single-factor monotonicity (IN-19)                                #
# --------------------------------------------------------------------------- #


def _sal(
    *,
    relevance: float = 0.8,
    importance: int = 3,
    last_accessed: int = NOW,
    pinned: bool = False,
    settings: RankingSettings | None = None,
) -> float:
    return rank.salience(
        normalized_relevance=relevance,
        importance=importance,
        last_accessed_epoch=last_accessed,
        now_epoch=NOW,
        pinned=pinned,
        settings=settings or RankingSettings(),
    )


def test_salience_increases_with_relevance() -> None:
    assert _sal(relevance=0.9) > _sal(relevance=0.4)


def test_salience_increases_with_importance() -> None:
    assert _sal(importance=5) > _sal(importance=2)


def test_salience_increases_with_recency() -> None:
    fresh = _sal(last_accessed=NOW)
    stale = _sal(last_accessed=NOW - 180 * 86400)
    assert fresh > stale


def test_salience_normalized_importance_curve_range() -> None:
    # importance 5 -> *1.0, importance 1 -> *0.2 with relevance=1, fresh, decay=1.
    top = _sal(relevance=1.0, importance=5, last_accessed=NOW)
    bottom = _sal(relevance=1.0, importance=1, last_accessed=NOW)
    assert top == pytest.approx(1.0)
    assert bottom == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# ranking_key / sort_by_ranking — pinned-first (IN-19 done-condition)          #
# --------------------------------------------------------------------------- #


def test_pinned_outranks_higher_salience_unpinned_after_decay() -> None:
    # The IN-19 done-condition: an OLD pinned record vs a FRESH unpinned record
    # whose computed salience is higher must still sort pinned-first.
    s = RankingSettings()
    old_pinned = _record(name="pin", importance=3, pinned=True)
    fresh_unpinned = _record(name="fresh", importance=5, pinned=False)

    pin_old_epoch = NOW - 365 * 86400  # a year stale
    pin_sal = rank.salience(
        normalized_relevance=0.5,
        importance=old_pinned.importance,
        last_accessed_epoch=pin_old_epoch,
        now_epoch=NOW,
        pinned=True,
        settings=s,
    )
    fresh_sal = rank.salience(
        normalized_relevance=0.95,
        importance=fresh_unpinned.importance,
        last_accessed_epoch=NOW,
        now_epoch=NOW,
        pinned=False,
        settings=s,
    )
    # Sanity: the unpinned record genuinely has higher raw salience.
    assert fresh_sal > pin_sal

    ordered = rank.sort_by_ranking(
        [(fresh_unpinned, fresh_sal), (old_pinned, pin_sal)]
    )
    assert ordered[0][0] is old_pinned  # pinned-first wins despite lower salience


def test_sort_breaks_ties_by_salience_desc_within_pin_group() -> None:
    a = _record(name="a", pinned=False)
    b = _record(name="b", pinned=False)
    ordered = rank.sort_by_ranking([(a, 0.2), (b, 0.8)])
    assert [r.name for r, _ in ordered] == ["b", "a"]


def test_ranking_key_pinned_rank() -> None:
    pinned = _record(pinned=True)
    unpinned = _record(pinned=False)
    assert rank.ranking_key(pinned, 0.1)[0] == 0
    assert rank.ranking_key(unpinned, 0.9)[0] == 1
