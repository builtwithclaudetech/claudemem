"""Tests for the §10.3 latency harness (T6.2).

Two distinct tiers, deliberately separated:

* **Fast structural tests** (always run) — exercise the harness LOGIC end to end
  on a TINY corpus with a tiny ``runs`` value, asserting the
  :class:`~tests.latency_harness.LatencyReport` has the right *shape* (p95
  computed by nearest-rank, per-query structure, max-p95 derivation) and that a
  real ``claudemem search`` runs as a fresh subprocess. These assert **nothing
  about the 200 ms bar** — timing is environment-dependent.

* **The certification test** (SKIPPED unless ``CLAUDEMEM_RUN_LATENCY_CERT`` is
  set) — the actual VPS-timing-dependent 50×5 cold-run cert at 2,000 records.
  It is the **morning-verify** step (tech-design §4.6 / §10.3), so the normal
  ``pytest`` suite SKIPS it. Run it deliberately with::

      CLAUDEMEM_RUN_LATENCY_CERT=1 uv run --python 3.11 pytest \\
          tests/test_latency_harness.py -q -k certification

  or, attended and standalone::

      uv run --python 3.11 python -m tests.latency_harness
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.fixtures.claudemem_2k import (
    DEFAULT_COUNT,
    NO_MATCH_TOKEN,
    REPRESENTATIVE_QUERIES,
)
from tests.latency_harness import (
    DEFAULT_RUNS,
    LATENCY_BAR_MS,
    LatencyReport,
    _p95_nearest_rank_index,
    run_latency_cert,
)

#: A tiny corpus + run count for the fast structural test — proves the harness
#: logic + the subprocess round-trip without the slow 2,000-record / 50×5 cert.
_TINY_COUNT = 20
_TINY_RUNS = 3
_TINY_QUERIES = (REPRESENTATIVE_QUERIES[0], NO_MATCH_TOKEN)


# --------------------------------------------------------------------------- #
# p95 nearest-rank: the load-bearing statistic (must be sorted[47] at n=50).    #
# --------------------------------------------------------------------------- #


def test_p95_index_is_47_at_50_runs() -> None:
    """The certifying n=50 case MUST select sorted[47] (tech-design §4.6)."""
    assert _p95_nearest_rank_index(50) == 47


@pytest.mark.parametrize(
    ("n", "expected"),
    [
        (1, 0),
        (3, 2),  # ceil(0.95*3)=ceil(2.85)=3 -> index 2
        (20, 18),  # ceil(0.95*20)=19 -> index 18
        (50, 47),  # the certification case
        (100, 94),  # ceil(0.95*100)=95 -> index 94
    ],
)
def test_p95_index_nearest_rank_generalized(n: int, expected: int) -> None:
    """Nearest-rank index is ceil(0.95*n)-1, clamped to [0, n-1]."""
    assert _p95_nearest_rank_index(n) == expected


def test_p95_index_rejects_zero_runs() -> None:
    with pytest.raises(ValueError):
        _p95_nearest_rank_index(0)


# --------------------------------------------------------------------------- #
# Fast structural test — runs the harness on a tiny corpus, no <200ms assert.   #
# --------------------------------------------------------------------------- #


def test_harness_returns_well_formed_report(tmp_path: Path) -> None:
    """The harness runs ``claudemem search`` via subprocess and reports cleanly.

    Tiny corpus + runs=3 + 2 queries. Asserts the report SHAPE only — never the
    200 ms bar (timing is environment-dependent; this test only proves the
    harness logic and that the real CLI runs end to end as a fresh subprocess).
    """
    home = tmp_path / "home"
    home.mkdir()
    report = run_latency_cert(
        home=home,
        runs=_TINY_RUNS,
        queries=_TINY_QUERIES,
        count=_TINY_COUNT,
    )

    assert isinstance(report, LatencyReport)
    assert report.corpus_count == _TINY_COUNT
    assert report.runs == _TINY_RUNS

    # One QueryLatency per query, each with the right run/sample shape.
    assert len(report.per_query) == len(_TINY_QUERIES)
    assert [q.query for q in report.per_query] == list(_TINY_QUERIES)
    for q in report.per_query:
        assert q.runs == _TINY_RUNS
        assert len(q.samples_ms) == _TINY_RUNS
        assert all(s > 0.0 for s in q.samples_ms)  # real spawn-inclusive timings
        # p95 by nearest-rank: index 2 at runs=3, and it must be a real sample.
        assert q.p95_index == _p95_nearest_rank_index(_TINY_RUNS)
        assert q.p95_ms == sorted(q.samples_ms)[q.p95_index]

    # max_p95 is the max over per-query p95s; passed is derived vs the bar.
    assert report.max_p95_ms == max(q.p95_ms for q in report.per_query)
    assert report.bar_ms == LATENCY_BAR_MS
    assert report.passed == (report.max_p95_ms < LATENCY_BAR_MS)
    assert report.p95_by_query == {q.query: q.p95_ms for q in report.per_query}


# --------------------------------------------------------------------------- #
# The certification — SKIPPED unless CLAUDEMEM_RUN_LATENCY_CERT is set.          #
# This is the VPS-timing-dependent morning-verify step (tech-design §4.6).       #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not os.environ.get("CLAUDEMEM_RUN_LATENCY_CERT"),
    reason="latency cert is morning-verify, VPS-timing-dependent "
    "(set CLAUDEMEM_RUN_LATENCY_CERT=1 to run the 50x5 cert at 2000 records)",
)
def test_sc2_latency_certification(tmp_path: Path) -> None:
    """Certify SC-2: max p95 over the 5 queries < 200 ms at ClaudeMem-2k.

    The full 50 cold runs × 5 representative queries at 2,000 records. Only runs
    when ``CLAUDEMEM_RUN_LATENCY_CERT`` is set (the normal suite skips it).
    """
    home = tmp_path / "home"
    home.mkdir()
    report = run_latency_cert(
        home=home,
        runs=DEFAULT_RUNS,
        queries=REPRESENTATIVE_QUERIES,
        count=DEFAULT_COUNT,
    )
    assert report.passed, (
        f"SC-2 FAIL: max p95 {report.max_p95_ms:.2f} ms >= {LATENCY_BAR_MS:.0f} ms "
        f"(per-query: {report.p95_by_query})"
    )
