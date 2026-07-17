"""ClaudeMem-2k latency harness — certifies ``SC-2`` (tech-design §4.6 / §10.3).

This module is the **spawn-inclusive cold-start latency certifier** for the
`claudemem search` read path. It discharges the PRD `SC-2` caveat ("PRD sign-off
does not certify SC-2"): cold-start recall **< 200 ms** in a single CLI call,
*spawn time included*, measured as the **p95 over a fixed run count** against the
named **ClaudeMem-2k** baseline corpus (tech-design §4.6).

What the harness measures (the contract, §4.6 / §10.3)
------------------------------------------------------
* **Corpus** — ClaudeMem-2k: 2,000 active Fork A records / ~6 MB body, seeded by
  :data:`tests.fixtures.claudemem_2k.SEED` (reproducible, stdlib-only — `C-13`).
* **Runs** — **50 cold runs × 5 representative queries**
  (:data:`tests.fixtures.claudemem_2k.REPRESENTATIVE_QUERIES`).
* **Cold-start IS the measured quantity** — **no warmup is discarded**. Each of
  the 50 runs per query is a **fresh `claudemem search` process spawn** (SC-1:
  stateless, no warm process survives between runs), timed with
  :func:`time.perf_counter` around the whole `subprocess.run` call so the sample
  includes interpreter startup + import + DB open + query + exit.
* **Statistic** — **p95, nearest-rank**: at exactly 50 runs the p95 sample is
  ``sorted(times)[47]`` (index 47 of 50, 0-based). The generalized nearest-rank
  index for ``n`` runs is :func:`_p95_nearest_rank_index` (documented below); at
  ``n == 50`` it returns 47 exactly.
* **PASS gate** — **max p95 over the 5 queries < 200 ms** on the VPS.

Why a fresh subprocess per run (NOT an in-process call, NOT hyperfine)
----------------------------------------------------------------------
SC-1 says every invocation boots → SQLite + files → exits, with no background
process. The only honest way to certify SC-2 is therefore to *spawn the real
console entry point* (`claudemem search "<q>"`) once per run and time the whole
spawn. We use stdlib :mod:`subprocess` + :func:`time.perf_counter` (NOT
`hyperfine`, per §4.6) so the harness has no external dependency and the timing
boundary is exactly "the cost a Claude Code Bash call would pay".

The spawned process inherits the parent environment **plus** ``CLAUDEMEM_HOME``
pointed at the materialized corpus, so it never touches the live ``~/.claude``.

Running the certification (the user — morning verify)
-------------------------------------------------
The certification is **VPS-timing-dependent** and is the morning-verify step; it
is NOT run by the normal `pytest` suite (the pytest cert is gated behind
``CLAUDEMEM_RUN_LATENCY_CERT`` — see ``tests/test_latency_harness.py``). To run
the full 2,000-record / 50×5 certification attended, from the project root::

    uv run --python 3.11 python -m tests.latency_harness

That materializes ClaudeMem-2k into a temp home, runs 50 cold spawns × 5
queries, prints the per-query p95s + the max-p95 + PASS/FAIL vs the 200 ms bar,
and exits 0 on PASS / 1 on FAIL.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

from claudemem import config

from tests.fixtures.claudemem_2k import (
    DEFAULT_COUNT,
    REPRESENTATIVE_QUERIES,
    SEED,
    generate_corpus,
    materialize_corpus,
)

#: The PASS gate: max p95 over the 5 queries must be strictly below this, in
#: milliseconds (tech-design §4.6 / §10.3, PRD SC-2).
LATENCY_BAR_MS = 200.0

#: The certifying run count (tech-design §4.6: 50 cold runs per query).
DEFAULT_RUNS = 50

#: The percentile the gate is keyed to (95th).
PERCENTILE = 0.95


# --------------------------------------------------------------------------- #
# Report dataclasses.                                                           #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class QueryLatency:
    """The latency result for one representative query over ``runs`` cold spawns."""

    query: str
    runs: int
    #: Every per-run wall time in milliseconds, in run order (no warmup dropped).
    samples_ms: list[float]
    #: The 0-based nearest-rank index used to pick the p95 from sorted samples.
    p95_index: int
    #: The p95 latency in milliseconds — ``sorted(samples_ms)[p95_index]``.
    p95_ms: float


@dataclass(frozen=True, slots=True)
class LatencyReport:
    """The full ClaudeMem-2k latency certification result (tech-design §4.6)."""

    corpus_count: int
    runs: int
    per_query: list[QueryLatency]
    #: The maximum p95 over all queries — the value compared against the bar.
    max_p95_ms: float
    bar_ms: float
    #: True iff ``max_p95_ms < bar_ms`` (the §4.6 PASS gate).
    passed: bool
    #: Per-query (query, p95_ms) pairs, for terse logging / assertions.
    p95_by_query: dict[str, float] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# p95 nearest-rank (the load-bearing statistic).                                #
# --------------------------------------------------------------------------- #


def _p95_nearest_rank_index(n: int) -> int:
    """0-based index into ``sorted(samples)`` for the p95 nearest-rank value.

    Nearest-rank definition: the rank (1-based) of the p-th percentile over ``n``
    ordered samples is ``ceil(p * n)``; the 0-based index is that minus one,
    clamped to ``[0, n - 1]``.

    At the certifying ``n == 50`` this MUST return **47**: ``ceil(0.95 * 50) =
    ceil(47.5) = 48`` (1-based) → ``47`` (0-based) — i.e. ``sorted(times)[47]``
    exactly, per tech-design §4.6 / §10.3. The generalized form lets the fast
    structural test exercise the same code path with a tiny ``runs`` value
    (e.g. ``n == 3`` → index ``2``) without special-casing.
    """
    if n < 1:
        raise ValueError(f"runs must be >= 1, got {n}")
    # ceil(p * n) without importing math: integer ceiling of a float product.
    rank_1based = int(-(-PERCENTILE * n // 1))  # ceil via floor-negate trick
    index_0based = rank_1based - 1
    return max(0, min(index_0based, n - 1))


# --------------------------------------------------------------------------- #
# Locating + spawning the real `claudemem search` entry point.                  #
# --------------------------------------------------------------------------- #


def _resolve_claudemem_argv() -> list[str]:
    """The argv prefix that spawns the real ``claudemem`` console entry point.

    Prefers the installed ``claudemem`` console script (the production C-5 entry
    point) so the harness times exactly what a Claude Code Bash call would pay.
    Resolution order:

    1. ``claudemem`` on ``$PATH`` (``shutil.which``);
    2. a ``claudemem`` sibling of the running interpreter (the project ``.venv``
       ``bin/claudemem`` — present after ``uv pip install -e .``);
    3. fall back to ``<python> -c "from claudemem.cli import main; ...; main()"``
       which still spawns a fresh interpreter (spawn-inclusive) and runs the same
       entry function, so SC-1's "fresh process per run" holds even if the
       console script is missing.

    The fallback is a genuine cold spawn — a new ``sys.executable`` process — so
    p95 stays spawn-inclusive in every branch.
    """
    on_path = shutil.which("claudemem")
    if on_path is not None:
        return [on_path]

    sibling = Path(sys.executable).with_name("claudemem")
    if sibling.is_file():
        return [str(sibling)]

    # Last-resort cold spawn of the same entry point via the current interpreter.
    return [
        sys.executable,
        "-c",
        "import sys; from claudemem.cli import main; sys.exit(main())",
    ]


def _time_one_search(argv_prefix: list[str], query: str, home: Path) -> float:
    """Spawn ONE fresh ``claudemem search "<query>"`` and return its wall time (ms).

    A brand-new process every call (SC-1: no warm process between runs). The
    child inherits the parent env plus ``CLAUDEMEM_HOME`` → ``home`` so it reads
    the materialized corpus and never the live ``~/.claude``. Timed with
    :func:`time.perf_counter` around the whole :func:`subprocess.run` so the
    sample is spawn-inclusive (interpreter start + import + DB open + query +
    exit). A non-zero exit aborts the cert (a broken read path must not be timed
    as if it were a pass).
    """
    env = {**os.environ, config.CONFIG_HOME_ENV: str(home)}
    start = time.perf_counter()
    proc = subprocess.run(
        [*argv_prefix, "search", query],
        env=env,
        capture_output=True,
        text=True,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if proc.returncode != 0:
        raise RuntimeError(
            f"`claudemem search {query!r}` exited {proc.returncode}; "
            f"stderr:\n{proc.stderr}"
        )
    return elapsed_ms


# --------------------------------------------------------------------------- #
# The harness entry point.                                                      #
# --------------------------------------------------------------------------- #


def run_latency_cert(
    *,
    home: Path,
    runs: int = DEFAULT_RUNS,
    queries: tuple[str, ...] = REPRESENTATIVE_QUERIES,
    materialize: bool = True,
    count: int = DEFAULT_COUNT,
    seed: int = SEED,
) -> LatencyReport:
    """Run the ClaudeMem-2k latency certification and return a :class:`LatencyReport`.

    For EACH query, spawns ``runs`` fresh ``claudemem search`` subprocesses
    (SC-1: cold, spawn-inclusive, no warmup discarded), times each with
    :func:`time.perf_counter`, then computes the per-query p95 by nearest-rank
    (``sorted(samples)[_p95_nearest_rank_index(runs)]`` — index 47 at runs=50).
    The report's :attr:`LatencyReport.max_p95_ms` is the max of those, and
    :attr:`LatencyReport.passed` is ``max_p95_ms < 200`` (tech-design §4.6).

    Args:
        home: a writable directory used as ``$CLAUDEMEM_HOME`` for the corpus and
            the timed child processes. Must already exist.
        runs: cold spawns per query (50 for the certification; small for the fast
            structural test).
        queries: the representative query set to time (defaults to the 5 §4.6
            queries).
        materialize: when True (default), generate + materialize a fresh
            ClaudeMem-2k corpus into ``home``; pass False if ``home`` already
            holds a materialized corpus.
        count: active-record count for the materialized corpus (2,000 for the
            certification).
        seed: the corpus seed (defaults to the pinned :data:`SEED`).
    """
    if materialize:
        records = generate_corpus(seed=seed, count=count)
        materialize_corpus(records, home, write_files=True, seed_forkb=True)

    argv_prefix = _resolve_claudemem_argv()
    p95_index = _p95_nearest_rank_index(runs)

    per_query: list[QueryLatency] = []
    for query in queries:
        samples_ms = [
            _time_one_search(argv_prefix, query, home) for _ in range(runs)
        ]
        p95_ms = sorted(samples_ms)[p95_index]
        per_query.append(
            QueryLatency(
                query=query,
                runs=runs,
                samples_ms=samples_ms,
                p95_index=p95_index,
                p95_ms=p95_ms,
            )
        )

    max_p95_ms = max(q.p95_ms for q in per_query)
    return LatencyReport(
        corpus_count=count,
        runs=runs,
        per_query=per_query,
        max_p95_ms=max_p95_ms,
        bar_ms=LATENCY_BAR_MS,
        passed=max_p95_ms < LATENCY_BAR_MS,
        p95_by_query={q.query: q.p95_ms for q in per_query},
    )


def format_report(report: LatencyReport) -> str:
    """A human-readable, fixed-width rendering of a :class:`LatencyReport`."""
    lines = [
        "ClaudeMem-2k latency certification (tech-design §4.6 / §10.3, SC-2)",
        "=" * 70,
        f"corpus records : {report.corpus_count}",
        f"runs per query : {report.runs} cold spawns (no warmup discarded)",
        f"statistic      : p95 nearest-rank (sorted[{report.per_query[0].p95_index}]"
        f" at runs={report.runs})",
        f"bar            : max p95 < {report.bar_ms:.0f} ms",
        "-" * 70,
        f"{'query':<28}{'min(ms)':>10}{'p95(ms)':>10}{'max(ms)':>10}",
    ]
    for q in report.per_query:
        lines.append(
            f"{q.query:<28}{min(q.samples_ms):>10.2f}"
            f"{q.p95_ms:>10.2f}{max(q.samples_ms):>10.2f}"
        )
    lines.append("-" * 70)
    verdict = "PASS" if report.passed else "FAIL"
    lines.append(f"max p95        : {report.max_p95_ms:.2f} ms")
    lines.append(f"verdict        : {verdict} (bar {report.bar_ms:.0f} ms)")
    return "\n".join(lines)


def main() -> int:
    """Attended entry point: certify SC-2 against the full 2,000-record corpus.

    Materializes ClaudeMem-2k into a temp ``$CLAUDEMEM_HOME``, runs the full
    50×5 cold certification, prints the report, and returns 0 on PASS / 1 on
    FAIL. Run with::

        uv run --python 3.11 python -m tests.latency_harness
    """
    with tempfile.TemporaryDirectory(prefix="claudemem-2k-cert-") as tmp:
        home = Path(tmp) / "home"
        home.mkdir(parents=True, exist_ok=True)
        report = run_latency_cert(home=home, runs=DEFAULT_RUNS, count=DEFAULT_COUNT)
    print(format_report(report))
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
