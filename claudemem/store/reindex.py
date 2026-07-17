"""claudemem.store.reindex — Fork A rebuild/flush phase (L2; §3.9, SC-4).

The read/flush half of ``reindex`` (architecture §2.4, §5.5 PHASE A): rebuild
``forkA.db`` from the markdown files (files-as-truth, C-11/SC-4) through the
§3.9 **sidecar → checkpoint → ``os.replace`` atomic swap** sequence. This phase
is **model-free** and sits on the read-path side of the SC-6/C-17 firewall
(architecture §4): it MUST NOT import ``enrich`` or ``anthropic``. The
enrichment-backfill phase (PHASE B) is a separate Phase-5 ``cli`` concern and is
the only enrich-importing reindex phase.

Module discipline (architecture §2.4 + §4): imports ``index``, ``store.forka``,
``files``, ``config``, and the standard library only.

**The §3.9 five-step sequence (the load-bearing crash-safety + orphan-WAL fix).**

1. **Sidecar build** — open ``forkA.db.rebuild`` (a fresh file next to
   ``forkA.db``) with the full DDL + standard PRAGMAs and the larger cold-path
   ``cache_size`` (§3.6 D — reindex is not SC-2-bound), then populate it from
   :func:`files.iter_records` via :func:`store.forka.upsert_record`. The live
   ``forkA.db`` is never touched here, so concurrent ``search`` stays
   readable lock-free against the old index until the swap.
2. **Checkpoint + clean close (sidecar)** — ``wal_checkpoint(TRUNCATE)`` on the
   sidecar then close it, so no ``-wal``/``-shm`` remain on the rebuild file.
3. **Checkpoint the live DB before the swap** — ``wal_checkpoint(TRUNCATE)`` on
   the **live** ``forkA.db`` to drain its WAL into the main file, then close the
   live handle. *This step is the orphan-WAL hazard fix* (see below).
4. **Atomic swap** — ``os.replace`` the sidecar over ``forkA.db`` (atomic on a
   POSIX same-filesystem rename).
5. **Cleanup of the old DB's WAL/SHM** — best-effort remove of the *old*
   ``forkA.db-wal`` / ``forkA.db-shm`` (ignore ``FileNotFoundError``). A crashed
   prior run's orphan ``forkA.db.rebuild`` is removed at the *start* of the next
   rebuild.

**Why step 3 + step 5 close the corruption hazard (SC-3/SC-4).** ``os.replace``
swaps the *main* DB file by inode but does not touch the sibling ``-wal``/
``-shm``. If the live ``forkA.db`` still had an undrained ``-wal`` at swap time,
that WAL — keyed to the *old* inode's pages — would be left next to the *new*
inode; the next open would try to replay a mismatched WAL onto the fresh DB:
silent corruption or a hard open error. Draining the live WAL first (step 3) and
removing any stale sidecar files afterward (step 5) makes the swapped file
self-consistent with no WAL to replay.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, replace
from pathlib import Path

from claudemem import config, files, index
from claudemem.store import forka


@dataclass(frozen=True, slots=True)
class RebuildResult:
    """Outcome of a :func:`rebuild_index` run (for the caller to report).

    ``records`` is the number of Fork A files rebuilt into the fresh index;
    ``enrich_pending`` is how many of those lacked a summary and were flagged
    ``EnrichPending=1`` for the Phase-B backfill (architecture §5.5).
    """

    records: int
    enrich_pending: int


def _sidecar_paths() -> tuple[Path, Path]:
    """Return ``(live_forkA_path, sidecar_rebuild_path)`` for the current home."""
    live = index.forka_path()
    sidecar = live.with_name(live.name + ".rebuild")
    return live, sidecar


def _remove_db_siblings(db_path: Path) -> None:
    """Best-effort remove ``<db>``, ``<db>-wal``, ``<db>-shm`` (ignore absence).

    Used both to clear a crashed prior run's orphan sidecar at the start of a
    rebuild (step 0) and to clean the *old* DB's stale WAL/SHM after the swap
    (step 5). ``FileNotFoundError`` is the expected, ignored case.
    """
    base = os.fspath(db_path)
    for suffix in ("", "-wal", "-shm"):
        try:
            os.remove(base + suffix)
        except FileNotFoundError:
            pass


def rebuild_index(
    scope_ctx: config.ScopeContext, *, settings: config.Settings | None = None
) -> RebuildResult:
    """Rebuild ``forkA.db`` from the markdown files via the §3.9 swap (SC-4).

    Implements the §3.9 five-step sidecar → checkpoint → ``os.replace`` sequence
    exactly. Every Fork A file enumerated by :func:`files.iter_records` is
    re-derived into a fresh sidecar at the current schema (§3.13: forkA "migrates"
    by reindex-from-files), reconstructing every IN-1 column from the file — all
    three alias forms and the ISO → epoch timestamp crossing — so this is the
    SC-4 files-as-truth round-trip. A file the user hand-deleted is simply absent from
    ``iter_records`` and so omitted from the rebuild (C-11/NG-7).

    **Counter/stale flush (IN-10).** ``hit_count``/``last_accessed``/
    ``access_count`` and the ``stale`` flag come straight from the file
    frontmatter (files are truth), so the rebuild inherently flushes file → index
    for *all* records. The index → file writeback of live session counters is a
    SessionEnd / Phase-5 ``cli`` concern handled outside this model-free phase.

    **Staleness-horizon sweep (SC-13/IN-16/IN-10).** For every record the rebuild
    *recomputes* the ``stale`` trust flag against the configured horizon
    (``settings.staleness.horizon_days``): a record whose ``last_accessed`` is more
    than ``horizon_days`` in the past becomes stale. The sweep is **monotonic** —
    it only *sets* stale, never clears it (clearing is the job of a confirmed
    ``used`` hit per SC-13/``forka.reinforce``), so a record already ``stale:
    true`` stays stale and a record within the horizon keeps its authored value.
    Because the sweep never clears stale, a record whose ``last_accessed`` is
    hand-edited back inside the horizon while ``stale: true`` is left in place keeps
    the flag until a confirmed ``used``/reflection hit clears it — reindex does not
    un-stale it. This is the SC-13/IN-16 reading (clear-on-confirmed-hit-only); it
    narrows IN-10's literal "recompute for all records" to "recompute the *set*
    direction on reindex, clear on hit."
    The recomputed flag is upserted into the fresh index and, **on a not-stale →
    stale flip only**, flushed back to the markdown frontmatter via
    :func:`files.set_stale` (IN-10/IN-16/SC-4); a file the user hand-deleted mid-rebuild
    yields ``None`` from ``set_stale`` and is tolerated without error. The flush
    stays inside the sidecar-build loop, before the atomic swap, consistent with
    files-as-truth (this is a pure file write, independent of the §3.9 swap, so it
    carries no WAL/swap hazard). ``now`` is sampled once via stdlib ``time.time``,
    which keeps this phase model-free (the §10.2 firewall forbids only
    ``enrich``/``anthropic`` imports, not the standard library).

    **Re-enrichment marking.** A record whose file has no ``summary`` was saved
    while degraded, so it is flagged ``EnrichPending=1`` for the Phase-B backfill
    to find (architecture §5.5); a record that already has a summary is left
    ``EnrichPending=0``. The backfill itself (the only enrich-importing phase) is
    NOT part of this module.

    ``settings`` supplies the staleness horizon for the sweep; when ``None`` it
    is resolved via :func:`config.load_config` (matching the recall modules'
    idiom). No ranking/enrich is performed here.

    Returns a :class:`RebuildResult` with the rebuilt-record and enrich-pending
    counts for the caller to report.
    """
    if settings is None:
        settings = config.load_config()
    horizon_seconds = settings.staleness.horizon_days * 86400
    now = int(time.time())

    live_path, sidecar_path = _sidecar_paths()

    # Step 0 — clear any orphan sidecar (+ its -wal/-shm) left by a crashed run.
    _remove_db_siblings(sidecar_path)

    records = 0
    enrich_pending = 0

    # A file's own frontmatter ``scope`` decides the row's Scope/ProjectId (a
    # global-dir file is a global record with ProjectId NULL; a project-dir file
    # belongs to the enumerated project). ``iter_records`` interleaves both dirs,
    # so we map each file onto the matching scope_ctx rather than stamping the
    # whole batch with the passed (project) context.
    global_ctx = config.ScopeContext(
        kind="global",
        project_id=None,
        global_dir=scope_ctx.global_dir,
        project_dir=None,
    )

    # Step 1 — sidecar build. Open the fresh rebuild DB on the cold path (larger
    # cache, §3.6 D) and populate it from files. The live forkA.db is NOT opened
    # here, so it stays readable lock-free until the swap.
    sidecar_conn = index.open_forkA_at(sidecar_path, cold_path=True)
    try:
        for record_file in files.iter_records(scope_ctx):
            row_ctx = global_ctx if record_file.scope == "global" else scope_ctx
            # Staleness-horizon sweep (SC-13/IN-16): a record past the horizon
            # becomes stale; the sweep only sets stale, never clears it.
            last_accessed_epoch = files.iso_to_epoch(record_file.last_accessed)
            is_past_horizon = (now - last_accessed_epoch) > horizon_seconds
            if is_past_horizon and not record_file.stale:
                record_file = replace(record_file, stale=True)
                # Flush the flip back to frontmatter (IN-10/IN-16/SC-4); a file
                # hand-deleted mid-rebuild returns None and is tolerated.
                files.set_stale(record_file, True)
            pending = record_file.summary is None
            forka.upsert_record(
                sidecar_conn, record_file, row_ctx, enrich_pending=pending
            )
            records += 1
            if pending:
                enrich_pending += 1
        # Step 2 — checkpoint + clean close the sidecar (no -wal/-shm left on it).
        sidecar_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    finally:
        sidecar_conn.close()

    # Step 3 — drain the LIVE WAL before the swap (the orphan-WAL hazard fix),
    # then close the live handle. Opening here only reads/checkpoints; concurrent
    # readers up to this point used the old index. If forkA.db does not exist yet
    # (first-ever reindex) there is no live WAL to drain — open_forkA_at would
    # create an empty one, so guard on existence.
    if os.path.exists(live_path):
        live_conn = index.open_forkA_at(live_path)
        try:
            live_conn.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        finally:
            live_conn.close()

    # Step 4 — atomic swap of the sidecar over the live DB (POSIX same-fs rename).
    os.replace(sidecar_path, live_path)

    # Step 5 — best-effort cleanup of the OLD DB's stale WAL/SHM (the swapped-out
    # inode's siblings). The new forkA.db was checkpointed+closed clean in step 2,
    # so it has none of its own; any remaining -wal/-shm belong to the old inode.
    base = os.fspath(live_path)
    for suffix in ("-wal", "-shm"):
        try:
            os.remove(base + suffix)
        except FileNotFoundError:
            pass

    return RebuildResult(records=records, enrich_pending=enrich_pending)


__all__ = ["RebuildResult", "rebuild_index"]
