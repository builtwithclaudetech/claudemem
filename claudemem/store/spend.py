"""claudemem.store.spend — the daemonless SpendLog ledger (L1, store layer).

Owns the ``SpendLog`` ledger in ``forkA.db`` (tech-design §3.4, architecture
§2.4 / §7.3): a per-call insert recording actual post-call usage across all four
token classes plus the ``Latency``/``RetryCount``/``Outcome`` observability
columns, and a windowed ``SUM`` *tally* over ET (``America/New_York``) day/month
bounds. There is **no resident state** (``NG-1``): the tally is recomputed from
the indexed ``SpendLog(Ts)`` range on every call.

Per architecture §2.4 this module imports ``index`` + ``config`` + the standard
library only — never a model, never ``enrich``/``recall``/``files``. The two
enrichment call sites (``save`` / ``reflect``) feed it post-call; the columns it
records never gate persistence (``SC-3``).

**Caps are warn-not-block** (``SC-10``, tech-design §5.10): :func:`near_cap_warnings`
returns advisory strings and never raises or blocks a save.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from time import time as _now
from zoneinfo import ZoneInfo

from claudemem import config, index

#: One ET calendar day — used to derive the next-day boundary by date (not by
#: a naive +86400 epoch shift, which would be wrong across a DST transition).
_ONE_DAY = timedelta(days=1)

# --------------------------------------------------------------------------- #
# Vocabularies — kept narrow + spec-pinned (tech-design §3.4)                   #
# --------------------------------------------------------------------------- #

#: ``SpendLog.CallSite`` values — the two Haiku call sites (tech-design §3.4).
CALL_SITES = ("save", "reflect")
#: ``SpendLog.Backend`` values (tech-design §3.4, §5.9).
BACKENDS = ("sdk", "cli")
#: ``SpendLog.Outcome`` values (tech-design §3.4, §5.6/§5.8).
OUTCOMES = ("ok", "deferred", "repaired")

# The single per-call insert. Columns enumerated explicitly (claude_coding §6 SQL).
# ``Ts`` is UTC epoch seconds; cache columns are always 0 in v1 (§2.3) but
# carried for the future cost model. ``IdempotencyKey`` is NULL on the CLI path.
_INSERT_SPEND_SQL = """
INSERT INTO SpendLog (
    Ts, CallSite, Model, InputTokens, OutputTokens,
    CacheWriteTokens, CacheReadTokens, IdempotencyKey, Backend,
    Latency, RetryCount, Outcome
) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""


# --------------------------------------------------------------------------- #
# Tally result + ET window computation (tech-design §3.4, plan §1)             #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class SpendTally:
    """A point-in-time spend tally (tech-design §3.4; ``IN-18``, no resident state).

    Token sums cover all four classes (input + output + cache-write +
    cache-read; the cache classes are 0 in v1, §2.3). ``cli_spawns_today`` /
    ``cli_records_today`` count CLI-backend rows in the ET day window — the
    frozen ``SpendLog`` schema stores one row per CLI call, so both CLI caps
    (``cli_daily_spawn_cap`` / ``cli_daily_record_cap``) are evaluated against
    that per-call row count (see module note in :func:`spend_tally`).
    """

    day_tokens: int
    month_tokens: int
    cli_spawns_today: int
    cli_records_today: int
    day_start_epoch: int
    month_start_epoch: int


def _et_window_bounds(now_epoch: int, tz_name: str) -> tuple[int, int, int]:
    """Compute ET day-start, month-start, and next-day-start as UTC epoch seconds.

    DST correctness (claude_coding §6 timezone footgun): a naive
    ``now - 86400`` is WRONG across a US DST transition. We instead build the ET
    *wall-clock* boundaries (00:00 ET) as timezone-aware datetimes in
    ``tz_name`` and let ``.timestamp()`` resolve each instant's real UTC offset
    — so the day that contains the spring-forward / fall-back transition gets
    its true 23- or 25-hour span, never an off-by-an-hour window.

    Returns ``(day_start, month_start, next_day_start)`` as UTC epoch seconds.
    The tally uses half-open ranges ``[day_start, next_day_start)`` and
    ``[month_start, next_day_start)``.
    """
    tz = ZoneInfo(tz_name)
    now_et = datetime.fromtimestamp(now_epoch, tz=tz)

    # Day start = 00:00 ET on the current ET calendar day.
    day_start_et = datetime.combine(now_et.date(), time.min, tzinfo=tz)
    # Month start = 00:00 ET on the first ET calendar day of the current month.
    month_start_et = datetime.combine(
        now_et.date().replace(day=1), time.min, tzinfo=tz
    )
    # Next day start = 00:00 ET on the following ET calendar day. Re-derive from
    # the date via timedelta on the *date* (not epoch) so the DST offset of the
    # following midnight is resolved independently, keeping the span correct.
    next_day_date = now_et.date() + _ONE_DAY
    next_day_start_et = datetime.combine(next_day_date, time.min, tzinfo=tz)

    return (
        int(day_start_et.timestamp()),
        int(month_start_et.timestamp()),
        int(next_day_start_et.timestamp()),
    )


# --------------------------------------------------------------------------- #
# T2.5 — record_spend + atomic EnrichPending-clear pairing (§3.4, §3.6)         #
# --------------------------------------------------------------------------- #


def _insert_spend_row(
    conn: sqlite3.Connection,
    *,
    call_site: str,
    model: str,
    backend: str,
    input_tokens: int,
    output_tokens: int,
    idempotency_key: str | None,
    latency_ms: int,
    retry_count: int,
    outcome: str,
) -> bool:
    """Insert one ``SpendLog`` row on an already-open transaction; idempotent.

    Cache columns are written as 0 (v1, §2.3). A duplicate ``IdempotencyKey``
    (SDK path) trips ``UX_SpendLog_Idem`` → ``IntegrityError``, which is caught
    and treated as already-recorded (no double-count, §5.8). Returns ``True`` if
    a row was inserted, ``False`` if it was a duplicate skip.

    Caller owns the transaction (``BEGIN IMMEDIATE`` via :func:`index.write_tx`)
    so this can be paired atomically with an ``EnrichPending`` clear (§3.6).
    """
    try:
        conn.execute(
            _INSERT_SPEND_SQL,
            (
                int(_now()),
                call_site,
                model,
                input_tokens,
                output_tokens,
                0,  # CacheWriteTokens — always 0 in v1 (§2.3)
                0,  # CacheReadTokens — always 0 in v1 (§2.3)
                idempotency_key,
                backend,
                latency_ms,
                retry_count,
                outcome,
            ),
        )
    except sqlite3.IntegrityError:
        # Duplicate IdempotencyKey: the SDK call was already recorded. Treat as a
        # clean no-op so a retried insert never double-counts (§5.8). NULL keys
        # (CLI path) never reach the partial unique index, so they never land here.
        return False
    return True


def record_spend(
    conn: sqlite3.Connection,
    *,
    call_site: str,
    model: str,
    backend: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    idempotency_key: str | None = None,
    latency_ms: int = 0,
    retry_count: int = 0,
    outcome: str = "ok",
) -> bool:
    """Record one post-call ``SpendLog`` row (T2.5; tech-design §3.4).

    All four token classes are recorded; the two cache classes are forced to 0
    in v1 (§2.3). ``backend='cli'`` callers pass ``idempotency_key=None`` (the
    ``EnrichPending`` flag is the CLI idempotency boundary, §5.8); ``backend='sdk'``
    callers pass the ``sha256`` header so a retried insert is a clean no-op.

    Opens its own ``BEGIN IMMEDIATE`` write transaction (``C-18``). Returns
    ``True`` if a row was inserted, ``False`` on an idempotent duplicate skip.
    These columns are pure telemetry and never gate persistence (``SC-3``).
    """
    with index.write_tx(conn):
        return _insert_spend_row(
            conn,
            call_site=call_site,
            model=model,
            backend=backend,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            idempotency_key=idempotency_key,
            latency_ms=latency_ms,
            retry_count=retry_count,
            outcome=outcome,
        )


def record_spend_and_clear_pending(
    conn: sqlite3.Connection,
    *,
    record_id: int | None,
    call_site: str,
    model: str,
    backend: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    idempotency_key: str | None = None,
    latency_ms: int = 0,
    retry_count: int = 0,
    outcome: str = "ok",
) -> bool:
    """Atomically clear a record's ``EnrichPending`` flag AND insert the spend row.

    Both happen in ONE ``BEGIN IMMEDIATE`` transaction (tech-design §3.6): a
    crash can no longer leave a record counted-but-unbilled or
    billed-but-still-pending — the pair flips together or not at all. If
    ``record_id`` is ``None`` (e.g. ``reflect``, which clears no ``Record`` row),
    only the spend insert runs.

    Returns ``True`` if the spend row was inserted, ``False`` on an idempotent
    duplicate skip. On a duplicate-skip the ``EnrichPending`` clear is also
    rolled back (single transaction), so a re-played SDK call neither
    double-counts nor double-clears — the prior successful pairing already
    cleared it.
    """
    with index.write_tx(conn):
        inserted = _insert_spend_row(
            conn,
            call_site=call_site,
            model=model,
            backend=backend,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            idempotency_key=idempotency_key,
            latency_ms=latency_ms,
            retry_count=retry_count,
            outcome=outcome,
        )
        if not inserted:
            # Idempotent duplicate: abandon the whole transaction so the clear
            # below never runs against an already-recorded call.
            conn.execute("ROLLBACK;")
            return False
        if record_id is not None:
            conn.execute(
                "UPDATE Record SET EnrichPending = 0 WHERE Id = ?;", (record_id,)
            )
    return True


# --------------------------------------------------------------------------- #
# T2.6 — windowed SUM tally + near-cap warnings (§3.4, §5.10)                   #
# --------------------------------------------------------------------------- #


def spend_tally(
    conn: sqlite3.Connection,
    *,
    now_epoch: int | None = None,
    tz_name: str = config.SPEND_WINDOW_TZ_DEFAULT,
) -> SpendTally:
    """Compute the ET day/month token tally via indexed range ``SUM`` (T2.6).

    No resident state (``NG-1``): ET day-start and month-start are computed in
    ``tz_name`` (DST-correct, see :func:`_et_window_bounds`), converted to
    UTC-epoch bounds, and fed to half-open range scans over ``SpendLog(Ts)``
    (served by ``IX_SpendLog_Ts``). Token sums cover all four classes (§3.4).

    ``now_epoch`` defaults to the current UTC epoch; tests pass a fixed value for
    determinism. ``tz_name`` defaults to the locked ``America/New_York`` (§3.4);
    callers with a custom ``[spend].window_tz`` pass ``settings.spend.window_tz``.

    The CLI counts are derived from CLI-backend rows in the day window. The
    frozen ``SpendLog`` schema stores one row per CLI call with no per-call
    record-count column, so ``cli_records_today == cli_spawns_today`` here; both
    CLI caps are evaluated against that per-call row count (the most
    defensible reading of the frozen §3.4 schema — see :class:`SpendTally`).
    """
    epoch = now_epoch if now_epoch is not None else int(_now())
    day_start, month_start, next_day_start = _et_window_bounds(epoch, tz_name)

    day_tokens = conn.execute(
        "SELECT COALESCE(SUM("
        "  InputTokens + OutputTokens + CacheWriteTokens + CacheReadTokens"
        "), 0) FROM SpendLog WHERE Ts >= ? AND Ts < ?;",
        (day_start, next_day_start),
    ).fetchone()[0]

    month_tokens = conn.execute(
        "SELECT COALESCE(SUM("
        "  InputTokens + OutputTokens + CacheWriteTokens + CacheReadTokens"
        "), 0) FROM SpendLog WHERE Ts >= ? AND Ts < ?;",
        (month_start, next_day_start),
    ).fetchone()[0]

    cli_rows_today = conn.execute(
        "SELECT COUNT(*) FROM SpendLog "
        "WHERE Ts >= ? AND Ts < ? AND Backend = 'cli';",
        (day_start, next_day_start),
    ).fetchone()[0]

    return SpendTally(
        day_tokens=int(day_tokens),
        month_tokens=int(month_tokens),
        cli_spawns_today=int(cli_rows_today),
        cli_records_today=int(cli_rows_today),
        day_start_epoch=day_start,
        month_start_epoch=month_start,
    )


def near_cap_warnings(tally: SpendTally, settings: config.Settings) -> list[str]:
    """Return advisory near-/over-cap warning strings (T2.6; ``SC-10`` warn-not-block).

    Pure function — NEVER raises and NEVER blocks. Warns at ``warn_fraction``
    (0.8 default) of each relevant cap and again at/over the cap, for all four
    caps: SDK daily/monthly token caps and CLI daily record/spawn caps
    (tech-design §5.10). The caller logs these; persistence is unaffected.
    """
    spend = settings.spend
    warnings: list[str] = []
    for label, used, cap in (
        ("SDK daily tokens", tally.day_tokens, spend.daily_token_cap),
        ("SDK monthly tokens", tally.month_tokens, spend.monthly_token_cap),
        ("CLI daily records", tally.cli_records_today, spend.cli_daily_record_cap),
        ("CLI daily spawns", tally.cli_spawns_today, spend.cli_daily_spawn_cap),
    ):
        if cap <= 0:
            continue
        if used >= cap:
            warnings.append(
                f"{label} at/over cap: {used:,} / {cap:,} (100%+) — "
                "enrichment defers; saves still persist (SC-10)."
            )
        elif used >= cap * spend.warn_fraction:
            pct = round(used / cap * 100)
            warnings.append(
                f"{label} near cap: {used:,} / {cap:,} ({pct}%)."
            )
    return warnings


__all__ = [
    "CALL_SITES",
    "BACKENDS",
    "OUTCOMES",
    "SpendTally",
    "record_spend",
    "record_spend_and_clear_pending",
    "spend_tally",
    "near_cap_warnings",
]
