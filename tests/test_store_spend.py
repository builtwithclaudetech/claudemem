"""Tests for claudemem.store.spend (T2.5 + T2.6).

Covers the daemonless SpendLog ledger (tech-design §3.4, §3.6, §5.8, §5.10):

* ``record_spend`` inserts a row with all columns; cache columns are 0 (§2.3).
* Idempotency — a repeated ``IdempotencyKey`` (SDK) inserts once; a NULL key
  (CLI) allows many rows (§5.8, ``UX_SpendLog_Idem``).
* ``record_spend_and_clear_pending`` flips ``EnrichPending`` AND inserts the
  spend row in one transaction (§3.6); a duplicate key rolls back both.
* ``spend_tally`` — ET day/month windowed SUM over ``SpendLog(Ts)``, including a
  US DST-transition boundary (asserting the ET day window is not off-by-an-hour),
  using a fixed ``now_epoch`` for determinism (§3.4).
* ``near_cap_warnings`` — silent below 0.8, warns near + at/over cap, never
  raises/blocks (``SC-10``).

``CLAUDEMEM_HOME`` is pointed at ``tmp_path`` so no test touches the real
``~/.claude``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from claudemem import config, index
from claudemem.store import spend

_ET = ZoneInfo("America/New_York")


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME at a tmp dir; return it."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def conn(home: Path) -> Iterator[sqlite3.Connection]:
    """An open forkA connection with schema ensured."""
    c = index.open_forkA()
    yield c
    c.close()


def _et_epoch(year: int, month: int, day: int, hour: int = 12) -> int:
    """UTC epoch seconds for a given ET wall-clock instant (DST resolved by tz)."""
    return int(datetime(year, month, day, hour, tzinfo=_ET).timestamp())


def _insert_raw(conn: sqlite3.Connection, ts: int, backend: str = "sdk",
                tokens: int = 0) -> None:
    """Insert a SpendLog row at an explicit Ts (bypasses record_spend's now())."""
    with index.write_tx(conn):
        conn.execute(
            "INSERT INTO SpendLog (Ts, CallSite, Model, InputTokens, "
            "OutputTokens, Backend) VALUES (?, 'save', 'haiku', ?, 0, ?);",
            (ts, tokens, backend),
        )


# --------------------------------------------------------------------------- #
# T2.5 — record_spend                                                          #
# --------------------------------------------------------------------------- #


def test_record_spend_inserts_all_columns(conn: sqlite3.Connection) -> None:
    inserted = spend.record_spend(
        conn,
        call_site="save",
        model="haiku",
        backend="sdk",
        input_tokens=120,
        output_tokens=45,
        idempotency_key="abc",
        latency_ms=873,
        retry_count=2,
        outcome="repaired",
    )
    assert inserted is True
    row = conn.execute(
        "SELECT CallSite, Model, InputTokens, OutputTokens, CacheWriteTokens, "
        "CacheReadTokens, IdempotencyKey, Backend, Latency, RetryCount, Outcome "
        "FROM SpendLog;"
    ).fetchone()
    assert row == ("save", "haiku", 120, 45, 0, 0, "abc", "sdk", 873, 2, "repaired")


def test_cache_columns_always_zero(conn: sqlite3.Connection) -> None:
    spend.record_spend(conn, call_site="reflect", model="haiku", backend="cli",
                       input_tokens=999, output_tokens=999)
    cw, cr = conn.execute(
        "SELECT CacheWriteTokens, CacheReadTokens FROM SpendLog;"
    ).fetchone()
    assert (cw, cr) == (0, 0)


def test_idempotent_sdk_key_inserts_once(conn: sqlite3.Connection) -> None:
    first = spend.record_spend(conn, call_site="save", model="haiku",
                               backend="sdk", idempotency_key="dup-key")
    second = spend.record_spend(conn, call_site="save", model="haiku",
                                backend="sdk", idempotency_key="dup-key")
    assert first is True
    assert second is False  # duplicate skip, no double-count
    assert conn.execute("SELECT COUNT(*) FROM SpendLog;").fetchone()[0] == 1


def test_null_key_cli_allows_multiple_rows(conn: sqlite3.Connection) -> None:
    for _ in range(3):
        assert spend.record_spend(conn, call_site="save", model="haiku",
                                  backend="cli", idempotency_key=None) is True
    assert conn.execute("SELECT COUNT(*) FROM SpendLog;").fetchone()[0] == 3


def _seed_pending_record(conn: sqlite3.Connection) -> int:
    with index.write_tx(conn):
        cur = conn.execute(
            "INSERT INTO Record (Name, Scope, Type, Source, Created, "
            "LastAccessed, EnrichPending, Body) VALUES "
            "('r1', 'global', 'note', 'save', 0, 0, 1, 'body');"
        )
    return int(cur.lastrowid)


def test_atomic_clear_pending_and_insert(conn: sqlite3.Connection) -> None:
    rec_id = _seed_pending_record(conn)
    assert conn.execute(
        "SELECT EnrichPending FROM Record WHERE Id = ?;", (rec_id,)
    ).fetchone()[0] == 1

    inserted = spend.record_spend_and_clear_pending(
        conn, record_id=rec_id, call_site="save", model="haiku",
        backend="cli", input_tokens=10, output_tokens=5,
    )
    assert inserted is True
    # Both halves flipped together (§3.6).
    assert conn.execute(
        "SELECT EnrichPending FROM Record WHERE Id = ?;", (rec_id,)
    ).fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM SpendLog;").fetchone()[0] == 1


def test_atomic_duplicate_key_rolls_back_clear(conn: sqlite3.Connection) -> None:
    rec_id = _seed_pending_record(conn)
    # First pairing succeeds and clears.
    assert spend.record_spend_and_clear_pending(
        conn, record_id=rec_id, call_site="save", model="haiku",
        backend="sdk", idempotency_key="k1",
    ) is True
    # Re-mark pending to prove the duplicate path does NOT re-clear it.
    with index.write_tx(conn):
        conn.execute("UPDATE Record SET EnrichPending = 1 WHERE Id = ?;", (rec_id,))
    second = spend.record_spend_and_clear_pending(
        conn, record_id=rec_id, call_site="save", model="haiku",
        backend="sdk", idempotency_key="k1",
    )
    assert second is False
    # Duplicate skip rolled back the clear → still pending; still only one row.
    assert conn.execute(
        "SELECT EnrichPending FROM Record WHERE Id = ?;", (rec_id,)
    ).fetchone()[0] == 1
    assert conn.execute("SELECT COUNT(*) FROM SpendLog;").fetchone()[0] == 1


def test_atomic_none_record_id_inserts_only(conn: sqlite3.Connection) -> None:
    inserted = spend.record_spend_and_clear_pending(
        conn, record_id=None, call_site="reflect", model="haiku", backend="cli",
    )
    assert inserted is True
    assert conn.execute("SELECT COUNT(*) FROM SpendLog;").fetchone()[0] == 1


# --------------------------------------------------------------------------- #
# T2.6 — spend_tally windowed SUM                                              #
# --------------------------------------------------------------------------- #


def test_tally_day_and_month_windows(conn: sqlite3.Connection) -> None:
    now = _et_epoch(2026, 3, 15, hour=14)  # mid-March ET
    _insert_raw(conn, _et_epoch(2026, 3, 15, hour=2), tokens=100)   # today
    _insert_raw(conn, _et_epoch(2026, 3, 15, hour=10), tokens=50)   # today
    _insert_raw(conn, _et_epoch(2026, 3, 14, hour=23), tokens=7)    # yesterday
    _insert_raw(conn, _et_epoch(2026, 2, 28, hour=12), tokens=1000)  # last month

    tally = spend.spend_tally(conn, now_epoch=now)
    assert tally.day_tokens == 150           # only the two same-ET-day rows
    assert tally.month_tokens == 157         # March rows: 100+50+7, not Feb's 1000


def test_tally_cli_counts(conn: sqlite3.Connection) -> None:
    now = _et_epoch(2026, 6, 10, hour=12)
    _insert_raw(conn, _et_epoch(2026, 6, 10, hour=9), backend="cli")
    _insert_raw(conn, _et_epoch(2026, 6, 10, hour=11), backend="cli")
    _insert_raw(conn, _et_epoch(2026, 6, 10, hour=8), backend="sdk")  # not CLI
    tally = spend.spend_tally(conn, now_epoch=now)
    assert tally.cli_spawns_today == 2
    assert tally.cli_records_today == 2


def test_tally_dst_spring_forward_day_window(conn: sqlite3.Connection) -> None:
    # 2026-03-08 02:00 ET springs forward to 03:00 — that ET day is only 23h.
    # A naive now-86400 would mis-place the day start by an hour. now = 10:00 ET.
    now = _et_epoch(2026, 3, 8, hour=10)
    day_start, _month, next_day = spend._et_window_bounds(now, "America/New_York")
    # Day start is exactly 00:00 ET on 2026-03-08.
    assert datetime.fromtimestamp(day_start, tz=_ET).hour == 0
    assert datetime.fromtimestamp(day_start, tz=_ET).day == 8
    # The DST day spans 23h, not 24h (the off-by-an-hour bug would make it 24h).
    assert next_day - day_start == 23 * 3600

    # A row at 01:30 ET (before the gap) is in today; 23:30 ET prior day is not.
    _insert_raw(conn, _et_epoch(2026, 3, 8, hour=1), tokens=11)
    _insert_raw(conn, _et_epoch(2026, 3, 7, hour=23), tokens=22)
    tally = spend.spend_tally(conn, now_epoch=now)
    assert tally.day_tokens == 11


def test_tally_dst_fall_back_day_window(conn: sqlite3.Connection) -> None:
    # 2026-11-01 02:00 ET falls back to 01:00 — that ET day is 25h.
    now = _et_epoch(2026, 11, 1, hour=12)
    day_start, _month, next_day = spend._et_window_bounds(now, "America/New_York")
    assert next_day - day_start == 25 * 3600


def test_tally_empty_is_zero(conn: sqlite3.Connection) -> None:
    tally = spend.spend_tally(conn, now_epoch=_et_epoch(2026, 1, 1))
    assert tally.day_tokens == 0
    assert tally.month_tokens == 0
    assert tally.cli_spawns_today == 0


# --------------------------------------------------------------------------- #
# T2.6 — near_cap_warnings (warn-not-block, never raises)                       #
# --------------------------------------------------------------------------- #


def _tally(day: int = 0, month: int = 0, cli: int = 0) -> spend.SpendTally:
    return spend.SpendTally(
        day_tokens=day, month_tokens=month,
        cli_spawns_today=cli, cli_records_today=cli,
        day_start_epoch=0, month_start_epoch=0,
    )


def test_below_warn_fraction_no_warning() -> None:
    settings = config.Settings()  # daily cap 1_000_000, warn 0.8
    assert spend.near_cap_warnings(_tally(day=500_000), settings) == []


def test_at_warn_fraction_warns() -> None:
    settings = config.Settings()
    warns = spend.near_cap_warnings(_tally(day=800_000), settings)
    assert any("near cap" in w for w in warns)


def test_over_cap_warns() -> None:
    settings = config.Settings()
    warns = spend.near_cap_warnings(_tally(day=1_200_000), settings)
    assert any("at/over cap" in w for w in warns)


def test_cli_caps_warn() -> None:
    settings = config.Settings()  # cli_daily_spawn_cap 200
    warns = spend.near_cap_warnings(_tally(cli=180), settings)
    assert any("CLI daily spawns" in w for w in warns)


def test_near_cap_never_raises_or_blocks() -> None:
    settings = config.Settings()
    # Absurd values, zero caps — must still return a list, never raise.
    huge = _tally(day=10**12, month=10**13, cli=10**6)
    result = spend.near_cap_warnings(huge, settings)
    assert isinstance(result, list)
