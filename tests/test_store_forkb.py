"""Tests for claudemem.store.forkb (T2.3 + T2.4).

Covers the model-free Fork B log path (architecture §2.4, §5.7; tech-design §3.5,
§3.7, §7.4; PRD IN-2/IN-12/SC-6/AS-4):

* ``append_activity`` under the cap (verbatim body, ``Truncated=0``);
* ``append_activity`` over the cap (head+tail with elision marker, ``Truncated=1``,
  stored length within cap, ``FullLen`` = original);
* tool-output events (``Body`` NULL, ``ToolRef`` populated);
* the per-session ``Cursor`` watermark (default 0, advance, per-session isolation);
* the 45-day window prune (old rows deleted, in-window kept, count correct,
  reclaim runs);
* ``rows_for_session`` returns only a session's in-window rows;
* SC-6: importing the module pulls in no ``anthropic`` and makes no model call.

``CLAUDEMEM_HOME`` is pointed at ``tmp_path`` so no test touches the real
``~/.claude``.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from claudemem import config, index
from claudemem.store import forkb

WINDOW_DAYS = config.FORKB_WINDOW_DAYS_DEFAULT  # 45
CAP = config.FORKB_ENTRY_CHAR_CAP_DEFAULT  # 4000
NOW = 1_700_000_000  # fixed UTC epoch for deterministic window tests


@pytest.fixture
def conn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[sqlite3.Connection]:
    """Open a fresh forkB.db under a tmp CLAUDEMEM_HOME; yield the connection."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    connection = index.open_forkB()
    yield connection
    connection.close()


def _count(conn: sqlite3.Connection) -> int:
    return int(conn.execute("SELECT COUNT(*) FROM Activity;").fetchone()[0])


def _row(conn: sqlite3.Connection, row_id: int) -> sqlite3.Row:
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT Id, SessionId, Ts, Role, Kind, Body, ToolRef, FullLen, Truncated "
        "FROM Activity WHERE Id = ?;",
        (row_id,),
    ).fetchone()
    conn.row_factory = None
    return row


# --------------------------------------------------------------------------- #
# T2.3 — append_activity                                                        #
# --------------------------------------------------------------------------- #


def test_append_under_cap_stored_verbatim(conn: sqlite3.Connection) -> None:
    body = "a short user prompt"
    row_id = forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="user", kind="prompt", body=body
    )
    row = _row(conn, row_id)
    assert row["Body"] == body
    assert row["Truncated"] == 0
    assert row["FullLen"] == len(body)
    assert row["ToolRef"] is None
    assert row["SessionId"] == "s1"
    assert row["Role"] == "user"
    assert row["Kind"] == "prompt"


def test_append_over_cap_head_tail_with_marker(conn: sqlite3.Connection) -> None:
    body = "H" * 3000 + "M" * 5000 + "T" * 3000  # 11_000 chars, well over cap
    row_id = forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="assistant", kind="text", body=body
    )
    row = _row(conn, row_id)
    stored = row["Body"]
    assert row["Truncated"] == 1
    assert row["FullLen"] == len(body)
    # Stored result stays within the cap.
    assert len(stored) <= CAP
    # Elision marker present, between a kept head and a kept tail.
    assert "chars elided" in stored
    assert stored.startswith("H")
    assert stored.endswith("T")
    # The dropped middle ('M' run) is gone.
    assert "M" * 100 not in stored
    # Head ~2800 + tail ~1000 at the 4000 cap.
    head, tail = forkb._head_tail_split(CAP)
    assert (head, tail) == (2800, 1000)
    assert stored[:head] == "H" * head
    assert stored[-tail:] == "T" * tail


def test_append_exactly_at_cap_not_truncated(conn: sqlite3.Connection) -> None:
    body = "x" * CAP
    row_id = forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="assistant", kind="text", body=body
    )
    row = _row(conn, row_id)
    assert row["Truncated"] == 0
    assert row["Body"] == body
    assert row["FullLen"] == CAP


def test_append_tool_output_skips_body(conn: sqlite3.Connection) -> None:
    body = "tool output payload " * 500
    row_id = forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="tool", kind="tool_result", body=body
    )
    row = _row(conn, row_id)
    assert row["Body"] is None
    assert row["ToolRef"] is not None
    assert row["ToolRef"].startswith("tool:tool_result ")
    assert f"len={len(body)}" in row["ToolRef"]
    assert "sha=" in row["ToolRef"]
    assert row["FullLen"] == len(body)
    assert row["Truncated"] == 0


# --------------------------------------------------------------------------- #
# T2.4 — Cursor watermark                                                       #
# --------------------------------------------------------------------------- #


def test_cursor_default_zero(conn: sqlite3.Connection) -> None:
    assert forkb.get_cursor(conn, "never-seen") == 0


def test_cursor_advance_and_read(conn: sqlite3.Connection) -> None:
    forkb.advance_cursor(conn, "s1", 42)
    assert forkb.get_cursor(conn, "s1") == 42
    forkb.advance_cursor(conn, "s1", 99)  # UPSERT updates in place
    assert forkb.get_cursor(conn, "s1") == 99


def test_cursor_per_session_isolation(conn: sqlite3.Connection) -> None:
    forkb.advance_cursor(conn, "s1", 10)
    forkb.advance_cursor(conn, "s2", 20)
    assert forkb.get_cursor(conn, "s1") == 10
    assert forkb.get_cursor(conn, "s2") == 20


# --------------------------------------------------------------------------- #
# T2.4 — prune_window                                                           #
# --------------------------------------------------------------------------- #


def test_prune_window_drops_old_keeps_in_window(conn: sqlite3.Connection) -> None:
    window_secs = WINDOW_DAYS * 86400
    old_ts = NOW - window_secs - 1  # just outside the window
    in_ts = NOW - window_secs + 1  # just inside the window
    for ts in (old_ts, old_ts, in_ts):
        forkb.append_activity(
            conn, session_id="s1", ts=ts, role="user", kind="prompt", body="x"
        )
    assert _count(conn) == 3

    pruned = forkb.prune_window(conn, now_epoch=NOW)
    assert pruned == 2
    assert _count(conn) == 1
    remaining_ts = conn.execute("SELECT Ts FROM Activity;").fetchone()[0]
    assert remaining_ts == in_ts


def test_prune_window_reclaim_runs_clean(conn: sqlite3.Connection) -> None:
    # No rows to prune — prune still calls reclaim without error, returns 0.
    assert forkb.prune_window(conn, now_epoch=NOW) == 0


# --------------------------------------------------------------------------- #
# T2.4 — rows_for_session                                                       #
# --------------------------------------------------------------------------- #


def test_rows_for_session_scopes_and_windows(conn: sqlite3.Connection) -> None:
    window_secs = WINDOW_DAYS * 86400
    # s1: two in-window rows + one out-of-window row.
    forkb.append_activity(
        conn, session_id="s1", ts=NOW - 100, role="user", kind="prompt", body="a"
    )
    forkb.append_activity(
        conn, session_id="s1", ts=NOW - 50, role="assistant", kind="text", body="b"
    )
    forkb.append_activity(
        conn, session_id="s1", ts=NOW - window_secs - 1, role="user",
        kind="prompt", body="old",
    )
    # s2: one in-window row — must not leak into s1's result.
    forkb.append_activity(
        conn, session_id="s2", ts=NOW - 10, role="user", kind="prompt", body="other"
    )

    rows = forkb.rows_for_session(conn, "s1", now_epoch=NOW)
    assert [r["Body"] for r in rows] == ["a", "b"]  # oldest-first, in-window only
    assert all(r["SessionId"] == "s1" for r in rows)


# --------------------------------------------------------------------------- #
# SC-6 — model-free (IN-12, SC-6)                                               #
# --------------------------------------------------------------------------- #


def test_module_imports_no_model_transport() -> None:
    """Importing claudemem.store.forkb pulls in no anthropic and no model call.

    Run in a fresh interpreter so the assertion is not contaminated by another
    test having already imported ``anthropic``. The module must also import no
    ``enrich``/``recall`` peer (architecture §2.4 / the read-path firewall).
    """
    code = (
        "import sys, json\n"
        "import claudemem.store.forkb  # noqa: F401\n"
        "bad = [m for m in sys.modules if m == 'anthropic'\n"
        "       or m.startswith('anthropic.')\n"
        "       or m == 'claudemem.enrich' or m.startswith('claudemem.enrich.')\n"
        "       or m == 'claudemem.recall' or m.startswith('claudemem.recall.')]\n"
        "print(json.dumps(bad))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "[]", result.stdout


# --------------------------------------------------------------------------- #
# Archive read accessors (§4.3, §5.2) — recall fallback delegates here          #
# --------------------------------------------------------------------------- #


def test_archive_matching_treats_like_wildcards_literally(
    conn: sqlite3.Connection,
) -> None:
    """The ``_like_escape`` contract: ``%``/``_``/``\\`` in a token match literally.

    A token like ``50%`` must NOT behave as a LIKE wildcard (which would match
    every row); it must match only bodies containing the literal substring
    ``50%``. Likewise ``a_b`` must not match ``axb`` via the ``_`` single-char
    wildcard.
    """
    hit = forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="user", kind="prompt",
        body="cpu sat at 50% during the build",
    )
    # A row that would be a false positive if `%` acted as a wildcard.
    forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="user", kind="prompt",
        body="totally unrelated activity with no percentages",
    )
    rows = forkb.archive_matching(conn, ["50%"], cutoff_epoch=NOW - 1, limit=20)
    ids = [r["Id"] for r in rows]
    assert ids == [hit]  # only the literal-substring match, not the wildcard sweep

    # `_` must also be literal: `a_b` matches "a_b", not "axb".
    lit = forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="user", kind="prompt", body="token a_b here",
    )
    forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="user", kind="prompt", body="token axb here",
    )
    rows2 = forkb.archive_matching(conn, ["a_b"], cutoff_epoch=NOW - 1, limit=20)
    assert [r["Id"] for r in rows2] == [lit]


def test_archive_matching_excludes_tool_rows_via_null_body(
    conn: sqlite3.Connection,
) -> None:
    """Tool rows (``Body`` NULL, §3.5) are excluded by the ``Body IS NULL`` filter
    alone — no ``Role`` predicate needed, since a tool row never has a body."""
    forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="tool", kind="bash", body="kubernetes output",
    )
    user = forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="user", kind="prompt", body="kubernetes question",
    )
    rows = forkb.archive_matching(conn, ["kubernetes"], cutoff_epoch=NOW - 1, limit=20)
    assert [r["Id"] for r in rows] == [user]  # tool row's NULL body is not scanned


def test_get_activity_returns_row_or_none(conn: sqlite3.Connection) -> None:
    rowid = forkb.append_activity(
        conn, session_id="s1", ts=NOW, role="user", kind="prompt", body="hello world",
    )
    row = forkb.get_activity(conn, rowid)
    assert row is not None
    assert row["Body"] == "hello world"
    assert forkb.get_activity(conn, 999_999) is None  # absent rowid → None
