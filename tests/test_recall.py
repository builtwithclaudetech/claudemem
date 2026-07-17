"""Integration tests for the recall read commands (T3.6).

Exercises ``claudemem.recall.search`` / ``get`` / ``menu`` end-to-end over the
**real** store layer — Fork A seeded via ``forka.upsert_record`` (through
``files.RecordFile``) and Fork B via ``forkb.append_activity`` — with a fixed
``now_epoch`` wherever salience matters, and ``CLAUDEMEM_HOME`` pointed at
``tmp_path`` so no test touches the real ``~/.claude``.

Coverage maps to the PRD ids:

* **SC-12** — a query matching both a global and a project record returns both,
  correctly ranked (scope merge).
* **IN-19** — single-factor salience variations order correctly; a pinned old
  record outranks a fresh unpinned one in both ``search`` re-rank and ``menu``.
* **SC-5** — with >30 active records, ``menu`` emits ≤30 entries, ≤600 tokens,
  ≤10,000 chars, pinned-first, no bodies; ``source='resume'`` → empty.
* **IN-3** — a query with no Fork A hit above the relevance floor falls back to
  Fork B, results labelled ``[archive]`` with ``b:`` ids; an empty/punctuation
  query also triggers the fallback.
* **IN-4** — ``get a:<name>`` bumps ``AccessCount`` in the index (not the file);
  an unknown ``a:`` / ``b:`` id → "not found", no exception.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from claudemem import config, files, index
from claudemem.recall import get as get_mod
from claudemem.recall import menu as menu_mod
from claudemem.recall import search as search_mod
from claudemem.store import forka, forkb

# A fixed "now" so recency_decay (and thus salience ordering) is deterministic.
NOW = files.iso_to_epoch("2026-05-30T12:00:00Z")
DAY = 86_400


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    return tmp_path


@pytest.fixture
def conn_a(home: Path) -> sqlite3.Connection:
    return index.open_forkA()


@pytest.fixture
def conn_b(home: Path) -> sqlite3.Connection:
    return index.open_forkB()


def _project_scope(pid: str = "proj-1") -> config.ScopeContext:
    return config.ScopeContext(
        kind="project", project_id=pid, global_dir=Path("/g"), project_dir=Path("/p")
    )


def _global_scope() -> config.ScopeContext:
    return config.ScopeContext(
        kind="global", project_id=None, global_dir=Path("/g"), project_dir=None
    )


def _epoch(days_ago: float) -> str:
    return files.epoch_to_iso(NOW - int(days_ago * DAY))


def _rf(
    name: str,
    *,
    scope: str = "project",
    summary: str | None = "a summary",
    body: str = "the body text",
    aliases: list[str] | None = None,
    pinned: bool = False,
    importance: int = 3,
    last_accessed_days_ago: float = 0.0,
) -> files.RecordFile:
    return files.RecordFile(
        path=Path(f"/tmp/{name}.md"),
        name=name,
        type="reference",
        scope=scope,
        importance=importance,
        pinned=pinned,
        source="explicit",
        created=_epoch(10),
        last_accessed=_epoch(last_accessed_days_ago),
        access_count=0,
        hit_count=0,
        summary=summary,
        aliases=aliases if aliases is not None else [],
        superseded_by=None,
        stale=False,
        body=body,
    )


def _seed(conn: sqlite3.Connection, rf: files.RecordFile, scope: config.ScopeContext) -> int:
    return forka.upsert_record(conn, rf, scope)


def _seed_distractors(conn: sqlite3.Connection, scope: config.ScopeContext, n: int = 40) -> None:
    """Seed ``n`` non-matching records so bm25 IDF is realistic.

    SQLite bm25 IDF collapses toward 0 on a one/two-document corpus, so a single
    seeded record never clears ``relevance_floor=0.30`` (the floor is calibrated
    for the ClaudeMem-2k corpus, tech-design §4.6). These distractors give the
    matching term meaningful rarity so an above-floor Fork A hit is reachable.
    """
    for i in range(n):
        _seed(
            conn,
            _rf(f"distract{i:02d}", summary=f"distractor {i}", body=f"unrelated filler {i} " * 4),
            scope,
        )


# --------------------------------------------------------------------------- #
# SC-12 — scope merge (global + project)                                       #
# --------------------------------------------------------------------------- #


def test_search_scope_merges_global_and_project(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    proj = _project_scope()
    glob = _global_scope()
    _seed_distractors(conn_a, proj)
    _seed(conn_a, _rf("proj_dragon", summary="project dragon", body="a dragon in the project"), proj)
    _seed(conn_a, _rf("glob_dragon", scope="global", summary="global dragon", body="a dragon globally"), glob)
    # A record in *another* project must NOT appear (scope merge is own+global).
    _seed(conn_a, _rf("other_dragon", body="a dragon elsewhere"), _project_scope("other"))

    out = search_mod.search(conn_a, conn_b, "dragon", proj, now_epoch=NOW)
    ids = {line.split(" ", 1)[0] for line in out.splitlines()}
    assert "a:proj_dragon" in ids
    assert "a:glob_dragon" in ids  # global merged in (SC-12)
    assert "a:other_dragon" not in ids  # other project excluded


# --------------------------------------------------------------------------- #
# IN-19 — salience ordering + pinned-first                                     #
# --------------------------------------------------------------------------- #


def test_search_single_factor_importance_orders(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    scope = _project_scope()
    _seed_distractors(conn_a, scope)
    # Same relevance + recency; importance is the only varying factor.
    _seed(conn_a, _rf("hi", summary="needle one", body="needle term", importance=5), scope)
    _seed(conn_a, _rf("lo", summary="needle two", body="needle term", importance=1), scope)
    out = search_mod.search(conn_a, conn_b, "needle", scope, now_epoch=NOW)
    ids = [line.split(" ", 1)[0] for line in out.splitlines()]
    assert ids.index("a:hi") < ids.index("a:lo")


def test_search_pinned_old_outranks_fresh_unpinned(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    scope = _project_scope()
    _seed_distractors(conn_a, scope)
    _seed(
        conn_a,
        _rf("pinned_old", summary="needle pinned", body="needle term", pinned=True, last_accessed_days_ago=365),
        scope,
    )
    _seed(
        conn_a,
        _rf("fresh", summary="needle fresh", body="needle term", pinned=False, last_accessed_days_ago=0),
        scope,
    )
    out = search_mod.search(conn_a, conn_b, "needle", scope, now_epoch=NOW)
    ids = [line.split(" ", 1)[0] for line in out.splitlines()]
    assert ids[0] == "a:pinned_old"  # pinned-first regardless of age (IN-19)


def test_menu_pinned_old_outranks_fresh_unpinned(conn_a: sqlite3.Connection) -> None:
    scope = _project_scope()
    _seed(conn_a, _rf("pinned_old", pinned=True, last_accessed_days_ago=365), scope)
    _seed(conn_a, _rf("fresh", pinned=False, last_accessed_days_ago=0), scope)
    out = menu_mod.menu(conn_a, scope, now_epoch=NOW)
    ids = [line.split(" ", 1)[0] for line in out.splitlines()]
    assert ids[0] == "a:pinned_old"


# --------------------------------------------------------------------------- #
# SC-5 — menu budget                                                           #
# --------------------------------------------------------------------------- #


def test_menu_respects_caps_and_pinned_first(conn_a: sqlite3.Connection) -> None:
    scope = _project_scope()
    # >30 active records; a couple pinned so pinned-first is observable.
    for i in range(40):
        _seed(
            conn_a,
            _rf(
                f"rec{i:02d}",
                summary=f"summary number {i}",
                pinned=(i in (37, 38)),
                last_accessed_days_ago=i,  # later i => older => lower salience
            ),
            scope,
        )
    out = menu_mod.menu(conn_a, scope, now_epoch=NOW)
    lines = out.splitlines()
    assert len(lines) <= 30  # SC-5 max_entries
    assert len(out) <= 10_000  # SC-5 hard char ceiling
    assert (len(out) + 3) // 4 <= 600  # SC-5 token ceiling (chars/4, conservative)
    # Pinned-first: the two pinned records lead.
    leading = {lines[0].split(" ", 1)[0], lines[1].split(" ", 1)[0]}
    assert leading == {"a:rec37", "a:rec38"}
    # No bodies — every line is a single `id␠title` line (no blank-line+body).
    assert "" not in lines
    for line in lines:
        assert line.startswith("a:")


def test_menu_resume_source_is_empty(conn_a: sqlite3.Connection) -> None:
    scope = _project_scope()
    _seed(conn_a, _rf("anything"), scope)
    assert menu_mod.menu(conn_a, scope, source="resume", now_epoch=NOW) == ""


# --------------------------------------------------------------------------- #
# IN-3 — Fork A → Fork B archive fallback                                      #
# --------------------------------------------------------------------------- #


def _seed_activity(conn_b: sqlite3.Connection, body: str, *, role: str = "user") -> int:
    return forkb.append_activity(
        conn_b, session_id="s1", ts=NOW, role=role, kind="prompt", body=body
    )


def test_search_falls_back_to_archive_when_no_forka_match(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    scope = _project_scope()
    # Fork A has an unrelated record; the query won't match it at all → no
    # candidates → archive fallback.
    _seed(conn_a, _rf("unrelated", body="completely different content"), scope)
    rowid = _seed_activity(conn_b, "we discussed the kubernetes migration today")

    out = search_mod.search(conn_a, conn_b, "kubernetes", scope, now_epoch=NOW)
    lines = out.splitlines()
    assert lines, "expected an archive hit"
    assert lines[0].startswith(f"b:{rowid}")
    assert "[archive]" in lines[0]


def test_search_archive_fallback_below_relevance_floor(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    scope = _project_scope()
    # A Fork A record that matches the token only weakly (single body mention,
    # body weight 1) so normalized_relevance stays below relevance_floor=0.30,
    # forcing the A→B fallback even though there *is* a lexical candidate.
    weak_body = "lorem ipsum " * 200 + " rare_token " + "dolor sit " * 200
    _seed(conn_a, _rf("weak", summary="unrelated summary", body=weak_body), scope)
    _seed_activity(conn_b, "a clear note mentioning rare_token prominently")

    out = search_mod.search(conn_a, conn_b, "rare_token", scope, now_epoch=NOW)
    lines = out.splitlines()
    assert lines and lines[0].startswith("b:")
    assert "[archive]" in lines[0]


@pytest.mark.parametrize("q", ["", "   ", "!!!", "@#$%"])
def test_search_empty_or_punctuation_triggers_archive(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection, q: str
) -> None:
    scope = _project_scope()
    _seed(conn_a, _rf("present", body="some indexed content"), scope)
    rowid = _seed_activity(conn_b, "most recent activity line")
    out = search_mod.search(conn_a, conn_b, q, scope, now_epoch=NOW)
    lines = out.splitlines()
    # No usable MATCH token → archive fallback (recent rows), not Fork A.
    assert lines and lines[0].startswith(f"b:{rowid}")


def test_search_json_archive_marks_source(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    import json

    scope = _project_scope()
    rowid = _seed_activity(conn_b, "json archive hit about widgets")
    out = search_mod.search(conn_a, conn_b, "widgets", scope, json=True, now_epoch=NOW)
    obj = json.loads(out.splitlines()[0])
    assert obj["id"] == f"b:{rowid}"
    assert obj["archive"] is True


def test_search_forka_hit_does_not_fall_back(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    scope = _project_scope()
    _seed_distractors(conn_a, scope)
    # Strong match: token in Name + Summary + Body → relevance well above floor.
    _seed(conn_a, _rf("dragons", summary="all about dragons", body="dragons dragons dragons"), scope)
    _seed_activity(conn_b, "an archive line also mentioning dragons")
    out = search_mod.search(conn_a, conn_b, "dragons", scope, now_epoch=NOW)
    lines = out.splitlines()
    assert lines[0].startswith("a:dragons")  # curated hit, no archive fallback
    assert not any(line.startswith("b:") for line in lines)


# --------------------------------------------------------------------------- #
# IN-4 — get access refresh + not-found                                        #
# --------------------------------------------------------------------------- #


def test_get_forka_returns_body_and_bumps_access_in_index(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    scope = _project_scope()
    _seed(conn_a, _rf("notes", summary="my notes", body="the full body here"), scope)

    before = forka.select_record(conn_a, scope, "notes")
    assert before is not None
    assert before.access_count == 0

    out = get_mod.get(conn_a, conn_b, "a:notes", scope, now_epoch=NOW + 5)
    assert "the full body here" in out

    after = forka.select_record(conn_a, scope, "notes")
    assert after is not None
    assert after.access_count == before.access_count + 1  # IN-4 index bump
    assert after.last_accessed == NOW + 5


def test_get_archive_returns_body(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    scope = _project_scope()
    rowid = _seed_activity(conn_b, "archived discussion about caching")
    out = get_mod.get(conn_a, conn_b, f"b:{rowid}", scope, now_epoch=NOW)
    assert "archived discussion about caching" in out
    assert out.splitlines()[0].startswith(f"b:{rowid}")


def test_get_unknown_forka_name_is_not_found(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    out = get_mod.get(conn_a, conn_b, "a:nope", _project_scope(), now_epoch=NOW)
    assert out.startswith("not found")


def test_get_unknown_archive_rowid_is_not_found(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    out = get_mod.get(conn_a, conn_b, "b:999999", _project_scope(), now_epoch=NOW)
    assert out.startswith("not found")


def test_get_malformed_id_is_not_found_no_exception(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    for bad in ["garbage", "c:x", "a:", "b:notanint", ""]:
        out = get_mod.get(conn_a, conn_b, bad, _project_scope(), now_epoch=NOW)
        assert out.startswith("not found")


def test_get_json_forka_includes_body(
    conn_a: sqlite3.Connection, conn_b: sqlite3.Connection
) -> None:
    import json

    scope = _project_scope()
    _seed(conn_a, _rf("doc", summary="the doc", body="body content"), scope)
    out = get_mod.get(conn_a, conn_b, "a:doc", scope, json=True, now_epoch=NOW)
    obj = json.loads(out)
    assert obj["id"] == "a:doc"
    assert obj["body"] == "body content"
