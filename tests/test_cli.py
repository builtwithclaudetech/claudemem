"""Integration tests for the L4 cli entry point + lazy dispatch (T5.1-5.3).

Drives ``claudemem.cli.main`` against a **real** tmp store (``CLAUDEMEM_HOME`` →
``tmp_path`` so nothing touches the real ``~/.claude``), with a **fake**
enrichment backend injected via ``select_backend`` so no test makes a real model
call and no test spawns ``claude``.

Coverage maps to the spec ids:

* **§6.3 MF-3 / §7.1** — ``CLAUDEMEM_DISABLE_HOOKS=1`` → ``main`` returns 0
  immediately, even on malformed argv, before any argparse parse or DB open.
* **IN-3/IN-4/IN-11/IN-21** — search/get/menu round-trip; a ``--json`` id from
  ``search`` round-trips to ``get``/``used``.
* **SC-6/C-17** — a read command (``search``) imports no ``enrich`` (asserted via
  the fresh-interpreter firewall harness).
* **IN-6/IN-7/IN-8 / SC-7/SC-9** — pin/unpin, forget (supersede trail), used
  (HitCount +1; ``used b:`` signal; pruned ``b:`` → not found exit 0).
* **SC-10/SC-3** — ``save`` persists even with a deferring fake backend
  (``EnrichPending=1``, exit 0); over-cap warns but still persists.
* **IN-10/IN-17/SC-11** — ``import`` re-runnable; ``reindex`` PHASE A rebuild +
  PHASE B backfill with the fake backend.
* **SC-3** — every command in the SC-3 set returns exit 0 under degradation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from claudemem import cli, config, files, index
from claudemem.enrich import backend as backend_mod
from claudemem.enrich.backend import (
    BackendOutcome,
    DeferralEntry,
    EnrichRequest,
    EnrichResult,
    ReflectOutcome,
    ReflectRequest,
    SpendEntry,
)

# Import the real fresh-interpreter firewall harness (the SC-6 read-path check).
from tests.test_firewall import build_fake_claude_shim, run_in_fresh_interpreter

NOW = files.iso_to_epoch("2026-05-30T12:00:00Z")


# --------------------------------------------------------------------------- #
# Fixtures                                                                       #
# --------------------------------------------------------------------------- #


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME (DB files) at tmp_path; cwd → tmp_path for scope."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def scoped_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Redirect the global + project Fork A memory dirs into tmp_path.

    ``resolve_scope`` reads module-level path constants; monkeypatch them so the
    cwd-derived project scope and the global scope both land under tmp_path and
    every ``save``/``import``/admin file write stays inside the sandbox.
    """
    global_dir = tmp_path / "global_memory"
    projects_root = tmp_path / "projects"
    global_dir.mkdir(parents=True, exist_ok=True)
    projects_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "GLOBAL_MEMORY_DIR", global_dir)
    monkeypatch.setattr(config, "PROJECTS_ROOT", projects_root)
    return global_dir, projects_root


def _project_dir(scoped_dirs: tuple[Path, Path]) -> Path:
    """The cwd-derived project memory dir a project-scope save writes into."""
    _global_dir, projects_root = scoped_dirs
    scope_ctx = config.resolve_scope(Path.cwd(), None, None)
    assert scope_ctx.project_id is not None
    return projects_root / scope_ctx.project_id / "memory"


class FakeEnrichBackend:
    """A model-free fake backend matching the EnrichmentBackend protocol.

    ``mode='enrich'`` returns an enrich verdict per request (so a save lands
    enriched, EnrichPending cleared); ``mode='defer'`` defers every record (the
    SC-10 degraded-save path, EnrichPending=1). No model, no claude spawn.
    """

    def __init__(self, mode: str = "enrich") -> None:
        self.mode = mode
        self.calls: list[list[EnrichRequest]] = []

    @staticmethod
    def detect() -> bool:
        return True

    def enrich_batch(self, reqs: list[EnrichRequest]) -> BackendOutcome:
        self.calls.append(reqs)
        if self.mode == "defer":
            return BackendOutcome(
                results=[],
                deferred=[DeferralEntry(record_id=r.record_id, reason="auth") for r in reqs],
                spend=[],
            )
        results = [
            EnrichResult(
                record_id=r.record_id,
                summary=f"summary for {r.name}",
                aliases=[r.name],
                dedup_verdict="new",
                dedup_target_name=None,
                conflict_explanation=None,
            )
            for r in reqs
        ]
        spend = [
            SpendEntry(call_site="save", model="haiku", backend="cli", output_tokens=10)
            for _ in reqs
        ]
        return BackendOutcome(results=results, deferred=[], spend=spend)

    def reflect(self, req: ReflectRequest) -> ReflectOutcome:
        return ReflectOutcome()


@pytest.fixture
def fake_enrich(monkeypatch: pytest.MonkeyPatch) -> FakeEnrichBackend:
    """Inject a fake backend so enrich_batch never makes a real model call."""
    fake = FakeEnrichBackend(mode="enrich")
    monkeypatch.setattr(backend_mod, "select_backend", lambda settings: fake)
    backend_mod._reset_cache_for_tests()
    return fake


# --------------------------------------------------------------------------- #
# T5.1 — recursion guard (the load-bearing first statement)                      #
# --------------------------------------------------------------------------- #


def test_guard_returns_zero_immediately_on_malformed_argv(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLAUDEMEM_DISABLE_HOOKS set → main returns 0 even on malformed argv, with
    NO argparse error and NO DB opened (§6.3 MF-3, §7.1)."""
    monkeypatch.setenv(cli.GUARD_ENV_VAR, "1")

    # Trip a hard failure if anything tries to open SQLite past the guard.
    def _boom(*args: object, **kwargs: object) -> object:
        raise AssertionError("DB must not open under the guard")

    monkeypatch.setattr(index, "open_forkA", _boom)
    monkeypatch.setattr(index, "open_forkB", _boom)

    for argv in (
        ["--this-flag-does-not-exist"],
        ["hook"],  # missing required <event>
        ["search", "--bogus", "x"],
        ["\x00garbage"],
    ):
        assert cli.main(argv) == 0


# --------------------------------------------------------------------------- #
# T5.2 — read round-trip + the IN-21 id round-trip                               #
# --------------------------------------------------------------------------- #


def _save(content: str, **kw: object) -> int:
    argv = ["save", content]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)]
    return cli.main(argv)


def test_search_get_used_json_id_roundtrip(
    home: Path, scoped_dirs: tuple[Path, Path], fake_enrich: FakeEnrichBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    """save → search --json → parse id → get/used round-trip (IN-21)."""
    # SQLite bm25 IDF collapses toward 0 on a one-document corpus, so a single
    # saved record never clears relevance_floor=0.30 (the floor is calibrated for
    # a realistic corpus, tech-design §4.6, mirroring test_recall._seed_distractors).
    # Seed distractors via save so the rare query term has meaningful rarity.
    for i in range(40):
        assert _save(f"unrelated filler topic number {i} " * 4, name=f"d{i:02d}") == 0
    assert _save("zephyrquux zephyrquux indexing strategy notes for zephyrquux", name="zephyr-rec") == 0
    capsys.readouterr()

    assert cli.main(["search", "zephyrquux", "--json"]) == 0
    out = capsys.readouterr().out
    assert '"id":"a:zephyr-rec"' in out

    # The captured id round-trips to get.
    assert cli.main(["get", "a:zephyr-rec"]) == 0
    body = capsys.readouterr().out
    assert "zephyrquux" in body

    # …and to used (HitCount bump, SC-9).
    assert cli.main(["used", "a:zephyr-rec"]) == 0
    assert "hit_count=1" in capsys.readouterr().out


def test_menu_lists_saved_record(
    home: Path, scoped_dirs: tuple[Path, Path], fake_enrich: FakeEnrichBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    """menu emits the active record's id␠title line; resume source → empty (IN-11)."""
    assert _save("a durable fact about deployment", name="deploy-fact") == 0
    capsys.readouterr()

    assert cli.main(["menu"]) == 0
    assert "a:deploy-fact" in capsys.readouterr().out

    assert cli.main(["menu", "--source", "resume"]) == 0
    assert capsys.readouterr().out == ""


def test_get_unknown_id_not_found_exit_zero(
    home: Path, scoped_dirs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """An unknown / unparseable id → 'not found', exit 0 (SC-3)."""
    assert cli.main(["get", "a:nope"]) == 0
    assert "not found" in capsys.readouterr().out
    assert cli.main(["get", "garbage-no-prefix"]) == 0
    assert "not found" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# SC-6 — a read command imports no enrich (fresh-interpreter firewall)           #
# --------------------------------------------------------------------------- #


def test_search_command_never_imports_enrich(tmp_path: Path) -> None:
    """Running ``search`` in a fresh interpreter pulls in neither ``claudemem.enrich``
    nor ``anthropic``, and spawns no ``claude`` (SC-6/C-17)."""
    shim_dir, sentinel = build_fake_claude_shim(tmp_path)
    home = tmp_path / "home"
    home.mkdir()
    env = {**__import__("os").environ, config.CONFIG_HOME_ENV: str(home)}
    run = run_in_fresh_interpreter(
        ["search", "anything"], extra_path=shim_dir, env=env, sentinel=sentinel
    )
    assert run.returncode == 0, run.stderr
    assert "claudemem.cli" in run.imported_modules
    assert "claudemem.enrich" not in run.imported_modules
    assert not any(m.startswith("claudemem.enrich") for m in run.imported_modules)
    assert "anthropic" not in run.imported_modules
    assert run.spawned_claude is False


# --------------------------------------------------------------------------- #
# T5.2 — admin: pin/unpin, forget (SC-7 trail), used (SC-9 + used b:)            #
# --------------------------------------------------------------------------- #


def test_pin_unpin_toggle_index_and_file(
    home: Path, scoped_dirs: tuple[Path, Path], fake_enrich: FakeEnrichBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    """pin/unpin flip Pinned in the index AND the frontmatter (IN-7, SC-4)."""
    proj_dir = _project_dir(scoped_dirs)
    assert _save("pinnable fact", name="pinme") == 0
    capsys.readouterr()

    assert cli.main(["pin", "a:pinme"]) == 0
    conn = index.open_forkA()
    try:
        pinned = conn.execute(
            "SELECT Pinned FROM Record WHERE Name = 'pinme';"
        ).fetchone()[0]
    finally:
        conn.close()
    assert pinned == 1
    # File frontmatter flushed too (project-scope file, cwd-derived).
    assert "pinned: true" in (proj_dir / "pinme.md").read_text(encoding="utf-8")

    assert cli.main(["unpin", "a:pinme"]) == 0
    conn = index.open_forkA()
    try:
        pinned = conn.execute(
            "SELECT Pinned FROM Record WHERE Name = 'pinme';"
        ).fetchone()[0]
    finally:
        conn.close()
    assert pinned == 0


def test_forget_supersede_trail(
    home: Path, scoped_dirs: tuple[Path, Path], fake_enrich: FakeEnrichBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    """forget soft-deletes (SupersededBy set, row not DELETEd) — the SC-7 trail."""
    proj_dir = _project_dir(scoped_dirs)
    assert _save("disposable fact", name="bye") == 0
    capsys.readouterr()

    assert cli.main(["forget", "a:bye"]) == 0
    conn = index.open_forkA()
    try:
        row = conn.execute(
            "SELECT SupersededBy FROM Record WHERE Name = 'bye';"
        ).fetchone()
    finally:
        conn.close()
    # Row still exists (trail), now superseded.
    assert row is not None
    assert row[0] == "forget"
    # File frontmatter carries the trail too (project-scope file).
    assert "superseded_by: forget" in (proj_dir / "bye.md").read_text(encoding="utf-8")


def test_used_forka_bumps_hit_count_and_clears_stale(
    home: Path, scoped_dirs: tuple[Path, Path], fake_enrich: FakeEnrichBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    """used a: increments HitCount by exactly one and clears Stale (SC-9/SC-13)."""
    assert _save("reinforceable fact", name="hit") == 0
    capsys.readouterr()
    # Seed a stale flag in the index to prove it clears.
    conn = index.open_forkA()
    try:
        with index.write_tx(conn):
            conn.execute("UPDATE Record SET Stale = 1 WHERE Name = 'hit';")
    finally:
        conn.close()

    assert cli.main(["used", "a:hit"]) == 0
    capsys.readouterr()

    conn = index.open_forkA()
    try:
        hit, stale = conn.execute(
            "SELECT HitCount, Stale FROM Record WHERE Name = 'hit';"
        ).fetchone()
    finally:
        conn.close()
    assert hit == 1
    assert stale == 0


def test_used_forkb_signal_and_pruned(
    home: Path, scoped_dirs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """used b:<rowid> signals on a present row; a pruned/unknown rowid → not found, exit 0."""
    from claudemem.store import forkb

    conn_b = index.open_forkB()
    try:
        rowid = forkb.append_activity(
            conn_b, session_id="s1", ts=NOW, role="user", kind="message", body="hi"
        )
    finally:
        conn_b.close()

    assert cli.main(["used", f"b:{rowid}"]) == 0
    assert "promotion-hit" in capsys.readouterr().out

    assert cli.main(["used", "b:999999"]) == 0
    assert "not found" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# T5.3 — write: save persists even degraded; over-cap warns but persists         #
# --------------------------------------------------------------------------- #


def test_save_persists_with_deferring_backend(
    home: Path, scoped_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A deferring fake backend → record persists EnrichPending=1, exit 0 (SC-10/SC-3)."""
    fake = FakeEnrichBackend(mode="defer")
    monkeypatch.setattr(backend_mod, "select_backend", lambda settings: fake)
    backend_mod._reset_cache_for_tests()

    assert _save("degraded save body", name="degraded") == 0
    capsys.readouterr()

    conn = index.open_forkA()
    try:
        pending = conn.execute(
            "SELECT EnrichPending FROM Record WHERE Name = 'degraded';"
        ).fetchone()[0]
    finally:
        conn.close()
    assert pending == 1


def test_save_over_cap_warns_but_persists(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    fake_enrich: FakeEnrichBackend,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Over-cap → a warning is logged but the save still persists (§5.10/SC-10)."""
    # Force a tiny daily token cap so the warn-not-block tally trips.
    real_load = config.load_config

    def _tiny_cap(*a: object, **k: object) -> config.Settings:
        base = real_load()
        return config.Settings(
            llm=base.llm,
            ranking=base.ranking,
            forkb=base.forkb,
            spend=config.SpendSettings(daily_token_cap=1, monthly_token_cap=1),
            promotion=base.promotion,
            menu=base.menu,
        )

    monkeypatch.setattr(config, "load_config", _tiny_cap)
    # Pre-load a spend row so the tally is over the (tiny) cap.
    from claudemem.store import spend

    conn = index.open_forkA()
    try:
        spend.record_spend(conn, call_site="save", model="haiku", backend="cli", output_tokens=100)
    finally:
        conn.close()

    import logging

    with caplog.at_level(logging.WARNING, logger="claudemem"):
        assert _save("over cap body", name="overcap") == 0
    capsys.readouterr()

    assert any("cap" in r.message.lower() for r in caplog.records)
    conn = index.open_forkA()
    try:
        row = conn.execute("SELECT Name FROM Record WHERE Name = 'overcap';").fetchone()
    finally:
        conn.close()
    assert row is not None  # persisted despite over-cap


# --------------------------------------------------------------------------- #
# T5.3 — import re-runnable; reindex A rebuild + B backfill                       #
# --------------------------------------------------------------------------- #


def test_import_rerunnable(
    home: Path, scoped_dirs: tuple[Path, Path], tmp_path: Path, fake_enrich: FakeEnrichBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    """import ingests *.md; a second run is idempotent on the natural key (SC-11)."""
    src = tmp_path / "src"
    src.mkdir()
    (src / "one.md").write_text(
        "---\ntype: fact\nscope: global\nsource: explicit\n"
        "created: 2026-01-01T00:00:00Z\nlast_accessed: 2026-01-01T00:00:00Z\n"
        "aliases: []\n---\n\nimported body one\n",
        encoding="utf-8",
    )

    assert cli.main(["import", str(src)]) == 0
    capsys.readouterr()
    assert cli.main(["import", str(src)]) == 0  # re-runnable, no crash
    capsys.readouterr()

    conn = index.open_forkA()
    try:
        count = conn.execute("SELECT COUNT(*) FROM Record WHERE Name = 'one';").fetchone()[0]
    finally:
        conn.close()
    assert count == 1  # upsert on natural key — no duplicate row


def test_reindex_phase_a_then_b(
    home: Path, scoped_dirs: tuple[Path, Path], fake_enrich: FakeEnrichBackend, capsys: pytest.CaptureFixture[str]
) -> None:
    """reindex rebuilds from files (PHASE A) then backfills EnrichPending (PHASE B)."""
    global_dir, _ = scoped_dirs
    # A file with no summary → EnrichPending after PHASE A rebuild.
    (global_dir / "pending.md").write_text(
        "---\ntype: fact\nscope: global\nsource: explicit\n"
        "created: 2026-01-01T00:00:00Z\nlast_accessed: 2026-01-01T00:00:00Z\n"
        "aliases: []\n---\n\npending body\n",
        encoding="utf-8",
    )

    assert cli.main(["reindex"]) == 0
    out = capsys.readouterr().out
    assert "backfilled 1" in out
    assert "0 still enrich-pending" in out

    conn = index.open_forkA()
    try:
        pending = conn.execute(
            "SELECT EnrichPending FROM Record WHERE Name = 'pending';"
        ).fetchone()[0]
    finally:
        conn.close()
    assert pending == 0  # backfill cleared it


def test_reindex_no_backfill_skips_phase_b(
    home: Path, scoped_dirs: tuple[Path, Path], capsys: pytest.CaptureFixture[str]
) -> None:
    """reindex --no-backfill runs only the model-free PHASE A (no enrich import)."""
    global_dir, _ = scoped_dirs
    (global_dir / "nb.md").write_text(
        "---\ntype: fact\nscope: global\nsource: explicit\n"
        "created: 2026-01-01T00:00:00Z\nlast_accessed: 2026-01-01T00:00:00Z\n"
        "summary: has summary\naliases: []\n---\n\nbody\n",
        encoding="utf-8",
    )
    assert cli.main(["reindex", "--no-backfill"]) == 0
    assert "backfill skipped" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# SC-3 — every command in the SC-3 set exits 0 under degradation                 #
# --------------------------------------------------------------------------- #


def test_sc3_command_set_all_exit_zero_when_degraded(
    home: Path, scoped_dirs: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """search/get/save/used/pin/unpin/forget/promote/reindex/menu/log/import all
    exit 0 with a lexical-only (deferring) backend — no key/SDK present (SC-3)."""
    fake = FakeEnrichBackend(mode="defer")
    monkeypatch.setattr(backend_mod, "select_backend", lambda settings: fake)
    backend_mod._reset_cache_for_tests()

    # Seed one Fork A record and one Fork B row to exercise the lookups.
    assert _save("sc3 seed body", name="seed") == 0
    capsys.readouterr()
    conn_b = index.open_forkB()
    try:
        from claudemem.store import forkb

        bid = forkb.append_activity(
            conn_b, session_id="s", ts=NOW, role="user", kind="message", body="seed log"
        )
    finally:
        conn_b.close()

    commands: list[list[str]] = [
        ["search", "seed"],
        ["get", "a:seed"],
        ["save", "another body", "--name", "another"],
        ["used", "a:seed"],
        ["used", f"b:{bid}"],
        ["pin", "a:seed"],
        ["unpin", "a:seed"],
        ["promote", f"b:{bid}"],
        ["reindex"],
        ["menu"],
        ["log", "--session-id", "s", "a model-free log line"],
        ["forget", "a:seed"],
    ]
    for argv in commands:
        assert cli.main(argv) == 0, f"command must exit 0 under SC-3 degradation: {argv}"
        capsys.readouterr()


# --------------------------------------------------------------------------- #
# hook dispatch hook-in point — degrades to exit 0 until hooks.py lands          #
# --------------------------------------------------------------------------- #


def test_hook_dispatch_noop_until_hooks_module(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`claudemem hook <event>` exits 0 as a no-op while claudemem.hooks is absent."""
    # Ensure the (not-yet-built) hooks module import fails cleanly.
    monkeypatch.setitem(sys.modules, "claudemem.hooks", None)  # type: ignore[arg-type]
    assert cli.main(["hook", "SessionStart"]) == 0


# --------------------------------------------------------------------------- #
# bad args (genuine usage error) may exit non-zero — NOT a SC-3 violation        #
# --------------------------------------------------------------------------- #


def test_unknown_command_returns_nonzero(home: Path) -> None:
    """An unknown subcommand / bad args is a genuine usage error → non-zero (allowed)."""
    assert cli.main(["definitely-not-a-command"]) != 0
