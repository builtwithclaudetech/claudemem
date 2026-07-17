"""§10.7 supporting-gate coverage (T5.5) — the one place the §10.7 SC set is
certified at the cli/hooks integration level.

tech-design §10.7 enumerates a set of Success Criteria that are exercised by
"targeted unit/integration tests (not new harnesses)": SC-5 menu budget, SC-7
supersede trail, SC-8 promotion gate (never auto-promote), SC-9 reinforce-on-
confirmed-hit-only, SC-10 warn-not-block, SC-11 import re-runnable, SC-12 scope
merge, SC-13 staleness round-trip, and IN-19 salience ordering. Several of these
already have coverage elsewhere (``test_cli.py``, ``test_hooks.py``,
``test_recall.py``, ``test_enrich_routine.py``); this module **consolidates the
§10.7 SC assertions into one clearly-named gate** so the §10.7 contract is
verifiable in a single place, and **fills the gaps** those modules leave.

Everything runs against a **real** store/files (``CLAUDEMEM_HOME`` → ``tmp_path``
so nothing touches the real ``~/.claude``) with a **fake backend injected** for
any enrich path — **no real model call, no ``claude`` spawn**.

§10.7 SC coverage matrix (where each is asserted in THIS module — see the
docstring of each test for the angle, and the cross-references below for the SCs
already covered elsewhere that this module references rather than duplicates):

* **SC-5** (menu budget) — covered fully in ``test_recall.py`` at the recall
  level; here a cli-level smoke (``test_sc5_menu_cli_smoke_within_budget``)
  confirms the budget holds end-to-end through ``cli main menu``.
* **SC-7** (supersede trail) — ``test_sc7_forget_drops_from_active_but_recoverable``
  + ``test_sc7_conflict_supersede_drops_from_active_history_intact``.
* **SC-8** (promotion gate) — ``test_sc8_reflection_proposes_never_auto_promotes``
  + ``test_sc8_unattended_session_end_defers_no_forka_record``.
* **SC-9** (reinforce on confirmed hit only) —
  ``test_sc9_used_bumps_hitcount_by_one`` (active hit),
  ``test_sc9_miss_does_not_bump_any_hitcount`` (the miss half — the gap), and
  ``test_sc9_reflection_passive_hit_reinforces_at_hooks_level`` (passive hit via
  the SessionEnd hook).
* **SC-10** (warn-not-block) — ``test_sc10_artificially_low_cap_save_still_persists``.
* **SC-11** (import re-runnable) — ``test_sc11_import_twice_no_net_new_with_backend``
  (with a backend) + ``test_sc11_import_offline_persists_pending_then_resolves``
  (offline → EnrichPending, dedup deferred to reindex).
* **SC-12** (scope merge) — covered in ``test_recall.py``; referenced, not
  duplicated.
* **SC-13** (staleness round-trip + horizon sweep) —
  ``test_sc13_stale_flag_round_trips_through_reindex`` (the ``stale`` flag round-
  trips through reindex) + ``test_sc13_used_clears_stale_after_reindex`` (a
  confirmed ``used`` hit clears it) + ``test_sc13_staleness_horizon_sweep`` (a
  record untouched beyond ``settings.staleness.horizon_days`` auto-acquires
  ``stale`` on reindex — ``store.reindex.rebuild_index`` recomputes Stale against
  the horizon and flushes the flip back to frontmatter).
* **IN-19** (salience ordering + pinned outranks) — covered in ``test_recall.py``;
  referenced, not duplicated.
* **exit-0 + source-aware skip** — covered in ``test_hooks.py``; referenced.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import pytest

from claudemem import cli, config, files, hooks, index
from claudemem.enrich import backend as backend_mod
from claudemem.enrich.backend import (
    BackendOutcome,
    DeferralEntry,
    EnrichRequest,
    EnrichResult,
    PassiveHit,
    PromotionCandidate,
    ReflectOutcome,
    ReflectRequest,
    SpendEntry,
)
from claudemem.store import forka

NOW = files.iso_to_epoch("2026-05-30T12:00:00Z")


# --------------------------------------------------------------------------- #
# Fixtures (mirror the established test_cli.py / test_hooks.py sandbox)          #
# --------------------------------------------------------------------------- #


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point CLAUDEMEM_HOME (DB files) at tmp_path; cwd → tmp_path for scope."""
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    monkeypatch.delenv(cli.GUARD_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def scoped_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    """Redirect the global + project Fork A memory dirs into tmp_path."""
    global_dir = tmp_path / "global_memory"
    projects_root = tmp_path / "projects"
    global_dir.mkdir(parents=True, exist_ok=True)
    projects_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(config, "GLOBAL_MEMORY_DIR", global_dir)
    monkeypatch.setattr(config, "PROJECTS_ROOT", projects_root)
    return global_dir, projects_root


def _scope_ctx() -> config.ScopeContext:
    """The cwd-derived project scope_ctx used to read the index back."""
    return config.resolve_scope(Path.cwd(), None, None)


def _project_dir(scoped_dirs: tuple[Path, Path]) -> Path:
    """The cwd-derived project memory dir a project-scope save writes into."""
    _global_dir, projects_root = scoped_dirs
    scope_ctx = _scope_ctx()
    assert scope_ctx.project_id is not None
    return projects_root / scope_ctx.project_id / "memory"


# --------------------------------------------------------------------------- #
# Fake backends (no model call, no claude spawn)                                 #
# --------------------------------------------------------------------------- #


class FakeEnrichBackend:
    """Model-free fake matching the EnrichmentBackend protocol.

    ``mode='enrich'`` returns one ``new``-verdict result per request (the record
    lands enriched, ``EnrichPending`` cleared); ``mode='defer'`` defers every
    record (the SC-10/SC-3 degraded-save path, ``EnrichPending=1``). A
    ``conflict_for`` map turns the named record's verdict into a ``conflict``
    against a supplied target (drives the SC-7 conflict-supersede angle).
    """

    def __init__(
        self,
        mode: str = "enrich",
        *,
        conflict_for: dict[str, str] | None = None,
    ) -> None:
        self.mode = mode
        self.conflict_for = conflict_for or {}
        self.calls: list[list[EnrichRequest]] = []

    @staticmethod
    def detect() -> bool:
        return True

    def enrich_batch(self, reqs: list[EnrichRequest]) -> BackendOutcome:
        self.calls.append(reqs)
        if self.mode == "defer":
            return BackendOutcome(
                results=[],
                deferred=[
                    DeferralEntry(record_id=r.record_id, reason="auth") for r in reqs
                ],
                spend=[],
            )
        results: list[EnrichResult] = []
        for r in reqs:
            target = self.conflict_for.get(r.record_id)
            results.append(
                EnrichResult(
                    record_id=r.record_id,
                    summary=f"summary for {r.name}",
                    aliases=[r.name],
                    dedup_verdict="conflict" if target else "new",
                    dedup_target_name=target,
                    conflict_explanation="contradicts the prior record"
                    if target
                    else None,
                )
            )
        spend = [
            SpendEntry(call_site="save", model="haiku", backend="cli", output_tokens=10)
            for _ in reqs
        ]
        return BackendOutcome(results=results, deferred=[], spend=spend)

    def reflect(self, req: ReflectRequest) -> ReflectOutcome:
        return ReflectOutcome()


class FakeReflectBackend:
    """A fake whose ``reflect`` returns a scripted outcome (passive hits +
    promotion candidates); ``enrich_batch`` defers (no model)."""

    def __init__(self, outcome: ReflectOutcome) -> None:
        self.outcome = outcome
        self.reflect_reqs: list[ReflectRequest] = []

    @staticmethod
    def detect() -> bool:
        return True

    def enrich_batch(self, reqs: list[EnrichRequest]) -> BackendOutcome:
        return BackendOutcome(
            results=[],
            deferred=[DeferralEntry(record_id=r.record_id, reason="auth") for r in reqs],
            spend=[],
        )

    def reflect(self, req: ReflectRequest) -> ReflectOutcome:
        self.reflect_reqs.append(req)
        return self.outcome


def _inject(monkeypatch: pytest.MonkeyPatch, backend: object) -> None:
    """Inject ``backend`` as the resolved enrichment backend (the test seam)."""
    monkeypatch.setattr(backend_mod, "select_backend", lambda settings: backend)
    backend_mod._reset_cache_for_tests()


@pytest.fixture
def fake_enrich(monkeypatch: pytest.MonkeyPatch) -> FakeEnrichBackend:
    """Inject an enriching fake so save/import/reindex never call a real model."""
    fake = FakeEnrichBackend(mode="enrich")
    _inject(monkeypatch, fake)
    return fake


# --------------------------------------------------------------------------- #
# Save / index read helpers                                                      #
# --------------------------------------------------------------------------- #


def _save(content: str, **kw: object) -> int:
    argv = ["save", content]
    for k, v in kw.items():
        argv += [f"--{k}", str(v)]
    return cli.main(argv)


def _row(name: str, columns: str):  # type: ignore[no-untyped-def]
    """Fetch ``columns`` for record ``name`` from the live index (or None)."""
    conn = index.open_forkA()
    try:
        return conn.execute(
            f"SELECT {columns} FROM Record WHERE Name = ?;", (name,)
        ).fetchone()
    finally:
        conn.close()


def _active_names() -> set[str]:
    """Names in the active set (``SupersededBy IS NULL``) for the cwd scope."""
    conn = index.open_forkA()
    try:
        return {rec.name for rec in forka.active_set(conn, _scope_ctx())}
    finally:
        conn.close()


def _search_ids(query: str, capsys: pytest.CaptureFixture[str]) -> str:
    """Run ``search --json`` and return the captured stdout (the id-bearing lines)."""
    capsys.readouterr()
    assert cli.main(["search", query, "--json"]) == 0
    return capsys.readouterr().out


# --------------------------------------------------------------------------- #
# SC-7 — supersede trail (forget + conflict)                                     #
# --------------------------------------------------------------------------- #


def test_sc7_forget_drops_from_active_but_recoverable(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    fake_enrich: FakeEnrichBackend,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-7: ``forget a:<name>`` removes the record from the active search set yet
    leaves a recoverable trail — the row is superseded (never DELETEd), so
    ``select_record`` still returns it with ``SupersededBy`` set, and the file
    trail is preserved on disk."""
    proj_dir = _project_dir(scoped_dirs)
    # Seed distractors so the rare query term clears the relevance floor (a single
    # document corpus collapses bm25 IDF — see test_cli rationale, tech-design §4.6).
    for i in range(40):
        assert _save(f"unrelated filler topic number {i} " * 4, name=f"d{i:02d}") == 0
    assert _save("zephyrquux trail record about zephyrquux retention", name="trailme") == 0
    capsys.readouterr()

    # Present in the active set + findable via search before forget.
    assert "trailme" in _active_names()
    assert '"id":"a:trailme"' in _search_ids("zephyrquux", capsys)

    assert cli.main(["forget", "a:trailme"]) == 0
    capsys.readouterr()

    # Dropped from the active search set...
    assert "trailme" not in _active_names()
    assert '"id":"a:trailme"' not in _search_ids("zephyrquux", capsys)

    # ...but recoverable: the row survives with SupersededBy set (the SC-7 trail).
    conn = index.open_forkA()
    try:
        record = forka.select_record(conn, _scope_ctx(), "trailme")
    finally:
        conn.close()
    assert record is not None
    assert record.superseded_by == "forget"
    assert record.body  # the body is preserved, not destroyed

    # The file trail is preserved on disk too (files-as-truth, SC-4).
    on_disk = (proj_dir / "trailme.md").read_text(encoding="utf-8")
    assert "superseded_by: forget" in on_disk
    assert "trailme" in (proj_dir / "trailme.md").read_text(encoding="utf-8")


def test_sc7_conflict_supersede_drops_from_active_history_intact(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-7: a conflict-superseded record is likewise dropped from the active set,
    while its history stays intact (the row is superseded, not deleted; both the
    index row and the file trail survive)."""
    # First save lands the target as a plain ``new`` record.
    _inject(monkeypatch, FakeEnrichBackend(mode="enrich"))
    assert _save("old shipping address is 1 old street", name="old-addr") == 0
    capsys.readouterr()
    assert "old-addr" in _active_names()

    # The next save returns a ``conflict`` verdict targeting the prior record, so
    # the routine supersedes ``old-addr`` (SupersededBy = the new record's name).
    _inject(
        monkeypatch,
        FakeEnrichBackend(mode="enrich", conflict_for={"new-addr": "old-addr"}),
    )
    assert _save("new shipping address is 2 new avenue", name="new-addr") == 0
    out = capsys.readouterr().out
    # The conflict is surfaced NON-BLOCKING (a resolution line, exit 0).
    assert "conflict" in out.lower()

    # The superseded target dropped from the active set; the new record is active.
    active = _active_names()
    assert "old-addr" not in active
    assert "new-addr" in active

    # History intact: the superseded row still exists, pointing at the superseding
    # record's name (recoverable trail), and the file carries the same trail.
    conn = index.open_forkA()
    try:
        superseded = forka.select_record(conn, _scope_ctx(), "old-addr")
    finally:
        conn.close()
    assert superseded is not None
    assert superseded.superseded_by == "new-addr"
    proj_dir = _project_dir(scoped_dirs)
    assert "superseded_by: new-addr" in (proj_dir / "old-addr.md").read_text(
        encoding="utf-8"
    )


# --------------------------------------------------------------------------- #
# SC-8 — promotion gate: never auto-promote                                      #
# --------------------------------------------------------------------------- #


def _seed_forkb_row(body: str = "a recurring fact worth promoting") -> int:
    """Append one Fork B activity row; return its rowid (the b:<rowid> id)."""
    from claudemem.store import forkb

    conn_b = index.open_forkB()
    try:
        return forkb.append_activity(
            conn_b, session_id="S1", ts=NOW, role="user", kind="prompt", body=body
        )
    finally:
        conn_b.close()


def test_sc8_reflection_proposes_never_auto_promotes(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SC-8: a reflection that proposes a promotion candidate does NOT create a
    Fork A record — it is proposed only. No new Fork A record appears without an
    explicit ``promote``."""
    rowid = _seed_forkb_row()
    before = _active_names()

    backend = FakeReflectBackend(
        ReflectOutcome(
            passive_hits=[],
            promotion_candidates=[
                PromotionCandidate(
                    archive_id=f"b:{rowid}",
                    proposed_summary="a recurring fact worth promoting",
                    rationale="recurs across sessions",
                )
            ],
            spend=[],
        )
    )

    # Drive reflect directly through the SessionEnd hook flow's routine seam.
    from claudemem.enrich import routine

    conn_a = index.open_forkA()
    conn_b = index.open_forkB()
    try:
        result = routine.reflect(
            conn_b, "S1", conn_a, _scope_ctx(), config.load_config(), backend=backend
        )
    finally:
        conn_a.close()
        conn_b.close()

    # The candidate is PROPOSED (surfaced for approval)...
    assert len(result.proposed_promotions) == 1
    assert result.proposed_promotions[0].archive_id == f"b:{rowid}"
    # ...but NOT auto-applied: no new active Fork A record was created.
    assert _active_names() == before


def test_sc8_unattended_session_end_defers_no_forka_record(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SC-8: an unattended SessionEnd (the hook path) with a reflection that
    proposes a promotion candidate defers — it NEVER auto-promotes. After the
    hook runs, no Fork A record exists for the proposed candidate."""
    import io
    import json

    rowid = _seed_forkb_row("promote candidate body for unattended path")
    before = _active_names()

    backend = FakeReflectBackend(
        ReflectOutcome(
            promotion_candidates=[
                PromotionCandidate(
                    archive_id=f"b:{rowid}",
                    proposed_summary="promote candidate body for unattended path",
                    rationale="recurs",
                )
            ],
        )
    )
    _inject(monkeypatch, backend)

    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"session_id": "S1", "cwd": str(home)}))
    )
    assert hooks.dispatch("SessionEnd") == 0

    # The unattended SessionEnd did NOT create any Fork A record (SC-8 / NG-6).
    assert _active_names() == before
    # The reflection actually ran (the candidate was seen), proving the no-promote
    # is a deliberate defer, not a path that never reached reflection.
    assert backend.reflect_reqs


# --------------------------------------------------------------------------- #
# SC-9 — reinforce on confirmed hit ONLY                                         #
# --------------------------------------------------------------------------- #


def test_sc9_used_bumps_hitcount_by_one(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    fake_enrich: FakeEnrichBackend,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-9: ``used a:<name>`` (an active confirmed hit) bumps HitCount by exactly
    one and clears the stale flag."""
    assert _save("reinforceable confirmed-hit fact", name="hitrec") == 0
    capsys.readouterr()
    # Seed a stale flag so we can prove the confirmed hit clears it (SC-13 clear).
    conn = index.open_forkA()
    try:
        with index.write_tx(conn):
            conn.execute("UPDATE Record SET Stale = 1 WHERE Name = 'hitrec';")
    finally:
        conn.close()

    assert cli.main(["used", "a:hitrec"]) == 0
    capsys.readouterr()
    assert _row("hitrec", "HitCount, Stale") == (1, 0)

    # A SECOND confirmed hit bumps by exactly one more (not more, not a reset).
    assert cli.main(["used", "a:hitrec"]) == 0
    capsys.readouterr()
    assert _row("hitrec", "HitCount")[0] == 2


def test_sc9_miss_does_not_bump_any_hitcount(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    fake_enrich: FakeEnrichBackend,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-9 (the miss half — the gap this gate fills): a ``search`` that does NOT
    match, and a ``used`` against an unknown id (a miss), bump NO HitCount.

    Reinforcement is reserved for a confirmed hit (``used <id>`` on a real record
    or a reflection passive hit); a search — match or miss — never reinforces."""
    assert _save("a record nobody confirms a hit on", name="quiet") == 0
    capsys.readouterr()
    assert _row("quiet", "HitCount")[0] == 0

    # A matching search does NOT reinforce — search is read-only (no confirmed hit).
    capsys.readouterr()
    assert cli.main(["search", "record"]) == 0
    capsys.readouterr()
    assert _row("quiet", "HitCount")[0] == 0

    # A no-match search reinforces nothing either.
    assert cli.main(["search", "zzzznomatchquux"]) == 0
    capsys.readouterr()
    assert _row("quiet", "HitCount")[0] == 0

    # A ``used`` against an unknown id is a miss → not found, exit 0, no bump
    # anywhere (the real record's HitCount is untouched).
    assert cli.main(["used", "a:does-not-exist"]) == 0
    assert "not found" in capsys.readouterr().out
    assert _row("quiet", "HitCount")[0] == 0


def test_sc9_reflection_passive_hit_reinforces_at_hooks_level(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    fake_enrich: FakeEnrichBackend,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-9: a reflection passive hit (confirmed, validated against the active set)
    reinforces — asserted at the SessionEnd hooks level. A passive hit naming a
    NON-active / unknown record reinforces nothing (validated-against-set)."""
    import io
    import json

    assert _save("passively-hit fact about deployment", name="passive-rec") == 0
    capsys.readouterr()
    assert _row("passive-rec", "HitCount")[0] == 0

    backend = FakeReflectBackend(
        ReflectOutcome(
            passive_hits=[
                PassiveHit(record_id="a:passive-rec", evidence="referenced in session"),
                # An out-of-set id is dropped — it must NOT bump anything.
                PassiveHit(record_id="a:ghost-record", evidence="not a real record"),
            ],
        )
    )
    _inject(monkeypatch, backend)

    monkeypatch.setattr(
        "sys.stdin", io.StringIO(json.dumps({"session_id": "S1", "cwd": str(home)}))
    )
    assert hooks.dispatch("SessionEnd") == 0

    # The valid passive hit reinforced exactly once; the ghost id reinforced nothing.
    assert _row("passive-rec", "HitCount")[0] == 1
    # No record named ``ghost-record`` was created or reinforced.
    assert "ghost-record" not in _active_names()


# --------------------------------------------------------------------------- #
# SC-10 — warn-not-block (artificially low cap; save still persists)             #
# --------------------------------------------------------------------------- #


def test_sc10_artificially_low_cap_save_still_persists(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    fake_enrich: FakeEnrichBackend,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-10: an artificially low spend cap (config override) → ``save`` STILL
    persists the record and a warning is logged; the save NEVER blocks or exits
    non-zero. Caps are warn-not-block (tech-design §5.10)."""
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

    # Pre-load a spend row so the windowed tally is already over the (tiny) cap.
    from claudemem.store import spend

    conn = index.open_forkA()
    try:
        spend.record_spend(
            conn, call_site="save", model="haiku", backend="cli", output_tokens=100
        )
    finally:
        conn.close()

    with caplog.at_level(logging.WARNING, logger="claudemem"):
        rc = _save("over-cap body still persists", name="overcap")
    capsys.readouterr()

    # The save did NOT block and exited 0 (warn-not-block, never non-zero).
    assert rc == 0
    # A near-/over-cap warning was logged (advisory only).
    assert any("cap" in r.message.lower() for r in caplog.records)
    # The record persisted despite being over cap.
    assert _row("overcap", "Name") is not None


# --------------------------------------------------------------------------- #
# SC-11 — import re-runnable                                                     #
# --------------------------------------------------------------------------- #


def _write_import_source(src: Path, name: str, body: str) -> None:
    """Write one minimal-frontmatter *.md import source file (global scope)."""
    src.mkdir(parents=True, exist_ok=True)
    (src / f"{name}.md").write_text(
        "---\ntype: fact\nscope: global\nsource: explicit\n"
        "created: 2026-01-01T00:00:00Z\nlast_accessed: 2026-01-01T00:00:00Z\n"
        f"aliases: []\n---\n\n{body}\n",
        encoding="utf-8",
    )


def test_sc11_import_twice_no_net_new_with_backend(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    tmp_path: Path,
    fake_enrich: FakeEnrichBackend,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-11: ``import`` re-run on the same source with a backend present produces
    NO net-new duplicate Fork A record — the second run upserts on the natural key
    ``(Scope, ProjectId, Name)``."""
    src = tmp_path / "src"
    _write_import_source(src, "imp-one", "imported body one about widgets")
    _write_import_source(src, "imp-two", "imported body two about gadgets")

    assert cli.main(["import", str(src)]) == 0
    capsys.readouterr()
    count_after_first = _count_active_records()

    # Second run with the (enriching) backend present — re-runnable, no crash.
    assert cli.main(["import", str(src)]) == 0
    capsys.readouterr()
    count_after_second = _count_active_records()

    # No net-new duplicate rows: the upsert on the natural key is idempotent.
    assert count_after_first == count_after_second == 2
    assert _row("imp-one", "Name") is not None
    # Each name resolves to exactly one row (no dupes).
    assert _name_row_count("imp-one") == 1
    assert _name_row_count("imp-two") == 1


def test_sc11_import_offline_persists_pending_then_resolves(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-11: an OFFLINE import (no backend → deferring lexical-only) persists each
    record with ``EnrichPending=1`` (dedup deferred), does not error, and a re-run
    with a backend present resolves cleanly without net-new duplicates."""
    src = tmp_path / "src"
    _write_import_source(src, "off-one", "offline import body about pgvector")

    # Offline: the resolved backend defers every record (lexical-only).
    _inject(monkeypatch, backend_mod.LexicalOnlyBackend())

    assert cli.main(["import", str(src)]) == 0  # does NOT error offline
    capsys.readouterr()
    # Persisted lexical-only with EnrichPending=1 (dedup/enrichment deferred).
    assert _row("off-one", "EnrichPending")[0] == 1
    assert _name_row_count("off-one") == 1

    # A second offline import re-run is still re-runnable (no crash, no net-new
    # dup): the upsert on the natural key folds the re-import into the same row,
    # leaving it pending — the dedup is what defers to a later backend-backed run.
    assert cli.main(["import", str(src)]) == 0
    capsys.readouterr()
    assert _name_row_count("off-one") == 1
    assert _row("off-one", "EnrichPending")[0] == 1

    # And the deferred record resolves once a backend is present: re-importing
    # with an enriching backend clears EnrichPending (the dedup deferred to the
    # later run now runs), still without a net-new duplicate row (SC-11).
    _inject(monkeypatch, FakeEnrichBackend(mode="enrich"))
    assert cli.main(["import", str(src)]) == 0
    capsys.readouterr()
    assert _name_row_count("off-one") == 1
    assert _row("off-one", "EnrichPending")[0] == 0


def _count_active_records() -> int:
    conn = index.open_forkA()
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM Record WHERE SupersededBy IS NULL;"
            ).fetchone()[0]
        )
    finally:
        conn.close()


def _name_row_count(name: str) -> int:
    conn = index.open_forkA()
    try:
        return int(
            conn.execute(
                "SELECT COUNT(*) FROM Record WHERE Name = ?;", (name,)
            ).fetchone()[0]
        )
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# SC-13 — staleness round-trip + horizon sweep                                   #
# --------------------------------------------------------------------------- #


def test_sc13_stale_flag_round_trips_through_reindex(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-13: the ``stale`` flag round-trips faithfully through ``reindex`` for
    records within the staleness horizon. A file authored ``stale: true`` rebuilds
    with Stale=1; a file authored ``stale: false`` (and recently accessed, so the
    horizon sweep leaves it alone) rebuilds with Stale=0. (Files-as-truth: reindex
    carries the within-horizon ``stale`` value through, tech-design §3.9.)"""
    global_dir, _ = scoped_dirs
    # last_accessed safely within the horizon (one day ago, computed against real
    # now) so the staleness-horizon sweep never touches these records — this test
    # exercises the stale-flag round-trip, not the horizon recompute.
    recent = files.epoch_to_iso(int(time.time()) - 86400)
    (global_dir / "stale-rec.md").write_text(
        "---\ntype: fact\nscope: global\nsource: explicit\n"
        f"created: {recent}\nlast_accessed: {recent}\n"
        "summary: has summary\nstale: true\naliases: []\n---\n\nstale body\n",
        encoding="utf-8",
    )
    (global_dir / "fresh-rec.md").write_text(
        "---\ntype: fact\nscope: global\nsource: explicit\n"
        f"created: {recent}\nlast_accessed: {recent}\n"
        "summary: has summary\nstale: false\naliases: []\n---\n\nfresh body\n",
        encoding="utf-8",
    )

    assert cli.main(["reindex", "--no-backfill"]) == 0
    capsys.readouterr()

    assert _row("stale-rec", "Stale")[0] == 1  # stale flag survived the rebuild
    assert _row("fresh-rec", "Stale")[0] == 0  # not-stale survived as not-stale


def test_sc13_used_clears_stale_after_reindex(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-13: a confirmed ``used`` hit clears the ``stale`` flag — even on a record
    that carried ``stale`` through a ``reindex``."""
    global_dir, _ = scoped_dirs
    # Recent last_accessed (within the horizon) so Stale=1 here is the authored
    # flag round-tripping, not a horizon auto-flag — this isolates the used-clears.
    recent = files.epoch_to_iso(int(time.time()) - 86400)
    (global_dir / "rotten.md").write_text(
        "---\ntype: fact\nscope: global\nsource: explicit\n"
        f"created: {recent}\nlast_accessed: {recent}\n"
        "summary: has summary\nstale: true\naliases: []\n---\n\nrotten body\n",
        encoding="utf-8",
    )
    assert cli.main(["reindex", "--no-backfill"]) == 0
    capsys.readouterr()
    assert _row("rotten", "Stale")[0] == 1  # stale after reindex

    # A confirmed hit clears the stale flag in the index AND the file (SC-13).
    assert cli.main(["used", "a:rotten", "--scope", "global"]) == 0
    capsys.readouterr()
    assert _row("rotten", "Stale")[0] == 0
    assert "stale: false" in (global_dir / "rotten.md").read_text(encoding="utf-8")


def test_sc13_staleness_horizon_sweep(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-13 (horizon sweep): a record untouched beyond the configured staleness
    horizon (``settings.staleness.horizon_days``, default 180) auto-acquires
    ``stale`` on ``reindex``. ``store.reindex.rebuild_index`` recomputes Stale
    against the horizon and flushes the flip back to frontmatter (IN-10/IN-16).

    The fixture authors a record with a far-past ``last_accessed`` (2020) and
    ``stale: false``; well past the 180-day horizon, the sweep flips Stale=1 in
    both the index and the markdown file."""
    global_dir, _ = scoped_dirs
    (global_dir / "ancient.md").write_text(
        "---\ntype: fact\nscope: global\nsource: explicit\n"
        "created: 2020-01-01T00:00:00Z\nlast_accessed: 2020-01-01T00:00:00Z\n"
        "summary: has summary\nstale: false\naliases: []\n---\n\nancient body\n",
        encoding="utf-8",
    )
    assert cli.main(["reindex", "--no-backfill"]) == 0
    capsys.readouterr()
    # A record well past the horizon is auto-flagged stale by the sweep, in the
    # index and flushed back to the file (files-as-truth, SC-4).
    assert _row("ancient", "Stale")[0] == 1
    assert "stale: true" in (global_dir / "ancient.md").read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# SC-5 — menu budget (cli-level smoke; full coverage in test_recall.py)          #
# --------------------------------------------------------------------------- #


def test_sc5_menu_cli_smoke_within_budget(
    home: Path,
    scoped_dirs: tuple[Path, Path],
    fake_enrich: FakeEnrichBackend,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """SC-5 (cli smoke; the recall-level budget proof lives in test_recall.py):
    with more than 30 active records, ``cli main menu`` emits ≤30 entry lines,
    ≤10 KB chars, and ≤~600 tokens (chars/4 conservative)."""
    for i in range(45):
        assert _save(f"menu budget record number {i}", name=f"menurec{i:02d}") == 0
    capsys.readouterr()

    assert cli.main(["menu"]) == 0
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert len(lines) <= 30  # SC-5 max_entries
    assert len(out) <= 10_000  # SC-5 hard char ceiling
    assert (len(out) + 3) // 4 <= 600  # SC-5 token ceiling (chars/4, conservative)


# --------------------------------------------------------------------------- #
# SC-12 / IN-19 — referenced, not duplicated                                     #
# --------------------------------------------------------------------------- #
#
# SC-12 (scope merge: a query matching both a global and a project record returns
# both) is certified in ``test_recall.py::test_search_scope_merges_global_and_project``.
# IN-19 (single-factor salience ordering + a pinned old record outranks a fresh
# unpinned one in both search re-rank and menu) is certified in
# ``test_recall.py::test_search_single_factor_importance_orders``,
# ``::test_search_pinned_old_outranks_fresh_unpinned``, and
# ``::test_menu_pinned_old_outranks_fresh_unpinned``. The §10.7 gate references
# those rather than re-running the same recall-level assertions here.
#
# exit-0 + source-aware menu skip are certified in ``test_hooks.py`` (the guard /
# always-exit-0 / SessionStart-resume-no-injection tests) and ``test_cli.py``
# (the SC-3 command-set exit-0 sweep). Referenced, not duplicated.
