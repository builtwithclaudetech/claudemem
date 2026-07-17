"""claudemem.cli — the L4 console entry point + lazy command dispatch (T5.1-5.3).

The `claudemem = claudemem.cli:main` console script (`C-5`). This is the layer
that **wires the whole system together**: it parses argv, resolves scope, opens
the SQLite connections (`cli` OWNS connection opening — `recall`/`enrich` cannot
import `index`), and lazily dispatches to exactly one command handler so a read
command never drags the `enrich`/model path into `sys.modules` (architecture
§2.7, §4; tech-design §6.3).

**The recursion-guard is the literal first statement of :func:`main`** (ahead of
`argparse`, any import, any DB open, any payload read) — `CLAUDEMEM_DISABLE_HOOKS`
set → return 0 immediately (tech-design §6.3 MF-3, §7.1). It must hold even on
malformed argv, so it CANNOT parse argv first; an `argparse`-first design would
``sys.exit(2)`` on bad args and break the exit-0 no-op contract that
``test_recursion_guard`` asserts. `cli` itself is not a hook, but a guarded
context (the spawned ``claude -p`` inner session inherits the env var via the
MF-2 merge) must exit 0 regardless of how it is invoked.

**The read/admin/write firewall (architecture §2.7, §4).** Command handlers are
grouped by what they may import:

* **Read flows** (`search`, `get`, `menu`, `log`, `reindex` PHASE A) import only
  `recall`/`store`/`files`/`config` — NEVER `enrich`. The lazy dispatch + the
  per-handler function-local imports are what enforce this at runtime (the
  import-linter forbidden contract enforces it statically on `recall`).
* **Write flows** (`save`, `import`, `promote`, `reindex` PHASE B backfill)
  import `enrich.routine` **inside the handler function** — never at module top
  level (the §2.7 "Must NOT" rule).
* **Admin** (`pin`/`unpin`, `forget`, `used`) are pure `store`/`files`
  mutations, model-free, no `enrich` import.

**SC-3.** No command errors or exits non-zero *solely* because a key/SDK is
absent — degradation is the `enrich` layer's job and a degraded save still
persists (`EnrichPending=1`) and returns 0. Genuine usage errors (bad args) may
exit non-zero (`argparse` returns 2).
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from claudemem.config import ScopeContext, Settings

#: The recursion-guard env var (tech-design §7.1). When set to any non-empty
#: value, every entry point no-ops at exit 0 (§6.3 MF-3).
GUARD_ENV_VAR = "CLAUDEMEM_DISABLE_HOOKS"

_log = logging.getLogger("claudemem")

#: Lazy dispatch table: subcommand → the name of the handler function in THIS
#: module (resolved via ``importlib.import_module(__name__)`` + ``getattr`` at
#: dispatch time, §6.3). The handler itself does its heavy imports
#: (recall/store/enrich) function-locally, so merely building this dict — and
#: importing ``claudemem.cli`` — pulls in none of them. Read handlers
#: (``search``/``get``/``menu``/``log``/``reindex`` phase A) never touch the
#: ``enrich`` import; write handlers import ``enrich.routine`` inside the body.
DISPATCH: dict[str, str] = {
    # Read flows (no enrich import).
    "search": "_cmd_search",
    "get": "_cmd_get",
    "menu": "_cmd_menu",
    "log": "_cmd_log",
    # reindex runs PHASE A (read/flush, model-free) then optional PHASE B
    # (backfill, the ONLY enrich-importing reindex phase — imported lazily).
    "reindex": "_cmd_reindex",
    # Admin (no enrich import).
    "pin": "_cmd_pin",
    "unpin": "_cmd_unpin",
    "forget": "_cmd_forget",
    "used": "_cmd_used",
    # Write flows (lazy enrich.routine import INSIDE the handler).
    "save": "_cmd_save",
    "import": "_cmd_import",
    "promote": "_cmd_promote",
    # Hook dispatch — lazily imports claudemem.hooks (built next task); degrades
    # to a clean exit-0 no-op until that module exists (SC-3).
    "hook": "_cmd_hook",
}


# --------------------------------------------------------------------------- #
# Entry point                                                                   #
# --------------------------------------------------------------------------- #


def main(argv: list[str] | None = None) -> int:
    """The console entry point ``claudemem`` (`C-5`); returns a process exit code.

    The recursion-guard env check is the **literal first statement** — ahead of
    ``argparse``, any import, any DB open, any payload read (tech-design §6.3
    MF-3 / §7.1). This is why it cannot parse argv first: a guarded context may
    be handed malformed argv, and an ``argparse``-first design would
    ``sys.exit(2)`` on bad args, breaking the exit-0 no-op contract.

    After the guard, parse args, configure logging to stderr (so stdout stays
    the clean machine-readable command output, IN-21), and lazily dispatch to
    exactly one handler. A handler returns its own exit code; ``argparse``
    returns 2 on a genuine usage error. A missing key/SDK never raises here
    (SC-3) — that degradation is owned inside the ``enrich`` layer.
    """
    # ── Recursion guard — MUST be first (tech-design §6.3 MF-3, §7.1). ──────
    # Set → no-op exit 0, before ANY argparse/import/DB-open/payload-read, so
    # the exit-0 contract holds even on malformed argv (an argparse-first design
    # would sys.exit(2) on bad args). cli is not itself a hook, but a guarded
    # context (the spawned `claude -p` inner session inherits this env var via
    # the MF-2 merge) must no-op here regardless of how it is invoked.
    if os.environ.get(GUARD_ENV_VAR):
        return 0

    parser = _build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        # argparse exits 2 on a usage error / 0 on --help. A genuine usage error
        # is allowed to be non-zero (SC-3 only forbids non-zero *solely* because
        # a key/SDK is absent). Normalize to an int code.
        return int(exc.code) if isinstance(exc.code, int) else 2

    _configure_logging(args.verbose)

    handler = _resolve_handler(args.command)
    if handler is None:
        parser.print_help(sys.stderr)
        return 2
    return handler(args)


def _resolve_handler(command: str) -> Callable[[argparse.Namespace], int] | None:
    """Resolve a subcommand to its handler via the lazy :data:`DISPATCH` table.

    ``importlib.import_module(__name__)`` returns this already-loaded module;
    the ``getattr`` then yields the handler function. The heavy work (importing
    ``recall``/``store``/``enrich``) happens INSIDE the handler, so resolving a
    read command never pulls the ``enrich`` path into ``sys.modules`` (§6.3).
    """
    handler_name = DISPATCH.get(command)
    if handler_name is None:
        return None
    module = importlib.import_module(__name__)
    return getattr(module, handler_name)  # type: ignore[no-any-return]


# --------------------------------------------------------------------------- #
# argparse construction                                                          #
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    """Build the stdlib ``argparse`` parser (NOT click/typer — cold-start cost, SC-2).

    Every subcommand carries the shared ``--scope``/``--project`` scope override
    (PRD IN-3/IN-18) and a ``--verbose`` toggle. The ``hook`` subcommand is wired
    here so ``claudemem hook <event>`` dispatches cleanly into ``claudemem.hooks``
    once that module exists (next task); see :func:`_cmd_hook`.
    """
    parser = argparse.ArgumentParser(
        prog="claudemem",
        description="Stateless, file-based memory for Claude Code (lexical + Haiku).",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="verbose logging to stderr"
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    def _add_scope_flags(p: argparse.ArgumentParser) -> None:
        p.add_argument("--scope", choices=("global", "project"), default=None)
        p.add_argument("--project", default=None, metavar="<project-id>")

    # ── Read flows ──────────────────────────────────────────────────────────
    p_search = sub.add_parser("search", help="salience-ranked recall (IN-3)")
    p_search.add_argument("query")
    p_search.add_argument("--json", action="store_true", help="JSONL output (IN-21)")
    _add_scope_flags(p_search)

    p_get = sub.add_parser("get", help="full body of one record by id (IN-4)")
    p_get.add_argument("id")
    p_get.add_argument("--json", action="store_true", help="JSON output (IN-21)")
    _add_scope_flags(p_get)

    p_menu = sub.add_parser("menu", help="session-start titles menu (IN-11)")
    p_menu.add_argument(
        "--source", default=None, help="hook source (startup|resume|clear|compact)"
    )
    _add_scope_flags(p_menu)

    p_log = sub.add_parser("log", help="append one model-free Fork B activity row")
    p_log.add_argument("--session-id", required=True)
    p_log.add_argument("--role", default="user")
    p_log.add_argument("--kind", default="message")
    p_log.add_argument("body")

    p_reindex = sub.add_parser("reindex", help="rebuild index from files (IN-10)")
    p_reindex.add_argument(
        "--reenrich-all",
        action="store_true",
        help="re-enrich EVERY record in backfill, not just EnrichPending (§3.9; off by default)",
    )
    p_reindex.add_argument(
        "--no-backfill",
        action="store_true",
        help="run only the model-free PHASE A rebuild; skip PHASE B enrichment backfill",
    )
    _add_scope_flags(p_reindex)

    # ── Admin (model-free) ────────────────────────────────────────────────────
    p_pin = sub.add_parser("pin", help="pin a record (IN-7)")
    p_pin.add_argument("id")
    _add_scope_flags(p_pin)

    p_unpin = sub.add_parser("unpin", help="unpin a record (IN-7)")
    p_unpin.add_argument("id")
    _add_scope_flags(p_unpin)

    p_forget = sub.add_parser("forget", help="soft-delete with a trail (IN-8/SC-7)")
    p_forget.add_argument("id")
    _add_scope_flags(p_forget)

    p_used = sub.add_parser("used", help="confirmed-hit reinforcement (IN-6/SC-9)")
    p_used.add_argument("id")
    _add_scope_flags(p_used)

    # ── Write flows (handlers lazy-import enrich.routine) ─────────────────────
    p_save = sub.add_parser("save", help="create/update a Fork A record (IN-5)")
    p_save.add_argument("content")
    p_save.add_argument("--name", default=None, help="record name (a:<name>); derived from content if absent")
    p_save.add_argument("--type", default="reference")
    p_save.add_argument("--importance", type=int, default=3)
    p_save.add_argument(
        "--resolve",
        choices=("replace", "keep-both", "supersede"),
        default=None,
        help="resolution for a surfaced conflict (non-blocking; re-issued separately)",
    )
    _add_scope_flags(p_save)

    p_import = sub.add_parser("import", help="bulk markdown ingest (IN-17)")
    p_import.add_argument("path")
    _add_scope_flags(p_import)

    p_promote = sub.add_parser("promote", help="distill a Fork B entry into Fork A (IN-9)")
    p_promote.add_argument("id")
    _add_scope_flags(p_promote)

    # ── Hook dispatch (delegates to claudemem.hooks, built next task) ─────────
    p_hook = sub.add_parser("hook", help="dispatch a Claude Code hook event")
    p_hook.add_argument("event")
    p_hook.set_defaults(_is_hook=True)

    return parser


def _configure_logging(verbose: bool) -> None:
    """Send human logs to **stderr** so stdout stays the clean IN-21 contract.

    Idempotent across re-entry (a guarded test that calls ``main`` repeatedly in
    one process must not stack handlers). Defaults to WARNING; ``--verbose``
    raises it to INFO. The near-cap warnings and SC-3 degradation notices go
    through this logger, never to stdout.
    """
    root = logging.getLogger("claudemem")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        root.addHandler(handler)
    root.setLevel(logging.INFO if verbose else logging.WARNING)


# --------------------------------------------------------------------------- #
# Shared helpers                                                                 #
# --------------------------------------------------------------------------- #


def _scope(args: argparse.Namespace) -> ScopeContext:
    """Resolve the active scope from cwd + the shared ``--scope``/``--project`` flags."""
    from claudemem import config

    return config.resolve_scope(Path.cwd(), args.project, args.scope)


def _settings() -> Settings:
    """Load the frozen settings (locked defaults when ``config.toml`` is absent)."""
    from claudemem import config

    return config.load_config()


def _emit(text: str) -> None:
    """Print one command's stdout payload (the IN-21 contract surface).

    The empty string (no results / clean no-op) prints nothing — Claude sees a
    clean empty result, not a blank line, matching the recall serializers'
    empty-result convention.
    """
    if text:
        print(text)


# --------------------------------------------------------------------------- #
# Read handlers (NO enrich import) — T5.2                                        #
# --------------------------------------------------------------------------- #


def _cmd_search(args: argparse.Namespace) -> int:
    """``search <query>`` — salience-ranked recall, Fork A → Fork B fallback (IN-3).

    Opens BOTH connections (``conn_b`` is non-optional — the archive fallback can
    fire on any query, search.py module docstring) and threads them into
    ``recall.search``. Model-free by construction (SC-3): a missing key/SDK
    changes nothing on this path, and no ``enrich`` import occurs.
    """
    from claudemem import index
    from claudemem.recall import search as search_mod

    scope_ctx = _scope(args)
    conn_a = index.open_forkA()
    conn_b = index.open_forkB()
    try:
        out = search_mod.search(
            conn_a, conn_b, args.query, scope_ctx, json=args.json, settings=_settings()
        )
    finally:
        conn_a.close()
        conn_b.close()
    _emit(out)
    return 0


def _cmd_get(args: argparse.Namespace) -> int:
    """``get <id>`` — full body of one record by unified id (IN-4).

    Threads ``conn_a`` (the ``a:`` path + the index-only access refresh) and
    ``conn_b`` (the ``b:`` archive path). An unparseable / unknown / pruned id
    resolves to a clean "not found" line, exit 0 (SC-3) — never raises.
    """
    from claudemem import index
    from claudemem.recall import get as get_mod

    scope_ctx = _scope(args)
    conn_a = index.open_forkA()
    conn_b = index.open_forkB()
    try:
        out = get_mod.get(conn_a, conn_b, args.id, scope_ctx, json=args.json)
    finally:
        conn_a.close()
        conn_b.close()
    _emit(out)
    return 0


def _cmd_menu(args: argparse.Namespace) -> int:
    """``menu`` — the session-start titles-only menu (IN-11, SC-5).

    Reads only Fork A (``menu`` does no FTS MATCH and consults no other store).
    ``--source resume`` yields the empty string (no injection on a resumed
    session, IN-11). Model-free, no ``enrich`` import.
    """
    from claudemem import index
    from claudemem.recall import menu as menu_mod

    scope_ctx = _scope(args)
    conn_a = index.open_forkA()
    try:
        out = menu_mod.menu(conn_a, scope_ctx, args.source, settings=_settings())
    finally:
        conn_a.close()
    _emit(out)
    return 0


def _cmd_log(args: argparse.Namespace) -> int:
    """``log`` — append one **model-free** Fork B activity row.

    The transcript-driven Fork B logging is owned by the ``Stop`` hook
    (``claudemem.hooks``, next task) — this subcommand is the direct
    forkb append path that ``hooks`` also calls. It stays model-free and imports
    no ``enrich`` (architecture §2.4 / §4). ``--session-id`` is required (Fork B
    rows are session-scoped, §7.4); the body is capped/tool-skipped at write time
    inside ``store.forkb.append_activity``.
    """
    from claudemem import index
    from claudemem.store import forkb

    conn_b = index.open_forkB()
    try:
        forkb.append_activity(
            conn_b,
            session_id=args.session_id,
            ts=int(time.time()),
            role=args.role,
            kind=args.kind,
            body=args.body,
            settings=_settings(),
        )
    finally:
        conn_b.close()
    return 0


def _cmd_reindex(args: argparse.Namespace) -> int:
    """``reindex`` — PHASE A model-free rebuild, then optional PHASE B backfill (IN-10).

    PHASE A (``store.reindex.rebuild_index``) is model-free and imports no
    ``enrich`` — it is the read-path side of the SC-6 firewall. PHASE B (the
    enrichment backfill) is the ONLY enrich-importing reindex phase and is
    delegated to :func:`_reindex_backfill`, which imports ``enrich.routine``
    lazily inside its own body. ``--no-backfill`` runs only PHASE A; the model-
    free rebuild therefore never imports ``enrich`` on that path.
    """
    from claudemem.store import reindex

    scope_ctx = _scope(args)
    settings = _settings()

    # PHASE A — model-free rebuild from files (no enrich import on this path).
    result = reindex.rebuild_index(scope_ctx, settings=settings)
    _log.info(
        "reindex PHASE A: %d records rebuilt, %d enrich-pending",
        result.records,
        result.enrich_pending,
    )

    if args.no_backfill:
        _emit(f"reindex: {result.records} records; backfill skipped")
        return 0

    # PHASE B — enrichment backfill (the ONLY enrich-importing reindex phase).
    backfilled, remaining = _reindex_backfill(
        scope_ctx, settings, reenrich_all=args.reenrich_all
    )
    _emit(
        f"reindex: {result.records} records; "
        f"backfilled {backfilled}, {remaining} still enrich-pending"
    )
    return 0


# --------------------------------------------------------------------------- #
# Admin handlers (model-free, NO enrich import) — T5.2                           #
# --------------------------------------------------------------------------- #


def _resolve_forka_name(raw_id: str) -> str | None:
    """Parse a unified id and return the Fork A ``name`` (or ``None`` if not ``a:``).

    Admin commands (``pin``/``unpin``/``forget``) act only on Fork A records, so
    a non-``a:`` id (or an unparseable one) is not a valid target. ``recall.output``
    owns the id contract; a parse failure becomes ``None`` here (the caller maps
    it to a clean "not found", exit 0, SC-3).
    """
    from claudemem.recall import output

    try:
        fork, key = output.parse_id(raw_id)
    except output.InvalidId:
        return None
    return key if fork == "a" else None


def _set_pinned(args: argparse.Namespace, pinned: bool) -> int:
    """Toggle the ``Pinned`` flag on a Fork A record in the index + frontmatter (IN-7).

    The index UPDATE (under ``BEGIN IMMEDIATE``) keeps RecordFts in sync via the
    §3.3 triggers; the file frontmatter is flushed too so the files-as-truth
    contract (SC-4) survives a later ``reindex``. An unknown name → clean "not
    found", exit 0 (SC-3).
    """
    from claudemem import index
    from claudemem.store import forka

    name = _resolve_forka_name(args.id)
    scope_ctx = _scope(args)
    if name is None:
        _emit(f"not found: {args.id}")
        return 0

    conn_a = index.open_forkA()
    try:
        record = forka.select_record(conn_a, scope_ctx, name)
        if record is None:
            _emit(f"not found: {args.id}")
            return 0
        forka.set_pinned(conn_a, record.id, pinned)
    finally:
        conn_a.close()

    _flush_pinned_to_file(scope_ctx, name, pinned)
    _emit(f"{'pinned' if pinned else 'unpinned'}: a:{name}")
    return 0


def _flush_pinned_to_file(scope_ctx: ScopeContext, name: str, pinned: bool) -> None:
    """Flush the toggled ``pinned`` flag into the record's markdown (SC-4, best-effort).

    A hand-deleted file is not an error (C-11): the index already recorded the
    flag and the next ``reindex`` is the backstop. ``dataclasses.replace`` keeps
    every other frontmatter field intact.
    """
    from dataclasses import replace

    from claudemem import files

    for directory in (scope_ctx.project_dir, scope_ctx.global_dir):
        if directory is None:
            continue
        path = directory / f"{name}.md"
        if path.is_file():
            files.write_record(replace(files.read_record(path), pinned=pinned))
            return


def _cmd_pin(args: argparse.Namespace) -> int:
    """``pin <id>`` — set the immortal ranking flag (IN-7)."""
    return _set_pinned(args, True)


def _cmd_unpin(args: argparse.Namespace) -> int:
    """``unpin <id>`` — clear the ranking flag (IN-7)."""
    return _set_pinned(args, False)


def _cmd_forget(args: argparse.Namespace) -> int:
    """``forget <id>`` — soft-delete with a trail (IN-8/SC-7; supersede-not-delete).

    ``store.mark_superseded`` retires the row from the active set (the row is
    NEVER DELETEd — the SC-7 trail survives) and ``files.set_superseded`` writes
    the same trail into the markdown. ``superseded_by`` is set to a sentinel
    (``"forget"``) so the soft-delete is distinguishable from a conflict-driven
    supersede. An unknown id → clean "not found", exit 0 (SC-3).
    """
    from claudemem import files, index
    from claudemem.store import forka

    name = _resolve_forka_name(args.id)
    scope_ctx = _scope(args)
    if name is None:
        _emit(f"not found: {args.id}")
        return 0

    conn_a = index.open_forkA()
    try:
        record = forka.select_record(conn_a, scope_ctx, name)
        if record is None:
            _emit(f"not found: {args.id}")
            return 0
        forka.mark_superseded(conn_a, name, "forget", scope_ctx)
    finally:
        conn_a.close()

    target_file = _read_forka_file(scope_ctx, name)
    if target_file is not None:
        files.set_superseded(target_file, "forget")
    _emit(f"forgotten: a:{name}")
    return 0


def _read_forka_file(scope_ctx: ScopeContext, name: str):  # type: ignore[no-untyped-def]
    """Read a Fork A record's markdown by name from the scope dirs, or ``None``."""
    from claudemem import files

    for directory in (scope_ctx.project_dir, scope_ctx.global_dir):
        if directory is None:
            continue
        path = directory / f"{name}.md"
        if path.is_file():
            return files.read_record(path)
    return None


def _cmd_used(args: argparse.Namespace) -> int:
    """``used <id>`` — confirmed-hit reinforcement (IN-6/SC-9).

    Model-free, no ``enrich`` import. Dispatch by fork (``recall.output.parse_id``):

    * ``a:<name>`` → bump ``HitCount`` by exactly one + clear the ``stale`` flag
      in the INDEX, then flush ``hit_count`` + cleared ``stale`` to frontmatter
      (SC-9 reinforce-on-confirmed-hit; SC-13 stale clear). Unknown name → clean
      "not found", exit 0.
    * ``b:<rowid>`` → the IN-15 Fork B promotion-hit signal. Fork B's frozen §3.5
      schema has no per-row hit counter (promotion is reflection-driven, IN-14/
      IN-15), so the model-free ``used b:`` action is to acknowledge a present
      in-window row (and a pruned/unknown rowid → clean "not found"), exit 0 in
      both cases. The signal is surfaced; the promotion decision stays with the
      reflection/reindex gate (SC-8).
    """
    from claudemem.recall import output

    try:
        fork, key = output.parse_id(args.id)
    except output.InvalidId:
        _emit(f"not found: {args.id}")
        return 0

    if fork == "a":
        return _used_forka(args, key)
    return _used_forkb(key, args.id)


def _used_forka(args: argparse.Namespace, name: str) -> int:
    """Fork A confirmed-hit: HitCount +1 + clear stale, index then frontmatter (SC-9/SC-13)."""
    from claudemem import files, index
    from claudemem.store import forka

    scope_ctx = _scope(args)
    conn_a = index.open_forkA()
    try:
        record = forka.select_record(conn_a, scope_ctx, name)
        if record is None:
            _emit(f"not found: {args.id}")
            return 0
        new_hit_count = forka.bump_hit(conn_a, record.id)
    finally:
        conn_a.close()

    target_file = _read_forka_file(scope_ctx, name)
    if target_file is not None:
        updated = files.writeback_counters(target_file, hit_count=new_hit_count)
        if updated is not None:
            files.set_stale(updated, False)
    _emit(f"reinforced: a:{name} (hit_count={new_hit_count})")
    return 0


def _used_forkb(key: str, raw_id: str) -> int:
    """Fork B promotion-hit signal (IN-15); pruned/unknown/non-numeric → not found, exit 0."""
    from claudemem import index
    from claudemem.store import forkb

    try:
        rowid = int(key)
    except ValueError:
        _emit(f"not found: {raw_id}")
        return 0

    conn_b = index.open_forkB()
    try:
        row = forkb.get_activity(conn_b, rowid)
    finally:
        conn_b.close()
    if row is None:
        _emit(f"not found: {raw_id}")
        return 0
    _emit(f"promotion-hit signalled: b:{rowid}")
    return 0


# --------------------------------------------------------------------------- #
# Write handlers (lazy-import enrich.routine) — T5.3                              #
# --------------------------------------------------------------------------- #


def _slug(text: str) -> str:
    """Derive a filename-stem slug (the ``a:<name>`` id) from free-text content.

    Used when ``save`` is given no explicit ``--name``: lowercase the first line,
    keep alphanumerics, collapse other runs to ``-``, trim, and cap the length so
    the stem stays a sane filename. A degenerate (all-punctuation) first line
    falls back to ``"note"`` so a name always exists.
    """
    import re

    first_line = text.strip().splitlines()[0] if text.strip() else ""
    slug = re.sub(r"[^a-z0-9]+", "-", first_line.lower()).strip("-")[:60]
    return slug or "note"


def _build_record_file(args: argparse.Namespace, scope_ctx: ScopeContext):  # type: ignore[no-untyped-def]
    """Build the frontmatter scaffold :class:`files.RecordFile` for ``save`` (§5.1).

    The model-free scaffold: ISO-8601 timestamps (MF-1 boundary, ``files`` owns
    the epoch crossing at index time), the scope's frontmatter ``scope`` kind, an
    empty summary/aliases (enrichment fills these — a degraded save leaves them
    empty and flags ``EnrichPending=1``). The record path lands in the scope's
    project dir (project scope) or the global dir, matching ``iter_records``.
    """
    from claudemem import files

    name = args.name or _slug(args.content)
    now_iso = files.epoch_to_iso(int(time.time()))
    directory = (
        scope_ctx.project_dir
        if scope_ctx.kind == "project" and scope_ctx.project_dir is not None
        else scope_ctx.global_dir
    )
    return files.RecordFile(
        path=directory / f"{name}.md",
        name=name,
        type=args.type,
        scope=scope_ctx.kind,
        importance=args.importance,
        pinned=False,
        source="explicit",
        created=now_iso,
        last_accessed=now_iso,
        access_count=0,
        hit_count=0,
        summary=None,
        aliases=[],
        superseded_by=None,
        stale=False,
        body=args.content,
    )


def _warn_spend(conn_a: sqlite3.Connection, settings: Settings) -> None:
    """Log near-/over-cap warnings (warn-not-block, §5.10 / SC-10) — the cap throttle.

    This is where the Phase-4 carry-forward ``cap`` throttle is wired into ``cli``
    (architecture §5.10): compute the daemonless ET-windowed tally and log every
    advisory string. ``near_cap_warnings`` NEVER raises and NEVER blocks — a save
    over the cap still persists (the ``enrich`` routine defers the enrichment,
    marking ``EnrichPending=1``), so this is purely advisory output to stderr.
    """
    from claudemem.store import spend

    tally = spend.spend_tally(conn_a, tz_name=settings.spend.window_tz)
    for warning in spend.near_cap_warnings(tally, settings):
        _log.warning("claudemem spend: %s", warning)


def _report_enrich(result, label: str) -> None:  # type: ignore[no-untyped-def]
    """Emit a one-line summary + any NON-BLOCKING conflict options (IN-13, SC-10).

    A surfaced conflict prints its resolution options on stdout but does NOT
    block (no ``input()``); the record already persisted (superseding the target),
    and a resolution is re-issued via a separate ``--resolve`` invocation. The
    deferred count is the SC-3/SC-10 degraded-save tally (``EnrichPending=1``).
    """
    lines = [
        f"{label}: enriched={result.enriched}, "
        f"deferred={len(result.deferred)}, conflicts={len(result.conflicts)}"
    ]
    for conflict in result.conflicts:
        lines.append(
            f"  conflict: a:{conflict.record_name} supersedes a:{conflict.target_name}"
            f" — resolve with --resolve replace|keep-both|supersede"
        )
    _emit("\n".join(lines))


def _cmd_save(args: argparse.Namespace) -> int:
    """``save <content>`` — create/update a Fork A record via the shared routine (IN-5).

    Builds the frontmatter scaffold, lazily imports ``enrich.routine`` (the
    write-flow boundary — never a module-level import, §2.7), and routes the one
    record through ``enrich_batch`` (the single Haiku call: enrich + dedup +
    contradiction, IN-13). The save **always persists** and **never blocks**
    (SC-10): a degraded / over-cap / deferred outcome leaves the record
    ``EnrichPending=1`` and still returns 0 (SC-3). The cap throttle
    (:func:`_warn_spend`) warns before and after but never blocks (§5.10).
    """
    from claudemem import index
    from claudemem.enrich import routine  # lazy — write-flow boundary (§2.7)

    scope_ctx = _scope(args)
    settings = _settings()
    record = _build_record_file(args, scope_ctx)

    conn_a = index.open_forkA()
    try:
        _warn_spend(conn_a, settings)
        result = routine.enrich_batch(conn_a, [record], scope_ctx, settings)
        _warn_spend(conn_a, settings)
    finally:
        conn_a.close()

    _report_enrich(result, f"save a:{record.name}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    """``import <path>`` — bulk markdown ingest via the SHARED routine (IN-17, SC-11).

    Reads each ``*.md`` under ``<path>`` defensively (AS-7 — ``files.read_record``
    tolerates a minimal / alternate frontmatter shape), then routes the whole set
    through ``enrich_batch`` in ONE batch (NOT a per-file model request, SC-6).
    Re-runnable (SC-11): a second run re-upserts on the same natural key and the
    next ``reindex`` resolves any duplicates. Offline → every record persists
    ``EnrichPending=1``. A missing path → clean "no records", exit 0 (SC-3 — not a
    crash). The ``enrich.routine`` import is lazy, inside this write-flow handler.
    """
    from claudemem import files, index
    from claudemem.enrich import routine  # lazy — write-flow boundary (§2.7)

    scope_ctx = _scope(args)
    settings = _settings()
    source = Path(args.path)

    records = []
    if source.is_dir():
        paths = sorted(source.glob("*.md"))
    elif source.is_file():
        paths = [source]
    else:
        paths = []
    for path in paths:
        if path.name == "MEMORY.md":
            continue
        try:
            records.append(files.read_record(path))
        except OSError as exc:
            _log.warning("import: skipping unreadable %s: %s", path, exc)

    if not records:
        _emit(f"import: no records under {args.path}")
        return 0

    conn_a = index.open_forkA()
    try:
        _warn_spend(conn_a, settings)
        result = routine.enrich_batch(conn_a, records, scope_ctx, settings)
        _warn_spend(conn_a, settings)
    finally:
        conn_a.close()

    _report_enrich(result, f"import {len(records)} records")
    return 0


def _cmd_promote(args: argparse.Namespace) -> int:
    """``promote <id>`` — distill a Fork B entry into a Fork A candidate (IN-9).

    Reads the Fork B ``Activity`` body for a ``b:<rowid>`` id, builds a Fork A
    frontmatter scaffold from it (``source='promotion'``), and routes it through
    the SAME ``enrich_batch`` routine (no own model request, SC-6/NG-8). A non-
    ``b:`` id, a non-numeric rowid, or a pruned/unknown row → clean "not found",
    exit 0 (SC-3). The ``enrich.routine`` import is lazy, inside this write-flow
    handler. Promotion is the explicit-trigger half of the SC-8 gate.
    """
    from claudemem import files, index
    from claudemem.enrich import routine  # lazy — write-flow boundary (§2.7)
    from claudemem.recall import output
    from claudemem.store import forkb

    scope_ctx = _scope(args)
    settings = _settings()

    try:
        fork, key = output.parse_id(args.id)
    except output.InvalidId:
        _emit(f"not found: {args.id}")
        return 0
    if fork != "b":
        _emit(f"not a Fork B id: {args.id}")
        return 0
    try:
        rowid = int(key)
    except ValueError:
        _emit(f"not found: {args.id}")
        return 0

    conn_b = index.open_forkB()
    try:
        row = forkb.get_activity(conn_b, rowid)
    finally:
        conn_b.close()
    if row is None or row["Body"] is None:
        _emit(f"not found: {args.id}")
        return 0

    body = row["Body"]
    name = _slug(body)
    now_iso = files.epoch_to_iso(int(time.time()))
    directory = (
        scope_ctx.project_dir
        if scope_ctx.kind == "project" and scope_ctx.project_dir is not None
        else scope_ctx.global_dir
    )
    record = files.RecordFile(
        path=directory / f"{name}.md",
        name=name,
        type="reference",
        scope=scope_ctx.kind,
        importance=3,
        pinned=False,
        source="promotion",
        created=now_iso,
        last_accessed=now_iso,
        access_count=0,
        hit_count=0,
        summary=None,
        aliases=[],
        superseded_by=None,
        stale=False,
        body=body,
    )

    conn_a = index.open_forkA()
    try:
        _warn_spend(conn_a, settings)
        result = routine.enrich_batch(conn_a, [record], scope_ctx, settings)
        _warn_spend(conn_a, settings)
    finally:
        conn_a.close()

    _report_enrich(result, f"promote b:{rowid} -> a:{name}")
    return 0


def _reindex_backfill(
    scope_ctx: ScopeContext, settings: Settings, *, reenrich_all: bool
) -> tuple[int, int]:
    """PHASE B enrichment backfill — the ONLY enrich-importing reindex phase (IN-10).

    Imports ``enrich.routine`` **lazily inside this body** (the §2.7 write-flow
    boundary; PHASE A in :func:`_cmd_reindex` never reaches this import). Selects
    the ``EnrichPending=1`` records (or ALL active records when ``--reenrich-all``,
    §3.9, default off), re-reads their markdown (files-as-truth), and routes them
    through ``enrich_batch`` up to the spend cap. The cap throttle is the warn-
    not-block tally (:func:`_warn_spend`); the remainder carries forward, still
    ``EnrichPending=1``, and is reported (IN-10 convergence). Returns
    ``(backfilled, still_pending)``.
    """
    from claudemem import config, index
    from claudemem.enrich import routine  # lazy — write-flow boundary (§2.7)
    from claudemem.store import forka

    # A global-scope record must be enriched under a GLOBAL scope_ctx (its row
    # has ProjectId NULL), exactly as ``reindex.rebuild_index`` maps each file
    # onto the matching scope — otherwise ``enrich_batch``'s upsert would create
    # a duplicate project-scoped row and leave the original EnrichPending. We
    # therefore group the backfill by the record's own scope.
    global_ctx = config.ScopeContext(
        kind="global",
        project_id=None,
        global_dir=scope_ctx.global_dir,
        project_dir=None,
    )

    conn_a = index.open_forkA()
    try:
        _warn_spend(conn_a, settings)
        rows = forka.pending_names(conn_a, all_active=reenrich_all)

        global_records = []
        project_records = []
        for name, scope_kind in rows:
            target_file = _read_forka_file(scope_ctx, name)
            if target_file is None:
                continue
            if scope_kind == "global":
                global_records.append(target_file)
            else:
                project_records.append(target_file)

        if not global_records and not project_records:
            return 0, 0

        enriched = 0
        if global_records:
            enriched += routine.enrich_batch(
                conn_a, global_records, global_ctx, settings
            ).enriched
        if project_records:
            enriched += routine.enrich_batch(
                conn_a, project_records, scope_ctx, settings
            ).enriched
        _warn_spend(conn_a, settings)

        still_pending = forka.count_pending(conn_a)
    finally:
        conn_a.close()

    return enriched, still_pending


# --------------------------------------------------------------------------- #
# Hook dispatch hook-in point (claudemem.hooks lands next task)                  #
# --------------------------------------------------------------------------- #


def _cmd_hook(args: argparse.Namespace) -> int:
    """``hook <event>`` — dispatch a Claude Code hook event to ``claudemem.hooks``.

    The single hook-in point for the parallel ``claudemem.hooks`` task: when that
    module exists, ``hooks.dispatch(event)`` reads the JSON hook payload from
    stdin and routes to the per-event handler (SessionStart→menu,
    UserPromptSubmit/Stop→log, SessionEnd→reflect). The import is lazy so a read
    command never pulls ``hooks`` (and its eventual ``enrich`` reach for
    SessionEnd) into ``sys.modules``. Until ``hooks.py`` exists this degrades to
    a clean exit 0 (SC-3 — a guarded/headless context must never error), so the
    ``hook`` route is already wired into :data:`DISPATCH`: the moment ``hooks.py``
    lands with a ``dispatch(event) -> int`` function, this delegates to it with no
    edit here.
    """
    try:
        hooks = importlib.import_module("claudemem.hooks")
    except ImportError:
        _log.info("claudemem hook %s: hooks module not present yet (no-op)", args.event)
        return 0
    return int(hooks.dispatch(args.event))


__all__ = ["main", "GUARD_ENV_VAR", "DISPATCH"]


if __name__ == "__main__":  # pragma: no cover — `python -m claudemem.cli` entry
    raise SystemExit(main())
