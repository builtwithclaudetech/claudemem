"""Runtime import-firewall harness + read-path firewall assertions.

This file is the executable counterpart to the static ``import-linter`` contract
in ``pyproject.toml``. It enforces the single most load-bearing property of
ClaudeMem at *runtime* (tech-design §6.2/§10.2, architecture §4.3a, PRD
SC-6/C-17): a read command must reach **no** model transport.

Two transports must be proven absent after every read command:

1. **Anthropic SDK** — ``anthropic`` must not be in the child interpreter's
   ``sys.modules`` (it is only ever function-imported inside the backend
   wrapper, so a clean module-presence check is valid for this transport).
2. **``claude`` subprocess spawn** — a ``sys.modules`` check is *invalid* here
   because stdlib ``subprocess`` is always imported, so its presence proves
   nothing. Instead we PATH-shim a fake ``claude`` executable that writes a
   sentinel file the instant it is invoked, and assert the sentinel is never
   written. This shim is the load-bearing half (tech-design §6.2 MF-1).

State of the world (T5.6): ``claudemem.cli`` and the read-command modules
(``search``/``get``/``menu``/``log``) now exist, so the cli-invocation-level
firewall assertions below are **live GREEN** (the Phase-0 ``xfail`` markers were
removed at T5.6). The harness utilities themselves (``build_fake_claude_shim``,
``run_in_fresh_interpreter``) are real, and a non-xfail self-test proves the
shim works.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

# The four read commands that must run model-free (architecture §4.1, §2.7;
# tech-design §10.2). Each entry is the argv that would follow ``claudemem`` and
# is WELL-FORMED so the handler actually runs (``log`` requires ``--session-id``;
# a bare ``("log",)`` would exit 2 at argparse and never exercise its handler,
# making the firewall assertion vacuous for that command).
READ_COMMANDS: tuple[tuple[str, ...], ...] = (
    ("search", "foo"),
    ("get", "a:x"),
    ("menu",),
    ("log", "--session-id", "FW", "a firewall log line"),
)

#: Read commands whose handler imports the ``recall`` layer (vs ``store`` only).
#: ``log`` is the store-backed read command (``store.forkb``); the others reach
#: ``recall``. Used by the "teeth" half of the firewall assertion below.
_RECALL_BACKED = {"search", "get", "menu"}


# ---------------------------------------------------------------------------
# Fake `claude` PATH shim — real, tested utility.
# ---------------------------------------------------------------------------
def build_fake_claude_shim(tmp_path: Path) -> tuple[Path, Path]:
    """Create a temp dir holding an executable named ``claude`` (the spawn trap).

    The shim writes a sentinel file the instant it is invoked, then exits 0.
    It also handles ``claude auth status`` (tech-design §7.3) — it still drops
    the sentinel so *any* spawn, including an availability probe, is detected —
    and exits 0 there too so a probe cannot be distinguished from a real spawn
    on the trap side.

    Returns ``(shim_dir, sentinel_path)``. Prepend ``shim_dir`` to the child's
    ``PATH``; after a read command, assert ``sentinel_path`` does not exist.
    """
    shim_dir = tmp_path / "fake_bin"
    shim_dir.mkdir(parents=True, exist_ok=True)
    sentinel = tmp_path / "claude_spawned.sentinel"

    shim = shim_dir / "claude"
    # Pure-stdlib shim; records argv for diagnostics, always drops the sentinel,
    # always exits 0 (so `claude auth status` "succeeds" yet is still trapped).
    shim.write_text(
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"open({str(sentinel)!r}, 'a').write(' '.join(sys.argv[1:]) + chr(10))\n"
        "sys.exit(0)\n",
        encoding="utf-8",
    )
    shim.chmod(0o755)
    return shim_dir, sentinel


@dataclass(frozen=True)
class FreshRun:
    """Result of running a command in a fresh interpreter subprocess."""

    returncode: int
    stdout: str
    stderr: str
    imported_modules: frozenset[str]
    spawned_claude: bool


def run_in_fresh_interpreter(
    cmd_args: list[str],
    *,
    extra_path: Path | None = None,
    env: dict[str, str] | None = None,
    sentinel: Path | None = None,
) -> FreshRun:
    """Run ``claudemem <cmd_args>`` in a fresh interpreter with clean module state.

    The command is launched via ``python -c`` shim (not the console script) so
    we can intercept the child's ``sys.modules`` deterministically: the shim
    invokes ``claudemem.cli.main(cmd_args)`` and then dumps the
    ``claudemem``-and-``anthropic``-relevant module keys as JSON on a private
    marker line. This matches the intended ``CLAUDEMEM_DUMP_MODULES`` diagnostic
    (tech-design §6.2) without depending on the cli implementing it yet.

    ``extra_path`` is prepended to the child's ``PATH`` (the fake-``claude``
    shim dir). ``sentinel`` is checked after the run to set ``spawned_claude``.
    """
    child_env = dict(os.environ if env is None else env)
    if extra_path is not None:
        child_env["PATH"] = f"{extra_path}{os.pathsep}{child_env.get('PATH', '')}"

    marker = "__CLAUDEMEM_MODULES__"
    shim = (
        "import sys, json\n"
        "rc = 0\n"
        "try:\n"
        "    from claudemem.cli import main\n"
        "    rc = main(sys.argv[1:])\n"
        "except SystemExit as e:\n"
        "    rc = e.code if isinstance(e.code, int) else 1\n"
        "finally:\n"
        "    relevant = sorted(\n"
        "        m for m in sys.modules\n"
        "        if m == 'anthropic' or m.startswith('anthropic.')\n"
        "        or m == 'claudemem' or m.startswith('claudemem.')\n"
        "    )\n"
        f"    sys.stderr.write({marker!r} + json.dumps(relevant) + chr(10))\n"
        "sys.exit(rc if isinstance(rc, int) else 0)\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", shim, *cmd_args],
        capture_output=True,
        text=True,
        env=child_env,
        timeout=60,
    )

    imported: frozenset[str] = frozenset()
    for line in proc.stderr.splitlines():
        if line.startswith(marker):
            imported = frozenset(json.loads(line[len(marker) :]))
            break

    spawned = bool(sentinel is not None and sentinel.exists())
    return FreshRun(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        imported_modules=imported,
        spawned_claude=spawned,
    )


# ---------------------------------------------------------------------------
# Harness self-tests — REAL (not xfail). These prove the trap actually fires
# so that the xfail firewall assertions below are meaningful once they flip.
# ---------------------------------------------------------------------------
def test_fake_claude_shim_writes_sentinel_when_invoked(tmp_path: Path) -> None:
    """If `claude` is actually spawned, the shim drops its sentinel — proving the
    trap can detect a spawn (so a later 'sentinel absent' assertion has teeth)."""
    shim_dir, sentinel = build_fake_claude_shim(tmp_path)
    assert not sentinel.exists()

    result = subprocess.run(
        ["claude", "--some-flag", "hi"],
        env={
            **os.environ,
            "PATH": f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert sentinel.exists(), "fake claude shim must write its sentinel when spawned"
    assert "--some-flag hi" in sentinel.read_text(encoding="utf-8")


def test_fake_claude_shim_handles_auth_status(tmp_path: Path) -> None:
    """`claude auth status` (the availability probe, tech-design §7.3) is also
    trapped: it still drops the sentinel and exits 0, so a read-path probe is
    detectable, not silently allowed."""
    shim_dir, sentinel = build_fake_claude_shim(tmp_path)

    result = subprocess.run(
        ["claude", "auth", "status"],
        env={
            **os.environ,
            "PATH": f"{shim_dir}{os.pathsep}{os.environ.get('PATH', '')}",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0
    assert sentinel.exists()
    assert "auth status" in sentinel.read_text(encoding="utf-8")


def test_run_in_fresh_interpreter_reports_imported_modules(tmp_path: Path) -> None:
    """The harness correctly captures the child's claudemem/anthropic module set.

    `claudemem.cli` does not exist yet, so the shim's import raises ImportError;
    that is fine for this self-test — we only assert the harness MECHANISM:
    it runs a clean subprocess, returns a FreshRun, and never spawns claude when
    the command does nothing (no sentinel)."""
    shim_dir, sentinel = build_fake_claude_shim(tmp_path)
    run = run_in_fresh_interpreter(["menu"], extra_path=shim_dir, sentinel=sentinel)

    assert isinstance(run, FreshRun)
    # Importing claudemem.cli (absent today) only ever pulls the thin top-level
    # `claudemem` package marker before failing, never `anthropic`.
    assert "anthropic" not in run.imported_modules
    # Nothing was spawned by a failed import.
    assert run.spawned_claude is False
    assert not sentinel.exists()


# ---------------------------------------------------------------------------
# T3.7 — recall-module-level read-path firewall, REAL (not xfail).
#
# The cli does not exist yet (the argv-level READ_COMMANDS tests below stay
# xfail until Phase 5), but the recall read MODULES exist now, so the §10.2
# "read path imports no enrich/anthropic" property is assertable today at the
# import boundary: a fresh interpreter that imports every recall read module
# must NOT pull `anthropic` or `claudemem.enrich` into sys.modules, and a pure
# import must spawn no `claude` subprocess. This is the Phase-3 realization of
# the runtime firewall (tech-design §10.2, architecture §4.3a); the static half
# is the `lint-imports` forbidden contract on `claudemem.recall`.
# ---------------------------------------------------------------------------
#: Every recall read module imported by the T3.7 assertion.
_RECALL_READ_MODULES: tuple[str, ...] = (
    "claudemem.recall.search",
    "claudemem.recall.get",
    "claudemem.recall.menu",
    "claudemem.recall.rank",
    "claudemem.recall.output",
)


def _import_recall_in_fresh_interpreter(
    *, extra_path: Path | None = None, sentinel: Path | None = None
) -> FreshRun:
    """Import the recall read modules in a clean child interpreter (T3.7).

    Mirrors :func:`run_in_fresh_interpreter`'s module-capture mechanism but
    drives a bare ``import`` of each recall read module instead of a (not-yet-
    existing) cli command, so the §10.2 read-path firewall can be asserted at
    the import boundary today. Returns a :class:`FreshRun` with the child's
    ``claudemem``/``anthropic`` ``sys.modules`` keys and the spawn flag.
    """
    child_env = dict(os.environ)
    if extra_path is not None:
        child_env["PATH"] = f"{extra_path}{os.pathsep}{child_env.get('PATH', '')}"

    marker = "__CLAUDEMEM_MODULES__"
    imports = "".join(f"    import {m}\n" for m in _RECALL_READ_MODULES)
    shim = (
        "import sys, json\n"
        "rc = 0\n"
        "try:\n"
        f"{imports}"
        "except Exception:\n"
        "    rc = 1\n"
        "finally:\n"
        "    relevant = sorted(\n"
        "        m for m in sys.modules\n"
        "        if m == 'anthropic' or m.startswith('anthropic.')\n"
        "        or m == 'claudemem' or m.startswith('claudemem.')\n"
        "    )\n"
        f"    sys.stderr.write({marker!r} + json.dumps(relevant) + chr(10))\n"
        "sys.exit(rc)\n"
    )

    proc = subprocess.run(
        [sys.executable, "-c", shim],
        capture_output=True,
        text=True,
        env=child_env,
        timeout=60,
    )

    imported: frozenset[str] = frozenset()
    for line in proc.stderr.splitlines():
        if line.startswith(marker):
            imported = frozenset(json.loads(line[len(marker) :]))
            break
    spawned = bool(sentinel is not None and sentinel.exists())
    return FreshRun(
        returncode=proc.returncode,
        stdout=proc.stdout,
        stderr=proc.stderr,
        imported_modules=imported,
        spawned_claude=spawned,
    )


def test_recall_read_modules_never_import_anthropic_or_enrich(tmp_path: Path) -> None:
    """T3.7: importing every recall read module pulls in neither `anthropic` nor
    `claudemem.enrich` (tech-design §10.2, architecture §4.3a; SC-6/C-17). This
    is the GREEN recall-level realization of the read-path firewall — the cli
    argv-level assertions stay xfail until Phase 5."""
    shim_dir, sentinel = build_fake_claude_shim(tmp_path)
    run = _import_recall_in_fresh_interpreter(extra_path=shim_dir, sentinel=sentinel)

    # The import itself must succeed (the modules exist as of Phase 3).
    assert run.returncode == 0, run.stderr
    # The recall read modules were actually imported (proves the check has teeth).
    assert "claudemem.recall.search" in run.imported_modules
    assert "claudemem.recall.get" in run.imported_modules
    # Neither model transport leaked into the child's module set.
    assert "anthropic" not in run.imported_modules
    assert "claudemem.enrich" not in run.imported_modules
    assert not any(m.startswith("anthropic.") for m in run.imported_modules)
    # A pure import spawns nothing — the fake `claude` shim's sentinel is absent.
    assert run.spawned_claude is False
    assert not sentinel.exists()


# ---------------------------------------------------------------------------
# Read-path firewall assertions at the cli-invocation level — REAL (T5.6).
#
# These were authored xfail in Phase 0 (cli + read-command modules were absent)
# and flipped to live GREEN assertions at T5.6 now that ``claudemem.cli`` and
# the ``search``/``get``/``menu``/``log`` handlers exist and are clean: each read
# command run via the cli in a fresh interpreter imports neither ``anthropic``
# nor ``claudemem.enrich`` and never spawns the PATH-shimmed fake ``claude``.
# The xpass→pass flip the Phase-0 carry-forward asked for: they pass because the
# cli is present and the lazy per-handler imports keep enrich off the read path
# (tech-design §10.2, architecture §4.3a; PRD SC-6/C-17), NOT vacuously because
# the cli is absent.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "cmd_args", [list(c) for c in READ_COMMANDS], ids=[c[0] for c in READ_COMMANDS]
)
def test_read_command_never_imports_anthropic(
    cmd_args: list[str], tmp_path: Path
) -> None:
    """SDK transport absent: after a read command, `anthropic` and
    `claudemem.enrich` are not in the child's sys.modules (tech-design §10.2,
    architecture §4.3a)."""
    shim_dir, sentinel = build_fake_claude_shim(tmp_path)
    # Isolate the store in tmp_path so `log`'s real INSERT never touches the real
    # ~/.claude (the other read commands open the DB read-only too).
    env = dict(os.environ)
    env["CLAUDEMEM_HOME"] = str(tmp_path / "home")
    run = run_in_fresh_interpreter(
        cmd_args, extra_path=shim_dir, env=env, sentinel=sentinel
    )

    # The handler actually ran (rc 0, not an argparse exit-2): the read module it
    # owns is in sys.modules, proving the firewall check has teeth (not vacuous).
    assert run.returncode == 0, run.stderr
    layer = "claudemem.recall" if cmd_args[0] in _RECALL_BACKED else "claudemem.store"
    assert any(m.startswith(layer) for m in run.imported_modules), (
        f"read command {cmd_args} should reach {layer}: "
        f"{sorted(run.imported_modules)}"
    )
    assert "anthropic" not in run.imported_modules
    assert "claudemem.enrich" not in run.imported_modules
    # No anthropic.* submodule leaked in either.
    assert not any(m.startswith("anthropic.") for m in run.imported_modules)


@pytest.mark.parametrize(
    "cmd_args", [list(c) for c in READ_COMMANDS], ids=[c[0] for c in READ_COMMANDS]
)
def test_read_command_never_spawns_claude(cmd_args: list[str], tmp_path: Path) -> None:
    """CLI transport absent (the load-bearing half): a PATH-shimmed fake
    `claude` drops a sentinel iff spawned; after a read command the sentinel is
    never written (tech-design §6.2 MF-1, §10.2; architecture §4.3a)."""
    shim_dir, sentinel = build_fake_claude_shim(tmp_path)
    env = dict(os.environ)
    env["CLAUDEMEM_HOME"] = str(tmp_path / "home")
    run = run_in_fresh_interpreter(
        cmd_args, extra_path=shim_dir, env=env, sentinel=sentinel
    )

    assert run.spawned_claude is False, "read command must never spawn `claude`"
    assert not sentinel.exists()


def test_save_write_command_still_exits_zero_with_fake_claude(tmp_path: Path) -> None:
    """A WRITE command (`save`) MAY reach a transport — that is allowed (§10.2).

    With the fake-claude shim on PATH and no API key, the enrich layer degrades
    (the spawn is the auth-status probe / a deferral) and the save still persists
    lexical-only and exits 0 (SC-3). This is the asymmetric half of the firewall:
    read commands reach zero transports, a write command is permitted to.
    """
    shim_dir, sentinel = build_fake_claude_shim(tmp_path)
    env = dict(os.environ)
    # Isolate the store + memory dirs in tmp_path so the real ~/.claude is
    # untouched and the save has a writable scope.
    env["CLAUDEMEM_HOME"] = str(tmp_path / "home")
    env.pop("ANTHROPIC_API_KEY", None)
    (tmp_path / "home").mkdir(parents=True, exist_ok=True)

    run = run_in_fresh_interpreter(
        ["save", "remember the deploy port is 6380"],
        extra_path=shim_dir,
        env=env,
        sentinel=sentinel,
    )

    # SC-3: a degraded save never errors solely because a key/SDK is absent.
    assert run.returncode == 0, run.stderr
