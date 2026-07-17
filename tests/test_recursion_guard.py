"""Recursion-guard contract — protects SC-1 (runaway) / SC-6 and the spawn loop.

ClaudeMem spawns ``claude -p`` for CLI enrichment; without a guard, that spawned
session would fire ClaudeMem's own hooks -> infinite recursion + Fork B
pollution. The guard has two halves (tech-design §7.1, §6.3 MF-2/MF-3, §10.5):

1. **No-op-on-guard (entrypoint side).** With ``CLAUDEMEM_DISABLE_HOOKS=1`` set,
   every cli/hook entrypoint exits **0 as a no-op** — *before* any argparse
   parse, import, DB open, or payload read. The check is the literal first
   statement of ``main()`` (ahead of argparse), so the exit-0 contract holds
   even when handed malformed argv and malformed stdin (an argparse-first design
   would ``sys.exit(2)`` on bad args and break the contract). This bounds
   recursion at **depth 1**: the outer invocation spawns one guarded
   ``claude -p``; the inner session's hooks all no-op, so no third level.

2. **Guard-env-on-spawn (backend side).** ``ClaudeCliBackend`` sets the guard
   env via a **merge** — ``{**os.environ, "CLAUDEMEM_DISABLE_HOOKS": "1"}`` (MF-2)
   — never a bare ``env={...}`` that would wipe ``PATH``/``HOME`` so the child
   could not even locate ``claude``. The spawn also passes ``--bare``,
   ``--max-turns 1``, and ``--no-session-persistence`` (belt-and-suspenders).

State of the world (T5.6): ``claudemem.cli``, ``claudemem.hooks`` (Phase 5) and
``claudemem.enrich.backend_cli.ClaudeCliBackend`` (Phase 4) now exist, so every
contract below is **live GREEN** — the Phase-0 ``xfail`` markers were removed at
T5.6 and the assertions run against the real entrypoints + backend. A NEGATIVE
guard test (``test_unguarded_hook_does_real_work``) proves the guard is not an
always-no-op: with the env ABSENT, ``user-prompt-submit`` actually appends a
Fork B row, so the exit-0 tests above are not vacuously satisfied.
"""

from __future__ import annotations

import io
import os
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

GUARD_ENV_VAR = "CLAUDEMEM_DISABLE_HOOKS"

# Every cli/hook entrypoint invocation that must no-op under the guard env.
# `claudemem hook <event>` is the single hook dispatch (architecture §2.8);
# the bare `claudemem <read-cmd>` entrypoints share the same first-statement
# guard check (tech-design §6.3).
GUARDED_ENTRYPOINTS: tuple[tuple[str, ...], ...] = (
    ("hook", "SessionStart"),
    ("hook", "UserPromptSubmit"),
    ("hook", "Stop"),
    ("hook", "SessionEnd"),
    ("menu",),
    ("log",),
)

# Argv/stdin that would normally break argparse or payload parsing. The exit-0
# guard must hold REGARDLESS of these (§6.3 MF-3).
# NB: a literal NUL byte (\x00) is NOT used here — an argv element physically
# cannot contain NUL (execve uses NUL-terminated C strings, so the parent
# ``subprocess.run`` rejects it with ``ValueError: embedded null byte`` before
# the child ever starts). That is an OS-undeliverable byte, not an argparse-
# rejectable arg, so it does not exercise the guard. A non-NUL control char is
# OS-deliverable and still unknown to argparse, so it tests the same intent: the
# first-statement guard wins even on bizarre argv (§6.3 MF-3, §10.5).
MALFORMED_ARGV: tuple[list[str], ...] = (
    ["--this-flag-does-not-exist"],
    ["hook"],  # missing required <event>
    ["hook", "NotARealEvent", "--bogus", "x"],
    ["\x01garbage", "--"],  # OS-deliverable control char; argparse rejects it.
)
MALFORMED_STDIN: tuple[str, ...] = (
    "",
    "not json at all",
    "{unterminated",
    "\x00\x01\x02 binary noise",
)


def _run_guarded_entrypoint(
    argv: list[str], stdin: str
) -> subprocess.CompletedProcess[str]:
    """Invoke ``claudemem <argv>`` with the guard env set and given stdin.

    Uses the intended ``claudemem.cli.main`` entrypoint via a ``python -c`` shim
    so the test does not depend on the console script being installed. The guard
    env is a MERGE onto the real environment (mirroring the production spawn
    contract, §7.1 MF-2) so PATH/HOME survive."""
    shim = "import sys\nfrom claudemem.cli import main\nsys.exit(main(sys.argv[1:]))\n"
    return subprocess.run(
        [sys.executable, "-c", shim, *argv],
        input=stdin,
        capture_output=True,
        text=True,
        env={**os.environ, GUARD_ENV_VAR: "1"},
        timeout=60,
    )


# ---------------------------------------------------------------------------
# Half 1: no-op-on-guard (entrypoint side) — REAL GREEN (T5.6; cli/hooks exist).
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "argv",
    [list(e) for e in GUARDED_ENTRYPOINTS],
    ids=["_".join(e) for e in GUARDED_ENTRYPOINTS],
)
def test_guarded_entrypoint_exits_zero_noop(argv: list[str]) -> None:
    """With CLAUDEMEM_DISABLE_HOOKS=1, every entrypoint exits 0 as a no-op on
    well-formed argv (tech-design §7.1, §10.5)."""
    result = _run_guarded_entrypoint(argv, stdin="{}")
    assert result.returncode == 0


@pytest.mark.parametrize(
    "argv", MALFORMED_ARGV, ids=[f"argv{i}" for i in range(len(MALFORMED_ARGV))]
)
def test_guarded_exit_zero_survives_malformed_argv(argv: list[str]) -> None:
    """The exit-0 guard holds for MALFORMED argv: the env check is the first
    statement of main(), ahead of argparse, so bad args never reach a
    sys.exit(2) (tech-design §6.3 MF-3, §10.5)."""
    result = _run_guarded_entrypoint(argv, stdin="{}")
    assert result.returncode == 0


@pytest.mark.parametrize(
    "stdin", MALFORMED_STDIN, ids=[f"stdin{i}" for i in range(len(MALFORMED_STDIN))]
)
def test_guarded_exit_zero_survives_malformed_stdin(stdin: str) -> None:
    """The exit-0 guard holds for MALFORMED stdin: the env check precedes any
    payload read, so a hook handed garbage stdin still no-ops at 0
    (tech-design §6.3 MF-3, §10.5)."""
    result = _run_guarded_entrypoint(["hook", "Stop"], stdin=stdin)
    assert result.returncode == 0


# ---------------------------------------------------------------------------
# Half 2: guard-env-on-spawn (backend side) — REAL GREEN (T5.6; backend exists).
# ---------------------------------------------------------------------------
def test_cli_backend_spawn_merges_guard_env_and_sets_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A ClaudeCliBackend spawn sets the guard env via MERGE (not replace) and
    passes the belt-and-suspenders flags (tech-design §7.1 MF-2, §10.5).

    Injects a fake runner via the backend's documented ``runner=`` spawn seam
    (the production default ``runner`` is bound to ``subprocess.run`` at
    definition time, so monkeypatching the module attribute would not reach it —
    the seam is the supported interception point). The fake captures the env +
    argv the backend hands to ``claude -p`` and returns a well-formed
    ``--output-format json`` envelope. Asserts:
      - env is a merge: PATH/HOME survive AND CLAUDEMEM_DISABLE_HOOKS == "1"
        (a bare env={...} that wiped PATH/HOME is the bug this guards against).
      - argv carries --bare, --max-turns 1, --no-session-persistence.
    """
    import json

    from claudemem.enrich.backend import EnrichRequest
    from claudemem.enrich.backend_cli import ClaudeCliBackend

    captured: dict[str, object] = {}

    def fake_run(cmd, *args, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env") or {})
        # A well-formed `claude -p --output-format json` envelope: `_parse_envelope`
        # requires a top-level object with a string `result`. The `result` carries
        # the model's JSON array text; an empty array is a valid (if empty) parse.
        envelope = {"result": "[]", "usage": {"input_tokens": 0, "output_tokens": 0}}
        return subprocess.CompletedProcess(
            cmd, 0, stdout=json.dumps(envelope), stderr=""
        )

    # Ensure PATH/HOME are present in the parent env so a merge can be observed.
    monkeypatch.setenv("PATH", os.environ.get("PATH", "/usr/bin"))
    monkeypatch.setenv("HOME", os.environ.get("HOME", "/root"))

    backend = ClaudeCliBackend(runner=fake_run)
    backend.enrich_batch(
        [EnrichRequest(record_id="a:x", name="x", body="hello", candidates=[])]
    )

    env = captured["env"]
    cmd = captured["cmd"]
    assert isinstance(env, dict) and isinstance(cmd, list)

    # MF-2: merge, not replace — guard var set AND parent PATH/HOME survived.
    assert env.get(GUARD_ENV_VAR) == "1"
    assert "PATH" in env and env["PATH"], (
        "spawn env must inherit PATH (merge, not bare env)"
    )
    assert "HOME" in env, "spawn env must inherit HOME (merge, not bare env)"

    # §7.1 belt-and-suspenders flags. `--bare` is intentionally absent: it forces
    # API-key-only auth and breaks subscription/OAuth enrichment (the billing path).
    assert "--bare" not in cmd
    assert "--no-session-persistence" in cmd
    # --max-turns 1 may be one token or split across two argv entries.
    joined = " ".join(str(c) for c in cmd)
    assert "--max-turns 1" in joined or ("--max-turns" in cmd and "1" in cmd)


# ---------------------------------------------------------------------------
# Depth-1 termination — documented reasoning encoded as a REAL test (T5.6). The
# outer invocation spawns ONE guarded `claude -p`; the inner session's hooks all
# no-op under the env var; there is no third level.
# ---------------------------------------------------------------------------
def test_depth1_termination_inner_hooks_all_noop() -> None:
    """Depth-1 proof (tech-design §7.1): under the guard env (the environment a
    spawned `claude -p` inherits via the MF-2 merge), EVERY hook event no-ops at
    exit 0. Since no inner hook does work, none can spawn a further `claude`, so
    recursion is bounded at depth 1."""
    for event in ("SessionStart", "UserPromptSubmit", "Stop", "SessionEnd"):
        result = _run_guarded_entrypoint(["hook", event], stdin="{}")
        assert result.returncode == 0, (
            f"inner hook {event} must no-op at exit 0 under guard"
        )


# ---------------------------------------------------------------------------
# NEGATIVE guard test (Phase-0 carry-forward) — proves the guard is NOT an
# always-no-op. With the env ABSENT, a hook entrypoint does REAL work (opens the
# DB + appends a Fork B Activity row). Without this, the exit-0 tests above would
# be vacuously satisfied by a main() that always returns 0 regardless of the env.
# Driven in-process against ``claudemem.hooks.dispatch`` with CLAUDEMEM_HOME and
# the memory dirs isolated into tmp_path (no real ~/.claude, no real claude spawn
# — UserPromptSubmit is model-free, NO enrich import).
# ---------------------------------------------------------------------------
def test_unguarded_hook_does_real_work(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guard ABSENT → ``user-prompt-submit`` actually appends a Fork B row.

    This is the teeth behind the exit-0 contract: the guard short-circuits REAL
    work, so with it unset the same entrypoint must perform that work (here, one
    ``Activity`` INSERT). If the guard were an always-no-op, this would write
    nothing and the assertion would fail — proving the exit-0 tests above are not
    vacuous (SC-1 bounded recursion is a real guard, not a dead branch)."""
    from claudemem import config, hooks, index

    # Isolate the store + cwd-derived scope into tmp_path; ensure the guard is
    # genuinely ABSENT for this entrypoint (the negative half of §10.5).
    monkeypatch.setenv(config.CONFIG_HOME_ENV, str(tmp_path))
    monkeypatch.delenv(GUARD_ENV_VAR, raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.stdin", io.StringIO('{"session_id": "S1", "prompt": "real work"}'))

    assert hooks.dispatch("user-prompt-submit") == 0

    conn = index.open_forkB()
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT SessionId, Role, Kind, Body FROM Activity ORDER BY Id;"
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1, "unguarded UserPromptSubmit must append exactly one row"
    assert rows[0]["SessionId"] == "S1"
    assert rows[0]["Role"] == "user"
    assert rows[0]["Kind"] == "prompt"
    assert rows[0]["Body"] == "real work"
