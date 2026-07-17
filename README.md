# ClaudeMem

Stateless, file-based memory for Claude Code — durable, searchable memory across sessions with no background daemon, no GPU, and no required external API.

![License: MIT](https://img.shields.io/badge/license-MIT-green)
![Python: 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![no daemon · no GPU · lexical core](https://img.shields.io/badge/no%20daemon%20%C2%B7%20no%20GPU%20%C2%B7%20lexical%20core-informational)

---

## What it is

ClaudeMem gives Claude Code durable, searchable memory across sessions. It works in two modes:

- **Authored memories** — markdown files you (or Claude) create deliberately. These are the source of truth. An SQLite FTS5 index is a rebuildable cache derived from them.
- **Auto-captured activity archive** — a rolling SQLite log of per-session I/O, written automatically by Claude Code hooks with no model calls at write time.

Both are searchable from the `claudemem` CLI. Claude Code can call `claudemem search` during a session to retrieve relevant context on demand.

### Design principles

- **Stateless per invocation.** Every call boots, opens SQLite and files, does its work, and exits. No background process, no resident RAM between calls.
- **Files are the source of truth.** Authored memories are plain markdown with YAML frontmatter. You can edit them directly, put them in version control, or delete them by hand. `claudemem reindex` fully rebuilds the SQLite index from the files at any time.
- **Graceful degradation.** The lexical FTS5 core requires zero external dependencies. With no API key set, ClaudeMem runs fully offline. Write-time enrichment via a cheap language model is an optional enhancement — not a hard requirement.
- **Supersede, never hard-delete.** Conflicts and forgotten memories leave a trail. You can hard-delete files by hand; ClaudeMem itself only soft-retires records.

---

## Features

- **Zero-dependency lexical core** — SQLite FTS5 with no required external packages
- **Stateless per invocation** — boots, runs, exits; nothing persists in RAM between calls
- **Optional write-time enrichment** — a single cheap model call per `save` generates a summary, aliases, and a dedup/conflict check; deferred automatically if no key is available
- **Files as truth with full `reindex` rebuild** — edit memories in any text editor; `reindex` reconstructs the index from the markdown files
- **Salience-ranked search** — relevance, importance, and recency decay combined; pinned memories always surface first
- **Activity archive fallback** — if no authored memory clears the relevance floor, search falls back to the auto-captured session archive
- **Claude Code hook integration** — `SessionStart`, `UserPromptSubmit`, `Stop`, and `SessionEnd` hooks wired to a single `claudemem hook <event>` entry point
- **Supersede-not-delete** — soft retirement with a trail; hard deletion is always your choice

---

## How it works

```
  claudemem save "..."
       │
       ▼
  [optional] write-time enrichment
  (summary + aliases + dedup check)
       │
       ▼
  markdown file written          ←── source of truth
  SQLite FTS5 index updated      ←── rebuildable cache
       │
       ▼
  claudemem search "..."
       │
       ▼
  FTS5 lexical search
  salience ranking (relevance × importance × recency)
  output to stdout
  (no model call on the read path — works fully offline)
```

The read path never touches a model. Search, get, menu, and all admin commands are model-free by construction.

---

## Prerequisites

- **Python 3.11 or later**
- **`uv`** (recommended) or **`pipx`** for installation
- **`ANTHROPIC_API_KEY`** in your environment — optional; only needed if you want write-time enrichment

---

## Install

### Clone and install with uv (recommended)

```sh
git clone https://github.com/builtwithclaudetech/claudemem.git
cd claudemem
uv tool install .
```

### Install with pipx

```sh
git clone https://github.com/builtwithclaudetech/claudemem.git
cd claudemem
pipx install .
```

After installation, confirm the entry point is on your PATH:

```sh
claudemem --help
```

### Install with enrichment support

To enable write-time Haiku enrichment, install with the `llm` extra:

```sh
uv tool install ".[llm]"
# or
pipx install ".[llm]"
```

This adds the `anthropic` Python SDK as a dependency. Set `ANTHROPIC_API_KEY` in your environment before use. Without the key, ClaudeMem degrades silently to lexical-only mode.

---

## Quick start

### Save a memory

```sh
claudemem save "We use pytest with uv run pytest; never invoke pytest directly."
```

```
save a:we-use-pytest-with-uv-run-pytest: enriched=1, deferred=0, conflicts=0
```

Save with explicit metadata:

```sh
claudemem save "The staging database is at db.staging.example.com port 5432." \
  --name staging-db-host \
  --type reference \
  --importance 4
```

### Search memories

```sh
claudemem search "database host"
```

```
a:staging-db-host  [0.91]  The staging database is at db.staging.example.com port 5432.
```

Search returns plain text by default. Use `--json` for JSONL output suitable for Claude:

```sh
claudemem search "pytest" --json
```

### Retrieve a full memory

```sh
claudemem get a:staging-db-host
```

### Mark a memory as used (reinforces its ranking)

```sh
claudemem used a:staging-db-host
```

```
reinforced: a:staging-db-host (hit_count=1)
```

### Pin a memory (stays at the top; never decays)

```sh
claudemem pin a:staging-db-host
```

### Soft-delete a memory

```sh
claudemem forget a:staging-db-host
```

The markdown file is marked superseded but not deleted. The record is removed from active search results.

### Rebuild the index from files

```sh
claudemem reindex
```

```
reindex: 42 records; backfilled 3, 0 still enrich-pending
```

Run `reindex` after editing markdown files directly, after adding `ANTHROPIC_API_KEY` to catch up on deferred enrichment, or to run the staleness sweep.

### Bulk import existing markdown memories

```sh
claudemem import /home/you/projects/my-app/memory/
```

Records already indexed are re-upserted safely. Run `reindex` afterward to fill in enrichment if the import ran without a key.

---

## Claude Code integration

### How the hooks work

ClaudeMem registers four hooks in Claude Code settings. Each hook pipes its JSON payload to `claudemem hook <event>` on stdin:

| Event | What happens |
|---|---|
| `SessionStart` | A salience-ranked titles menu is injected as `additionalContext` (capped at 30 entries / 600 tokens). On a resumed session, injection is skipped. |
| `UserPromptSubmit` | The submitted prompt is appended to the activity archive (model-free, no API call). |
| `Stop` | New transcript turns since the last watermark are appended to the activity archive (model-free). |
| `SessionEnd` | An end-of-session reflection runs (one model call): passive hits are reinforced and promotion candidates from the archive are proposed. Pending enrichment is then backfilled. |

All hook paths always exit 0. A failure logs to `~/.claude/claudemem/claudemem.log` and is never surfaced to Claude Code.

### Recursion guard

When ClaudeMem's `SessionEnd` handler spawns a model to run reflection, that spawned session also fires hooks. The `CLAUDEMEM_DISABLE_HOOKS` environment variable is set in the spawned environment so all hooks in that inner session are no-ops. This bounds recursion to depth one.

### Directory layout

```
~/.claude/
  memory/                        # Global-scope authored memories (markdown)
    <name>.md
  projects/
    <cwd-slug>/
      memory/                    # Per-project authored memories (markdown)
        <name>.md
  claudemem/
    config.toml                  # All configuration tunables
    forkA.db                     # Curated index (rebuildable cache)
    forkB.db                     # Activity archive (45-day rolling window)
    claudemem.log                # Hook error log
```

The `cwd-slug` is derived from the absolute path of your project directory, matching Claude Code's own project slug convention (e.g., `/home/you/projects/my-app` becomes `-home-you-projects-my-app`).

Global memories are available in every session. Project memories are merged with global memories at query time and are visible only when your working directory resolves to that project.

### Hook registration

Register the four hooks in your Claude Code settings (`.claude/settings.json` or `.claude/settings.local.json`):

```json
{
  "hooks": {
    "SessionStart": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "claudemem hook session-start"
          }
        ]
      }
    ],
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "claudemem hook user-prompt-submit"
          }
        ]
      }
    ],
    "Stop": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "claudemem hook stop"
          }
        ]
      }
    ],
    "SessionEnd": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "claudemem hook session-end"
          }
        ]
      }
    ]
  }
}
```

---

## Configuration

Configuration lives at `~/.claude/claudemem/config.toml`. All keys are optional; defaults are applied when the file or a key is absent.

Override the home directory with the `CLAUDEMEM_HOME` environment variable (useful for testing or multi-profile setups):

```sh
CLAUDEMEM_HOME=/tmp/test-mem claudemem search "example"
```

### Selected defaults

```toml
[ranking]
recency_half_life_days = 90   # how fast recency score decays
relevance_floor = 0.30        # below this, falls back to activity archive
salience_floor = 0.05         # below this, excluded from menu

[forkb]
window_days = 45              # activity archive rolling window
entry_char_cap = 4000         # per-entry character cap (head+tail truncation)

[staleness]
horizon_days = 180            # days without access before a trust-decay flag is set

[menu]
max_entries = 30              # maximum entries in the SessionStart menu
token_ceiling = 600           # approximate token ceiling for the menu

[spend]
daily_token_cap = 1000000     # SDK token cap per ET day (warn-not-block)
monthly_token_cap = 15000000  # SDK token cap per ET month (warn-not-block)

[promotion]
hit_threshold = 3             # hits in window_days before a promotion is proposed
```

Spend caps are advisory — a save over the cap still persists with enrichment deferred to the next `reindex`.

---

## Scope

Most commands accept `--scope` and `--project` flags to override automatic scope detection:

```sh
claudemem search "api key" --scope global
claudemem save "..." --scope project --project -home-you-projects-my-app
```

By default, scope is derived from your current working directory.

---

## Development

```sh
# Clone and install dev + llm extras
uv sync --extra dev --extra llm

# Run tests
uv run pytest

# Lint
uv run ruff check .

# Type check
uv run mypy claudemem

# Static import-linter check (enforces the read-path firewall)
uv run lint-imports
```

---

## Project structure

```
claudemem/
  config.py       # Config parsing, scope resolution, locked defaults
  files.py        # Markdown I/O (source of truth for authored memories)
  index.py        # SQLite schema, connection helpers, FTS5 DDL
  store/
    forka.py      # Authored-memory CRUD against the index
    forkb.py      # Activity archive append, cursor watermark, pruning
    reindex.py    # Atomic index rebuild from markdown files
    spend.py      # Spend ledger and token-cap tally
  recall/
    search.py     # Salience-ranked FTS5 search with archive fallback
    get.py        # Full-body fetch by record id
    menu.py       # Session-start titles menu (model-free)
    rank.py       # Salience scoring (relevance × importance × recency decay)
    output.py     # Output serialization and unified id scheme (a: / b:)
  enrich/
    backend.py    # EnrichmentBackend protocol and auto-selection
    backend_cli.py  # Claude CLI transport (subscription-covered, no extra dep)
    backend_sdk.py  # Anthropic SDK transport (requires claudemem[llm])
    routine.py    # enrich_batch (save-time) and reflect (session-end) routines
  cli.py          # Console entry point, argparse, lazy command dispatch
  hooks.py        # Claude Code hook dispatch (SessionStart / Stop / SessionEnd)

tests/            # pytest suite
```

---

## Links

- [Architecture](docs/ARCHITECTURE.md) — layered module graph, read-path firewall, data-flow paths
- [Security](SECURITY.md) — vulnerability disclosure and security model
- [License](LICENSE) — MIT
