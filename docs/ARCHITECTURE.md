# Architecture — ClaudeMem

## Design goals

ClaudeMem is a stateless, file-based memory system for Claude Code. Three hard constraints shaped every structural decision:

1. **No daemon or resident process.** Every invocation boots, opens SQLite and flat files, does its work, and exits. Nothing persists in RAM between calls. `pgrep -f claudemem` returns nothing between invocations.
2. **No local language model inference.** There is no GPU requirement and no on-device inference. Write-time enrichment uses a cheap hosted model via the Anthropic SDK or the `claude` CLI — both are optional.
3. **No hard dependency on any external API.** The lexical FTS5 core operates fully offline. With no API key present, ClaudeMem degrades silently to lexical-only mode. The system must never error solely because a key or SDK is absent.

---

## Stateless invocation model

```
boot
  → read config (~/.claude/claudemem/config.toml)
  → open SQLite connection(s) as needed
  → read/write markdown files as needed
  → do work
  → exit
```

There is no warm-up, no cache warm, and no background task. Cold start for a `search` is under 200 ms. All maintenance work (pruning the activity archive, running staleness checks, opportunistic vacuuming) is inline and bounded — never scheduled, never a separate process.

---

## Layered module graph

The package is organized into four layers. Every dependency edge points downward. No layer imports anything above it.

```
┌──────────────────────────────────────────────────────────────────┐
│  L4  ENTRY / DISPATCH                                            │
│      cli · hooks                                                 │
│      (process boot, argparse, recursion guard, command dispatch) │
├──────────────────────────────────────────────────────────────────┤
│  L3  WRITE-SIDE ORCHESTRATION                                    │
│      enrich                                                      │
│      (the only module that constructs model requests)            │
├──────────────────────────────────────────────────────────────────┤
│  L2  READ / RANK / SERIALIZE                                     │
│      recall                                                      │
│      (search · get · menu · rank · output — model-free)          │
├──────────────────────────────────────────────────────────────────┤
│  L1  STORE / FILES / CONFIG                                      │
│      store · files · index · config                              │
│      (SQLite + markdown + TOML; zero model calls)                │
└──────────────────────────────────────────────────────────────────┘
```

### Dependency summary

| Module | Depends on | Layer |
|---|---|---|
| `cli` | `recall`, `enrich`, `store`, `files`, `config` | L4 → L3/L2/L1 |
| `hooks` | `recall`, `enrich`, `store`, `config` | L4 → L3/L2/L1 |
| `enrich` | `store`, `recall`, `config` | L3 → L2/L1 |
| `recall` | `store`, `config` | L2 → L1 |
| `store` | `index`, `files`, `config` | L1 → L1 |
| `files` | `config` | L1 → L1 |
| `index` | `config` | L1 → L1 |
| `config` | stdlib only | leaf |

The layering is statically enforced by `import-linter` contracts in `pyproject.toml`. A forbidden contract ensures `recall` (and all its sub-modules) can never import `enrich` or the Anthropic SDK, even transitively. A layers contract encodes the L1→L4 ordering. Both contracts run as a build gate alongside the test suite.

---

## Authored memories vs. the activity archive

ClaudeMem maintains two distinct stores.

### Authored memories

Authored memories are individual markdown files with YAML frontmatter. They live under `~/.claude/memory/` (global scope) and `~/.claude/projects/<cwd-slug>/memory/` (per-project scope).

The **markdown files are the source of truth**. The SQLite FTS5 index is a derived cache. Running `claudemem reindex` fully rebuilds the index from the files — the system can always recover from a corrupt or deleted database by re-running `reindex`. You can edit files directly in any text editor, check them into version control, or delete them by hand.

A typical authored memory file:

```markdown
---
name: staging-db-host
type: reference
scope: project
importance: 4
pinned: false
source: explicit
created: 2026-06-01
last_accessed: 2026-06-01
access_count: 0
hit_count: 0
summary: "Staging database hostname and port for direct connection."
aliases: [staging, database, db, host, postgres, connection]
---

The staging database is at db.staging.example.com port 5432.
```

Everything in the frontmatter — including the model-generated `summary` and `aliases` — is visible, editable, and regenerated on `reindex`. Pinned memories never decay in salience ranking and are never pruned.

### Activity archive

The activity archive is a rolling SQLite log of per-session prompts and transcript turns. It is captured automatically by the `UserPromptSubmit` and `Stop` hooks with no model calls at write time. Entries are capped at 4,000 characters (head + tail truncation) and pruned to a 45-day window.

The activity archive has no markdown file backing. It is intentionally ephemeral — the raw material from which authored memories can be distilled, not a long-term store. `claudemem search` falls back to the archive when no authored memory clears the relevance threshold, labeling those results `[archive]`.

---

## The read-path firewall

The most important architectural property is the **absence** of a dependency from the read path to the enrichment layer.

**Invariant:** `recall` (and all its sub-modules: `search`, `get`, `menu`, `rank`, `output`) has no import path to `enrich`, the Anthropic SDK, or any subprocess spawn of `claude`.

This absence is load-bearing for three reasons:

1. **Model-free by construction.** Every read command (`search`, `get`, `menu`, `pin`, `unpin`, `forget`, `used`) works without any API key or SDK package installed. A missing key or a missing `anthropic` package changes nothing on the read path — there is no code path to reach.
2. **Cold-start performance.** Not importing the SDK keeps the read-path cold start well under 200 ms. The cost of the import never taxes a `search` call.
3. **Structural enforcement of the two-call-site rule.** Because the import itself is impossible from the read side, a future change cannot quietly add a third model-calling code path through a read command. The two sanctioned model call sites (`enrich_batch` at save time; `reflect` at session end) are the only places in the codebase that can reach a model.

The import-linter forbidden contract in `pyproject.toml` enforces this statically on every lint run. A runtime test suite (`test_firewall.py`) also asserts, in a fresh subprocess for each read command, that neither `claudemem.enrich` nor `anthropic` appears in `sys.modules` after the command completes.

---

## The two model call sites

ClaudeMem has exactly two places that construct a model request. Both are in the `enrich` package. Both are optional and degrade gracefully.

### 1. Write-time enrichment (`enrich_batch`)

Called by the `save`, `import`, and `promote` commands, and by the enrichment backfill phase of `reindex`.

A single model call per batch of records performs three jobs: generate a `summary` and `aliases` for the frontmatter, check for duplicate or conflicting existing memories, and return a verdict (`new` / `duplicate-of:<name>` / `conflicts-with:<name>`).

If the call is unavailable (no key, no SDK, over spend cap, transient failure), the record is still persisted immediately with enrichment marked as pending. The next `reindex` backfills pending records.

### 2. Session-end reflection (`reflect`)

Called by the `SessionEnd` hook.

One model call over the current session's bounded activity archive rows: identify any memories that were implicitly confirmed useful during the session (passive hits) and propose activity archive entries as promotion candidates for authored memories. Proposals are never auto-applied — they are presented for your approval.

---

## Enrichment backend selection

The enrichment backend is pluggable. The auto-selection order is:

1. **Claude CLI backend** — spawns `claude -p --output-format json` via subprocess. Requires the `claude` CLI to be authenticated (subscription-covered; no additional Python dependency).
2. **Anthropic SDK backend** — uses the `anthropic` Python package. Requires `claudemem[llm]` and `ANTHROPIC_API_KEY`.
3. **Lexical-only fallback** — no model call; enrichment deferred to `reindex`.

The backend can be forced via `[llm].backend = cli|sdk|none` in `config.toml`. A forced but unavailable backend warns once and falls through to lexical-only rather than erroring.

---

## Files-as-truth and `reindex`

`claudemem reindex` rebuilds the FTS5 index in two phases:

**Phase A (model-free):** A sidecar copy of the database is built from the markdown files using an atomic `os.replace` swap. Every authored memory is re-read from disk, upserted into the new index, and the staleness flag is recomputed for all records. The live database remains readable throughout the rebuild.

**Phase B (optional enrichment backfill):** Records that were saved while enrichment was unavailable (marked `EnrichPending`) are routed through `enrich_batch`. Records still over the spend cap carry forward to the next `reindex`. The count of remaining pending records is reported.

Because Phase A never imports `enrich`, running `reindex --no-backfill` is model-free and works fully offline.

---

## Graceful degradation

The system degrades along a spectrum rather than failing hard:

| Condition | Behavior |
|---|---|
| No `ANTHROPIC_API_KEY` | Lexical-only mode; saves persist without enrichment (`EnrichPending=1`); search works normally |
| `anthropic` SDK not installed | Same as no key; the SDK import is inside a `try/except ImportError` |
| Over spend cap | Enrichment deferred; save persists; `reindex` catches up later |
| Transient model failure | Bounded retry, then defer with reason logged |
| Missing or corrupt `forkA.db` | `reindex` rebuilds from markdown files |
| Hook error | Logged to `~/.claude/claudemem/claudemem.log`; hook exits 0; Claude Code session unaffected |

The one condition that is not degradable is a missing FTS5 extension in the SQLite build. FTS5 is the lexical core; without it there is nothing to fall back to. This does not occur in practice when ClaudeMem is installed via `uv` using its own Python interpreter.

---

## Hook topology

Four Claude Code hooks are registered, each dispatching to `claudemem hook <event>` with the JSON payload on stdin.

```
Claude Code session
  │
  ├─ SessionStart ──────────► claudemem hook session-start
  │                                  └─ recall.menu → additionalContext (≤600 tokens)
  │                                     [model-free]
  │
  ├─ UserPromptSubmit ──────► claudemem hook user-prompt-submit
  │                                  └─ append prompt to activity archive
  │                                     [model-free]
  │
  ├─ Stop ──────────────────► claudemem hook stop
  │                                  └─ append new transcript turns to activity archive
  │                                     [model-free]
  │
  └─ SessionEnd ────────────► claudemem hook session-end
                                     ├─ enrich.reflect (model call #2)
                                     └─ enrich_batch backfill of pending records
```

Every hook path checks the recursion guard (`CLAUDEMEM_DISABLE_HOOKS` environment variable) as the very first statement before reading stdin, importing any module, or opening a database. When the guard is set, the hook exits 0 immediately.

When the `SessionEnd` handler spawns a model process (via the CLI backend), that process inherits the `CLAUDEMEM_DISABLE_HOOKS=1` environment variable. This means all hooks fired by that inner session are immediate no-ops, bounding recursion to exactly one level deep.

---

## On-disk layout

```
~/.claude/claudemem/
  config.toml     # Configuration tunables ([llm] [ranking] [forkb] [spend] [promotion] [menu])
  forkA.db        # Authored-memory index: FTS5 + salience fields (rebuildable cache)
  forkB.db        # Activity archive: per-session turns (45-day rolling window)
  claudemem.log   # Hook error log (errors logged here, never raised)

~/.claude/memory/
  <name>.md       # Global-scope authored memories (markdown — source of truth)

~/.claude/projects/<cwd-slug>/memory/
  <name>.md       # Per-project authored memories (markdown — source of truth)
```

`forkA.db` is always recoverable by running `claudemem reindex`. `forkB.db` has no file backing; it is the ephemeral activity archive.

---

## Concurrency

Multiple Claude Code sessions can run concurrently against the same databases. Safety properties:

- Both databases use **WAL journal mode** so readers are never blocked by writers.
- Write transactions use `BEGIN IMMEDIATE` to take the writer lock up front, avoiding upgrade deadlocks.
- A `busy_timeout` of 5 seconds lets a concurrent writer wait rather than failing immediately.
- Activity archive rows are scoped by session ID, so two concurrent sessions never mix their archive rows or reinforce off each other's activity.
- The `reindex` sidecar swap (`os.replace`) is atomic at the filesystem level; concurrent readers see a consistent database until the moment of swap.
