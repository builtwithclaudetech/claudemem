# Changelog

All notable changes to ClaudeMem are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).
ClaudeMem uses [Semantic Versioning](https://semver.org/).

---

## [1.0.0] - 2026-06-01

### Added

- **Lexical FTS5 core** — salience-ranked search over authored memories using SQLite FTS5 with zero required external dependencies. Search works fully offline with no API key.
- **Optional write-time enrichment** — a single model call per `save` generates a `summary` and `aliases`, checks for duplicate or conflicting memories, and returns a dedup verdict. Enrichment is deferred automatically when no key or SDK is available; records persist immediately and are backfilled on the next `reindex`.
- **Files-as-truth authored memories** — each memory is a plain markdown file with YAML frontmatter stored under `~/.claude/memory/` (global) or `~/.claude/projects/<cwd-slug>/memory/` (per-project). The SQLite index is a rebuildable cache; `reindex` fully reconstructs it from the files at any time.
- **Auto-captured activity archive** — a rolling 45-day SQLite log of per-session prompts and transcript turns, written by Claude Code hooks with no model calls at write time. Search falls back to the archive when no authored memory clears the relevance threshold.
- **Claude Code hook integration** — `SessionStart` (salience-ranked titles menu injected as `additionalContext`), `UserPromptSubmit` and `Stop` (activity archive logging), and `SessionEnd` (end-of-session reflection with passive-hit reinforcement and promotion candidate proposals). All hooks always exit 0.
- **Stateless design** — every invocation boots, opens SQLite and files, does its work, and exits. No background process, no resident RAM between calls.
- **`claudemem` CLI** with subcommands: `search`, `get`, `save`, `import`, `promote`, `reindex`, `pin`, `unpin`, `forget`, `used`, `menu`, `log`, `hook`.
- **Pluggable enrichment backend** — auto-selects between the `claude` CLI (subscription-covered, no extra Python dependency) and the Anthropic SDK (`claudemem[llm]`), falling through to lexical-only when neither is available.
- **Spend guardrails** — configurable daily and monthly token caps; saves over the cap persist with enrichment deferred rather than blocking.
- **Supersede-not-delete** — conflicts and forgotten memories leave a soft-retirement trail. Hard deletion is always the user's choice.
- **Recursion guard** — `CLAUDEMEM_DISABLE_HOOKS` environment variable prevents hook re-entry when ClaudeMem spawns a model subprocess during `SessionEnd` reflection.
- **Read-path firewall** — the recall / search path has no import path to the enrichment layer or the Anthropic SDK; enforced statically by `import-linter` and at runtime by the test suite.
