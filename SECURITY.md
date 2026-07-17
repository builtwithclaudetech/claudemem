# Security Policy

## Supported versions

Security fixes are applied to the `main` branch. Only the latest release is actively maintained.

## Reporting a vulnerability

If you discover a security vulnerability in ClaudeMem, please report it through **GitHub Security Advisories** rather than a public issue:

1. Go to the [Security tab](https://github.com/builtwithclaudetech/claudemem/security) of this repository.
2. Click **Report a vulnerability**.
3. Fill in the details of the issue.

**Do not disclose the vulnerability publicly until it has been addressed.**

If you are not sure whether an issue is a security vulnerability, report it privately anyway — it is easier to downgrade than to un-disclose.

## Response timeline

- **Acknowledgment**: Within 7 days of report
- **Initial assessment**: Within 14 days
- **Fix**: Depends on severity; critical issues are prioritized for an immediate patch

## Security model

ClaudeMem is a local CLI tool. Its security posture differs significantly from a network-facing application.

### What ClaudeMem does

- Stores all data locally under the user's home directory (`~/.claude/`).
- Has no server component, no network listener, and no remote storage.
- Never writes secrets to disk. The only credential it reads is `ANTHROPIC_API_KEY` from the environment, used transiently for optional write-time enrichment and never persisted.
- With no API key set, ClaudeMem runs entirely offline in lexical-only mode with no outbound network activity.

### What memory content contains

Memory content is whatever the user or Claude saves — notes, facts, preferences, project context. Users should treat the memory store with the same care they would treat any local notes file:

- **Do not save secrets, credentials, tokens, or private keys as memories.** The markdown files and SQLite databases are plain-text on disk and are not encrypted at rest by ClaudeMem.
- Memories are readable by any process running as the same user.

### Hook subprocess safety

The `SessionEnd` hook may spawn a `claude` subprocess to run the end-of-session reflection. That subprocess inherits a sanitized environment with the `CLAUDEMEM_DISABLE_HOOKS` variable set, preventing recursive hook invocations. Hooks always exit 0 regardless of internal failures, and errors are logged locally rather than propagated to Claude Code.

## Dependencies

**Core (zero required external dependencies):** The lexical FTS5 core depends only on the Python standard library and the SQLite FTS5 extension bundled with the Python interpreter.

**Optional enrichment (`claudemem[llm]`):** Installs the `anthropic` Python SDK. When installed, the SDK is the only outbound network dependency. It is used transiently for enrichment calls and never receives memory content in bulk — only individual records at write time.

There are no vendored JavaScript dependencies, no web server, and no daemon process.
