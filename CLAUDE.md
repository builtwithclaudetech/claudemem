
## Secret Scanning

This repo is enrolled in the opt-in secrets-commit gate: a `git commit` carrying a real-secret-shaped staged change is blocked before it reaches history. Enrolled via `~/.claude/scripts/secrets_registry.py enroll`.

- Enforce level: `block`
- Enrolled: 2026-08-09
- Remote: `https://github.com/builtwithclaudetech/claudemem.git`
- Marker: `.secrets-gate` (committed — the fact travels with the repo)
- Registry: `~/.claude/secrets-gated-projects.json`
- Manual scan: `python3 ~/.claude/scripts/secrets_scan.py --staged --json`
