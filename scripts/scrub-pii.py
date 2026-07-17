#!/usr/bin/env python3
"""Generic PII scrub for an open-source release.

Walks a directory tree and replaces every match of a caller-supplied denylist
of (regex, replacement) pairs, in place. The denylist itself is NOT part of
this file — it never contains real personal data, so it is safe to commit to
a public repo as-is.

The denylist is loaded, in order, from:
  1. the path in the ``PII_DENYLIST`` environment variable, or
  2. a ``.pii-denylist`` file in the scrubbed root, or
  3. neither present -> exit with an explanatory error.

Denylist format (TSV, one rule per line):
    <python-regex><TAB><replacement>
Lines starting with ``#`` and blank lines are ignored. Every pattern is
compiled with ``re.IGNORECASE`` and applied, in file order, via ``re.subn``.

Usage::

    PII_DENYLIST=/path/to/denylist.tsv python3 scrub-pii.py [root] [--dry-run]

``root`` defaults to the current working directory. ``--dry-run`` reports
what would change without writing anything.

Requires only the Python 3.11+ standard library.
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

# Directory names pruned from the walk entirely. Any directory whose name
# contains "cache" (case-insensitive) is also pruned.
SKIP_DIR_NAMES = {
    ".git",
    ".venv",
    "__pycache__",
    "node_modules",
    "target",
    "bin",
    "obj",
    "dist",
}

# Binary / generated file suffixes that are never text-scrubbed.
BINARY_SUFFIXES = {
    ".db",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".woff",
    ".woff2",
    ".gz",
    ".br",
    ".so",
    ".pyc",
    ".wasm",
    ".ico",
    ".pdf",
}

LOCK_FILE_NAMES = {"uv.lock", "package-lock.json"}

THIS_FILE_NAME = Path(__file__).name
DENYLIST_FILE_NAME = ".pii-denylist"


def load_denylist() -> list[tuple[re.Pattern[str], str]]:
    """Locate and parse the denylist TSV. Exits with a clear message if absent."""
    env_path = os.environ.get("PII_DENYLIST")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            print(f"PII_DENYLIST is set to {path!s} but that file does not exist.", file=sys.stderr)
            sys.exit(1)
    else:
        local_path = Path.cwd() / DENYLIST_FILE_NAME
        if local_path.is_file():
            path = local_path
        else:
            print(
                "No PII denylist found. Set the PII_DENYLIST environment variable "
                f"to a denylist TSV path, or drop a {DENYLIST_FILE_NAME!r} file in "
                "the scrub root.",
                file=sys.stderr,
            )
            sys.exit(1)

    rules: list[tuple[re.Pattern[str], str]] = []
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.rstrip("\n")
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) != 2:
            print(f"{path}:{lineno}: expected '<regex>\\t<replacement>', got: {raw_line!r}", file=sys.stderr)
            sys.exit(1)
        pattern_text, replacement = parts
        rules.append((re.compile(pattern_text, re.IGNORECASE), replacement))
    return rules


def is_vendor_or_binary(path: Path, root: Path) -> bool:
    rel_parts = path.relative_to(root).parts
    if "vendor" in rel_parts:
        return True
    name = path.name
    if name.endswith(".min.js") or name.endswith(".min.css"):
        return True
    if name.endswith(".lock") or name in LOCK_FILE_NAMES:
        return True
    if path.suffix.lower() in BINARY_SUFFIXES:
        return True
    if name in (THIS_FILE_NAME, DENYLIST_FILE_NAME):
        return True
    return False


def iter_candidate_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d
            for d in dirnames
            if d not in SKIP_DIR_NAMES and "cache" not in d.lower()
        ]
        for filename in filenames:
            file_path = Path(dirpath) / filename
            if is_vendor_or_binary(file_path, root):
                continue
            yield file_path


def scrub_file(path: Path, rules: list[tuple[re.Pattern[str], str]], *, dry_run: bool) -> int:
    """Apply every rule to the file's contents. Returns the replacement count.

    Writes the result back to disk unless ``dry_run`` is set.
    """
    try:
        original = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return 0

    updated = original
    total_count = 0
    for pattern, replacement in rules:
        updated, count = pattern.subn(replacement, updated)
        total_count += count

    if total_count and updated != original:
        if not dry_run:
            path.write_text(updated, encoding="utf-8")
        return total_count
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", nargs="?", default=".", help="Directory to scrub (default: cwd)")
    parser.add_argument("--dry-run", action="store_true", help="Report changes without writing")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"Not a directory: {root}", file=sys.stderr)
        sys.exit(1)

    rules = load_denylist()

    total_files_changed = 0
    total_replacements = 0

    for file_path in iter_candidate_files(root):
        count = scrub_file(file_path, rules, dry_run=args.dry_run)
        if count:
            total_files_changed += 1
            total_replacements += count
            rel = file_path.relative_to(root)
            print(f"({count}): {rel}")

    verb = "Would change" if args.dry_run else "Changed"
    print(f"\n{verb} {total_files_changed} file(s), {total_replacements} replacement(s) total.")


if __name__ == "__main__":
    main()
