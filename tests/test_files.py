"""Tests for claudemem.files (T1.4).

Covers the file-truth contract: IN-1 frontmatter round-trip (full + minimal),
field-level SC-4 round-trip, the alias dual-form (§3.11) including a multi-word
alias, the ISO ⇄ epoch boundary (MF-1) at second granularity, and hand-deletion
robustness for ``iter_records`` / the ``writeback_*`` helpers.

All filesystem use is under ``tmp_path``; the scope dirs are pointed at tmp dirs
via a hand-built :class:`~claudemem.config.ScopeContext`, so no test touches the
real ``~/.claude``.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from claudemem.config import ScopeContext
from claudemem.files import (
    RecordFile,
    aliases_flat,
    aliases_from_json,
    aliases_json,
    epoch_to_iso,
    iso_to_epoch,
    iter_records,
    read_record,
    set_stale,
    set_superseded,
    write_record,
    writeback_counters,
)

# --------------------------------------------------------------------------- #
# Fixtures / builders.                                                          #
# --------------------------------------------------------------------------- #

# Every IN-1 field present, including a multi-word alias and a body.
_FULL_FRONTMATTER = """\
---
type: reference
scope: global
importance: 5
pinned: true
source: explicit
created: 2026-05-29T12:00:00Z
last_accessed: 2026-05-29T12:00:00Z
access_count: 0
hit_count: 0
summary: "Default fal.ai image endpoint preference."
aliases: [fal, fal.ai, image generation, txt2img, endpoint, model]
superseded_by: newer-record
stale: false
---

We prefer the `fal-ai/flux/dev` endpoint for general image generation.

Related: [[local-ai-stack]]
"""

# Minimal: only required-ish keys; optional fields absent (AS-7 defaults apply).
_MINIMAL_FRONTMATTER = """\
---
type: user
scope: project
source: session
created: 2026-01-02T03:04:05Z
last_accessed: 2026-01-02T03:04:05Z
---

A bare-bones record body.
"""


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------------- #
# Frontmatter round-trip (SC-4).                                                #
# --------------------------------------------------------------------------- #


def test_full_record_roundtrip_byte_stable(tmp_path: Path) -> None:
    """write_record(read_record(x)) reproduces every IN-1 field; second pass stable."""
    src = _write(tmp_path / "fal-preferred-endpoint.md", _FULL_FRONTMATTER)
    rec = read_record(src)

    # Field-level SC-4: every IN-1 field round-trips with the same value.
    assert rec.name == "fal-preferred-endpoint"
    assert rec.type == "reference"
    assert rec.scope == "global"
    assert rec.importance == 5
    assert rec.pinned is True
    assert rec.source == "explicit"
    assert rec.created == "2026-05-29T12:00:00Z"
    assert rec.last_accessed == "2026-05-29T12:00:00Z"
    assert rec.access_count == 0
    assert rec.hit_count == 0
    assert rec.summary == "Default fal.ai image endpoint preference."
    assert rec.aliases == [
        "fal",
        "fal.ai",
        "image generation",
        "txt2img",
        "endpoint",
        "model",
    ]
    assert rec.superseded_by == "newer-record"
    assert rec.stale is False
    assert "fal-ai/flux/dev" in rec.body

    # Round-trip: write, re-read, every field identical.
    out = tmp_path / "out.md"
    write_record(replace(rec, path=out))
    rec2 = read_record(out)
    for f in (
        "type",
        "scope",
        "importance",
        "pinned",
        "source",
        "created",
        "last_accessed",
        "access_count",
        "hit_count",
        "summary",
        "aliases",
        "superseded_by",
        "stale",
        "body",
    ):
        assert getattr(rec2, f) == getattr(rec, f), f"field {f} did not round-trip"

    # A second write is byte-for-byte identical (deterministic key order).
    first = out.read_text(encoding="utf-8")
    write_record(rec2)
    assert out.read_text(encoding="utf-8") == first


def test_minimal_record_applies_defaults(tmp_path: Path) -> None:
    """A minimal file parses with IN-1 defaults and round-trips its present fields."""
    src = _write(tmp_path / "bare.md", _MINIMAL_FRONTMATTER)
    rec = read_record(src)

    assert rec.name == "bare"
    assert rec.type == "user"
    assert rec.scope == "project"
    assert rec.source == "session"
    assert rec.importance == 3  # default
    assert rec.pinned is False  # default
    assert rec.access_count == 0
    assert rec.hit_count == 0
    assert rec.summary is None  # absent optional
    assert rec.superseded_by is None  # absent optional
    assert rec.aliases == []
    assert rec.stale is False
    assert rec.body.strip() == "A bare-bones record body."

    out = tmp_path / "bare-out.md"
    write_record(replace(rec, path=out))
    rec2 = read_record(out)
    # Optional Nones stay None (omitted on write, defaulted on read).
    assert rec2.summary is None
    assert rec2.superseded_by is None
    assert rec2.aliases == []
    assert rec2.type == "user"
    assert rec2.importance == 3


def test_summary_with_special_chars_survives_quoting(tmp_path: Path) -> None:
    """A summary containing a colon and quotes round-trips through write/read."""
    rec = RecordFile(
        path=tmp_path / "q.md",
        name="q",
        type="reference",
        scope="global",
        importance=3,
        pinned=False,
        source="explicit",
        created="2026-05-29T00:00:00Z",
        last_accessed="2026-05-29T00:00:00Z",
        access_count=0,
        hit_count=0,
        summary='Note: it said "hi" — keep it',
        aliases=["a"],
        superseded_by=None,
        stale=False,
        body="body text",
    )
    write_record(rec)
    rec2 = read_record(rec.path)
    assert rec2.summary == 'Note: it said "hi" — keep it'


def test_metadata_nested_block_is_hoisted(tmp_path: Path) -> None:
    """The brief.md alternate shape (metadata: nested block) parses defensively."""
    nested = """\
---
name: fal-preferred-endpoint
metadata:
  type: reference
  scope: global
  importance: 5
  pinned: true
  source: explicit
  created: 2026-05-29T00:00:00Z
  last_accessed: 2026-05-29T00:00:00Z
  aliases: [fal, image generation]
---

body
"""
    src = _write(tmp_path / "nested.md", nested)
    rec = read_record(src)
    assert rec.type == "reference"
    assert rec.importance == 5
    assert rec.pinned is True
    assert rec.aliases == ["fal", "image generation"]


def test_block_list_aliases_parse(tmp_path: Path) -> None:
    """Block-list aliases (- item lines) parse to the same list form."""
    block = """\
---
type: reference
scope: global
source: explicit
created: 2026-05-29T00:00:00Z
last_accessed: 2026-05-29T00:00:00Z
aliases:
  - fal
  - image generation
  - txt2img
---

body
"""
    src = _write(tmp_path / "blocklist.md", block)
    rec = read_record(src)
    assert rec.aliases == ["fal", "image generation", "txt2img"]


# --------------------------------------------------------------------------- #
# Alias dual-form (§3.11).                                                       #
# --------------------------------------------------------------------------- #


def test_alias_dual_form_multiword() -> None:
    """A multi-word alias survives list → json → list; flat tokenizes it apart."""
    aliases = ["fal", "image generation", "txt2img"]
    j = aliases_json(aliases)
    assert aliases_from_json(j) == aliases  # exact reconstruction
    flat = aliases_flat(aliases)
    assert flat == "fal image generation txt2img"
    # Flat is FTS-only: "image" and "generation" are separate tokens.
    assert "image" in flat.split()
    assert "generation" in flat.split()


def test_aliases_from_json_empty_and_none() -> None:
    assert aliases_from_json(None) == []
    assert aliases_from_json("") == []
    assert aliases_from_json("[]") == []


def _alias_record(tmp_path: Path, aliases: list[str]) -> RecordFile:
    return RecordFile(
        path=tmp_path / "alias.md",
        name="alias",
        type="reference",
        scope="global",
        importance=3,
        pinned=False,
        source="explicit",
        created="2026-05-29T00:00:00Z",
        last_accessed="2026-05-29T00:00:00Z",
        access_count=0,
        hit_count=0,
        summary=None,
        aliases=aliases,
        superseded_by=None,
        stale=False,
        body="body text",
    )


def test_alias_with_comma_roundtrips_exactly(tmp_path: Path) -> None:
    """An alias containing a comma round-trips as one alias, not split (SC-4)."""
    rec = _alias_record(tmp_path, ["a, b", "plain"])
    write_record(rec)
    rec2 = read_record(rec.path)
    assert rec2.aliases == ["a, b", "plain"]  # 2 aliases, not 3


def test_alias_with_double_quote_roundtrips_exactly(tmp_path: Path) -> None:
    """An alias containing a double-quote round-trips exactly (SC-4)."""
    rec = _alias_record(tmp_path, ['he said "hi"', "plain"])
    write_record(rec)
    rec2 = read_record(rec.path)
    assert rec2.aliases == ['he said "hi"', "plain"]


def test_legacy_bare_inline_list_aliases_parse(tmp_path: Path) -> None:
    """A hand-authored bare inline list still parses (AS-7 importer back-compat)."""
    legacy = """\
---
type: reference
scope: global
source: explicit
created: 2026-05-29T00:00:00Z
last_accessed: 2026-05-29T00:00:00Z
aliases: [fal, fal.ai]
---

body
"""
    src = _write(tmp_path / "legacy.md", legacy)
    rec = read_record(src)
    assert rec.aliases == ["fal", "fal.ai"]


def test_multiword_space_alias_roundtrips(tmp_path: Path) -> None:
    """A multi-word (space-separated) alias still round-trips through write/read."""
    rec = _alias_record(tmp_path, ["image generation", "txt2img"])
    write_record(rec)
    rec2 = read_record(rec.path)
    assert rec2.aliases == ["image generation", "txt2img"]


# --------------------------------------------------------------------------- #
# ISO ⇄ epoch (MF-1).                                                           #
# --------------------------------------------------------------------------- #


def test_iso_epoch_roundtrip_z_suffix() -> None:
    """Z-suffixed UTC ISO round-trips exactly at second granularity."""
    iso = "2026-05-29T12:00:00Z"
    epoch = iso_to_epoch(iso)
    assert isinstance(epoch, int)
    assert epoch_to_iso(epoch) == iso
    # 2026-05-29T12:00:00Z is a known epoch.
    assert iso_to_epoch(epoch_to_iso(epoch)) == epoch


def test_iso_epoch_offset_and_naive() -> None:
    """Explicit offset and timezone-naive (assumed UTC) both parse correctly."""
    # +00:00 offset equals the Z form.
    assert iso_to_epoch("2026-05-29T12:00:00+00:00") == iso_to_epoch(
        "2026-05-29T12:00:00Z"
    )
    # A +05:00 time is 5h earlier in UTC.
    assert (
        iso_to_epoch("2026-05-29T17:00:00+05:00")
        == iso_to_epoch("2026-05-29T12:00:00Z")
    )
    # Naive is treated as UTC.
    assert iso_to_epoch("2026-05-29T12:00:00") == iso_to_epoch("2026-05-29T12:00:00Z")


# --------------------------------------------------------------------------- #
# iter_records + hand-deletion (C-11/SC-4).                                      #
# --------------------------------------------------------------------------- #


def _scope(global_dir: Path, project_dir: Path | None) -> ScopeContext:
    return ScopeContext(
        kind="project" if project_dir else "global",
        project_id="testproj" if project_dir else None,
        global_dir=global_dir,
        project_dir=project_dir,
    )


def test_iter_records_merges_scopes_and_skips_non_md(tmp_path: Path) -> None:
    gdir = tmp_path / "global"
    pdir = tmp_path / "project"
    gdir.mkdir()
    pdir.mkdir()
    _write(gdir / "g1.md", _MINIMAL_FRONTMATTER)
    _write(pdir / "p1.md", _MINIMAL_FRONTMATTER)
    _write(gdir / "notes.txt", "not a record")
    _write(gdir / "MEMORY.md", "the CC memory index — skipped")

    names = sorted(r.name for r in iter_records(_scope(gdir, pdir)))
    assert names == ["g1", "p1"]


def test_iter_records_missing_dirs_yield_nothing(tmp_path: Path) -> None:
    """Absent global + project dirs yield no records and never raise (C-11/SC-4)."""
    gdir = tmp_path / "nope-global"
    pdir = tmp_path / "nope-project"
    assert list(iter_records(_scope(gdir, pdir))) == []
    # Global scope (project_dir None) with absent global dir also yields nothing.
    assert list(iter_records(_scope(gdir, None))) == []


# --------------------------------------------------------------------------- #
# Writeback helpers + hand-deletion robustness (IN-4/IN-6/IN-10/IN-14).          #
# --------------------------------------------------------------------------- #


def test_writeback_counters_updates_only_given_fields(tmp_path: Path) -> None:
    src = _write(tmp_path / "rec.md", _FULL_FRONTMATTER)
    rec = read_record(src)
    updated = writeback_counters(
        rec, hit_count=7, last_accessed="2026-06-01T00:00:00Z", access_count=3
    )
    assert updated is not None
    again = read_record(src)
    assert again.hit_count == 7
    assert again.last_accessed == "2026-06-01T00:00:00Z"
    assert again.access_count == 3
    # Other fields preserved.
    assert again.summary == rec.summary
    assert again.aliases == rec.aliases
    assert again.body == rec.body
    assert again.superseded_by == rec.superseded_by


def test_set_superseded_and_set_stale(tmp_path: Path) -> None:
    src = _write(tmp_path / "rec2.md", _MINIMAL_FRONTMATTER)
    rec = read_record(src)
    assert set_superseded(rec, "winner") is not None
    assert read_record(src).superseded_by == "winner"
    # Clear it again.
    assert set_superseded(read_record(src), None) is not None
    assert read_record(src).superseded_by is None
    # Stale flag flips and persists.
    assert set_stale(read_record(src), True) is not None
    assert read_record(src).stale is True


def test_writeback_against_deleted_file_no_crash(tmp_path: Path) -> None:
    """A file hand-deleted between read and writeback is skipped, not raised."""
    src = _write(tmp_path / "gone.md", _MINIMAL_FRONTMATTER)
    rec = read_record(src)
    src.unlink()  # the user hand-deletes it.
    assert writeback_counters(rec, hit_count=9) is None
    assert set_superseded(rec, "x") is None
    assert set_stale(rec, True) is None
    # File was not recreated by any of the skipped writebacks.
    assert not src.exists()
