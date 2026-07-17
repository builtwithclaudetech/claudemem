"""claudemem.recall.output — the machine-readable output + id contract (L2).

The serialization half of the read path (architecture §2.5, tech-design §8,
IN-21). It owns two things, both of which Claude depends on as a *stable
contract*:

1. **The unified id scheme** (§8.2): every record is addressed as ``a:<name>``
   (Fork A, the filename stem — durable, C-11) or ``b:<rowid>`` (Fork B archive,
   the SQLite rowid — ephemeral within the 45-day window). :func:`parse_id`
   splits on the **first** ``:`` and dispatches by fork; :func:`make_id` /
   :func:`make_id_b` build the same forms so ``search``/``get``/``menu`` all emit
   identical ids, and an id from ``search`` round-trips to ``get``/``used``.

2. **The serialization formats** (§8.1): id-first human text by default and
   ``--json`` JSONL for ``search``/``get`` (so Claude captures an id and replays
   it), plus the compact ``id␠title`` ``menu`` lines (no bodies, protecting the
   SC-5 ≤600-token budget).

**Firewall (architecture §2.5, §4; SC-6/SC-2).** Imports the standard library
(``json`` only) and — under ``TYPE_CHECKING`` for the ``Record`` type — nothing
else: never ``enrich``, never ``anthropic``, no model request. ``Record`` is a
type-only import so loading ``output`` does not drag in the ``store`` layer,
keeping the SC-2 cold path light. ``lint-imports`` asserts the firewall.

**Format stability (IN-21).** The id-first text and the JSONL object shape below
are a contract the next module (``search``/``get``/``menu``) and Claude itself
read against. The shapes are documented inline and pinned by tests; treat any
change here as a contract revision.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from claudemem.store.forka import Record

#: Field separator between an id and its title in human/menu lines (a single
#: space, U+0020). The "id-first" property (§8.1) is load-bearing: Claude parses
#: the leading token as the id and replays it to ``get``/``used``.
ID_TITLE_SEP = " "


class InvalidId(ValueError):
    """Raised by :func:`parse_id` for a syntactically invalid id (§8.2).

    A specific :class:`ValueError` subclass so the command layer can catch *just*
    this — an unparseable / unknown-fork id becomes a clean "not found", exit 0
    (SC-3), never an unhandled crash. Raising (rather than returning a sentinel)
    keeps the happy-path return type a plain ``(fork, key)`` tuple with no
    in-band error value for callers to forget to check.
    """


# --------------------------------------------------------------------------- #
# Unified id scheme (§8.2)                                                      #
# --------------------------------------------------------------------------- #


def make_id(record: Record) -> str:
    """Build the Fork A id ``a:<name>`` for a curated record (§8.2).

    The single source of the ``a:`` form so ``search``/``get``/``menu`` all emit
    an identical id that round-trips through :func:`parse_id`.
    """
    return f"a:{record.name}"


def make_id_b(rowid: int) -> str:
    """Build the Fork B archive id ``b:<rowid>`` for an ``Activity`` row (§8.2).

    The ``b:`` counterpart to :func:`make_id`. The rowid is ephemeral within the
    45-day window (§8.2); a later-pruned id resolves to a clean "not found"
    downstream (SC-3), which is why the durability split lives in the id itself.
    """
    return f"b:{rowid}"


def parse_id(raw: str) -> tuple[str, str]:
    """Split a unified id into ``(fork, key)`` on the **first** ``:`` (§8.2).

    ``a:<name>`` → ``("a", name)``; ``b:<rowid>`` → ``("b", rowid)``. The split is
    on the first colon only, so a name that itself contains a colon survives
    intact: ``parse_id("a:weird:name") == ("a", "weird:name")``. The fork is
    lower-cased before validation. ``b:`` keys are returned as the raw string (the
    caller coerces to ``int`` for the rowid lookup); this module does not assume a
    numeric Fork B key.

    Raises :class:`InvalidId` when there is no ``:`` separator, when the fork
    prefix is neither ``a`` nor ``b``, or when the key is empty — the command
    layer catches it and emits a clean "not found", exit 0 (SC-3). This never
    raises any *other* exception type, so the SC-3 contract is total.
    """
    fork, sep, key = raw.partition(":")
    if not sep:
        raise InvalidId(f"id has no fork prefix: {raw!r}")
    fork = fork.lower()
    if fork not in ("a", "b"):
        raise InvalidId(f"unknown fork prefix {fork!r} in id: {raw!r}")
    if not key:
        raise InvalidId(f"id has empty key: {raw!r}")
    return (fork, key)


# --------------------------------------------------------------------------- #
# Record → contract dict (the JSONL object shape, §8.1/IN-21)                    #
# --------------------------------------------------------------------------- #


def _search_obj(record: Record) -> dict[str, object]:
    """The stable per-record JSONL object for ``search``/``get`` (§8.1, IN-21).

    Keys (insertion order = emission order; the shape is the IN-21 contract):

    * ``id``         — the ``a:<name>`` unified id (round-trips to ``get``/``used``).
    * ``name``       — the filename stem (the Fork A natural key).
    * ``type``       — record type (``fact``/``decision``/...).
    * ``importance`` — integer 1–5.
    * ``pinned``     — bool (raw 0/1 storage normalized to JSON bool).
    * ``summary``    — the one-line summary, or ``null`` when absent.

    Body is intentionally excluded from the *search* object (search lists hits;
    ``get`` returns the body) so a result set stays compact.
    """
    return {
        "id": make_id(record),
        "name": record.name,
        "type": record.type,
        "importance": record.importance,
        "pinned": bool(record.pinned),
        "summary": record.summary,
    }


def _search_line(record: Record) -> str:
    """One id-first human text line for ``search`` (§8.1).

    Format (stable): ``a:<name>␠<summary-or-name>``. The id is the leading
    whitespace-delimited token so Claude (and humans) can lift it verbatim; the
    remainder is the summary, falling back to the bare name when no summary
    exists. No body — a search line is an index entry, not the record.
    """
    title = record.summary if record.summary else record.name
    return f"{make_id(record)}{ID_TITLE_SEP}{title}"


# --------------------------------------------------------------------------- #
# search serialization (§8.1)                                                   #
# --------------------------------------------------------------------------- #


def serialize_search(records: Sequence[Record], *, json: bool) -> str:
    """Serialize ``search`` hits — id-first text (default) or JSONL (§8.1, IN-21).

    Human (``json=False``): one id-first line per record (:func:`_search_line`),
    newline-joined. JSONL (``json=True``): one :func:`_search_obj` JSON object per
    line — line-delimited JSON, *not* a JSON array, so Claude can stream/parse a
    record at a time and replay each ``id``. An empty result set yields the empty
    string in both modes (the caller decides whether that triggers the Fork B
    ``[archive]`` fallback — that branch is ``search``'s job, not this module's).
    """
    if json:
        return "\n".join(_json_dumps(_search_obj(r)) for r in records)
    return "\n".join(_search_line(r) for r in records)


def serialize_get(record: Record, *, json: bool) -> str:
    """Serialize a single ``get`` result — full body (human) or JSON (§8.1).

    Human (``json=False``): an id-first header line (so the id is still
    lift-able) followed by a blank line and the full record body — this is the
    one serializer that emits the body. JSON (``json=True``): a single JSON
    object that extends the :func:`_search_obj` shape with a ``body`` key, so the
    ``get`` object is a superset of the ``search`` object (same ``id`` semantics,
    plus the body the caller asked for).
    """
    if json:
        obj = _search_obj(record)
        obj["body"] = record.body
        return _json_dumps(obj)
    header = f"{make_id(record)}{ID_TITLE_SEP}{record.summary or record.name}"
    return f"{header}\n\n{record.body}"


# --------------------------------------------------------------------------- #
# menu serialization (§8.1, SC-5)                                               #
# --------------------------------------------------------------------------- #


def archive_title(body: str) -> str:
    """A one-line ``[archive]`` title from an activity body (first non-empty line).

    Collapses the (possibly multi-line, possibly head/tail-elided) Fork B body to
    its first non-empty line so a ``search``/``get`` archive line stays single-line
    and compact; the full body is still reachable via ``get b:<rowid>``. Shared by
    both ``recall.search`` (the ``[archive]`` fallback lines) and ``recall.get``
    (the ``b:`` header), so the archive title format has one definition.
    """
    for line in body.splitlines():
        stripped = line.strip()
        if stripped:
            return f"[archive] {stripped}"
    return "[archive]"


def serialize_menu(entries: Iterable[tuple[str, str]]) -> str:
    """Serialize the session-start ``menu`` — compact ``id␠title`` lines (SC-5).

    ``entries`` is an iterable of ``(id, title)`` pairs already ranked + capped by
    the caller (``menu`` does pinned-first, salience-only ordering and applies the
    SC-5 cap). Emits exactly one ``id␠title`` line per entry — **no bodies** — so
    the menu stays within the ≤600-token / ≤10,000-char SC-5 budget. The caller
    passes ids built via :func:`make_id` / :func:`make_id_b`, so every menu id
    round-trips through :func:`parse_id`.
    """
    return "\n".join(f"{eid}{ID_TITLE_SEP}{title}" for eid, title in entries)


def _json_dumps(obj: dict[str, object]) -> str:
    """Compact, deterministic ``json.dumps`` for one JSONL object.

    ``ensure_ascii=False`` keeps Unicode literal (smaller, human-legible);
    ``separators`` strips insignificant whitespace so each object is a single
    tight line. Keys are emitted in insertion order (the §8.1 contract order),
    so ``sort_keys`` is intentionally left off.
    """
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "ID_TITLE_SEP",
    "InvalidId",
    "make_id",
    "make_id_b",
    "parse_id",
    "archive_title",
    "serialize_search",
    "serialize_get",
    "serialize_menu",
]
