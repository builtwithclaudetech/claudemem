"""Tests for claudemem.recall.output (T3.2) — id scheme + serialization (§8, IN-21).

Covers: ``parse_id`` round-trips ``a:foo``/``b:123``, splits on the FIRST ``:``
(so ``a:weird:name`` keeps ``weird:name`` as the key), and raises ``InvalidId``
(a ValueError subclass — never any other exception) for malformed ids so the
caller can map them to a clean "not found", exit 0 (SC-3); the id-first search
text and the JSONL object shape are pinned to exact formats (the IN-21 contract);
``get`` emits the body (text) / a body-bearing superset object (JSON); ``menu``
lines are ``id␠title`` with no body (SC-5); and an id emitted by
``serialize_search`` parses back via ``parse_id`` (the IN-21 round-trip).

``output`` is pure; ``Record`` values are constructed directly, no fixtures.
"""

from __future__ import annotations

import json

import pytest

from claudemem.recall import output
from claudemem.store.forka import Record


def _record(
    *,
    name: str = "auth-decision",
    importance: int = 4,
    pinned: bool = False,
    summary: str | None = "Use JWT for service auth",
    body: str = "Full body text.\nSecond line.",
    type_: str = "decision",
) -> Record:
    return Record(
        id=7,
        name=name,
        scope="project",
        project_id="p",
        type=type_,
        importance=importance,
        pinned=1 if pinned else 0,
        source="user",
        created=1_700_000_000,
        last_accessed=1_700_000_000,
        access_count=0,
        hit_count=0,
        summary=summary,
        aliases_json=None,
        aliases_flat=None,
        superseded_by=None,
        stale=0,
        enrich_pending=0,
        body=body,
    )


# --------------------------------------------------------------------------- #
# parse_id / make_id (§8.2)                                                     #
# --------------------------------------------------------------------------- #


def test_parse_id_fork_a() -> None:
    assert output.parse_id("a:foo") == ("a", "foo")


def test_parse_id_fork_b() -> None:
    # b key is returned as a raw string; the caller coerces to int.
    assert output.parse_id("b:123") == ("b", "123")


def test_parse_id_splits_on_first_colon() -> None:
    assert output.parse_id("a:weird:name") == ("a", "weird:name")


def test_parse_id_fork_lowercased() -> None:
    assert output.parse_id("A:foo") == ("a", "foo")


@pytest.mark.parametrize("bad", ["", "foo", "c:thing", "a:", ":x", "nofork"])
def test_parse_id_malformed_raises_invalidid(bad: str) -> None:
    with pytest.raises(output.InvalidId):
        output.parse_id(bad)


def test_invalidid_is_valueerror() -> None:
    # The caller may catch ValueError broadly; InvalidId must qualify.
    assert issubclass(output.InvalidId, ValueError)


def test_make_id_forms() -> None:
    assert output.make_id(_record(name="foo")) == "a:foo"
    assert output.make_id_b(123) == "b:123"


# --------------------------------------------------------------------------- #
# serialize_search — text + JSONL (§8.1, IN-21)                                 #
# --------------------------------------------------------------------------- #


def test_search_text_is_id_first_line_with_summary() -> None:
    rec = _record(name="auth-decision", summary="Use JWT for service auth")
    text = output.serialize_search([rec], json=False)
    assert text == "a:auth-decision Use JWT for service auth"


def test_search_text_falls_back_to_name_when_no_summary() -> None:
    rec = _record(name="bare", summary=None)
    assert output.serialize_search([rec], json=False) == "a:bare bare"


def test_search_text_multiple_records_one_line_each() -> None:
    recs = [_record(name="a1", summary="one"), _record(name="a2", summary="two")]
    text = output.serialize_search(recs, json=False)
    assert text.splitlines() == ["a:a1 one", "a:a2 two"]


def test_search_empty_is_empty_string() -> None:
    assert output.serialize_search([], json=False) == ""
    assert output.serialize_search([], json=True) == ""


def test_search_jsonl_object_shape() -> None:
    rec = _record(
        name="auth-decision",
        importance=4,
        pinned=True,
        summary="Use JWT",
        type_="decision",
    )
    line = output.serialize_search([rec], json=True)
    obj = json.loads(line)
    assert obj == {
        "id": "a:auth-decision",
        "name": "auth-decision",
        "type": "decision",
        "importance": 4,
        "pinned": True,
        "summary": "Use JWT",
    }
    # No body in the search object.
    assert "body" not in obj


def test_search_jsonl_is_line_delimited_not_array() -> None:
    recs = [_record(name="a1"), _record(name="a2")]
    out = output.serialize_search(recs, json=True)
    lines = out.splitlines()
    assert len(lines) == 2
    for line in lines:
        json.loads(line)  # each line is independently valid JSON
    assert not out.startswith("[")


def test_search_jsonl_null_summary() -> None:
    rec = _record(name="bare", summary=None)
    obj = json.loads(output.serialize_search([rec], json=True))
    assert obj["summary"] is None


# --------------------------------------------------------------------------- #
# serialize_get (§8.1)                                                          #
# --------------------------------------------------------------------------- #


def test_get_text_has_id_header_then_body() -> None:
    rec = _record(name="auth-decision", summary="Use JWT", body="Line A\nLine B")
    out = output.serialize_get(rec, json=False)
    assert out == "a:auth-decision Use JWT\n\nLine A\nLine B"


def test_get_json_is_search_object_plus_body() -> None:
    rec = _record(name="auth-decision", body="BODY")
    obj = json.loads(output.serialize_get(rec, json=True))
    assert obj["id"] == "a:auth-decision"
    assert obj["body"] == "BODY"
    # superset of the search object keys.
    search_obj = json.loads(output.serialize_search([rec], json=True))
    assert search_obj.keys() <= obj.keys()


# --------------------------------------------------------------------------- #
# serialize_menu (§8.1, SC-5)                                                   #
# --------------------------------------------------------------------------- #


def test_menu_lines_are_id_title_no_body() -> None:
    entries = [("a:foo", "Foo title"), ("b:42", "An archived turn")]
    out = output.serialize_menu(entries)
    assert out == "a:foo Foo title\nb:42 An archived turn"


def test_menu_empty() -> None:
    assert output.serialize_menu([]) == ""


# --------------------------------------------------------------------------- #
# Round-trip (IN-21): a search id parses back                                   #
# --------------------------------------------------------------------------- #


def test_search_id_roundtrips_through_parse_id() -> None:
    rec = _record(name="auth-decision")
    obj = json.loads(output.serialize_search([rec], json=True))
    fork, key = output.parse_id(obj["id"])
    assert (fork, key) == ("a", "auth-decision")


def test_search_text_leading_token_roundtrips() -> None:
    rec = _record(name="some-name", summary="a summary with spaces")
    line = output.serialize_search([rec], json=False)
    leading = line.split(output.ID_TITLE_SEP, 1)[0]
    assert output.parse_id(leading) == ("a", "some-name")
