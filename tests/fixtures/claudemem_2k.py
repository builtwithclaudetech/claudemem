"""ClaudeMem-2k — the seeded, reproducible benchmark corpus (T6.1; tech-design §4.6).

The named baseline that discharges ``SC-2`` (tech-design §4.6 / §10.3): **2,000
active Fork A records / ~6 MB of body text**, generated from a single integer
seed with **stdlib ``random.Random(seed)`` ONLY** — no faker, no external dep
(PRD ``C-13``, so any adopter reproduces a byte-identical corpus from the seed).

This module is consumed by:

* **T6.2** — the 50 cold-run × 5-query spawn-inclusive latency harness
  (``§10.3``); it imports :data:`REPRESENTATIVE_QUERIES` and materializes the
  corpus into a tmp ``CLAUDEMEM_HOME``, then times ``claudemem search`` against
  the real ``forkA.db``.
* **T6.4** — the ``EXPLAIN QUERY PLAN`` validation (``§3.8``); it materializes
  the corpus and runs ``EXPLAIN QUERY PLAN`` over the §4.3 ``search`` query and
  the ``menu`` active-set scan against the 2,000-row index.

Determinism contract
---------------------
:func:`generate_corpus` derives **every** random choice — names, types,
importance, pinned, scope, summaries, aliases, body length, body vocabulary, and
the created/last-accessed epochs — from one ``random.Random(seed)``. Same seed →
identical corpus (same order, same bytes). No ``time.time``, no module-level
randomness, no ``os.urandom``.

Body sizing — how ~6 MB is hit at 2,000
---------------------------------------
The per-record body is a sequence of sentences drawn from a topic-keyed word
bank until the record reaches a per-record target byte budget. The targets are
drawn from a triangular distribution centred on
``target_body_bytes / count`` (≈ 3,000 B at the 2,000/6 MB point) with a spread
that keeps the *sum* of the targets ≈ ``target_body_bytes`` regardless of
``count`` — so the same generator produces ~6 MB at ``count=2000`` and a small,
fast corpus at ``count=50`` whose total still tracks ``target_body_bytes`` × the
count fraction. The realized total lands within a few percent of the target
(sentence granularity rounds each body up to the next whole sentence).

bm25 / IDF diversity (Phase-3 note)
-----------------------------------
bm25 IDF needs corpus diversity to behave like a real corpus. Each record is
assigned ONE primary topic from :data:`_TOPICS`; its name, summary, aliases, and
the bulk of its body are drawn from that topic's word bank, with a thin tail of
shared filler vocabulary. That gives:

* a **single-token** query (a topic keyword) that matches a meaningful subset,
* a **multi-token** query (two topic keywords) that matches via the OR-ed FTS
  MATCH, and
* a **no-match** query — :data:`NO_MATCH_TOKEN` is a coined nonsense token that
  appears in **no** body/name/summary/alias — which drives the Fork A → Fork B
  archive fallback path (``search.py`` §4.3/§5.2).

Materialization
---------------
:func:`materialize_corpus` writes the records as Fork A markdown files under a
self-contained ``CLAUDEMEM_HOME`` (``<home>/memory`` for global rows,
``<home>/projects/<slug>/memory`` for project rows) AND upserts them into
``<home>/forkA.db`` (so ``RecordFts`` is populated via the §3.3 triggers and a
real ``claudemem search`` returns hits). It optionally seeds a handful of Fork B
``Activity`` rows so the no-match query exercises the archive fallback and
returns ``[archive]`` hits rather than an empty result. The function sets
``CLAUDEMEM_HOME`` for the duration of the DB writes and restores the prior value
on exit, so it is tmp-safe and never touches the real ``~/.claude``.
"""

from __future__ import annotations

import os
import random
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from claudemem import config, files, index
from claudemem.store import forka, forkb

# --------------------------------------------------------------------------- #
# Pinned seed + scale constants (tech-design §4.6).                             #
# --------------------------------------------------------------------------- #

#: The pinned ClaudeMem-2k seed. The corpus T6.2 certifies SC-2 against is the
#: one produced by this exact seed; an adopter reproduces it byte-for-byte.
SEED = 20260530

#: Target active-record count (tech-design §4.6: 2,000 active records).
DEFAULT_COUNT = 2000

#: Target total body-text size in bytes (tech-design §4.6: ~6 MB body).
DEFAULT_TARGET_BODY_BYTES = 6_000_000

#: A few project scopes so the SC-12 scope-merge (global + active project) is
#: exercised by the harness. Index 0 (the empty string) means "global scope".
#: The remaining entries are explicit Claude Code project slugs.
_PROJECT_SLUGS: tuple[str, ...] = (
    "",  # global
    "-home-you-projects-my-app",
    "-home-you-projects-example-app",
    "-home-you-work-atlas",
)

#: The Fork A record types (tech-design type vocabulary).
_TYPES: tuple[str, ...] = ("user", "feedback", "project", "reference")

#: A fixed reference "now" epoch so created/last_accessed spreads are reproducible
#: without reading the wall clock. 2026-05-30T00:00:00Z.
_NOW_EPOCH = int(datetime(2026, 5, 30, tzinfo=timezone.utc).timestamp())

_SECONDS_PER_DAY = 86_400

# --------------------------------------------------------------------------- #
# Topic word banks — the IDF-diversity backbone (Phase-3 note).                 #
# --------------------------------------------------------------------------- #

#: Each topic is (topic_key, [distinctive vocabulary]). The topic_key is itself a
#: distinctive single token used in names/aliases; the vocabulary words are the
#: distinctive body terms. These are deliberately distinct per topic so bm25 IDF
#: separates them — a single-token topic query returns only that topic's slice.
_TOPICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("postgres", ("vacuum", "replication", "wraparound", "autovacuum",
                  "checkpoint", "toast", "btree", "pgbouncer")),
    ("kubernetes", ("kubelet", "pod", "ingress", "namespace", "sidecar",
                    "configmap", "scheduler", "etcd")),
    ("typescript", ("generics", "narrowing", "interface", "discriminated",
                    "structural", "tsconfig", "decorator", "inference")),
    ("rendering", ("rasterizer", "shader", "tessellation", "framebuffer",
                   "occlusion", "anisotropic", "viewport", "mipmap")),
    ("phonetics", ("fricative", "diphthong", "allophone", "prosody",
                   "sibilant", "rhotic", "glottal", "morpheme")),
    ("logistics", ("freight", "palletized", "manifest", "warehousing",
                   "dispatch", "customs", "intermodal", "drayage")),
    ("biochem", ("enzyme", "substrate", "catalysis", "phosphorylation",
                 "membrane", "ligand", "allosteric", "denature")),
    ("synth", ("oscillator", "lowpass", "envelope", "wavetable",
               "polyphony", "modulation", "resonance", "arpeggiator")),
)

#: Shared, low-distinctiveness filler — appears across all topics so bodies read
#: like prose and short queries still find a sentence frame. Intentionally
#: high-frequency (low IDF) so it does not dominate ranking.
_FILLER: tuple[str, ...] = (
    "the", "system", "value", "note", "when", "after", "before", "result",
    "should", "configured", "default", "behaviour", "observed", "during",
    "session", "record", "context", "summary", "detail", "example",
)

#: Multi-word alias fragments (some aliases are multi-word per the task).
_ALIAS_ADJECTIVES: tuple[str, ...] = (
    "deep", "quick", "legacy", "core", "shared", "nightly", "primary",
)

# --------------------------------------------------------------------------- #
# The 5 representative queries (tech-design §4.6 / §10.3).                       #
# --------------------------------------------------------------------------- #

#: A coined nonsense token guaranteed absent from every name/summary/alias/body
#: (it is not in any topic bank, the filler, or the slugs). Used to build the
#: no-match query that exercises the Fork A → Fork B archive fallback.
NO_MATCH_TOKEN = "zqxwflumph"

#: The 5 representative queries the latency harness times (§4.6: a mix of
#: single-token, multi-token, and a no-match query that exercises the Fork B
#: fallback). Order is stable so the harness can label results positionally.
#:
#: 1. single-token, matches a meaningful Fork A subset (the ``postgres`` topic);
#: 2. single-token, a different topic slice (``kubernetes``);
#: 3. multi-token (two same-topic keywords, OR-ed in the FTS MATCH);
#: 4. multi-token cross-topic (two distinct topic keywords);
#: 5. NO-MATCH — a nonsense token with no Fork A hit, routing to the Fork B
#:    archive fallback (the §4.3/§5.2 path).
REPRESENTATIVE_QUERIES: tuple[str, ...] = (
    "postgres",
    "kubernetes",
    "shader tessellation",
    "enzyme oscillator",
    NO_MATCH_TOKEN,
)

#: The index of the no-match query within :data:`REPRESENTATIVE_QUERIES`.
NO_MATCH_QUERY_INDEX = 4


# --------------------------------------------------------------------------- #
# Generated-record payload (decoupled from on-disk path).                       #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class GeneratedRecord:
    """One synthetic Fork A record, scope-aware but path-free.

    Mirrors the writable subset of :class:`claudemem.files.RecordFile` plus the
    derived ``project_id`` (``None`` for global scope). :func:`to_record_file`
    binds it to a concrete on-disk ``path``; :func:`scope_context` derives the
    :class:`~claudemem.config.ScopeContext` used for the ``forkA.db`` upsert.
    """

    name: str
    scope: str
    project_id: str | None
    type: str
    importance: int
    pinned: bool
    source: str
    created: str
    last_accessed: str
    summary: str
    aliases: list[str]
    body: str

    def to_record_file(self, path: Path) -> files.RecordFile:
        """Bind this record to ``path`` as a writable :class:`files.RecordFile`."""
        return files.RecordFile(
            path=path,
            name=self.name,
            type=self.type,
            scope=self.scope,
            importance=self.importance,
            pinned=self.pinned,
            source=self.source,
            created=self.created,
            last_accessed=self.last_accessed,
            access_count=0,
            hit_count=0,
            summary=self.summary,
            aliases=list(self.aliases),
            superseded_by=None,
            stale=False,
            body=self.body,
        )

    def scope_context(self, home: Path) -> config.ScopeContext:
        """The :class:`ScopeContext` for this record's scope, rooted at ``home``.

        Global rows live in ``<home>/memory``; project rows live in
        ``<home>/projects/<slug>/memory`` — a self-contained corpus under one
        root, so the fixture never touches the real ``~/.claude`` tree.
        """
        if self.scope == "global":
            return config.ScopeContext(
                kind="global",
                project_id=None,
                global_dir=home / "memory",
                project_dir=None,
            )
        assert self.project_id is not None
        return config.ScopeContext(
            kind="project",
            project_id=self.project_id,
            global_dir=home / "memory",
            project_dir=home / "projects" / self.project_id / "memory",
        )


# --------------------------------------------------------------------------- #
# Body sizing.                                                                  #
# --------------------------------------------------------------------------- #


def _record_body_targets(rng: random.Random, count: int, total: int) -> list[int]:
    """Per-record body byte targets whose mean is a FIXED bytes-per-record.

    The per-record mean is ``total / DEFAULT_COUNT`` (≈ 3,000 B at the 6 MB /
    2,000 spec point) — deliberately keyed to the *pinned* count, NOT to the
    requested ``count``. That is what makes the SAME generator produce ~6 MB at
    ``count=2000`` (2000 × 3,000) and a proportionally-small ~150 KB at
    ``count=50`` (50 × 3,000): the realized total tracks
    ``total × count / DEFAULT_COUNT``, so the unit test asserts the 2,000/6 MB
    target by arithmetic on a tiny, fast corpus.

    Each target is drawn from a triangular distribution centred on that mean with
    a ±50% spread (varied body lengths, §4.6), then the list is rescaled so its
    mean is exactly the per-record mean (cancelling the sampling drift) before
    sentence-granularity rounding.
    """
    per_record_mean = total / DEFAULT_COUNT
    raw = [
        rng.triangular(per_record_mean * 0.5, per_record_mean * 1.5, per_record_mean)
        for _ in range(count)
    ]
    target_sum = per_record_mean * count
    scale = target_sum / sum(raw)
    return [max(1, round(x * scale)) for x in raw]


def _build_body(rng: random.Random, vocab: tuple[str, ...], target_bytes: int) -> str:
    """Assemble a prose-ish body of ≈ ``target_bytes`` from ``vocab`` + filler.

    Emits whole sentences (8–18 words, ~70% drawn from the topic ``vocab`` and
    ~30% from the shared filler) until the UTF-8 byte length reaches
    ``target_bytes``; the body rounds up to the next whole sentence. All word and
    length choices come from ``rng`` so the body is reproducible.
    """
    pool_topic = vocab
    pool_filler = _FILLER
    parts: list[str] = []
    size = 0
    while size < target_bytes:
        n_words = rng.randint(8, 18)
        words: list[str] = []
        for _ in range(n_words):
            if rng.random() < 0.7:
                words.append(rng.choice(pool_topic))
            else:
                words.append(rng.choice(pool_filler))
        sentence = " ".join(words).capitalize() + "."
        parts.append(sentence)
        size += len(sentence.encode("utf-8")) + 1  # +1 for the joining space
    return " ".join(parts)


# --------------------------------------------------------------------------- #
# generate_corpus.                                                              #
# --------------------------------------------------------------------------- #


def generate_corpus(
    *,
    seed: int = SEED,
    count: int = DEFAULT_COUNT,
    target_body_bytes: int = DEFAULT_TARGET_BODY_BYTES,
) -> list[GeneratedRecord]:
    """Generate ``count`` active Fork A records, ~``target_body_bytes`` total body.

    Stdlib-only and fully seeded (``C-13``): every field derives from
    ``random.Random(seed)``, so the same ``(seed, count, target_body_bytes)``
    triple yields a byte-identical list in a stable order. Each record is
    assigned one primary topic (bm25-IDF diversity, Phase-3 note); its name,
    summary, aliases, and body bulk come from that topic. Scopes spread across
    global + a few project slugs (SC-12 scope merge); importance spans 1–5;
    ~18% are pinned; created/last_accessed spread over the trailing year so
    recency-decay (§4.4) is exercised. No record is superseded (all active).
    """
    rng = random.Random(seed)
    targets = _record_body_targets(rng, count, target_body_bytes)

    records: list[GeneratedRecord] = []
    for i in range(count):
        topic_key, vocab = _TOPICS[rng.randrange(len(_TOPICS))]
        slug = _PROJECT_SLUGS[rng.randrange(len(_PROJECT_SLUGS))]
        scope = "global" if slug == "" else "project"
        project_id = None if scope == "global" else slug

        # Distinctive, collision-free name: topic + ordinal + a topic word.
        name = f"{topic_key}-{i:04d}-{rng.choice(vocab)}"

        importance = rng.randint(1, 5)
        pinned = rng.random() < 0.18
        rec_type = _TYPES[rng.randrange(len(_TYPES))]
        source = "explicit" if rng.random() < 0.6 else "auto"

        # Recency spread: created up to 365 days back, last_accessed at or after
        # created and up to "now" — a genuine spread so salience/recency varies.
        created_age = rng.randint(0, 365)
        created_epoch = _NOW_EPOCH - created_age * _SECONDS_PER_DAY
        accessed_epoch = rng.randint(created_epoch, _NOW_EPOCH)

        # Summary: a topic-keyworded one-liner (FTS Summary column, weight 5).
        summary = (
            f"{topic_key.capitalize()} {rng.choice(vocab)} "
            f"{rng.choice(vocab)} {rng.choice(_FILLER)}"
        )

        # Aliases: 1–3, at least one multi-word (e.g. "deep postgres vacuum").
        n_aliases = rng.randint(1, 3)
        aliases: list[str] = []
        for j in range(n_aliases):
            if j == 0:
                aliases.append(
                    f"{rng.choice(_ALIAS_ADJECTIVES)} {topic_key} "
                    f"{rng.choice(vocab)}"
                )
            else:
                aliases.append(rng.choice(vocab))

        body = _build_body(rng, vocab, targets[i])

        records.append(
            GeneratedRecord(
                name=name,
                scope=scope,
                project_id=project_id,
                type=rec_type,
                importance=importance,
                pinned=pinned,
                source=source,
                created=files.epoch_to_iso(created_epoch),
                last_accessed=files.epoch_to_iso(accessed_epoch),
                summary=summary,
                aliases=aliases,
                body=body,
            )
        )
    return records


# --------------------------------------------------------------------------- #
# Materialization into a tmp CLAUDEMEM_HOME.                                    #
# --------------------------------------------------------------------------- #


@contextmanager
def _home_env(home: Path) -> Iterator[None]:
    """Temporarily point ``$CLAUDEMEM_HOME`` at ``home`` (restored on exit)."""
    prior = os.environ.get(config.CONFIG_HOME_ENV)
    os.environ[config.CONFIG_HOME_ENV] = str(home)
    try:
        yield
    finally:
        if prior is None:
            os.environ.pop(config.CONFIG_HOME_ENV, None)
        else:
            os.environ[config.CONFIG_HOME_ENV] = prior


def _record_path(home: Path, record: GeneratedRecord) -> Path:
    """The on-disk markdown path for ``record`` under the self-contained ``home``."""
    ctx = record.scope_context(home)
    directory = ctx.global_dir if record.scope == "global" else ctx.project_dir
    assert directory is not None
    return directory / f"{record.name}.md"


def materialize_corpus(
    records: list[GeneratedRecord],
    home: Path,
    *,
    write_files: bool = True,
    seed_forkb: bool = True,
) -> Path:
    """Materialize ``records`` into a self-contained ``CLAUDEMEM_HOME`` at ``home``.

    Writes each record as a Fork A markdown file under ``home`` (``write_files``;
    ``<home>/memory`` for global, ``<home>/projects/<slug>/memory`` for project)
    AND upserts every record into ``<home>/forkA.db`` so ``RecordFts`` is
    populated by the §3.3 sync triggers and a real ``claudemem search`` (which
    reads the index) returns hits. ``$CLAUDEMEM_HOME`` is set to ``home`` for the
    duration of the DB open and restored afterward, so the live ``~/.claude`` is
    never touched.

    ``seed_forkb=True`` also opens ``<home>/forkB.db`` and inserts a few
    ``Activity`` rows whose bodies mention the topic vocab, so the no-match query
    (:data:`NO_MATCH_TOKEN`) routes through the Fork A → Fork B archive fallback
    and the latency harness times a realistic fallback rather than an instant
    empty result. Returns the resolved ``home`` path.
    """
    home.mkdir(parents=True, exist_ok=True)

    if write_files:
        for record in records:
            path = _record_path(home, record)
            files.write_record(record.to_record_file(path))

    with _home_env(home):
        conn_a = index.open_forkA()
        try:
            for record in records:
                path = _record_path(home, record)
                forka.upsert_record(
                    conn_a,
                    record.to_record_file(path),
                    record.scope_context(home),
                )
        finally:
            conn_a.close()

        if seed_forkb:
            conn_b = index.open_forkB()
            try:
                _seed_forkb_activity(conn_b, records)
            finally:
                conn_b.close()

    return home


def _seed_forkb_activity(
    conn_b: object, records: list[GeneratedRecord], *, limit: int = 50
) -> None:
    """Insert a handful of recent ``Activity`` rows for the archive-fallback path.

    Reuses the first ``limit`` records' summaries as activity bodies so the
    no-match query (which finds nothing in Fork A) still has *something* recent
    to fall back to. Deterministic: ts values are derived from ``_NOW_EPOCH`` and
    the row index, not the wall clock.
    """
    import sqlite3

    assert isinstance(conn_b, sqlite3.Connection)
    for i, record in enumerate(records[:limit]):
        forkb.append_activity(
            conn_b,
            session_id="claudemem-2k-seed",
            ts=_NOW_EPOCH - i * 60,
            role="assistant",
            kind="message",
            body=f"{record.summary} {record.body[:200]}",
        )


__all__ = [
    "SEED",
    "DEFAULT_COUNT",
    "DEFAULT_TARGET_BODY_BYTES",
    "NO_MATCH_TOKEN",
    "NO_MATCH_QUERY_INDEX",
    "REPRESENTATIVE_QUERIES",
    "GeneratedRecord",
    "generate_corpus",
    "materialize_corpus",
]
