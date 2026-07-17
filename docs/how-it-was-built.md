# How ClaudeMem was built

ClaudeMem was produced by a pipeline that front-loads specification and review before any code is
written: interviews, then a PRD, then a technical design, then an architecture document, each
iterated under adversarial review until it is settled — and only then a phased, verify-gated build.
This document walks through how that process produced the CLI in this repo, because ClaudeMem is a
clean example of the part of the method that does the real work: the specification, not the coding.
It started not from a blank page but from a post-mortem.

## Why it was worth building: a failure, written down

A prior memory system had the right *concept* — tiered memory, reindexable files, semantic-flavored
recall — and failed operationally for two concrete, mechanical reasons:

- **Per-session backend bloat.** It spun up a new backend process per session, each holding 1+ GB of
  RAM resident.
- **No viable local inference.** The host had no GPU; running a retrieval model locally was unusably
  slow and saturated the CPU.

Those two failures became the project's non-negotiable constraints *before* the design was written:
no daemon and nothing persistent in RAM; no local model and no GPU dependency; and — the real
insurance against repeating the failure — no hard dependency on any external API, so the system must
degrade to fully-offline lexical-only search when no API key is present. The keeper features of the
old system (tiering, files as truth, salience-ranked recall) were carried over; the architecture that
caused the failures was removed. Semantic strength comes from cheap **write-time** enrichment over
lexical search, not from a model on the read path.

## The spec came first, and it was large

ClaudeMem followed the full documentation pipeline before a line of implementation was scheduled: a
brainstorm brief, then a PRD, then a technical design, then an architecture document — the design and
brand stages skipped deliberately because it is a backend CLI with no user-facing surface. The
specification artifacts dwarf the eventual glue: the technical design and architecture documents
alone run to tens of thousands of words, and the build tasklist was broken into **54 tasks** before
any of them ran.

The point of that volume is not thoroughness for its own sake. By the time coding was scheduled, the
contentious decisions were already settled: the two-store model, the exact write pipeline, the
scoring formula, the degradation behavior, the hook wiring. The build was not asked to make
architecture decisions mid-stream.

## The design, in one screen

- **Two stores.** *Fork A* is curated markdown with YAML frontmatter — the source of truth, mirrored
  into an SQLite FTS5 index that can be rebuilt from the files at any time with `reindex`. *Fork B* is
  an SQLite-only rolling activity archive (a 45-day window, per-entry size cap) written by hooks with
  no model call.
- **Stateless per invocation.** Every CLI call boots, opens SQLite and the files, does its work, and
  exits. No resident process; nothing persists in RAM between calls. This is the direct answer to the
  prior system's 1+ GB-per-session footprint.
- **Exactly two model call sites, both on the write side.** One cheap enrichment call per `save`
  (generate a summary and aliases, and run a dedup / contradiction check in the same call), and one
  reflection call at session end. The read path — `search`, `get`, `menu`, every admin command —
  never calls a model, so recall works fully offline.
- **Salience ranking.** Results are scored `relevance × importance × recency-decay`; pinned memories
  never decay and always surface first; search falls back to the activity archive only when nothing
  authored clears a relevance floor.
- **Supersede, never hard-delete.** Conflicts and forgets leave a trail; the system soft-retires
  records and leaves hard deletion to the human.

## An architectural invariant made mechanical

The most important design rule — *the read path never touches a model* — is not left to code review
to catch. It is enforced in CI by a static import-linter check (`lint-imports`, run via
`uv run lint-imports`) that fails the build if a read-path module imports an enrichment module. An
architectural promise that would ordinarily erode over time is instead a mechanical gate: turn the
invariant you care about into something a machine refuses to let you break.

The headline success criterion is deliberately quantitative and points straight back at the original
failure: **per-session RAM overhead is ~0**, because there is no resident process. "Works offline
with no API key" and "the session-start menu stays under a small token ceiling" round out the
measurable bar the project was held to.

## What this shows about the process

- **A post-mortem became constraints, and constraints became architecture.** The two ways the prior
  system failed are the two things this one is structurally incapable of doing.
- **The spec was settled before the build.** Tens of thousands of words of design and a 54-task list
  existed before the first phase ran.
- **Invariants are enforced, not hoped for.** The read-path firewall is a CI gate, and the
  anti-regression metric (near-zero resident RAM) is quantitative.
- **The result is inspectable end to end.** This repo, its architecture document (`docs/ARCHITECTURE.md`),
  its test suite, and its import-linter configuration are all open.
