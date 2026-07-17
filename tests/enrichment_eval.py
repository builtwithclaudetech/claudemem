"""tests.enrichment_eval — the §10.4 backend-parameterized enrichment eval.

A **runnable** eval module (tech-design §10.4) that scores the LIVE enrichment
backend (CLI or SDK) over a curated fixture set of enrichment cases. It is
**backend-gated**: the pytest test that runs it against a real transport is
skipped offline (see ``tests/test_enrichment_eval.py``), so the unattended suite
never spawns ``claude`` / calls the API (consistent with ``SC-3`` — absence of a
backend is normal, not a failure).

What it certifies (tech-design §10.4, parameterized over the live backend):

* **CLI structured output** — ``≥90%`` first-attempt valid structured output and
  ``≥99%`` after the single repair re-spawn (§5.6). First-attempt vs repaired vs
  deferred is read off the backend's own :class:`SpendEntry.outcome`
  (``ok`` / ``repaired`` / ``deferred``, §3.4) plus per-element ``parse``
  :class:`DeferralEntry` s.
* **SDK content quality** — a rubric over summary / aliases / dedup-verdict
  correctness (forced tool-use guarantees structure, so the SDK gate is content
  quality, not parse rate, §5.2). Some checks are **auto-graded** (exact verdict
  match where the expectation is unambiguous, ∈-candidates, alias bounds, summary
  non-empty); fuzzier "is the summary relevant" is **reported for human review**.
* **Both backends** — the deterministic structural gates that hold for ANY
  backend: ``dedup_target_name ∈ candidate set`` (or null) and **all enrichment
  fields present regardless of verdict** (§5.2). These also run against the FAKE
  backend in the unattended structural test, proving the eval LOGIC offline.

Run it attended against the live backend (morning verify):

    uv run --python 3.11 python -m tests.enrichment_eval

It auto-detects the backend (``auto`` order: authenticated ``claude`` CLI → SDK
key → lexical-only) and prints a per-case report + the pass/fail verdict against
the gates. With no backend configured it prints a clear "no live backend" notice
and exits non-zero (so an attended run that forgot to configure a backend is
visible — distinct from the *pytest* skip, which is the unattended-suite path).

ASSUMPTION (noted per the unattended directive): the eval drives each backend
through its public :meth:`EnrichmentBackend.enrich_batch` (one
:class:`EnrichRequest` per case) and reads the returned :class:`BackendOutcome`.
This is the spec-consistent seam — §10.4 is "parameterized over the live
backend", and ``enrich_batch`` is the only backend entry that produces the
``results`` / ``deferred`` / ``spend`` the eval scores. The eval does NOT touch
``store`` / ``files`` (no persistence) — it scores the transport in isolation.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

from claudemem import config
from claudemem.enrich.backend import (
    BackendOutcome,
    Candidate,
    DedupVerdict,
    EnrichmentBackend,
    EnrichRequest,
    EnrichResult,
    select_backend,
)

# --------------------------------------------------------------------------- #
# Gates (tech-design §10.4)                                                     #
# --------------------------------------------------------------------------- #

#: CLI structured-output gates (§5.6 / §10.4).
CLI_FIRST_ATTEMPT_GATE = 0.90
CLI_POST_REPAIR_GATE = 0.99

#: SDK content-quality auto-graded pass gate. The auto-gradable checks (verdict
#: exact-match where unambiguous, ∈-candidates, alias bounds, non-empty summary)
#: must pass on ``≥90%`` of cases; the residual is surfaced for human review
#: rather than failing the gate (rubric eval, not a hard parse gate — §10.4).
SDK_CONTENT_GATE = 0.90


# --------------------------------------------------------------------------- #
# Fixture set — a curated, stdlib-only set of enrichment cases (§10.4)          #
#                                                                               #
# Each case is a record + its model-free dedup candidate set + the expected     #
# properties. ``expected_verdict`` is the deterministic expectation where the    #
# verdict is unambiguous (used for exact-match auto-grading); ``None`` means the  #
# verdict is judgement-dependent and only the structural gates apply.            #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class EvalCase:
    """One enrichment fixture case (record + candidates + expectations, §10.4).

    ``expected_verdict`` is the deterministic dedup expectation when it is
    unambiguous (a verbatim restatement of a candidate fact → ``duplicate``; a
    directly contradicting fact → ``conflict``; an unrelated fact with no
    matching candidate → ``new``). ``None`` marks a case whose verdict is a
    judgement call — only the structural gates and presence/bounds checks apply,
    and the produced verdict is reported for human review, not auto-graded.
    """

    record_id: str
    name: str
    body: str
    candidates: list[Candidate]
    expected_verdict: DedupVerdict | None
    note: str

    def to_request(self) -> EnrichRequest:
        return EnrichRequest(
            record_id=self.record_id,
            name=self.name,
            body=self.body,
            candidates=list(self.candidates),
        )


def _cand(name: str, summary: str, *aliases: str) -> Candidate:
    return Candidate(name=name, summary=summary, aliases=list(aliases), excerpt=summary)


def default_cases() -> list[EvalCase]:
    """The curated fixture set: 15 representative enrichment cases (§10.4).

    Spread across the three verdicts (``new`` / ``duplicate`` / ``conflict``)
    with varied content so summary/alias quality is judgeable, plus a few
    judgement-call cases (``expected_verdict=None``) that exercise only the
    structural gates. Stdlib-only construction — no external dependency.
    """
    redis_cand = _cand(
        "redis-port", "Redis listens on port 6380 on this host.", "redis", "cache port"
    )
    pg_cand = _cand(
        "postgres-port", "Postgres listens on port 5432 on this host.", "pg", "db port"
    )
    deploy_cand = _cand(
        "deploy-cmd",
        "Deploy with `make deploy` from the repo root.",
        "deploy",
        "release",
    )
    license_cand = _cand("license", "The project is MIT licensed.", "licence", "mit")

    return [
        # ---- new: unrelated facts, no matching candidate ----------------- #
        EvalCase(
            "ci-runner",
            "ci-runner",
            "CI runs on a self-hosted GitHub Actions runner named vps-1.",
            [redis_cand, pg_cand],
            "new",
            "Unrelated to either port candidate; should be new.",
        ),
        EvalCase(
            "editor-pref",
            "editor-pref",
            "the user edits with Neovim and uses two-space indentation in Python.",
            [deploy_cand, license_cand],
            "new",
            "No candidate touches editor prefs; new.",
        ),
        EvalCase(
            "backup-window",
            "backup-window",
            "Encrypted backups run nightly at 5am ET via the /backup script.",
            [redis_cand, deploy_cand],
            "new",
            "Backups are unrelated to ports or deploy command; new.",
        ),
        EvalCase(
            "tz-default",
            "tz-default",
            "Spend windows are computed in America/New_York (ET).",
            [pg_cand],
            "new",
            "Timezone fact unrelated to the postgres port; new.",
        ),
        EvalCase(
            "uv-toolchain",
            "uv-toolchain",
            "Dependencies and the virtualenv are managed with uv on Python 3.11.",
            [deploy_cand, license_cand],
            "new",
            "Toolchain fact, no matching candidate; new.",
        ),
        # ---- duplicate: restates a candidate fact ------------------------ #
        EvalCase(
            "redis-restated",
            "redis-restated",
            "Redis is bound to port 6380 on this machine.",
            [redis_cand, pg_cand],
            "duplicate",
            "Verbatim restatement of redis-port (6380); duplicate of redis-port.",
        ),
        EvalCase(
            "pg-restated",
            "pg-restated",
            "The Postgres server is reachable on port 5432.",
            [pg_cand, redis_cand],
            "duplicate",
            "Restates postgres-port (5432); duplicate of postgres-port.",
        ),
        EvalCase(
            "deploy-restated",
            "deploy-restated",
            "To ship a release, run `make deploy` from the repository root.",
            [deploy_cand, license_cand],
            "duplicate",
            "Restates deploy-cmd; duplicate of deploy-cmd.",
        ),
        EvalCase(
            "license-restated",
            "license-restated",
            "This codebase is released under the MIT license.",
            [license_cand, deploy_cand],
            "duplicate",
            "Restates license; duplicate of license.",
        ),
        # ---- conflict: directly contradicts a candidate fact ------------- #
        EvalCase(
            "redis-moved",
            "redis-moved",
            "Redis now listens on port 6390 (moved off 6380).",
            [redis_cand, pg_cand],
            "conflict",
            "Contradicts redis-port (6380 vs 6390); conflict with redis-port.",
        ),
        EvalCase(
            "pg-moved",
            "pg-moved",
            "Postgres was migrated to port 5433; it is no longer on 5432.",
            [pg_cand, redis_cand],
            "conflict",
            "Contradicts postgres-port (5432 vs 5433); conflict with postgres-port.",
        ),
        EvalCase(
            "license-changed",
            "license-changed",
            "The project relicensed to Apache-2.0; it is not MIT.",
            [license_cand, deploy_cand],
            "conflict",
            "Contradicts license (MIT vs Apache-2.0); conflict with license.",
        ),
        EvalCase(
            "deploy-changed",
            "deploy-changed",
            "Deploys now go through `./scripts/release.sh`, not `make deploy`.",
            [deploy_cand, license_cand],
            "conflict",
            "Contradicts deploy-cmd; conflict with deploy-cmd.",
        ),
        # ---- judgement calls: only structural gates apply ---------------- #
        EvalCase(
            "redis-related",
            "redis-related",
            "Redis on this host is configured with maxmemory 256mb and an LRU policy.",
            [redis_cand, pg_cand],
            None,
            "Related to redis-port but adds new config facts; verdict is a judgement call.",
        ),
        EvalCase(
            "no-candidates",
            "no-candidates",
            "The CLI entry point is `claudemem` installed on PATH.",
            [],
            None,
            "Empty candidate set; must produce a non-null summary/aliases and a new verdict.",
        ),
    ]


# --------------------------------------------------------------------------- #
# Per-case result + report                                                      #
# --------------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class CaseResult:
    """The scored outcome of one case (§10.4).

    ``outcome`` mirrors the backend's :class:`SpendEntry.outcome` where one was
    recorded (``ok`` / ``repaired`` / ``deferred``); ``"parse_defer"`` marks an
    element-level parse deferral that produced no spend row (the chunk parsed but
    this element failed validation, §5.6); ``"no_result"`` marks a case the
    backend neither enriched nor explained (degraded / lexical-only).
    """

    case: EvalCase
    result: EnrichResult | None
    outcome: str
    # Structural gates (apply to ANY backend, §5.2):
    target_in_candidates: bool
    all_fields_present: bool
    # SDK rubric auto-graded checks (None when not applicable, e.g. CLI / deferred):
    summary_nonempty: bool | None
    aliases_in_bounds: bool | None
    verdict_matches_expected: bool | None
    # Human-review notes (fuzzier signals — reported, not gated):
    human_review: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class EvalReport:
    """The full eval result over a fixture set against one backend (§10.4)."""

    backend_kind: str
    cases: list[CaseResult]

    # ---- CLI rates (§5.6 / §10.4) ---------------------------------------- #

    @property
    def total(self) -> int:
        return len(self.cases)

    @property
    def first_attempt_valid(self) -> int:
        """Cases valid on the FIRST attempt (no repair): outcome 'ok'."""
        return sum(1 for c in self.cases if c.outcome == "ok")

    @property
    def post_repair_valid(self) -> int:
        """Cases valid after the one repair re-spawn too: 'ok' OR 'repaired'."""
        return sum(1 for c in self.cases if c.outcome in ("ok", "repaired"))

    @property
    def first_attempt_rate(self) -> float:
        return self.first_attempt_valid / self.total if self.total else 0.0

    @property
    def post_repair_rate(self) -> float:
        return self.post_repair_valid / self.total if self.total else 0.0

    # ---- SDK content-quality auto-grade (§10.4) -------------------------- #

    @property
    def content_auto_graded(self) -> list[CaseResult]:
        """Cases where every applicable auto-graded check is defined (not None)."""
        return [
            c
            for c in self.cases
            if c.summary_nonempty is not None and c.aliases_in_bounds is not None
        ]

    @property
    def content_pass(self) -> int:
        """Auto-graded cases passing all auto checks (verdict only when expected)."""
        count = 0
        for c in self.content_auto_graded:
            ok = bool(c.summary_nonempty) and bool(c.aliases_in_bounds)
            ok = ok and c.target_in_candidates and c.all_fields_present
            if c.verdict_matches_expected is not None:
                ok = ok and c.verdict_matches_expected
            if ok:
                count += 1
        return count

    @property
    def content_rate(self) -> float:
        graded = len(self.content_auto_graded)
        return self.content_pass / graded if graded else 0.0

    # ---- Structural gates (ANY backend, §5.2) ---------------------------- #

    @property
    def structural_failures(self) -> list[CaseResult]:
        """Cases that produced a result but violated a deterministic gate (§5.2)."""
        return [
            c
            for c in self.cases
            if c.result is not None
            and not (c.target_in_candidates and c.all_fields_present)
        ]

    # ---- Pass/fail vs the §10.4 gates ------------------------------------ #

    def passed(self) -> bool:
        """True iff the live backend meets its gate(s) AND no structural gate failed."""
        if self.structural_failures:
            return False
        if self.backend_kind == "cli":
            return (
                self.first_attempt_rate >= CLI_FIRST_ATTEMPT_GATE
                and self.post_repair_rate >= CLI_POST_REPAIR_GATE
            )
        if self.backend_kind == "sdk":
            return self.content_rate >= SDK_CONTENT_GATE
        # An unknown/fake backend has no live-quality gate — only the structural
        # gates apply (already checked above). This is the offline-structural path.
        return True


# --------------------------------------------------------------------------- #
# The eval driver                                                               #
# --------------------------------------------------------------------------- #


def _backend_kind(backend: EnrichmentBackend) -> str:
    """Classify the backend as ``"cli"`` / ``"sdk"`` / other (its lowercase class).

    Reads an explicit ``kind`` attribute if the backend declares one (the fake
    backends in the structural test do), else falls back to the class name so the
    eval picks the right gate for the live ``ClaudeCliBackend`` /
    ``AnthropicSdkBackend`` without importing those classes here (keeps this
    module free of the SDK/CLI spawn machinery — same C-17 discipline).
    """
    declared = getattr(backend, "kind", None)
    if isinstance(declared, str):
        return declared
    cls = type(backend).__name__
    if cls == "ClaudeCliBackend":
        return "cli"
    if cls == "AnthropicSdkBackend":
        return "sdk"
    if cls == "LexicalOnlyBackend":
        return "lexical"
    return cls.lower()


def _score_case(
    case: EvalCase,
    result: EnrichResult | None,
    outcome: str,
) -> CaseResult:
    """Score one case against the structural gates + the SDK content rubric.

    Structural gates (ANY backend, §5.2) — only meaningful when a result exists:
    ``dedup_target_name ∈ candidate set`` (or null) and all enrichment fields
    present (non-empty summary, 1..8 aliases, a valid verdict, nullable fields
    null-or-string). The :class:`EnrichResult` dataclass guarantees the *slots*
    exist; this checks they carry spec-valid *values* (§5.2).

    SDK content rubric (auto-graded where deterministic; else human-review):
    summary non-empty, aliases 1..8, and an exact verdict match where the case's
    ``expected_verdict`` is unambiguous. Relevance of the summary text is fuzzier
    — it is appended to ``human_review`` rather than gated.
    """
    candidate_names = {c.name for c in case.candidates}
    human: list[str] = []

    if result is None:
        return CaseResult(
            case=case,
            result=None,
            outcome=outcome,
            target_in_candidates=False,
            all_fields_present=False,
            summary_nonempty=None,
            aliases_in_bounds=None,
            verdict_matches_expected=None,
            human_review=[f"no result ({outcome}); nothing to grade"],
        )

    # --- structural gate 1: dedup_target_name ∈ candidates (or null) (§5.2) ---
    target = result.dedup_target_name
    target_in_candidates = target is None or target in candidate_names

    # --- structural gate 2: all enrichment fields present + spec-valid (§5.2) ---
    summary_nonempty = bool(result.summary and result.summary.strip())
    aliases_in_bounds = 1 <= len(result.aliases) <= 8 and all(
        isinstance(a, str) and a.strip() for a in result.aliases
    )
    verdict_valid = result.dedup_verdict in ("new", "duplicate", "conflict")
    # Nullable-but-present fields: present regardless of verdict; a conflict must
    # carry an explanation, a non-new verdict must carry a target (§5.2).
    nullable_consistent = True
    if result.dedup_verdict == "new":
        # new must not assert a target (the backend coerces an out-of-set one).
        if result.dedup_target_name is not None:
            nullable_consistent = False
    if result.dedup_verdict == "conflict" and not result.conflict_explanation:
        human.append("conflict verdict with no conflict_explanation")
    all_fields_present = (
        summary_nonempty and aliases_in_bounds and verdict_valid and nullable_consistent
    )

    # --- SDK content rubric: exact verdict match where unambiguous ---------- #
    verdict_matches_expected: bool | None
    if case.expected_verdict is None:
        verdict_matches_expected = None
        human.append(
            f"verdict judgement call (got {result.dedup_verdict!r}): {case.note}"
        )
    else:
        verdict_matches_expected = result.dedup_verdict == case.expected_verdict
        if not verdict_matches_expected:
            human.append(
                f"verdict {result.dedup_verdict!r} != expected "
                f"{case.expected_verdict!r}: {case.note}"
            )

    # Summary relevance is fuzzy — reported, never gated.
    human.append(f"summary for review: {result.summary!r}")

    return CaseResult(
        case=case,
        result=result,
        outcome=outcome,
        target_in_candidates=target_in_candidates,
        all_fields_present=all_fields_present,
        summary_nonempty=summary_nonempty,
        aliases_in_bounds=aliases_in_bounds,
        verdict_matches_expected=verdict_matches_expected,
        human_review=human,
    )


def _outcome_for(
    case: EvalCase,
    outcome: BackendOutcome,
) -> tuple[EnrichResult | None, str]:
    """Reconcile one case against the backend's keyed :class:`BackendOutcome`.

    Returns ``(result_or_None, outcome_label)``. The label mirrors the backend's
    own :class:`SpendEntry.outcome` (``ok`` / ``repaired`` / ``deferred``, §3.4)
    so the CLI first-attempt / repaired / deferred rates are read from what the
    backend *actually did*, not re-derived. A record present in ``results`` with
    an ``ok``/``repaired`` spend is a first-attempt / post-repair success; a
    ``parse`` :class:`DeferralEntry` is a structured-output miss.
    """
    result = next((r for r in outcome.results if r.record_id == case.record_id), None)
    deferral = next((d for d in outcome.deferred if d.record_id == case.record_id), None)
    # A backend reports one spend per save call. For the per-case eval the eval
    # drives one request at a time, so spend is 0 or 1 entries and applies to
    # this case directly.
    spend = outcome.spend[0] if outcome.spend else None

    if result is not None:
        # 'ok' (first-attempt valid) or 'repaired' (valid after the repair spawn).
        label = spend.outcome if spend is not None else "ok"
        return result, label
    if deferral is not None:
        if deferral.reason == "parse":
            # A chunk-level parse failure carries a 'deferred' spend; an
            # element-level parse failure carries no spend (chunk parsed, element
            # didn't). Distinguish so the CLI rates classify both as misses.
            if spend is not None and spend.outcome == "deferred":
                return None, "deferred"
            return None, "parse_defer"
        # auth / cap / transient — not a structured-output quality signal.
        return None, deferral.reason
    return None, "no_result"


def run_enrichment_eval(
    backend: EnrichmentBackend, cases: list[EvalCase]
) -> EvalReport:
    """Run the §10.4 enrichment eval over ``cases`` against ``backend``.

    Drives one :class:`EnrichRequest` per case through the backend's public
    :meth:`EnrichmentBackend.enrich_batch` (so the backend owns its own
    chunking / retry / repair), reconciles each case against the returned
    :class:`BackendOutcome` (reading ``results`` / ``deferred`` / ``spend``), and
    scores it against the structural gates + the SDK content rubric. Returns an
    :class:`EvalReport` (rates, per-case results, pass/fail vs the gates).

    No persistence: the eval scores the transport in isolation (it never touches
    ``store`` / ``files``). Never spawns / calls a model itself — the backend
    does, exactly as production would.
    """
    kind = _backend_kind(backend)
    results: list[CaseResult] = []
    for case in cases:
        outcome = backend.enrich_batch([case.to_request()])
        result, label = _outcome_for(case, outcome)
        results.append(_score_case(case, result, label))
    return EvalReport(backend_kind=kind, cases=results)


# --------------------------------------------------------------------------- #
# Pretty-print                                                                  #
# --------------------------------------------------------------------------- #


def format_report(report: EvalReport) -> str:
    """Render the :class:`EvalReport` as a human-readable block (§10.4)."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append(f"ClaudeMem §10.4 enrichment eval — backend: {report.backend_kind}")
    lines.append("=" * 72)
    lines.append(f"cases: {report.total}")
    lines.append("")

    if report.backend_kind == "cli":
        lines.append("CLI structured-output rates (§5.6):")
        lines.append(
            f"  first-attempt valid : {report.first_attempt_valid}/{report.total}"
            f" = {report.first_attempt_rate:.1%}  (gate ≥ {CLI_FIRST_ATTEMPT_GATE:.0%})"
        )
        lines.append(
            f"  post-repair valid   : {report.post_repair_valid}/{report.total}"
            f" = {report.post_repair_rate:.1%}  (gate ≥ {CLI_POST_REPAIR_GATE:.0%})"
        )
    elif report.backend_kind == "sdk":
        lines.append("SDK content-quality rubric (§5.2 — auto-graded subset):")
        lines.append(
            f"  auto-graded pass    : {report.content_pass}/{len(report.content_auto_graded)}"
            f" = {report.content_rate:.1%}  (gate ≥ {SDK_CONTENT_GATE:.0%})"
        )
    else:
        lines.append(f"(no live-quality gate for backend kind {report.backend_kind!r};")
        lines.append(" only the deterministic structural gates apply.)")

    lines.append("")
    lines.append(f"structural gate failures (§5.2): {len(report.structural_failures)}")
    for cr in report.structural_failures:
        lines.append(
            f"  ! {cr.case.record_id}: target_in_candidates="
            f"{cr.target_in_candidates} all_fields_present={cr.all_fields_present}"
        )

    lines.append("")
    lines.append("per-case:")
    for cr in report.cases:
        verdict = cr.result.dedup_verdict if cr.result is not None else "-"
        lines.append(
            f"  {cr.case.record_id:<16} outcome={cr.outcome:<11} "
            f"verdict={verdict:<9} expected={cr.case.expected_verdict or '-'}"
        )

    lines.append("")
    lines.append("human-review notes (fuzzy signals — NOT gated):")
    for cr in report.cases:
        for note in cr.human_review:
            lines.append(f"  [{cr.case.record_id}] {note}")

    lines.append("")
    lines.append("=" * 72)
    lines.append(f"VERDICT: {'PASS' if report.passed() else 'FAIL'}")
    lines.append("=" * 72)
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Entry point — attended morning verify (NOT run by the unattended suite)       #
# --------------------------------------------------------------------------- #


def main() -> int:
    """Attended entry point: run the eval against the auto-detected live backend.

    ``uv run --python 3.11 python -m tests.enrichment_eval``

    Resolves the backend via ``select_backend`` (``auto`` order: authenticated
    ``claude`` CLI → SDK key → lexical-only). A lexical-only resolution means no
    live backend is configured — prints a notice and exits non-zero so an
    attended run that forgot to configure a transport is visible (this is
    distinct from the *pytest* skip, which is the unattended-suite path and stays
    green per ``SC-3``). Exit 0 on a passing live eval, 1 otherwise.
    """
    settings = config.load_config()
    backend = select_backend(settings)
    kind = _backend_kind(backend)
    if kind not in ("cli", "sdk"):
        print(
            "No live enrichment backend configured (resolved to "
            f"{kind!r}). Configure an authenticated `claude` CLI or set "
            "ANTHROPIC_API_KEY, then re-run. (This is the attended path; the "
            "pytest suite SKIPS this eval offline per SC-3.)",
            file=sys.stderr,
        )
        return 1
    report = run_enrichment_eval(backend, default_cases())
    print(format_report(report))
    return 0 if report.passed() else 1


def live_backend_available() -> bool:
    """True iff an enrichment-capable live backend (CLI or SDK) is detectable.

    Used by the backend-gated test's skip predicate. Honors the
    ``CLAUDEMEM_RUN_ENRICH_EVAL`` opt-in flag at the test layer (this helper only
    probes availability). Never raises — a probe failure is ``False`` (SC-3).
    """
    settings = config.load_config()
    try:
        return _backend_kind(select_backend(settings)) in ("cli", "sdk")
    except Exception:  # noqa: BLE001 — a probe must never error (SC-3).
        return False


#: The opt-in env flag that forces the backend-gated eval test to run even when
#: availability detection is inconclusive (tech-design §10.4 backend-gating).
RUN_ENRICH_EVAL_ENV = "CLAUDEMEM_RUN_ENRICH_EVAL"


def should_run_live_eval() -> bool:
    """Detect-or-flag predicate for the ATTENDED path: live backend OR the opt-in flag.

    Used only by attended callers (``main`` resolves the backend itself; this is the
    detect-or-flag convenience). The *pytest* test must NOT use this — auto-running on
    mere backend presence is what spawned ``claude`` during the unattended suite. The
    test gates on :func:`live_eval_opt_in` (explicit flag only) instead.
    """
    return live_backend_available() or bool(os.environ.get(RUN_ENRICH_EVAL_ENV))


def live_eval_opt_in() -> bool:
    """Skip predicate for the backend-gated *pytest* test: the explicit opt-in flag ONLY.

    NEVER gated on backend detection — a box with an authenticated ``claude`` CLI must
    still SKIP the live eval in the unattended suite (§10.4 is build-only / morning-verify,
    SC-3). The morning run opts in with ``CLAUDEMEM_RUN_ENRICH_EVAL=1``.
    """
    return bool(os.environ.get(RUN_ENRICH_EVAL_ENV))


if __name__ == "__main__":
    raise SystemExit(main())
