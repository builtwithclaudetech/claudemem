"""Tests for tests.enrichment_eval — the §10.4 backend-parameterized eval.

Two halves, per the §10.4 backend-gating contract:

* A **fast structural test** (NOT skipped, NO live backend) that runs
  :func:`run_enrichment_eval` against a deterministic **fake backend** (canned
  :class:`BackendOutcome` s, reusing the fake-backend pattern from
  ``test_enrich_routine.py``). This proves the eval LOGIC — CLI rate
  computation, SDK rubric scoring, the ∈-candidates + all-fields structural
  gates — works WITHOUT any real model call. This is the part the unattended
  suite runs.

* The **backend-gated live test** that drives the eval against a real transport.
  It is ``@pytest.mark.skipif`` 'd on backend availability (and the
  ``CLAUDEMEM_RUN_ENRICH_EVAL`` opt-in flag), so the unattended suite SKIPS it
  and never spawns ``claude`` / calls the API (consistent with ``SC-3``).
"""

from __future__ import annotations

import pytest

from claudemem.enrich.backend import (
    BackendOutcome,
    DedupVerdict,
    DeferralEntry,
    EnrichRequest,
    EnrichResult,
    ReflectOutcome,
    ReflectRequest,
    SpendEntry,
)
from tests import enrichment_eval as ev


# --------------------------------------------------------------------------- #
# Scripted fake backend (the fake-backend pattern from test_enrich_routine.py)  #
# --------------------------------------------------------------------------- #


class ScriptedBackend:
    """A fake backend that returns a canned outcome PER record_id (offline).

    ``kind`` lets the eval pick the right gate (cli / sdk) without any real
    transport. Each ``enrich_batch`` call carries exactly one request (the eval
    drives one case at a time); the scripted outcome for that record_id is
    returned. The only model touch in the structural test — no ``anthropic``
    import, no ``claude`` spawn.
    """

    def __init__(self, *, kind: str, outcomes: dict[str, BackendOutcome]) -> None:
        self.kind = kind
        self._outcomes = outcomes
        self.seen: list[str] = []

    @staticmethod
    def detect() -> bool:
        return True

    def enrich_batch(self, reqs: list[EnrichRequest]) -> BackendOutcome:
        assert len(reqs) == 1, "the eval drives one case per call"
        rid = reqs[0].record_id
        self.seen.append(rid)
        return self._outcomes[rid]

    def reflect(self, req: ReflectRequest) -> ReflectOutcome:  # pragma: no cover
        raise AssertionError("the enrichment eval never calls reflect")


def _result(
    rid: str,
    *,
    verdict: DedupVerdict = "new",
    summary: str = "a tight one-sentence summary",
    aliases: list[str] | None = None,
    target: str | None = None,
    explanation: str | None = None,
) -> EnrichResult:
    return EnrichResult(
        record_id=rid,
        summary=summary,
        aliases=aliases if aliases is not None else ["alias-one", "alias-two"],
        dedup_verdict=verdict,
        dedup_target_name=target,
        conflict_explanation=explanation,
    )


def _spend(outcome: str) -> SpendEntry:
    return SpendEntry(call_site="save", model="haiku", backend="cli", outcome=outcome)  # type: ignore[arg-type]


def _ok_outcome(result: EnrichResult, *, spend_outcome: str = "ok") -> BackendOutcome:
    return BackendOutcome(results=[result], deferred=[], spend=[_spend(spend_outcome)])


# --------------------------------------------------------------------------- #
# Fast structural test — CLI rate computation (offline, NOT skipped)            #
# --------------------------------------------------------------------------- #


def test_cli_rate_computation_first_attempt_and_repair() -> None:
    """CLI rates read off SpendEntry.outcome: ok / repaired / deferred (§5.6)."""
    cases = ev.default_cases()
    # Script: most ok, one repaired, one chunk-parse deferral. That yields a
    # first-attempt rate below 1.0 but a post-repair rate that counts the repair.
    outcomes: dict[str, BackendOutcome] = {}
    for i, case in enumerate(cases):
        verdict = case.expected_verdict or "new"
        target = case.candidates[0].name if verdict in ("duplicate", "conflict") else None
        explanation = "they disagree" if verdict == "conflict" else None
        result = _result(
            case.record_id, verdict=verdict, target=target, explanation=explanation
        )
        if i == 0:
            outcomes[case.record_id] = _ok_outcome(result, spend_outcome="repaired")
        elif i == 1:
            # chunk-level parse failure → deferred spend, no result.
            outcomes[case.record_id] = BackendOutcome(
                results=[],
                deferred=[DeferralEntry(record_id=case.record_id, reason="parse")],
                spend=[_spend("deferred")],
            )
        else:
            outcomes[case.record_id] = _ok_outcome(result)

    backend = ScriptedBackend(kind="cli", outcomes=outcomes)
    report = ev.run_enrichment_eval(backend, cases)

    assert report.backend_kind == "cli"
    assert report.total == len(cases)
    # One repaired + one deferred → first-attempt = total-2; post-repair = total-1.
    assert report.first_attempt_valid == len(cases) - 2
    assert report.post_repair_valid == len(cases) - 1
    assert report.first_attempt_rate == pytest.approx((len(cases) - 2) / len(cases))
    assert report.post_repair_rate == pytest.approx((len(cases) - 1) / len(cases))


def test_cli_all_ok_passes_gates() -> None:
    """An all-'ok' CLI run clears both ≥90% / ≥99% gates and passes (§10.4)."""
    cases = ev.default_cases()
    outcomes: dict[str, BackendOutcome] = {}
    for case in cases:
        verdict = case.expected_verdict or "new"
        target = case.candidates[0].name if verdict in ("duplicate", "conflict") else None
        explanation = "they disagree" if verdict == "conflict" else None
        outcomes[case.record_id] = _ok_outcome(
            _result(case.record_id, verdict=verdict, target=target, explanation=explanation)
        )
    report = ev.run_enrichment_eval(ScriptedBackend(kind="cli", outcomes=outcomes), cases)
    assert report.first_attempt_rate == 1.0
    assert report.post_repair_rate == 1.0
    assert report.passed()


def test_cli_below_first_attempt_gate_fails() -> None:
    """A CLI run under the 90% first-attempt gate FAILS even if repair recovers it."""
    cases = ev.default_cases()
    n = len(cases)
    # Defer (then repair) enough cases to push first-attempt below 0.90 but keep
    # post-repair at 1.0: mark 'repaired' on > 10% of cases.
    n_repaired = int(n * 0.2) + 1
    outcomes: dict[str, BackendOutcome] = {}
    for i, case in enumerate(cases):
        verdict = case.expected_verdict or "new"
        target = case.candidates[0].name if verdict in ("duplicate", "conflict") else None
        explanation = "they disagree" if verdict == "conflict" else None
        result = _result(
            case.record_id, verdict=verdict, target=target, explanation=explanation
        )
        spend_outcome = "repaired" if i < n_repaired else "ok"
        outcomes[case.record_id] = _ok_outcome(result, spend_outcome=spend_outcome)
    report = ev.run_enrichment_eval(ScriptedBackend(kind="cli", outcomes=outcomes), cases)
    assert report.first_attempt_rate < ev.CLI_FIRST_ATTEMPT_GATE
    assert report.post_repair_rate == 1.0  # repair recovered everything
    assert not report.passed()  # but the first-attempt gate still fails


# --------------------------------------------------------------------------- #
# Fast structural test — SDK rubric scoring (offline, NOT skipped)              #
# --------------------------------------------------------------------------- #


def test_sdk_rubric_all_correct_passes() -> None:
    """SDK rubric: correct verdicts + non-empty summary + 1..8 aliases → PASS."""
    cases = ev.default_cases()
    outcomes: dict[str, BackendOutcome] = {}
    for case in cases:
        verdict = case.expected_verdict or "new"
        target = case.candidates[0].name if verdict in ("duplicate", "conflict") else None
        explanation = "they disagree" if verdict == "conflict" else None
        outcomes[case.record_id] = _ok_outcome(
            _result(case.record_id, verdict=verdict, target=target, explanation=explanation)
        )
    report = ev.run_enrichment_eval(ScriptedBackend(kind="sdk", outcomes=outcomes), cases)
    assert report.backend_kind == "sdk"
    assert report.content_rate == 1.0
    assert report.passed()


def test_sdk_rubric_wrong_verdicts_below_gate_fails() -> None:
    """Wrong verdicts on the unambiguous cases drop content-rate below gate."""
    cases = ev.default_cases()
    outcomes: dict[str, BackendOutcome] = {}
    for case in cases:
        # Force every unambiguous case to the WRONG verdict ('new' with no target),
        # which still satisfies the structural gates but fails verdict match.
        outcomes[case.record_id] = _ok_outcome(_result(case.record_id, verdict="new"))
    report = ev.run_enrichment_eval(ScriptedBackend(kind="sdk", outcomes=outcomes), cases)
    # The duplicate/conflict cases now mismatch; content-rate falls below 90%.
    assert report.content_rate < ev.SDK_CONTENT_GATE
    assert not report.passed()


def test_sdk_judgement_call_cases_not_auto_graded_on_verdict() -> None:
    """expected_verdict=None cases skip verdict auto-grading (human review only)."""
    cases = ev.default_cases()
    judgement = [c for c in cases if c.expected_verdict is None]
    assert judgement, "fixture set must include judgement-call cases"
    outcomes = {
        c.record_id: _ok_outcome(_result(c.record_id, verdict="duplicate", target=None))
        for c in cases
    }
    # A 'duplicate' with target=None violates nullable consistency? No — target
    # None is allowed; the structural gate only forbids a 'new' WITH a target.
    report = ev.run_enrichment_eval(ScriptedBackend(kind="sdk", outcomes=outcomes), cases)
    for cr in report.cases:
        if cr.case.expected_verdict is None:
            assert cr.verdict_matches_expected is None


# --------------------------------------------------------------------------- #
# Fast structural test — the deterministic gates (ANY backend, offline)         #
# --------------------------------------------------------------------------- #


def test_structural_gate_out_of_set_target_flagged() -> None:
    """A dedup_target_name NOT in the candidate set fails the ∈-candidates gate."""
    cases = ev.default_cases()
    bad = next(c for c in cases if c.candidates)
    outcomes: dict[str, BackendOutcome] = {}
    for case in cases:
        if case.record_id == bad.record_id:
            outcomes[case.record_id] = _ok_outcome(
                _result(case.record_id, verdict="duplicate", target="not-a-candidate")
            )
        else:
            outcomes[case.record_id] = _ok_outcome(_result(case.record_id))
    report = ev.run_enrichment_eval(ScriptedBackend(kind="cli", outcomes=outcomes), cases)
    failures = {cr.case.record_id for cr in report.structural_failures}
    assert bad.record_id in failures
    assert not report.passed()  # a structural failure fails ANY backend


def test_structural_gate_empty_summary_flagged() -> None:
    """An empty summary fails the all-fields-present gate (§5.2)."""
    cases = ev.default_cases()
    target_case = cases[0]
    outcomes = {c.record_id: _ok_outcome(_result(c.record_id)) for c in cases}
    outcomes[target_case.record_id] = _ok_outcome(
        _result(target_case.record_id, summary="   ")
    )
    report = ev.run_enrichment_eval(ScriptedBackend(kind="sdk", outcomes=outcomes), cases)
    failures = {cr.case.record_id for cr in report.structural_failures}
    assert target_case.record_id in failures


def test_structural_gate_alias_bounds_flagged() -> None:
    """Aliases outside 1..8 fail the all-fields-present gate (§5.2)."""
    cases = ev.default_cases()
    target_case = cases[0]
    outcomes = {c.record_id: _ok_outcome(_result(c.record_id)) for c in cases}
    outcomes[target_case.record_id] = _ok_outcome(
        _result(target_case.record_id, aliases=[f"a{i}" for i in range(9)])
    )
    report = ev.run_enrichment_eval(ScriptedBackend(kind="sdk", outcomes=outcomes), cases)
    failures = {cr.case.record_id for cr in report.structural_failures}
    assert target_case.record_id in failures


def test_fixture_set_shape() -> None:
    """The curated fixture set is representative: 10..20 cases, all 3 verdicts."""
    cases = ev.default_cases()
    assert 10 <= len(cases) <= 20
    expected = {c.expected_verdict for c in cases}
    assert {"new", "duplicate", "conflict"}.issubset(expected)
    assert None in expected  # at least one judgement-call case
    # Every duplicate/conflict expectation has a non-empty candidate set with a
    # target the model could legitimately pick.
    for c in cases:
        if c.expected_verdict in ("duplicate", "conflict"):
            assert c.candidates, f"{c.record_id} needs candidates for its verdict"


def test_format_report_runs() -> None:
    """The pretty-printer renders without error and includes the verdict line."""
    cases = ev.default_cases()
    outcomes = {c.record_id: _ok_outcome(_result(c.record_id)) for c in cases}
    report = ev.run_enrichment_eval(ScriptedBackend(kind="cli", outcomes=outcomes), cases)
    text = ev.format_report(report)
    assert "enrichment eval" in text
    assert "VERDICT:" in text


# --------------------------------------------------------------------------- #
# Backend-gated live eval — SKIPPED offline (the morning-verify path)           #
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(
    not ev.live_eval_opt_in(),
    reason=(
        "§10.4 enrichment eval is morning-verify; needs a live backend, opt in "
        "with CLAUDEMEM_RUN_ENRICH_EVAL=1 (skipped offline/by default per SC-3)"
    ),
)
def test_live_enrichment_eval_meets_gates() -> None:
    """Run the eval against the AUTO-DETECTED live backend; assert it meets §10.4.

    SKIPPED in the unattended suite (no backend + no opt-in flag). When a live
    backend is present this drives the real transport: CLI ≥90% first-attempt /
    ≥99% post-repair, SDK content rubric ≥90% auto-graded, plus the structural
    gates for either.
    """
    from claudemem import config
    from claudemem.enrich.backend import select_backend

    backend = select_backend(config.load_config())
    report = ev.run_enrichment_eval(backend, ev.default_cases())
    print("\n" + ev.format_report(report))  # noqa: T201 — eval evidence for the run log.
    assert report.passed(), (
        f"enrichment eval FAILED for backend {report.backend_kind!r}; "
        f"see the printed per-case report"
    )
