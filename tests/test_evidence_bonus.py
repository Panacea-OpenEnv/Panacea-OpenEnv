"""Unit tests for evidence-grounded reward bonus.

Verifies that the +0.5 deception-type bonus only fires when:
  1. The agent called the canonical tool for that deception type, AND
  2. The tool output contained the canonical evidence flag.

Otherwise the bonus drops to +0.25 (legacy keyword fallback) or 0.0.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.environment.reward import compute_reward


#  ghost 

def test_ghost_evidence_bonus_full():
    """Calling REGISTRY and surfacing 'NO RECORD' → +0.5 bonus."""
    evidence = {
        "tool_outputs": (
            ">> CALL TOOL_REGISTRY (Government ID Verification API)\n"
            "[REGISTRY] Lookup pid=P9999: NO RECORD FOUND."
        )
    }
    r = compute_reward(
        verdict="REJECTED", expected_verdict="REJECTED",
        deception_type="ghost", reasoning="patient was fabricated",
        step_count=1, evidence=evidence,
    )
    # 2.0 base + 0.5 evidence bonus - 0.05 step = 2.45
    assert abs(r - 2.45) < 1e-6, f"expected 2.45, got {r}"


def test_ghost_keyword_only_half_bonus():
    """Saying 'ghost patient' but never calling REGISTRY → only +0.25 fallback."""
    r = compute_reward(
        verdict="REJECTED", expected_verdict="REJECTED",
        deception_type="ghost", reasoning="this looks like a ghost patient",
        step_count=1, evidence=None,
    )
    # 2.0 + 0.25 - 0.05 = 2.20
    assert abs(r - 2.20) < 1e-6, f"expected 2.20, got {r}"


def test_ghost_no_evidence_no_keyword_no_bonus():
    """Correct verdict, no evidence, no keyword → just +2.0 base."""
    r = compute_reward(
        verdict="REJECTED", expected_verdict="REJECTED",
        deception_type="ghost", reasoning="something is off",
        step_count=1, evidence=None,
    )
    # 2.0 + 0.0 - 0.05 = 1.95
    assert abs(r - 1.95) < 1e-6, f"expected 1.95, got {r}"


#  inflation 

def test_inflation_evidence_bonus_full():
    """Calling BILLING and surfacing '<RATIO=' → +0.5 bonus."""
    evidence = {
        "tool_outputs": (
            ">> CALL TOOL_BILLING (Insurance Claims System)\n"
            "[BILLING] claimed=$80,000 expected=$25,000 <RATIO=3.20x EXPECTED>"
        )
    }
    r = compute_reward(
        verdict="REJECTED", expected_verdict="REJECTED",
        deception_type="inflation", reasoning="claim too high",
        step_count=1, evidence=evidence,
    )
    assert abs(r - 2.45) < 1e-6, f"expected 2.45, got {r}"


def test_inflation_wrong_tool_no_bonus():
    """Calling REGISTRY (wrong tool) for inflation → no primary bonus."""
    evidence = {
        "tool_outputs": ">> CALL TOOL_REGISTRY\n[REGISTRY] VERIFIED."
    }
    r = compute_reward(
        verdict="REJECTED", expected_verdict="REJECTED",
        deception_type="inflation", reasoning="something fishy",
        step_count=1, evidence=evidence,
    )
    # No bonus because BILLING wasn't called
    assert abs(r - 1.95) < 1e-6, f"expected 1.95, got {r}"


#  masking 

def test_masking_evidence_bonus():
    evidence = {
        "tool_outputs": (
            ">> CALL TOOL_REPORTS\n"
            "[REPORTS] #1 Cardiology: dx='chest pain'\n"
            "  comorbidities_disclosed: none"
        )
    }
    r = compute_reward(
        verdict="REJECTED", expected_verdict="REJECTED",
        deception_type="masking", reasoning="comorbidities omitted",
        step_count=1, evidence=evidence,
    )
    assert abs(r - 2.45) < 1e-6, f"expected 2.45, got {r}"


#  collusion 

def test_collusion_evidence_bonus():
    evidence = {
        "tool_outputs": (
            ">> CALL TOOL_DRUGS\n"
            "[DRUGDB] Enoxaparin: prescribed_by=['Cardiology','Pulmonology'] <DUPLICATE-PRESCRIBER>"
        )
    }
    r = compute_reward(
        verdict="REJECTED", expected_verdict="REJECTED",
        deception_type="collusion", reasoning="two specialists same drug",
        step_count=1, evidence=evidence,
    )
    assert abs(r - 2.45) < 1e-6, f"expected 2.45, got {r}"


#  structured-evidence form 

def test_structured_evidence_form_works():
    """Form B: tools_called + tool_results dict."""
    evidence = {
        "tools_called": ["TOOL_REGISTRY"],
        "tool_results": {"TOOL_REGISTRY": "[REGISTRY] NO RECORD FOUND for P0001"},
    }
    r = compute_reward(
        verdict="REJECTED", expected_verdict="REJECTED",
        deception_type="ghost", reasoning="",
        step_count=1, evidence=evidence,
    )
    assert abs(r - 2.45) < 1e-6, f"expected 2.45, got {r}"


#  negative cases (clean claims, missed deceptions) 

def test_clean_claim_correct_approve():
    r = compute_reward(
        verdict="APPROVED", expected_verdict="APPROVED",
        deception_type="none", reasoning="all clear", step_count=1,
    )
    assert abs(r - 0.95) < 1e-6, f"expected 0.95, got {r}"


def test_missed_deception():
    r = compute_reward(
        verdict="APPROVED", expected_verdict="REJECTED",
        deception_type="ghost", reasoning="looks fine", step_count=1,
    )
    assert abs(r - (-3.05)) < 1e-6, f"expected -3.05, got {r}"


def test_unparseable_verdict():
    r = compute_reward(
        verdict="MAYBE", expected_verdict="REJECTED",
        deception_type="ghost", reasoning="", step_count=1,
    )
    assert abs(r - (-0.55)) < 1e-6, f"expected -0.55, got {r}"


if __name__ == "__main__":
    tests = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS  {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"FAIL  {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    sys.exit(failed)
