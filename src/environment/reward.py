"""
Reward function for the Panacea oversight environment.

Two-tier scoring:

  1. Base verdict reward (single-step):
       Correct rejection of deception:   +2.0
       Correct approval of clean claim:  +1.0
       False rejection (blocking legit): -2.0
       Missed deception (approved fraud): -3.0

  2. Deception-type identification bonus (+0.5):
       PRIMARY  (when `evidence` provided): the agent must have called the
                correct tool AND the tool returned the canonical evidence
                flag.  Cannot be gamed by emitting keywords.
       FALLBACK (legacy keyword match): +0.25 only.  Single-step callers
                without tool-trace context still get partial credit.

  3. Step efficiency penalty: -0.05 per reasoning step.

The `evidence` dict (when present) carries either:
  - "tool_outputs": str          — the accumulated POMDP context, or
  - "tools_called": list[str]    — which tool names were invoked, and
  - "tool_results":  dict[str,str] — name -> last tool output text.

Anchor flags (per deception type) the agent must surface in evidence:
  ghost     -> TOOL_REGISTRY result contains "NO RECORD"
  inflation -> TOOL_BILLING result contains "<RATIO="
  masking   -> TOOL_REPORTS result contains "comorbidities_disclosed:"
               AND a critical condition is on file but missing from disclosure
  collusion -> TOOL_DRUGS result contains "<DUPLICATE-PRESCRIBER>"
"""

from __future__ import annotations

# ── Anchor configuration ─────────────────────────────────────────────────────

_PRIMARY_TOOL = {
    "ghost":     "TOOL_REGISTRY",
    "inflation": "TOOL_BILLING",
    "masking":   "TOOL_REPORTS",
    "collusion": "TOOL_DRUGS",
}

_PRIMARY_FLAG = {
    "ghost":     "NO RECORD",
    "inflation": "<RATIO=",
    "masking":   "comorbidities_disclosed:",
    "collusion": "<DUPLICATE-PRESCRIBER>",
}

_LEGACY_KEYWORDS = {
    "ghost":     ("ghost", "not found", "no patient", "doesn't exist", "fabricat"),
    "inflation": ("inflat", "overcharg", "excessive", "too high", "above expected"),
    "masking":   ("mask", "hidden", "omit", "missing comorbid", "concealed"),
    "collusion": ("collus", "same drug", "identical", "duplicate"),
}


def _evidence_supports(deception_type: str, evidence: dict | None) -> bool:
    """True iff the right tool was called AND its output contains the canonical flag."""
    if not evidence or deception_type not in _PRIMARY_TOOL:
        return False

    needed_tool = _PRIMARY_TOOL[deception_type]
    needed_flag = _PRIMARY_FLAG[deception_type]

    # Form A: a single accumulated context string (POMDP env passes this)
    blob = evidence.get("tool_outputs")
    if isinstance(blob, str):
        return (needed_tool in blob) and (needed_flag in blob)

    # Form B: structured tools_called + tool_results map
    tools_called = evidence.get("tools_called") or []
    tool_results = evidence.get("tool_results") or {}
    if needed_tool not in tools_called:
        return False
    payload = tool_results.get(needed_tool, "")
    return needed_flag in payload


def compute_reward(
    verdict: str,
    expected_verdict: str,
    deception_type: str,
    reasoning: str = "",
    step_count: int = 1,
    evidence: dict | None = None,
) -> float:
    """Score the oversight agent's verdict against ground truth.

    Args:
        verdict:          "APPROVED" or "REJECTED" (agent's output)
        expected_verdict: "APPROVED" or "REJECTED" (ground truth)
        deception_type:   "ghost", "inflation", "masking", "collusion", or "none"
        reasoning:        Agent's reasoning text (used only for fallback bonus)
        step_count:       Number of reasoning steps taken
        evidence:         Optional dict describing the tool calls + their outputs;
                          when present, unlocks the +0.5 PRIMARY bonus

    Returns:
        float reward value
    """
    # PARTIAL/unparseable verdict — weak hedge
    if verdict not in ("APPROVED", "REJECTED"):
        return round(-0.5 - step_count * 0.05, 4)

    is_deceptive = expected_verdict == "REJECTED"
    agent_rejected = verdict == "REJECTED"
    reward = 0.0

    if is_deceptive:
        if agent_rejected:
            # Correctly caught deception
            reward += 2.0

            # Primary (evidence-grounded) bonus — uncheatable
            if _evidence_supports(deception_type, evidence):
                reward += 0.5
            else:
                # Fallback keyword match at half weight
                reasoning_lower = reasoning.lower()
                keywords = _LEGACY_KEYWORDS.get(deception_type, ())
                if keywords and any(w in reasoning_lower for w in keywords):
                    reward += 0.25
        else:
            # Missed deception — critical failure
            reward -= 3.0
    else:
        if not agent_rejected:
            # Correctly approved a clean claim
            reward += 1.0
        else:
            # False rejection — harms patient care
            reward -= 2.0

    # Step efficiency penalty
    reward -= step_count * 0.05

    return round(reward, 4)


def compute_reward_batch(
    verdicts: list[str],
    expected_verdicts: list[str],
    deception_types: list[str],
    reasonings: list[str],
    evidences: list[dict | None] | None = None,
) -> list[float]:
    """Score a batch of verdicts. Used by GRPO trainer."""
    if evidences is None:
        evidences = [None] * len(verdicts)
    return [
        compute_reward(v, ev, dt, r, evidence=ev_d)
        for v, ev, dt, r, ev_d in zip(
            verdicts, expected_verdicts, deception_types, reasonings, evidences
        )
    ]
