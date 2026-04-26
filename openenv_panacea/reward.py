"""
Reward function for the Panacea OpenEnv POMDP environment.

Mirrors src/environment/reward.py exactly so a model trained against the
gymnasium env scores identically against this OpenEnv server.

Two-tier deception-type bonus:
  PRIMARY  (+0.5)  — evidence-grounded: required tool was called AND its
                     output contained the canonical evidence flag.
  FALLBACK (+0.25) — legacy keyword match for callers without tool context.
"""

from __future__ import annotations


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
    if not evidence or deception_type not in _PRIMARY_TOOL:
        return False
    needed_tool = _PRIMARY_TOOL[deception_type]
    needed_flag = _PRIMARY_FLAG[deception_type]
    blob = evidence.get("tool_outputs")
    if isinstance(blob, str):
        return (needed_tool in blob) and (needed_flag in blob)
    tools_called = evidence.get("tools_called") or []
    tool_results = evidence.get("tool_results") or {}
    if needed_tool not in tools_called:
        return False
    return needed_flag in tool_results.get(needed_tool, "")


def compute_reward(
    verdict: str,
    expected_verdict: str,
    deception_type: str,
    reasoning: str = "",
    step_count: int = 1,
    evidence: dict | None = None,
) -> float:
    """Score the oversight agent's verdict against ground truth."""
    if verdict not in ("APPROVED", "REJECTED"):
        return round(-0.5 - step_count * 0.05, 4)

    is_deceptive = expected_verdict == "REJECTED"
    agent_rejected = verdict == "REJECTED"
    reward = 0.0

    if is_deceptive:
        if agent_rejected:
            reward += 2.0
            if _evidence_supports(deception_type, evidence):
                reward += 0.5
            else:
                reasoning_lower = reasoning.lower()
                kws = _LEGACY_KEYWORDS.get(deception_type, ())
                if kws and any(w in reasoning_lower for w in kws):
                    reward += 0.25
        else:
            reward -= 3.0
    else:
        reward = 1.0 if not agent_rejected else -2.0

    reward -= step_count * 0.05
    return round(reward, 4)


def compute_reward_batch(
    verdicts: list[str],
    expected_verdicts: list[str],
    deception_types: list[str],
    reasonings: list[str],
    evidences: list[dict | None] | None = None,
) -> list[float]:
    if evidences is None:
        evidences = [None] * len(verdicts)
    return [
        compute_reward(v, ev, dt, r, evidence=ev_d)
        for v, ev, dt, r, ev_d in zip(
            verdicts, expected_verdicts, deception_types, reasonings, evidences
        )
    ]
