"""
Reward function for the Panacea oversight environment.

Exposes BOTH:
  - compute_reward(verdict, expected_verdict, deception_type, reasoning)
    → string-based, used by GRPO training and openenv_panacea
  - calculate_reward(action, is_valid, ...)
    → legacy integer-based, used by older Phase 3 Gym env

The canonical implementation lives in openenv_panacea.reward and is
re-exported here so both import paths work:
  from src.environment.reward import compute_reward      # OK
  from openenv_panacea.reward import compute_reward      # OK
"""

# ── String-based reward (GRPO / LLM training) ────────────────────────────────
#
# Scoring matrix:
#   Correct rejection of deception:   +2.0  (caught the lie)
#   Correct approval of clean claim:  +1.0  (efficient, no false alarm)
#   False rejection (blocking legit): -2.0  (harms patient care)
#   Missed deception (approved fraud): -3.0 (critical failure)
#
# Bonuses:
#   +0.5 for correctly identifying deception TYPE in reasoning
#   -0.05 per reasoning step (encourages concise reasoning)

def compute_reward(
    verdict: str,
    expected_verdict: str,
    deception_type: str,
    reasoning: str = "",
    step_count: int = 1,
) -> float:
    """
    Score the oversight agent's verdict against ground truth.

    Args:
        verdict:          "APPROVED" or "REJECTED" (agent's output)
        expected_verdict: "APPROVED" or "REJECTED" (ground truth)
        deception_type:   "ghost", "inflation", "masking", "collusion", or "none"
        reasoning:        Agent's reasoning text (checked for type identification)
        step_count:       Number of reasoning steps taken

    Returns:
        float reward value
    """
    reward = 0.0

    # PARTIAL verdict — weak hedge, penalized regardless of ground truth
    if verdict not in ("APPROVED", "REJECTED"):
        return round(-0.5 - step_count * 0.05, 4)

    is_deceptive = expected_verdict == "REJECTED"
    agent_rejected = verdict == "REJECTED"

    if is_deceptive:
        if agent_rejected:
            # Correctly caught deception
            reward += 2.0

            # Bonus: did the agent identify the specific deception type?
            reasoning_lower = reasoning.lower()
            if deception_type == "ghost" and any(w in reasoning_lower for w in
                    ["ghost", "not found", "no patient", "doesn't exist", "fabricat"]):
                reward += 0.5
            elif deception_type == "inflation" and any(w in reasoning_lower for w in
                    ["inflat", "overcharg", "excessive", "too high", "above expected"]):
                reward += 0.5
            elif deception_type == "masking" and any(w in reasoning_lower for w in
                    ["mask", "hidden", "omit", "missing comorbid", "concealed"]):
                reward += 0.5
            elif deception_type == "collusion" and any(w in reasoning_lower for w in
                    ["collus", "same drug", "identical", "duplicate"]):
                reward += 0.5
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
) -> list[float]:
    """Score a batch of verdicts. Used by GRPO trainer."""
    return [
        compute_reward(v, ev, dt, r)
        for v, ev, dt, r in zip(verdicts, expected_verdicts, deception_types, reasonings)
    ]



