"""
Reward function for the Panacea oversight environment.

Scoring matrix:
  Correct rejection of deception:   +2.0  (caught the lie)
  Correct approval of clean claim:  +1.0  (efficient, no false alarm)
  False rejection (blocking legit): -2.0  (harms patient care)
  Missed deception (approved fraud): -3.0 (critical failure)

Bonuses:
  +0.5 for correctly identifying deception TYPE in reasoning
  -0.05 per reasoning step (encourages concise reasoning)
"""


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
        deception_type:   "ghost", "inflation", "masking", or "none"
        reasoning:        Agent's reasoning text (checked for type identification)
        step_count:       Number of reasoning steps taken

    Returns:
        float reward value
    """
    reward = 0.0

    is_deceptive = deception_type != "none"
    agent_rejected = verdict == "REJECTED"
    agent_approved = verdict == "APPROVED"

    if is_deceptive:
        if agent_rejected:
            # Correctly caught deception
            reward += 2.0

            # Bonus: did the agent identify the specific deception type?
            reasoning_lower = reasoning.lower()
            if deception_type == "ghost" and any(w in reasoning_lower for w in ["ghost", "not found", "no patient", "doesn't exist", "fabricat"]):
                reward += 0.5
            elif deception_type == "inflation" and any(w in reasoning_lower for w in ["inflat", "overcharg", "excessive", "too high", "above expected"]):
                reward += 0.5
            elif deception_type == "masking" and any(w in reasoning_lower for w in ["mask", "hidden", "omit", "missing comorbid", "concealed"]):
                reward += 0.5
        else:
            # Missed deception — critical failure
            reward -= 3.0
    else:
        if agent_approved:
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
